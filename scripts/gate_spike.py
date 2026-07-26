"""Phase 3 framework gate — the ADK spike.

Five boxes, each evidenced by running rather than argued:

1. an agent answers via Groq/OpenAI through our own BaseLlm adapter
2. trace rows exist from both agents in a two-agent hub-and-spoke
3. restart the process and a follow-up message still works
4. a state[...] value written before the restart is readable after it
5. the mock BaseLlm short-circuits the real LLM and drives the same loop

Run:  python scripts/gate_spike.py            (boxes 2-5, mock only)
      python scripts/gate_spike.py --live     (adds box 1 against Groq)

Boxes 3 and 4 re-invoke this script as a **separate process**. Deleting an
object in the same interpreter is not a restart: imports, engines, and caches
all survive it, which is exactly what the box exists to rule out. The reported
PIDs are the evidence that two interpreters were involved.

Deliberately a script rather than a test: the gate is a decision made once, and
its evidence should be re-runnable by hand and readable in the terminal.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import DatabaseSessionService  # noqa: E402
from google.adk.tools import FunctionTool  # noqa: E402
from google.genai import types  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import TraceEvent  # noqa: E402
from app.models.enums import TraceAuthor  # noqa: E402
from app.providers import get_provider  # noqa: E402
from app.trace import TraceWriter, check_session  # noqa: E402

APP_NAME = "agentcare-gate"
USER_ID = "gate-user"

_writer: TraceWriter | None = None


# --- tools the spike exposes -------------------------------------------


def resolve_department(text: str) -> dict:
    """Work out which hospital department a request belongs to."""
    from app.tools.departments import resolve_department as resolve

    session = SessionLocal()
    try:
        return resolve(session, text)
    finally:
        session.close()


def list_patient_documents() -> dict:
    """List the documents the patient already has on file."""
    from app.tools.documents import list_patient_documents as listing

    session = SessionLocal()
    try:
        return {"documents": listing(session, patient_id=1)}
    finally:
        session.close()


def list_departments() -> dict:
    """List every hospital department that accepts appointments."""
    from app.tools.departments import list_departments as listing

    session = SessionLocal()
    try:
        return {"departments": [d["name"] for d in listing(session)]}
    finally:
        session.close()


def remember_preference_tool(department: str, tool_context) -> dict:
    """Remember the patient's preferred department for later in the session."""
    tool_context.state["preferred_department"] = department
    return {"remembered": department}


# --- ADK callbacks: the trace capture points ---------------------------
# Parameter names must match ADK's keywords EXACTLY. Renaming one to `ctx` is a
# mid-turn TypeError, and these callbacks are the capture points.


def before_model_callback(callback_context, llm_request):  # noqa: ANN001
    from app.providers.base import request_snapshot

    correlation = _writer.llm_request(
        agent_name=callback_context.agent_name, payload=request_snapshot(llm_request)
    )
    callback_context.state["last_request_id"] = correlation
    return None


def after_model_callback(callback_context, llm_response):  # noqa: ANN001
    from app.providers.base import response_snapshot

    correlation = callback_context.state.get("last_request_id")
    if correlation:
        _writer.llm_response(
            correlation,
            agent_name=callback_context.agent_name,
            payload=response_snapshot(llm_response),
        )
    return None


def before_tool_callback(tool, args, tool_context):  # noqa: ANN001
    correlation = _writer.tool_call(
        tool.name, args=dict(args or {}), agent_name=tool_context.agent_name
    )
    tool_context.state["tool_" + tool.name] = correlation
    return None


def after_tool_callback(tool, args, tool_context, tool_response):  # noqa: ANN001
    correlation = tool_context.state.get("tool_" + tool.name)
    _writer.tool_result(
        correlation, name=tool.name, result=tool_response, agent_name=tool_context.agent_name
    )
    return None


def build_agent(provider_name: str) -> LlmAgent:
    """Two agents, hub and spoke: a coordinator delegating to a specialist."""
    specialist = LlmAgent(
        name="department_specialist",
        model=get_provider(provider_name),
        description="Answers questions about which hospital departments exist.",
        instruction=(
            "You answer questions about which departments exist, using the tools "
            "provided. Administrative information only; never give clinical advice."
        ),
        tools=[
            FunctionTool(resolve_department),
            FunctionTool(list_departments),
            FunctionTool(list_patient_documents),
            FunctionTool(remember_preference_tool),
        ],
        before_model_callback=before_model_callback,
        after_model_callback=after_model_callback,
        before_tool_callback=before_tool_callback,
        after_tool_callback=after_tool_callback,
    )

    return LlmAgent(
        name="coordinator",
        model=get_provider(provider_name),
        description="Coordinates administrative patient requests.",
        instruction=(
            "You are a hospital administration coordinator. For anything about "
            "departments, transfer to department_specialist. Never give clinical advice."
        ),
        sub_agents=[specialist],
        before_model_callback=before_model_callback,
        after_model_callback=after_model_callback,
    )


async def one_turn(provider: str, session_id: str, text: str, session_service) -> str:
    """Run one turn end to end, bracketing it in the trace."""
    global _writer

    trace_session = SessionLocal()
    _writer = TraceWriter(trace_session, session_id=session_id)
    _writer.inbound(text, author=TraceAuthor.PATIENT_MESSAGE)

    runner = Runner(
        agent=build_agent(provider), app_name=APP_NAME, session_service=session_service
    )
    message = types.Content(role="user", parts=[types.Part(text=text)])
    reply = ""
    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=message
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        reply += part.text
        _writer.outbound(reply, author=TraceAuthor.LLM)
        return reply
    finally:
        trace_session.commit()
        trace_session.close()


async def run_turn_only(provider: str, session_id: str, text: str, create: bool) -> None:
    """One turn in this process, then exit. The child half of the restart test."""
    settings = get_settings()
    service = DatabaseSessionService(db_url=settings.adk_session_db_url)
    if create:
        await service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    reply = await one_turn(provider, session_id, text, service)
    print("PID:" + str(os.getpid()))
    print("REPLY:" + " ".join(reply.split())[:120])

    session = await service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    print("STATE_KEYS:" + ",".join(sorted((session.state or {}).keys())))
    print("EVENTS:" + str(len(session.events)))


def spawn(args: list[str]) -> tuple[int, str]:
    """Run this script again in a brand-new interpreter."""
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *args],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    return completed.returncode, completed.stdout + completed.stderr


def parse(output: str, key: str) -> str:
    for line in output.splitlines():
        if line.startswith(key + ":"):
            return line[len(key) + 1 :]
    return ""


def report(box: str, passed: bool, evidence: str) -> bool:
    print("  [" + ("PASS" if passed else "FAIL") + "] " + box)
    print("         " + evidence)
    return passed


async def main(live: bool) -> int:
    settings = get_settings()
    results: list[bool] = []
    session_id = "gate-" + uuid.uuid4().hex[:8]

    print("")
    print("ADK session store: " + settings.adk_session_db_url)
    print("Provider adapters: our own BaseLlm subclasses (LiteLLM not installed)")
    print("=" * 74)

    # --- Box 5 (also produces turn 1 for the restart) --------------------
    print("")
    print("Box 5 - mock BaseLlm short-circuits the real LLM and drives the loop")
    code, out = spawn(
        [
            "--turn", "--session", session_id, "--create", "--provider", "mock",
            "--text", "I need a cardiology appointment next week",
        ]
    )
    first_reply = parse(out, "REPLY")
    first_pid = parse(out, "PID")
    state_before = [k for k in parse(out, "STATE_KEYS").split(",") if k]
    results.append(
        report(
            "mock drives the ADK loop",
            code == 0 and bool(first_reply),
            ("pid " + first_pid + ", reply: " + repr(first_reply[:70]))
            if code == 0
            else out[-500:],
        )
    )

    # --- Box 2 -----------------------------------------------------------
    print("")
    print("Box 2 - trace rows from both agents in a two-agent hub-and-spoke")
    check = SessionLocal()
    try:
        # Scoped to this run's session: rows left by an earlier run must never
        # be able to vouch for this one.
        mine = check.query(TraceEvent).filter(TraceEvent.session_id == session_id)
        agents = {row.agent_name for row in mine.all() if row.agent_name}
        rows = mine.count()
        turn_ids = {row.turn_id for row in mine.all()}
        malformed = [
            r for t in turn_ids for r in check_session(check, turn_id=t) if not r.ok
        ]
    finally:
        check.close()
    results.append(
        report(
            "both agents appear in the trace",
            len(agents) >= 2,
            str(rows) + " trace rows; agents: " + str(sorted(agents) or "none"),
        )
    )

    # --- Boxes 3 and 4: a genuinely new process --------------------------
    print("")
    print("Boxes 3 & 4 - restart the process; follow-up works, state survives")
    code2, out2 = spawn(
        [
            "--turn", "--session", session_id, "--provider", "mock",
            "--text", "which departments did you say?",
        ]
    )
    follow_up = parse(out2, "REPLY")
    second_pid = parse(out2, "PID")
    state_after = [k for k in parse(out2, "STATE_KEYS").split(",") if k]
    events_after = parse(out2, "EVENTS") or "0"

    results.append(
        report(
            "follow-up works after a real process restart",
            code2 == 0 and bool(follow_up) and second_pid != first_pid,
            (
                "pid " + first_pid + " -> " + second_pid + ", replayed " + events_after
                + " events, reply: " + repr(follow_up[:60])
            )
            if code2 == 0
            else out2[-500:],
        )
    )

    # last_request_id is rewritten by every turn, so its presence after the
    # restart proves nothing. The evidence has to be a key only turn 1 wrote.
    rewritten_each_turn = {"last_request_id"}
    carried = sorted((set(state_before) & set(state_after)) - rewritten_each_turn)
    results.append(
        report(
            "state[...] written before the restart is readable after",
            bool(carried),
            "before: " + str(sorted(set(state_before) - rewritten_each_turn) or "none")
            + " | survived (excluding per-turn keys): " + str(carried or "none"),
        )
    )

    # --- Box 1 -----------------------------------------------------------
    print("")
    print("Box 1 - an agent answers via Groq through our own BaseLlm adapter")
    if not live:
        results.append(report("live provider call", False, "skipped (run with --live)"))
    else:
        live_session = "live-" + uuid.uuid4().hex[:8]
        code3, out3 = spawn(
            [
                "--turn", "--session", live_session, "--create", "--provider", "groq",
                "--text", "Which departments can I book an appointment with?",
            ]
        )
        live_reply = parse(out3, "REPLY")
        from app.providers.groq_provider import GroqLlm

        chain = " -> ".join(c.__name__ for c in GroqLlm.__mro__[:4])
        results.append(
            report(
                "live Groq call through GroqLlm, not LiteLLM",
                code3 == 0 and bool(live_reply),
                ("adapter: " + chain + " | reply: " + repr(live_reply[:70]))
                if code3 == 0
                else out3[-700:],
            )
        )

    print("")
    print("=" * 74)
    print(
        "Trace well-formedness: "
        + ("clean" if not malformed else str(len(malformed)) + " malformed turn(s)")
    )
    passed = sum(results)
    print("GATE: " + str(passed) + "/" + str(len(results)) + " boxes ticked")
    if passed != len(results):
        print("Not all boxes ticked - report the evidence, do not push through.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="include the live provider box")
    parser.add_argument("--turn", action="store_true", help="internal: run one turn and exit")
    parser.add_argument("--session", default="")
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--text", default="")
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONUTF8", "1")

    if args.turn:
        asyncio.run(run_turn_only(args.provider, args.session, args.text, args.create))
        raise SystemExit(0)

    raise SystemExit(asyncio.run(main(args.live)))
