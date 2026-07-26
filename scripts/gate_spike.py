"""Framework gate evidence — the five boxes, run against the real system.

Originally this was a spike: a throwaway two-agent hub-and-spoke wired with
ADK's ``transfer_to_agent``, built to answer "does ADK work here?" before any
of the system existed. It answered yes, and Phase 4 then built the real thing —
which dispatches delegation from a validated plan instead of transferring,
because a transfer hands the sub-agent the whole session history and the
context contract forbids that.

So the spike stopped running, and a script reporting a failed gate for a
decision that had not changed is worse than no script. It now drives
``run_workflow``: the same five boxes, evidenced against the code that ships
rather than against scaffolding that no longer resembles it.

Five boxes, each evidenced by running rather than argued:

1. an agent answers via Groq through our own BaseLlm adapter
2. trace rows from multiple agents in one hub-and-spoke turn
3. restart the process and a follow-up message still works
4. state written before the restart is readable after it
5. the mock BaseLlm short-circuits the real LLM and drives the same loop

Run:  python scripts/gate_spike.py            (boxes 2-5, mock only)
      python scripts/gate_spike.py --live     (adds box 1 against Groq)

Boxes 3 and 4 re-invoke this script as a **separate process**. Deleting an
object in the same interpreter is not a restart: imports, engines, and caches
all survive it, which is exactly what the box exists to rule out. The reported
PIDs are the evidence that two interpreters were involved.

The parent resets and re-seeds the database first. Without that, a run left
active by a previous invocation would make turn 1 classify against it instead
of planning fresh, and the gate would be measuring its own leftovers.
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

from app.config import get_settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import TraceEvent, TraceEventType, User, WorkflowRun  # noqa: E402
from app.orchestrator import active_run, run_workflow  # noqa: E402
from app.trace import check_session  # noqa: E402

#: Patient asha drives the restart pair; rohan is kept for the live box so a
#: billed call never has to share state with the mock boxes.
MOCK_PATIENT = "asha.patient@example.invalid"
LIVE_PATIENT = "rohan.patient@example.invalid"

BOOKING = "I need a cardiology appointment next week"
CONFIRM = "yes"
LIVE_MESSAGE = "I would like to book a cardiology appointment next week"


# --- the child half: one turn in its own interpreter --------------------


def _patient_profile_id(session, email: str) -> int:
    from app.models import PatientProfile

    user = session.query(User).filter(User.email == email).one()
    return (
        session.query(PatientProfile)
        .filter(PatientProfile.user_id == user.id)
        .one()
        .id
    )


def _pre_turn_state(email: str) -> dict[str, str]:
    """What this process can already see, before it does anything.

    Read *first*, so anything reported here was written by somebody else. This
    is the box-4 evidence, and getting it in the wrong order is precisely how
    the earlier version of this script fooled itself.
    """
    session = SessionLocal()
    try:
        run = active_run(session, _patient_profile_id(session, email))
        if run is None:
            return {"PRE_STATUS": "none", "PRE_DEPT": "", "PRE_SLOT": ""}
        state = run.state or {}
        return {
            "PRE_STATUS": run.status.value,
            "PRE_DEPT": str(state.get("department_name") or ""),
            "PRE_SLOT": str(run.proposed_slot_id or ""),
        }
    finally:
        session.close()


async def _adk_event_count(user_id: str, session_id: str) -> int:
    from app.agents import memory

    service = memory.conversation_service()
    session = await service.get_session(
        app_name=memory.APP_NAME, user_id=user_id, session_id=session_id
    )
    return len(session.events) if session else 0


def _trace_facts(session_id: str) -> dict[str, str]:
    session = SessionLocal()
    try:
        rows = (
            session.query(TraceEvent)
            .filter(TraceEvent.session_id == session_id)
            .all()
        )
        agents = sorted({row.agent_name for row in rows if row.agent_name})
        tool_calls = [
            row for row in rows if row.event_type is TraceEventType.TOOL_CALL
        ]
        return {
            "AGENTS": ",".join(agents),
            "TOOL_CALLS": str(len(tool_calls)),
            "TRACE_ROWS": str(len(rows)),
        }
    finally:
        session.close()


async def run_turn_only(provider: str, session_id: str, text: str, email: str) -> None:
    """One turn in this process, then exit. The child half of the restart test."""
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.email == email).one()
        user_id = str(user.id)
    finally:
        session.close()

    # ADK sessions are keyed by the numeric user id, which is what
    # ``run_workflow`` passes. Asking with the email finds nothing and reports
    # zero — a check that fails for a reason having nothing to do with the box.
    facts = _pre_turn_state(email)
    facts["PRE_ADK_EVENTS"] = str(await _adk_event_count(user_id, session_id))

    result = await run_workflow(user, text, session_id, provider=provider)

    facts.update(_trace_facts(session_id))
    facts["PID"] = str(os.getpid())
    facts["STATUS"] = str(result.status)
    facts["RUN_ID"] = str(result.run_id)
    facts["STEPS"] = ",".join(result.steps_run)
    facts["ADK_EVENTS"] = str(await _adk_event_count(user_id, session_id))
    facts["REPLY"] = " ".join(result.reply.split())[:140]

    for key, value in facts.items():
        print(key + ":" + value)


# --- the parent half -----------------------------------------------------


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
    for line in evidence.splitlines():
        print("         " + line)
    return passed


def reset_database() -> None:
    from scripts.seed import run as seed_run

    seed_run(reset=True)


def main(live: bool) -> int:
    settings = get_settings()
    results: list[bool] = []
    session_id = "gate-" + uuid.uuid4().hex[:8]

    print("")
    print("Resetting the database so the gate measures this run, not the last one.")
    reset_database()

    print("")
    print("ADK session store: " + settings.adk_session_db_url)
    print("Provider adapters: our own BaseLlm subclasses (LiteLLM not installed)")
    print("Topology: orchestrator-dispatched plan steps, not transfer_to_agent")
    print("=" * 74)

    # --- Box 5 (also produces turn 1 for the restart) --------------------
    print("")
    print("Box 5 - the mock BaseLlm short-circuits the real LLM and drives the loop")
    code, out = spawn(
        ["--turn", "--session", session_id, "--provider", "mock",
         "--email", MOCK_PATIENT, "--text", BOOKING]
    )
    first_pid = parse(out, "PID")
    first_reply = parse(out, "REPLY")
    first_status = parse(out, "STATUS")
    first_steps = parse(out, "STEPS")
    tool_calls = int(parse(out, "TOOL_CALLS") or 0)

    results.append(
        report(
            "mock drives the whole workflow with no network call",
            code == 0 and tool_calls > 0 and bool(first_reply),
            (
                "pid " + first_pid + ", status " + first_status
                + ", steps [" + first_steps + "], " + str(tool_calls) + " tool calls\n"
                + "reply: " + repr(first_reply[:90])
            )
            if code == 0
            else out[-700:],
        )
    )

    # --- Box 2 -----------------------------------------------------------
    print("")
    print("Box 2 - trace rows from multiple agents in one hub-and-spoke turn")
    agents = [a for a in parse(out, "AGENTS").split(",") if a]
    rows = parse(out, "TRACE_ROWS")

    check = SessionLocal()
    try:
        # Scoped to this run's session: rows left by an earlier run must never
        # be able to vouch for this one.
        turn_ids = {
            row.turn_id
            for row in check.query(TraceEvent)
            .filter(TraceEvent.session_id == session_id)
            .all()
        }
        malformed = [
            r for t in turn_ids for r in check_session(check, turn_id=t) if not r.ok
        ]
    finally:
        check.close()

    results.append(
        report(
            "more than one agent appears in this session's trace",
            len(agents) >= 2,
            rows + " trace rows scoped to " + session_id + "\nagents: " + str(agents),
        )
    )

    # --- Boxes 3 and 4: a genuinely new process --------------------------
    print("")
    print("Boxes 3 & 4 - restart the process; the follow-up works, state survives")
    code2, out2 = spawn(
        ["--turn", "--session", session_id, "--provider", "mock",
         "--email", MOCK_PATIENT, "--text", CONFIRM]
    )
    second_pid = parse(out2, "PID")
    second_status = parse(out2, "STATUS")
    second_steps = parse(out2, "STEPS")
    replayed = parse(out2, "PRE_ADK_EVENTS") or "0"

    results.append(
        report(
            "a follow-up works after a real process restart",
            code2 == 0
            and second_pid != first_pid
            and second_status == "completed",
            (
                "pid " + first_pid + " -> " + second_pid + "; the confirmation landed "
                "on a run the first process created\n"
                + "status " + first_status + " -> " + second_status
                + ", steps [" + second_steps + "]"
            )
            if code2 == 0
            else out2[-700:],
        )
    )

    # Box 4. The values below are written only while routing and proposing —
    # both of which happened in pid 1 — and are read by pid 2 *before* it acts.
    # The earlier version of this script was fooled by a key every turn writes;
    # nothing here is written by the reading process before it reads.
    pre_dept = parse(out2, "PRE_DEPT")
    pre_slot = parse(out2, "PRE_SLOT")
    pre_status = parse(out2, "PRE_STATUS")

    results.append(
        report(
            "state written before the restart is readable after it",
            bool(pre_dept) and bool(pre_slot) and int(replayed) > 0,
            (
                "pid " + second_pid + " read, before acting: status=" + pre_status
                + " department=" + repr(pre_dept) + " proposed_slot=" + repr(pre_slot)
                + "\n" + replayed + " ADK conversation events replayed from "
                "sqlite+aiosqlite\n"
                "(department and proposed slot are written only during routing and "
                "proposal, both in pid " + first_pid + ")"
            ),
        )
    )

    # --- Box 1 -----------------------------------------------------------
    print("")
    print("Box 1 - an agent answers via Groq through our own BaseLlm adapter")
    if not live:
        results.append(report("live provider call", False, "skipped (run with --live)"))
    elif not settings.groq_api_key:
        results.append(
            report("live provider call", False, "no GROQ_API_KEY in the environment")
        )
    else:
        live_session = "live-" + uuid.uuid4().hex[:8]
        code3, out3 = spawn(
            ["--turn", "--session", live_session, "--provider", "groq",
             "--email", LIVE_PATIENT, "--text", LIVE_MESSAGE]
        )
        live_reply = parse(out3, "REPLY")
        live_tools = int(parse(out3, "TOOL_CALLS") or 0)
        live_agents = [a for a in parse(out3, "AGENTS").split(",") if a]

        from app.providers.groq_provider import GroqLlm

        chain = " -> ".join(c.__name__ for c in GroqLlm.__mro__[:4])
        results.append(
            report(
                "a live Groq call through GroqLlm, not LiteLLM",
                code3 == 0 and bool(live_reply) and live_tools > 0,
                (
                    "adapter: " + chain + "\n"
                    "model: " + settings.groq_model + ", " + str(live_tools)
                    + " tool calls, agents: " + str(live_agents) + "\n"
                    "reply: " + repr(live_reply[:90])
                )
                if code3 == 0
                else out3[-900:],
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
        print("Not all boxes ticked - report the evidence, do not push through it.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="include the live provider box")
    parser.add_argument("--turn", action="store_true", help="internal: run one turn and exit")
    parser.add_argument("--session", default="")
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--email", default=MOCK_PATIENT)
    parser.add_argument("--text", default="")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONUTF8", "1")

    if args.turn:
        asyncio.run(
            run_turn_only(args.provider, args.session, args.text, args.email)
        )
        raise SystemExit(0)

    raise SystemExit(main(args.live))
