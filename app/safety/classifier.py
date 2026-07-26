"""The LLM half of the hybrid screen — a second opinion, never the first.

The phrase list in :mod:`app.safety.screen` catches what can be written down.
This catches what cannot: "there's a lot of pressure across my chest since this
morning" carries no phrase from that list and is the same emergency.

Three properties make this a guard rather than an agent:

* **It runs second.** Whatever it says, the deterministic screen has already
  had its turn, so a model that has been argued out of its instructions cannot
  unblock a message the phrase list already blocked.
* **It proposes a class, and only a class.** The verdict is one of a closed
  set, validated by code before anything is applied. It never writes a reply,
  never touches a run, and never calls a tool that changes anything.
* **It is not in the topology.** Nothing delegates to it and it delegates to
  nothing; it has no plan and no place in the hub-and-spoke.

It is driven through :func:`app.agents.base.run_agent` on purpose. Sharing the
runner means sharing the nine capture points, the iteration budget, and the
provider seam — a guard with its own private call path would be the one part of
the system whose model calls did not appear in the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agents.base import build, run_agent
from app.agents.callbacks import TurnCallbacks
from app.agents.memory import task_service
from app.safety.screen import PASSED, SafetyCategory, SafetyVerdict
from app.trace import TraceWriter

#: The closed set the model is allowed to answer with. ``safe`` is a real
#: verdict, not the absence of one: "the classifier said safe" and "the
#: classifier never ran" are different facts and the trace must tell them apart.
SAFETY_CLASSES = ("emergency", "clinical_advice", "safe")


@dataclass
class _Holder:
    """Where the validated proposal lands, for the caller to read.

    The verdict is read from here rather than parsed out of the agent's reply,
    for the same reason plans are: a proposal that exists only in prose is a
    proposal nobody can validate.
    """

    verdict: SafetyVerdict | None = None
    rejections: list[str] = field(default_factory=list)


def _tools(holder: _Holder, writer: TraceWriter) -> list:
    def submit_safety_verdict(category: str, rationale: str) -> dict:
        """Say whether this message is safe for an administrative assistant to
        handle. `category` must be one of: emergency, clinical_advice, safe.
        `rationale` is one short administrative sentence."""
        proposed = str(category or "").strip().lower()
        if proposed not in SAFETY_CLASSES:
            writer.validation(
                "safety_class",
                accepted=False,
                detail={"proposed": category, "problem": "unknown class"},
            )
            holder.rejections.append(f"{category!r} is not a safety class")
            return {
                "accepted": False,
                "problem": f"Not a safety class. Use one of: {', '.join(SAFETY_CLASSES)}.",
            }

        writer.validation(
            "safety_class", accepted=True, detail={"class": proposed}
        )
        holder.verdict = (
            PASSED
            if proposed == "safe"
            else SafetyVerdict(
                fired=True,
                category=SafetyCategory(proposed),
                rule="llm_classification",
                source="llm",
                detail=str(rationale or "")[:200],
            )
        )
        return {"accepted": True, "category": proposed}

    return [submit_safety_verdict]


async def llm_screen(
    message: str,
    *,
    callbacks: TurnCallbacks,
    writer: TraceWriter,
    user_id: str,
    provider: str | None = None,
) -> SafetyVerdict:
    """Ask the model whether this message is safe to handle administratively.

    Returns :data:`app.safety.screen.PASSED` when the model says so *and* when
    it says nothing usable. That default is deliberate and it is the weaker
    direction on purpose: this layer exists to catch what the phrase list
    missed, so a silent model leaves the deterministic verdict standing rather
    than replacing it with a guess.
    """
    holder = _Holder()
    agent = build(
        name="safety_screen",
        prompt_key="safety",
        description="Guardrail: classifies one message for safety. Not an agent "
        "in the workflow topology and never acts on anything.",
        tools=_tools(holder, writer),
        callbacks=callbacks,
        provider=provider,
    )

    await run_agent(
        agent,
        task_text=message,
        callbacks=callbacks,
        session_service=task_service(),
        # A throwaway session: the screen judges the message in front of it and
        # has no business reading the conversation it arrived in.
        session_id=f"{writer.turn_id}-safety",
        user_id=user_id,
        create=True,
    )

    return holder.verdict if holder.verdict is not None else PASSED


__all__ = ["SAFETY_CLASSES", "llm_screen"]
