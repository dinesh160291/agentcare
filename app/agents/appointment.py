"""The Appointment specialist.

Finds times and *proposes* one. It never books: ``propose_appointment`` writes
a typed proposal onto the run row and pauses the workflow, and the commit
happens only after code has read the patient's confirmation deterministically.

Schemas catch malformed arguments; only confirmation catches well-formed
fabrications. A model handed "next week" with no other anchor can resolve it
differently across two runs while staying schema-valid and sounding certain —
which is why the date comes from ``resolve_date`` and the booking comes from a
human saying yes.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from app.agents import base
from app.agents.callbacks import TurnCallbacks
from app.agents.toolbelt import Toolbelt

NAME = "appointment"


def build_agent(
    toolbelt: Toolbelt, callbacks: TurnCallbacks, *, provider: str | None = None
) -> LlmAgent:
    return base.build(
        name=NAME,
        prompt_key="appointment",
        description="Finds appointment times and proposes one for confirmation.",
        tools=toolbelt.appointment_tools(),
        callbacks=callbacks,
        provider=provider,
    )


__all__ = ["NAME", "build_agent"]
