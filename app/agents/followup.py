"""The Follow-up specialist.

Summarises what is outstanding: reminders due, tasks still open, what the
patient should expect next. Its inputs are the other agents' outputs made
durable — the appointment id and the missing-documents list — which is the
cross-agent data flow arriving through persisted rows rather than through
peer-to-peer chatter.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from app.agents import base
from app.agents.callbacks import TurnCallbacks
from app.agents.toolbelt import Toolbelt

NAME = "followup"


def build_agent(
    toolbelt: Toolbelt, callbacks: TurnCallbacks, *, provider: str | None = None
) -> LlmAgent:
    return base.build(
        name=NAME,
        prompt_key="followup",
        description="Summarises reminders, outstanding tasks, and next steps.",
        tools=toolbelt.followup_tools(),
        callbacks=callbacks,
        provider=provider,
    )


__all__ = ["NAME", "build_agent"]
