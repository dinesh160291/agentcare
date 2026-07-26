"""The Department Routing specialist.

The one specialist whose task carries the patient's own words, because reading
them *is* the job. It still receives no transcript — the current request text
rides inside the typed task, and nothing else does.

It proposes a department; ``validate_department`` disposes, matching exactly
against the Department table. A model-invented department never reaches a slot
search, and a low-confidence match becomes a staff review rather than a
confident guess.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from app.agents import base
from app.agents.callbacks import TurnCallbacks
from app.agents.toolbelt import Toolbelt

NAME = "department_routing"


def build_agent(
    toolbelt: Toolbelt, callbacks: TurnCallbacks, *, provider: str | None = None
) -> LlmAgent:
    return base.build(
        name=NAME,
        prompt_key="routing",
        description="Works out which hospital department handles a request.",
        tools=toolbelt.routing_tools(),
        callbacks=callbacks,
        provider=provider,
    )


__all__ = ["NAME", "build_agent"]
