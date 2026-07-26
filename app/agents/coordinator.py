"""The Coordinator — the hub.

It is the only agent that sees the conversation. Its job is to decide what a
request needs and say so as a *plan*, or, when a request is already live, to
say how the new message relates to it. Both answers arrive as tool calls
carrying typed arguments, never as prose: code cannot enforce dependency order
on a plan it cannot parse.

It holds no domain tools. It cannot book, route, or file anything — which is
what keeps "decides what is needed" and "does it" in different agents.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from app.agents import base
from app.agents.callbacks import TurnCallbacks
from app.agents.toolbelt import Toolbelt

NAME = "coordinator"


def build_agent(
    toolbelt: Toolbelt, callbacks: TurnCallbacks, *, provider: str | None = None
) -> LlmAgent:
    return base.build(
        name=NAME,
        prompt_key="coordinator",
        description="Decides which specialists a patient's request needs.",
        tools=toolbelt.coordinator_tools(),
        callbacks=callbacks,
        provider=provider,
    )


__all__ = ["NAME", "build_agent"]
