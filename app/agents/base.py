"""The one place an ``LlmAgent`` is constructed.

Every agent is the same six decisions — name, prompt, tools, model, callbacks —
and the callbacks are the trace capture points. Building them in five places
means four places to forget one, and a forgotten callback does not fail: it
produces a turn that runs correctly and cannot be explained afterwards.
"""

from __future__ import annotations

from typing import Callable, Sequence

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from app.agents import prompts
from app.agents.callbacks import TurnCallbacks
from app.providers import get_provider


def build(
    *,
    name: str,
    prompt_key: str,
    description: str,
    tools: Sequence[Callable],
    callbacks: TurnCallbacks,
    provider: str | None = None,
) -> LlmAgent:
    """Construct one agent with its capture points already wired.

    Note what is *not* here: ``sub_agents``. Delegation is dispatched by the
    orchestrator from a validated plan rather than by ADK's
    ``transfer_to_agent``, because a transfer hands the sub-agent the whole
    session history — and the context contract says specialists receive no
    history at all, only a typed task. The spike also showed what unbounded
    peer-to-peer transfer costs: agents bouncing work onward until ADK's
    session store failed.
    """
    return LlmAgent(
        name=name,
        model=get_provider(provider),
        description=description,
        instruction=prompts.instruction(prompt_key),
        tools=[FunctionTool(tool) for tool in tools],
        before_model_callback=callbacks.before_model,
        after_model_callback=callbacks.after_model,
        before_tool_callback=callbacks.before_tool,
        after_tool_callback=callbacks.after_tool,
    )


__all__ = ["build"]
