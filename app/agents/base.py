"""The one place an ``LlmAgent`` is constructed.

Every agent is the same six decisions — name, prompt, tools, model, callbacks —
and the callbacks are the trace capture points. Building them in five places
means four places to forget one, and a forgotten callback does not fail: it
produces a turn that runs correctly and cannot be explained afterwards.
"""

from __future__ import annotations

from typing import Callable, Sequence

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.tools import FunctionTool
from google.genai import types

from app.agents import memory, prompts
from app.agents.callbacks import TurnCallbacks
from app.errors import ProviderError
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


async def run_agent(
    agent: LlmAgent,
    *,
    task_text: str,
    callbacks: TurnCallbacks,
    session_service,
    session_id: str,
    user_id: str,
    create: bool,
    terminal_tool: str | None = None,
) -> str:
    """Drive one agent for one task and return whatever text it produced.

    Lives here rather than in the orchestrator because the safety guard drives
    an agent too, and it must do so through exactly the same plumbing — same
    capture points, same budget, same provider seam. A second copy of this loop
    would be a second place for a capture point to go missing.

    A provider failure is paired into the trace before it propagates: a request
    with no terminal partner is supposed to mean the process died mid-call, and
    that diagnosis is only worth anything if nothing else can produce it.

    ``terminal_tool``, where the caller has one, names the tool whose accepted
    result *is* this agent's output — so the loop can stop there rather than
    asking a model to comment on an answer already in hand.
    """
    if create:
        await session_service.create_session(
            app_name=memory.APP_NAME, user_id=user_id, session_id=session_id
        )

    callbacks.start_agent(terminal_tool=terminal_tool)
    runner = Runner(
        agent=agent, app_name=memory.APP_NAME, session_service=session_service
    )
    message = types.Content(role="user", parts=[types.Part(text=task_text)])

    reply = ""
    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=message
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        reply += part.text
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
        callbacks.fail_pending_request(
            f"{type(exc).__name__}: {exc}",
            kind="rate_limit" if isinstance(exc, ProviderError) else "error",
        )
        raise
    return reply.strip()


__all__ = ["build", "run_agent"]
