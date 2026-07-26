"""The provider seam.

All three providers — ``mock``, ``groq``, ``openai`` — are subclasses of
``google.adk.models.BaseLlm``. None of them goes through LiteLLM, which does
not install on this machine (its build chain wants a Rust toolchain). Owning
the adapters turns out to be the better design anyway: identical plumbing for
all three, and exact request/response capture for the trace, because the bytes
pass through our own code rather than a third party's callback.

This module holds what the three share — reading an ``LlmRequest`` and building
an ``LlmResponse`` — so the adapters differ only where the providers genuinely
differ.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types


@dataclass(frozen=True)
class ToolResult:
    """A tool result handed back to the model in a previous turn."""

    name: str
    payload: dict[str, Any]


def latest_user_text(llm_request: LlmRequest) -> str:
    """The most recent thing the user actually typed.

    After a transfer to a sub-agent, ADK hands the specialist a history whose
    user turns may not carry a ``user`` role — the request arrives wrapped in
    transfer machinery. Falling back to the earliest text in the conversation
    recovers the original request, which is what a specialist needs: the
    patient's own words are the input where language is the job.
    """
    for content in reversed(llm_request.contents or []):
        if content.role != "user":
            continue
        text = " ".join(part.text for part in (content.parts or []) if part.text)
        if text.strip():
            return text.strip()

    for content in llm_request.contents or []:
        text = " ".join(part.text for part in (content.parts or []) if part.text)
        if text.strip():
            return text.strip()
    return ""


def tool_results(llm_request: LlmRequest) -> list[ToolResult]:
    """Every tool result in the conversation so far, oldest first.

    This is the mock's evidence base. A reply templated from these is a reply
    templated from what the database actually returned.
    """
    results: list[ToolResult] = []
    for content in llm_request.contents or []:
        for part in content.parts or []:
            response = getattr(part, "function_response", None)
            if response is None:
                continue
            payload = response.response or {}
            if not isinstance(payload, dict):
                payload = {"result": payload}
            results.append(ToolResult(name=response.name or "", payload=payload))
    return results


def latest_tool_result(llm_request: LlmRequest, name: str | None = None) -> ToolResult | None:
    """The most recent tool result, optionally filtered by tool name."""
    for result in reversed(tool_results(llm_request)):
        if name is None or result.name == name:
            return result
    return None


def called_tools(llm_request: LlmRequest) -> set[str]:
    """Names of every tool already called this turn."""
    return {result.name for result in tool_results(llm_request)}


def available_tool_names(llm_request: LlmRequest) -> set[str]:
    """Tools the agent has been given, from the request's tool dictionary."""
    return set((llm_request.tools_dict or {}).keys())


def system_instruction_text(llm_request: LlmRequest) -> str:
    """The system instruction as plain text, however it was supplied."""
    instruction = getattr(llm_request.config, "system_instruction", None)
    if instruction is None:
        return ""
    if isinstance(instruction, str):
        return instruction
    parts = getattr(instruction, "parts", None) or []
    return " ".join(part.text for part in parts if getattr(part, "text", None))


#: ADK advertises delegation targets in the system instruction it builds, in
#: the form "Agent name: <name>". Reading them from there is exactly how a real
#: model learns which agents exist — there is no enum on the tool declaration.
_AGENT_NAME = re.compile(r"Agent name:\s*(\S+)")


def transferable_agents(llm_request: LlmRequest) -> list[str]:
    """Sub-agent names this agent is allowed to transfer to."""
    return _AGENT_NAME.findall(system_instruction_text(llm_request))


def text_response(text: str) -> LlmResponse:
    """A final, user-facing reply."""
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        turn_complete=True,
    )


def function_call_response(name: str, args: dict[str, Any]) -> LlmResponse:
    """A request to run a tool.

    The mock emits these for real: under ``LLM_PROVIDER=mock`` the tools run,
    the database is written, and the reply is built from what came back. A mock
    that skipped this and returned prose would be a hardcoded final response,
    which scores zero.
    """
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
        ),
        turn_complete=False,
    )


def request_snapshot(llm_request: LlmRequest) -> dict[str, Any]:
    """A JSON-able record of an outgoing request, for the trace.

    Captured identically for every provider, so mock-mode and live-mode traces
    are shape-identical and can be diffed against each other.
    """
    messages: list[dict[str, Any]] = []
    for content in llm_request.contents or []:
        entry: dict[str, Any] = {"role": content.role, "text": "", "function_calls": [],
                                 "function_responses": []}
        for part in content.parts or []:
            if part.text:
                entry["text"] += part.text
            call = getattr(part, "function_call", None)
            if call is not None:
                entry["function_calls"].append({"name": call.name, "args": dict(call.args or {})})
            response = getattr(part, "function_response", None)
            if response is not None:
                entry["function_responses"].append({"name": response.name})
        messages.append(entry)

    return {
        "model": llm_request.model,
        "tools": sorted(available_tool_names(llm_request)),
        "messages": messages,
    }


def response_snapshot(response: LlmResponse) -> dict[str, Any]:
    """A JSON-able record of a response, for the trace."""
    text = ""
    calls: list[dict[str, Any]] = []
    for part in (response.content.parts if response.content else []) or []:
        if part.text:
            text += part.text
        call = getattr(part, "function_call", None)
        if call is not None:
            calls.append({"name": call.name, "args": dict(call.args or {})})
    return {"text": text, "function_calls": calls, "turn_complete": response.turn_complete}


class AgentCareLlm(BaseLlm):
    """Base for our three adapters.

    Subclasses implement ``generate_content_async``. Everything above is shared
    so the three differ only in how they reach a model — or, for the mock, in
    not reaching one at all.
    """

    @classmethod
    def supported_models(cls) -> list[str]:
        return []
