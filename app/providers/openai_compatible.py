"""Shared adapter for OpenAI-compatible chat APIs.

Groq and OpenAI expose the same request shape, so the translation between ADK's
genai types and chat-completions lives here once and the two concrete providers
differ only in how they build a client.

The translation is the whole job, and it runs in both directions:

* ADK gives us ``contents`` — genai ``Content`` objects whose parts may be text,
  a ``function_call``, or a ``function_response`` — plus tool declarations
  carrying genai ``Schema`` objects. Chat APIs want ``messages`` with
  ``tool_calls``/``tool`` roles and JSON Schema.
* Coming back, ``tool_calls`` become ``function_call`` parts so ADK's runner can
  dispatch them.

Owning this is why the trace can record exact request and response payloads:
nothing passes through a third-party callback we would have to reverse-engineer.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator

from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from app.errors import ProviderError, RateLimited
from app.providers.base import AgentCareLlm, function_call_response, text_response

#: Retries for genuinely broken calls. Rate limits do NOT consume one of these —
#: a retry budget exists for failures, and being throttled is not a failure.
MAX_ATTEMPTS = 3
#: Separate, larger allowance for 429s, which resolve by waiting.
MAX_RATE_LIMIT_WAITS = 3
DEFAULT_RETRY_AFTER = 2.0


def _json_schema(schema: Any) -> dict[str, Any]:
    """Convert a genai ``Schema`` to JSON Schema.

    genai spells types as upper-case enums (``STRING``, ``OBJECT``); JSON Schema
    wants them lower-case. Everything else survives the round trip unchanged.
    """
    if schema is None:
        return {"type": "object", "properties": {}}

    raw = schema.model_dump(exclude_none=True, mode="json") if hasattr(schema, "model_dump") else dict(schema)

    def normalise(node: Any) -> Any:
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                if key == "type" and isinstance(value, str):
                    out[key] = value.lower()
                else:
                    out[key] = normalise(value)
            return out
        if isinstance(node, list):
            return [normalise(item) for item in node]
        return node

    return normalise(raw)


def _parameters(declaration: Any) -> dict[str, Any]:
    """The argument schema, whichever of the two shapes ADK built it in.

    ADK declares a function either as a genai ``Schema`` in ``parameters`` or
    as plain JSON Schema in ``parameters_json_schema``, depending on a feature
    flag. The installed version fills the second and leaves the first ``None``,
    so reading only ``parameters`` sent every tool to the provider declaring no
    arguments at all — leaving the model to infer the shape from the docstring.
    That is how ``submit_plan`` arrived holding a list of objects rather than a
    list of step names, and it cost a turn each time: rejected, resubmitted
    identically, until the iteration budget fired.

    It was invisible locally because the mock provider does not read schemas.
    """
    json_schema = getattr(declaration, "parameters_json_schema", None)
    if json_schema:
        return _json_schema(json_schema)
    return _json_schema(declaration.parameters)


def _tools_payload(llm_request: LlmRequest) -> list[dict[str, Any]]:
    """Tool declarations in chat-completions form."""
    tools: list[dict[str, Any]] = []
    config = llm_request.config
    for tool in (getattr(config, "tools", None) or []):
        for declaration in (getattr(tool, "function_declarations", None) or []):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": declaration.name,
                        "description": declaration.description or "",
                        "parameters": _parameters(declaration),
                    },
                }
            )
    return tools


def _messages_payload(llm_request: LlmRequest) -> list[dict[str, Any]]:
    """Translate genai contents into chat-completions messages.

    Tool call ids are synthesised and matched by position: a ``function_response``
    binds to the most recent unmatched ``function_call`` of the same name. The
    genai side does not always carry an id, and the chat API requires one.

    A response whose call is not in this request is re-paired with a
    synthesised one rather than emitted alone — see the comment below. The
    invariant the whole function owes its caller is that **every** tool message
    has an assistant ``tool_call`` bearing its id before it.
    """
    messages: list[dict[str, Any]] = []

    system = getattr(llm_request.config, "system_instruction", None)
    if system:
        text = system if isinstance(system, str) else " ".join(
            part.text for part in getattr(system, "parts", []) if getattr(part, "text", None)
        )
        if text:
            messages.append({"role": "system", "content": text})

    pending: dict[str, list[str]] = {}
    counter = 0

    for content in llm_request.contents or []:
        text_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for part in content.parts or []:
            if getattr(part, "text", None):
                text_chunks.append(part.text)

            call = getattr(part, "function_call", None)
            if call is not None:
                counter += 1
                call_id = f"call_{counter}"
                pending.setdefault(call.name, []).append(call_id)
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(dict(call.args or {}), default=str),
                        },
                    }
                )

            response = getattr(part, "function_response", None)
            if response is not None:
                name = response.name or "unknown_tool"
                queued = pending.get(response.name or "", [])
                if queued:
                    call_id = queued.pop(0)
                else:
                    # An orphan: the call this answers is not in this request.
                    # It happens where a history begins mid-turn — a stage
                    # boundary, or a window that started at a tool result,
                    # since ADK carries results as ``role="user"`` contents.
                    # The chat API rejects a tool message with no assistant
                    # tool_call before it, and says so as a 400 naming the
                    # index rather than the cause. Re-paired rather than
                    # dropped: the result is real and the model needs it, and
                    # only the call's arguments are unrecoverable.
                    counter += 1
                    call_id = f"call_{counter}"
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": "{}"},
                                }
                            ],
                        }
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(response.response or {}, default=str),
                    }
                )

        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": " ".join(text_chunks) or None,
                    "tool_calls": tool_calls,
                }
            )
        elif text_chunks:
            role = "assistant" if content.role == "model" else "user"
            messages.append({"role": role, "content": " ".join(text_chunks)})

    return messages


def _to_llm_response(message: Any) -> LlmResponse:
    """Turn a chat-completions message back into an ADK response."""
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        call = tool_calls[0]
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            # A malformed argument blob is a model error, not a crash: hand it
            # back as an empty call so the validation layer rejects it into the
            # retry ladder rather than the process dying here.
            args = {}
        return function_call_response(call.function.name, args)

    return text_response(message.content or "")


class OpenAICompatibleLlm(AgentCareLlm):
    """Base for the two live providers."""

    #: Set by subclasses.
    provider_name: str = "openai-compatible"

    def _client(self):  # pragma: no cover - overridden
        raise NotImplementedError

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        client = self._client()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_payload(llm_request),
            "temperature": 0,
        }
        tools = _tools_payload(llm_request)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        attempts = 0
        rate_limit_waits = 0
        last_error: Exception | None = None

        while attempts < MAX_ATTEMPTS and rate_limit_waits < MAX_RATE_LIMIT_WAITS:
            try:
                completion = client.chat.completions.create(**payload)
                yield _to_llm_response(completion.choices[0].message)
                return
            except Exception as exc:  # noqa: BLE001 - re-raised below, classified first
                last_error = exc
                if _is_rate_limit(exc):
                    # Throttling is not a failure: wait it out without spending
                    # a retry slot that exists for genuinely broken calls.
                    rate_limit_waits += 1
                    time.sleep(_retry_after(exc))
                    continue
                if not _is_retryable(exc):
                    raise ProviderError(f"{self.provider_name} call failed: {exc}") from exc
                attempts += 1
                time.sleep(min(2**attempts, 8))

        if last_error is not None and _is_rate_limit(last_error):
            raise RateLimited(_retry_after(last_error))
        raise ProviderError(
            f"{self.provider_name} call failed after {attempts} attempts: {last_error}"
        )


def _is_rate_limit(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return status == 429 or exc.__class__.__name__ == "RateLimitError"


def _retry_after(exc: Exception) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    for key in ("retry-after", "Retry-After"):
        if key in headers:
            try:
                return float(headers[key])
            except (TypeError, ValueError):
                break
    return DEFAULT_RETRY_AFTER


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None and 500 <= int(status) < 600:
        return True
    return exc.__class__.__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
    }
