"""The provider seam's translation layer and its ADK-shape assumptions.

Two things are pinned here, and both are pinned because they depend on
somebody else's wording or format rather than on our own code:

* **ADK's transfer narration.** After a transfer, ADK does not hand the
  sub-agent a ``function_response`` — it injects ``role="user"`` prose ("For
  context: [coordinator] called tool ...`"). We filter that out so a specialist
  receives the patient's words. If an ADK upgrade rewords the narration, these
  tests fail loudly instead of the specialist quietly classifying the
  delegation machinery instead of the request.
* **The chat-completions translation.** genai types in, messages and JSON
  Schema out, tool calls back again. Exercised with recorded shapes so it is
  tested without a network call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from google.adk.models import LlmRequest
from google.genai import types

from app.providers.base import (
    ADK_CONTEXT_MARKERS,
    is_framing,
    latest_user_text,
    request_snapshot,
    response_snapshot,
    text_response,
    transferable_agents,
)
from app.providers.openai_compatible import (
    _json_schema,
    _messages_payload,
    _to_llm_response,
    _is_rate_limit,
    _retry_after,
)

#: Recorded verbatim from a real ADK run (see docs/framework-gate.md). The
#: specialist's history after `transfer_to_agent`.
TRANSFER_HISTORY = [
    types.Content(
        role="user", parts=[types.Part(text="I need a cardiology appointment next week")]
    ),
    types.Content(
        role="user",
        parts=[
            types.Part(text="For context:"),
            types.Part(
                text="[coordinator] called tool `transfer_to_agent` with parameters: "
                "{'agent_name': 'department_specialist'}"
            ),
        ],
    ),
    types.Content(
        role="user",
        parts=[
            types.Part(text="For context:"),
            types.Part(
                text="[coordinator] `transfer_to_agent` tool returned result: {'result': None}"
            ),
        ],
    ),
]


class TestTransferNarration:
    def test_the_specialist_receives_the_patients_words(self):
        """The bug this pins: without filtering, the specialist classifies
        ADK's delegation narration instead of the request, and routing decides
        a department from the word "transfer_to_agent"."""
        request = LlmRequest(model="m", contents=TRANSFER_HISTORY)
        assert latest_user_text(request) == "I need a cardiology appointment next week"

    def test_every_recorded_narration_line_is_recognised_as_framing(self):
        for content in TRANSFER_HISTORY[1:]:
            text = " ".join(p.text for p in content.parts if p.text)
            assert is_framing(text), f"not recognised as framing: {text!r}"

    def test_a_genuine_message_is_not_mistaken_for_framing(self):
        assert is_framing("I need a cardiology appointment next week") is False

    def test_the_markers_are_not_empty(self):
        """An empty marker list would silently filter nothing."""
        assert ADK_CONTEXT_MARKERS
        assert all(marker.strip() for marker in ADK_CONTEXT_MARKERS)

    def test_a_history_of_pure_framing_still_yields_something(self):
        request = LlmRequest(model="m", contents=TRANSFER_HISTORY[1:])
        assert latest_user_text(request) == ""

    def test_agent_names_are_read_from_the_system_instruction(self):
        """ADK advertises delegation targets there; the tool declaration has no
        enum, so this is the only place the names appear."""
        config = types.GenerateContentConfig(
            system_instruction=(
                "You coordinate.\n\nAgent name: department_specialist\n"
                "Agent description: Answers department questions.\n"
            )
        )
        request = LlmRequest(model="m", contents=[], config=config)
        assert transferable_agents(request) == ["department_specialist"]


class TestSchemaTranslation:
    def test_genai_type_names_are_lowercased_for_json_schema(self):
        """genai spells types as STRING/OBJECT; JSON Schema wants lower case,
        and a provider given "STRING" rejects the whole tool definition."""
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={"phrase": types.Schema(type=types.Type.STRING)},
            required=["phrase"],
        )
        out = _json_schema(schema)
        assert out["type"] == "object"
        assert out["properties"]["phrase"]["type"] == "string"
        assert out["required"] == ["phrase"]

    def test_a_missing_schema_becomes_an_empty_object(self):
        assert _json_schema(None) == {"type": "object", "properties": {}}

    def test_nested_types_are_lowercased_too(self):
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "slots": types.Schema(
                    type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER)
                )
            },
        )
        out = _json_schema(schema)
        assert out["properties"]["slots"]["type"] == "array"
        assert out["properties"]["slots"]["items"]["type"] == "integer"


class TestMessageTranslation:
    def test_a_plain_exchange_becomes_user_and_assistant_messages(self):
        request = LlmRequest(
            model="m",
            contents=[
                types.Content(role="user", parts=[types.Part(text="hello")]),
                types.Content(role="model", parts=[types.Part(text="hi")]),
            ],
        )
        assert _messages_payload(request) == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_the_system_instruction_becomes_a_system_message(self):
        request = LlmRequest(
            model="m",
            contents=[],
            config=types.GenerateContentConfig(system_instruction="be helpful"),
        )
        assert _messages_payload(request)[0] == {"role": "system", "content": "be helpful"}

    def test_a_function_call_becomes_an_assistant_tool_call(self):
        request = LlmRequest(
            model="m",
            contents=[
                types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name="resolve_date", args={"phrase": "next week"}
                            )
                        )
                    ],
                )
            ],
        )
        message = _messages_payload(request)[0]
        assert message["role"] == "assistant"
        call = message["tool_calls"][0]
        assert call["function"]["name"] == "resolve_date"
        assert json.loads(call["function"]["arguments"]) == {"phrase": "next week"}

    def test_a_function_response_binds_to_its_call_id(self):
        """The chat API requires a tool_call_id; genai does not always carry
        one, so ids are synthesised and matched by position."""
        request = LlmRequest(
            model="m",
            contents=[
                types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(name="resolve_date", args={})
                        )
                    ],
                ),
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name="resolve_date", response={"resolved": True}
                            )
                        )
                    ],
                ),
            ],
        )
        messages = _messages_payload(request)
        tool_message = next(m for m in messages if m["role"] == "tool")
        assistant = next(m for m in messages if m["role"] == "assistant")
        assert tool_message["tool_call_id"] == assistant["tool_calls"][0]["id"]
        assert json.loads(tool_message["content"]) == {"resolved": True}


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    function: FakeFunction


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list | None = None


class TestResponseTranslation:
    def test_plain_text_comes_back_as_text(self):
        response = _to_llm_response(FakeMessage(content="Here are some times."))
        assert response_snapshot(response)["text"] == "Here are some times."

    def test_a_tool_call_comes_back_as_a_function_call(self):
        message = FakeMessage(
            tool_calls=[FakeToolCall(FakeFunction("book_appointment", '{"slot_id": 7}'))]
        )
        snapshot = response_snapshot(_to_llm_response(message))
        assert snapshot["function_calls"] == [
            {"name": "book_appointment", "args": {"slot_id": 7}}
        ]

    def test_malformed_arguments_do_not_crash_the_adapter(self):
        """A model emitting broken JSON is a model error. It belongs in the
        validation layer's retry ladder, not in a traceback here."""
        message = FakeMessage(tool_calls=[FakeToolCall(FakeFunction("book", "{not json"))])
        snapshot = response_snapshot(_to_llm_response(message))
        assert snapshot["function_calls"] == [{"name": "book", "args": {}}]


class TestFailureClassification:
    def test_a_429_is_recognised_as_a_rate_limit(self):
        """Rate limits must not consume a retry slot: retry budgets exist for
        broken calls, and being throttled is not a failure."""

        class Throttled(Exception):
            status_code = 429

        assert _is_rate_limit(Throttled()) is True

    def test_a_500_is_not_a_rate_limit(self):
        class Broken(Exception):
            status_code = 500

        assert _is_rate_limit(Broken()) is False

    def test_retry_after_is_honoured_when_the_provider_sends_it(self):
        class WithHeader(Exception):
            status_code = 429

            class response:  # noqa: N801
                headers = {"retry-after": "7"}

        assert _retry_after(WithHeader()) == 7.0

    def test_a_default_delay_is_used_when_no_header_is_sent(self):
        assert _retry_after(Exception()) > 0


class TestSnapshotsAreProviderIdentical:
    def test_a_request_snapshot_is_json_serialisable(self):
        """Trace payloads are persisted and diffed between mock and live."""
        request = LlmRequest(model="m", contents=TRANSFER_HISTORY)
        json.dumps(request_snapshot(request))

    def test_a_response_snapshot_is_json_serialisable(self):
        json.dumps(response_snapshot(text_response("hello")))

    def test_snapshots_have_stable_keys(self):
        """Mock and live traces are compared by shape, so the keys must not
        depend on which provider produced them."""
        assert set(request_snapshot(LlmRequest(model="m", contents=[]))) == {
            "model", "tools", "messages",
        }
        assert set(response_snapshot(text_response("x"))) == {
            "text", "function_calls", "turn_complete",
        }
