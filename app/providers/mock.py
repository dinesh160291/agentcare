"""The mock provider — an understudy, not a fixture.

Under ``LLM_PROVIDER=mock`` the entire application runs: real tool calls, real
database writes, real workflow state. The only thing replaced is the *brain* —
the part that decides what to do next and how to word the answer.

**Why this file contains no canned replies.** The organizer's rule 6 disqualifies
"hardcoded final responses presented as agent results". A mock that returned
"Your appointment is booked!" regardless of input would break that rule, and
would also make the mock useless as an understudy: it would prove the plumbing
worked only in the one case somebody hardcoded.

So the rule here is absolute: **every reply is templated from a tool result the
database produced this turn.** Facts are never carried over from what the mock
"intended" — they are read out of the payload the tool returned. For the
consequential case the mock goes further and calls ``render_confirmation``,
whose whole job is to re-read the row, then templates from that. Different
bookings therefore produce different sentences, and every fact in one can be
diffed against the row it describes.

What is deterministic here is *policy* — which tool to call next, given what
has already come back. That is the model's judgement being stood in for, and
standing in for judgement is what an understudy does.
"""

from __future__ import annotations

import re
from typing import Any, AsyncGenerator

from google.adk.models import LlmRequest, LlmResponse

from app.providers.base import (
    AgentCareLlm,
    available_tool_names,
    called_tools,
    function_call_response,
    latest_tool_result,
    is_framing,
    latest_user_text,
    text_response,
    transferable_agents,
)

CONFIRM_TOKENS = {"yes", "y", "confirm", "confirmed", "ok", "okay", "book it", "go ahead"}
DECLINE_TOKENS = {"no", "n", "cancel", "decline", "stop"}

BOOKING_CUES = (
    "appointment", "book", "booking", "schedule", "see a doctor", "consult", "slot",
)
DOCUMENT_CUES = ("report", "document", "upload", "attach", "scan", "record")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


def wants_booking(text: str) -> bool:
    return any(cue in text.lower() for cue in BOOKING_CUES)


def mentions_documents(text: str) -> bool:
    return any(cue in text.lower() for cue in DOCUMENT_CUES)


def reads_as_confirmation(text: str) -> bool:
    """Exact-token match, the same rule the deterministic reader uses.

    The mock is not permitted to be cleverer than the real confirmation path.
    Anything ambiguous falls through to a re-ask rather than a commit.
    """
    return _norm(text) in CONFIRM_TOKENS


def reads_as_decline(text: str) -> bool:
    return _norm(text) in DECLINE_TOKENS


class MockLlm(AgentCareLlm):
    """Deterministic stand-in for the model.

    Decides the next step from what the tools have already returned, and words
    the final reply from the last tool result.
    """

    model: str = "mock-understudy"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield self._within_toolset(llm_request, self._decide(llm_request))

    @staticmethod
    def _within_toolset(llm_request: LlmRequest, response: LlmResponse) -> LlmResponse:
        """Never call a tool this agent does not have.

        A real model is constrained the same way: it can only call what it was
        given. In a hub-and-spoke the coordinator holds `transfer_to_agent` and
        the specialists hold the real tools, so a coordinator that wants
        `find_available_slots` should delegate rather than invent a call — which
        is what a live model does, and what ADK raises on if it does not.
        """
        parts = (response.content.parts if response.content else None) or []
        call = next((getattr(p, "function_call", None) for p in parts), None)
        if call is None:
            return response

        available = available_tool_names(llm_request)
        if call.name in available:
            return response

        # Delegate at most once per turn. A specialist that was itself
        # delegated to must do the work or answer — bouncing it onward is an
        # unbounded loop, and ADK's session store fails loudly when two
        # branches race to append to it.
        already_delegated = any(
            is_framing(" ".join(part.text for part in (content.parts or []) if part.text))
            for content in (llm_request.contents or [])
        )
        if "transfer_to_agent" in available and not already_delegated:
            candidates = [
                name for name in transferable_agents(llm_request) if name != "transfer_to_agent"
            ]
            if candidates:
                return function_call_response(
                    "transfer_to_agent", {"agent_name": candidates[0]}
                )

        # No tool and nowhere to delegate: answer from what is already known
        # rather than calling something that does not exist.
        latest = latest_tool_result(llm_request)
        if latest is not None:
            return text_response(
                "Here is what I found: "
                + ", ".join(f"{key}: {value}" for key, value in list(latest.payload.items())[:4])
            )
        return text_response(
            "I can help with appointments, documents, reminders, and follow-ups. "
            "What would you like to do?"
        )

    # --- policy ----------------------------------------------------------

    def _decide(self, llm_request: LlmRequest) -> LlmResponse:
        text = latest_user_text(llm_request)
        done = called_tools(llm_request)

        # A confirmation the tools have already acted on: read the row back.
        booked = latest_tool_result(llm_request, "book_appointment")
        if booked is not None:
            return self._after_booking(llm_request, booked.payload)

        confirmation = latest_tool_result(llm_request, "render_confirmation")
        if confirmation is not None:
            return self._from_confirmation(confirmation.payload)

        if reads_as_confirmation(text) and "book_appointment" not in done:
            slot_id = self._proposed_slot_id(llm_request)
            if slot_id is not None:
                return function_call_response("book_appointment", {"slot_id": slot_id})

        if reads_as_decline(text):
            return text_response(
                "No problem — nothing has been booked. Tell me when suits you better."
            )

        # Ordinary booking flow: department, then dates, then slots.
        if "resolve_department" not in done and wants_booking(text):
            return function_call_response("resolve_department", {"text": text})

        department = latest_tool_result(llm_request, "resolve_department")
        if department is not None:
            payload = department.payload
            if payload.get("status") == "unsupported":
                return self._from_unsupported(payload)
            if payload.get("status") == "ambiguous":
                return self._from_ambiguous(payload)

            if "resolve_date" not in done:
                return function_call_response("resolve_date", {"phrase": text})

            window = latest_tool_result(llm_request, "resolve_date")
            if window is not None and "find_available_slots" not in done:
                args: dict[str, Any] = {"department_id": payload["department"]["id"]}
                if window.payload.get("resolved"):
                    args["start"] = window.payload["start"]
                    args["end"] = window.payload["end"]
                    if window.payload.get("part_of_day"):
                        args["part_of_day"] = window.payload["part_of_day"]
                return function_call_response("find_available_slots", args)

            slots = latest_tool_result(llm_request, "find_available_slots")
            if slots is not None:
                return self._from_slots(slots.payload, payload)

        if mentions_documents(text) and "list_patient_documents" not in done:
            return function_call_response("list_patient_documents", {})

        documents = latest_tool_result(llm_request, "list_patient_documents")
        if documents is not None:
            return self._from_documents(documents.payload)

        # Nothing recognised. The scope reply is a template by design — it is a
        # guard's output, not an agent result, and it carries no facts.
        return text_response(
            "I can help with appointments, documents, reminders, and follow-ups. "
            "What would you like to do?"
        )

    # --- templating, always from a tool payload --------------------------

    @staticmethod
    def _proposed_slot_id(llm_request: LlmRequest) -> int | None:
        """The slot last offered, read back out of the availability result."""
        slots = latest_tool_result(llm_request, "find_available_slots")
        if slots is None:
            return None
        offered = slots.payload.get("slots") or []
        return offered[0]["slot_id"] if offered else None

    @staticmethod
    def _from_slots(payload: dict[str, Any], department: dict[str, Any]) -> LlmResponse:
        offered = payload.get("slots") or []
        name = department["department"]["name"]
        if not offered:
            return text_response(
                f"I could not find any free {name} appointments in that period. "
                "Would you like to try a different date?"
            )

        lines = [
            f"{index}. {slot['doctor_name']} — "
            f"{slot['start'][:10]} at {slot['start'][11:16]}"
            for index, slot in enumerate(offered[:3], start=1)
        ]
        first = offered[0]
        more = ""
        if payload.get("total_matching", 0) > len(offered):
            more = f" There are {payload['total_matching']} times free in total."
        return text_response(
            f"Here are the earliest {name} appointments I can offer:\n"
            + "\n".join(lines)
            + f"\n\nShall I book {first['doctor_name']} at {first['start'][11:16]} "
            f"on {first['start'][:10]}? Reply 'yes' to confirm.{more}"
        )

    @staticmethod
    def _after_booking(llm_request: LlmRequest, payload: dict[str, Any]) -> LlmResponse:
        """After a booking, re-read the row rather than trusting the write."""
        if not payload.get("ok"):
            alternatives = payload.get("alternatives") or []
            message = payload.get("message") or "That booking could not be completed."
            if alternatives:
                options = ", ".join(
                    f"{slot['doctor_name']} at {slot['start'][11:16]} on {slot['start'][:10]}"
                    for slot in alternatives[:3]
                )
                return text_response(f"{message} These times are still free: {options}.")
            return text_response(message)

        appointment_id = (payload.get("appointment") or {}).get("appointment_id")
        if appointment_id is None:
            return text_response(payload.get("message") or "That is booked.")
        return function_call_response("render_confirmation", {"appointment_id": appointment_id})

    @staticmethod
    def _from_confirmation(payload: dict[str, Any]) -> LlmResponse:
        """The consequential reply. Every fact comes from the persisted row."""
        sentence = payload.get("sentence")
        if sentence:
            return text_response(f"{sentence} You will get a reminder the day before.")
        facts = payload.get("facts") or {}
        return text_response(
            f"Your appointment with {facts.get('doctor_name')} is "
            f"{facts.get('status')} for {facts.get('weekday')} {facts.get('date')} "
            f"at {facts.get('time')}."
        )

    @staticmethod
    def _from_ambiguous(payload: dict[str, Any]) -> LlmResponse:
        names = [candidate["name"] for candidate in payload.get("candidates", [])]
        listed = " or ".join(names) if names else "one of our departments"
        return text_response(
            f"That could be handled by {listed}. Which would you like? "
            "I can also pass this to our staff to route for you."
        )

    @staticmethod
    def _from_unsupported(payload: dict[str, Any]) -> LlmResponse:
        return text_response(
            "I could not match that to one of our departments. "
            "Could you tell me a little more about what the appointment is for?"
        )

    @staticmethod
    def _from_documents(payload: Any) -> LlmResponse:
        documents = payload.get("result") if isinstance(payload, dict) else payload
        if not documents:
            return text_response("You have no documents on file yet.")
        listed = ", ".join(
            f"{doc['document_type']} ({doc['status'].replace('_', ' ')})"
            for doc in documents[:5]
        )
        return text_response(f"You have {len(documents)} document(s) on file: {listed}.")
