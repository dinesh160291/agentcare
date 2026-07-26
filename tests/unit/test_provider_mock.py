"""The mock provider's understudy contract.

The point of these tests is rule 6: a reply that does not vary with its input,
or that states a fact not present in the database, is a hardcoded final
response. Both are checked here against real rows produced by the real tools.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime

import pytest
from google.adk.models import LlmRequest
from google.genai import types

from app import clock
from app.models import User
from app.providers.mock import MockLlm, reads_as_confirmation, reads_as_decline
from app.tools.appointments import book_appointment
from app.tools.availability import find_available_slots
from app.tools.confirmations import render_confirmation
from app.tools.departments import resolve_department

MONDAY = date(2026, 8, 3)


@pytest.fixture(autouse=True)
def world(db):
    clock.freeze(datetime(2026, 8, 3, 8, 0))
    from scripts.seed import seed

    seed(db, anchor=MONDAY)
    db.commit()
    return db


#: The tools a specialist agent holds. The request has to advertise them: a
#: model may only call what it was given, and the mock is held to the same rule.
SPECIALIST_TOOLS = (
    "resolve_department",
    "resolve_date",
    "find_available_slots",
    "book_appointment",
    "render_confirmation",
    "list_patient_documents",
)


def _tools_dict(names):
    """Build a tools_dict of real FunctionTools with the given names."""
    from google.adk.tools import FunctionTool

    tools = {}
    for name in names:
        def stub(**kwargs) -> dict:
            """Stub tool used only to advertise availability."""
            return {}

        stub.__name__ = name
        tools[name] = FunctionTool(stub)
    return tools


def build_request(
    user_text: str, *tool_results: tuple[str, dict], tools=SPECIALIST_TOOLS
) -> LlmRequest:
    """An LlmRequest shaped the way ADK hands one to a provider."""
    contents = [types.Content(role="user", parts=[types.Part(text=user_text)])]
    for name, payload in tool_results:
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(name=name, response=payload)
                    )
                ],
            )
        )
    return LlmRequest(
        model="mock-understudy", contents=contents, tools_dict=_tools_dict(tools)
    )


def ask(request: LlmRequest):
    async def run():
        async for response in MockLlm().generate_content_async(request):
            return response

    return asyncio.run(run())


def reply_text(response) -> str:
    return "".join(part.text or "" for part in (response.content.parts or []))


def calls(response) -> list[str]:
    return [
        part.function_call.name
        for part in (response.content.parts or [])
        if getattr(part, "function_call", None)
    ]


class TestRealToolCalls:
    def test_a_booking_request_calls_a_tool_rather_than_answering(self):
        """A mock that replied straight away would prove nothing works."""
        response = ask(build_request("I need a cardiology appointment next week"))
        assert calls(response) == ["resolve_department"]

    def test_it_progresses_through_the_tool_chain(self, db):
        text = "I need a cardiology appointment next week"
        department = resolve_department(db, text)

        response = ask(build_request(text, ("resolve_department", department)))
        assert calls(response) == ["resolve_date"]

    def test_a_confirmation_triggers_the_booking_tool(self, db):
        slots = find_available_slots(db, department_id=1, limit=3)
        response = ask(build_request("yes", ("find_available_slots", slots)))
        assert calls(response) == ["book_appointment"]
        assert response.content.parts[0].function_call.args["slot_id"] == (
            slots["slots"][0]["slot_id"]
        )

    def test_after_booking_it_re_reads_the_row(self, db):
        """It does not trust its own write: the confirmation is rendered from
        the persisted row, not from the booking tool's return value."""
        patient = db.query(User).filter(User.id == 2).one()
        slot_id = find_available_slots(db, department_id=1, limit=1)["slots"][0]["slot_id"]
        booked = book_appointment(db, patient, slot_id=slot_id, reason="x")

        response = ask(build_request("yes", ("book_appointment", booked)))
        assert calls(response) == ["render_confirmation"]


class TestRepliesAreTemplatedFromPersistedResults:
    def _confirmation_reply(self, db, patient_id: int, slot_index: int) -> tuple[str, dict]:
        patient = db.query(User).filter(User.id == patient_id).one()
        slots = find_available_slots(db, department_id=1, limit=6)["slots"]
        booked = book_appointment(
            db, patient, slot_id=slots[slot_index]["slot_id"], reason="follow-up"
        )
        appointment_id = booked["appointment"]["appointment_id"]
        rendered = render_confirmation(db, appointment_id)
        response = ask(build_request("yes", ("render_confirmation", rendered)))
        return reply_text(response), rendered["facts"]

    def test_every_fact_in_the_reply_comes_from_the_row(self, db):
        reply, facts = self._confirmation_reply(db, 2, 0)
        for key in ("doctor_name", "weekday", "date", "time", "reference_code"):
            assert str(facts[key]) in reply, f"{key} missing from the reply"

    def test_different_bookings_produce_different_replies(self, db):
        """The rule-6 check: identical replies for different inputs is a
        hardcoded response wearing a template's clothes."""
        first, first_facts = self._confirmation_reply(db, 2, 0)
        second, second_facts = self._confirmation_reply(db, 3, 4)

        assert first != second
        assert first_facts["time"] != second_facts["time"] or (
            first_facts["doctor_name"] != second_facts["doctor_name"]
        )

    def test_the_reply_states_no_fact_the_row_does_not_hold(self, db):
        """The other half: a doctor who is not on the appointment must not
        appear in its confirmation."""
        from app.models import Doctor

        reply, facts = self._confirmation_reply(db, 2, 0)
        others = [
            d.name
            for d in db.query(Doctor).all()
            if d.name != facts["doctor_name"]
        ]
        assert not [name for name in others if name in reply]

    def test_slot_offers_are_built_from_the_availability_result(self, db):
        text = "I need a cardiology appointment next week"
        department = resolve_department(db, text)
        slots = find_available_slots(db, department_id=1, limit=3)

        reply = reply_text(
            ask(
                build_request(
                    text,
                    ("resolve_department", department),
                    ("resolve_date", {"resolved": True, "start": "2026-08-10",
                                      "end": "2026-08-16", "part_of_day": None}),
                    ("find_available_slots", slots),
                )
            )
        )
        for slot in slots["slots"][:3]:
            assert slot["doctor_name"] in reply

    def test_an_empty_availability_result_is_not_dressed_up_as_success(self, db):
        empty = {"slots": [], "total_matching": 0}
        department = resolve_department(db, "cardiology")
        reply = reply_text(
            ask(
                build_request(
                    "cardiology next week",
                    ("resolve_department", department),
                    ("resolve_date", {"resolved": True, "start": "2027-01-01",
                                      "end": "2027-01-07", "part_of_day": None}),
                    ("find_available_slots", empty),
                )
            )
        )
        assert "could not find" in reply.lower()

    def test_a_failed_booking_reports_the_failure_and_the_alternatives(self, db):
        slots = find_available_slots(db, department_id=1, limit=3)["slots"]
        failed = {
            "ok": False,
            "reason": "slot_taken",
            "message": "That time was taken while you were confirming.",
            "appointment": None,
            "alternatives": slots,
        }
        reply = reply_text(ask(build_request("yes", ("book_appointment", failed))))
        assert "taken" in reply.lower()
        assert slots[0]["doctor_name"] in reply

    def test_ambiguity_is_reported_with_the_real_candidates(self, db):
        ambiguous = resolve_department(db, "my kid has ear pain")
        reply = reply_text(
            ask(build_request("my kid has ear pain", ("resolve_department", ambiguous)))
        )
        assert "Pediatrics" in reply and "ENT" in reply


class TestConfirmationReading:
    @pytest.mark.parametrize("text", ["yes", "Yes", " confirm ", "book it", "OK"])
    def test_exact_tokens_confirm(self, text):
        assert reads_as_confirmation(text) is True

    @pytest.mark.parametrize("text", ["no wait — yes, the Tuesday one", "maybe", "yes but later"])
    def test_ambiguous_text_never_confirms(self, text):
        """The mock is not permitted to be cleverer than the deterministic
        reader it stands in for."""
        assert reads_as_confirmation(text) is False

    @pytest.mark.parametrize("text", ["no", "cancel", "Decline"])
    def test_declines_are_read(self, text):
        assert reads_as_decline(text) is True

    def test_an_ambiguous_reply_does_not_reach_the_booking_tool(self, db):
        slots = find_available_slots(db, department_id=1, limit=3)
        response = ask(
            build_request("no wait — yes, the Tuesday one", ("find_available_slots", slots))
        )
        assert "book_appointment" not in calls(response)


class TestScopeReply:
    def test_unrecognised_input_gets_the_scope_template(self):
        reply = reply_text(ask(build_request("what's the weather like?")))
        assert "appointments" in reply.lower()

    def test_the_scope_reply_carries_no_clinical_language(self):
        reply = reply_text(ask(build_request("what's the weather like?"))).lower()
        for word in ("diagnos", "prescrib", "dosage", "symptom"):
            assert word not in reply
