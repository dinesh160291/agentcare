"""The mock provider's understudy contract.

The point of these tests is rule 6: a reply that does not vary with its input,
or that states a fact not present in the database, is a hardcoded final
response. Both are checked here against real rows produced by the real tools.

They drive the provider directly with hand-built ``LlmRequest`` objects, which
is what ADK hands it. The toolset in each request is the real one that agent
receives, because the mock dispatches on the toolset — a model may only call
what it was given, and the understudy is held to the same rule.

**What moved in Phase 4.** Booking used to be a tool the model could call; it
is now code, triggered by a confirmation that code read. The properties that
mattered did not move — the receipt is still re-read from the row, an ambiguous
answer still never commits, a refused proposal still reports its refusal — but
the seam they are checked at did. Their end-to-end versions are in
``test_orchestrator.py``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime

import pytest
from google.adk.models import LlmRequest
from google.genai import types

from app import clock
from app.providers.mock import MockLlm, reads_as_confirmation, reads_as_decline
from app.tools.appointments import book_appointment
from app.tools.availability import find_available_slots
from app.tools.confirmations import render_confirmation
from app.tools.dates import resolve_date
from app.tools.departments import resolve_department

MONDAY = date(2026, 8, 3)

# The real toolsets, per agent. If ``Toolbelt`` renames a tool the mock's
# dispatch changes with it and these fixtures must change too — which is the
# point: a mock built against tools nobody has proves nothing.
COORDINATOR_NEW = ("load_patient_context", "submit_plan")
COORDINATOR_ACTIVE = ("load_patient_context", "classify_message")
ROUTING = ("resolve_department", "list_departments", "submit_routing")
APPOINTMENT = (
    "resolve_date",
    "find_available_slots",
    "propose_appointment",
    "render_confirmation",
)
#: Mirrors what Toolbelt.document_tools() actually hands over. A constant that
#: drifted from the real toolset would let these tests describe a mock policy
#: the running system never takes.
DOCUMENTS = (
    "list_patient_documents",
    "list_unverified_documents",
    "read_document_text",
    "submit_document_verification",
    "diff_required_documents",
    "record_missing_documents",
)
#: "Nothing waiting to be verified" — the precondition for the diff stage.
NOTHING_PENDING = ("list_unverified_documents", {"documents": []})
FOLLOWUP = ("list_patient_reminders", "list_open_tasks")


@pytest.fixture(autouse=True)
def world(db):
    clock.freeze(datetime(2026, 8, 3, 8, 0))
    from scripts.seed import seed

    seed(db, anchor=MONDAY)
    db.commit()
    return db


def _tools_dict(names):
    """A tools_dict of real FunctionTools with the given names."""
    from google.adk.tools import FunctionTool

    tools = {}
    for name in names:

        def stub(**kwargs) -> dict:
            """Stub tool used only to advertise availability."""
            return {}

        stub.__name__ = name
        tools[name] = FunctionTool(stub)
    return tools


def build_request(user_text, *tool_results, tools) -> LlmRequest:
    """An LlmRequest shaped the way ADK hands one to a provider.

    A dict ``user_text`` is serialised as a specialist's typed task, which is
    the only thing a specialist ever receives.
    """
    if isinstance(user_text, dict):
        user_text = json.dumps(user_text, sort_keys=True)
    contents = [types.Content(role="user", parts=[types.Part(text=user_text)])]
    for name, payload in tool_results:
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=name, response=payload
                        )
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


def call_args(response) -> dict:
    return response.content.parts[0].function_call.args


class TestTheCoordinatorPlans:
    def test_it_loads_patient_context_first(self):
        """Tier-2 memory: a returning patient re-explains nothing, because the
        first thing the Coordinator does is read the database."""
        response = ask(
            build_request("I need a cardiology appointment", tools=COORDINATOR_NEW)
        )
        assert calls(response) == ["load_patient_context"]

    def test_a_booking_request_produces_a_booking_plan(self):
        """A mock that replied straight away would prove nothing works."""
        response = ask(
            build_request(
                "I need a cardiology appointment next week",
                ("load_patient_context", {"profile": {}}),
                tools=COORDINATOR_NEW,
            )
        )
        assert calls(response) == ["submit_plan"]
        assert "book" in call_args(response)["steps"]

    def test_a_two_intent_message_proposes_both(self):
        """Intent extraction must not stop at whichever matched first."""
        response = ask(
            build_request(
                "I need a cardiology appointment next week. I'll attach my ECG report.",
                ("load_patient_context", {"profile": {}}),
                tools=COORDINATOR_NEW,
            )
        )
        steps = call_args(response)["steps"]
        assert "book" in steps and "documents" in steps

    def test_a_document_only_message_does_not_ask_for_a_booking(self):
        response = ask(
            build_request(
                "I want to upload my blood test report",
                ("load_patient_context", {"profile": {}}),
                tools=COORDINATOR_NEW,
            )
        )
        steps = call_args(response)["steps"]
        assert "documents" in steps and "book" not in steps

    def test_the_acknowledgement_is_built_from_the_accepted_plan(self):
        """From what code accepted, not from what the mock asked for."""
        response = ask(
            build_request(
                "I need an appointment",
                ("load_patient_context", {"profile": {}}),
                ("submit_plan", {"accepted": True, "plan": ["route", "book"]}),
                tools=COORDINATOR_NEW,
            )
        )
        text = reply_text(response).lower()
        assert "department" in text and "appointment" in text

    def test_a_rejected_plan_is_not_reported_as_success(self):
        response = ask(
            build_request(
                "I need an appointment",
                ("load_patient_context", {"profile": {}}),
                ("submit_plan", {"accepted": False, "problem": "Unknown plan step"}),
                tools=COORDINATOR_NEW,
            )
        )
        assert "Unknown plan step" in reply_text(response)


class TestTheCoordinatorClassifies:
    def _classify(self, text) -> dict:
        response = ask(
            build_request(
                text,
                ("load_patient_context", {"profile": {}}),
                tools=COORDINATOR_ACTIVE,
            )
        )
        assert calls(response) == ["classify_message"]
        return call_args(response)

    def test_a_withdrawal_outranks_everything(self):
        assert self._classify("actually never mind, forget it")["message_class"] == (
            "withdrawal"
        )

    def test_an_off_topic_message_is_not_a_continuation(self):
        """The ordering that stops an off-topic message being appended to the
        run's request text and read later by routing."""
        assert self._classify("what's the weather like?")["message_class"] == "off_topic"

    def test_a_document_mention_is_complementary(self):
        args = self._classify("also, here's my old ECG report")
        assert args["message_class"] == "complementary"
        assert args["incoming_steps"] == ["documents"]

    def test_a_second_booking_request_is_conflicting(self):
        assert (
            self._classify("actually book a dermatology appointment instead")[
                "message_class"
            ]
            == "conflicting"
        )

    def test_a_read_only_question_stays_a_side_question(self):
        assert self._classify("what documents do I have on file?")["message_class"] == (
            "side_question"
        )

    def test_an_ordinary_answer_is_a_continuation(self):
        assert self._classify("Tuesday works for the appointment")["message_class"] == (
            "continuation"
        )


class TestRoutingUsesRealResolution:
    def test_it_reads_the_request_text_from_its_typed_task(self):
        """Routing is the one specialist whose task carries the patient's
        words, because classifying them is its job."""
        response = ask(
            build_request(
                {"step": "route", "request_text": "my heart has been bothering me"},
                tools=ROUTING,
            )
        )
        assert calls(response) == ["resolve_department"]
        assert "heart" in call_args(response)["text"]

    def test_a_resolved_department_is_submitted_with_high_confidence(self, db):
        resolved = resolve_department(db, "I need a cardiology appointment")
        response = ask(
            build_request(
                {"step": "route", "request_text": "cardiology please"},
                ("resolve_department", resolved),
                tools=ROUTING,
            )
        )
        args = call_args(response)
        assert args["department_name"] == "Cardiology"
        assert args["confidence"] == "high"

    def test_ambiguity_hands_over_rather_than_guessing_confidently(self, db):
        """The seeded ambiguous case exists to demonstrate this path. Guessing
        confidently to avoid the handover is the failure being refused, and the
        candidate submitted has to be one the resolver actually returned."""
        ambiguous = resolve_department(db, "my kid has ear pain")
        assert ambiguous["status"] == "ambiguous"

        response = ask(
            build_request(
                {"step": "route", "request_text": "my kid has ear pain"},
                ("resolve_department", ambiguous),
                tools=ROUTING,
            )
        )
        args = call_args(response)
        assert args["confidence"] == "low"
        assert args["department_name"] in [c["name"] for c in ambiguous["candidates"]]

    def test_an_unsupported_request_submits_nothing(self, db):
        unsupported = resolve_department(db, "qqqq zzzz")
        response = ask(
            build_request(
                {"step": "route", "request_text": "qqqq zzzz"},
                ("resolve_department", unsupported),
                tools=ROUTING,
            )
        )
        assert calls(response) == []
        assert "department" in reply_text(response).lower()

    def test_the_reply_names_the_department_code_accepted(self):
        response = ask(
            build_request(
                {"step": "route"},
                ("resolve_department", {"status": "resolved"}),
                (
                    "submit_routing",
                    {
                        "accepted": True,
                        "department": {"id": 1, "name": "Cardiology"},
                        "confidence": "high",
                    },
                ),
                tools=ROUTING,
            )
        )
        assert "Cardiology" in reply_text(response)

    def test_a_rejected_department_is_not_reported_as_routed(self):
        response = ask(
            build_request(
                {"step": "route"},
                ("resolve_department", {"status": "resolved"}),
                (
                    "submit_routing",
                    {
                        "accepted": False,
                        "problem": "'Cardiology Unit' is not a department",
                    },
                ),
                tools=ROUTING,
            )
        )
        assert "not a department" in reply_text(response)

    def test_a_low_confidence_reply_promises_a_human(self):
        response = ask(
            build_request(
                {"step": "route"},
                ("resolve_department", {"status": "ambiguous"}),
                (
                    "submit_routing",
                    {
                        "accepted": True,
                        "department": {"id": 5, "name": "Pediatrics"},
                        "confidence": "low",
                    },
                ),
                tools=ROUTING,
            )
        )
        text = reply_text(response).lower()
        assert "staff" in text and "pediatrics" in text


class TestAppointmentProposesButNeverBooks:
    def test_the_appointment_toolset_has_no_booking_tool(self):
        """The commit is code's, triggered by a confirmation code read. The
        model is never handed the tool that books."""
        assert "book_appointment" not in APPOINTMENT

    def test_it_resolves_dates_rather_than_working_them_out(self):
        response = ask(
            build_request(
                {"step": "book", "request_text": "next week please", "department_id": 1},
                tools=APPOINTMENT,
            )
        )
        assert calls(response) == ["resolve_date"]

    def test_it_searches_with_the_resolved_window(self):
        window = resolve_date("next week", today=MONDAY)
        response = ask(
            build_request(
                {"step": "book", "request_text": "next week", "department_id": 1},
                ("resolve_date", window),
                tools=APPOINTMENT,
            )
        )
        assert calls(response) == ["find_available_slots"]
        assert call_args(response)["start"] == window["start"]

    def test_it_proposes_a_slot_from_the_availability_result(self, db):
        window = resolve_date("next week", today=MONDAY)
        slots = find_available_slots(db, department_id=1)
        response = ask(
            build_request(
                {"step": "book", "department_id": 1},
                ("resolve_date", window),
                ("find_available_slots", slots),
                tools=APPOINTMENT,
            )
        )
        assert calls(response) == ["propose_appointment"]
        assert call_args(response)["slot_id"] == slots["slots"][0]["slot_id"]

    def test_the_offer_states_the_slot_code_accepted(self, db):
        first = find_available_slots(db, department_id=1)["slots"][0]
        response = ask(
            build_request(
                {"step": "book", "department_id": 1},
                ("propose_appointment", {"accepted": True, "proposed": first}),
                tools=APPOINTMENT,
            )
        )
        text = reply_text(response)
        assert first["doctor_name"] in text
        assert first["start"][:10] in text
        assert first["start"][11:16] in text

    def test_an_empty_availability_result_is_not_dressed_up_as_success(self):
        response = ask(
            build_request(
                {"step": "book", "department_id": 1},
                ("resolve_date", {"resolved": False}),
                ("find_available_slots", {"slots": [], "total_matching": 0}),
                tools=APPOINTMENT,
            )
        )
        assert calls(response) == []
        assert "find" in reply_text(response).lower()

    def test_a_refused_proposal_reports_the_refusal(self):
        """The commit-time failure path in miniature: a proposal pointing at a
        slot somebody else took must not be reported as an offer."""
        response = ask(
            build_request(
                {"step": "book", "department_id": 1},
                (
                    "propose_appointment",
                    {"accepted": False, "problem": "That time has just been taken."},
                ),
                tools=APPOINTMENT,
            )
        )
        assert "just been taken" in reply_text(response)


class TestTheReceiptIsReadBackFromTheRow:
    def _receipt(self, db, patient_id: int, slot_index: int) -> tuple[str, dict]:
        from app.models import User

        patient = db.query(User).filter(User.id == patient_id).one()
        slots = find_available_slots(db, department_id=1, limit=6)["slots"]
        booked = book_appointment(
            db, patient, slot_id=slots[slot_index]["slot_id"], reason="follow-up"
        )
        appointment_id = booked["appointment"]["appointment_id"]
        rendered = render_confirmation(db, appointment_id)
        response = ask(
            build_request(
                {"step": "book", "appointment_id": appointment_id},
                ("render_confirmation", rendered),
                tools=APPOINTMENT,
            )
        )
        return reply_text(response), rendered["facts"]

    def test_a_booked_appointment_triggers_a_re_read(self):
        """The mock does not template from what the booking returned — it calls
        the seam whose whole job is to re-read the row."""
        response = ask(
            build_request(
                {"step": "book", "appointment_id": 1, "department_id": 1},
                tools=APPOINTMENT,
            )
        )
        assert calls(response) == ["render_confirmation"]
        assert call_args(response)["appointment_id"] == 1

    def test_every_fact_in_the_reply_comes_from_the_row(self, db):
        reply, facts = self._receipt(db, 2, 0)
        for key in ("doctor_name", "weekday", "date", "time", "reference_code"):
            assert str(facts[key]) in reply, f"{key} missing from the reply"

    def test_different_bookings_produce_different_replies(self, db):
        """The rule-6 check: identical replies for different inputs is a
        hardcoded response wearing a template's clothes."""
        first, first_facts = self._receipt(db, 2, 0)
        second, second_facts = self._receipt(db, 3, 4)

        assert first != second
        assert first_facts["time"] != second_facts["time"] or (
            first_facts["doctor_name"] != second_facts["doctor_name"]
        )

    def test_the_reply_states_no_fact_the_row_does_not_hold(self, db):
        """The other half: a doctor who is not on the appointment must not
        appear in its confirmation."""
        from app.models import Doctor

        reply, facts = self._receipt(db, 2, 0)
        others = [
            doctor.name
            for doctor in db.query(Doctor).all()
            if doctor.name != facts["doctor_name"]
        ]
        assert not [name for name in others if name in reply]


class TestDocumentVerificationPolicy:
    """The mock's reading judgement. It stands in for the one thing a real
    model is genuinely better at, so what matters is that it is *honest* about
    the limits rather than that it is clever."""

    PENDING = (
        "list_unverified_documents",
        {"documents": [{"document_id": 7, "declared_type": "ECG report"}]},
    )

    def test_pending_documents_are_read_before_anything_else(self):
        response = ask(
            build_request(
                {"step": "documents", "department_id": 1},
                self.PENDING,
                tools=DOCUMENTS,
            )
        )
        assert calls(response) == ["read_document_text"]
        assert call_args(response)["document_id"] == 7

    def test_a_mismatch_is_proposed_from_the_extracted_text(self):
        response = ask(
            build_request(
                {"step": "documents", "department_id": 1},
                self.PENDING,
                (
                    "read_document_text",
                    {"extracted": True, "text": "SYNTHETIC X-RAY REPORT - chest radiograph"},
                ),
                tools=DOCUMENTS,
            )
        )
        submitted = call_args(response)
        assert submitted["matches"] is False
        assert submitted["detected_type"] == "X-ray report"

    def test_matching_content_is_proposed_as_a_match(self):
        response = ask(
            build_request(
                {"step": "documents", "department_id": 1},
                self.PENDING,
                (
                    "read_document_text",
                    {"extracted": True, "text": "Report type: Electrocardiogram summary"},
                ),
                tools=DOCUMENTS,
            )
        )
        submitted = call_args(response)
        assert submitted["matches"] is True

    def test_an_unreadable_file_is_accepted_at_its_declared_type(self):
        """No OCR. Flagging every photo a patient uploads would fill the review
        queue with things nobody did wrong."""
        response = ask(
            build_request(
                {"step": "documents", "department_id": 1},
                self.PENDING,
                ("read_document_text", {"extracted": False, "reason": "not_extractable", "text": ""}),
                tools=DOCUMENTS,
            )
        )
        submitted = call_args(response)
        assert submitted["matches"] is True
        assert submitted["detected_type"] == "ECG report"

    def test_unrecognisable_text_is_not_guessed_into_a_mismatch(self):
        """An unexplained flag costs a human the time to work out why."""
        response = ask(
            build_request(
                {"step": "documents", "department_id": 1},
                self.PENDING,
                ("read_document_text", {"extracted": True, "text": "Lorem ipsum dolor sit amet"}),
                tools=DOCUMENTS,
            )
        )
        assert call_args(response)["matches"] is True

    def test_verification_is_skipped_when_the_tool_was_not_handed_over(self):
        """Toolset dispatch, like every other branch: asking for a tool the
        agent does not have is how a mock starts describing a system that does
        not exist."""
        response = ask(
            build_request(
                {"step": "documents", "department_id": 1},
                tools=("list_patient_documents", "record_missing_documents"),
            )
        )
        assert calls(response) == ["list_patient_documents"]


class TestDocumentsAndFollowUp:
    def test_documents_are_listed_before_anything_is_diffed(self):
        response = ask(
            build_request(
                {"step": "documents", "department_id": 1},
                NOTHING_PENDING,
                tools=DOCUMENTS,
            )
        )
        assert calls(response) == ["list_patient_documents"]

    def test_the_diff_runs_once_a_department_is_known(self):
        response = ask(
            build_request(
                {"step": "documents", "department_id": 1},
                NOTHING_PENDING,
                ("list_patient_documents", {"documents": []}),
                tools=DOCUMENTS,
            )
        )
        assert calls(response) == ["diff_required_documents"]

    def test_the_diff_is_skipped_when_no_department_is_known(self):
        """The required-documents rules are per department; diffing without one
        would compare against nothing and report everything satisfied."""
        response = ask(
            build_request(
                {"step": "documents"},
                NOTHING_PENDING,
                ("list_patient_documents", {"documents": []}),
                tools=DOCUMENTS,
            )
        )
        assert calls(response) == []

    def test_missing_documents_are_recorded_not_just_mentioned(self):
        """A missing-documents list that lives only in a reply is lost the
        moment the conversation ends."""
        response = ask(
            build_request(
                {"step": "documents", "department_id": 1},
                NOTHING_PENDING,
                ("list_patient_documents", {"documents": []}),
                ("diff_required_documents", {"missing_mandatory": ["ECG report"]}),
                tools=DOCUMENTS,
            )
        )
        assert calls(response) == ["record_missing_documents"]

    def test_the_document_reply_counts_what_the_tool_returned(self):
        documents = [
            {"document_type": "ECG report", "status": "verified"},
            {"document_type": "Blood test report", "status": "pending_verification"},
        ]
        response = ask(
            build_request(
                {"step": "documents"},
                NOTHING_PENDING,
                ("list_patient_documents", {"documents": documents}),
                tools=DOCUMENTS,
            )
        )
        text = reply_text(response)
        assert "2 document" in text
        assert "ECG report" in text

    def test_follow_up_summarises_only_what_it_was_given(self):
        response = ask(
            build_request(
                {"step": "follow_up"},
                ("list_patient_reminders", {"reminders": []}),
                ("list_open_tasks", {"tasks": []}),
                tools=FOLLOWUP,
            )
        )
        assert "nothing is outstanding" in reply_text(response).lower()

    def test_follow_up_counts_what_the_tools_returned(self):
        response = ask(
            build_request(
                {"step": "follow_up"},
                (
                    "list_patient_reminders",
                    {"reminders": [{"scheduled_at": "2026-08-09T09:00:00"}]},
                ),
                ("list_open_tasks", {"tasks": [{"task_type": "missing_documents"}]}),
                tools=FOLLOWUP,
            )
        )
        text = reply_text(response)
        assert "1 reminder" in text and "2026-08-09" in text
        assert "missing documents" in text


class TestConfirmationReadingIsDelegated:
    """The mock is not permitted to be cleverer than the real reader."""

    @pytest.mark.parametrize("text", ["yes", "Yes", " confirm ", "book it", "OK"])
    def test_exact_tokens_confirm(self, text):
        assert reads_as_confirmation(text) is True

    @pytest.mark.parametrize(
        "text", ["no wait — yes, the Tuesday one", "maybe", "yes but later"]
    )
    def test_ambiguous_text_never_confirms(self, text):
        assert reads_as_confirmation(text) is False

    @pytest.mark.parametrize("text", ["no", "cancel", "Decline"])
    def test_declines_are_read(self, text):
        assert reads_as_decline(text) is True

    def test_it_uses_the_workflow_token_set_rather_than_its_own(self):
        """Two copies of a token list are two lists that drift apart."""
        from app.workflow.confirmation import (
            CONFIRM_TOKENS,
            ConfirmationAnswer,
            read_confirmation,
        )

        for token in CONFIRM_TOKENS:
            assert reads_as_confirmation(token) is True
            assert read_confirmation(token) is ConfirmationAnswer.CONFIRM


class TestScopeReply:
    def test_unrecognised_input_gets_the_scope_template(self):
        response = ask(
            build_request(
                "what's the weather like?",
                ("load_patient_context", {"profile": {}}),
                tools=COORDINATOR_NEW,
            )
        )
        assert calls(response) == []
        assert "appointments" in reply_text(response).lower()

    def test_off_topic_input_submits_no_plan(self):
        """No plan submitted means no run created and no tools fired.
        Off-topic is noise, not a human-review case."""
        response = ask(
            build_request(
                "tell me a joke",
                ("load_patient_context", {"profile": {}}),
                tools=COORDINATOR_NEW,
            )
        )
        assert "submit_plan" not in calls(response)

    def test_the_scope_reply_carries_no_clinical_language(self):
        from app.providers.mock import SCOPE_TEXT

        lowered = SCOPE_TEXT.lower()
        for word in ("diagnos", "prescrib", "dosage", "symptom", "treatment"):
            assert word not in lowered
