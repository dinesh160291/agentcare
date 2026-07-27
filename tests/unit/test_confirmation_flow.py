"""The rest of confirm-before-commit: buttons, the model's half, and the stall.

The exact-token reader shipped in Phase 4. What is pinned here is everything
around it, and the reading order it belongs to:

  (a) ✅ Confirm / ❌ Decline buttons — a typed action, zero interpretation;
  (b) exact-token match on typed text, in code;
  (c) only what the tokens cannot read reaches the model, whose **only**
      permitted verdicts are decline and non-answer.

The one rule everything here serves: **a wrongly re-asked "yes" costs one tap;
a wrongly committed "no" books an appointment against the patient's word at the
exact step built to prevent that.** So the tests below spend most of their
effort trying to get a booking committed by something other than the patient,
and expect to fail every time.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest
from google.adk.models import LlmRequest, LlmResponse

from app.config import get_settings
from app.db import SessionLocal
from app.errors import ValidationFailed
from app.models import (
    Appointment,
    AppointmentStatus,
    AuditEvent,
    TraceAuthor,
    TraceEvent,
    TraceEventType,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.orchestrator import (
    DECLINED_REPLY,
    NOTHING_TO_CONFIRM_REPLY,
    apply_patient_action,
    run_workflow,
)
from app.providers.base import (
    AgentCareLlm,
    available_tool_names,
    called_tools,
    function_call_response,
    text_response,
)
from app.trace import assert_well_formed
from app.workflow.replies import render_reask

PATIENT_EMAIL = "asha.patient@example.invalid"
BOOKING = "I need a cardiology appointment next week"
SEEDED_APPOINTMENT_ID = 1


@pytest.fixture
def patient(seeded_db):
    seeded_db.commit()
    return seeded_db.query(User).filter(User.email == PATIENT_EMAIL).one()


def turn(user, message, session_id):
    return asyncio.run(run_workflow(user, message, session_id))


def press(user, action, session_id):
    return asyncio.run(apply_patient_action(user, action, session_id))


def fresh():
    return SessionLocal()


def booked_for(session, run_id: int) -> list[Appointment]:
    run = session.get(WorkflowRun, run_id)
    return (
        session.query(Appointment)
        .filter(Appointment.patient_id == run.patient_id)
        .filter(Appointment.id != SEEDED_APPOINTMENT_ID)
        .all()
    )


def _guard(session, turn_id, name):
    for event in (
        session.query(TraceEvent)
        .filter(TraceEvent.turn_id == turn_id)
        .order_by(TraceEvent.seq)
        .all()
    ):
        if event.event_type is TraceEventType.GUARD_VERDICT:
            if (event.payload or {}).get("guard") == name:
                return event.payload
    raise AssertionError(f"no {name!r} guard verdict in turn {turn_id}")


class TestTheConfirmButton:
    """A click arrives as a typed action. There is nothing to interpret, so
    nothing interprets it."""

    def test_pressing_confirm_books_the_appointment(self, patient):
        turn(patient, BOOKING, "s-btn-1")
        result = press(patient, "confirm", "s-btn-1")

        session = fresh()
        try:
            booked = booked_for(session, result.run_id)
            assert len(booked) == 1
            assert booked[0].status is AppointmentStatus.CONFIRMED
        finally:
            session.close()

    def test_the_click_is_an_inbound_event_of_its_own_kind(self, patient):
        """The system has two front doors and the grammar recognises both: an
        inbound event is a chat message **or** a typed action."""
        turn(patient, BOOKING, "s-btn-2")
        result = press(patient, "confirm", "s-btn-2")

        session = fresh()
        try:
            inbound = [
                e
                for e in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if e.event_type is TraceEventType.INBOUND
            ]
        finally:
            session.close()

        assert len(inbound) == 1
        assert inbound[0].author is TraceAuthor.PATIENT_ACTION

    def test_the_turn_is_bracketed_like_any_other(self, patient):
        turn(patient, BOOKING, "s-btn-3")
        result = press(patient, "confirm", "s-btn-3")

        session = fresh()
        try:
            kinds = [
                e.event_type
                for e in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .order_by(TraceEvent.seq)
                .all()
            ]
        finally:
            session.close()

        assert kinds[0] is TraceEventType.INBOUND
        assert kinds[-1] is TraceEventType.OUTBOUND

    def test_pressing_decline_clears_the_proposal(self, patient):
        turn(patient, BOOKING, "s-btn-4")
        result = press(patient, "decline", "s-btn-4")

        assert result.reply == DECLINED_REPLY
        assert result.status == WorkflowStatus.IN_PROGRESS.value

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.proposed_slot_id is None
            assert booked_for(session, result.run_id) == []
        finally:
            session.close()

    def test_a_second_confirm_click_is_a_calm_no_op(self, patient):
        """The motivating trace is a double-clicked button. The second request
        finds the proposal already committed and must say so, not crash and not
        book twice."""
        first = turn(patient, BOOKING, "s-btn-5")
        press(patient, "confirm", "s-btn-5")
        again = press(patient, "confirm", "s-btn-5")

        assert again.reply == NOTHING_TO_CONFIRM_REPLY

        session = fresh()
        try:
            assert len(booked_for(session, first.run_id)) == 1
        finally:
            session.close()

    def test_a_stale_click_is_audited(self, patient):
        press(patient, "confirm", "s-btn-6")

        session = fresh()
        try:
            assert (
                session.query(AuditEvent)
                .filter(AuditEvent.action == "patient_action_stale")
                .count()
                == 1
            )
        finally:
            session.close()

    def test_an_unknown_action_is_refused(self, patient):
        turn(patient, BOOKING, "s-btn-7")

        with pytest.raises(ValidationFailed):
            press(patient, "book it immediately", "s-btn-7")

    def test_the_typed_action_turn_parses_against_the_grammar(self, patient):
        """The checker's docstring has always claimed typed-action turns follow
        the same rules. Until there were typed actions, nothing exercised the
        claim — and a checker that only understood chatty turns would pass the
        system's least deterministic flows while ignoring its most."""
        turn(patient, BOOKING, "s-btn-grammar")
        result = press(patient, "confirm", "s-btn-grammar")

        session = fresh()
        try:
            assert_well_formed(session, turn_id=result.turn_id)
        finally:
            session.close()

    def test_a_typed_action_costs_no_model_call(self, patient):
        """Zero interpretation means zero interpretation."""
        turn(patient, BOOKING, "s-btn-8")
        result = press(patient, "decline", "s-btn-8")

        session = fresh()
        try:
            requests = [
                e
                for e in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if e.event_type is TraceEventType.LLM_REQUEST
            ]
        finally:
            session.close()
        assert requests == []


class TestTheModelMayNeverConfirm:
    """(c) of the reading order. The model's two permitted verdicts are
    decline and non-answer; ``confirm`` is not in the enum, so the refusal is
    structural rather than a matter of the prompt holding."""

    def test_a_model_read_decline_returns_to_slot_selection(self, patient):
        """Text the exact tokens cannot read — "not that one, thanks" carries
        no token — still has to be able to mean no."""
        turn(patient, BOOKING, "s-model-1")
        result = turn(patient, "not that one, thanks", "s-model-1")

        assert result.status == WorkflowStatus.IN_PROGRESS.value

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.proposed_slot_id is None
            assert booked_for(session, result.run_id) == []
        finally:
            session.close()

    def test_a_confirm_verdict_is_rejected_by_code(self, patient, monkeypatch):
        """The adversarial case: a provider that tries to confirm on the
        patient's behalf. It must be told no, and nothing must be booked."""
        first = turn(patient, BOOKING, "s-model-2")

        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: EagerConfirmer()
        )
        result = turn(patient, "hmm, I suppose so, maybe?", "s-model-2")

        session = fresh()
        try:
            assert booked_for(session, first.run_id) == []
            run = session.get(WorkflowRun, first.run_id)
            assert run.status is WorkflowStatus.PENDING_CONFIRMATION
            rejections = [
                e.payload
                for e in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if e.event_type is TraceEventType.VALIDATION
                and e.payload["what"] == "confirmation_verdict"
            ]
        finally:
            session.close()

        assert rejections, "the attempt must be recorded, not merely ignored"
        assert all(r["accepted"] is False for r in rejections)


class TestStallContainment:
    """The re-ask loop is bounded. A stall never resolves implicitly in either
    direction — no auto-commit, no auto-decline."""

    def test_each_non_answer_is_counted(self, patient):
        result = turn(patient, BOOKING, "s-stall-1")
        for _ in range(2):
            turn(patient, "hmm, maybe", "s-stall-1")

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.non_answer_count == 2
        finally:
            session.close()

    def test_at_the_cap_the_re_ask_becomes_code_templated(self, patient):
        """Still exact equality, against what code actually writes now.

        The stalled wording used to be a bare constant. It is the facts-
        carrying re-ask instead — a patient who has been asked three times is
        the one who most needs telling *what* they are being asked about — so
        the pin moved to ``render_reask`` rather than being loosened to a
        substring, which would have stopped noticing whether code or the model
        wrote it at all.
        """
        cap = get_settings().max_confirmation_non_answers
        result = turn(patient, BOOKING, "s-stall-2")

        replies = [turn(patient, "hmm, maybe", "s-stall-2") for _ in range(cap + 1)]

        session = fresh()
        try:
            expected = render_reask(session, session.get(WorkflowRun, result.run_id))
        finally:
            session.close()

        assert replies[-1].reply == expected
        assert replies[-1].author is TraceAuthor.TEMPLATE

    def test_the_templated_re_ask_costs_no_model_wording(self, patient):
        """"Costing zero further LLM turns" is about the re-ask, which is now
        code's. Classification still runs — a withdrawal after four stalls must
        still be heard."""
        cap = get_settings().max_confirmation_non_answers
        turn(patient, BOOKING, "s-stall-3")
        for _ in range(cap):
            turn(patient, "hmm, maybe", "s-stall-3")
        result = turn(patient, "hmm, maybe", "s-stall-3")

        assert result.author is TraceAuthor.TEMPLATE

    def test_a_withdrawal_still_lands_after_the_cap(self, patient):
        """The trap the class-independent counter exists to avoid: bounding the
        loop by skipping the model would also stop the system hearing "forget
        it", leaving a zombie run claiming the patient is still waiting."""
        cap = get_settings().max_confirmation_non_answers
        first = turn(patient, BOOKING, "s-stall-4")
        for _ in range(cap + 1):
            turn(patient, "hmm, maybe", "s-stall-4")
        turn(patient, "actually forget it", "s-stall-4")

        session = fresh()
        try:
            run = session.get(WorkflowRun, first.run_id)
            assert run.status is WorkflowStatus.CANCELLED
            assert run.cancellation_reason == "withdrawn"
        finally:
            session.close()

    def test_an_off_topic_stall_still_counts(self, patient):
        """Counted independently of the class. An "hmm, maybe" read as
        off-topic must not buy an extra free turn — that is how the bound
        becomes reachable only through correctly-classified paths."""
        result = turn(patient, BOOKING, "s-stall-5")
        turn(patient, "what's the weather like?", "s-stall-5")

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.non_answer_count == 1
        finally:
            session.close()

    def test_the_stall_never_resolves_itself(self, patient):
        cap = get_settings().max_confirmation_non_answers
        first = turn(patient, BOOKING, "s-stall-6")
        for _ in range(cap + 3):
            turn(patient, "hmm, maybe", "s-stall-6")

        session = fresh()
        try:
            run = session.get(WorkflowRun, first.run_id)
            assert run.status is WorkflowStatus.PENDING_CONFIRMATION
            assert run.proposed_slot_id is not None
            assert booked_for(session, first.run_id) == []
        finally:
            session.close()

    def test_the_counter_is_recorded_as_a_guard_verdict(self, patient):
        turn(patient, BOOKING, "s-stall-7")
        result = turn(patient, "hmm, maybe", "s-stall-7")

        session = fresh()
        try:
            stall = _guard(session, result.turn_id, "confirmation_stall")
        finally:
            session.close()
        assert stall["detail"]["non_answers"] == 1
        assert stall["passed"] is True

    def test_declining_resets_the_counter(self, patient):
        """The count belongs to the proposal. A patient who stalled once should
        not meet the terse framing on their first look at a new time."""
        first = turn(patient, BOOKING, "s-stall-8")
        turn(patient, "hmm, maybe", "s-stall-8")
        press(patient, "decline", "s-stall-8")

        session = fresh()
        try:
            run = session.get(WorkflowRun, first.run_id)
            assert run.non_answer_count == 0
        finally:
            session.close()

    def test_confirming_resets_the_counter(self, patient):
        first = turn(patient, BOOKING, "s-stall-9")
        turn(patient, "hmm, maybe", "s-stall-9")
        turn(patient, "yes", "s-stall-9")

        session = fresh()
        try:
            run = session.get(WorkflowRun, first.run_id)
            assert run.non_answer_count == 0
        finally:
            session.close()


class EagerConfirmer(AgentCareLlm):
    """A provider that tries to book on the patient's behalf.

    Not a strawman: "no wait — yes, the Tuesday one" is exactly the input that
    invites a helpful model to decide the patient meant yes. The enum is what
    stops it, and this is how we find out whether the enum is real.
    """

    model: str = "eager-confirmer-stub"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        available = available_tool_names(llm_request)
        done = called_tools(llm_request)

        if "submit_safety_verdict" in available:
            if "submit_safety_verdict" in done:
                yield text_response("screened")
            else:
                yield function_call_response(
                    "submit_safety_verdict",
                    {"category": "safe", "rationale": "stub"},
                )
            return

        if "submit_confirmation_verdict" in available:
            if "submit_confirmation_verdict" in done:
                yield text_response("I've gone ahead and booked that for you.")
            else:
                yield function_call_response(
                    "submit_confirmation_verdict",
                    {"verdict": "confirm", "reason": "they sounded willing"},
                )
            return

        yield text_response("ok")
