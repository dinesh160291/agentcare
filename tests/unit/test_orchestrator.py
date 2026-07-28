"""The orchestrator seam — ``run_workflow``, driven end to end under mock.

These run the *whole* system: five agents, real tools, real database writes,
real state transitions, real trace rows. Under ``LLM_PROVIDER=mock`` the only
thing standing in is the judgement about what to do next. If something works
live but not here, it is not done.

Two checks in this file are the ones the mock has to survive to be a provider
rather than a fixture:

* **Fact diffing.** Every fact in a booking reply — doctor, weekday, date,
  time, reference — is compared against the row it claims to describe, on two
  different bookings. Identical replies for different inputs, or facts absent
  from the database, both fail.
* **The budget cap.** A stub provider that calls a tool forever must produce a
  ``failed`` run at exactly the cap, never an unbounded loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from collections import Counter
from typing import AsyncGenerator

import pytest
from google.adk.models import LlmRequest, LlmResponse

from app import clock
from app.db import SessionLocal
from app.models import (
    Appointment,
    AppointmentSlot,  # noqa: F401  (slot rows are read directly by the round-6 tests)
    AppointmentStatus,
    AuditEvent,
    DocumentStatus,
    Escalation,
    EscalationKind,
    EscalationStatus,
    MessageClass,
    PatientDocument,
    ProposedAction,
    Reminder,
    SlotStatus,
    TraceAuthor,
    TraceEvent,
    TraceEventType,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.orchestrator import (
    FAILED_REPLY,
    NO_PLAN_REPLY,
    SCOPE_REPLY,
    active_run,
    apply_patient_action,
    run_workflow,
)
from app.workflow.replies import clock_time
from app.providers.mock import MockLlm
from app.providers.base import (
    AgentCareLlm,
    available_tool_names,
    called_tools,
    function_call_response,
    latest_tool_result,
    text_response,
)
from app.trace import assert_well_formed

PATIENT_EMAIL = "asha.patient@example.invalid"
OTHER_EMAIL = "rohan.patient@example.invalid"
BOOKING = "I need a cardiology appointment next week"


@pytest.fixture
def patient(seeded_db):
    seeded_db.commit()
    return seeded_db.query(User).filter(User.email == PATIENT_EMAIL).one()


@pytest.fixture
def other_patient(seeded_db):
    return seeded_db.query(User).filter(User.email == OTHER_EMAIL).one()


def turn(user, message, session_id):
    """One turn, run to completion."""
    return asyncio.run(run_workflow(user, message, session_id))


def fresh():
    """A session that has seen none of the objects the turn wrote."""
    return SessionLocal()


def appointments_for(session, run_id: int) -> list[Appointment]:
    """This run's appointments.

    Absolute counts would be wrong: the seed ships one booked appointment on
    purpose, so that a reschedule/cancel flow has something to act on.
    """
    run = session.get(WorkflowRun, run_id)
    return (
        session.query(Appointment)
        .filter(Appointment.patient_id == run.patient_id)
        .filter(Appointment.id != SEEDED_APPOINTMENT_ID)
        .all()
    )


#: The seed's one pre-existing appointment.
SEEDED_APPOINTMENT_ID = 1


def _guard(session, turn_id: str, name: str) -> dict:
    """The payload of one named guard verdict in a turn."""
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


class TestBookingHappyPath:
    def test_the_first_turn_proposes_and_pauses(self, patient):
        result = turn(patient, BOOKING, "s-happy-1")

        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        assert result.plan == ["route", "book", "documents", "follow_up"]
        assert result.steps_run == ["route", "book"]

    def test_nothing_is_booked_before_confirmation(self, patient):
        """Confirm-before-commit. The proposal exists; the appointment does not."""
        result = turn(patient, BOOKING, "s-happy-2")

        session = fresh()
        try:
            assert appointments_for(session, result.run_id) == []
            run = session.get(WorkflowRun, result.run_id)
            assert run.proposed_slot_id is not None
        finally:
            session.close()

    def test_confirming_books_and_completes_the_plan(self, patient):
        turn(patient, BOOKING, "s-happy-3")
        result = turn(patient, "yes", "s-happy-3")

        assert result.status == WorkflowStatus.COMPLETED.value
        assert result.steps_run == ["book", "documents", "follow_up"]

        session = fresh()
        try:
            booked = appointments_for(session, result.run_id)
            assert len(booked) == 1
            assert booked[0].status is AppointmentStatus.CONFIRMED
        finally:
            session.close()

    def test_the_proposal_is_cleared_after_commit(self, patient):
        """A proposal that survives its own commit is confirmable twice."""
        turn(patient, BOOKING, "s-happy-4")
        turn(patient, "yes", "s-happy-4")

        session = fresh()
        try:
            run = session.query(WorkflowRun).order_by(WorkflowRun.id.desc()).first()
            assert run.proposed_slot_id is None
            assert run.proposed_action is None
        finally:
            session.close()

    def test_the_department_is_recorded_on_the_run(self, patient):
        turn(patient, BOOKING, "s-happy-5")
        session = fresh()
        try:
            run = session.query(WorkflowRun).order_by(WorkflowRun.id.desc()).first()
            assert run.state["department_name"] == "Cardiology"
        finally:
            session.close()

    def test_the_trace_is_well_formed(self, patient):
        turn(patient, BOOKING, "s-happy-6")
        turn(patient, "yes", "s-happy-6")

        session = fresh()
        try:
            assert_well_formed(session)
        finally:
            session.close()

    def test_both_ledgers_record_the_booking(self, patient):
        turn(patient, BOOKING, "s-happy-7")
        turn(patient, "yes", "s-happy-7")

        session = fresh()
        try:
            transitions = [
                event
                for event in session.query(TraceEvent).all()
                if event.event_type is TraceEventType.TRANSITION
            ]
            audits = [
                event
                for event in session.query(AuditEvent).all()
                if event.action == "workflow_transition"
            ]
            assert transitions and audits
        finally:
            session.close()


class TestRepliesAreTemplatedFromTheDatabase:
    """Rule 6, checked rather than asserted.

    A tool that returns a fixed value regardless of input scores zero, and the
    mock provider is bound by that rule as much as any other provider.
    """

    def _book(self, user, session_id, message=BOOKING):
        turn(user, message, session_id)
        return turn(user, "yes", session_id)

    def test_every_fact_in_the_reply_is_in_the_row(self, patient):
        result = self._book(patient, "s-facts-1")
        reply = result.reply

        session = fresh()
        try:
            appointment = appointments_for(session, result.run_id)[0]
            slot = appointment.slot
            doctor = appointment.doctor
            start = slot.start_time
        finally:
            session.close()

        assert doctor.name in reply
        assert start.strftime("%A") in reply
        # Through the shared formatter, not a second copy of the format: the
        # point of the diff is that the reply agrees with the row, and a test
        # holding its own notation would drift from both.
        assert clock_time(start) in reply
        assert str(start.year) in reply
        assert f"AC-{appointment.id:06d}" in reply

    def test_two_different_bookings_produce_different_replies(
        self, patient, other_patient
    ):
        """The check a canned string cannot pass."""
        first = self._book(patient, "s-facts-2").reply
        second = self._book(
            other_patient, "s-facts-3", "I would like a dermatology appointment"
        ).reply

        assert first != second

    def test_the_reply_names_the_department_that_was_booked(
        self, patient, other_patient
    ):
        self._book(patient, "s-facts-4")
        second = self._book(
            other_patient, "s-facts-5", "I would like a dermatology appointment"
        )
        assert "Dermatology" in second.reply

    def test_a_reference_code_matches_its_own_appointment(
        self, patient, other_patient
    ):
        """Two bookings, two references. A reply carrying the other patient's
        reference would pass a "looks right" reading and fail here."""
        first = self._book(patient, "s-facts-6")
        second = self._book(
            other_patient, "s-facts-7", "I would like a dermatology appointment"
        )

        session = fresh()
        try:
            runs = {
                run.id: run.state.get("appointment_id")
                for run in session.query(WorkflowRun).all()
            }
        finally:
            session.close()

        assert f"AC-{runs[first.run_id]:06d}" in first.reply
        assert f"AC-{runs[second.run_id]:06d}" in second.reply


class TestWithdrawal:
    def test_withdrawing_mid_booking_cancels_the_run(self, patient):
        turn(patient, BOOKING, "s-withdraw-1")
        result = turn(patient, "actually never mind, forget it", "s-withdraw-1")

        assert result.message_class is MessageClass.WITHDRAWAL
        assert result.status == WorkflowStatus.CANCELLED.value

    def test_the_reason_is_recorded(self, patient):
        turn(patient, BOOKING, "s-withdraw-2")
        turn(patient, "never mind", "s-withdraw-2")

        session = fresh()
        try:
            run = session.query(WorkflowRun).order_by(WorkflowRun.id.desc()).first()
            assert run.cancellation_reason == "withdrawn"
        finally:
            session.close()

    def test_nothing_is_booked(self, patient):
        result = turn(patient, BOOKING, "s-withdraw-3")
        turn(patient, "never mind", "s-withdraw-3")

        session = fresh()
        try:
            assert appointments_for(session, result.run_id) == []
        finally:
            session.close()

    def test_the_patient_has_no_active_run_afterwards(self, patient):
        turn(patient, BOOKING, "s-withdraw-4")
        turn(patient, "forget it", "s-withdraw-4")

        session = fresh()
        try:
            profile_id = (
                session.query(WorkflowRun)
                .order_by(WorkflowRun.id.desc())
                .first()
                .patient_id
            )
            assert active_run(session, profile_id) is None
        finally:
            session.close()


class TestOffTopicDuringARun:
    """The PRD's named Layer-1 scenario, run through the real orchestrator."""

    def test_state_and_request_text_are_byte_identical(self, patient):
        turn(patient, BOOKING, "s-offtopic-1")

        session = fresh()
        try:
            run = session.query(WorkflowRun).order_by(WorkflowRun.id.desc()).first()
            before = (run.status, run.request_text, json.dumps(run.state, sort_keys=True))
        finally:
            session.close()

        turn(patient, "what's the weather like today?", "s-offtopic-1")

        session = fresh()
        try:
            run = session.query(WorkflowRun).order_by(WorkflowRun.id.desc()).first()
            after = (run.status, run.request_text, json.dumps(run.state, sort_keys=True))
        finally:
            session.close()

        assert before == after

    def test_the_reply_is_a_scope_template(self, patient):
        turn(patient, BOOKING, "s-offtopic-2")
        result = turn(patient, "who won the football last night?", "s-offtopic-2")

        assert result.message_class is MessageClass.OFF_TOPIC
        assert "appointments" in result.reply.lower()

    def test_no_escalation_is_created(self, patient):
        """Off-topic is noise, not a human-review case. The staff queue stays
        as clean of it as of panic-repeats."""
        turn(patient, BOOKING, "s-offtopic-3")
        turn(patient, "what's on television?", "s-offtopic-3")

        session = fresh()
        try:
            assert session.query(Escalation).count() == 0
        finally:
            session.close()

    def test_the_turn_is_still_traced(self, patient):
        """No run spawned, but the turn is bracketed like any other."""
        turn(patient, BOOKING, "s-offtopic-4")
        result = turn(patient, "what's the weather?", "s-offtopic-4")

        session = fresh()
        try:
            events = (
                session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
            )
            kinds = {event.event_type for event in events}
            assert TraceEventType.INBOUND in kinds
            assert TraceEventType.OUTBOUND in kinds
        finally:
            session.close()


class TestOffTopicWithNoRun:
    def test_no_run_is_spawned(self, patient):
        turn(patient, "what's the weather like?", "s-noscope-1")

        session = fresh()
        try:
            assert session.query(WorkflowRun).count() == 0
        finally:
            session.close()

    def test_the_turn_is_traced_with_a_null_run_id(self, patient):
        result = turn(patient, "tell me a joke", "s-noscope-2")

        session = fresh()
        try:
            events = (
                session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
            )
            assert events
            assert all(event.workflow_run_id is None for event in events)
        finally:
            session.close()

    def test_the_refusal_is_a_template_and_says_so(self, patient):
        """The scope reply is code-authored. It has to read identically under
        mock and live, and the timeline must not imply a model wrote it."""
        result = turn(patient, "tell me a joke", "s-noscope-3")

        assert result.reply == SCOPE_REPLY
        assert result.author is TraceAuthor.GUARD

    def test_the_gate_records_a_verdict(self, patient):
        result = turn(patient, "who won the football last night?", "s-noscope-4")

        session = fresh()
        try:
            gate = _guard(session, result.turn_id, "scope_gate")
        finally:
            session.close()
        assert gate["passed"] is False

    def test_the_gate_records_its_passes_too(self, patient):
        """A guard that logs only its firings is half an instrument: the
        expensive question is always "why did it not fire?"."""
        result = turn(patient, BOOKING, "s-noscope-5")

        session = fresh()
        try:
            gate = _guard(session, result.turn_id, "scope_gate")
        finally:
            session.close()
        assert gate["passed"] is True
        assert "book" in gate["detail"]["steps"]

    def test_an_off_topic_message_is_audited(self, patient):
        """Off-topic spawns no run, but the turn still happened."""
        turn(patient, "what's the weather like?", "s-noscope-6")

        session = fresh()
        try:
            assert (
                session.query(AuditEvent)
                .filter(AuditEvent.action == "scope_gate_refused")
                .count()
                == 1
            )
        finally:
            session.close()


class TestLowConfidenceRouting:
    """The second staff-approval trigger: routing that a human should decide.

    The seed carries an ambiguous case on purpose ("my kid has ear pain" —
    Pediatrics or ENT). Guessing confidently to avoid the handover is the
    failure this path exists to refuse.
    """

    AMBIGUOUS = "book an appointment, my kid has ear pain"

    def test_the_run_pauses_for_a_human(self, patient):
        result = turn(patient, self.AMBIGUOUS, "s-lowconf-1")

        assert result.status == WorkflowStatus.PENDING_REVIEW.value

    def test_an_escalation_is_opened_for_the_run(self, patient):
        result = turn(patient, self.AMBIGUOUS, "s-lowconf-2")

        session = fresh()
        try:
            escalations = (
                session.query(Escalation)
                .filter(Escalation.workflow_run_id == result.run_id)
                .all()
            )
            assert len(escalations) == 1
            assert escalations[0].kind is EscalationKind.LOW_CONFIDENCE_ROUTING
            assert escalations[0].status is EscalationStatus.OPEN
        finally:
            session.close()

    def test_nothing_is_booked_while_review_is_pending(self, patient):
        result = turn(patient, self.AMBIGUOUS, "s-lowconf-3")

        session = fresh()
        try:
            assert appointments_for(session, result.run_id) == []
        finally:
            session.close()


class TestAmbiguousConfirmation:
    def test_an_unreadable_answer_does_not_commit(self, patient):
        """"no wait — yes, the Tuesday one" can decline or trigger a re-ask.
        It can never book."""
        result = turn(patient, BOOKING, "s-ambig-1")
        turn(patient, "no wait - yes, the Tuesday one", "s-ambig-1")

        session = fresh()
        try:
            assert appointments_for(session, result.run_id) == []
        finally:
            session.close()

    def test_the_run_stays_in_pending_confirmation(self, patient):
        """A stall never resolves implicitly in either direction."""
        turn(patient, BOOKING, "s-ambig-2")
        turn(patient, "hmm, maybe", "s-ambig-2")

        session = fresh()
        try:
            run = session.query(WorkflowRun).order_by(WorkflowRun.id.desc()).first()
            assert run.status is WorkflowStatus.PENDING_CONFIRMATION
            assert run.proposed_slot_id is not None
        finally:
            session.close()

    def test_declining_clears_the_proposal(self, patient):
        turn(patient, BOOKING, "s-ambig-3")
        result = turn(patient, "no", "s-ambig-3")

        assert result.status == WorkflowStatus.IN_PROGRESS.value
        session = fresh()
        try:
            run = session.query(WorkflowRun).order_by(WorkflowRun.id.desc()).first()
            assert run.proposed_slot_id is None
        finally:
            session.close()

    def test_a_confirmation_is_read_before_any_model_call(self, patient):
        """The reader is code. Its verdict is a guard verdict, and it is
        recorded whether it fired or passed."""
        turn(patient, BOOKING, "s-ambig-4")
        result = turn(patient, "yes", "s-ambig-4")

        session = fresh()
        try:
            guards = [
                event
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.GUARD_VERDICT
            ]
            assert any(g.payload["guard"] == "confirmation_reader" for g in guards)
        finally:
            session.close()


class TestSlotSabotagedBetweenProposalAndConfirm:
    """Every commit-time failure exits the same way: clear the proposal, return
    to selection, offer alternatives.

    A failure path may never leave a stale proposal confirmable. A proposal
    pointing at a dead slot plus a patient pressing Confirm is a loop with no
    exit, which is the boundedness invariant applied to the unhappy path.
    """

    def _sabotage(self, run_id: int) -> None:
        """Somebody else takes the proposed slot before the patient answers."""
        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            slot = session.get(AppointmentSlot, run.proposed_slot_id)
            slot.status = SlotStatus.BOOKED
            session.commit()
        finally:
            session.close()

    def test_the_booking_does_not_happen(self, patient):
        proposal = turn(patient, BOOKING, "s-sabotage-1")
        self._sabotage(proposal.run_id)
        turn(patient, "yes", "s-sabotage-1")

        session = fresh()
        try:
            assert appointments_for(session, proposal.run_id) == []
        finally:
            session.close()

    def test_the_proposal_is_cleared(self, patient):
        proposal = turn(patient, BOOKING, "s-sabotage-2")
        self._sabotage(proposal.run_id)
        turn(patient, "yes", "s-sabotage-2")

        session = fresh()
        try:
            run = session.get(WorkflowRun, proposal.run_id)
            assert run.proposed_slot_id is None
            assert run.proposed_action is None
        finally:
            session.close()

    def test_a_second_confirm_click_no_ops(self, patient):
        """The double-click that would otherwise book against a dead proposal."""
        proposal = turn(patient, BOOKING, "s-sabotage-3")
        self._sabotage(proposal.run_id)
        turn(patient, "yes", "s-sabotage-3")
        turn(patient, "yes", "s-sabotage-3")

        session = fresh()
        try:
            assert appointments_for(session, proposal.run_id) == []
        finally:
            session.close()

    def test_the_run_returns_to_selection(self, patient):
        proposal = turn(patient, BOOKING, "s-sabotage-4")
        self._sabotage(proposal.run_id)
        result = turn(patient, "yes", "s-sabotage-4")

        assert result.status != WorkflowStatus.PENDING_CONFIRMATION.value
        assert result.status != WorkflowStatus.COMPLETED.value


class LoopingLlm(AgentCareLlm):
    """A provider that always wants one more tool call.

    Every call it makes succeeds, so no retry ladder is ever tripped. That is
    precisely the loop the iteration budget exists for.

    It answers the safety screen honestly and loops everywhere else. The
    behaviour under test is a budget blowing *inside the workflow* — a stub
    that stalled the guard instead would stop the turn one layer earlier and
    quietly test something else.
    """

    model: str = "looping-stub"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        if "submit_safety_verdict" in available_tool_names(llm_request):
            if "submit_safety_verdict" in called_tools(llm_request):
                yield text_response("screened")
            else:
                yield function_call_response(
                    "submit_safety_verdict",
                    {"category": "safe", "rationale": "stub: nothing clinical"},
                )
            return
        yield function_call_response("load_patient_context", {})


class TestIterationBudget:
    @pytest.fixture(autouse=True)
    def _loop_forever(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: LoopingLlm()
        )

    def test_the_run_fails_rather_than_looping(self, patient, settings):
        result = turn(patient, BOOKING, "s-budget-1")
        assert result.budget_exhausted is True

    def test_the_cap_is_the_configured_one(self, patient, settings):
        """Exactly the cap — not "eventually", not "about".

        Counted per agent, because that is what the budget is: a booking turn
        legitimately spends calls in the Coordinator, in Routing, and in
        Appointment, and one shared counter would put an ordinary request
        within a call of its own budget. So the looping agent must stop at
        exactly the cap while the guard that ran before it spends exactly one.
        """
        result = turn(patient, BOOKING, "s-budget-2")

        session = fresh()
        try:
            calls = [
                event
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.TOOL_CALL
            ]
        finally:
            session.close()

        per_agent = Counter(event.agent_name for event in calls)
        assert per_agent["coordinator"] == settings.max_tool_iterations
        assert per_agent["safety_screen"] == 1

    def test_the_exhaustion_is_recorded(self, patient):
        result = turn(patient, BOOKING, "s-budget-3")

        session = fresh()
        try:
            rejections = [
                event
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.VALIDATION
                and event.payload["what"] == "tool_iteration_budget"
            ]
            assert rejections
        finally:
            session.close()

    def test_the_patient_gets_a_graceful_message(self, patient):
        result = turn(patient, BOOKING, "s-budget-4")
        assert "staff" in result.reply.lower()


class ResubmittingLlm(AgentCareLlm):
    """A provider that re-calls a submit tool whose result was already accepted.

    Observed live on ``openai/gpt-4o-mini``, reduced to a stub: holding a
    single mandatory tool and asked again, it volunteers no text — it calls the
    tool again. In the trace, ``submit_safety_verdict`` was accepted at seq
    5-7, then re-called eight more times until the iteration budget fired and
    the turn failed with a template. The function_response round-trip was
    intact, so this is the model's behaviour and not the adapter's.

    The mock and llama happen to answer with text at that point. That is the
    only reason a defect living in *every* submit-style loop stayed invisible
    until a third provider ran.
    """

    model: str = "resubmitting-stub"
    category: str = "safe"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        available = available_tool_names(llm_request)
        if "submit_safety_verdict" in available:
            yield function_call_response(
                "submit_safety_verdict",
                {"category": self.category, "rationale": "stub: re-submitting"},
            )
            return
        if "submit_plan" in available:
            if "load_patient_context" not in called_tools(llm_request):
                yield function_call_response("load_patient_context", {})
                return
            yield function_call_response("submit_plan", {"steps": ["route", "book"]})
            return
        # Specialists are not the subject here; leave them a plain reply so the
        # turn ends on the loop control being tested rather than on a stall.
        yield text_response("stub")


class EmergencyResubmittingLlm(ResubmittingLlm):
    """The same defect, on the verdict whose loss would be silent and worst."""

    model: str = "resubmitting-emergency-stub"
    category: str = "emergency"


def _validations(session, turn_id: str, what: str) -> list[TraceEvent]:
    return [
        event
        for event in session.query(TraceEvent)
        .filter(TraceEvent.turn_id == turn_id)
        .order_by(TraceEvent.seq)
        .all()
        if event.event_type is TraceEventType.VALIDATION
        and event.payload["what"] == what
    ]


def _tool_calls(session, turn_id: str) -> list[TraceEvent]:
    return [
        event
        for event in session.query(TraceEvent)
        .filter(TraceEvent.turn_id == turn_id)
        .order_by(TraceEvent.seq)
        .all()
        if event.event_type is TraceEventType.TOOL_CALL
    ]


class TestAcceptedSubmitEndsTheLoop:
    """A submit tool's acceptance is the end of that decision, and code says so.

    The iteration budget is the outer bound and it works — but reaching it
    costs eight wasted calls and *fails a turn that had already succeeded*.
    The verdict was in the belt at call one; everything after it is the model
    being asked a question it has already answered.
    """

    @pytest.fixture(autouse=True)
    def _resubmit(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: ResubmittingLlm()
        )

    def test_the_turn_completes_instead_of_exhausting_the_budget(self, patient):
        result = turn(patient, BOOKING, "s-resubmit-1")
        assert result.budget_exhausted is False

    def test_the_safety_screen_calls_its_tool_once(self, patient):
        """Terminal on acceptance: the screen's prompt forbids it a reply and
        ``llm_screen`` discards the text, so the model call that would follow
        acceptance has nothing left to produce."""
        result = turn(patient, BOOKING, "s-resubmit-2")

        session = fresh()
        try:
            calls = Counter(
                event.payload["tool"]
                for event in _tool_calls(session, result.turn_id)
                if event.agent_name == "safety_screen"
            )
        finally:
            session.close()

        assert calls["submit_safety_verdict"] == 1

    def test_the_screen_is_asked_once_and_not_asked_again(self, patient):
        """The saving is a whole LLM call per turn, on every provider: the
        screen runs on every message, and its second call never had a use."""
        result = turn(patient, BOOKING, "s-resubmit-3")

        session = fresh()
        try:
            requests = [
                event
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.LLM_REQUEST
                and event.agent_name == "safety_screen"
            ]
        finally:
            session.close()

        assert len(requests) == 1

    def test_a_repeat_of_an_accepted_call_is_refused(self, patient):
        """The Coordinator's ``submit_plan`` is not terminal — it may be
        followed by an acknowledgement — so the bound there is repetition:
        the same call, already accepted, is not run a second time."""
        result = turn(patient, BOOKING, "s-resubmit-4")

        session = fresh()
        try:
            plans = [
                event
                for event in _tool_calls(session, result.turn_id)
                if event.payload["tool"] == "submit_plan"
            ]
            refusals = _validations(session, result.turn_id, "repeated_tool_call")
        finally:
            session.close()

        assert len(plans) == 1
        assert refusals, "the refusal must be in the trace, not merely happen"

    def test_the_accepted_plan_survives_the_early_exit(self, patient):
        """Distrust green: ending the loop must not throw away what the tool
        accepted. A turn that completes but silently loses the plan would pass
        every assertion above."""
        result = turn(patient, BOOKING, "s-resubmit-5")
        # The submitted steps, in canonical order, ahead of the ones
        # ``validate_plan`` implies for a booking.
        assert result.plan[:2] == ["route", "book"]
        assert result.run_id is not None

    def test_the_trace_is_still_well_formed(self, patient):
        turn(patient, BOOKING, "s-resubmit-6")

        session = fresh()
        try:
            assert_well_formed(session)
        finally:
            session.close()


class TestTheTerminalExitKeepsTheVerdict:
    """The safety verdict is the one whose loss would be silent and worst: the
    turn would carry on booking as though nothing had been said."""

    @pytest.fixture(autouse=True)
    def _resubmit(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider",
            lambda name=None: EmergencyResubmittingLlm(),
        )

    def test_an_emergency_still_escalates(self, patient):
        result = turn(patient, BOOKING, "s-resubmit-safety")

        session = fresh()
        try:
            escalations = (
                session.query(Escalation)
                .filter(Escalation.kind == EscalationKind.SAFETY)
                .all()
            )
        finally:
            session.close()

        assert escalations, "the accepted verdict was dropped with the loop"
        assert result.budget_exhausted is False


RESTART_SCRIPT = """
import asyncio, os, sys
sys.path.insert(0, {root!r})
from app.db import SessionLocal
from app.models import User
from app.orchestrator import run_workflow

session = SessionLocal()
user = session.query(User).filter(User.email == {email!r}).one()
session.close()
result = asyncio.run(run_workflow(user, {message!r}, {session_id!r}))
print("PID:" + str(os.getpid()))
print("STATUS:" + str(result.status))
print("REPLY:" + " ".join(result.reply.split())[:100])
"""


class TestTier1MemorySurvivesARestart:
    """PRD story 12. A conversation continues after the process that started
    it is gone — which is only provable in a second interpreter, because
    imports, engines, and caches all survive anything smaller."""

    def _run_in_new_process(self, message, session_id, tmp_root):
        script = RESTART_SCRIPT.format(
            root=str(os.getcwd()),
            email=PATIENT_EMAIL,
            message=message,
            session_id=session_id,
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return {
            line.split(":", 1)[0]: line.split(":", 1)[1]
            for line in completed.stdout.splitlines()
            if ":" in line
        }

    def test_a_confirmation_lands_on_a_run_started_by_a_dead_process(
        self, patient, tmp_root
    ):
        first = self._run_in_new_process(BOOKING, "s-restart-1", tmp_root)
        assert first["STATUS"] == WorkflowStatus.PENDING_CONFIRMATION.value

        second = self._run_in_new_process("yes", "s-restart-1", tmp_root)
        assert second["PID"] != first["PID"], "not a real restart"
        assert second["STATUS"] == WorkflowStatus.COMPLETED.value

    def test_the_appointment_exists_afterwards(self, patient, tmp_root):
        self._run_in_new_process(BOOKING, "s-restart-2", tmp_root)
        self._run_in_new_process("yes", "s-restart-2", tmp_root)

        session = fresh()
        try:
            run = session.query(WorkflowRun).order_by(WorkflowRun.id.desc()).first()
            assert appointments_for(session, run.id)
        finally:
            session.close()


class TestTheRowWinsOverSessionState:
    def test_a_second_conversation_sees_the_same_active_run(self, patient):
        """Sessions are cheap and disposable; continuity lives in the database.
        A patient opening a fresh conversation still has their live request."""
        turn(patient, BOOKING, "s-authority-1")
        result = turn(patient, "yes", "s-authority-2")

        assert result.status == WorkflowStatus.COMPLETED.value

    def test_the_clock_reaches_the_prompt(self, patient):
        """A frozen clock has to reach the prompt as well as the tools, or a
        golden run and the model disagree about what day it is."""
        from app.agents import prompts

        clock.freeze(clock.today())
        assert clock.today().isoformat() in prompts.instruction("coordinator")


class TestTheProposalReplyOffersAChoice:
    """A proposal goes out with the shortlist it was drawn from.

    The typed proposal is still exactly one slot — the state machine is not
    involved in any of this. What changed is that "no" stopped being the only
    thing a patient could say to a time that did not suit them.
    """

    def test_three_times_are_named_and_the_first_is_the_one_held(self, patient):
        result = turn(patient, BOOKING, "s-offer-1")

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            held = session.get(AppointmentSlot, run.proposed_slot_id)
            offered = run.state["offered_slot_ids"]
        finally:
            session.close()

        lines = [
            line for line in result.reply.splitlines() if line[:2] in ("1.", "2.", "3.")
        ]
        assert len(lines) == 3
        assert "holding" in lines[0], "the held time is the first option"
        assert offered[0] == held.id

    def test_every_time_shown_is_recorded_as_offered(self, patient):
        """The set a re-proposal is checked against is built here. If showing
        and recording could come apart, the patient would be offered a time the
        guard would then refuse."""
        result = turn(patient, BOOKING, "s-offer-2")

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            recorded = set(run.state["offered_slot_ids"])
            shown = {
                slot_id
                for slot_id in recorded
                if clock_time(session.get(AppointmentSlot, slot_id).start_time)
                in result.reply
            }
        finally:
            session.close()

        # Recorded is deliberately the *wider* set: everything the search
        # returned is answerable, not only the three lines the reply prints.
        # "Lets go with 4pm slot" named a time from further down the payload,
        # and recording only what was printed made the guard refuse a slot the
        # tool had just produced. What must hold is the other direction —
        # nothing is printed that was not recorded, or the patient is offered
        # a time the guard would then reject.
        assert shown, "the reply printed none of the recorded slots"
        assert shown <= recorded

    def test_the_options_are_distinct_times(self, patient):
        """Three doctors free at 09:00 is one appointment three times, not a
        choice — and offering it as one hides the 10:00 they would have taken."""
        result = turn(patient, BOOKING, "s-offer-3")

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            starts = [
                session.get(AppointmentSlot, slot_id).start_time
                for slot_id in run.state["offered_slot_ids"]
            ]
        finally:
            session.close()

        assert len(set(starts)) == len(starts)


class TestTheReAskCarriesTheProposal:
    """Item 4's guarantee, checked against the row rather than a fixed string."""

    def test_it_names_the_doctor_and_day_being_held(self, patient):
        turn(patient, BOOKING, "s-reask-1")
        result = turn(patient, "looks good. lets book that time", "s-reask-1")

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            slot = session.get(AppointmentSlot, run.proposed_slot_id)
            doctor = slot.doctor.name
            day = f"{slot.start_time:%A} {slot.start_time.day}"
        finally:
            session.close()

        assert doctor in result.reply
        assert day in result.reply

    def test_it_promises_no_action(self, patient):
        from app.workflow.replies import promises_action

        turn(patient, BOOKING, "s-reask-2")
        result = turn(patient, "looks good. lets book that time", "s-reask-2")

        assert promises_action(result.reply) is False

    def test_nothing_is_booked_by_any_of_it(self, patient):
        """The rule underneath everything in this batch. Paraphrase never
        commits — the improvements are around that, never through it."""
        result = turn(patient, BOOKING, "s-reask-3")
        for words in ("looks good. lets book that time", "that works for me"):
            turn(patient, words, "s-reask-3")

        session = fresh()
        try:
            assert appointments_for(session, result.run_id) == []
            run = session.get(WorkflowRun, result.run_id)
            assert run.status is WorkflowStatus.PENDING_CONFIRMATION
        finally:
            session.close()


class TestTheReceiptIsAssembled:
    """Item 2, end to end: what the patient is told after a commit."""

    def test_it_is_code_authored(self, patient):
        turn(patient, BOOKING, "s-receipt-1")
        result = turn(patient, "yes", "s-receipt-1")
        assert result.author is TraceAuthor.TEMPLATE

    def test_the_only_date_in_it_is_the_appointments(self, patient):
        """The reminder fires the day *before* the visit, and the live receipt
        printed that date as the appointment's. A patient reading it arrives a
        day early to an empty clinic."""
        turn(patient, BOOKING, "s-receipt-2")
        result = turn(patient, "yes", "s-receipt-2")

        session = fresh()
        try:
            appointment = appointments_for(session, result.run_id)[0]
            starts = appointment.slot.start_time
            reminders = [
                reminder.scheduled_at
                for reminder in session.query(Reminder)
                .filter(Reminder.appointment_id == appointment.id)
                .all()
            ]
        finally:
            session.close()

        assert reminders, "a booking schedules a reminder; without one this proves nothing"
        for fires in reminders:
            if fires.date() != starts.date():
                assert f"{fires.day} {fires:%B}" not in result.reply

    def test_the_upload_pointer_rides_with_a_missing_document(self, patient):
        """Item 6. Cardiology's rules leave something outstanding for the
        seeded patient, and being told what is missing without being told
        where to put it is a chore rather than an instruction."""
        turn(patient, BOOKING, "s-receipt-3")
        result = turn(patient, "yes", "s-receipt-3")

        if "before your visit" in result.reply or "Optional but helpful" in result.reply:
            assert "Documents page" in result.reply

    def test_it_never_contradicts_itself_about_follow_up(self, patient):
        """Live, one receipt said a thing was "recorded for follow-up" and that
        there were "no outstanding follow-up tasks", in the same breath."""
        turn(patient, BOOKING, "s-receipt-4")
        result = turn(patient, "yes", "s-receipt-4")

        lowered = result.reply.lower()
        assert not ("no outstanding" in lowered and "recorded for follow" in lowered)


class ReproposingLlm(MockLlm):
    """The mock, deprived of the one hint ``gpt-4o-mini`` ignored.

    The specialist's typed task carries ``committed`` — which verb has already
    landed — and the mock reads it and states the outcome instead of proposing
    again. The live model was handed the same field and proposed anyway.

    That difference is the whole reason this stub exists. A field in a JSON
    task is a *proposal-side* mitigation: advisory, and only as good as the
    model's attention. This is what declining the advice looks like, and
    without it the mock cannot reproduce the defect at all — the scenario for
    it passes with the guard removed.
    """

    model: str = "reproposing-stub"

    def _change_appointment(self, llm_request, done, task, step):  # noqa: ANN001
        return super()._change_appointment(
            llm_request,
            done,
            {key: value for key, value in task.items() if key != "committed"},
            step,
        )


def _audit_count(session, action: str) -> int:
    return session.query(AuditEvent).filter(AuditEvent.action == action).count()


class TestACommittedVerbIsNotReEntered:
    """Item 1, against a provider that behaves the way the live one did.

    The live shape: Confirm committed a reschedule to the 4 August 11:00 slot
    the patient had chosen; the reschedule step then ran *again* in the same
    turn, called ``find_slots_for_reschedule`` with no window, proposed the
    earliest free slot, and put the run back into ``pending_confirmation``. The
    patient — seeing a receipt and a fresh proposal card — clicked Confirm a
    second time and their appointment moved to 28 July 09:00, a time nothing
    had ever offered them. Two ``appointment_rescheduled`` audits, eighteen
    seconds apart.

    The audit count is the assertion that can see it. The appointment's final
    status is ``confirmed`` either way, and its slot after one bad move looks
    exactly like a slot after one good one.
    """

    @pytest.fixture(autouse=True)
    def _repropose(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: ReproposingLlm()
        )

    def _held_slot(self, session) -> int:
        return session.get(Appointment, SEEDED_APPOINTMENT_ID).slot_id

    def test_the_run_does_not_put_itself_back_up_for_confirmation(self, patient):
        first = turn(patient, "please reschedule my appointment to next week", "s-twice-1")
        assert first.status == WorkflowStatus.PENDING_CONFIRMATION.value

        result = asyncio.run(apply_patient_action(patient, "confirm", "s-twice-1"))

        assert result.status != WorkflowStatus.PENDING_CONFIRMATION.value
        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.proposed_action is None
            assert run.proposed_slot_id is None
        finally:
            session.close()

    def test_a_second_click_moves_nothing(self, patient):
        turn(patient, "please reschedule my appointment to next week", "s-twice-2")
        asyncio.run(apply_patient_action(patient, "confirm", "s-twice-2"))

        session = fresh()
        try:
            after_first = self._held_slot(session)
        finally:
            session.close()

        asyncio.run(apply_patient_action(patient, "confirm", "s-twice-2"))

        session = fresh()
        try:
            assert self._held_slot(session) == after_first
            assert _audit_count(session, "appointment_rescheduled") == 1
        finally:
            session.close()

    def test_the_second_click_says_there_is_nothing_to_confirm(self, patient):
        """Layer (c), which is what makes a double-click harmless even if the
        two layers above it were ever to regress."""
        turn(patient, "please reschedule my appointment to next week", "s-twice-3")
        asyncio.run(apply_patient_action(patient, "confirm", "s-twice-3"))
        second = asyncio.run(apply_patient_action(patient, "confirm", "s-twice-3"))

        assert second.author is TraceAuthor.TEMPLATE
        assert "nothing waiting for your confirmation" in second.reply.lower()

    def test_the_specialist_is_not_dispatched_for_the_committed_verb(self, patient):
        """The mechanism, not just the outcome: the step settles from
        ``committed_action`` without a second LLM request for it."""
        turn(patient, "please reschedule my appointment to next week", "s-twice-4")
        result = asyncio.run(apply_patient_action(patient, "confirm", "s-twice-4"))

        session = fresh()
        try:
            requests = [
                event
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.LLM_REQUEST
                and event.agent_name == "appointment"
            ]
            settled = _validations(session, result.turn_id, "step_already_committed")
        finally:
            session.close()

        assert requests == []
        assert len(settled) == 1


class TestAFailedRunReachesAHuman:
    """Item 4. ``FAILED_REPLY`` says "I've flagged it for them" — so something
    has to have been flagged.

    Live, a run failed on its re-plan budget and left **zero** escalation rows.
    The patient was told staff had been told; staff had not been told. A
    template that makes a promise is the thing that has to keep it, and until
    this existed the sentence was about nothing.
    """

    @pytest.fixture(autouse=True)
    def _loop_forever(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: LoopingLlm()
        )

    def test_it_opens_a_system_failure_escalation(self, patient, settings):
        result = turn(patient, BOOKING, "s-failesc-1")
        assert result.budget_exhausted is True

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.status is WorkflowStatus.FAILED
            kinds = [escalation.kind for escalation in run.escalations]
        finally:
            session.close()

        assert kinds == [EscalationKind.SYSTEM_FAILURE]

    def test_the_queue_a_human_reads_actually_shows_it(self, patient, settings):
        """The row existing and the row being *in the queue* are different
        claims, and only the second one is the promise the template made."""
        turn(patient, BOOKING, "s-failesc-2")

        session = fresh()
        try:
            queued = [
                escalation
                for escalation in session.query(Escalation).all()
                if escalation.status in (EscalationStatus.OPEN, EscalationStatus.ACKNOWLEDGED)
            ]
        finally:
            session.close()

        assert [escalation.kind for escalation in queued] == [
            EscalationKind.SYSTEM_FAILURE
        ]

    def test_repeated_failures_are_one_item_with_two_occurrences(
        self, patient, settings
    ):
        """Bounded like every other escalation. A system failing twice must not
        be able to fill the queue a human is supposed to read."""
        turn(patient, BOOKING, "s-failesc-3")
        turn(patient, BOOKING, "s-failesc-3")

        session = fresh()
        try:
            rows = (
                session.query(Escalation)
                .filter(Escalation.kind == EscalationKind.SYSTEM_FAILURE)
                .all()
            )
            counts = sorted(row.occurrence_count for row in rows)
        finally:
            session.close()

        # A second failing turn either re-fails the same run or opens its own;
        # either way the queue grows by records, not by noise.
        assert sum(counts) == 2


class TestADeadRunLeavesNothingBehind:
    """Item 3, through the seam. The derivation invariant applied to a run's
    own leftovers: an escalation a human would work on, and a proposal on a row
    whose status says it is over."""

    def test_withdrawing_closes_the_review_it_was_waiting_for(self, patient):
        turn(patient, "book an appointment, my kid has ear pain", "s-dead-1")
        turn(patient, "actually never mind, forget it", "s-dead-1")

        session = fresh()
        try:
            open_rows = [
                escalation
                for escalation in session.query(Escalation).all()
                if escalation.status in (EscalationStatus.OPEN, EscalationStatus.ACKNOWLEDGED)
            ]
            resolved = session.query(Escalation).all()
        finally:
            session.close()

        assert open_rows == []
        assert resolved[0].resolution_note == "Withdrawn by the patient."

    def test_a_withdrawn_run_keeps_no_proposal(self, patient):
        turn(patient, BOOKING, "s-dead-2")
        result = turn(patient, "actually never mind, forget it", "s-dead-2")

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.status is WorkflowStatus.CANCELLED
            assert run.proposed_action is None
            assert run.proposed_slot_id is None
            assert run.proposed_appointment_id is None
        finally:
            session.close()

    def test_a_safety_escalation_survives_a_withdrawal(self, patient):
        """The exemption, end to end. ``escalated`` is terminal for automation,
        so the withdrawal cannot even reach the mapping — and if it ever could,
        the queue item would still be there."""
        turn(patient, "I'm having chest pain and my left arm is numb", "s-dead-3")
        turn(patient, "actually never mind, forget it", "s-dead-3")

        session = fresh()
        try:
            safety = (
                session.query(Escalation)
                .filter(Escalation.kind == EscalationKind.SAFETY)
                .all()
            )
        finally:
            session.close()

        assert len(safety) == 1
        assert safety[0].status is EscalationStatus.OPEN


class UnplanningLlm(AgentCareLlm):
    """Safe on the screen, silent on the plan. The live Coordinator's failure
    mode, isolated: it answers the safety screen and then says something
    conversational instead of calling ``submit_plan``."""

    model: str = "unplanning-stub"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        available = available_tool_names(llm_request)
        if "submit_safety_verdict" in available:
            yield function_call_response(
                "submit_safety_verdict",
                {"category": "safe", "rationale": "stub: administrative"},
            )
            return
        yield text_response("stub: no plan")


class TestTheScopeGateDoesNotRefuseItsOwnSubject:
    """Item 6 at the gate — which is where the live refusals happened.

    "can you tell me my appointments" produced three ``scope_gate_refused``
    audits across one session and no run at all, while a differently-worded ask
    for the same thing worked. The Coordinator failing to plan for a message is
    a different fact from the message being out of scope, and the two had the
    same answer.

    Code cannot supply the plan the Coordinator did not produce — that would
    put the deterministic layer in the planning bin — so what changes is the
    reply: "tell me more", not "I don't do that". (Round 5 carved out the one
    exception: a message that *states* an appointment verb beside an
    appointment noun, for a patient who has one. See
    ``TestAStatedVerbSurvivesThePlan``. These phrasings state no verb, so the
    veto is still what answers them.)

    The live message itself has since moved on: "can you tell me my
    appointments" is a *listing question* and is now answered from the rows
    before planning is reached at all (see
    ``TestListingQuestionsAreAnsweredFromTheRows``). What remains for the veto
    is the wider case — a message that names something this system owns and
    still produced no plan — so these use a phrasing that names the subject
    without asking for a list.
    """

    @pytest.fixture
    def unplanning(self, monkeypatch):
        """A Coordinator that submits no plan — which is what the live one did.

        The mock plans "can you tell me my appointments" happily, so under it
        the gate is never reached and these assertions would pass however the
        gate behaved. gpt-4o-mini produced no plan for that wording and a plan
        for another, which is the whole defect: a message's scope must not
        depend on which sentence the model happened to find plannable.
        """
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: UnplanningLlm()
        )

    def test_a_message_naming_appointments_is_not_refused(self, patient, unplanning):
        result = turn(patient, "my appointment situation is confusing", "s-veto-1")
        assert result.reply != SCOPE_REPLY
        assert result.author is TraceAuthor.TEMPLATE

    def test_it_is_never_answered_with_a_scope_refusal_audit(self, patient, unplanning):
        turn(patient, "my appointment situation is confusing", "s-veto-2")

        session = fresh()
        try:
            refused = (
                session.query(AuditEvent)
                .filter(AuditEvent.action == "scope_gate_refused")
                .count()
            )
            vetoed = (
                session.query(AuditEvent)
                .filter(AuditEvent.action == "scope_gate_vetoed")
                .count()
            )
        finally:
            session.close()

        assert refused == 0
        assert vetoed == 1

    def test_a_genuinely_off_topic_message_is_byte_identical(self, patient):
        """The direction that must not move. A veto that swallowed these would
        have traded a working refusal for a system that refuses nothing."""
        for index, message in enumerate(
            ("what's the weather like?", "how is nvidia stock doing", "who won the fifa final")
        ):
            result = turn(patient, message, f"s-veto-off-{index}")
            assert result.reply == SCOPE_REPLY
            assert result.author is TraceAuthor.GUARD

        session = fresh()
        try:
            assert (
                session.query(AuditEvent)
                .filter(AuditEvent.action == "scope_gate_refused")
                .count()
                == 3
            )
            assert session.query(WorkflowRun).count() == 0
        finally:
            session.close()

    def test_the_gate_records_which_way_it_went(self, patient, unplanning):
        result = turn(patient, "my appointment situation is confusing", "s-veto-4")

        session = fresh()
        try:
            gate = _guard(session, result.turn_id, "scope_gate")
        finally:
            session.close()

        assert gate["detail"]["domain_subject"] is True


class MisplanningLlm(AgentCareLlm):
    """A Coordinator that plans a *booking* for a reschedule request.

    The live failure, isolated. "Please reschedule my appointment to next week"
    reached ``submit_plan`` as ``["route", "book"]``, routed to General Medicine
    with low confidence, and queued for a staff decision about which department
    should own an appointment that already has one. The same sentence had
    worked hours before and nothing in between touched planning: replayed live
    three times, one phrasing produced a correct plan, one produced this, and
    one produced no plan at all.

    The mock cannot show any of that — it plans reschedules correctly every
    time, so the correction's own scenario passes with the correction removed.
    """

    model: str = "misplanning-stub"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        available = available_tool_names(llm_request)
        if "submit_safety_verdict" in available:
            yield function_call_response(
                "submit_safety_verdict",
                {"category": "safe", "rationale": "stub: administrative"},
            )
            return
        if "submit_plan" in available and "submit_plan" not in called_tools(llm_request):
            yield function_call_response("submit_plan", {"steps": ["route", "book"]})
            return
        yield text_response("stub: planned a booking")


class TestAStatedVerbSurvivesThePlan:
    """Round 5 item 2(c). A verb the patient typed is not the model's to reread.

    Two provider failures, one rule. The plan came back naming the wrong verb
    (``MisplanningLlm``) or naming nothing at all (``UnplanningLlm``), and both
    left a patient with an appointment on the books being asked which
    department their reschedule belonged to — or told to rephrase. Code now
    checks the plan against the verb the message states, and states it in the
    trace either way.

    The correction is narrow on purpose and the last test is what keeps it
    narrow: a patient with nothing booked is left entirely to the model,
    because "reschedule my appointment" from someone who has none is a badly
    worded booking and only the model can tell.
    """

    RESCHEDULE = "please reschedule my appointment to next week"

    def _plan_of(self, session) -> list[str]:
        run = session.query(WorkflowRun).order_by(WorkflowRun.id.desc()).first()
        return list(run.plan or []) if run else []

    def test_a_booking_plan_for_a_reschedule_is_corrected(self, patient, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: MisplanningLlm()
        )
        turn(patient, self.RESCHEDULE, "s-verb-1")

        session = fresh()
        try:
            assert self._plan_of(session) == ["reschedule", "follow_up"]
        finally:
            session.close()

    def test_the_corrected_run_never_enters_routing(self, patient, monkeypatch):
        """The consequence that was actually costing the patient: routing a
        reschedule asks a human which department owns an appointment that
        already names one."""
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: MisplanningLlm()
        )
        result = turn(patient, self.RESCHEDULE, "s-verb-2")

        session = fresh()
        try:
            assert "route" not in self._plan_of(session)
            assert session.query(Escalation).count() == 0
        finally:
            session.close()
        assert result.status != WorkflowStatus.PENDING_REVIEW.value

    def test_no_plan_at_all_is_also_corrected(self, patient, monkeypatch):
        """The other live outcome for the same sentence. "Tell me more" is the
        honest answer to a message nobody could plan for; it is the wrong
        answer to one that names its verb outright."""
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: UnplanningLlm()
        )
        result = turn(patient, self.RESCHEDULE, "s-verb-3")

        session = fresh()
        try:
            assert self._plan_of(session) == ["reschedule", "follow_up"]
        finally:
            session.close()
        assert result.reply != NO_PLAN_REPLY

    def test_the_override_is_traced(self, patient, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: MisplanningLlm()
        )
        result = turn(patient, self.RESCHEDULE, "s-verb-4")

        session = fresh()
        try:
            events = [
                event.payload
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.VALIDATION
                and event.payload.get("what") == "plan_change_verb"
            ]
        finally:
            session.close()

        assert events and events[0]["accepted"] is False
        assert events[0]["detail"]["applied"] == ["reschedule", "follow_up"]

    def test_agreement_is_recorded_too(self, patient):
        """Under the mock the plan is already right, so this records an
        agreement — which is the event that says the check ran at all. Without
        it, "the correction never fired" and "the correction is not wired in"
        look identical in the trace."""
        result = turn(patient, self.RESCHEDULE, "s-verb-5")

        session = fresh()
        try:
            events = [
                event.payload
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.VALIDATION
                and event.payload.get("what") == "plan_change_verb"
            ]
        finally:
            session.close()

        assert events and events[0]["accepted"] is True

    def test_a_patient_with_nothing_booked_is_left_to_the_model(
        self, other_patient, monkeypatch
    ):
        """The falsification. Rohan has no appointment, so there is nothing to
        reschedule and the model's plan stands — a correction that fired here
        would be inventing a target."""
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: MisplanningLlm()
        )
        turn(other_patient, self.RESCHEDULE, "s-verb-6")

        session = fresh()
        try:
            assert self._plan_of(session) == ["route", "book", "documents", "follow_up"]
        finally:
            session.close()


class SupersedingLlm(AgentCareLlm):
    """A Coordinator that calls every message during a run a new request.

    Live, "can you give me slots for next week?" — asked while an Orthopedics
    proposal stood — was classified ``conflicting``, and the booking the
    patient was in the middle of making was cancelled to start a fresh search
    for the same thing. The mock reads that message as a side question, so
    under it the refinement rule is never even consulted and its tests would
    pass with the rule deleted.
    """

    model: str = "superseding-stub"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        available = available_tool_names(llm_request)
        if "submit_safety_verdict" in available:
            yield function_call_response(
                "submit_safety_verdict",
                {"category": "safe", "rationale": "stub: administrative"},
            )
            return
        if "classify_message" in available and "classify_message" not in called_tools(
            llm_request
        ):
            yield function_call_response(
                "classify_message",
                {"message_class": "conflicting", "incoming_steps": ["book"]},
            )
            return
        yield text_response("stub: treating this as a new request")


class TestARefinementDoesNotReplaceTheRunItRefines:
    """Round 5 item 2, (a) and (b), at the orchestrator.

    A question about times, asked while a time is being held, must leave the
    offer standing and come back with times. It did neither: the run was
    superseded, the replacement had no department to route by, routing queued
    it for review, and the patient was told *"a member of staff will assist you
    with that"* — a referral to a human for a question the slot table answers.
    """

    QUESTION = "can you give me slots for next week?"

    def _run_row(self, session):
        return session.query(WorkflowRun).order_by(WorkflowRun.id.desc()).first()

    def _hold(self, patient, session_id: str):
        result = turn(patient, BOOKING, session_id)
        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        session = fresh()
        try:
            run = self._run_row(session)
            return run.id, run.proposed_slot_id
        finally:
            session.close()

    def test_the_run_survives_and_the_offer_is_still_held(self, patient, monkeypatch):
        run_id, held = self._hold(patient, "s-refine-1")
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: SupersedingLlm()
        )
        result = turn(patient, self.QUESTION, "s-refine-1")

        session = fresh()
        try:
            assert session.query(WorkflowRun).count() == 1
            run = self._run_row(session)
            assert (run.id, run.proposed_slot_id) == (run_id, held)
            assert run.status is WorkflowStatus.PENDING_CONFIRMATION
        finally:
            session.close()
        assert result.run_id == run_id

    def test_the_question_is_answered_with_times(self, patient, monkeypatch):
        """Refusing the supersede alone would leave the message unanswered,
        which is half a fix: the model called it a new request, so it never
        asked for the times. Code runs the search itself."""
        self._hold(patient, "s-refine-2")
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: SupersedingLlm()
        )
        result = turn(patient, self.QUESTION, "s-refine-2")

        assert "free" in result.reply.lower()
        assert result.author is TraceAuthor.TEMPLATE

    def test_nobody_is_referred_to_staff(self, patient, monkeypatch):
        self._hold(patient, "s-refine-3")
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: SupersedingLlm()
        )
        result = turn(patient, self.QUESTION, "s-refine-3")

        assert "member of staff" not in result.reply.lower()
        session = fresh()
        try:
            assert session.query(Escalation).count() == 0
        finally:
            session.close()

    def test_the_request_text_is_untouched(self, patient, monkeypatch):
        """A refused supersede that edited the request text would contaminate
        what the next search reads — the write the refusal exists to prevent,
        arriving by another door."""
        self._hold(patient, "s-refine-4")
        session = fresh()
        try:
            before = self._run_row(session).request_text
        finally:
            session.close()

        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: SupersedingLlm()
        )
        turn(patient, self.QUESTION, "s-refine-4")

        session = fresh()
        try:
            assert self._run_row(session).request_text == before
        finally:
            session.close()

    def test_a_different_department_still_replaces_it(self, patient, monkeypatch):
        """The counterexample, through the whole turn rather than the mapping
        alone. "Instead" names a subject this run is not about, and a rule that
        swallowed it would have made the system unable to change its mind."""
        self._hold(patient, "s-refine-5")
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: SupersedingLlm()
        )
        turn(patient, "book me a dermatology appointment instead", "s-refine-5")

        session = fresh()
        try:
            assert session.query(WorkflowRun).count() == 2
        finally:
            session.close()


class ThoroughDocumentLlm(MockLlm):
    """The mock, doing what the prompt used to ask for: *every* pending upload.

    ``MockLlm._verify_next`` has always taken ``pending[0]`` and documented why
    — a bound, not a shortcut. So the understudy could never reproduce this,
    and the live failure lived entirely in the gap between that bound and the
    prompt, which said "For each one". ``gpt-4o-mini`` obliged: three seeded
    documents, nine tool calls, a cap of eight, and the ninth was the diff.

    This stub is the mock with the bound taken out of it — which is exactly
    what the live provider was.
    """

    model: str = "thorough-document-stub"

    def _verify_next(self, llm_request, available, done):  # noqa: ANN001
        if "list_unverified_documents" not in available:
            return None
        if "list_unverified_documents" not in done:
            return function_call_response("list_unverified_documents", {})

        listed = latest_tool_result(llm_request, "list_unverified_documents")
        pending = (listed.payload.get("documents") if listed else None) or []
        verified = {
            call.args.get("document_id")
            for call in _calls_to(llm_request, "submit_document_verification")
        }
        for target in pending:
            if target["document_id"] in verified:
                continue
            reads = {
                call.args.get("document_id")
                for call in _calls_to(llm_request, "read_document_text")
            }
            if target["document_id"] not in reads:
                return function_call_response(
                    "read_document_text", {"document_id": target["document_id"]}
                )
            extracted = latest_tool_result(llm_request, "read_document_text")
            payload = extracted.payload if extracted else {}
            detected, matches = self._detect_type(
                payload.get("text") or "",
                declared=target.get("declared_type") or "",
                extractable=bool(payload.get("extracted")),
            )
            return function_call_response(
                "submit_document_verification",
                {
                    "document_id": target["document_id"],
                    "detected_type": detected,
                    "matches": matches,
                },
            )
        return None


def _calls_to(llm_request, name: str) -> list:
    """Every function call to ``name`` in this agent turn's history."""
    found = []
    for content in llm_request.contents or []:
        for part in content.parts or []:
            call = getattr(part, "function_call", None)
            if call is not None and call.name == name:
                found.append(call)
    return found


class TestOneDocumentPerTurn:
    """A booking that succeeded must not end on the failure notice.

    The live shape, from the round-5 sweep and reproduced in three separate
    conversations: Confirm commits the appointment, the documents step works
    through all three seeded uploads, and the budget fires on
    ``diff_required_documents`` — so the run goes ``failed``, a
    ``system_failure`` escalation opens for a booking that had *worked*, and
    ``record_missing_documents`` never runs, leaving the patient with no idea
    what to bring. The receipt is assembled from rows, so it goes out looking
    perfect on top of all of it: the patient sees nothing wrong and the staff
    queue sees a failure that isn't one.
    """

    @pytest.fixture(autouse=True)
    def _thorough(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: ThoroughDocumentLlm()
        )

    def test_the_run_completes(self, patient):
        turn(patient, BOOKING, "s-docs-1")
        result = turn(patient, "yes", "s-docs-1")

        assert result.status == WorkflowStatus.COMPLETED.value
        assert result.budget_exhausted is False

    def test_no_failure_escalation_is_opened_for_a_booking_that_worked(self, patient):
        turn(patient, BOOKING, "s-docs-2")
        turn(patient, "yes", "s-docs-2")

        session = fresh()
        try:
            assert (
                session.query(Escalation)
                .filter(Escalation.kind == EscalationKind.SYSTEM_FAILURE)
                .count()
                == 0
            )
        finally:
            session.close()

    def test_the_diff_still_runs(self, patient):
        """The call the budget was firing on. Everything downstream of it —
        the missing-documents task, and the patient being told what to bring —
        was silently not happening."""
        turn(patient, BOOKING, "s-docs-3")
        result = turn(patient, "yes", "s-docs-3")

        session = fresh()
        try:
            names = [
                (event.payload or {}).get("tool")
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.TOOL_RESULT
            ]
        finally:
            session.close()

        assert "diff_required_documents" in names
        assert "record_missing_documents" in names

    def test_only_one_document_is_verified_per_turn(self, patient):
        """The bound itself, at the seam both providers go through. It lived in
        the mock alone, which is the same as not living anywhere."""
        turn(patient, BOOKING, "s-docs-4")
        result = turn(patient, "yes", "s-docs-4")

        session = fresh()
        try:
            verifications = [
                event
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.VALIDATION
                and (event.payload or {}).get("what") == "document_verification"
            ]
        finally:
            session.close()

        assert len(verifications) == 1

    def test_the_rest_are_still_pending_and_get_picked_up(self, patient):
        """One per turn is a pace, not a cap on the work. The seed ships three
        unverified documents and the third is deliberately misfiled — a bound
        that quietly stopped after the first would hide it forever."""
        turn(patient, BOOKING, "s-docs-5")
        turn(patient, "yes", "s-docs-5")
        for index in range(4):
            turn(patient, "what documents do I have on file?", f"s-docs-5-{index}")

        session = fresh()
        try:
            flagged = (
                session.query(PatientDocument)
                .filter(
                    PatientDocument.patient_id == 1,
                    PatientDocument.status == DocumentStatus.FLAGGED,
                )
                .count()
            )
        finally:
            session.close()

        assert flagged == 1


class GreedyCoordinatorLlm(MockLlm):
    """The mock, plus a Coordinator that will not stop asking for slot lists.

    Live, ``gpt-4o-mini`` burned a turn's whole iteration budget inside the
    Coordinator while the run sat at ``pending_confirmation`` holding a slot.
    ``_budget_failure`` then tried to fail a run the table gives no edge from —
    ``pending_confirmation -> failed`` is not a transition — so the refusal
    raised straight through the turn envelope and the patient got an HTTP 500
    where a failure notice was meant to be. The sweep caught it on two separate
    conversations once the document loop stopped absorbing the budget first.
    """

    model: str = "greedy-coordinator-stub"

    def _coordinate(self, llm_request, available, done, text):  # noqa: ANN001
        if "submit_confirmation_verdict" in available:
            # A *different* phrase every time, which is what a live model asking
            # "and what about the week after?" looks like, and what makes this a
            # budget case rather than a repeat case. The accepted-repeat bound
            # ends the loop on identical arguments — so a stub that asked the
            # same thing twice would prove that guard works and say nothing at
            # all about this one.
            asked = len(_calls_to(llm_request, "submit_confirmation_verdict"))
            return function_call_response(
                "submit_confirmation_verdict",
                {
                    "verdict": "slot_question",
                    "reason": "asked about times",
                    "phrase": f"in {asked + 2} weeks",
                },
            )
        return super()._coordinate(llm_request, available, done, text)


class TestABlownBudgetNeverRaisesThroughTheTurn:
    """A failure path that fails is the whole point of a failure path.

    Two shapes, one class: a transition the pinned table does not allow,
    driven by code that assumed the run was somewhere else. Both were live
    500s. Neither edits the table — the table is right, and what was wrong was
    asking it for an edge that should never have been needed.
    """

    def test_a_budget_blown_while_a_proposal_stands_is_a_notice_not_a_crash(
        self, patient, monkeypatch
    ):
        turn(patient, BOOKING, "s-blown-1")
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: GreedyCoordinatorLlm()
        )
        result = turn(patient, "what else have you got", "s-blown-1")

        assert result.budget_exhausted is True
        assert result.reply == FAILED_REPLY

    def test_the_held_proposal_survives_the_bad_turn(self, patient, monkeypatch):
        """Not merely legal — right. The patient had already decided about that
        time; one misbehaving turn must not throw their decision away."""
        turn(patient, BOOKING, "s-blown-2")
        session = fresh()
        try:
            held = (
                session.query(WorkflowRun).order_by(WorkflowRun.id.desc()).first()
            ).proposed_slot_id
        finally:
            session.close()

        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: GreedyCoordinatorLlm()
        )
        turn(patient, "what else have you got", "s-blown-2")

        session = fresh()
        try:
            run = session.query(WorkflowRun).order_by(WorkflowRun.id.desc()).first()
            assert run.status is WorkflowStatus.PENDING_CONFIRMATION
            assert run.proposed_slot_id == held
        finally:
            session.close()

    def test_the_promise_in_the_template_is_still_kept(self, patient, monkeypatch):
        """FAILED_REPLY says "I've flagged it for them". The run not moving
        must not quietly take the queue item with it."""
        turn(patient, BOOKING, "s-blown-3")
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: GreedyCoordinatorLlm()
        )
        turn(patient, "what else have you got", "s-blown-3")

        session = fresh()
        try:
            assert (
                session.query(Escalation)
                .filter(Escalation.kind == EscalationKind.SYSTEM_FAILURE)
                .count()
                == 1
            )
        finally:
            session.close()

    def test_the_trace_says_the_run_was_left_alone(self, patient, monkeypatch):
        """"The run failed" and "a turn failed while the run stood" are
        different facts. Skipping the transition silently would make them look
        identical afterwards."""
        turn(patient, BOOKING, "s-blown-4")
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: GreedyCoordinatorLlm()
        )
        result = turn(patient, "what else have you got", "s-blown-4")

        session = fresh()
        try:
            verdict = _guard(session, result.turn_id, "run_failable")
        finally:
            session.close()

        assert verdict["passed"] is False
        assert verdict["detail"]["status"] == WorkflowStatus.PENDING_CONFIRMATION.value

    def test_an_in_progress_run_still_fails_properly(self, patient, monkeypatch):
        """The falsification. A guard that skipped the transition everywhere
        would pass all of the above while quietly deleting the `failed` state
        from the system — the run would sit `in_progress` forever with a queue
        item beside it and nothing to say it was over."""
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: LoopingLlm()
        )
        result = turn(patient, BOOKING, "s-blown-5")

        assert result.budget_exhausted is True
        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.status is WorkflowStatus.FAILED
        finally:
            session.close()


class QueuedRoutingLlm(MockLlm):
    """A message during a staff review that the Coordinator calls a continuation.

    The plan then carries on, the route step re-runs, routing is ambiguous
    again — and `pending_review -> pending_review` is not an edge. Live: an
    HTTP 500 for the sentence "looks good. lets book that time".
    """

    model: str = "queued-routing-stub"

    def _classify(self, llm_request, available, done, task):  # noqa: ANN001
        return function_call_response(
            "classify_message",
            {"message_class": "continuation", "incoming_steps": []},
        )


class TestRoutingDoesNotReRunOnAQueuedRun:
    AMBIGUOUS = "book an appointment, my kid has ear pain"

    @pytest.fixture(autouse=True)
    def _queued(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: QueuedRoutingLlm()
        )

    def test_a_message_during_a_review_does_not_crash_the_turn(self, patient):
        first = turn(patient, self.AMBIGUOUS, "s-queued-1")
        assert first.status == WorkflowStatus.PENDING_REVIEW.value

        result = turn(patient, "looks good. lets book that time", "s-queued-1")
        assert result.status == WorkflowStatus.PENDING_REVIEW.value

    def test_the_queue_item_is_not_duplicated(self, patient):
        turn(patient, self.AMBIGUOUS, "s-queued-2")
        turn(patient, "looks good. lets book that time", "s-queued-2")

        session = fresh()
        try:
            assert (
                session.query(Escalation)
                .filter(Escalation.kind == EscalationKind.LOW_CONFIDENCE_ROUTING)
                .count()
                == 1
            )
        finally:
            session.close()


class TestClarifyingIntoAnotherDepartment:
    """The live dead end, end to end: three turns, and the third must escape.

    A General Medicine proposal, declined, then "let me clarify — appointment
    for vision issues". Before the fix this test's third turn stayed on run 1
    and re-proposed General Medicine slots — the mock reproduced it exactly, so
    the failure was structural rather than a live model's bad day. What made it
    a *dead end* rather than a wrong answer is that clarifying again did the
    same thing: the model reads a clarification as a continuation, and a
    continuation was fed into a run whose routing step had already closed.
    """

    SESSION = "s-clarify-department"

    def _declined_general_medicine(self, patient):
        first = turn(patient, "I need a general medicine appointment", self.SESSION)
        assert first.status == WorkflowStatus.PENDING_CONFIRMATION.value
        declined = turn(patient, "no", self.SESSION)
        assert declined.status == WorkflowStatus.IN_PROGRESS.value
        return first.run_id

    def test_naming_another_department_re_routes(self, patient):
        original = self._declined_general_medicine(patient)

        result = turn(
            patient, "let me clarify - appointment for vision issues", self.SESSION
        )

        assert result.run_id != original, "the eye request stayed in the GM run"
        session = fresh()
        try:
            assert session.get(WorkflowRun, original).status is WorkflowStatus.CANCELLED
            replacement = session.get(WorkflowRun, result.run_id)
            assert replacement.state["department_name"] == "Ophthalmology"
        finally:
            session.close()

    def test_the_reply_is_about_the_department_the_patient_named(self, patient):
        self._declined_general_medicine(patient)
        result = turn(
            patient, "let me clarify - appointment for vision issues", self.SESSION
        )
        assert "General Medicine" not in result.reply

    def test_a_timing_refinement_keeps_the_run(self, patient):
        """The negative control, and the one that keeps the rule narrow. "Some
        time next week" names no department, so it is what it was classified as
        — a continuation of the request the patient is still making."""
        original = self._declined_general_medicine(patient)

        result = turn(patient, "some time next week please", self.SESSION)

        assert result.run_id == original
        session = fresh()
        try:
            run = session.get(WorkflowRun, original)
            assert run.status is not WorkflowStatus.CANCELLED
            assert run.state["department_name"] == "General Medicine"
        finally:
            session.close()


class TestASelectionAtInProgressReEntersTheProposal:
    """Round 6, item 1 — the state that could not hear an answer.

    A run at ``pending_confirmation`` holds a slot, and "lets book the 4pm one"
    moves the offer. A run at ``in_progress`` holds nothing, and there the same
    sentence had nowhere to land: the Coordinator called it a new request, the
    mapping correctly refused to supersede a run with itself, and the refusal's
    consequence is to answer with *more times*. So the reply asked for a time, a
    time arrived, and the reply asked for a time again.

    Live, run 4: seven consecutive messages — "okay lets book at 3pm then",
    "lets do 3pm and book it", "option 2", "3pm will work for me", "2pm will
    work for me", "confirm 2pm slot", "close the previous request" — each drew
    the identical "Other times that are free... Nothing is booked yet. Tell me a
    time" template. Run 9 died the same way in the same session. A patient
    factually could not complete a booking from that state by any wording.

    The state is reached here by declining a proposal, which is how run 4
    reached it — a booking conflict cleared the proposal. Nothing below books
    anything: the confirmation gate is untouched, and that gate is what makes
    reading a selection this freely the cheap direction.
    """

    def _offered_and_waiting(self, patient, session_id: str) -> int:
        """A run at ``in_progress`` that has already shown the patient times."""
        first = turn(patient, BOOKING, session_id)
        assert first.status == WorkflowStatus.PENDING_CONFIRMATION.value
        declined = asyncio.run(apply_patient_action(patient, "decline", session_id))
        assert declined.status == WorkflowStatus.IN_PROGRESS.value
        return declined.run_id

    def _shortlist_times(self, run_id: int) -> list[str]:
        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            return [
                clock_time(session.get(AppointmentSlot, slot_id).start_time)
                for slot_id in run.state["shortlist_slot_ids"]
            ]
        finally:
            session.close()

    def test_a_named_time_is_held_rather_than_re_listed(self, patient):
        run_id = self._offered_and_waiting(patient, "s-select-1")
        when = self._shortlist_times(run_id)[1]

        result = turn(patient, f"okay lets book at {when.lower()} then", "s-select-1")

        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        assert result.run_id == run_id, "the run was replaced instead of advanced"
        assert "Other times that are free" not in result.reply

    def test_the_slot_held_is_the_one_the_patient_named(self, patient):
        run_id = self._offered_and_waiting(patient, "s-select-2")
        when = self._shortlist_times(run_id)[2]

        turn(patient, f"{when.lower()} will work for me", "s-select-2")

        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            held = session.get(AppointmentSlot, run.proposed_slot_id)
            assert clock_time(held.start_time) == when
        finally:
            session.close()

    def test_a_list_number_is_answerable(self, patient):
        """"option 2" means something only against a list somebody recorded,
        which is why ``render_proposal`` writes the numbering down in the same
        breath as drawing it — ``offered_slot_ids`` is a union over the whole
        run and a union has no second element."""
        run_id = self._offered_and_waiting(patient, "s-select-3")
        expected = self._shortlist_times(run_id)[1]

        result = turn(patient, "option 2", "s-select-3")

        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            held = session.get(AppointmentSlot, run.proposed_slot_id)
            assert clock_time(held.start_time) == expected
        finally:
            session.close()

    def test_the_xpm_slot_phrasing_lands(self, patient):
        """Run 9's last message, on run 9's shape."""
        run_id = self._offered_and_waiting(patient, "s-select-4")
        when = self._shortlist_times(run_id)[0]

        result = turn(patient, f"lets book the {when.lower()} slot", "s-select-4")

        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        assert result.run_id == run_id

    def test_no_model_is_asked_what_a_selection_meant(self, patient):
        """The mechanism, not only the outcome. Matching a number against a
        list this run rendered needs no judgement, and a turn that needs no
        model should not spend one — the same argument the listing questions
        use for running before the Coordinator rather than after it."""
        self._offered_and_waiting(patient, "s-select-5")

        result = turn(patient, "option 1", "s-select-5")

        session = fresh()
        try:
            agents = {
                event.agent_name
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.LLM_REQUEST
            }
        finally:
            session.close()
        assert "coordinator" not in agents
        assert "appointment" not in agents

    def test_the_read_is_traced(self, patient):
        """A selection nobody can see is a selection nobody can review."""
        self._offered_and_waiting(patient, "s-select-6")
        taken = turn(patient, "option 1", "s-select-6")

        session = fresh()
        try:
            assert _guard(session, taken.turn_id, "slot_selection")["passed"] is True
        finally:
            session.close()

    def test_a_question_is_not_a_selection(self, patient):
        """The negative control. "What documents do I need" names no time and
        no position, so it goes where it always went — and if this ever starts
        holding slots, the reader has become a trap."""
        run_id = self._offered_and_waiting(patient, "s-select-7")

        result = turn(patient, "what documents do I need to bring?", "s-select-7")

        assert result.status != WorkflowStatus.PENDING_CONFIRMATION.value
        session = fresh()
        try:
            assert session.get(WorkflowRun, run_id).proposed_slot_id is None
        finally:
            session.close()

    def test_a_withdrawal_is_honoured_rather_than_answered_with_times(self, patient):
        """"close the previous request" was run 4's seventh message, and it drew
        the same availability list as the six before it — a patient asking to be
        let go, answered with more of what they were trying to leave.

        Read exactly, never by containment: the message has to *be* the phrase.
        """
        run_id = self._offered_and_waiting(patient, "s-select-8")

        result = turn(patient, "close the previous request", "s-select-8")

        assert "Other times that are free" not in result.reply
        assert result.status == WorkflowStatus.CANCELLED.value
        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            assert run.status is WorkflowStatus.CANCELLED
            assert run.proposed_slot_id is None
        finally:
            session.close()

    def test_a_cue_inside_a_longer_sentence_does_not_close_the_run(self, patient):
        """The other direction of the same rule, and the expensive one: "I
        changed my mind, I'd like something later" contains a cue and is a
        refinement. Applying it there would destroy a live request and tell the
        patient it was their idea."""
        run_id = self._offered_and_waiting(patient, "s-select-9")

        turn(patient, "I changed my mind, can I have something later", "s-select-9")

        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            assert run.status is not WorkflowStatus.CANCELLED
        finally:
            session.close()

    def test_the_confirmation_gate_is_untouched(self, patient):
        """Nothing here books. A selection holds a time and asks; only an exact
        token or the button commits it."""
        run_id = self._offered_and_waiting(patient, "s-select-10")
        turn(patient, "option 1", "s-select-10")

        session = fresh()
        try:
            assert appointments_for(session, run_id) == []
        finally:
            session.close()


class ChangePlanningLlm(MockLlm):
    """The mock, classifying the way ``gpt-4o-mini`` classified two messages:
    a new request, with steps naming a verb the patient did not.

    It exists because the understudy reproduces neither failure. Asked about
    "lets book the 10am slot" the mock classifies sensibly; asked for the steps
    behind "lets reschedule my Ophthalmology appointment" it returns
    ``[reschedule, follow_up]`` and always has. The live model returned
    ``[route, cancel]`` for the first and ``[route, book, ...]`` for the second,
    and both went into a fresh run unchecked — so a guard falsified only against
    the mock is falsified against a provider that never makes the mistake.
    """

    model: str = "change-planning-stub"

    def _classify(self, llm_request, available, done, text):  # noqa: ANN001
        if latest_tool_result(llm_request, "classify_message") is None:
            return function_call_response(
                "classify_message",
                {"message_class": "conflicting", "incoming_steps": ["route", "cancel"]},
            )
        return super()._classify(llm_request, available, done, text)


class TestASupersedeGetsThePlanCheckAFirstMessageGets:
    """Round 6, items 3 and 4 — the other door into ``create_run``.

    ``_corrected_change_plan`` guards the plan of a run born from a first
    message. A run born from a *supersede* was built straight from the model's
    ``incoming_steps``, and that gap produced the session's two worst runs:

    * run 6 — "lets reschedule my Ophthalmology appointment", arriving over a
      live run, became a run whose plan said ``book``. The Appointment agent
      then correctly proposed a *reschedule* under that ``book`` step, and the
      mismatch is what let the step re-enter after the commit;
    * run 10 — "lets book the 10am slot" became a ``[route, cancel]`` run that
      routed to General Medicine with low confidence and died in a staff queue.

    A correction that runs on one door and not the other is not a correction; it
    is a coin flip on which door the message came through.
    """

    @pytest.fixture(autouse=True)
    def _misplan(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: ChangePlanningLlm()
        )

    def test_a_stated_reschedule_supersedes_into_a_reschedule_plan(self, patient):
        turn(patient, BOOKING, "s-supersede-1")
        result = turn(
            patient, "lets reschedule my Ophthalmology appointment", "s-supersede-1"
        )

        session = fresh()
        try:
            plan = session.get(WorkflowRun, result.run_id).plan
        finally:
            session.close()
        assert "reschedule" in plan
        assert "book" not in plan

    def test_the_replacement_run_never_routes_a_change(self, patient):
        """Item 2's rule, reaching the run item 3 creates."""
        turn(patient, BOOKING, "s-supersede-2")
        result = turn(
            patient, "lets reschedule my Ophthalmology appointment", "s-supersede-2"
        )

        session = fresh()
        try:
            assert "route" not in session.get(WorkflowRun, result.run_id).plan
        finally:
            session.close()

    def test_a_booking_selection_is_not_superseded_into_a_cancel(self, patient):
        """Run 9 to run 10, on run 9's shape: a routed run at ``in_progress``
        that has shown times, and a message naming one of them. It never reaches
        classification — which is the point, because the class that came back
        was the wrong one and nothing downstream could tell."""
        first = turn(patient, BOOKING, "s-supersede-3")
        declined = asyncio.run(apply_patient_action(patient, "decline", "s-supersede-3"))
        run_id = declined.run_id
        assert first.run_id == run_id

        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            slot_id = run.state["shortlist_slot_ids"][0]
            when = clock_time(session.get(AppointmentSlot, slot_id).start_time)
        finally:
            session.close()

        result = turn(patient, f"lets book the {when.lower()} slot", "s-supersede-3")

        assert result.run_id == run_id, "a selection replaced the run it was answering"
        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            assert run.status is WorkflowStatus.PENDING_CONFIRMATION
            assert "cancel" not in (run.plan or [])
        finally:
            session.close()


class BookPlanReschedulingLlm(MockLlm):
    """Proposes a *reschedule* while the plan step is ``book``.

    Run 6's actual shape, and reachable only with a provider that reads the
    request text under a step that does not match it. The mock cannot: its
    Appointment agent dispatches on ``task["step"]``, so a ``book`` step always
    books. The live model dispatches on the sentence, and the appointment
    toolset hands it ``propose_reschedule`` whatever the step says — which is
    how a booking plan came to hold a reschedule proposal.

    ``committed`` is stripped from the task for the same reason
    :class:`ReproposingLlm` strips it: a hint is advisory, the live model
    ignored it, and a stub that honours it cannot reproduce a defect that
    depends on it being ignored.
    """

    model: str = "book-plan-rescheduling-stub"

    def _appointment(self, llm_request, done, task):  # noqa: ANN001
        if "list_my_appointments" not in done:
            return function_call_response("list_my_appointments", {})
        listed = latest_tool_result(llm_request, "list_my_appointments")
        appointments = (listed.payload.get("appointments") if listed else None) or []
        if not appointments:
            return super()._appointment(llm_request, done, task)
        return self._change_appointment(
            llm_request,
            done,
            {
                **{key: value for key, value in task.items() if key != "committed"},
                "appointments": appointments,
            },
            "reschedule",
        )


class TestACommitSettlesTheStepWhateverThePlanCallsIt:
    """Round 6, item 3 — a run whose plan verb and committed verb disagree.

    Run 6's trace, in order: ``reschedule_appointment`` committed at seq 5; the
    run transitioned ``pending_confirmation -> in_progress`` at seq 7, so the
    transition was never the problem; and then the ``book`` step, still
    incomplete and not matching ``committed_action == "reschedule"``, was
    dispatched again. It proposed a second time at seq 22 and the run went back
    to ``pending_confirmation`` at seq 24. The receipt for the commit that *had*
    happened was rendered beside a fresh proposal card.

    The Decline pressed next was therefore applicable — there really was an open
    proposal — and answered "That's fine, nothing has been booked" about an
    appointment that had already moved.

    One run commits at most one appointment action, because ``validate_plan``
    refuses a plan naming two. So a committed verb settles this run's single
    appointment step whatever the plan happens to call it.
    """

    @pytest.fixture(autouse=True)
    def _misverb(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: BookPlanReschedulingLlm()
        )

    def _proposed_under_a_book_plan(self, patient, session_id: str):
        """An ordinary booking request, answered with a reschedule proposal.

        No supersede is involved: the plan is the one a first message produces,
        ``names_change_verb`` finds no change verb to correct it with, and the
        specialist proposes the other verb anyway. That is the residue the
        supersede fix cannot reach, and it is what this guard is for."""
        result = turn(patient, BOOKING, session_id)
        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert "book" in run.plan and "reschedule" not in run.plan
            assert run.proposed_action is ProposedAction.RESCHEDULE
        finally:
            session.close()
        return result

    def test_a_chat_yes_leaves_no_open_proposal(self, patient):
        proposed = self._proposed_under_a_book_plan(patient, "s-mismatch-1")

        result = turn(patient, "yes", "s-mismatch-1")

        assert result.status != WorkflowStatus.PENDING_CONFIRMATION.value
        session = fresh()
        try:
            run = session.get(WorkflowRun, proposed.run_id)
            assert run.proposed_action is None
            assert run.proposed_slot_id is None
        finally:
            session.close()

    def test_the_commit_is_not_followed_by_a_second_proposal(self, patient):
        self._proposed_under_a_book_plan(patient, "s-mismatch-2")

        turn(patient, "yes", "s-mismatch-2")

        session = fresh()
        try:
            assert _audit_count(session, "appointment_rescheduled") == 1
        finally:
            session.close()

    def test_a_decline_afterwards_does_not_claim_nothing_happened(self, patient):
        """The sentence the patient was actually shown. "Nothing has been
        booked" was false — the appointment had moved — and it was reachable
        only because the run had been put back up for confirmation."""
        self._proposed_under_a_book_plan(patient, "s-mismatch-3")
        turn(patient, "yes", "s-mismatch-3")

        declined = asyncio.run(apply_patient_action(patient, "decline", "s-mismatch-3"))

        assert "nothing has been booked" not in declined.reply.lower()
        assert "nothing waiting for your confirmation" in declined.reply.lower()

    def test_the_step_settles_from_the_commit_and_says_so(self, patient):
        """Verify-by-revert's target: the validation row naming both verbs. If
        the guard ever narrows back to an exact match this is what disappears,
        and the assertions above would go on passing for a while first."""
        self._proposed_under_a_book_plan(patient, "s-mismatch-4")

        result = turn(patient, "yes", "s-mismatch-4")

        session = fresh()
        try:
            settled = _validations(session, result.turn_id, "step_already_committed")
            assert len(settled) == 1
            detail = settled[0].payload["detail"]
        finally:
            session.close()
        assert detail["committed"] == "reschedule"
        assert detail["step"] != detail["committed"], (
            "this test is only meaningful while the plan and the commit disagree"
        )
