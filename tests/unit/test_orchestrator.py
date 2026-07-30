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
from datetime import timedelta
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
    Department,
    DocumentStatus,
    Doctor,
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
from app.agents.toolbelt import Toolbelt
from app.orchestrator import (
    AWAITING_REVIEW_REPLY,
    FAILED_REPLY,
    NO_PLAN_REPLY,
    SCOPE_REPLY,
    UNSUPPORTED_TOPIC_REPLY,
    active_run,
    apply_patient_action,
    run_workflow,
)
from app.workflow.plan import PlanStep
from app.workflow.replies import clock_time
from app.providers.mock import MockLlm
from app.providers.base import (
    AgentCareLlm,
    available_tool_names,
    called_tools,
    current_turn_start,
    function_call_response,
    latest_tool_result,
    text_response,
    tool_results,
)
from app.trace import TraceWriter, assert_well_formed

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


def _appointment_count(patient_id: int) -> int:
    """Live appointments for one patient, baselined per patient because the
    seed ships one already."""
    session = fresh()
    try:
        return (
            session.query(Appointment)
            .filter(Appointment.patient_id == patient_id)
            .count()
        )
    finally:
        session.close()


def _guard_or_none(session, turn_id: str, name: str) -> dict | None:
    """The same, for a guard whose *absence* is the claim."""
    try:
        return _guard(session, turn_id, name)
    except AssertionError:
        return None


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


class SearchLoopingLlm(MockLlm):
    """A Coordinator that answers a live run with ``list_other_slots``, forever.

    Live run #6, reduced. At an ``in_progress`` **cancel** run — one that had
    asked which of two appointments the patient meant and not been told — "book
    me a cardiology appointment" produced eight ``list_other_slots("next week")``
    calls. The same argument every time, and the same refusal every time,
    because a cancel run's department lives on the appointment nobody had picked
    yet. The iteration budget fired and ``_fail_run`` tombstoned the run.

    It keeps the mock's planning and safety screen, so the loop under test is the
    classifier's — which is where it happened.
    """

    model: str = "search-looping-stub"

    def _classify(self, llm_request, available, done, text):  # noqa: ANN001
        if "list_other_slots" in available:
            return function_call_response("list_other_slots", {"phrase": "next week"})
        return super()._classify(llm_request, available, done, text)


class WideningSearchLlm(SearchLoopingLlm):
    """The same loop, correcting its argument every call.

    The control for the identical-refusal bound. A model that *changes* its
    argument is doing what the retry ladder is for — every proposal tool in the
    system depends on being told no and trying again — so this one must run all
    the way to the iteration budget exactly as it did before.
    """

    model: str = "widening-search-stub"

    def _classify(self, llm_request, available, done, text):  # noqa: ANN001
        if "list_other_slots" in available:
            tried = len(
                [
                    result
                    for result in tool_results(llm_request)
                    if result.name == "list_other_slots"
                ]
            )
            return function_call_response(
                "list_other_slots", {"phrase": "week %d" % (tried + 1)}
            )
        return super(SearchLoopingLlm, self)._classify(
            llm_request, available, done, text
        )


def _waiting_cancel_run(patient, session_id: str):
    """A cancel run at ``in_progress``, still waiting to learn which appointment.

    Two live appointments is what makes it that: the specialist has nothing to
    propose, so the run asks and stays put — and the run therefore carries no
    department, which is why the search refuses. One appointment auto-targets and
    the run would be holding a proposal instead, which is a different state with
    a different toolbelt.

    Every turn here is readable by the looping stub without reaching its loop:
    planning happens with no active run, and the confirmation is read in code
    before any model call.
    """
    booked = turn(patient, BOOKING, session_id)
    assert booked.status == WorkflowStatus.PENDING_CONFIRMATION.value
    confirmed = turn(patient, "yes", session_id)
    assert confirmed.status == WorkflowStatus.COMPLETED.value

    started = turn(patient, "please cancel my appointment", session_id)
    assert started.status == WorkflowStatus.IN_PROGRESS.value
    return started


class TestAnIdenticalRefusalEndsTheLoop:
    """Round 11 item 2a — the refusal twin of the accepted-repeat bound.

    An accepted decision is settled, and asking again is waste. A refusal is
    narrower and just as final: it is a fact about *these arguments*, so the same
    call returns the same dict, and eight of them is one call's worth of
    information at eight calls' cost.
    """

    @pytest.fixture(autouse=True)
    def _loop(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: SearchLoopingLlm()
        )

    def test_the_search_runs_once_and_the_repeat_is_refused(self, patient, settings):
        """One call runs; the second is refused before it runs and ends the loop.

        The cap is eight, so a passing count of one could not have come from the
        budget — which is the thing this has to be distinguishable from.
        """
        _waiting_cancel_run(patient, "s-refusal-1")
        result = turn(patient, "book me a cardiology appointment", "s-refusal-1")

        session = fresh()
        try:
            searches = [
                event
                for event in _tool_calls(session, result.turn_id)
                if event.payload["tool"] == "list_other_slots"
            ]
            refusals = _validations(session, result.turn_id, "repeated_refusal")
        finally:
            session.close()

        assert len(searches) == 1
        assert refusals, "the bound must be in the trace, not merely happen"
        assert settings.max_tool_iterations > 2

    def test_the_budget_never_blows(self, patient):
        """Asserted on the trace, not on ``budget_exhausted``.

        Sabotage is how that distinction was found: with this bound removed the
        loop reaches the cap, item 2b's recovery supersedes anyway, and the
        replacement run's ``TurnResult`` reports ``budget_exhausted=False``. The
        flag was describing the second half of the turn. The absence of the
        exhaustion event is the claim this test is actually making.
        """
        _waiting_cancel_run(patient, "s-refusal-2")
        result = turn(patient, "book me a cardiology appointment", "s-refusal-2")

        session = fresh()
        try:
            exhausted = _validations(
                session, result.turn_id, "tool_iteration_budget"
            )
        finally:
            session.close()

        assert exhausted == []


class TestACorrectedRetryIsStillAllowed:
    """The direction that would break the retry ladder rather than the loop.

    Every proposal tool in this system is built on being refused and called again
    with a better argument. The bound is keyed on the arguments for exactly that
    reason, and this is what keeps the key honest: a widening search must still be
    able to spend its whole budget.
    """

    @pytest.fixture(autouse=True)
    def _widen(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: WideningSearchLlm()
        )

    def test_a_differing_argument_is_never_refused_as_a_repeat(self, patient):
        """The ladder runs all the way to the outer bound, refusing nothing.

        Counted as "the budget is what stopped it" rather than as an exact number
        of searches: the Coordinator legitimately spends one call loading the
        patient's context first, and an assertion on the total would be pinning
        that incidental call rather than this one.
        """
        _waiting_cancel_run(patient, "s-widen-1")
        result = turn(patient, "the earliest the better", "s-widen-1")

        session = fresh()
        try:
            searches = [
                event
                for event in _tool_calls(session, result.turn_id)
                if event.payload["tool"] == "list_other_slots"
            ]
            refusals = _validations(session, result.turn_id, "repeated_refusal")
            exhausted = _validations(
                session, result.turn_id, "tool_iteration_budget"
            )
        finally:
            session.close()

        assert refusals == []
        assert exhausted, "the outer bound is what must have stopped this"
        assert len(searches) > 2, "and not the identical-refusal bound"

    def test_an_unreadable_message_still_fails_as_a_last_resort(self, patient):
        """The negative control for item 2b, and the reason ``_fail_run`` stays.

        "The earliest the better" names no verb code can act on, so there is
        nothing for the recovery to read and the honest answer is the failure
        notice — with the queue item ``FAILED_REPLY`` promises behind it.
        """
        _waiting_cancel_run(patient, "s-widen-2")
        result = turn(patient, "the earliest the better", "s-widen-2")

        assert result.reply == FAILED_REPLY
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

    def test_a_genuinely_blown_budget_still_reaches_a_proposal(self, patient):
        """The one path ``release_budget`` is reachable on, and its whole point.

        The identical-refusal bound means an identical loop no longer blows the
        budget at all — so this is the shape that still can: a loop that corrects
        its argument every call, on a turn whose message code *can* read. Without
        the release every specialist dispatched after it short-circuits to "I
        couldn't complete this request", and the recovery is a cancelled run with
        an apology where the proposal should be.
        """
        _waiting_cancel_run(patient, "s-widen-3")
        result = turn(patient, "book me a cardiology appointment", "s-widen-3")

        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        session = fresh()
        try:
            exhausted = _validations(
                session, result.turn_id, "tool_iteration_budget"
            )
            released = _validations(session, result.turn_id, "budget_released")
        finally:
            session.close()

        assert exhausted, "the budget must actually have blown for this to mean it"
        assert released


class TestABlownLoopIsNotAVerdict:
    """Round 11 item 2b — the clearest sentence in the transcript killed a run.

    "Book me a cardiology appointment", sent to an ``in_progress`` cancel run,
    was never classified: the loop spent the budget, ``_fail_run`` transitioned
    the run to ``failed``, a ``system_failure`` escalation was raised against
    work nobody had failed at, and the patient got the apology template. A verb,
    a department, and nothing ambiguous about it.

    Whether the class is missing because the budget blew, because the
    identical-refusal bound ended the loop, or because the model never called the
    tool makes no difference to the message — so code reads it with the two
    readers that answer this question everywhere else.
    """

    @pytest.fixture(autouse=True)
    def _loop(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: SearchLoopingLlm()
        )

    def test_a_different_verb_supersedes_instead_of_failing(self, patient):
        started = _waiting_cancel_run(patient, "s-unclassified-1")
        result = turn(patient, "book me a cardiology appointment", "s-unclassified-1")

        assert result.run_id != started.run_id
        session = fresh()
        try:
            old = session.get(WorkflowRun, started.run_id)
            new = session.get(WorkflowRun, result.run_id)
            assert old.status is WorkflowStatus.CANCELLED
            assert new.plan[:2] == ["route", "book"]
            assert (
                session.query(Escalation)
                .filter(Escalation.kind == EscalationKind.SYSTEM_FAILURE)
                .count()
                == 0
            )
        finally:
            session.close()

    def test_the_replacement_reaches_a_proposal(self, patient):
        """The point of releasing the budget. Without it every specialist
        dispatched here short-circuits to "I couldn't complete this request", and
        the supersede leaves a cancelled run with nothing in its place."""
        _waiting_cancel_run(patient, "s-unclassified-2")
        result = turn(patient, "book me a cardiology appointment", "s-unclassified-2")

        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.proposed_slot_id is not None
            assert run.state["department_name"] == "Cardiology"
        finally:
            session.close()

    def test_the_decision_is_traced(self, patient):
        _waiting_cancel_run(patient, "s-unclassified-3")
        result = turn(patient, "book me a cardiology appointment", "s-unclassified-3")

        session = fresh()
        try:
            verdict = _guard(session, result.turn_id, "unclassified_turn")
        finally:
            session.close()

        assert verdict["detail"]["applied"] == "supersede"
        assert verdict["detail"]["verb"] == "book"
        assert verdict["detail"]["run_intent"] == "cancel"

    def test_the_same_verb_is_a_refinement_and_the_run_stands(self, patient):
        """Difference is what earns a supersede; sameness earns an answer.

        A second cancellation request against a cancel run replaces nothing, so
        the run survives — and it is still not failed, which is the half of the
        item that is about ``_fail_run`` rather than about superseding. What it
        gets back is the question it was already asking.
        """
        started = _waiting_cancel_run(patient, "s-unclassified-4")
        result = turn(patient, "please cancel my appointment", "s-unclassified-4")

        assert result.run_id == started.run_id
        assert result.reply != FAILED_REPLY
        assert "1." in result.reply
        session = fresh()
        try:
            run = session.get(WorkflowRun, started.run_id)
            assert run.status is WorkflowStatus.IN_PROGRESS
        finally:
            session.close()

    def test_the_trace_is_still_well_formed(self, patient):
        _waiting_cancel_run(patient, "s-unclassified-5")
        turn(patient, "book me a cardiology appointment", "s-unclassified-5")

        session = fresh()
        try:
            assert_well_formed(session)
        finally:
            session.close()


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


class FixatedCoordinatorLlm(MockLlm):
    """A Coordinator stuck on a word from the wrong vocabulary.

    Live: "I wanted to book an appointment for knee pain" reached
    ``submit_plan`` as ``["conflicting"]`` four times — a *classifier* class,
    not a plan step, refused each time by the closed enum — then once as
    ``["cancel", "book"]``, refused by the one-verb rule, and finally as prose
    the scope gate discarded. The patient rephrased and got the identical
    fixation. Nothing was wrong with any guard; the run simply never started.

    Three attempts then prose, because that is the shape of the live turn.
    Submitting until the iteration budget fires would test the budget path
    instead, which already has its own tests and produces a different reply.
    """

    model: str = "fixated-coordinator-stub"

    def _plan(self, llm_request, done, text):  # noqa: ANN001
        submitted = latest_tool_result(llm_request, "submit_plan")
        if submitted is not None and submitted.payload.get("accepted"):
            return self._from_plan(submitted.payload)
        attempts = sum(
            1 for result in tool_results(llm_request) if result.name == "submit_plan"
        )
        if attempts < 3:
            return function_call_response("submit_plan", {"steps": ["conflicting"]})
        return text_response(
            "Your knee pain request conflicts with your existing appointment."
        )


class HistoryFixatedCoordinatorLlm(FixatedCoordinatorLlm):
    """The same fixation, but *caused* by the transcript — which is the claim.

    ``FixatedCoordinatorLlm`` pins the floor: it misplans no matter what it can
    see, so it says nothing about why the live model misplanned. This one
    fixates only while it can see a turn older than the current one, and plans
    perfectly well when it cannot. That is the difference between a stub that
    reproduces the symptom and one that reproduces the *cause*, and only the
    second can falsify a bound on context.
    """

    model: str = "history-fixated-coordinator-stub"

    def _plan(self, llm_request, done, text):  # noqa: ANN001
        contents = llm_request.contents or []
        if current_turn_start(llm_request) > 0 and len(contents) > 1:
            return super()._plan(llm_request, done, text)
        return MockLlm._plan(self, llm_request, done, text)


class TestAFreshTurnPlansOnItsOwnMessage:
    """Round 9, item 2a — the planner reasons over one message, not a session.

    The live Coordinator's context began at a *previous run's* first message,
    so it was still reasoning about a conversation in which "conflicting" had
    been a valid answer. A fresh turn has nothing to learn from that: whether
    this message needs routing and a booking is a fact about this message and
    the patient's own record, both of which it is handed.

    Mid-run turns are untouched. There the Coordinator is a classifier, its
    whole question is how the new message relates to what came before, and
    ``TestTheMidRunWindowSurvives`` is what holds that.
    """

    @pytest.fixture(autouse=True)
    def _fixated(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider",
            lambda name=None: HistoryFixatedCoordinatorLlm(),
        )

    def _with_history(self, patient, session_id: str):
        """Two turns of unrelated conversation, so a transcript exists."""
        turn(patient, "what's the weather like?", session_id)
        turn(patient, "who won the fifa final", session_id)

    def test_a_planner_that_needs_history_to_fail_now_succeeds(self, patient):
        self._with_history(patient, "s-bounded-1")

        result = turn(
            patient, "I wanted to book an appointment for knee pain", "s-bounded-1"
        )

        assert result.run_id is not None
        assert result.reply != NO_PLAN_REPLY

    def test_the_run_is_routed_by_routing(self, patient):
        """Orthopedics has to come from the Department table, through the
        routing step — not from anything the floor guessed."""
        self._with_history(patient, "s-bounded-2")

        result = turn(
            patient, "I wanted to book an appointment for knee pain", "s-bounded-2"
        )

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            department = session.get(Department, run.state.get("department_id"))
            assert department.name == "Orthopedics"
        finally:
            session.close()

    def test_the_coordinators_request_carries_only_this_turn(self, patient):
        """The bound itself, read off the trace rather than inferred from the
        reply — the recorded request is what was actually sent."""
        self._with_history(patient, "s-bounded-3")

        result = turn(
            patient, "I wanted to book an appointment for knee pain", "s-bounded-3"
        )

        session = fresh()
        try:
            requests = [
                event.payload
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .order_by(TraceEvent.seq)
                .all()
                if event.event_type is TraceEventType.LLM_REQUEST
                and event.agent_name == "coordinator"
            ]
        finally:
            session.close()

        assert requests, "the coordinator made no recorded request"
        first = json.dumps(requests[0])
        assert "weather" not in first
        assert "fifa" not in first


class TestTheFloorUnderAFailedPlan:
    """Round 9, item 2b — when nothing is accepted, the words still said it.

    Bounding the context (2a) removes the *cause* observed live. It cannot
    promise the model will plan: a fresh session can misplan on its own, and
    then the patient meets the clarify, rephrases, and meets it again. So where
    the message plainly states a verb and a subject this system administers,
    code supplies that verb's canonical plan rather than asking a question
    whose answer is already in the sentence.

    Narrow on purpose, and it is the second carve-out of the same shape as
    round 5's ``_corrected_change_plan`` — which already covers the change
    verbs for a patient who has something to change. What it adds is *book*,
    which closes over nothing that exists yet and so had no equivalent. Every
    downstream guard runs unchanged: the plan goes through ``validate_plan``
    like any other, and routing still decides the department from the table.
    """

    @pytest.fixture(autouse=True)
    def _fixated(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: FixatedCoordinatorLlm()
        )

    def test_a_stated_booking_survives_a_coordinator_that_never_plans(self, patient):
        result = turn(
            patient, "I wanted to book an appointment for knee pain", "s-floor-1"
        )

        assert result.run_id is not None
        assert result.reply != NO_PLAN_REPLY

    def test_the_department_still_comes_from_routing(self, patient):
        result = turn(
            patient, "I wanted to book an appointment for knee pain", "s-floor-2"
        )

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            department = session.get(Department, run.state.get("department_id"))
            assert department.name == "Orthopedics"
            assert run.plan[0] == PlanStep.ROUTE.value
        finally:
            session.close()

    def test_the_floor_is_recorded_as_a_guard_verdict(self, patient):
        """Code supplying a plan is exactly the kind of thing that must not be
        invisible: the trace has to say the plan was not the model's."""
        result = turn(
            patient, "I wanted to book an appointment for knee pain", "s-floor-3"
        )

        session = fresh()
        try:
            floor = _guard(session, result.turn_id, "plan_floor")
        finally:
            session.close()

        assert floor["detail"]["verb"] == "book"

    def test_a_vague_message_still_gets_the_clarify(self, patient):
        """The negative control, and the thing that keeps the floor narrow.

        The message names the subject — so the veto applies and the answer is
        "tell me more" — but it states no verb, so there is nothing for code to
        supply and asking is the honest answer.
        """
        result = turn(patient, "my appointment situation is confusing", "s-floor-4")

        assert result.reply == NO_PLAN_REPLY
        assert result.run_id is None

    def test_an_off_topic_message_is_not_dragged_in(self, patient):
        """Naming no subject at all is still a refusal, not a booking."""
        result = turn(patient, "who won the fifa final", "s-floor-6")

        assert result.reply == SCOPE_REPLY
        assert result.run_id is None


class TestTheClarifyLoopIsBounded:
    """Round 9, item 2b's other half — a question asked forever is not an answer.

    The clarify is a good reply once. Given to a rephrase of the message that
    produced it, it is a loop, and the live session shows what that costs: the
    patient rephrased, met the identical sentence, and stopped. Nothing about
    the turn changes between iterations, which is what makes it a loop rather
    than an unlucky answer — so it is bounded like every other automated writer
    here, and the bound hands over to a person rather than trying again.

    Deliberately *not* applied to the off-topic refusal: there is nothing for a
    human to review about the FIFA final, and a queue of that is a queue nobody
    reads. This fires only where the patient keeps naming something this system
    genuinely administers and the planner keeps failing to plan it.
    """

    @pytest.fixture(autouse=True)
    def _fixated(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: FixatedCoordinatorLlm()
        )

    VAGUE = "my appointment situation is confusing"

    def test_the_third_vague_message_reaches_a_human(self, patient):
        assert turn(patient, self.VAGUE, "s-clarify-1").reply == NO_PLAN_REPLY
        assert turn(patient, self.VAGUE, "s-clarify-1").reply == NO_PLAN_REPLY

        third = turn(patient, self.VAGUE, "s-clarify-1")

        assert third.reply != NO_PLAN_REPLY
        assert third.status == WorkflowStatus.ESCALATED.value

    def test_the_escalation_is_a_queue_item_a_human_can_act_on(self, patient):
        for _ in range(3):
            turn(patient, self.VAGUE, "s-clarify-2")

        session = fresh()
        try:
            escalations = session.query(Escalation).all()
            assert len(escalations) == 1
            assert escalations[0].kind is EscalationKind.UNSUPPORTED_REQUEST
            assert escalations[0].status is EscalationStatus.OPEN
        finally:
            session.close()

    def test_a_turn_that_did_something_resets_the_count(self, patient):
        """Consecutive means consecutive. A patient who gets stuck twice, is
        answered by something else, and is puzzled again later must not be
        escalated by a tally left over from before.

        The turn in between is an off-topic one because it leaves no run: a
        booking would, and the next message would then be a *mid-run* turn
        answering a held proposal, which never reaches this branch at all and
        would prove nothing about the counter.
        """
        turn(patient, self.VAGUE, "s-clarify-3")
        turn(patient, self.VAGUE, "s-clarify-3")
        turn(patient, "who won the fifa final", "s-clarify-3")

        result = turn(patient, self.VAGUE, "s-clarify-3")

        assert result.reply == NO_PLAN_REPLY

    def test_an_off_topic_message_is_never_escalated(self, patient):
        """The direction that would make the queue useless."""
        for index in range(3):
            result = turn(patient, "who won the fifa final", "s-clarify-4")
            assert result.reply == SCOPE_REPLY

        session = fresh()
        try:
            assert session.query(Escalation).count() == 0
        finally:
            session.close()

    def test_an_accepted_plan_is_never_second_guessed(self, patient, monkeypatch):
        """The floor may only fill a hole. A Coordinator that plans properly
        must reach the same outcome it always did, so the guard has to be
        unreachable whenever a plan was accepted."""
        monkeypatch.setattr("app.agents.base.get_provider", lambda name=None: MockLlm())

        result = turn(patient, "I need a dermatology appointment next week", "s-floor-5")

        session = fresh()
        try:
            assert _guard_or_none(session, result.turn_id, "plan_floor") is None
        finally:
            session.close()


class FollowUpPlanningLlm(MockLlm):
    """A Coordinator that answers anything at all with ``["follow_up"]``.

    Live, runs #10 to #12: "what is the capital city of India?", "who is the
    ceo of nvidia?" and "tell me about google stock" each produced a plan of
    ``["follow_up"]``. It is a real step, so the plan guard accepted it; the
    scope gate only asked for a domain subject when the plan was *empty*, so it
    passed; and the follow-up agent dutifully listed the patient's reminders.
    Three completed runs of noise.

    Containment held — the agent's only tools are reminders and tasks, so the
    question was never answered — which is exactly why this was invisible. The
    reply was wrong, not leaky.
    """

    model: str = "followup-planning-stub"

    def _plan(self, llm_request, done, text):  # noqa: ANN001
        submitted = latest_tool_result(llm_request, "submit_plan")
        if submitted is not None:
            return self._from_plan(submitted.payload)
        return function_call_response("submit_plan", {"steps": ["follow_up"]})


class TestAFollowUpPlanMustBeEarned:
    """Round 9, item 3 — ``follow_up`` was a free pass through the scope gate.

    Every other step names something the message has to be about. ``follow_up``
    names the one specialist that can run on any patient at any time, so a plan
    containing only it asserts nothing about the request — and the gate, which
    checks for a domain subject only when there is no plan at all, had nothing
    to disagree with.

    So the message has to earn it: name something this system administers, or
    be a listing question. Otherwise the plan is refused, no run is created,
    and nothing goes to staff — there is nothing for a human to review about
    the CEO of nvidia.
    """

    @pytest.fixture(autouse=True)
    def _followup_planner(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: FollowUpPlanningLlm()
        )

    OFF_TOPIC = (
        "what is the capital city of India?",
        "who is the ceo of nvidia?",
        "tell me about google stock",
    )

    def test_an_off_topic_message_spawns_no_run(self, patient):
        for index, message in enumerate(self.OFF_TOPIC):
            result = turn(patient, message, f"s-earn-{index}")
            assert result.run_id is None, message

        session = fresh()
        try:
            assert session.query(WorkflowRun).count() == 0
            assert session.query(Escalation).count() == 0
        finally:
            session.close()

    def test_the_reply_says_so_plainly(self, patient):
        result = turn(patient, self.OFF_TOPIC[0], "s-earn-reply")

        assert result.reply == UNSUPPORTED_TOPIC_REPLY
        assert result.author is TraceAuthor.GUARD

    def test_the_refusal_is_recorded(self, patient):
        result = turn(patient, self.OFF_TOPIC[0], "s-earn-trace")

        session = fresh()
        try:
            gate = _guard(session, result.turn_id, "plan_earned")
        finally:
            session.close()

        assert gate["passed"] is False

    def test_a_question_about_the_patients_own_tasks_is_answered(self, patient):
        """Run #14's exact wording, and the reason the gate cannot simply reuse
        ``mentions_domain_subject``: "task" is in neither that vocabulary nor
        the listing detector's, so a gate built from those alone would refuse
        the one follow-up question the session got right."""
        result = turn(patient, "do I have any pending tasks?", "s-earn-tasks")

        assert result.reply != UNSUPPORTED_TOPIC_REPLY
        assert result.run_id is not None

    def test_a_reminders_question_never_reaches_this_gate_at_all(self, patient):
        """Named for what it actually pins, which sabotage is how I found out.

        With ``_earns_plan`` forced to refuse everything, this still
        passed: a listing question is answered from the rows *before* the
        Coordinator runs, so the gate never sees it. That makes this a
        regression guard on the query router's precedence rather than on the
        gate — worth keeping, worth not mistaking for the other thing.
        """
        result = turn(patient, "what reminders do I have?", "s-earn-reminders")

        assert result.reply != UNSUPPORTED_TOPIC_REPLY
        session = fresh()
        try:
            assert _guard_or_none(session, result.turn_id, "plan_earned") is None
        finally:
            session.close()

    def test_a_follow_up_step_inside_a_larger_plan_is_untouched(
        self, patient, monkeypatch
    ):
        """The gate is about a plan that is *only* follow-up. Every booking
        plan ends with one, and closing over that would stop the system
        working entirely."""
        monkeypatch.setattr("app.agents.base.get_provider", lambda name=None: MockLlm())

        result = turn(patient, BOOKING, "s-earn-booking")

        assert result.run_id is not None
        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert PlanStep.FOLLOW_UP.value in run.plan
        finally:
            session.close()


class SubjectlessPlanningLlm(MockLlm):
    """A Coordinator that answers anything at all with ``["documents", "follow_up"]``.

    Live run #5, and the reason round 9's fix was one door wide. "Now do the
    other thing I asked" came back as exactly this plan — two real steps, so the
    plan validator accepted it, and the gate asked for a domain subject only
    when the plan was ``["follow_up"]`` *alone*. So the run was created, the
    documents agent ran with no department to diff against and shipped its own
    internal refusal ("no department has been decided") to the patient as an
    answer, and the follow-up agent dumped their open tasks underneath it. The
    recall memory that exists for that exact sentence never fired, because it is
    consulted only when the gate refuses.

    The mock cannot show this: it produces no plan for that sentence and lands
    on the offer already. A stub that plans *around* the gate is the only thing
    that can falsify a claim about the gate.
    """

    model: str = "subjectless-planning-stub"

    def _plan(self, llm_request, done, text):  # noqa: ANN001
        submitted = latest_tool_result(llm_request, "submit_plan")
        if submitted is not None:
            return self._from_plan(submitted.payload)
        return function_call_response(
            "submit_plan", {"steps": ["documents", "follow_up"]}
        )


class TestASubjectlessPlanMustBeEarned:
    """Round 11 item 1 — the earned rule is about the *plan*, not about one step.

    A plan made only of ``documents`` and ``follow_up`` names no appointment
    verb, so nothing in it says what the message was about. Round 9 read that
    rule off ``follow_up`` and wrote the condition for ``follow_up``; the model
    then planned one step over.
    """

    @staticmethod
    def subjectless(monkeypatch):
        """Swap the provider *after* any setup the stub could not have planned.

        Round 10's decline tests learned this the hard way as a fixture: a stub
        that plans one thing for every sentence cannot also perform the setup,
        and eleven tests failed on a run that was never created.
        """
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: SubjectlessPlanningLlm()
        )

    def test_a_pointing_message_reaches_the_offer_instead_of_a_run(
        self, patient, monkeypatch
    ):
        """The live sentence, with the memory the live session also had.

        This is the whole item: the refusal is what consults the memory, so a
        plan that walks past the refusal walks past the memory too.
        """
        first = turn(
            patient,
            "okay lets cancel that appointment and book a new one for cardiology",
            "s-subjectless-1",
        )
        assert "One change at a time" in first.reply
        done = turn(patient, "yes", "s-subjectless-1")
        assert done.status == WorkflowStatus.COMPLETED.value

        self.subjectless(monkeypatch)
        result = turn(patient, "now do the other thing I asked", "s-subjectless-1")

        assert "booking a Cardiology appointment" in result.reply
        assert result.run_id is None

        session = fresh()
        try:
            patient_id = session.get(WorkflowRun, done.run_id).patient_id
            runs = (
                session.query(WorkflowRun)
                .filter(WorkflowRun.patient_id == patient_id)
                .all()
            )
            assert [run.id for run in runs] == [done.run_id]
        finally:
            session.close()

    def test_the_refusal_names_the_steps_it_refused(self, patient, monkeypatch):
        self.subjectless(monkeypatch)
        result = turn(patient, "who is the ceo of nvidia?", "s-subjectless-2")

        assert result.reply == UNSUPPORTED_TOPIC_REPLY
        session = fresh()
        try:
            gate = _guard(session, result.turn_id, "plan_earned")
        finally:
            session.close()
        assert gate["passed"] is False
        assert gate["detail"]["steps"] == ["documents", "follow_up"]

    def test_a_documents_question_still_earns_its_plan(self, patient, monkeypatch):
        """The direction that would break the feature rather than the gate.

        "Bring" and "submit" are in the documents vocabulary because a patient
        naming paperwork has named a subject — and the gate reads that list
        rather than a second one of its own.
        """
        self.subjectless(monkeypatch)
        result = turn(
            patient, "what documents do I need to bring?", "s-subjectless-3"
        )

        assert result.reply != UNSUPPORTED_TOPIC_REPLY
        assert result.run_id is not None

    def test_handing_something_over_is_not_a_question_and_still_earns_it(
        self, patient, monkeypatch
    ):
        """The one limb of ``_earns_plan`` the other three cannot cover.

        "Please file this" names paperwork and asks nothing, so
        ``mentions_domain_subject`` misses it (no "document"),
        ``names_followup_subject`` misses it, and ``detect_query`` misses it —
        it applies an "is this a question" filter this gate must not. Delete
        ``names_document_subject`` from the gate and only this test goes red,
        which is what makes that limb a guard rather than a decoration.
        """
        self.subjectless(monkeypatch)
        result = turn(
            patient, "please file this with my other paperwork", "s-subjectless-5"
        )

        assert result.reply != UNSUPPORTED_TOPIC_REPLY
        assert result.run_id is not None

    def test_a_message_with_nothing_behind_it_gets_the_generic_reply(
        self, patient, monkeypatch
    ):
        """No memory, so the offer has nothing to name and the refusal stands.

        The negative control for item 3 of round 10 as much as for this one: a
        memory that answered every refusal would be a memory that answers
        "who won the fifa final" with an offer to restart a booking.
        """
        self.subjectless(monkeypatch)
        result = turn(patient, "now do the other thing", "s-subjectless-4")

        assert result.reply == UNSUPPORTED_TOPIC_REPLY
        assert result.run_id is None


class HistoryReadingClassifierLlm(MockLlm):
    """A classifier that can only do its job with the transcript in front of it.

    The other half of item 2a, and the half that could have been broken
    silently. Bounding the *planner* to one message is safe because planning is
    a question about that message; bounding the classifier would not be,
    because "how does this relate to the request already running" is
    unanswerable without the thing it relates to.

    This stub makes that consequence visible instead of assumed: shown nothing
    older than the current message, it answers ``off_topic`` — which is what a
    trimmed mid-run turn would look like from inside a model, and what the
    round-4 zombie run was made of.
    """

    model: str = "history-reading-classifier-stub"

    def _classify(self, llm_request, available, done, text):  # noqa: ANN001
        if current_turn_start(llm_request) == 0:
            return function_call_response(
                "classify_message",
                {"message_class": "off_topic", "incoming_steps": []},
            )
        return super()._classify(llm_request, available, done, text)


class TestTheMidRunWindowSurvives:
    """Item 2a must not follow the planner's bound into the classifier."""

    @pytest.fixture(autouse=True)
    def _history_reading(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider",
            lambda name=None: HistoryReadingClassifierLlm(),
        )

    def test_a_classification_that_needs_the_transcript_still_gets_it(self, patient):
        first = turn(patient, BOOKING, "s-midrun-window")
        assert first.run_id is not None

        result = turn(patient, "what else is free that week?", "s-midrun-window")

        assert result.message_class != MessageClass.OFF_TOPIC.value
        assert result.run_id == first.run_id

    def test_the_mid_run_request_still_carries_the_earlier_turn(self, patient):
        """Read off the recorded request, so it says what was sent rather than
        what the reply implies."""
        first = turn(patient, BOOKING, "s-midrun-window-2")
        result = turn(patient, "what else is free that week?", "s-midrun-window-2")

        session = fresh()
        try:
            requests = [
                event.payload
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .order_by(TraceEvent.seq)
                .all()
                if event.event_type is TraceEventType.LLM_REQUEST
                and event.agent_name == "coordinator"
            ]
        finally:
            session.close()

        assert requests
        assert "cardiology" in json.dumps(requests[0]).lower()
        assert first.run_id is not None


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


class TestTheReviewWall:
    """Round 10 item 2. A run in front of staff answers, and does nothing else.

    Run 6 of the live transcript generated half its confusion here. A booking
    message arriving at ``pending_review`` was classified a *continuation*, so
    the plan carried on **inside the queued run**: routing re-ran and was
    accepted, ``propose_appointment(slot_id=7)`` was accepted, and the model
    shipped "shall I book it?" — an offer the state could not honour. The exact
    "yes" that followed had no proposal to land in and leaked classifier prose
    instead; "the earliest the better" looped ``list_other_slots`` nine times
    into the iteration budget.

    Reproduced under the mock before anything was changed, and it failed
    *worse* than live: with the class forced to continuation, **every** message
    re-dispatched the routing specialist and shipped its prose — including the
    status question that live got right.

    The three things that must still get through are pinned in
    :class:`TestTheReviewWallLetsThePatientOut`, deliberately without the stub:
    each one depends on the mock's own classification, and forcing continuation
    would make them unreachable and the tests vacuous.
    """

    AMBIGUOUS = "book an appointment, my kid has ear pain"
    SPECIALIST_TOOLS = {
        "resolve_department",
        "submit_routing",
        "find_available_slots",
        "propose_appointment",
        "list_other_slots",
    }

    @pytest.fixture(autouse=True)
    def _queued(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: QueuedRoutingLlm()
        )

    def _queued_run(self, patient, session_id: str):
        first = turn(patient, self.AMBIGUOUS, session_id)
        assert first.status == WorkflowStatus.PENDING_REVIEW.value
        return first

    @pytest.mark.parametrize(
        "message",
        [
            "book me a cardiology appointment",
            "yes",
            "option 3",
            "the earliest the better",
        ],
    )
    def test_all_four_live_shapes_get_the_wall(self, patient, message):
        session_id = f"s-wall-{abs(hash(message))}"
        first = self._queued_run(patient, session_id)

        result = turn(patient, message, session_id)

        assert result.reply == AWAITING_REVIEW_REPLY
        assert result.run_id == first.run_id
        assert result.status == WorkflowStatus.PENDING_REVIEW.value

    @pytest.mark.parametrize(
        "message",
        [
            "book me a cardiology appointment",
            "yes",
            "option 3",
            "the earliest the better",
        ],
    )
    def test_no_specialist_runs_behind_the_wall(self, patient, message):
        """The Coordinator still classifies — a supersede has to be *heard* to
        be honoured. What may not happen is anything downstream of it."""
        session_id = f"s-wall-tools-{abs(hash(message))}"
        self._queued_run(patient, session_id)

        result = turn(patient, message, session_id)

        assert self.SPECIALIST_TOOLS.isdisjoint(_tools_called(result.turn_id))

    def test_nothing_the_model_wrote_reaches_the_patient(self, patient):
        """Zero ``author: llm`` outbounds at this state, ever. The live leak was
        "It seems that your response was not a confirmation…" — true, useless,
        and about a proposal that should never have existed."""
        self._queued_run(patient, "s-wall-author")
        result = turn(patient, "book me a cardiology appointment", "s-wall-author")

        session = fresh()
        try:
            outbound = [
                event
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.OUTBOUND
            ]
        finally:
            session.close()

        assert [event.author for event in outbound] == [TraceAuthor.TEMPLATE]

    def test_the_queued_run_keeps_its_one_queue_item(self, patient):
        self._queued_run(patient, "s-wall-queue")
        turn(patient, "book me a cardiology appointment", "s-wall-queue")

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

    def test_a_status_question_is_answered_from_the_rows(self, patient):
        """Read-only, so it touches the queue item as little as any other side
        question — and it is detected here rather than trusted to the class,
        because at this state the class is exactly what went wrong. Under the
        stub it *is* a continuation, which is why this test needs the stub."""
        self._queued_run(patient, "s-wall-query")
        result = turn(patient, "show my upcoming appointments", "s-wall-query")

        assert "AC-000001" in result.reply
        assert result.reply != AWAITING_REVIEW_REPLY
        assert result.author is TraceAuthor.TEMPLATE

    def test_the_wall_is_recorded_as_a_guard_verdict(self, patient):
        self._queued_run(patient, "s-wall-trace")
        result = turn(patient, "book me a cardiology appointment", "s-wall-trace")

        session = fresh()
        try:
            wall = _guard(session, result.turn_id, "review_wall")
        finally:
            session.close()

        assert wall["passed"] is False
        assert wall["detail"]["answered"] == "wall"

    def test_no_slot_search_is_even_offered_to_the_model(self, patient):
        """Absent from the toolbelt, not merely unused — the rule that put
        ``submit_plan`` out of reach mid-run. A capability the state cannot
        honour is a capability the model will spend the budget on."""
        first = self._queued_run(patient, "s-wall-toolset")

        session = fresh()
        try:
            run = session.get(WorkflowRun, first.run_id)
            belt = Toolbelt(
                session,
                user=session.get(User, patient.id),
                patient_id=run.patient_id,
                writer=TraceWriter(session, session_id="s-wall-toolset"),
                run=run,
                message="the earliest the better",
            )
            names = {tool.__name__ for tool in belt.coordinator_tools()}
        finally:
            session.close()

        assert "list_other_slots" not in names
        assert "classify_message" in names


class TestTheReviewWallLetsThePatientOut:
    """The three exceptions, under the mock's own judgement.

    Deliberately without ``QueuedRoutingLlm``: each of these depends on the
    classification being something other than continuation, so forcing
    continuation would make every one of them unreachable and the tests would
    pass while proving nothing.
    """

    AMBIGUOUS = "book an appointment, my kid has ear pain"

    def _queued_run(self, patient, session_id: str):
        first = turn(patient, self.AMBIGUOUS, session_id)
        assert first.status == WorkflowStatus.PENDING_REVIEW.value
        return first

    def test_a_new_subject_still_supersedes(self, patient):
        """The eye-test supersede from the same live transcript. A patient may
        always replace a request that is waiting, and naming another department
        is what makes it a replacement rather than a nag."""
        first = self._queued_run(patient, "s-wall-out-1")

        result = turn(
            patient, "book me a dermatology appointment instead", "s-wall-out-1"
        )

        assert result.run_id != first.run_id
        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value

    def test_a_withdrawal_still_closes_the_run(self, patient):
        first = self._queued_run(patient, "s-wall-out-2")

        turn(patient, "actually never mind", "s-wall-out-2")

        session = fresh()
        try:
            assert session.get(WorkflowRun, first.run_id).status is (
                WorkflowStatus.CANCELLED
            )
        finally:
            session.close()

    def test_off_topic_still_gets_the_off_topic_reply(self, patient):
        """Walling this one would answer "who won the fifa final" with a note
        about a booking. The off-topic branch keeps its own answer, which is why
        the wall sits *after* it rather than in front of it."""
        self._queued_run(patient, "s-wall-out-3")

        result = turn(patient, "who won the fifa final", "s-wall-out-3")

        assert result.reply == SCOPE_REPLY


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


class TestOfferedMeansShown:
    """Round 7, item 2 — the union grew four times faster than the patient knew.

    ``_list_other_slots`` used to record everything the search returned, on the
    reasoning that a time below the fold is still answerable. A search returns
    up to twenty slots and a reply renders three, so a run that had shown six
    slots held an offered set of **twenty**, spanning two doctors and two days.

    That is not merely untidy. Both readers of the set — the re-proposal guard
    and ``read_selection`` — mean "times this patient has seen" by it, and the
    selection reader resolves a clock time only when it names *one* slot. Live,
    the patient pasted a rendered line back verbatim, "Monday 3 August at 04:00
    PM with Dr. Rahul Bose"; two slots in the union sat at 16:00, the unique
    rule correctly refused to guess, and the turn fell through to the classifier
    and drew another list. The slot they meant had been rendered. The one that
    made it ambiguous never was.
    """

    def _run(self, run_id: int) -> WorkflowRun:
        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            session.expunge(run)
            return run
        finally:
            session.close()

    def test_a_search_of_twenty_records_only_what_was_drawn(self, patient):
        """The proposal reply renders three of what the search found, and three
        is what the set holds — the held one among them."""
        result = turn(patient, BOOKING, "s-shown-1")
        run = self._run(result.run_id)

        offered = run.state["offered_slot_ids"]
        rendered = [
            line for line in result.reply.splitlines() if line[:2] in ("1.", "2.", "3.")
        ]
        assert len(rendered) == 3
        assert len(offered) == 3
        assert run.proposed_slot_id in offered

    def test_asking_for_other_times_records_the_three_it_lists(self, patient):
        """Not the twenty it found. The search is unchanged; only what is
        remembered as *shown* is."""
        first = turn(patient, BOOKING, "s-shown-2")
        before = set(self._run(first.run_id).state["offered_slot_ids"])

        turn(patient, "what else is free that week?", "s-shown-2")
        after = set(self._run(first.run_id).state["offered_slot_ids"])

        assert len(after - before) <= 3, "slots nobody was shown entered the set"

    def test_every_offered_slot_appears_in_something_the_patient_read(
        self, patient
    ):
        """The invariant, stated over a whole conversation rather than a turn.
        Every id in the set has to be traceable to a line somebody read."""
        replies = [turn(patient, BOOKING, "s-shown-3").reply]
        replies.append(turn(patient, "any other times?", "s-shown-3").reply)
        run = self._run(active_run(fresh(), 1).id)

        session = fresh()
        try:
            for slot_id in run.state["offered_slot_ids"]:
                slot = session.get(AppointmentSlot, slot_id)
                when = clock_time(slot.start_time)
                assert any(when in reply for reply in replies), (
                    f"slot {slot_id} at {when} is recorded as offered but was "
                    "never in a reply"
                )
        finally:
            session.close()


class TestASelectionIsReadWhileASlotIsHeld:
    """Round 7, item 3 — the reader ran one state too late.

    Round 6 put ``read_selection`` at ``in_progress`` and left
    ``pending_confirmation`` alone, on the reasoning that a held slot means the
    model's ``slot_question`` path already covers it. Live run 4 says otherwise,
    twice in consecutive turns, with a three-slot list on screen:

    * **"option 3"** — the exact-token reader said unread, the Coordinator
      called it *conflicting*, the refinement rule correctly refused the
      supersede, and the patient got a fresh morning list. The afternoon they
      had asked for was gone from it.
    * **"I want the 4pm appointment on august 3"** — read as a **decline**, with
      the reason "Patient specified a different time". The verdict enum is
      ``decline | non_answer | slot_question`` and has no selection in it, so a
      choice among the times on screen had no verdict it could be expressed as.

    They then booked 9:00 AM. They had asked for 4:00 PM four times.

    Nothing about the commit gate moves: the reader runs *before* the exact
    tokens and neither "yes" nor "no" names a time or a position, so they pass
    straight through it. The last two tests are that statement, falsifiable.
    """

    def _holding(self, patient, session_id: str) -> tuple[int, list[str]]:
        """A run at ``pending_confirmation`` with a numbered list on screen."""
        first = turn(patient, BOOKING, session_id)
        assert first.status == WorkflowStatus.PENDING_CONFIRMATION.value
        session = fresh()
        try:
            run = session.get(WorkflowRun, first.run_id)
            times = [
                clock_time(session.get(AppointmentSlot, slot_id).start_time)
                for slot_id in run.state["shortlist_slot_ids"]
            ]
        finally:
            session.close()
        return first.run_id, times

    def _held_time(self, run_id: int) -> str:
        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            return clock_time(session.get(AppointmentSlot, run.proposed_slot_id).start_time)
        finally:
            session.close()

    def test_a_list_number_moves_the_offer(self, patient):
        run_id, times = self._holding(patient, "s-hold-1")

        result = turn(patient, "option 3", "s-hold-1")

        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        assert result.run_id == run_id, "the run was replaced instead of moved"
        assert self._held_time(run_id) == times[2]

    def test_a_named_time_is_not_read_as_a_decline(self, patient):
        """The live sentence's shape. It named a time on the list and was
        recorded as a refusal of the list."""
        run_id, times = self._holding(patient, "s-hold-2")

        result = turn(patient, f"I want the {times[2].lower()} appointment", "s-hold-2")

        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        assert self._held_time(run_id) == times[2]
        assert "nothing has been booked" not in result.reply.lower()

    def test_no_model_is_asked(self, patient):
        """Read before the Coordinator, so a turn that needs no judgement
        spends none."""
        self._holding(patient, "s-hold-3")

        result = turn(patient, "option 2", "s-hold-3")

        session = fresh()
        try:
            agents = {
                event.agent_name
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
            }
        finally:
            session.close()
        assert "coordinator" not in agents

    def test_an_exact_yes_still_commits_the_held_slot(self, patient):
        """The pinned rule, unmoved. "yes" names no time and no position, so
        the reader above never sees it."""
        run_id, _ = self._holding(patient, "s-hold-4")
        held = self._held_time(run_id)

        result = turn(patient, "yes", "s-hold-4")

        assert result.status == WorkflowStatus.COMPLETED.value
        session = fresh()
        try:
            booked = appointments_for(session, run_id)
            assert len(booked) == 1
            slot = session.get(AppointmentSlot, booked[0].slot_id)
            assert clock_time(slot.start_time) == held
        finally:
            session.close()

    def test_an_exact_no_still_declines(self, patient):
        run_id, _ = self._holding(patient, "s-hold-5")

        result = turn(patient, "no", "s-hold-5")

        assert result.status == WorkflowStatus.IN_PROGRESS.value

    def test_a_number_outside_the_list_is_not_rounded_into_range(self, patient):
        """A near miss is not a choice. It falls through to the model, which is
        where a message nobody anticipated belongs."""
        run_id, times = self._holding(patient, "s-hold-6")

        turn(patient, "option 9", "s-hold-6")

        assert self._held_time(run_id) == times[0], "the offer moved on a number nobody showed"

    def test_a_true_non_answer_still_counts_against_the_cap(self, patient):
        """The bound is untouched. A selection is not a non-answer, and a
        non-answer is still one."""
        run_id, _ = self._holding(patient, "s-hold-7")

        turn(patient, "hmm let me think about it", "s-hold-7")

        session = fresh()
        try:
            assert session.get(WorkflowRun, run_id).non_answer_count == 1
        finally:
            session.close()


class TestATimingQuestionWhileHoldingIsAnswered:
    """Round 7, item 4 — a question about days met a nag.

    Live, at ``pending_confirmation``: "do you have any appointments at the end
    of the same week like thursday or friday preferably in the afternoon?" The
    Coordinator wrote prose about availability with no slot search behind it,
    the grounding guard correctly threw the prose away, and the fallback was the
    bare re-ask — with a stall counted against the patient for asking. The same
    words one turn later, at ``in_progress``, produced a real Thursday list.

    The trigger is the *destination*, not the class. Live it was a side
    question; under the mock the same sentence is a continuation whose
    confirmation verdict is ``non_answer``. Both end at the re-ask, so the
    re-ask is what this stands in front of.
    """

    QUESTION = (
        "do you have any appointments at the end of the same week like thursday "
        "or friday preferably in the afternoon?"
    )

    def _holding(self, patient, session_id: str) -> int:
        result = turn(patient, BOOKING, session_id)
        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        return result.run_id

    def test_the_question_gets_times(self, patient):
        self._holding(patient, "s-timing-1")

        result = turn(patient, self.QUESTION, "s-timing-1")

        assert "Other times that are free" in result.reply
        assert "Thursday" in result.reply

    def test_the_proposal_still_stands_afterwards(self, patient):
        """Answer-and-stay. A question is not an answer, and the time being
        held is still the patient's."""
        run_id = self._holding(patient, "s-timing-2")
        session = fresh()
        try:
            before = session.get(WorkflowRun, run_id).proposed_slot_id
        finally:
            session.close()

        result = turn(patient, self.QUESTION, "s-timing-2")

        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        session = fresh()
        try:
            assert session.get(WorkflowRun, run_id).proposed_slot_id == before
        finally:
            session.close()

    def test_asking_is_not_stalling(self, patient):
        """The counter bounds a re-ask loop, and a turn that rendered times is
        not a re-ask. The message that used to be counted here was a patient
        being told nothing and then charged for it."""
        run_id = self._holding(patient, "s-timing-3")

        turn(patient, self.QUESTION, "s-timing-3")

        session = fresh()
        try:
            assert session.get(WorkflowRun, run_id).non_answer_count == 0
        finally:
            session.close()

    def test_a_true_non_answer_still_counts(self, patient):
        """The control that keeps the bound a bound. "hmm let me think" names
        no day, so nothing here touches it."""
        run_id = self._holding(patient, "s-timing-4")

        turn(patient, "hmm let me think about it", "s-timing-4")

        session = fresh()
        try:
            assert session.get(WorkflowRun, run_id).non_answer_count == 1
        finally:
            session.close()

    def test_an_off_topic_message_naming_a_day_is_still_refused(self, patient):
        """"today" is a timing word and a question about the weather is not a
        question about this clinic's diary. Answering it with a slot list would
        be a second defect introduced while fixing the first."""
        self._holding(patient, "s-timing-5")

        result = turn(patient, "what is the weather like today?", "s-timing-5")

        assert "Other times that are free" not in result.reply


class TestTheEngineDoesNotOfferATimeThePatientHas:
    """Round 7, item 5 — a proposal built to bounce.

    Live run 8: the proposal held Monday 9:00 AM for a patient who already had a
    Dermatology appointment at Monday 9:00 AM with a different doctor. The
    Confirm failed with "You already have an appointment at that time" — the
    conflict guard working exactly as intended, on a collision the proposal
    engine had arranged.
    """

    REQUEST = "I need a dermatology appointment for a skin rash next week"
    AGAIN = "I need another dermatology appointment for a skin rash next week"

    def _book(self, patient, session_id: str) -> AppointmentSlot:
        turn(patient, self.REQUEST, session_id)
        result = asyncio.run(apply_patient_action(patient, "confirm", session_id))
        assert result.status == WorkflowStatus.COMPLETED.value
        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            appointment = session.get(Appointment, run.state["appointment_id"])
            slot = session.get(AppointmentSlot, appointment.slot_id)
            session.expunge(slot)
            return slot
        finally:
            session.close()

    def test_the_second_proposal_avoids_the_first_appointments_time(self, patient):
        held = self._book(patient, "s-clash-1")

        second = turn(patient, self.AGAIN, "s-clash-2")

        session = fresh()
        try:
            run = session.get(WorkflowRun, second.run_id)
            offered = session.get(AppointmentSlot, run.proposed_slot_id)
            assert offered.start_time != held.start_time
        finally:
            session.close()

    def test_the_second_booking_confirms_first_try(self, patient):
        """Run 8's shape, end to end. The bounce was not a near miss — it was
        the only possible outcome of the offer."""
        self._book(patient, "s-clash-3")
        turn(patient, self.AGAIN, "s-clash-4")

        result = asyncio.run(apply_patient_action(patient, "confirm", "s-clash-4"))

        assert result.status == WorkflowStatus.COMPLETED.value
        assert "already have an appointment" not in result.reply

    def test_a_clashing_time_is_not_even_listed(self, patient):
        """Not only the held one. A time in the shortlist is a time the patient
        may pick, and picking it would meet the same refusal."""
        held = self._book(patient, "s-clash-5")

        second = turn(patient, self.AGAIN, "s-clash-6")

        assert clock_time(held.start_time) not in second.reply


class TestATwoVerbMessageSaysWhatItDropped:
    """Round 7, item 6 — one verb was done and the other vanished.

    "One request confirms one thing" is a rule with a cost, and the cost was
    invisible. Live: "okay lets cancel that appointment and book a new one for
    skin rash" — the booking proceeded, the cancellation was neither done nor
    mentioned. The model dropped the verb at *classification*, so the plan
    validator never saw two and had nothing to report; only the sentence shows
    that two things were asked for.
    """

    MESSAGE = "okay lets cancel that appointment and book a new one for skin rash"

    VERBS = {"book": "the booking", "cancel": "the cancellation",
             "reschedule": "the reschedule"}

    def test_the_dropped_verb_is_acknowledged(self, patient):
        result = turn(patient, self.MESSAGE, "s-twoverb-1")

        assert "One change at a time" in result.reply
        assert "ask me about the other one right after" in result.reply

    def test_it_names_the_one_being_done(self, patient):
        result = turn(patient, self.MESSAGE, "s-twoverb-2")

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            doing = next(step for step in run.plan if step in self.VERBS)
        finally:
            session.close()
        assert f"I'll start with {self.VERBS[doing]}" in result.reply

    def test_the_line_is_part_of_what_the_trace_recorded(self, patient):
        """Added inside the turn envelope, so the outbound event carries it. A
        sentence appended after the turn was written down is a sentence the
        trace does not vouch for."""
        result = turn(patient, self.MESSAGE, "s-twoverb-3")

        session = fresh()
        try:
            outbound = [
                event
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.OUTBOUND
            ]
            assert len(outbound) == 1
            assert "One change at a time" in outbound[0].payload["content"]
        finally:
            session.close()

    def test_a_one_verb_message_says_nothing_of_the_kind(self, patient):
        """The control. Most messages name one verb, and a note that appeared
        beside them would read as the system being confused."""
        result = turn(patient, BOOKING, "s-twoverb-4")

        assert "One change at a time" not in result.reply


class TestTheSelectionReaderNeverChangesTheVerb:
    """Round 7 stopped this reader touching a change proposal at all; round 8
    narrows that to what it was protecting.

    The danger was never the *reading* — it was the hold. ``hold_offered_slot``
    went through ``_propose_appointment``, which sets ``proposed_action`` to
    BOOK, so a "3" answering three alternatives under a held reschedule would
    have turned it into a booking and the patient's Confirm would have
    committed something they were never shown.

    Standing aside entirely cost a live run instead: the alternatives were
    rendered, the patient said "3", this reader matched it and was suppressed,
    the turn fell to the classifier, which called it a **withdrawal** — and two
    turns later the patient re-stated the same time in words and the run was
    superseded into a routed staff review. So the slot moves now and the verb
    does not, which is the rule the old skip was standing in for.
    """

    def _run_holding(
        self, action: ProposedAction, session_id: str
    ) -> tuple[int, int]:
        """A run holding a change proposal, and the id of its second option.

        The options are drawn from the seeded appointment's own department: a
        reschedule may only move within it, so a shortlist from anywhere else
        would be refused for a reason that has nothing to do with this rule.
        """
        session = fresh()
        try:
            appointment = session.get(Appointment, SEEDED_APPOINTMENT_ID)
            slots = (
                session.query(AppointmentSlot)
                .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
                .filter(
                    Doctor.department_id == appointment.department_id,
                    AppointmentSlot.status == SlotStatus.AVAILABLE,
                    AppointmentSlot.start_time > clock.now(),
                )
                .order_by(AppointmentSlot.start_time)
                .limit(3)
                .all()
            )
            run = WorkflowRun(
                patient_id=1,
                status=WorkflowStatus.PENDING_CONFIRMATION,
                plan=["reschedule", "follow_up"],
                completed_steps=[],
                state={
                    "offered_slot_ids": [slot.id for slot in slots],
                    "shortlist_slot_ids": [slot.id for slot in slots],
                },
                request_text="reschedule my appointment",
                proposed_action=action,
                proposed_appointment_id=SEEDED_APPOINTMENT_ID,
                proposed_slot_id=slots[0].id if action is ProposedAction.RESCHEDULE else None,
                session_id=session_id,
            )
            session.add(run)
            session.commit()
            return run.id, slots[1].id
        finally:
            session.close()

    def test_a_list_number_moves_the_time_and_nothing_else(self, patient):
        run_id, second = self._run_holding(ProposedAction.RESCHEDULE, "s-nochange-1")

        turn(patient, "option 2", "s-nochange-1")

        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            assert run.proposed_action is ProposedAction.RESCHEDULE
            assert run.proposed_appointment_id == SEEDED_APPOINTMENT_ID
            assert run.proposed_slot_id == second
        finally:
            session.close()

    def test_a_cancellation_has_no_time_to_move(self, patient):
        """Nothing is held, so a number cannot be an answer to it — and reading
        one as though it were would attach a slot to a cancellation."""
        run_id, _ = self._run_holding(ProposedAction.CANCEL, "s-nochange-2")

        turn(patient, "option 2", "s-nochange-2")

        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            assert run.proposed_action is ProposedAction.CANCEL
            assert run.proposed_slot_id is None
        finally:
            session.close()


def _tools_called(turn_id: str) -> list[str]:
    """Every tool call in one turn, in order."""
    session = fresh()
    try:
        return [
            event.payload["tool"]
            for event in session.query(TraceEvent)
            .filter(TraceEvent.turn_id == turn_id)
            .order_by(TraceEvent.seq)
            .all()
            if event.event_type is TraceEventType.TOOL_CALL
        ]
    finally:
        session.close()


class TestTheAnswerToWhichAppointment:
    """Round 8, item 1 — the numbered list finally has a reader.

    ``render_appointment_choice`` numbers the patient's appointments and records
    the ids in the same breath, precisely so that "2" can mean something. For a
    whole round nothing read it. The number went to the Coordinator like any
    other message, and live it came back classified as a *new cancellation
    request*: the supersede was refused for naming no new subject, the refusal's
    recovery searched for slots on a run with no department and was told so, the
    re-plan budget went, and the run ended **failed** — "I'm sorry, I couldn't
    complete this request", to a patient who had answered the question they were
    asked one message earlier.
    """

    def _asked(self, patient, session_id: str, verb: str = "cancel"):
        """A change run that has drawn the list. Returns (run_id, listed ids).

        Two appointments in two departments, so the department-name path below
        has something to distinguish.
        """
        turn(patient, "I need a dermatology appointment for a rash", session_id)
        turn(patient, "yes", session_id)
        result = turn(patient, f"please {verb} my appointment", session_id)

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            listed = list(run.state.get("listed_appointment_ids") or [])
        finally:
            session.close()
        assert len(listed) == 2, "the run did not ask which appointment"
        return result.run_id, listed

    def _proposal(self, run_id: int) -> tuple:
        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            return run.proposed_action, run.proposed_appointment_id, run.status
        finally:
            session.close()

    def test_a_bare_number_proposes_that_appointment(self, patient):
        run_id, listed = self._asked(patient, "s-choice-1")

        turn(patient, "2", "s-choice-1")

        action, appointment_id, status = self._proposal(run_id)
        assert action is ProposedAction.CANCEL
        assert appointment_id == listed[1]
        assert status is WorkflowStatus.PENDING_CONFIRMATION

    def test_the_classifier_is_never_asked(self, patient):
        """The whole point of reading it in code. A number answering a list this
        run drew is not a judgement call, and the live failure is what asking
        cost: the model called it a new request, and everything after that was
        downstream of a wrong answer to a question nobody needed to ask."""
        self._asked(patient, "s-choice-2")

        result = turn(patient, "2", "s-choice-2")

        assert "classify_message" not in _tools_called(result.turn_id)

    def test_an_announced_position(self, patient):
        run_id, listed = self._asked(patient, "s-choice-3")

        turn(patient, "option 1", "s-choice-3")

        assert self._proposal(run_id)[1] == listed[0]

    def test_a_position_with_an_instruction_after_it(self, patient):
        """Run 4's sentence, in shape: a choice and then what to do with it.
        The leading numeral is the answer; the rest is the errand."""
        run_id, listed = self._asked(patient, "s-choice-4")

        turn(patient, "1. and please make it quick", "s-choice-4")

        assert self._proposal(run_id)[1] == listed[0]

    def test_a_number_outside_the_list_goes_to_the_model(self, patient):
        """Not a near miss to be rounded into range. Seven against a list of two
        is a message nobody anticipated, and the model is where those belong."""
        run_id, _ = self._asked(patient, "s-choice-5")

        result = turn(patient, "7", "s-choice-5")

        assert "classify_message" in _tools_called(result.turn_id)
        assert self._proposal(run_id)[0] is not ProposedAction.CANCEL

    def test_naming_the_department_still_works(self, patient):
        """The other way to answer the list, and it was never broken — this
        reader must not have taken it away. The cue path owns that sentence."""
        run_id, _ = self._asked(patient, "s-choice-6")

        turn(patient, "the dermatology one", "s-choice-6")

        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            chosen = session.get(Appointment, run.proposed_appointment_id)
            assert run.proposed_action is ProposedAction.CANCEL
            assert chosen.department.name == "Dermatology"
        finally:
            session.close()

    def test_a_chosen_reschedule_searches_for_that_appointment(self, patient):
        """The verb decides what the answer *does*. A cancel proposes; a
        reschedule carries on into its own plan, where the slot search is — with
        the target settled, so the search is for the right department."""
        run_id, listed = self._asked(patient, "s-choice-7", verb="reschedule")

        result = turn(patient, "1", "s-choice-7")

        assert "find_slots_for_reschedule" in _tools_called(result.turn_id)
        action, appointment_id, status = self._proposal(run_id)
        assert action is ProposedAction.RESCHEDULE
        assert appointment_id == listed[0]
        assert status is WorkflowStatus.PENDING_CONFIRMATION

    def test_the_choice_is_not_asked_again_once_it_is_answered(self, patient):
        """A patient who has answered must not be re-asked; re-drawing the list
        reads as the answer having been thrown away."""
        self._asked(patient, "s-choice-8", verb="reschedule")

        result = turn(patient, "1", "s-choice-8")

        assert "so I want to be sure which one" not in result.reply


class RefusingCoordinatorLlm(MockLlm):
    """Run 6's Coordinator: every message on a live run is a fresh cancellation.

    The mock reads "2" correctly through the specialist, so it cannot show what
    happened live — the classifier path has to be forced. ``incoming_steps`` of
    ``[cancel]`` against a cancel run is what makes ``shows_no_difference``
    refuse the supersede, which is the branch that then had nothing to do.
    """

    model: str = "refusing-coordinator-stub"

    def _classify(self, llm_request, available, done, text):  # noqa: ANN001
        if latest_tool_result(llm_request, "classify_message") is None:
            return function_call_response(
                "classify_message",
                {"message_class": "conflicting", "incoming_steps": ["cancel"]},
            )
        return super()._classify(llm_request, available, done, text)


class TestARefusedSupersedeAsksTheQuestionAgain:
    """Round 8, item 1b — the recovery that had nothing to recover with.

    A refused supersede is answered with times. On a change run that has not
    yet learned *which* appointment there are no times to answer with: the
    search needs a department and the department lives on the appointment
    nobody has picked. Falling through from there re-dispatched the specialist,
    which asked "which one?" again, proposed nothing, and spent the run's last
    re-plan — so the answer to "which one?" was "I couldn't complete this
    request."
    """

    @pytest.fixture(autouse=True)
    def _refusing(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: RefusingCoordinatorLlm()
        )

    def _asked(self, patient, session_id: str) -> int:
        turn(patient, "I need a dermatology appointment for a rash", session_id)
        turn(patient, "yes", session_id)
        result = turn(patient, "need help to cancel my appointment", session_id)
        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert len(run.state["listed_appointment_ids"]) == 2
        finally:
            session.close()
        return result.run_id

    def test_the_run_survives_a_message_the_reader_cannot_read(self, patient):
        run_id = self._asked(patient, "s-refuse-1")

        result = turn(patient, "please go ahead with that", "s-refuse-1")

        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            assert run.status is WorkflowStatus.IN_PROGRESS
        finally:
            session.close()
        assert result.reply != FAILED_REPLY

    def test_it_answers_by_asking_the_question_again(self, patient):
        """A re-ask is not progress, but it is a turn the patient can act on —
        and the list it draws is the one the code reader can read."""
        self._asked(patient, "s-refuse-2")

        result = turn(patient, "please go ahead with that", "s-refuse-2")

        assert "so I want to be sure which one" in result.reply

    def test_no_escalation_is_raised_for_a_run_that_is_fine(self, patient):
        """The failure path opens a ``system_failure`` escalation, correctly —
        so a run that should not have failed leaves a queue item behind that a
        human then works on for nothing."""
        run_id = self._asked(patient, "s-refuse-3")

        turn(patient, "please go ahead with that", "s-refuse-3")

        session = fresh()
        try:
            assert (
                session.query(Escalation)
                .filter(Escalation.workflow_run_id == run_id)
                .count()
                == 0
            )
        finally:
            session.close()

    def test_no_specialist_is_dispatched_to_ask_it(self, patient):
        """What the recovery is *for*, now that the re-plan is no longer spent
        on asking: the answer is already known to code, so the turn does not
        spend a model call rediscovering it.

        Without this the turn falls through to ``_execute_plan``, the
        Appointment agent runs, asks "which one?" in its own words, and the
        list is drawn afterwards regardless — the same reply for the price of a
        dispatch. That was the *whole* difference once the budget stopped being
        charged for a question, and a difference nobody asserts is a line
        nobody can defend.
        """
        self._asked(patient, "s-refuse-4")

        result = turn(patient, "please go ahead with that", "s-refuse-4")

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
        assert "appointment" not in agents


class TestASelectionMovesAHeldReschedule:
    """Round 8, item 2 — the same verb, a different time.

    Round 7 stopped the selection reader touching a run holding a reschedule,
    because the hold would have converted it into a booking. Live, that skip
    cost the run: three alternatives were rendered under a held reschedule, the
    patient answered "3", the reader matched it — the trace records the slot id
    — and was suppressed, so the turn fell to the classifier, which called it a
    withdrawal. Two turns later the patient re-stated the same time in words and
    the run was superseded into a routed staff review.
    """

    def _holding(self, patient, session_id: str) -> tuple[int, list[int]]:
        """A run holding a reschedule, with alternatives on screen."""
        turn(patient, "lets reschedule my appointment to next week", session_id)
        turn(
            patient,
            "can you show me other times for this appointment in the afternoon?",
            session_id,
        )
        session = fresh()
        try:
            run = (
                session.query(WorkflowRun)
                .filter(WorkflowRun.session_id == session_id)
                .order_by(WorkflowRun.id.desc())
                .first()
            )
            assert run.proposed_action is ProposedAction.RESCHEDULE
            shortlist = list(run.state.get("shortlist_slot_ids") or [])
            assert len(shortlist) >= 3, "no alternatives were rendered"
            return run.id, shortlist
        finally:
            session.close()

    def test_the_number_moves_the_time_and_keeps_the_verb(self, patient):
        run_id, shortlist = self._holding(patient, "s-rehold-1")

        turn(patient, "3", "s-rehold-1")

        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            assert run.proposed_slot_id == shortlist[2]
            assert run.proposed_action is ProposedAction.RESCHEDULE
            assert run.proposed_appointment_id == SEEDED_APPOINTMENT_ID
            assert run.status is WorkflowStatus.PENDING_CONFIRMATION
        finally:
            session.close()

    def test_the_reply_still_asks_about_moving_it(self, patient):
        self._holding(patient, "s-rehold-2")

        result = turn(patient, "3", "s-rehold-2")

        assert "move it" in result.reply
        assert "book it" not in result.reply

    def test_confirming_moves_the_appointment_to_the_chosen_time(self, patient):
        """The whole point of reading it: the patient's choice is what commits.
        The reader itself commits nothing — the run is still waiting on the
        exact word."""
        run_id, shortlist = self._holding(patient, "s-rehold-3")

        turn(patient, "3", "s-rehold-3")
        session = fresh()
        try:
            assert session.get(Appointment, SEEDED_APPOINTMENT_ID).slot_id != shortlist[2]
        finally:
            session.close()

        turn(patient, "yes", "s-rehold-3")

        session = fresh()
        try:
            assert session.get(Appointment, SEEDED_APPOINTMENT_ID).slot_id == shortlist[2]
        finally:
            session.close()

    def test_a_different_verb_mid_reschedule_still_goes_to_the_model(self, patient):
        """The control. This reader answers "which of these times"; deciding
        that a sentence means something else entirely is language."""
        self._holding(patient, "s-rehold-4")

        result = turn(patient, "actually just cancel it instead", "s-rehold-4")

        assert "classify_message" in _tools_called(result.turn_id)


#: The sentence gpt-4o-mini wrote to go with a classification code refused.
WITHDRAWAL_PROSE = (
    "It seems you've decided to withdraw your request again. If you have any "
    "other questions, feel free to reach out!"
)


class OverrulingLlm(MockLlm):
    """Classifies every message on a live run as a withdrawal, and says so.

    Run 4's turn, in a stub: the patient answered a question, the model called
    it an abandonment, and the cue guard refused the class. What survived the
    refusal was the sentence.
    """

    model: str = "overruling-stub"
    proposed_class: str = "withdrawal"

    def _classify(self, llm_request, available, done, text):  # noqa: ANN001
        if latest_tool_result(llm_request, "classify_message") is None:
            return function_call_response(
                "classify_message",
                {"message_class": self.proposed_class, "incoming_steps": []},
            )
        return text_response(WITHDRAWAL_PROSE)


class AcceptedClassLlm(OverrulingLlm):
    """The control, differing in exactly one thing: a class code accepts.

    Same prose, same shape, same turn. If the sentence disappears here too then
    the rule is "the model never speaks", which is a different and much larger
    change than the one being made.
    """

    model: str = "accepted-class-stub"
    proposed_class: str = "side_question"


class TestAnOverruledClassificationLosesItsProse:
    """Round 8, item 3 — a rejected verdict takes its sentence with it.

    The model wrote its reply *believing* the verdict code has just rejected,
    so the two cannot both be right. Live, a patient choosing option 3 was told
    "it seems you've decided to withdraw your request again", above a re-ask
    offering them the time they had just picked. Nothing was withdrawn — the
    guard did its job — and the words went out anyway.
    """

    def _held(self, patient, session_id: str):
        return turn(patient, BOOKING, session_id)

    def test_the_refused_withdrawal_says_nothing(self, patient, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: OverrulingLlm()
        )
        self._held(patient, "s-overrule-1")

        result = turn(patient, "hmm ok", "s-overrule-1")

        assert "withdraw" not in result.reply
        assert result.author == TraceAuthor.TEMPLATE.value

    def test_the_facts_still_reach_the_patient(self, patient, monkeypatch):
        """Discarding the prose is not discarding the turn: the re-ask states
        what is held and what would settle it, as it does for every other
        non-answer."""
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: OverrulingLlm()
        )
        self._held(patient, "s-overrule-2")

        result = turn(patient, "hmm ok", "s-overrule-2")

        assert "The time I'm holding is" in result.reply

    def test_the_outbound_event_names_the_template(self, patient, monkeypatch):
        """Trace completeness: an author of ``llm`` on a reply the model did not
        write vouches for a sentence nobody said."""
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: OverrulingLlm()
        )
        self._held(patient, "s-overrule-3")
        result = turn(patient, "hmm ok", "s-overrule-3")

        session = fresh()
        try:
            outbound = [
                event
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .order_by(TraceEvent.seq)
                .all()
                if event.event_type is TraceEventType.OUTBOUND
            ]
            assert [event.author for event in outbound] == [TraceAuthor.TEMPLATE]
        finally:
            session.close()

    def test_an_accepted_class_keeps_its_prose(self, patient, monkeypatch):
        """The control. One stub, one difference — the class code accepts — and
        the model's sentence is the patient's reply exactly as before."""
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: AcceptedClassLlm()
        )
        self._held(patient, "s-overrule-4")

        result = turn(patient, "hmm ok", "s-overrule-4")

        assert WITHDRAWAL_PROSE in result.reply
        assert result.author == TraceAuthor.LLM.value

    def test_a_real_withdrawal_is_still_applied(self, patient, monkeypatch):
        """Item 3 must not suppress a withdrawal the patient asked for. The cue
        is in the message, so nothing is overruled and the run closes."""
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: OverrulingLlm()
        )
        first = self._held(patient, "s-overrule-5")

        turn(patient, "never mind, forget it", "s-overrule-5")

        session = fresh()
        try:
            run = session.get(WorkflowRun, first.run_id)
            assert run.status is WorkflowStatus.CANCELLED
        finally:
            session.close()


class ClashProposingLlm(MockLlm):
    """``gpt-4o-mini``, proposing a time the search had already withheld.

    The mock cannot reproduce this defect on its own, and that is the point of
    the stub. The mock proposes out of ``slots``, and the clashing time is not
    in there — the search removed it correctly. The live model read
    ``withheld_for_patient``, the field that exists so *code* can explain a
    missing time, and treated it as part of the menu.

    It proposes the clash exactly once. A stub that kept proposing it would be
    testing the rejected-repeat budget instead, which is a different open item.
    """

    model: str = "clash-proposing-stub"
    clashing_slot_id: int = 0
    moving_appointment_id: int = 0

    def _change_appointment(self, llm_request, done, task, step):  # noqa: ANN001
        if (
            step == "reschedule"
            and "find_slots_for_reschedule" in done
            and task.get("committed") != step
            and latest_tool_result(llm_request, "propose_reschedule") is None
        ):
            return function_call_response(
                "propose_reschedule",
                {
                    "appointment_id": self.moving_appointment_id,
                    "slot_id": self.clashing_slot_id,
                },
            )
        return super()._change_appointment(llm_request, done, task, step)


class TestARescheduleCannotLandOnTheirOwnAppointment:
    """Round 9, item 1, end to end — runs #6 and #7 of the live session.

    The patient booked Ophthalmology for Thursday 6 August at 9:00 AM, then
    asked to move their Dermatology appointment "to August 6th". The search
    withheld the 9:00 and said why; the model proposed it anyway; the proposal
    was recorded, the patient was shown a card offering it, they confirmed, and
    ``reschedule_appointment`` committed. Appointments 3 and 5, both confirmed,
    both at 2026-08-06 09:00 — a state the booking path had refused since
    Phase 2 and the reschedule path had never been taught to.

    The department of the clashing slot matches the appointment being moved on
    purpose: a cross-department slot is refused one guard earlier, and a test
    that trips *that* refusal says nothing at all about this one.
    """

    @pytest.fixture
    def two_appointments(self, seeded_db, patient):
        """Asha's seeded Cardiology appointment, plus one elsewhere — and the
        Cardiology slot that collides with the second."""
        seeded = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID)
        elsewhere = (
            seeded_db.query(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .join(Department, Department.id == Doctor.department_id)
            .filter(
                Department.name == "Ophthalmology",
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                AppointmentSlot.start_time > clock.now(),
                AppointmentSlot.start_time != seeded.slot.start_time,
            )
            .order_by(AppointmentSlot.start_time)
            .first()
        )
        clashing = (
            seeded_db.query(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .filter(
                Doctor.department_id == seeded.department_id,
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                AppointmentSlot.start_time == elsewhere.start_time,
            )
            .first()
        )
        assert clashing is not None, "no Cardiology slot at the Ophthalmology time"

        elsewhere.status = SlotStatus.BOOKED
        seeded_db.add(
            Appointment(
                patient_id=1,
                doctor_id=elsewhere.doctor_id,
                slot_id=elsewhere.id,
                department_id=elsewhere.doctor.department_id,
                status=AppointmentStatus.CONFIRMED,
                reference_code="AC-000099",
                reason="vision test",
            )
        )
        seeded_db.commit()
        return clashing.id, elsewhere.start_time

    @pytest.fixture(autouse=True)
    def _propose_the_clash(self, monkeypatch, two_appointments):
        clashing_slot_id, _ = two_appointments
        monkeypatch.setattr(
            "app.agents.base.get_provider",
            lambda name=None: ClashProposingLlm(
                clashing_slot_id=clashing_slot_id,
                moving_appointment_id=SEEDED_APPOINTMENT_ID,
            ),
        )

    def _ask(self, patient, when, session_id: str):
        """The live shape, both turns of it.

        Naming the destination date is naming the *other* appointment's date,
        so turn 1 is refused as ambiguous — a date cue matching one appointment
        and a department cue matching the other — and the numbered list is
        drawn. That is what run #7 did, and the patient answered "2". Turn 2
        carries no window, which is why the live search came back with all 138
        slots and the model reached past them into the withheld list.

        Without this second turn the run never proposes anything at all, and
        every assertion below passes against a refusal one guard too early.
        """
        first = turn(
            patient,
            f"move my cardiology appointment to {when:%B} {when.day}",
            session_id,
        )
        assert "which one" in first.reply.lower() or "more than one" in first.reply
        choice = next(
            line.strip()[0]
            for line in first.reply.splitlines()
            if line.strip()[:1].isdigit() and "Cardiology" in line
        )
        return turn(patient, choice, session_id)

    def test_the_clashing_slot_is_never_held(self, patient, two_appointments):
        clashing_slot_id, when = two_appointments

        result = self._ask(patient, when, "s-clash-e2e-1")

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.proposed_slot_id != clashing_slot_id
        finally:
            session.close()

    def test_the_patient_is_never_shown_the_clashing_time(
        self, patient, two_appointments
    ):
        """The reply is the part the patient acts on. A card offering a time
        the commit is guaranteed to refuse is a promise the system cannot
        keep, and they press Confirm on it.

        The whole rendered instant, not just the clock time: the seed lays the
        same hours down every day, so "11:00 AM" alone appears in perfectly
        good offers for other days and would fail this for nothing.
        """
        _, when = two_appointments

        result = self._ask(patient, when, "s-clash-e2e-2")

        assert f"{when:%A} {when.day} {when:%B} at {clock_time(when)}" not in result.reply

    def test_no_two_live_appointments_share_a_start_time(self, patient, two_appointments):
        """The state itself, after the whole turn — the query that found the
        live defect, asked of the system rather than of one function."""
        _, when = two_appointments
        self._ask(patient, when, "s-clash-e2e-3")
        asyncio.run(apply_patient_action(patient, "confirm", "s-clash-e2e-3"))

        session = fresh()
        try:
            starts = [
                start
                for (start,) in session.query(AppointmentSlot.start_time)
                .join(Appointment, Appointment.slot_id == AppointmentSlot.id)
                .filter(
                    Appointment.patient_id == 1,
                    Appointment.status.in_(
                        (AppointmentStatus.PENDING, AppointmentStatus.CONFIRMED)
                    ),
                )
                .all()
            ]
            assert len(starts) == len(set(starts))
        finally:
            session.close()

    def test_the_refusal_is_in_the_trace(self, patient, two_appointments):
        """A refused proposal changes nothing, so the trace row is the only
        evidence the guard ran at all."""
        _, when = two_appointments
        result = self._ask(patient, when, "s-clash-e2e-4")

        session = fresh()
        try:
            rejected = [
                event
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.VALIDATION
                and event.payload.get("what") == "appointment_change_proposal"
                and event.payload.get("accepted") is False
            ]
            assert rejected, "the clash refusal left no trace row"
        finally:
            session.close()


class TestATimingConstraintIsHonouredOrNamed:
    """Round 9, item 5 — end to end, through a real turn.

    The live failures, in one place: "Thursday next week or after August 6th"
    parsed to nothing and was answered with Monday slots; "after August 6th"
    included the 6th; "more slots in the afternoon?" came back with 10 and
    11 AM. Each is a constraint the patient stated and the reply ignored
    without saying so.
    """

    def _holding(self, patient, session_id: str):
        """A run holding a proposal, which is where timing questions land."""
        first = turn(patient, BOOKING, session_id)
        assert first.status == WorkflowStatus.PENDING_CONFIRMATION.value
        return first

    def test_an_unreadable_constraint_is_admitted_not_hidden(self, patient):
        self._holding(patient, "s-window-1")

        # Names a timing word ("later", "week") that `resolve_date` cannot turn
        # into a window, and is phrased as the availability question the
        # classifier routes to the timing path — both are needed, and the first
        # phrasing I tried satisfied only one of them and was answered as
        # off-topic.
        result = turn(patient, "what else is free later in the week?", "s-window-1")

        assert "couldn't read that as a day or time" in result.reply

    def test_a_readable_constraint_is_answered_without_apology(self, patient):
        """The negative control. A window that worked must not be narrated —
        a note on every list is a note nobody reads."""
        self._holding(patient, "s-window-2")

        result = turn(patient, "what else is free next week?", "s-window-2")

        assert "couldn't read that" not in result.reply
        assert "Nothing free" not in result.reply

    def test_reading_a_window_never_costs_an_llm_call(self, patient):
        """Layer order, falsified from the trace rather than from the reply.

        Layer (a) is deterministic vocabulary, and the whole point of putting
        it first is that a phrase it can read must never reach a model. Two
        turns, and the second one's request count is the assertion.
        """
        self._holding(patient, "s-window-3")

        result = turn(patient, "anything on thursday next week?", "s-window-3")

        session = fresh()
        try:
            calls = [
                event
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.LLM_REQUEST
                and event.agent_name == "appointment"
            ]
        finally:
            session.close()

        assert not calls, "a phrase layer (a) can read reached a specialist"

    def test_after_a_date_starts_the_day_after(self, patient):
        """The off-by-one, end to end: "after August 6th" showed August 6th."""
        self._holding(patient, "s-window-4")

        result = turn(patient, "any availability after august 6th?", "s-window-4")

        assert "6 August" not in result.reply


class WindowProposingLlm(MockLlm):
    """A model that answers a timing question by proposing a window.

    The mock never does this on its own — layer (a) reads the phrases it knows
    and the mock asks for a list — so without a stub, layer (b) is code with no
    caller and its wiring is unproven. ``window`` is what this one proposes,
    so a test can hand it a good window or an impossible one.
    """

    model: str = "window-proposing-stub"
    window_start: str = ""
    window_end: str = ""

    def _classify(self, llm_request, available, done, text):  # noqa: ANN001
        if (
            "propose_search_window" in available
            and "propose_search_window" not in done
        ):
            return function_call_response(
                "propose_search_window",
                {"start": self.window_start, "end": self.window_end},
            )
        return super()._classify(llm_request, available, done, text)


class TestAModelProposedWindowIsDisposedByCode:
    """Round 9, item 5b, through a real turn — the tool exists and is reachable.

    The validation itself is pinned at the toolbelt seam. What only a turn can
    show is that a model at ``pending_confirmation`` can actually reach the
    tool, and that a window code refuses leaves the patient with the honest
    answer rather than with silence.
    """

    def _holding(self, patient, session_id: str):
        first = turn(patient, BOOKING, session_id)
        assert first.status == WorkflowStatus.PENDING_CONFIRMATION.value

    def test_an_impossible_window_still_answers_the_patient(self, patient, monkeypatch):
        self._holding(patient, "s-propwin-1")
        monkeypatch.setattr(
            "app.agents.base.get_provider",
            lambda name=None: WindowProposingLlm(
                window_start="2029-06-01", window_end="2029-06-07"
            ),
        )

        result = turn(patient, "what else is free around then?", "s-propwin-1")

        assert result.reply
        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value

    def test_the_refusal_is_recorded(self, patient, monkeypatch):
        self._holding(patient, "s-propwin-2")
        monkeypatch.setattr(
            "app.agents.base.get_provider",
            lambda name=None: WindowProposingLlm(
                window_start="2029-06-01", window_end="2029-06-07"
            ),
        )

        result = turn(patient, "what else is free around then?", "s-propwin-2")

        session = fresh()
        try:
            refused = [
                event.payload
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.VALIDATION
                and event.payload.get("what") == "search_window"
            ]
        finally:
            session.close()

        assert refused and refused[0]["accepted"] is False

    def test_a_turn_that_showed_times_is_not_a_non_answer(self, patient, monkeypatch):
        """The third instance of one shape, and the live sweep found this one too.

        The stall counter bounds a *re-ask loop*, and its own comment says a
        turn that rendered times is not a re-ask. The code said something
        narrower — a turn where **code** rendered times — because
        ``answered_timing`` is set only by the code-driven search, which is
        skipped when the model has already answered. Those two readings agreed
        until round 9 handed the model its own way to render times, and then a
        patient who asked about timing, was shown times, and was told nothing
        was wrong still had a non-answer charged against them.

        Pre-round-9 this conversation passed live and it fails now, which is
        what makes it a regression rather than another chronic red.
        """
        self._holding(patient, "s-propwin-4")
        monkeypatch.setattr(
            "app.agents.base.get_provider",
            lambda name=None: WindowProposingLlm(
                window_start=clock.today().isoformat(),
                window_end=(clock.today() + timedelta(days=6)).isoformat(),
            ),
        )

        result = turn(patient, "what else is free around then?", "s-propwin-4")

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.non_answer_count == 0
        finally:
            session.close()

    def test_nothing_is_booked_by_a_window(self, patient, monkeypatch):
        """The bound that makes reading a window freely safe: it searches, and
        that is all it can do."""
        self._holding(patient, "s-propwin-3")
        before = _appointment_count(1)
        monkeypatch.setattr(
            "app.agents.base.get_provider",
            lambda name=None: WindowProposingLlm(
                window_start=clock.today().isoformat(),
                window_end=(clock.today() + timedelta(days=6)).isoformat(),
            ),
        )

        turn(patient, "what else is free around then?", "s-propwin-3")

        assert _appointment_count(1) == before


class TestSayingWhyATimeIsMissing:
    """Round 8, item 4 — the withheld time is explained, not just subtracted.

    Live: "how about 11am on july 29?" came back as 9:00, 10:00 and 2:00. The
    11:00 was withheld correctly — the patient had an Orthopedics appointment
    at 11:00 that day and the commit would have refused it — but the reply said
    nothing about it, so a patient who asked a specific question got a list that
    reads as though they had said nothing at all.
    """

    def _busy_at_eleven(self, patient, session_id: str):
        """Book the patient into 11:00, then start a booking on the same day."""
        turn(patient, "I need a cardiology appointment next week", session_id)
        turn(patient, "3", session_id)
        turn(patient, "yes", session_id)

        session = fresh()
        try:
            booked = (
                session.query(Appointment)
                .filter(Appointment.patient_id == 1)
                .order_by(Appointment.id.desc())
                .first()
            )
            when = booked.slot.start_time
        finally:
            session.close()

        turn(
            patient,
            f"I need a dermatology appointment on {when:%B} {when.day}",
            session_id,
        )
        return when

    def test_the_clash_is_named(self, patient):
        when = self._busy_at_eleven(patient, "s-clash-1")
        hour = when.hour % 12 or 12

        result = turn(
            patient,
            f"how about {hour}{'am' if when.hour < 12 else 'pm'} on {when:%B} {when.day}?",
            "s-clash-1",
        )

        assert "clashes with your Cardiology appointment" in result.reply

    def test_the_times_still_follow(self, patient):
        """The sentence is a prefix, not a replacement: the patient asked what
        is free and still needs the answer. Two newlines, because the chat
        renders CommonMark and one would weld the sentence to the first row."""
        when = self._busy_at_eleven(patient, "s-clash-2")
        hour = when.hour % 12 or 12

        result = turn(
            patient,
            f"how about {hour}{'am' if when.hour < 12 else 'pm'} on {when:%B} {when.day}?",
            "s-clash-2",
        )

        assert "Other times that are free:" in result.reply
        assert result.reply.index("clashes") < result.reply.index("Other times")
        assert "that day.\n\n" in result.reply

    def test_a_time_that_is_simply_not_free_claims_no_clash(self, patient):
        """The control, and the direction that matters: this sentence asserts
        something about the patient's own diary. A time nobody is free at must
        never be reported as one they are busy at."""
        when = self._busy_at_eleven(patient, "s-clash-3")

        result = turn(
            patient, f"how about 8pm on {when:%B} {when.day}?", "s-clash-3"
        )

        # The search has to have run, or this passes for the wrong reason: a
        # turn that never looked says nothing about clashes either.
        assert "Other times that are free:" in result.reply
        assert "clashes" not in result.reply


class DriftingChoiceLlm(MockLlm):
    """Proposes the appointment the patient did *not* pick.

    A hint in a typed task is a proposal, and a model may decline it: the task
    carries the chosen row alone and ``gpt-4o-mini`` has already been seen
    ignoring exactly this kind of hint (round 6's ``committed`` field). The
    listed-ids check cannot catch the drift — the other appointment is on the
    list too, by construction — so this exists to falsify the one check that
    can: the patient's answer outranks the model's argument.

    It drifts only on the proposal. The search still runs for the right
    appointment, which is what makes the override's result a *correct* proposal
    rather than a refusal about departments.
    """

    model: str = "drifting-choice-stub"

    def _appointment(self, llm_request, done, task):  # noqa: ANN001
        listed = task.get("listed_appointment_ids") or []
        response = super()._appointment(llm_request, done, task)
        if not listed:
            return response
        for call in response.content.parts if response.content else []:
            if call.function_call and call.function_call.name in (
                "propose_reschedule",
                "propose_cancellation",
            ):
                call.function_call.args["appointment_id"] = listed[-1]
        return response


class TestThePatientsAnswerOutranksTheModelsArgument:
    """The chosen appointment is a decision, not a hint.

    ``_settle_choice`` records which appointment the patient picked and the
    typed task carries that row alone — but a task field is advisory, and the
    listed-ids check under ``_propose_change`` cannot catch a model that
    proposes the *other* listed appointment, because that one was listed too.
    """

    @pytest.fixture(autouse=True)
    def _drifting(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: DriftingChoiceLlm()
        )

    def test_the_chosen_appointment_is_the_one_proposed(self, patient):
        turn(patient, "I need a dermatology appointment for a rash", "s-drift-1")
        turn(patient, "yes", "s-drift-1")
        result = turn(patient, "please reschedule my appointment", "s-drift-1")

        session = fresh()
        try:
            listed = list(
                session.get(WorkflowRun, result.run_id).state["listed_appointment_ids"]
            )
        finally:
            session.close()
        assert len(listed) == 2

        turn(patient, "1", "s-drift-1")

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.proposed_appointment_id == listed[0]
            assert run.proposed_appointment_id != listed[-1]
        finally:
            session.close()

    def test_the_override_is_traced(self, patient):
        """A correction nobody can see is a correction nobody can audit."""
        turn(patient, "I need a dermatology appointment for a rash", "s-drift-2")
        turn(patient, "yes", "s-drift-2")
        turn(patient, "please reschedule my appointment", "s-drift-2")

        result = turn(patient, "1", "s-drift-2")

        session = fresh()
        try:
            targets = [
                event.payload
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .order_by(TraceEvent.seq)
                .all()
                if event.event_type is TraceEventType.VALIDATION
                and event.payload.get("what") == "appointment_target"
            ]
        finally:
            session.close()
        assert targets, "the override left no trace"
        assert targets[-1]["accepted"] is False
        assert targets[-1]["detail"]["reason"] == "chosen"


class TaskTrustingLlm(MockLlm):
    """Works from the appointments the task hands it — and would work from the
    wrong one if the task still offered a choice.

    The twin of :class:`DriftingChoiceLlm`, falsifying the other half. That one
    ignores the task; this one obeys it, so it can only go wrong if the task
    still contains an appointment the patient did not pick. Narrowing the task
    is what makes the *search* right, and the search is where the damage would
    be: a reschedule searched in the wrong appointment's department produces a
    slot the target override then has to refuse, and the turn dead-ends with no
    proposal at all.

    The test that uses it picks position 1, so anything else on the list is the
    row the patient rejected.
    """

    model: str = "task-trusting-stub"

    def _appointment(self, llm_request, done, task):  # noqa: ANN001
        listed = task.get("listed_appointment_ids") or []
        rows = task.get("appointments") or []
        if listed and rows:
            rejected = [row for row in rows if row["appointment_id"] != listed[0]]
            if rejected:
                task = {**task, "appointments": [rejected[0]]}
        return super()._appointment(llm_request, done, task)


class SilentSpecialistLlm(MockLlm):
    """Proposes nothing for a change verb — an empty window, in effect.

    Booking is left alone: the scenario has to *get* two appointments before it
    can ask which one, and a stub silent everywhere books neither.
    """

    model: str = "silent-specialist-stub"

    def _appointment(self, llm_request, done, task):  # noqa: ANN001
        if task.get("step") in ("reschedule", "cancel"):
            return text_response("I've noted that.")
        return super()._appointment(llm_request, done, task)


class TestTheChoiceIsMadeOnce:
    """What the answer buys, past the proposal it produces.

    Two things follow, and neither shows in the happy path: the specialist is
    handed the chosen row *alone*, and a turn that proposes nothing anyway must
    not re-draw the list the patient has already answered.

    The second was briefly deleted for being unfalsifiable, and it was — while
    asking "which one?" still spent the run's only re-plan, the answering turn
    died of the budget before any reply was assembled, so the test written for
    it was passing against a **failure notice** rather than against a re-drawn
    list. Both the guard and its test came back with the budget fix that made
    the turn survivable. A guard is only as testable as the path it sits on.
    """

    def _answered(self, patient, session_id: str) -> tuple[int, list[int]]:
        turn(patient, "I need a dermatology appointment for a rash", session_id)
        turn(patient, "yes", session_id)
        result = turn(patient, "please reschedule my appointment", session_id)
        session = fresh()
        try:
            listed = list(
                session.get(WorkflowRun, result.run_id).state["listed_appointment_ids"]
            )
        finally:
            session.close()
        assert len(listed) == 2
        return result.run_id, listed

    def test_the_specialist_searches_for_the_chosen_appointment(
        self, patient, monkeypatch
    ):
        """With the rejected appointment still in the task, a model that trusts
        its task searches the wrong department — and the target override then
        has to refuse the slot it comes back with, leaving the patient with
        nothing at all."""
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: TaskTrustingLlm()
        )
        run_id, listed = self._answered(patient, "s-once-1")

        turn(patient, "1", "s-once-1")

        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            assert run.status is WorkflowStatus.PENDING_CONFIRMATION
            assert run.proposed_appointment_id == listed[0]
        finally:
            session.close()

    def test_a_turn_that_proposes_nothing_does_not_re_ask(self, patient, monkeypatch):
        """The question has been answered. Re-drawing the list reads as the
        answer having been thrown away, and the patient answers it again."""
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: SilentSpecialistLlm()
        )
        self._answered(patient, "s-once-2")

        result = turn(patient, "1", "s-once-2")

        assert "so I want to be sure which one" not in result.reply
        assert result.reply != FAILED_REPLY, "the run died instead of answering"

    def test_asking_which_one_does_not_spend_the_runs_retry(self, patient, monkeypatch):
        """Live, run 2 of the re-check: the patient answered "1. and please make
        it sometime the week after", the reader took the 1, and the specialist
        read "the week after" as unparseable and stopped without searching. One
        fumble — and no budget left, because drawing the list had spent it. The
        reply to a correct answer was "I couldn't complete this request".

        A step that ends by asking the patient is waiting on them, exactly as a
        proposal waits on a "yes". The turn ends either way; nothing loops."""
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: SilentSpecialistLlm()
        )
        run_id, _ = self._answered(patient, "s-once-3")

        session = fresh()
        try:
            assert session.get(WorkflowRun, run_id).replan_count == 0
        finally:
            session.close()

        result = turn(patient, "1", "s-once-3")

        session = fresh()
        try:
            run = session.get(WorkflowRun, run_id)
            # The fumble does spend it, and the run survives to be asked again.
            assert run.replan_count == 1
            assert run.status is WorkflowStatus.IN_PROGRESS
        finally:
            session.close()
        assert result.reply != FAILED_REPLY
