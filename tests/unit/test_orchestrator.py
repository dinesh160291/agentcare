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
    AppointmentSlot,
    AppointmentStatus,
    AuditEvent,
    Escalation,
    EscalationKind,
    EscalationStatus,
    MessageClass,
    Reminder,
    SlotStatus,
    TraceAuthor,
    TraceEvent,
    TraceEventType,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.orchestrator import SCOPE_REPLY, active_run, run_workflow
from app.workflow.replies import clock_time
from app.providers.base import (
    AgentCareLlm,
    available_tool_names,
    called_tools,
    function_call_response,
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
