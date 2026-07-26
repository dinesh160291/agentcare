"""Staff typed actions: LLM-free, precondition-checked, and lazy-continue.

The path these cover is the one a judge reaches by clicking Approve. What is
pinned:

* **No model is consulted.** A human's decision enters the system as validated
  structured input, like every other consequential change.
* **A resume is a fresh proposal, not a replay.** Preconditions are re-checked
  against current state, and a failed one comes back to staff with the changed
  context rather than landing quietly on stale ground.
* **Approval stops.** It changes state and notifies; the run advances on the
  patient's next message. Humans resume conversations; state changes do not.
* **Nobody approves an emergency.** The two escalation lifecycles cannot be
  crossed, and that is enforced in code rather than in UI copy.
"""

from __future__ import annotations

import asyncio

import pytest

from app.db import SessionLocal
from app.errors import PermissionDenied, RecordNotFound, ValidationFailed
from app.models import (
    Appointment,
    AuditEvent,
    Department,
    Escalation,
    EscalationKind,
    EscalationStatus,
    Notification,
    NotificationKind,
    TraceAuthor,
    TraceEvent,
    TraceEventType,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.orchestrator import run_workflow
from app.trace import TraceWriter
from app.workflow.staff import apply_staff_decision, resolve_escalation

PATIENT_EMAIL = "asha.patient@example.invalid"
STAFF_EMAIL = "staff@example.invalid"
#: The seed's deliberately ambiguous case: Pediatrics or ENT, a human decides.
AMBIGUOUS = "book an appointment, my kid has ear pain"
EMERGENCY = "I have chest pain and my left arm hurts"
SEEDED_APPOINTMENT_ID = 1


@pytest.fixture
def patient(seeded_db):
    seeded_db.commit()
    return seeded_db.query(User).filter(User.email == PATIENT_EMAIL).one()


@pytest.fixture
def staff(seeded_db):
    return seeded_db.query(User).filter(User.email == STAFF_EMAIL).one()


def turn(user, message, session_id):
    return asyncio.run(run_workflow(user, message, session_id))


def fresh():
    return SessionLocal()


def decide(session, staff_user, run_id, action, **kwargs):
    """One staff decision, with the turn's writer supplied by the caller."""
    run = session.get(WorkflowRun, run_id)
    writer = TraceWriter(session, session_id=run.session_id if run else None)
    result = apply_staff_decision(
        session, staff=staff_user, run_id=run_id, action=action, writer=writer, **kwargs
    )
    session.commit()
    return result, writer


@pytest.fixture
def paused(patient, staff):
    """A run sitting in pending_review, awaiting a human."""
    result = turn(patient, AMBIGUOUS, "s-staff-base")
    assert result.status == WorkflowStatus.PENDING_REVIEW.value
    return result


class TestApproval:
    def test_the_run_returns_to_in_progress(self, paused, staff):
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            decision, _ = decide(session, staff_row, paused.run_id, "approve")
        finally:
            session.close()

        assert decision.status == WorkflowStatus.IN_PROGRESS.value
        assert decision.department_name in ("ENT", "Pediatrics")

    def test_the_decision_costs_no_model_call(self, paused, staff):
        """The staff decision path is LLM-free, and the trace proves it."""
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            _, writer = decide(session, staff_row, paused.run_id, "approve")
            events = (
                session.query(TraceEvent)
                .filter(TraceEvent.turn_id == writer.turn_id)
                .all()
            )
            assert not [
                e for e in events if e.event_type is TraceEventType.LLM_REQUEST
            ]
        finally:
            session.close()

    def test_the_turn_is_bracketed_by_a_staff_action(self, paused, staff):
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            _, writer = decide(session, staff_row, paused.run_id, "approve")
            events = (
                session.query(TraceEvent)
                .filter(TraceEvent.turn_id == writer.turn_id)
                .order_by(TraceEvent.seq)
                .all()
            )
        finally:
            session.close()

        assert events[0].event_type is TraceEventType.INBOUND
        assert events[0].author is TraceAuthor.STAFF_ACTION
        assert events[-1].event_type is TraceEventType.OUTBOUND

    def test_the_patient_is_notified(self, paused, staff):
        """A patient whose run moved while they were away learns through the
        in-app panel — the channel already exists."""
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            decision, _ = decide(session, staff_row, paused.run_id, "approve")
            notification = session.get(Notification, decision.notification_id)

            assert notification.kind is NotificationKind.STAFF_DECISION
            assert notification.workflow_run_id == paused.run_id
            assert decision.department_name in notification.title
        finally:
            session.close()

    def test_the_escalation_is_closed_as_approved(self, paused, staff):
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            decide(session, staff_row, paused.run_id, "approve")
            escalation = (
                session.query(Escalation)
                .filter(Escalation.workflow_run_id == paused.run_id)
                .one()
            )
            assert escalation.status is EscalationStatus.APPROVED
            assert escalation.reviewed_by == staff_row.id
        finally:
            session.close()

    def test_the_routing_step_is_marked_done(self, paused, staff):
        """Otherwise the resumed run walks straight back into routing, reaches
        the same ambiguity, and re-escalates into the queue a human has just
        emptied."""
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            decide(session, staff_row, paused.run_id, "approve")
            run = session.get(WorkflowRun, paused.run_id)

            assert "route" in (run.completed_steps or [])
            assert run.state["department_id"] is not None
            assert run.state["routed_by"] == "staff"
        finally:
            session.close()

    def test_nothing_runs_and_nothing_is_booked(self, paused, staff):
        """Lazy-continue. No agent executes, no chat turn is written into an
        empty room, and no non-answer clock starts against a patient who does
        not know a question exists."""
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            decide(session, staff_row, paused.run_id, "approve")
            run = session.get(WorkflowRun, paused.run_id)

            assert run.proposed_slot_id is None
            assert run.non_answer_count == 0
            assert (
                session.query(Appointment)
                .filter(Appointment.patient_id == run.patient_id)
                .count()
                == 1  # the seed's pre-existing one, untouched
            )
        finally:
            session.close()


class TestRedirect:
    def test_staff_may_name_a_different_department(self, paused, staff):
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            decision, _ = decide(
                session,
                staff_row,
                paused.run_id,
                "redirect",
                department_name="Dermatology",
            )
            run = session.get(WorkflowRun, paused.run_id)

            assert decision.department_name == "Dermatology"
            assert run.state["department_name"] == "Dermatology"
            assert run.status is WorkflowStatus.IN_PROGRESS
        finally:
            session.close()

    def test_an_invented_department_is_refused(self, paused, staff):
        """Staff typing is input too, and it is checked against the table."""
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            with pytest.raises(ValidationFailed, match="not a department"):
                decide(
                    session,
                    staff_row,
                    paused.run_id,
                    "redirect",
                    department_name="Cardiothoracic Wizardry",
                )
        finally:
            session.rollback()
            session.close()

    def test_a_redirect_with_no_department_is_refused(self, paused, staff):
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            with pytest.raises(ValidationFailed, match="must name"):
                decide(session, staff_row, paused.run_id, "redirect")
        finally:
            session.rollback()
            session.close()


class TestRejection:
    def test_the_run_reaches_a_terminal_state(self, paused, staff):
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            decision, _ = decide(
                session, staff_row, paused.run_id, "reject", note="Not for this clinic."
            )
            assert decision.status == WorkflowStatus.REJECTED.value
        finally:
            session.close()

    def test_the_patient_learns_about_it(self, paused, staff):
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            decision, _ = decide(
                session, staff_row, paused.run_id, "reject", note="Not for this clinic."
            )
            notification = session.get(Notification, decision.notification_id)
            assert "Not for this clinic." in notification.body
        finally:
            session.close()

    def test_the_escalation_is_closed_as_rejected(self, paused, staff):
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            decide(session, staff_row, paused.run_id, "reject")
            escalation = (
                session.query(Escalation)
                .filter(Escalation.workflow_run_id == paused.run_id)
                .one()
            )
            assert escalation.status is EscalationStatus.REJECTED
        finally:
            session.close()


class TestPreconditionsAreRevalidated:
    """A resume is a fresh proposal, not a replay."""

    def test_a_run_the_patient_has_moved_past_cannot_be_approved(self, paused, staff):
        """Defence in depth, and the test says so.

        Today the ordinary supersede path cancels the old run, so the status
        check refuses this first and the "still the latest?" question never
        arises. This constructs the state that check exists for — two live runs
        for one patient — because a guard nothing can reach is a guard that
        passes whatever it is guarding, and Phase 6 is about to add more
        writers that could break the one-active-run rule.
        """
        session = fresh()
        try:
            from app.workflow.state_machine import create_run

            run = session.get(WorkflowRun, paused.run_id)
            create_run(
                session,
                patient_id=run.patient_id,
                status=WorkflowStatus.IN_PROGRESS,
                trigger="invariant_violation_for_test",
                writer=TraceWriter(session, session_id="s-staff-stale"),
                request_text="a dermatology appointment instead",
                plan=["route", "book"],
                session_id="s-staff-stale",
            )
            session.commit()

            staff_row = session.get(User, staff.id)
            with pytest.raises(ValidationFailed, match="no longer this patient's latest"):
                decide(session, staff_row, paused.run_id, "approve")
        finally:
            session.rollback()
            session.close()

    def test_a_closed_department_cannot_be_redirected_to(self, paused, staff):
        session = fresh()
        try:
            department = (
                session.query(Department).filter(Department.name == "Dermatology").one()
            )
            department.active = False
            session.commit()

            staff_row = session.get(User, staff.id)
            with pytest.raises(ValidationFailed, match="no longer accepting"):
                decide(
                    session,
                    staff_row,
                    paused.run_id,
                    "redirect",
                    department_name="Dermatology",
                )
        finally:
            session.rollback()
            session.close()

    def test_a_refusal_changes_nothing(self, paused, staff):
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            with pytest.raises(ValidationFailed):
                decide(
                    session, staff_row, paused.run_id, "redirect",
                    department_name="Nowhere",
                )
            session.rollback()

            run = session.get(WorkflowRun, paused.run_id)
            assert run.status is WorkflowStatus.PENDING_REVIEW
        finally:
            session.close()

    def test_a_refusal_is_audited(self, paused, staff):
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            with pytest.raises(ValidationFailed):
                decide(session, staff_row, paused.run_id, "redirect")
            # The refusal's own rows are what the caller keeps.
            session.commit()

            assert (
                session.query(AuditEvent)
                .filter(AuditEvent.action == "staff_decision_refused")
                .count()
                == 1
            )
        finally:
            session.close()

    def test_a_run_not_awaiting_review_is_refused(self, patient, staff):
        booked = turn(patient, "I need a cardiology appointment next week", "s-staff-np")

        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            with pytest.raises(ValidationFailed, match="not awaiting review"):
                decide(session, staff_row, booked.run_id, "approve")
        finally:
            session.rollback()
            session.close()


class TestRbacIsEnforcedInTheBackend:
    def test_a_patient_cannot_decide_their_own_review(self, paused, patient):
        session = fresh()
        try:
            patient_row = session.get(User, patient.id)
            with pytest.raises(PermissionDenied):
                decide(session, patient_row, paused.run_id, "approve")
        finally:
            session.rollback()
            session.close()

    def test_an_unknown_run_is_a_404_not_a_confirmation(self, staff):
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            with pytest.raises(RecordNotFound):
                decide(session, staff_row, 999_999, "approve")
        finally:
            session.rollback()
            session.close()

    def test_an_invented_action_is_refused(self, paused, staff):
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            with pytest.raises(ValidationFailed, match="not a staff decision"):
                decide(session, staff_row, paused.run_id, "do whatever seems best")
        finally:
            session.rollback()
            session.close()


class TestNobodyApprovesAnEmergency:
    """The two lifecycles are kept apart in code, not in UI copy. Review
    escalations are approved or rejected; safety ones are acknowledged and
    resolved."""

    def test_a_safety_run_is_not_reachable_by_the_review_queue_at_all(
        self, patient, staff
    ):
        """The first line of the guarantee is the state machine: a safety
        verdict lands a run in ``escalated``, which is terminal, so it is never
        sitting in the review queue to be approved from."""
        result = turn(patient, EMERGENCY, "s-staff-safety")

        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            with pytest.raises(ValidationFailed, match="not awaiting review"):
                decide(session, staff_row, result.run_id, "approve")
        finally:
            session.rollback()
            session.close()

    def test_a_safety_escalation_on_a_review_run_is_still_refused(
        self, paused, staff
    ):
        """Second line, constructed on purpose.

        The state machine makes this unreachable today. It is tested anyway
        because the phrase this prevents — "your chest pain was approved" —
        is the one no interface may ever be able to produce, and a check that
        cannot fail is not evidence that it holds.
        """
        session = fresh()
        try:
            escalation = (
                session.query(Escalation)
                .filter(Escalation.workflow_run_id == paused.run_id)
                .one()
            )
            escalation.kind = EscalationKind.SAFETY
            session.commit()

            staff_row = session.get(User, staff.id)
            with pytest.raises(ValidationFailed, match="acknowledged and resolved"):
                decide(session, staff_row, paused.run_id, "approve")

            run = session.get(WorkflowRun, paused.run_id)
            assert run.status is WorkflowStatus.PENDING_REVIEW
        finally:
            session.rollback()
            session.close()

    def test_a_safety_escalation_is_acknowledged_then_resolved(self, patient, staff):
        turn(patient, EMERGENCY, "s-staff-safety-2")

        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            escalation = session.query(Escalation).one()

            acknowledged = resolve_escalation(
                session,
                staff=staff_row,
                escalation_id=escalation.id,
                status=EscalationStatus.ACKNOWLEDGED,
            )
            assert acknowledged["status"] == "acknowledged"

            resolved = resolve_escalation(
                session,
                staff=staff_row,
                escalation_id=escalation.id,
                status=EscalationStatus.RESOLVED,
                note="Patient was called and directed to A&E.",
            )
            session.commit()
            assert resolved["status"] == "resolved"
        finally:
            session.close()

    def test_a_review_escalation_cannot_be_acknowledged(self, paused, staff):
        """The mirror rule: review escalations do not use the safety words."""
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            escalation = (
                session.query(Escalation)
                .filter(Escalation.workflow_run_id == paused.run_id)
                .one()
            )
            assert escalation.kind is EscalationKind.LOW_CONFIDENCE_ROUTING

            with pytest.raises(ValidationFailed, match="cannot be set to"):
                resolve_escalation(
                    session,
                    staff=staff_row,
                    escalation_id=escalation.id,
                    status=EscalationStatus.ACKNOWLEDGED,
                )
        finally:
            session.rollback()
            session.close()


class TestLazyContinue:
    """The patient's return is the inbound that wakes the run."""

    def test_the_patient_resumes_where_staff_left_it(self, paused, staff):
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            decision, _ = decide(session, staff_row, paused.run_id, "approve")
        finally:
            session.close()

        # The patient comes back. Routing is settled, so the run picks up at
        # the booking step rather than re-deciding the department.
        patient_row = None
        session = fresh()
        try:
            run = session.get(WorkflowRun, paused.run_id)
            from app.models import PatientProfile

            profile = session.get(PatientProfile, run.patient_id)
            patient_row = session.get(User, profile.user_id)
        finally:
            session.close()

        result = turn(patient_row, "next week works for the appointment", "s-staff-base")

        assert result.run_id == paused.run_id
        assert result.steps_run == ["book"]
        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value

    def test_the_slot_search_happens_while_the_patient_is_present(self, paused, staff):
        """Which is why lazy-continue composes with "a resume is a fresh
        proposal" rather than fighting it: the times offered are current by
        construction, not as of whenever staff happened to click."""
        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            decide(session, staff_row, paused.run_id, "approve")
            run = session.get(WorkflowRun, paused.run_id)
            assert run.proposed_slot_id is None  # nothing offered yet
            from app.models import PatientProfile

            profile = session.get(PatientProfile, run.patient_id)
            patient_row = session.get(User, profile.user_id)
        finally:
            session.close()

        turn(patient_row, "next week works for the appointment", "s-staff-base")

        session = fresh()
        try:
            run = session.get(WorkflowRun, paused.run_id)
            assert run.proposed_slot_id is not None  # offered now, on their return
        finally:
            session.close()
