"""Reschedule and cancel: the two verbs that had no path.

The tools were built and tested in Phase 2 and nothing ever called them —
``_commit_proposal`` hard-coded ``book_appointment`` and never read
``run.proposed_action``, so a proposal to cancel would have committed a
booking. That is the defect this module was written against, and the first
class below is deliberately the narrowest possible statement of it: a typed
proposal, a Confirm, and an assertion about the row.

The rest is the same discipline the booking path already carries, applied to
two verbs that never had it:

* **confirm-before-commit** — nothing mutates until the patient says so;
* **the same exit from every commit failure** — proposal cleared, back to
  selection, a second Confirm a calm no-op;
* **the derivation invariant** — reminders move inside the source transaction,
  so no window exists where a reminder points at an appointment that moved.
"""

from __future__ import annotations

import asyncio

import pytest

from app.db import SessionLocal
from app.models import (
    Appointment,
    AppointmentSlot,
    AppointmentStatus,
    PlanStep,
    ProposedAction,
    Reminder,
    ReminderStatus,
    SlotStatus,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.orchestrator import apply_patient_action, run_workflow

PATIENT_EMAIL = "asha.patient@example.invalid"
#: The seed ships exactly one booked appointment, for Asha, in Cardiology.
SEEDED_APPOINTMENT_ID = 1


@pytest.fixture
def patient(seeded_db):
    seeded_db.commit()
    return seeded_db.query(User).filter(User.email == PATIENT_EMAIL).one()


def fresh():
    return SessionLocal()


def press(user, action, session_id):
    return asyncio.run(apply_patient_action(user, action, session_id))


def turn(user, message, session_id):
    return asyncio.run(run_workflow(user, message, session_id))


def free_slot_in_cardiology(session, *, exclude_id: int | None = None) -> AppointmentSlot:
    """Any bookable Cardiology slot other than the one already taken."""
    from app.models import Department, Doctor

    query = (
        session.query(AppointmentSlot)
        .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
        .join(Department, Department.id == Doctor.department_id)
        .filter(
            Department.name == "Cardiology",
            AppointmentSlot.status == SlotStatus.AVAILABLE,
        )
        .order_by(AppointmentSlot.start_time)
    )
    if exclude_id is not None:
        query = query.filter(AppointmentSlot.id != exclude_id)
    return query.first()


def pending_run(
    session,
    *,
    action: ProposedAction,
    step: PlanStep,
    appointment_id: int,
    slot_id: int | None = None,
    session_id: str,
) -> int:
    """A run parked in pending_confirmation with a typed proposal on it.

    Built directly rather than driven through a conversation, so that what is
    under test is the commit dispatch and nothing else. The conversational
    route to the same state is covered by the scenarios.
    """
    run = WorkflowRun(
        patient_id=1,
        status=WorkflowStatus.PENDING_CONFIRMATION,
        plan=[step.value],
        completed_steps=[],
        state={},
        request_text="",
        proposed_action=action,
        proposed_appointment_id=appointment_id,
        proposed_slot_id=slot_id,
        session_id=session_id,
    )
    session.add(run)
    session.commit()
    return run.id


class TestTheCommitDispatchesOnTheProposedAction:
    """The dead path, stated as narrowly as it can be stated."""

    def test_confirming_a_cancellation_cancels(self, patient, seeded_db):
        pending_run(
            seeded_db,
            action=ProposedAction.CANCEL,
            step=PlanStep.CANCEL,
            appointment_id=SEEDED_APPOINTMENT_ID,
            session_id="s-cancel-1",
        )
        press(patient, "confirm", "s-cancel-1")

        seeded_db.expire_all()
        appointment = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID)
        assert appointment.status is AppointmentStatus.CANCELLED

    def test_the_dispatch_reads_the_action_not_whichever_field_is_populated(
        self, patient, seeded_db
    ):
        """A cancel proposal that also carries a slot id must still cancel.

        Written this way after a falsification pass showed the obvious version
        of this test — "confirming a cancellation books nothing" — could not
        fail. Removing the cancel branch made the code fall through to
        ``book_appointment(slot_id=None)``, which refuses, so no appointment
        was created and the assertion passed while the feature was broken.

        A slot id on the proposal is the state that tells the two apart: if the
        commit dispatched on "is there a slot?" rather than on the action, this
        would book. It is artificial state on purpose — the guard is about what
        the code *keys on*, and only artificial state can separate the two.
        """
        before = seeded_db.query(Appointment).count()
        target = free_slot_in_cardiology(seeded_db)
        target_id = target.id
        pending_run(
            seeded_db,
            action=ProposedAction.CANCEL,
            step=PlanStep.CANCEL,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=target_id,
            session_id="s-cancel-2",
        )
        press(patient, "confirm", "s-cancel-2")

        seeded_db.expire_all()
        assert seeded_db.query(Appointment).count() == before, "a cancel booked something"
        assert (
            seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).status
            is AppointmentStatus.CANCELLED
        )
        assert seeded_db.get(AppointmentSlot, target_id).status is SlotStatus.AVAILABLE

    def test_confirming_a_reschedule_moves_the_appointment(self, patient, seeded_db):
        original = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID)
        original_slot_id = original.slot_id
        target = free_slot_in_cardiology(seeded_db, exclude_id=original_slot_id)

        pending_run(
            seeded_db,
            action=ProposedAction.RESCHEDULE,
            step=PlanStep.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=target.id,
            session_id="s-resched-1",
        )
        target_id = target.id
        press(patient, "confirm", "s-resched-1")

        seeded_db.expire_all()
        appointment = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID)
        assert appointment.slot_id == target_id
        assert appointment.status is AppointmentStatus.CONFIRMED

    def test_a_reschedule_releases_the_slot_it_left(self, patient, seeded_db):
        original_slot_id = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).slot_id
        target = free_slot_in_cardiology(seeded_db, exclude_id=original_slot_id)
        pending_run(
            seeded_db,
            action=ProposedAction.RESCHEDULE,
            step=PlanStep.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=target.id,
            session_id="s-resched-2",
        )
        target_id = target.id
        press(patient, "confirm", "s-resched-2")

        seeded_db.expire_all()
        assert seeded_db.get(AppointmentSlot, original_slot_id).status is SlotStatus.AVAILABLE
        assert seeded_db.get(AppointmentSlot, target_id).status is SlotStatus.BOOKED


class TestNothingMutatesBeforeTheConfirmation:
    def test_a_pending_cancellation_has_not_cancelled_anything(self, patient, seeded_db):
        pending_run(
            seeded_db,
            action=ProposedAction.CANCEL,
            step=PlanStep.CANCEL,
            appointment_id=SEEDED_APPOINTMENT_ID,
            session_id="s-pending-1",
        )
        seeded_db.expire_all()
        assert (
            seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).status
            is AppointmentStatus.CONFIRMED
        )

    def test_declining_a_cancellation_leaves_the_appointment_alone(
        self, patient, seeded_db
    ):
        run_id = pending_run(
            seeded_db,
            action=ProposedAction.CANCEL,
            step=PlanStep.CANCEL,
            appointment_id=SEEDED_APPOINTMENT_ID,
            session_id="s-decline-1",
        )
        press(patient, "decline", "s-decline-1")

        seeded_db.expire_all()
        assert (
            seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).status
            is AppointmentStatus.CONFIRMED
        )
        run = seeded_db.get(WorkflowRun, run_id)
        assert run.proposed_action is None
        assert run.status is WorkflowStatus.IN_PROGRESS


class TestTheDerivationInvariant:
    """Reminders belong to the appointment and move in its transaction — or the
    reminder channel, which is pure code, delivers a fact frozen at booking
    time after the appointment moved."""

    def test_cancelling_retires_the_reminder(self, patient, seeded_db):
        pending_run(
            seeded_db,
            action=ProposedAction.CANCEL,
            step=PlanStep.CANCEL,
            appointment_id=SEEDED_APPOINTMENT_ID,
            session_id="s-rem-1",
        )
        press(patient, "confirm", "s-rem-1")

        seeded_db.expire_all()
        live = (
            seeded_db.query(Reminder)
            .filter(
                Reminder.appointment_id == SEEDED_APPOINTMENT_ID,
                Reminder.status == ReminderStatus.PENDING,
            )
            .count()
        )
        assert live == 0

    def test_rescheduling_retimes_the_reminder(self, patient, seeded_db):
        original_slot_id = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).slot_id
        target = free_slot_in_cardiology(seeded_db, exclude_id=original_slot_id)
        target_start = target.start_time
        pending_run(
            seeded_db,
            action=ProposedAction.RESCHEDULE,
            step=PlanStep.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=target.id,
            session_id="s-rem-2",
        )
        press(patient, "confirm", "s-rem-2")

        seeded_db.expire_all()
        pending = (
            seeded_db.query(Reminder)
            .filter(
                Reminder.appointment_id == SEEDED_APPOINTMENT_ID,
                Reminder.status == ReminderStatus.PENDING,
            )
            .all()
        )
        assert len(pending) == 1
        # 24 hours ahead of the *new* time, not the old one.
        assert pending[0].scheduled_at < target_start
        assert (target_start - pending[0].scheduled_at).days == 1


class TestCommitFailureExitsTheSameWayForEveryVerb:
    """PRD: "every commit-time failure exits the same way: clear the proposal,
    return to selection, offer alternatives" — for book, reschedule, and cancel
    alike. A proposal pointing at a dead target that stays confirmable is a
    loop with no exit."""

    def test_a_slot_taken_between_proposal_and_confirm_clears_the_proposal(
        self, patient, seeded_db
    ):
        original_slot_id = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).slot_id
        target = free_slot_in_cardiology(seeded_db, exclude_id=original_slot_id)
        target_id = target.id
        run_id = pending_run(
            seeded_db,
            action=ProposedAction.RESCHEDULE,
            step=PlanStep.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=target_id,
            session_id="s-sabotage-1",
        )

        # Somebody else takes the slot while the patient is deciding.
        seeded_db.get(AppointmentSlot, target_id).status = SlotStatus.BOOKED
        seeded_db.commit()

        press(patient, "confirm", "s-sabotage-1")

        seeded_db.expire_all()
        run = seeded_db.get(WorkflowRun, run_id)
        assert run.proposed_action is None, "a dead proposal stayed confirmable"
        assert run.status is WorkflowStatus.IN_PROGRESS
        # The appointment is untouched: a failed reschedule leaves the patient
        # holding what they already had.
        assert seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).slot_id == original_slot_id

    def test_a_second_confirm_after_a_failed_reschedule_is_a_no_op(
        self, patient, seeded_db
    ):
        original_slot_id = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).slot_id
        target = free_slot_in_cardiology(seeded_db, exclude_id=original_slot_id)
        target_id = target.id
        pending_run(
            seeded_db,
            action=ProposedAction.RESCHEDULE,
            step=PlanStep.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=target_id,
            session_id="s-sabotage-2",
        )
        seeded_db.get(AppointmentSlot, target_id).status = SlotStatus.BOOKED
        seeded_db.commit()

        press(patient, "confirm", "s-sabotage-2")
        second = press(patient, "confirm", "s-sabotage-2")

        seeded_db.expire_all()
        assert seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).slot_id == original_slot_id
        assert second.reply

    def test_cancelling_an_already_cancelled_appointment_refuses_cleanly(
        self, patient, seeded_db
    ):
        appointment = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID)
        appointment.status = AppointmentStatus.CANCELLED
        seeded_db.commit()

        run_id = pending_run(
            seeded_db,
            action=ProposedAction.CANCEL,
            step=PlanStep.CANCEL,
            appointment_id=SEEDED_APPOINTMENT_ID,
            session_id="s-double-cancel",
        )
        result = press(patient, "confirm", "s-double-cancel")

        seeded_db.expire_all()
        assert seeded_db.get(WorkflowRun, run_id).proposed_action is None
        assert result.reply
