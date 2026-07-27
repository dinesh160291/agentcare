"""Booking, rescheduling, and cancellation.

Three rules are pinned here, and each exists because of a specific way this
goes wrong:

* **The slot is re-checked inside the commit transaction.** Availability was
  answered when the slot was offered; between offering and confirming, someone
  else may have taken it.
* **A commit failure clears the proposal.** A proposal pointing at a dead slot
  that stays confirmable is a loop with no exit — the patient presses Confirm
  forever.
* **Reminders update in their appointment's transaction.** A reminder that
  survives a cancellation tells a patient to attend an appointment that no
  longer exists.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from app import clock
from app.db import SessionLocal
from app.models import (
    Appointment,
    AppointmentSlot,
    AppointmentStatus,
    AuditEvent,
    Reminder,
    ReminderStatus,
    SlotStatus,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.tools.appointments import (
    book_appointment,
    cancel_appointment,
    reschedule_appointment,
)
from app.tools.availability import find_available_slots

MONDAY = date(2026, 8, 3)


@pytest.fixture(autouse=True)
def world(db):
    """Seeded world with the clock pinned inside the slot window."""
    clock.freeze(datetime(2026, 8, 3, 8, 0))
    from scripts.seed import seed

    seed(db, anchor=MONDAY)
    db.commit()
    return db


@pytest.fixture
def patient_b(db):
    """Patient 2 — no seeded appointments, so their state is all ours."""
    return db.query(User).filter(User.id == 2).one()


def free_slot(db, department_id: int = 1, index: int = 0) -> int:
    return find_available_slots(db, department_id=department_id, limit=index + 1)["slots"][index][
        "slot_id"
    ]


def remindable_slot(db, department_id: int = 1, index: int = 0) -> int:
    """A free slot far enough out to *have* a day before it.

    The earliest free slot is today — the seed lays them from today at 09:00
    and the search only returns future ones — so a same-day booking is what
    ``free_slot`` returns for most of the working day. That is a real state
    with its own rule (no reminder is scheduled, because there is no day
    before to send it on), and a test about reminder derivation that silently
    lands in it is testing the other rule by accident.
    """
    slots = find_available_slots(db, department_id=department_id, limit=200)["slots"]
    cutoff = clock.now() + timedelta(hours=24)
    later = [s for s in slots if datetime.fromisoformat(s["start"]) > cutoff]
    assert len(later) > index, "the seed has no slot more than a day out"
    return later[index]["slot_id"]


class TestBooking:
    def test_a_booking_creates_a_confirmed_appointment(self, db, patient_b):
        slot_id = free_slot(db)
        result = book_appointment(db, patient_b, slot_id=slot_id, reason="follow-up")

        assert result["ok"] is True
        appointment = db.get(Appointment, result["appointment"]["appointment_id"])
        assert appointment.status == AppointmentStatus.CONFIRMED
        assert appointment.patient_id == 2

    def test_the_slot_is_marked_booked(self, db, patient_b):
        slot_id = free_slot(db)
        book_appointment(db, patient_b, slot_id=slot_id, reason="follow-up")
        assert db.get(AppointmentSlot, slot_id).status == SlotStatus.BOOKED

    def test_a_reminder_is_derived_from_the_appointment(self, db, patient_b):
        slot_id = remindable_slot(db)
        result = book_appointment(db, patient_b, slot_id=slot_id, reason="follow-up")

        reminder = (
            db.query(Reminder)
            .filter(Reminder.appointment_id == result["appointment"]["appointment_id"])
            .one()
        )
        slot = db.get(AppointmentSlot, slot_id)
        assert reminder.scheduled_at == slot.start_time - timedelta(hours=24)
        assert reminder.status == ReminderStatus.PENDING

    def test_the_appointment_gets_a_reference_code(self, db, patient_b):
        """Confirmations quote this, so it has to exist before the reply does."""
        slot_id = free_slot(db)
        result = book_appointment(db, patient_b, slot_id=slot_id, reason="follow-up")
        assert result["appointment"]["reference_code"]

    def test_booking_writes_an_audit_event(self, db, patient_b):
        slot_id = free_slot(db)
        book_appointment(db, patient_b, slot_id=slot_id, reason="follow-up")
        actions = {e.action for e in db.query(AuditEvent).all()}
        assert "appointment_booked" in actions

    def test_the_department_is_taken_from_the_slot_not_the_caller(self, db, patient_b):
        """Trusting a caller-supplied department would let a Cardiology slot be
        filed under Dermatology, and every later diff would read the wrong
        required-documents rules."""
        slot_id = free_slot(db, department_id=2)
        result = book_appointment(db, patient_b, slot_id=slot_id, reason="x")
        assert result["appointment"]["department_id"] == 2


class TestBookingRefusals:
    def test_a_slot_taken_since_it_was_offered_is_refused(self, db, patient_b):
        """The race the re-check exists for."""
        slot_id = free_slot(db)
        db.get(AppointmentSlot, slot_id).status = SlotStatus.BOOKED
        db.commit()

        result = book_appointment(db, patient_b, slot_id=slot_id, reason="follow-up")
        assert result["ok"] is False
        assert result["reason"] == "slot_taken"

    def test_a_refusal_offers_alternatives(self, db, patient_b):
        """Returning "no" without a next step strands the patient."""
        slot_id = free_slot(db)
        db.get(AppointmentSlot, slot_id).status = SlotStatus.BOOKED
        db.commit()

        result = book_appointment(db, patient_b, slot_id=slot_id, reason="follow-up")
        assert result["alternatives"]

    def test_a_refusal_creates_no_appointment(self, db, patient_b):
        slot_id = free_slot(db)
        db.get(AppointmentSlot, slot_id).status = SlotStatus.BOOKED
        db.commit()
        before = db.query(Appointment).count()

        book_appointment(db, patient_b, slot_id=slot_id, reason="follow-up")
        assert db.query(Appointment).count() == before

    def test_a_slot_whose_doctor_is_inactive_is_refused(self, db, patient_b):
        """Availability filters inactive doctors out, but a slot id can be
        booked directly — from a stale proposal, a retry, or the API — and that
        path never passes through the availability filter."""
        from app.models import Doctor

        slot_id = free_slot(db)
        db.get(Doctor, db.get(AppointmentSlot, slot_id).doctor_id).active = False
        db.commit()

        result = book_appointment(db, patient_b, slot_id=slot_id, reason="x")
        assert result["ok"] is False
        assert result["reason"] == "doctor_unavailable"

    def test_an_inactive_doctors_slot_is_left_unclaimed(self, db, patient_b):
        from app.models import Doctor

        slot_id = free_slot(db)
        db.get(Doctor, db.get(AppointmentSlot, slot_id).doctor_id).active = False
        db.commit()

        book_appointment(db, patient_b, slot_id=slot_id, reason="x")
        assert db.get(AppointmentSlot, slot_id).status == SlotStatus.AVAILABLE

    def test_a_slot_in_an_inactive_department_is_refused(self, db, patient_b):
        """Same hole, one level up: deactivating a department must stop new
        bookings immediately, not only stop them being suggested."""
        from app.models import Department, Doctor

        slot_id = free_slot(db)
        doctor = db.get(Doctor, db.get(AppointmentSlot, slot_id).doctor_id)
        db.get(Department, doctor.department_id).active = False
        db.commit()

        assert book_appointment(db, patient_b, slot_id=slot_id, reason="x")["ok"] is False

    def test_a_missing_slot_is_refused(self, db, patient_b):
        result = book_appointment(db, patient_b, slot_id=999_999, reason="x")
        assert result["ok"] is False
        assert result["reason"] == "slot_not_found"

    def test_a_past_slot_is_refused(self, db, patient_b):
        slot_id = free_slot(db)
        clock.freeze(datetime(2026, 8, 20, 8, 0))
        result = book_appointment(db, patient_b, slot_id=slot_id, reason="x")
        assert result["ok"] is False
        assert result["reason"] == "slot_in_the_past"

    def test_double_booking_the_same_time_is_refused(self, db, patient_b):
        """Two appointments at once is a conflict whoever the doctors are."""
        first = free_slot(db, department_id=1)
        book_appointment(db, patient_b, slot_id=first, reason="one")

        start = db.get(AppointmentSlot, first).start_time
        clash = (
            db.query(AppointmentSlot)
            .filter(
                AppointmentSlot.start_time == start,
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                AppointmentSlot.id != first,
            )
            .first()
        )
        result = book_appointment(db, patient_b, slot_id=clash.id, reason="two")
        assert result["ok"] is False
        assert result["reason"] == "patient_double_booked"

    def test_the_clashing_slot_is_left_available(self, db, patient_b):
        first = free_slot(db, department_id=1)
        book_appointment(db, patient_b, slot_id=first, reason="one")
        start = db.get(AppointmentSlot, first).start_time
        clash = (
            db.query(AppointmentSlot)
            .filter(
                AppointmentSlot.start_time == start,
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                AppointmentSlot.id != first,
            )
            .first()
        )
        book_appointment(db, patient_b, slot_id=clash.id, reason="two")
        assert db.get(AppointmentSlot, clash.id).status == SlotStatus.AVAILABLE


class TestProposalIsClearedOnFailure:
    def test_a_failed_commit_clears_the_pending_proposal(self, db, patient_b):
        """Otherwise the patient can press Confirm on a dead slot forever."""
        slot_id = free_slot(db)
        run = WorkflowRun(
            patient_id=2,
            status=WorkflowStatus.PENDING_CONFIRMATION,
            proposed_slot_id=slot_id,
        )
        db.add(run)
        db.commit()

        db.get(AppointmentSlot, slot_id).status = SlotStatus.BOOKED
        db.commit()

        book_appointment(db, patient_b, slot_id=slot_id, reason="x", run=run)

        db.refresh(run)
        assert run.proposed_slot_id is None
        assert run.proposed_action is None

    def test_a_cleared_proposal_survives_a_rollback(self, db, patient_b):
        """Clearing must be *durable*, not merely pending.

        A failure path that only flushes leaves the cleared proposal inside an
        uncommitted transaction. Anything that rolls back afterwards — an error
        handler, a request teardown, a retry — restores the proposal pointing
        at the dead slot, which is precisely the un-exitable loop the clearing
        exists to prevent. Re-read through a fresh session, because the calling
        session would show its own uncommitted state either way.
        """
        from app.db import SessionLocal
        from app.models import ProposedAction

        slot_id = free_slot(db)
        run = WorkflowRun(
            patient_id=2,
            status=WorkflowStatus.PENDING_CONFIRMATION,
            proposed_action=ProposedAction.BOOK,
            proposed_slot_id=slot_id,
        )
        db.add(run)
        db.commit()
        run_id = run.id

        db.get(AppointmentSlot, slot_id).status = SlotStatus.BOOKED
        db.commit()

        assert book_appointment(db, patient_b, slot_id=slot_id, reason="x", run=run)["ok"] is False

        db.rollback()

        fresh = SessionLocal()
        try:
            reloaded = fresh.get(WorkflowRun, run_id)
            assert reloaded.proposed_action is None
            assert reloaded.proposed_slot_id is None
        finally:
            fresh.close()

    def test_a_successful_commit_also_clears_the_proposal(self, db, patient_b):
        slot_id = free_slot(db)
        run = WorkflowRun(
            patient_id=2,
            status=WorkflowStatus.PENDING_CONFIRMATION,
            proposed_slot_id=slot_id,
        )
        db.add(run)
        db.commit()

        book_appointment(db, patient_b, slot_id=slot_id, reason="x", run=run)
        db.refresh(run)
        assert run.proposed_slot_id is None


class TestRescheduling:
    def test_rescheduling_moves_the_appointment(self, db, patient_b):
        original = remindable_slot(db)
        booked = book_appointment(db, patient_b, slot_id=original, reason="x")
        appointment_id = booked["appointment"]["appointment_id"]
        target = remindable_slot(db, index=3)

        result = reschedule_appointment(db, patient_b, appointment_id, new_slot_id=target)
        assert result["ok"] is True
        assert db.get(Appointment, appointment_id).slot_id == target

    def test_the_old_slot_is_released(self, db, patient_b):
        original = free_slot(db)
        booked = book_appointment(db, patient_b, slot_id=original, reason="x")
        target = free_slot(db, index=3)

        reschedule_appointment(db, patient_b, booked["appointment"]["appointment_id"],
                               new_slot_id=target)
        assert db.get(AppointmentSlot, original).status == SlotStatus.AVAILABLE

    def test_reminders_follow_the_appointment(self, db, patient_b):
        """The derivation invariant. A reminder still pointing at the old time
        would tell the patient to turn up on the wrong day."""
        original = remindable_slot(db)
        booked = book_appointment(db, patient_b, slot_id=original, reason="x")
        appointment_id = booked["appointment"]["appointment_id"]
        target = remindable_slot(db, index=3)

        reschedule_appointment(db, patient_b, appointment_id, new_slot_id=target)

        reminders = (
            db.query(Reminder)
            .filter(
                Reminder.appointment_id == appointment_id,
                Reminder.status == ReminderStatus.PENDING,
            )
            .all()
        )
        new_start = db.get(AppointmentSlot, target).start_time
        assert len(reminders) == 1
        assert reminders[0].scheduled_at == new_start - timedelta(hours=24)

    def test_rescheduling_onto_a_taken_slot_is_refused(self, db, patient_b):
        original = free_slot(db)
        booked = book_appointment(db, patient_b, slot_id=original, reason="x")
        target = free_slot(db, index=3)
        db.get(AppointmentSlot, target).status = SlotStatus.BOOKED
        db.commit()

        result = reschedule_appointment(
            db, patient_b, booked["appointment"]["appointment_id"], new_slot_id=target
        )
        assert result["ok"] is False
        assert result["reason"] == "slot_taken"

    def test_a_refused_reschedule_leaves_the_original_intact(self, db, patient_b):
        """A half-applied move is worse than a refused one."""
        original = remindable_slot(db)
        booked = book_appointment(db, patient_b, slot_id=original, reason="x")
        appointment_id = booked["appointment"]["appointment_id"]
        target = remindable_slot(db, index=3)
        db.get(AppointmentSlot, target).status = SlotStatus.BOOKED
        db.commit()

        reschedule_appointment(db, patient_b, appointment_id, new_slot_id=target)

        assert db.get(Appointment, appointment_id).slot_id == original
        assert db.get(AppointmentSlot, original).status == SlotStatus.BOOKED

    def test_another_patients_appointment_cannot_be_rescheduled(self, db, patient_b):
        """Appointment 1 belongs to patient 1."""
        from app.errors import RecordNotFound

        with pytest.raises(RecordNotFound):
            reschedule_appointment(db, patient_b, 1, new_slot_id=free_slot(db))


class TestReschedulingRefusals:
    def test_rescheduling_a_cancelled_appointment_is_refused(self, db, patient_b):
        slot_id = free_slot(db)
        booked = book_appointment(db, patient_b, slot_id=slot_id, reason="x")
        appointment_id = booked["appointment"]["appointment_id"]
        cancel_appointment(db, patient_b, appointment_id)

        result = reschedule_appointment(db, patient_b, appointment_id,
                                        new_slot_id=free_slot(db, index=2))
        assert result["ok"] is False
        assert result["reason"] == "not_reschedulable"

    def test_rescheduling_onto_a_missing_slot_is_refused(self, db, patient_b):
        booked = book_appointment(db, patient_b, slot_id=free_slot(db), reason="x")
        result = reschedule_appointment(
            db, patient_b, booked["appointment"]["appointment_id"], new_slot_id=999_999
        )
        assert result["ok"] is False
        assert result["reason"] == "slot_not_found"

    def test_rescheduling_onto_an_inactive_doctors_slot_is_refused(self, db, patient_b):
        """The same bypass as booking: a slot id supplied directly never passes
        through the availability filter."""
        from app.models import Doctor

        booked = book_appointment(db, patient_b, slot_id=free_slot(db), reason="x")
        target = free_slot(db, index=4)
        db.get(Doctor, db.get(AppointmentSlot, target).doctor_id).active = False
        db.commit()

        result = reschedule_appointment(
            db, patient_b, booked["appointment"]["appointment_id"], new_slot_id=target
        )
        assert result["ok"] is False
        assert result["reason"] == "doctor_unavailable"

    def test_rescheduling_into_the_past_is_refused(self, db, patient_b):
        booked = book_appointment(db, patient_b, slot_id=free_slot(db), reason="x")
        target = free_slot(db, index=4)
        clock.freeze(datetime(2026, 8, 20, 8, 0))
        result = reschedule_appointment(
            db, patient_b, booked["appointment"]["appointment_id"], new_slot_id=target
        )
        assert result["ok"] is False
        assert result["reason"] == "slot_in_the_past"


class TestCancellation:
    def test_cancelling_marks_the_appointment_cancelled(self, db, patient_b):
        slot_id = free_slot(db)
        booked = book_appointment(db, patient_b, slot_id=slot_id, reason="x")
        appointment_id = booked["appointment"]["appointment_id"]

        result = cancel_appointment(db, patient_b, appointment_id)
        assert result["ok"] is True
        assert db.get(Appointment, appointment_id).status == AppointmentStatus.CANCELLED

    def test_the_slot_returns_to_the_pool(self, db, patient_b):
        slot_id = free_slot(db)
        booked = book_appointment(db, patient_b, slot_id=slot_id, reason="x")
        cancel_appointment(db, patient_b, booked["appointment"]["appointment_id"])
        assert db.get(AppointmentSlot, slot_id).status == SlotStatus.AVAILABLE

    def test_pending_reminders_are_cancelled_with_it(self, db, patient_b):
        """A reminder outliving its appointment tells the patient to attend
        something that no longer exists."""
        slot_id = remindable_slot(db)
        booked = book_appointment(db, patient_b, slot_id=slot_id, reason="x")
        appointment_id = booked["appointment"]["appointment_id"]

        cancel_appointment(db, patient_b, appointment_id)

        statuses = {
            r.status
            for r in db.query(Reminder).filter(Reminder.appointment_id == appointment_id).all()
        }
        assert statuses == {ReminderStatus.CANCELLED}

    def test_cancelling_twice_is_refused_rather_than_repeated(self, db, patient_b):
        slot_id = free_slot(db)
        booked = book_appointment(db, patient_b, slot_id=slot_id, reason="x")
        appointment_id = booked["appointment"]["appointment_id"]

        cancel_appointment(db, patient_b, appointment_id)
        second = cancel_appointment(db, patient_b, appointment_id)
        assert second["ok"] is False
        assert second["reason"] == "not_cancellable"

    def test_a_double_cancel_does_not_re_release_the_slot(self, db, patient_b):
        """If someone rebooked the freed slot, a second cancel must not hand it
        back out from under them."""
        slot_id = free_slot(db)
        booked = book_appointment(db, patient_b, slot_id=slot_id, reason="x")
        appointment_id = booked["appointment"]["appointment_id"]
        cancel_appointment(db, patient_b, appointment_id)

        db.get(AppointmentSlot, slot_id).status = SlotStatus.BOOKED
        db.commit()

        cancel_appointment(db, patient_b, appointment_id)
        assert db.get(AppointmentSlot, slot_id).status == SlotStatus.BOOKED

    def test_another_patients_appointment_cannot_be_cancelled(self, db, patient_b):
        from app.errors import RecordNotFound

        with pytest.raises(RecordNotFound):
            cancel_appointment(db, patient_b, 1)

    def test_cancellation_is_audited(self, db, patient_b):
        slot_id = free_slot(db)
        booked = book_appointment(db, patient_b, slot_id=slot_id, reason="x")
        cancel_appointment(db, patient_b, booked["appointment"]["appointment_id"])
        assert "appointment_cancelled" in {e.action for e in db.query(AuditEvent).all()}


class TestContract:
    def test_results_are_json_serialisable(self, db, patient_b):
        import json

        json.dumps(book_appointment(db, patient_b, slot_id=free_slot(db), reason="x"))
        json.dumps(book_appointment(db, patient_b, slot_id=999_999, reason="x"))

    def test_success_and_refusal_share_a_shape(self, db, patient_b):
        ok = book_appointment(db, patient_b, slot_id=free_slot(db), reason="x")
        bad = book_appointment(db, patient_b, slot_id=999_999, reason="x")
        assert set(ok) == set(bad)


class TestSameDayBookingHasNothingToRemind:
    """A 24-hour lead and a booking three hours away do not both fit.

    Live: a 15:00 slot booked the same afternoon produced a reminder row dated
    *yesterday* — already due the moment it was written — while the receipt
    promised "we'll remind you the day before". Nothing failed, because a
    past-dated pending row is perfectly valid SQL.

    It would not have stayed harmless. Phase 8's poll job selects pending
    reminders whose time has passed, so its first sweep would deliver a
    reminder for a visit already under way. The row is the decision: none is
    written, and the receipt says what is true instead.
    """

    def _today_slot(self, db) -> int:
        slots = find_available_slots(db, department_id=1, limit=200)["slots"]
        today = [
            s
            for s in slots
            if datetime.fromisoformat(s["start"]).date() == clock.today()
        ]
        assert today, "the seed lays slots from today; without one this proves nothing"
        return today[0]["slot_id"]

    def test_no_reminder_row_is_written(self, db, patient_b):
        booked = book_appointment(
            db, patient_b, slot_id=self._today_slot(db), reason="x"
        )
        reminders = (
            db.query(Reminder)
            .filter(Reminder.appointment_id == booked["appointment"]["appointment_id"])
            .all()
        )
        assert reminders == []

    def test_no_row_is_left_already_due(self, db, patient_b):
        """The sharper version: not "none was written" but "none is deliverable
        the instant it exists". A future change that wrote a *cancelled* row
        would pass the test above and still be wrong if it ever flipped."""
        book_appointment(db, patient_b, slot_id=self._today_slot(db), reason="x")

        due_now = (
            db.query(Reminder)
            .filter(
                Reminder.status == ReminderStatus.PENDING,
                Reminder.scheduled_at <= clock.now(),
            )
            .count()
        )
        assert due_now == 0

    def test_a_later_booking_still_gets_one(self, db, patient_b):
        """Distrust green: a change that simply stopped creating reminders
        would pass both tests above."""
        booked = book_appointment(
            db, patient_b, slot_id=remindable_slot(db), reason="x"
        )
        assert (
            db.query(Reminder)
            .filter(Reminder.appointment_id == booked["appointment"]["appointment_id"])
            .count()
            == 1
        )


class TestCancellingClosesWhatItDerived:
    """Item 5, and the derivation invariant it belongs to.

    Reminders were retired on cancellation from the start. The
    missing-documents task was not — so a cancelled appointment left an open
    task telling the patient to bring a scan to a visit that no longer existed,
    and the follow-up screen went on showing it. Confirmed in the live
    database: AC-000002 cancelled, its reminder cancelled, the task still open.

    Every row derived from an appointment updates inside that appointment's
    transaction, or it is only derived until something goes wrong.
    """

    def _with_task(self, db, patient_b):
        from app.models import FollowUpTaskType
        from app.tools.tasks import upsert_followup_task

        booked = book_appointment(
            db, patient_b, slot_id=remindable_slot(db), reason="x"
        )
        appointment_id = booked["appointment"]["appointment_id"]
        upsert_followup_task(
            db,
            patient_id=2,
            task_type=FollowUpTaskType.MISSING_DOCUMENTS,
            details={"missing": ["Prior MRI or CT report"]},
            appointment_id=appointment_id,
        )
        db.commit()
        return appointment_id

    def test_the_task_is_closed(self, db, patient_b):
        from app.models import FollowUpTask, FollowUpTaskStatus

        appointment_id = self._with_task(db, patient_b)
        cancel_appointment(db, patient_b, appointment_id)

        # Read from a session that has seen none of this: an object still in
        # the unit of work can look closed without the change having landed.
        fresh = SessionLocal()
        try:
            statuses = {
                task.status
                for task in fresh.query(FollowUpTask)
                .filter(FollowUpTask.appointment_id == appointment_id)
                .all()
            }
        finally:
            fresh.close()
        assert statuses == {FollowUpTaskStatus.CLOSED}

    def test_a_task_for_another_appointment_stays_open(self, db, patient_b):
        """Scoped to this appointment. A cancellation is not a licence to close
        the patient's other obligations."""
        from app.models import FollowUpTask, FollowUpTaskStatus, FollowUpTaskType
        from app.tools.tasks import upsert_followup_task

        appointment_id = self._with_task(db, patient_b)
        other = book_appointment(
            db, patient_b, slot_id=remindable_slot(db, index=5), reason="y"
        )["appointment"]["appointment_id"]
        upsert_followup_task(
            db,
            patient_id=2,
            task_type=FollowUpTaskType.MISSING_DOCUMENTS,
            details={"missing": ["Referral letter"]},
            appointment_id=other,
        )
        db.commit()

        cancel_appointment(db, patient_b, appointment_id)

        fresh = SessionLocal()
        try:
            survivor = (
                fresh.query(FollowUpTask)
                .filter(FollowUpTask.appointment_id == other)
                .one()
            )
            assert survivor.status is FollowUpTaskStatus.OPEN
        finally:
            fresh.close()
