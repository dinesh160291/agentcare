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
        slot_id = free_slot(db)
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
        original = free_slot(db)
        booked = book_appointment(db, patient_b, slot_id=original, reason="x")
        appointment_id = booked["appointment"]["appointment_id"]
        target = free_slot(db, index=3)

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
        original = free_slot(db)
        booked = book_appointment(db, patient_b, slot_id=original, reason="x")
        appointment_id = booked["appointment"]["appointment_id"]
        target = free_slot(db, index=3)

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
        original = free_slot(db)
        booked = book_appointment(db, patient_b, slot_id=original, reason="x")
        appointment_id = booked["appointment"]["appointment_id"]
        target = free_slot(db, index=3)
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
        slot_id = free_slot(db)
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
