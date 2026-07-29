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


class TestReschedulingCannotDoubleBook:
    """Round 9, item 1 — the clash guard booking has always had, ported.

    Live, runs #6 and #7: the patient booked Ophthalmology for Thursday 6
    August at 9:00 AM, then asked to move their Dermatology appointment "to
    August 6th". The search withheld the 9:00 slot correctly and even named the
    clash; the model proposed it anyway — out of the withheld list, which the
    tool result handed it — and ``reschedule_appointment`` committed it without
    a word. Appointments 3 and 5 both confirmed at one instant, which the
    *booking* path has refused since Phase 2.

    This is the layer that makes it impossible rather than unlikely. Every
    guard above it can be bypassed by a stale proposal, a retry, or a direct
    call, which is why the refusal has to live here as well as there.
    """

    @staticmethod
    def _department_slot(db, department_id: int, *, start=None, not_at=None, exclude=()):
        """A free slot in a department, optionally at (or away from) a time.

        ``not_at`` excludes a whole start time rather than a slot id, and it is
        load-bearing: every department has two doctors, so excluding one slot
        leaves its twin at the same instant — and a "different time" chosen
        that way is the same time, which makes the *setup* booking bounce off
        the very guard under test.
        """
        from app.models import Doctor

        query = (
            db.query(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .filter(
                Doctor.department_id == department_id,
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                AppointmentSlot.start_time > clock.now() + timedelta(hours=24),
            )
        )
        if start is not None:
            query = query.filter(AppointmentSlot.start_time == start)
        if not_at is not None:
            query = query.filter(AppointmentSlot.start_time != not_at)
        if exclude:
            query = query.filter(AppointmentSlot.id.notin_(exclude))
        return query.order_by(AppointmentSlot.start_time, AppointmentSlot.id).first()

    def _the_live_shape(self, db, patient_b):
        """Two appointments in two departments, and a slot that collides.

        Returns (mover_appointment_id, clashing_slot_id, mover_slot_id) — the
        Dermatology appointment, the Dermatology slot sitting at the exact
        time of the Ophthalmology one, and where the mover started.
        """
        keeper_slot = self._department_slot(db, 8)
        keeper_start = keeper_slot.start_time
        book_appointment(db, patient_b, slot_id=keeper_slot.id, reason="vision test")

        clashing = self._department_slot(db, 3, start=keeper_start)
        assert clashing is not None, "the seed has no Dermatology slot at that time"

        mover_slot = self._department_slot(db, 3, not_at=keeper_start)
        mover = book_appointment(db, patient_b, slot_id=mover_slot.id, reason="skin rash")
        assert mover["ok"] is True, "the setup booking must succeed on its own"

        return mover["appointment"]["appointment_id"], clashing.id, mover_slot.id

    def test_moving_onto_another_appointments_time_is_refused(self, db, patient_b):
        appointment_id, clashing_slot_id, _ = self._the_live_shape(db, patient_b)

        result = reschedule_appointment(
            db, patient_b, appointment_id, new_slot_id=clashing_slot_id
        )

        assert result["ok"] is False
        assert result["reason"] == "patient_double_booked"

    def test_no_two_live_appointments_share_a_start_time(self, db, patient_b):
        """The claim the patient actually cares about, asserted about the data.

        The reason code above is this repo's word for it; this is the state
        itself, and it is the query that found the live defect.
        """
        appointment_id, clashing_slot_id, _ = self._the_live_shape(db, patient_b)
        reschedule_appointment(db, patient_b, appointment_id, new_slot_id=clashing_slot_id)

        starts = [
            start
            for (start,) in db.query(AppointmentSlot.start_time)
            .join(Appointment, Appointment.slot_id == AppointmentSlot.id)
            .filter(
                Appointment.patient_id == 2,
                Appointment.status.in_(
                    (AppointmentStatus.PENDING, AppointmentStatus.CONFIRMED)
                ),
            )
            .all()
        ]
        assert len(starts) == 2, "both appointments must still exist to be compared"
        assert len(starts) == len(set(starts))

    def test_the_clashing_slot_is_left_available(self, db, patient_b):
        """Refused before the claim, not after it.

        A refusal that has already flipped the slot to booked takes a time out
        of everyone's diary to punish a move that never happened.
        """
        appointment_id, clashing_slot_id, _ = self._the_live_shape(db, patient_b)
        reschedule_appointment(db, patient_b, appointment_id, new_slot_id=clashing_slot_id)

        assert db.get(AppointmentSlot, clashing_slot_id).status == SlotStatus.AVAILABLE

    def test_the_original_appointment_is_untouched(self, db, patient_b):
        appointment_id, clashing_slot_id, mover_slot_id = self._the_live_shape(db, patient_b)
        reschedule_appointment(db, patient_b, appointment_id, new_slot_id=clashing_slot_id)

        assert db.get(Appointment, appointment_id).slot_id == mover_slot_id
        assert db.get(AppointmentSlot, mover_slot_id).status == SlotStatus.BOOKED

    def test_an_appointment_does_not_clash_with_itself(self, db, patient_b):
        """Self-exclusion, pinned by the only case that can falsify it.

        Moving a 9:00 to a 10:00 cannot fail for want of this rule — the two
        slots do not overlap, so a check with no exclusion at all still lets it
        through. Changing *doctor* at the same time is the case that does:
        without excluding the appointment being moved, its own slot overlaps
        the destination and every such move is refused as a clash with itself.
        """
        original = self._department_slot(db, 3)
        booked = book_appointment(db, patient_b, slot_id=original.id, reason="skin rash")
        appointment_id = booked["appointment"]["appointment_id"]

        same_time_other_doctor = self._department_slot(
            db, 3, start=original.start_time, exclude=(original.id,)
        )
        assert same_time_other_doctor is not None, "the seed has one dermatologist"

        result = reschedule_appointment(
            db, patient_b, appointment_id, new_slot_id=same_time_other_doctor.id
        )

        assert result["ok"] is True
        assert db.get(Appointment, appointment_id).slot_id == same_time_other_doctor.id

    def test_an_ordinary_move_to_another_hour_still_succeeds(self, db, patient_b):
        """The regression the guard could plausibly cause.

        Not a pin on self-exclusion — see above — but on the overlap predicate
        itself: a helper that asked "does this patient have anything that day"
        would refuse this, and so would one that compared dates rather than
        instants.
        """
        original = self._department_slot(db, 3)
        booked = book_appointment(db, patient_b, slot_id=original.id, reason="skin rash")
        later_same_day = self._department_slot(
            db,
            3,
            start=original.start_time + timedelta(hours=1),
            exclude=(original.id,),
        )
        assert later_same_day is not None, "the seed has no next-hour slot that day"

        result = reschedule_appointment(
            db,
            patient_b,
            booked["appointment"]["appointment_id"],
            new_slot_id=later_same_day.id,
        )

        assert result["ok"] is True

    def test_a_cancelled_appointment_does_not_block_the_time(self, db, patient_b):
        """A released commitment is not a commitment.

        The same rule the search follows: live statuses only. Counting a
        cancelled visit would shrink the patient's own schedule permanently.
        """
        appointment_id, clashing_slot_id, _ = self._the_live_shape(db, patient_b)
        keeper = (
            db.query(Appointment)
            .filter(Appointment.patient_id == 2, Appointment.id != appointment_id)
            .order_by(Appointment.id.desc())
            .first()
        )
        cancel_appointment(db, patient_b, keeper.id)

        result = reschedule_appointment(
            db, patient_b, appointment_id, new_slot_id=clashing_slot_id
        )
        assert result["ok"] is True


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
