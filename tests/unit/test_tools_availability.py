"""Slot availability.

Read-only, but the answers here become the options a patient is offered, so a
stale or wrongly-ordered result surfaces as a booking against a slot that was
never really free. Ordering is asserted because the golden set diffs it and
because "the first slot offered" is what most patients accept.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from app import clock
from app.models import (
    Appointment,
    AppointmentSlot,
    AppointmentStatus,
    Doctor,
    SlotStatus,
)
from app.tools.availability import find_available_slots, get_slot

MONDAY = date(2026, 8, 3)


@pytest.fixture(autouse=True)
def _reseed_on_monday(db):
    """Seed with the clock already on the anchor Monday.

    The seed generates its window relative to the anchor, so freezing after
    seeding would leave the slots in the wrong place relative to "now".
    """
    clock.freeze(datetime(2026, 8, 3, 8, 0))
    from scripts.seed import seed

    seed(db, anchor=MONDAY)
    db.commit()
    return db


class TestFiltering:
    def test_only_available_slots_are_returned(self, db):
        result = find_available_slots(db, department_id=1)
        ids = {s["slot_id"] for s in result["slots"]}
        booked = (
            db.query(AppointmentSlot).filter(AppointmentSlot.status == SlotStatus.BOOKED).all()
        )
        assert ids.isdisjoint({b.id for b in booked})

    def test_department_filter_excludes_other_departments(self, db):
        result = find_available_slots(db, department_id=1)
        assert result["slots"]
        assert {s["department_id"] for s in result["slots"]} == {1}

    def test_doctor_filter_narrows_to_one_doctor(self, db):
        result = find_available_slots(db, doctor_id=1)
        assert {s["doctor_id"] for s in result["slots"]} == {1}

    def test_date_range_is_inclusive_at_both_ends(self, db):
        start = MONDAY + timedelta(days=1)
        end = MONDAY + timedelta(days=2)
        result = find_available_slots(db, department_id=1, start=start, end=end)
        days = {s["start"][:10] for s in result["slots"]}
        assert days == {start.isoformat(), end.isoformat()}

    def test_past_slots_are_never_offered(self, db):
        """Ten in the morning cannot be booked at eleven."""
        clock.freeze(datetime(2026, 8, 3, 11, 30))
        result = find_available_slots(db, department_id=1, start=MONDAY, end=MONDAY)
        times = [s["start"] for s in result["slots"]]
        assert all(t > "2026-08-03T11:30" for t in times)

    def test_an_inactive_doctors_slots_are_withheld(self, db):
        db.query(Doctor).filter(Doctor.id == 1).one().active = False
        db.flush()
        result = find_available_slots(db, doctor_id=1)
        assert result["slots"] == []


class TestPartOfDay:
    def test_morning_returns_only_slots_before_noon(self, db):
        result = find_available_slots(db, department_id=1, part_of_day="morning")
        assert result["slots"]
        assert all(int(s["start"][11:13]) < 12 for s in result["slots"])

    def test_afternoon_returns_only_afternoon_slots(self, db):
        result = find_available_slots(db, department_id=1, part_of_day="afternoon")
        assert result["slots"]
        assert all(12 <= int(s["start"][11:13]) < 17 for s in result["slots"])

    def test_an_unknown_part_of_day_is_ignored_not_guessed(self, db):
        unfiltered = find_available_slots(db, department_id=1)
        result = find_available_slots(db, department_id=1, part_of_day="teatime")
        assert len(result["slots"]) == len(unfiltered["slots"])


class TestOrderingAndLimits:
    def test_slots_are_ordered_by_start_time(self, db):
        result = find_available_slots(db, department_id=1)
        times = [s["start"] for s in result["slots"]]
        assert times == sorted(times)

    def test_ordering_is_stable_across_identical_calls(self, db):
        """Ties on start time must not reorder between runs, or golden files
        flap and 'the first slot offered' stops being reproducible."""
        first = find_available_slots(db, department_id=1)["slots"]
        second = find_available_slots(db, department_id=1)["slots"]
        assert first == second

    def test_limit_caps_the_result(self, db):
        result = find_available_slots(db, department_id=1, limit=3)
        assert len(result["slots"]) == 3

    def test_the_total_reports_matches_beyond_the_limit(self, db):
        """A patient offered three of forty should be told there are more."""
        result = find_available_slots(db, department_id=1, limit=3)
        assert result["total_matching"] > 3


class TestContract:
    def test_each_slot_carries_what_a_confirmation_needs(self, db):
        slot = find_available_slots(db, department_id=1, limit=1)["slots"][0]
        assert set(slot) >= {
            "slot_id", "doctor_id", "doctor_name",
            "department_id", "department_name", "start", "end",
        }

    def test_no_matches_is_an_empty_list_not_an_error(self, db):
        far_future = MONDAY + timedelta(days=400)
        result = find_available_slots(db, department_id=1, start=far_future, end=far_future)
        assert result["slots"] == []
        assert result["total_matching"] == 0

    def test_results_are_json_serialisable(self, db):
        import json

        json.dumps(find_available_slots(db, department_id=1, limit=2))

    def test_get_slot_returns_a_single_slot(self, db):
        first = find_available_slots(db, department_id=1, limit=1)["slots"][0]
        fetched = get_slot(db, first["slot_id"])
        assert fetched["slot_id"] == first["slot_id"]
        assert fetched["available"] is True

    def test_get_slot_reports_a_missing_slot_without_raising(self, db):
        """The commit path asks this about a slot that may have just vanished."""
        result = get_slot(db, 999_999)
        assert result["found"] is False
        assert result["available"] is False


class TestATimeThePatientAlreadyHas:
    """Round 7, item 5 — the engine offered a slot the commit would refuse.

    Live: a Dermatology proposal held Monday 9:00 AM for a patient who already
    had a Dermatology appointment at Monday 9:00 AM. The conflict guard did its
    job and the Confirm bounced with "You already have an appointment at that
    time" — a refusal the proposal engine had guaranteed before the patient ever
    saw the offer. A slot free in the department's diary is not necessarily a
    time the patient is free.

    The commit-time check stays exactly where it is. This only stops it being
    set up to fail.
    """

    def _book(self, db, slot: AppointmentSlot, patient_id: int = 1) -> Appointment:
        appointment = Appointment(
            patient_id=patient_id,
            department_id=slot.doctor.department_id,
            doctor_id=slot.doctor_id,
            slot_id=slot.id,
            status=AppointmentStatus.CONFIRMED,
            reference_code=f"AC-90{slot.id:04d}",
            reason="",
        )
        slot.status = SlotStatus.BOOKED
        db.add(appointment)
        db.flush()
        return appointment

    def _first_free(self, db, department_id: int) -> AppointmentSlot:
        found = find_available_slots(db, department_id=department_id)
        return db.get(AppointmentSlot, found["slots"][0]["slot_id"])

    def test_a_clashing_slot_in_another_department_is_withheld(self, db):
        taken = self._first_free(db, 1)
        self._book(db, taken)

        clashing = (
            db.query(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .filter(
                AppointmentSlot.start_time == taken.start_time,
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                Doctor.department_id != 1,
            )
            .first()
        )
        assert clashing is not None, "the seed has no same-time slot elsewhere"

        found = find_available_slots(
            db, department_id=clashing.doctor.department_id, free_for_patient=1
        )
        assert clashing.id not in {slot["slot_id"] for slot in found["slots"]}

    def test_it_is_still_free_for_everybody_else(self, db):
        """Scoped to the patient asking. Somebody else's diary is not this
        patient's business, and a global exclusion would empty the schedule."""
        taken = self._first_free(db, 1)
        self._book(db, taken)
        clashing = (
            db.query(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .filter(
                AppointmentSlot.start_time == taken.start_time,
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                Doctor.department_id != 1,
            )
            .first()
        )

        found = find_available_slots(
            db, department_id=clashing.doctor.department_id, free_for_patient=2
        )
        assert clashing.id in {slot["slot_id"] for slot in found["slots"]}

    def test_without_the_argument_nothing_changes(self, db):
        """The default is the old behaviour exactly. Callers opt in."""
        taken = self._first_free(db, 1)
        self._book(db, taken)
        clashing = (
            db.query(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .filter(
                AppointmentSlot.start_time == taken.start_time,
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                Doctor.department_id != 1,
            )
            .first()
        )

        found = find_available_slots(db, department_id=clashing.doctor.department_id)
        assert clashing.id in {slot["slot_id"] for slot in found["slots"]}

    def test_a_cancelled_appointment_frees_the_time_again(self, db):
        """Live statuses only. A cancelled visit is not a commitment, and
        treating it as one would shrink the schedule for good."""
        taken = self._first_free(db, 1)
        appointment = self._book(db, taken)
        clashing = (
            db.query(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .filter(
                AppointmentSlot.start_time == taken.start_time,
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                Doctor.department_id != 1,
            )
            .first()
        )
        appointment.status = AppointmentStatus.CANCELLED
        db.flush()

        found = find_available_slots(
            db, department_id=clashing.doctor.department_id, free_for_patient=1
        )
        assert clashing.id in {slot["slot_id"] for slot in found["slots"]}

    def test_what_was_withheld_is_reported_back(self, db):
        """Subtracting silently is how "how about 11am?" came back as 9, 10 and
        2 with no mention of the patient's own 11:00. A caller cannot say why a
        time is missing unless it is told which times went and to whom."""
        taken = self._first_free(db, 1)
        appointment = self._book(db, taken)
        clashing = (
            db.query(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .filter(
                AppointmentSlot.start_time == taken.start_time,
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                Doctor.department_id != 1,
            )
            .first()
        )

        found = find_available_slots(
            db, department_id=clashing.doctor.department_id, free_for_patient=1
        )

        withheld = found["withheld_for_patient"]
        assert clashing.id in {row["slot_id"] for row in withheld}
        row = next(row for row in withheld if row["slot_id"] == clashing.id)
        assert row["start"] == taken.start_time.isoformat()
        assert row["department_name"] == appointment.department.name

    def test_nothing_withheld_is_an_empty_list_not_a_missing_key(self, db):
        """Callers read it unconditionally; a key that appears only sometimes
        is the shape that silently reads as "nothing"."""
        found = find_available_slots(db, department_id=1, free_for_patient=2)
        assert found["withheld_for_patient"] == []
