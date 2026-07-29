"""Slot availability queries.

Read-only, but these results become the options a patient is offered, so they
are treated as consequential: past slots are never returned, inactive doctors
are withheld, and ordering is stable so that "the first slot offered" means the
same thing on every run.

Availability answered here is advisory. The booking transaction re-checks the
slot inside its own transaction — between offering and confirming, someone else
may have taken it.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

from sqlalchemy.orm import Session

from app import clock
from app.models import (
    Appointment,
    AppointmentSlot,
    AppointmentStatus,
    Department,
    Doctor,
    SlotStatus,
)

DEFAULT_LIMIT = 20

#: Hour boundaries for a part-of-day filter, end-exclusive.
PARTS_OF_DAY: dict[str, tuple[int, int]] = {
    "morning": (0, 12),
    "afternoon": (12, 17),
    "evening": (17, 24),
}


def _serialise_slot(slot: AppointmentSlot, doctor: Doctor, department: Department) -> dict[str, Any]:
    return {
        "slot_id": slot.id,
        "doctor_id": doctor.id,
        "doctor_name": doctor.name,
        "department_id": department.id,
        "department_name": department.name,
        "start": slot.start_time.isoformat(),
        "end": slot.end_time.isoformat(),
    }


def _patient_busy(
    session: Session, patient_id: int
) -> list[tuple[datetime, datetime, str]]:
    """When this patient is already committed, and with whom.

    The department name rides along because withholding a time silently reads
    as ignoring the question: "how about 11am?" answered with 9, 10 and 2
    tells the patient nothing about where their 11 went.
    """
    rows = (
        session.query(
            AppointmentSlot.start_time, AppointmentSlot.end_time, Department.name
        )
        .join(Appointment, Appointment.slot_id == AppointmentSlot.id)
        .join(Department, Department.id == Appointment.department_id)
        .filter(
            Appointment.patient_id == patient_id,
            Appointment.status.in_((AppointmentStatus.PENDING, AppointmentStatus.CONFIRMED)),
        )
        .all()
    )
    return [(start, end, name) for start, end, name in rows]


def find_available_slots(
    session: Session,
    *,
    department_id: int | None = None,
    doctor_id: int | None = None,
    start: date | None = None,
    end: date | None = None,
    part_of_day: str | None = None,
    limit: int = DEFAULT_LIMIT,
    free_for_patient: int | None = None,
) -> dict[str, Any]:
    """Available slots matching the filters, soonest first.

    ``total_matching`` counts everything that matched before ``limit`` was
    applied, so a patient shown three of forty can be told there are more
    rather than being left to assume that is all there is.

    An unrecognised ``part_of_day`` is ignored rather than guessed at: silently
    narrowing to a period the patient did not ask for hides the choice.

    ``free_for_patient`` drops the slots that overlap that patient's own live
    appointments. A slot free in the *department's* diary can still be a time
    the patient is already spoken for, and the booking transaction refuses that
    as ``patient_double_booked`` — correctly, but far too late to be useful.
    Live, a Dermatology proposal held Monday 9:00 AM against a patient who
    already had a Dermatology appointment at Monday 9:00 AM, so the Confirm was
    guaranteed to bounce before it was ever offered. The commit-time check
    stays exactly where it is; this only stops the engine setting it up to fail.

    Those drops are returned in ``withheld_for_patient`` rather than only
    subtracted. A patient who asks "how about 11am?" and gets 9, 10 and 2 back
    with no explanation has been answered as though they said nothing; the
    caller needs the times and the clashing department to say otherwise.
    """
    query = (
        session.query(AppointmentSlot, Doctor, Department)
        .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
        .join(Department, Department.id == Doctor.department_id)
        .filter(
            AppointmentSlot.status == SlotStatus.AVAILABLE,
            Doctor.active.is_(True),
            Department.active.is_(True),
            # Never offer a slot that has already started.
            AppointmentSlot.start_time > clock.now(),
        )
    )

    if department_id is not None:
        query = query.filter(Department.id == department_id)
    if doctor_id is not None:
        query = query.filter(Doctor.id == doctor_id)
    if start is not None:
        query = query.filter(AppointmentSlot.start_time >= datetime.combine(start, time.min))
    if end is not None:
        query = query.filter(AppointmentSlot.start_time <= datetime.combine(end, time.max))

    rows = query.order_by(AppointmentSlot.start_time, AppointmentSlot.id).all()

    # Part-of-day is filtered in Python: extracting an hour from a timestamp is
    # spelled differently on SQLite and PostgreSQL, and the row count here is
    # small enough that portability is worth more than pushing it into SQL.
    window = PARTS_OF_DAY.get((part_of_day or "").lower())
    if window is not None:
        first_hour, last_hour = window
        rows = [row for row in rows if first_hour <= row[0].start_time.hour < last_hour]

    # What was withheld, not merely how many: the caller has to be able to say
    # *why* a time the patient asked for is not on the list it got back.
    withheld: list[dict[str, Any]] = []
    if free_for_patient is not None:
        busy = _patient_busy(session, free_for_patient)
        keep = []
        for row in rows:
            clash = next(
                (
                    name
                    for start_at, finish, name in busy
                    if row[0].start_time < finish and start_at < row[0].end_time
                ),
                None,
            )
            if clash is None:
                keep.append(row)
            else:
                withheld.append(
                    {
                        "slot_id": row[0].id,
                        "start": row[0].start_time.isoformat(),
                        "department_name": clash,
                    }
                )
        rows = keep

    total = len(rows)
    limited = rows[: max(0, limit)] if limit is not None else rows

    return {
        "department_id": department_id,
        "doctor_id": doctor_id,
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "part_of_day": part_of_day,
        "total_matching": total,
        "slots": [_serialise_slot(slot, doctor, dept) for slot, doctor, dept in limited],
        "withheld_for_patient": withheld,
    }


def get_slot(session: Session, slot_id: int) -> dict[str, Any]:
    """Look up one slot.

    Never raises for a missing slot: the commit path calls this about a slot
    that may have been taken or withdrawn since it was offered, and that is an
    ordinary outcome to be reported, not an exception.
    """
    row = (
        session.query(AppointmentSlot, Doctor, Department)
        .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
        .join(Department, Department.id == Doctor.department_id)
        .filter(AppointmentSlot.id == slot_id)
        .one_or_none()
    )
    if row is None:
        return {
            "slot_id": slot_id,
            "found": False,
            "available": False,
            "doctor_id": None,
            "doctor_name": None,
            "department_id": None,
            "department_name": None,
            "start": None,
            "end": None,
        }

    slot, doctor, department = row
    payload = _serialise_slot(slot, doctor, department)
    payload["found"] = True
    payload["available"] = (
        slot.status == SlotStatus.AVAILABLE
        and doctor.active
        and department.active
        and slot.start_time > clock.now()
    )
    return payload
