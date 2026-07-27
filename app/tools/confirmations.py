"""Confirmation rendering — the pinned fallback for consequential wording.

Every fact in a confirmation is read back out of the persisted row: doctor
name, weekday, date, time, department, reference code. Nothing is carried over
from what the model believed a moment ago.

That matters because a confirmation is the one reply a patient acts on. A
model summarising its own intention can drift a weekday by one and produce a
sentence that is fluent, confident, and wrong — and the patient arrives on
Tuesday for a Wednesday appointment. Live evals grade these fact fields exactly
against the database, and when wording drifts, this renderer is what the system
falls back to.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.errors import RecordNotFound
from app.models import Appointment, AppointmentSlot, Department, Doctor


def _clock_time(moment) -> str:
    """12-hour, matching ``app.workflow.replies.clock_time``.

    Duplicated rather than imported: ``replies`` imports this module, and a
    tool reaching back up into the workflow layer would invert the dependency
    the whole ``tools`` package is built on. Four lines, pinned by a test that
    compares the two.
    """
    hour = moment.hour % 12 or 12
    return f"{hour}:{moment:%M} {'AM' if moment.hour < 12 else 'PM'}"


def render_confirmation(session: Session, appointment_id: int) -> dict[str, Any]:
    """Facts and a ready-made sentence for an appointment, read from the row.

    Returns both ``facts`` (for exact grading and for the UI to lay out itself)
    and ``sentence`` (the code-authored wording). A caller may let the model
    phrase things around the facts, but the facts come from here.
    """
    appointment = session.get(Appointment, appointment_id)
    if appointment is None:
        raise RecordNotFound("Appointment", appointment_id)

    slot = session.get(AppointmentSlot, appointment.slot_id) if appointment.slot_id else None
    doctor = session.get(Doctor, appointment.doctor_id)
    department = session.get(Department, appointment.department_id)

    facts = {
        "appointment_id": appointment.id,
        "reference_code": appointment.reference_code,
        "department_name": department.name if department else None,
        "doctor_name": doctor.name if doctor else None,
        "weekday": f"{slot.start_time:%A}" if slot else None,
        "date": f"{slot.start_time.day} {slot.start_time:%B} {slot.start_time.year}" if slot else None,
        # 12-hour, because this string is read by patients — the receipt, the
        # cancellation prompt and the proposal card all render it. The row
        # keeps the timestamp; only the way it is said changes.
        "time": _clock_time(slot.start_time) if slot else None,
        "status": appointment.status.value,
    }

    if slot is None:
        sentence = (
            f"Your {facts['department_name']} appointment "
            f"(reference {facts['reference_code']}) is {facts['status']}."
        )
    else:
        sentence = (
            f"Your {facts['department_name']} appointment with {facts['doctor_name']} "
            f"is {facts['status']} for {facts['weekday']} {facts['date']} "
            f"at {facts['time']}. Reference: {facts['reference_code']}."
        )

    return {"appointment_id": appointment.id, "facts": facts, "sentence": sentence}
