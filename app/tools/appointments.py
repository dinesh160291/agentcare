"""Booking, rescheduling, and cancellation — the mutating appointment tools.

These are the only functions that change appointment state, and they are the
ones a race condition finds first. Three rules hold throughout:

**The slot is claimed by compare-and-swap.** Availability was answered when the
slot was offered to the patient; by the time they confirm, someone else may
have taken it. The claim is a conditional update — mark this slot booked *only
if it is still available* — and zero affected rows means someone else won.
Checking first and updating second leaves a window between the two.

**Every failure clears the pending proposal.** A proposal pointing at a dead
slot which is still confirmable is a loop with no exit: the patient presses
Confirm, it fails, the proposal survives, and they press Confirm again.

**Derived rows move in their source's transaction.** Reschedule retimes the
reminders; cancel cancels them. A reminder that outlives its appointment tells
a patient to attend something that is not happening.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import update
from sqlalchemy.orm import Session

from app import clock
from app.audit import write_audit
from app.auth.ownership import get_owned_or_404, patient_profile_for
from app.models import (
    Appointment,
    AppointmentSlot,
    AppointmentStatus,
    Department,
    Doctor,
    Reminder,
    ReminderStatus,
    ReminderType,
    SlotStatus,
    User,
    WorkflowRun,
)

#: How far ahead of the appointment its reminder fires.
REMINDER_LEAD = timedelta(hours=24)

#: Appointment states a patient may still act on.
LIVE_STATUSES = (AppointmentStatus.PENDING, AppointmentStatus.CONFIRMED)


def _result(
    *,
    ok: bool,
    reason: str | None = None,
    appointment: dict[str, Any] | None = None,
    alternatives: list[dict[str, Any]] | None = None,
    message: str = "",
) -> dict[str, Any]:
    """Shape-stable result: success and refusal carry the same keys."""
    return {
        "ok": ok,
        "reason": reason,
        "appointment": appointment,
        "alternatives": alternatives or [],
        "message": message,
    }


def _serialise(session: Session, appointment: Appointment) -> dict[str, Any]:
    slot = session.get(AppointmentSlot, appointment.slot_id) if appointment.slot_id else None
    doctor = session.get(Doctor, appointment.doctor_id)
    return {
        "appointment_id": appointment.id,
        "reference_code": appointment.reference_code,
        "patient_id": appointment.patient_id,
        "department_id": appointment.department_id,
        "doctor_id": appointment.doctor_id,
        "doctor_name": doctor.name if doctor else None,
        "slot_id": appointment.slot_id,
        "start": slot.start_time.isoformat() if slot else None,
        "end": slot.end_time.isoformat() if slot else None,
        "status": appointment.status.value,
        "reason": appointment.reason,
    }


def _reference_code(appointment_id: int) -> str:
    return f"AC-{appointment_id:06d}"


def _claim_slot(session: Session, slot_id: int) -> bool:
    """Claim a slot by compare-and-swap. True if this caller won it.

    Conditional on the slot still being available, so two confirmations racing
    for the same slot cannot both succeed. Checking availability and then
    writing would leave a window between the read and the write.
    """
    affected = session.execute(
        update(AppointmentSlot)
        .where(
            AppointmentSlot.id == slot_id,
            AppointmentSlot.status == SlotStatus.AVAILABLE,
        )
        .values(status=SlotStatus.BOOKED)
    ).rowcount
    return bool(affected)


def _release_slot(session: Session, slot_id: int | None) -> None:
    """Return a slot to the pool, but only if this appointment still holds it.

    Conditional on the slot being booked: if a second cancellation arrives
    after someone rebooked the freed slot, this must not hand it out from
    under them.
    """
    if slot_id is None:
        return
    session.execute(
        update(AppointmentSlot)
        .where(AppointmentSlot.id == slot_id, AppointmentSlot.status == SlotStatus.BOOKED)
        .values(status=SlotStatus.AVAILABLE)
    )


def _alternatives(session: Session, *, department_id: int, around) -> list[dict[str, Any]]:
    """A few nearby slots to offer when a booking fails."""
    from app.tools.availability import find_available_slots

    return find_available_slots(
        session,
        department_id=department_id,
        start=around.date() if around else None,
        limit=5,
    )["slots"]


def _clear_proposal(session: Session, run: WorkflowRun | None) -> None:
    """Drop the pending proposal on every exit path, success or failure."""
    if run is None:
        return
    run.clear_proposal()
    session.flush()


def _refuse(
    session: Session,
    run: WorkflowRun | None,
    *,
    reason: str,
    message: str,
    alternatives: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Clear the proposal, **commit**, and return a refusal.

    The commit is the point. Flushing alone leaves the cleared proposal inside
    an uncommitted transaction, so any later rollback — an error handler, a
    request teardown, a retry — restores it still pointing at the dead slot,
    and the patient is back in the un-exitable Confirm loop this clearing
    exists to prevent.

    Committing here is safe because every refusal path reaches this function
    having changed nothing else: the slot claim either was not attempted or
    did not succeed.
    """
    _clear_proposal(session, run)
    session.commit()
    return _result(ok=False, reason=reason, message=message, alternatives=alternatives)


def _retire_reminders(session: Session, appointment_id: int) -> None:
    """Cancel every pending reminder for an appointment."""
    session.query(Reminder).filter(
        Reminder.appointment_id == appointment_id,
        Reminder.status == ReminderStatus.PENDING,
    ).update({Reminder.status: ReminderStatus.CANCELLED}, synchronize_session=False)


def _create_reminder(session: Session, appointment: Appointment, slot: AppointmentSlot) -> Reminder:
    doctor = session.get(Doctor, appointment.doctor_id)
    reminder = Reminder(
        patient_id=appointment.patient_id,
        appointment_id=appointment.id,
        reminder_type=ReminderType.APPOINTMENT,
        scheduled_at=slot.start_time - REMINDER_LEAD,
        status=ReminderStatus.PENDING,
        message=(
            f"Reminder: appointment with {doctor.name} on "
            f"{slot.start_time:%A %d %B at %H:%M}."
        ),
    )
    session.add(reminder)
    return reminder


def book_appointment(
    session: Session,
    user: User,
    *,
    slot_id: int,
    reason: str = "",
    run: WorkflowRun | None = None,
) -> dict[str, Any]:
    """Book ``slot_id`` for the acting patient.

    Called only after the patient has confirmed — this function is the commit,
    not the proposal. The department is read from the slot rather than taken
    from the caller: a caller-supplied department could file a Cardiology slot
    under Dermatology, and every later required-documents diff would then read
    the wrong rules.
    """
    profile = patient_profile_for(session, user)

    slot = session.get(AppointmentSlot, slot_id)
    if slot is None:
        return _refuse(session, run, reason="slot_not_found",
                       message="That appointment time is no longer listed.")

    if slot.start_time <= clock.now():
        return _refuse(session, run, reason="slot_in_the_past",
                       message="That appointment time has already passed.")

    doctor = session.get(Doctor, slot.doctor_id)
    department = session.get(Department, doctor.department_id)

    # Availability filters these out, but a slot id can arrive here directly —
    # from a stale proposal, a retry, or the API — and that path never passes
    # through the availability query.
    if not doctor.active or not department.active:
        return _refuse(session, run, reason="doctor_unavailable",
                       message="That appointment time is no longer available.")

    clash = (
        session.query(Appointment)
        .join(AppointmentSlot, AppointmentSlot.id == Appointment.slot_id)
        .filter(
            Appointment.patient_id == profile.id,
            Appointment.status.in_(LIVE_STATUSES),
            AppointmentSlot.start_time == slot.start_time,
        )
        .first()
    )
    if clash is not None:
        return _refuse(
            session, run,
            reason="patient_double_booked",
            alternatives=_alternatives(session, department_id=doctor.department_id,
                                       around=slot.start_time),
            message="You already have an appointment at that time.",
        )

    if not _claim_slot(session, slot_id):
        return _refuse(
            session, run,
            reason="slot_taken",
            alternatives=_alternatives(session, department_id=doctor.department_id,
                                       around=slot.start_time),
            message="That time was taken while you were confirming.",
        )

    appointment = Appointment(
        patient_id=profile.id,
        doctor_id=slot.doctor_id,
        slot_id=slot.id,
        department_id=doctor.department_id,
        status=AppointmentStatus.CONFIRMED,
        reason=reason,
    )
    session.add(appointment)
    session.flush()
    appointment.reference_code = _reference_code(appointment.id)

    _create_reminder(session, appointment, slot)

    write_audit(
        session,
        action="appointment_booked",
        entity_type="Appointment",
        entity_id=appointment.id,
        actor=user,
        metadata={"slot_id": slot.id, "department_id": doctor.department_id},
    )

    _clear_proposal(session, run)
    session.commit()

    return _result(ok=True, appointment=_serialise(session, appointment))


def reschedule_appointment(
    session: Session,
    user: User,
    appointment_id: int,
    *,
    new_slot_id: int,
    run: WorkflowRun | None = None,
) -> dict[str, Any]:
    """Move an existing appointment to ``new_slot_id``.

    The new slot is claimed before the old one is released, so a failure leaves
    the patient holding their original appointment rather than nothing.
    """
    appointment = get_owned_or_404(session, Appointment, appointment_id, user)

    if appointment.status not in LIVE_STATUSES:
        return _refuse(session, run, reason="not_reschedulable",
                       message="That appointment can no longer be changed.")

    new_slot = session.get(AppointmentSlot, new_slot_id)
    if new_slot is None:
        return _refuse(session, run, reason="slot_not_found",
                       message="That appointment time is no longer listed.")

    if new_slot.start_time <= clock.now():
        return _refuse(session, run, reason="slot_in_the_past",
                       message="That appointment time has already passed.")

    doctor = session.get(Doctor, new_slot.doctor_id)
    department = session.get(Department, doctor.department_id)
    if not doctor.active or not department.active:
        return _refuse(session, run, reason="doctor_unavailable",
                       message="That appointment time is no longer available.")

    # Claim first: if this fails, nothing has been given up.
    if not _claim_slot(session, new_slot_id):
        return _refuse(
            session, run,
            reason="slot_taken",
            alternatives=_alternatives(session, department_id=doctor.department_id,
                                       around=new_slot.start_time),
            message="That time was taken while you were confirming.",
        )

    previous_slot_id = appointment.slot_id
    _release_slot(session, previous_slot_id)

    appointment.slot_id = new_slot.id
    appointment.doctor_id = new_slot.doctor_id
    appointment.department_id = doctor.department_id

    # Derivation: reminders belong to the appointment and move with it, in the
    # same transaction, so no window exists where they point at the old time.
    _retire_reminders(session, appointment.id)
    _create_reminder(session, appointment, new_slot)

    write_audit(
        session,
        action="appointment_rescheduled",
        entity_type="Appointment",
        entity_id=appointment.id,
        actor=user,
        metadata={"from_slot_id": previous_slot_id, "to_slot_id": new_slot.id},
    )

    _clear_proposal(session, run)
    session.commit()

    return _result(ok=True, appointment=_serialise(session, appointment))


def cancel_appointment(
    session: Session,
    user: User,
    appointment_id: int,
    *,
    run: WorkflowRun | None = None,
) -> dict[str, Any]:
    """Cancel an appointment, release its slot, and retire its reminders."""
    appointment = get_owned_or_404(session, Appointment, appointment_id, user)

    if appointment.status not in LIVE_STATUSES:
        return _refuse(session, run, reason="not_cancellable",
                       message="That appointment is not active.")

    slot_id = appointment.slot_id
    appointment.status = AppointmentStatus.CANCELLED
    _release_slot(session, slot_id)
    _retire_reminders(session, appointment.id)

    write_audit(
        session,
        action="appointment_cancelled",
        entity_type="Appointment",
        entity_id=appointment.id,
        actor=user,
        metadata={"slot_id": slot_id},
    )

    _clear_proposal(session, run)
    session.commit()

    return _result(ok=True, appointment=_serialise(session, appointment))
