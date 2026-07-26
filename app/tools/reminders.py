"""Reminder queries.

The scheduler that *delivers* reminders is Phase 8; this module is the read
side, and it exists now because the golden set uses it to prove the derivation
invariant. Booking then rescheduling and asking what is due at the original
time is the cleanest way to show that a reminder followed its appointment
rather than being left behind pointing at a date nothing happens on.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app import clock
from app.models import Reminder, ReminderStatus


def _serialise(reminder: Reminder) -> dict[str, Any]:
    return {
        "reminder_id": reminder.id,
        "patient_id": reminder.patient_id,
        "appointment_id": reminder.appointment_id,
        "reminder_type": reminder.reminder_type.value,
        "scheduled_at": reminder.scheduled_at.isoformat(),
        "status": reminder.status.value,
        "attempts": reminder.attempts,
        "message": reminder.message,
    }


def list_due_reminders(
    session: Session, *, at: datetime | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Pending reminders whose time has arrived, oldest first.

    Only ``pending`` rows are due. A reminder cancelled alongside its
    appointment must never surface here — that is the whole point of cancelling
    it in the appointment's transaction.
    """
    moment = at or clock.now()
    reminders = (
        session.query(Reminder)
        .filter(
            Reminder.status == ReminderStatus.PENDING,
            Reminder.scheduled_at <= moment,
        )
        .order_by(Reminder.scheduled_at, Reminder.id)
        .limit(limit)
        .all()
    )
    return [_serialise(r) for r in reminders]


def list_patient_reminders(
    session: Session, *, patient_id: int, include_inactive: bool = False
) -> list[dict[str, Any]]:
    """Reminders belonging to one patient, soonest first."""
    query = session.query(Reminder).filter(Reminder.patient_id == patient_id)
    if not include_inactive:
        query = query.filter(Reminder.status == ReminderStatus.PENDING)
    reminders = query.order_by(Reminder.scheduled_at, Reminder.id).all()
    return [_serialise(r) for r in reminders]
