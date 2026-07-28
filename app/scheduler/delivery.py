"""Where a reminder actually goes.

Two channels, and the split is deliberate. The **in-app notification** is the
real one — a row the patient's Notifications panel reads, and the thing story 31
means by "reminders actually fire". The **log line** stands in for SMS or email:
this project sends no external messages, and a stub that pretended to would be
a hardcoded final response wearing a courier's uniform.

The insert is idempotent by construction rather than by checking first.
``Notification.reminder_id`` carries a unique constraint, so a second delivery
of the same reminder loses a race with itself instead of reaching the patient
twice — which is what makes "deliver, then mark" a safe ordering to crash in
the middle of.
"""

from __future__ import annotations

import logging

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Notification, NotificationKind, Reminder

logger = logging.getLogger("agentcare.reminders")


def deliver(session: Session, reminder: Reminder) -> Notification | None:
    """Put the reminder in front of the patient. ``None`` if it already was.

    Does not mark the reminder sent and does not commit: the caller owns both,
    because the ordering between them is the property being protected.
    """
    existing = (
        session.query(Notification)
        .filter(Notification.reminder_id == reminder.id)
        .one_or_none()
    )
    if existing is not None:
        # A previous tick delivered this and died before marking it. The
        # re-attempt is the bounded cost of that ordering; making it visible
        # to the patient would be the unbounded one.
        logger.info("reminder %s already delivered; skipping", reminder.id)
        return None

    notification = Notification(
        patient_id=reminder.patient_id,
        kind=NotificationKind.REMINDER,
        title="Appointment reminder",
        body=reminder.message,
        reminder_id=reminder.id,
    )
    session.add(notification)
    try:
        session.flush()
    except IntegrityError:
        # The unique constraint, doing the job the query above cannot promise
        # under concurrency. Two pollers is not a supported deployment, but a
        # guarantee that holds only when nobody tests it is not a guarantee.
        session.rollback()
        logger.info("reminder %s was delivered concurrently; skipping", reminder.id)
        return None

    # The simulated outbound channel. Deliberately a log line and deliberately
    # named: nothing here contacts anyone, and the demo says so.
    logger.info(
        "SIMULATED NOTIFICATION patient=%s reminder=%s: %s",
        reminder.patient_id,
        reminder.id,
        reminder.message,
    )
    return notification


__all__ = ["deliver"]
