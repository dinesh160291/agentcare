"""One tick of the poll job: reminders that are due, and visits that are over.

**Why polling a table rather than scheduling each reminder.** A per-reminder
in-memory job dies with the process, so a reminder created before a restart
never fires and nothing anywhere records that it didn't. The table is the
source of truth; the job is stateless and asks it the same two questions every
tick. That also makes the whole thing testable without waiting: seed a row,
call :func:`poll_once`, assert.

**Two duties, one tick, because both have the same cause.** Every other
appointment transition has an actor — the patient confirms, staff correct — but
"the visit is now over" is the passage of time, and so is "this reminder is
due". A process that watches the clock is the answer to both, and splitting
them into two schedules would be two things to configure and two things to
forget to start.

**The scheduler acts with no run and no session**, so its consequential writes
go to ``AuditEvent`` and never to ``TraceEvent``. Stated rule: agent activity →
trace, scheduler activity → audit. The turn timeline does not "miss" reminder
deliveries; they are in the other ledger by design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import clock
from app.audit import write_system_audit
from app.config import get_settings
from app.db import SessionLocal
from app.models import (
    Appointment,
    AppointmentStatus,
    FollowUpTaskType,
    Reminder,
    ReminderStatus,
)
from app.scheduler.delivery import deliver
from app.tools.tasks import upsert_followup_task

logger = logging.getLogger("agentcare.scheduler")


@dataclass
class PollResult:
    """What one tick did. Returned so a test can assert on it directly."""

    delivered: list[int] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)
    completed: list[int] = field(default_factory=list)


def poll_once(*, session: Session | None = None) -> PollResult:
    """Run both duties once. Owns its transactions; safe to call from a thread.

    A session may be passed in for a caller that already has one, but the
    default is to open one: the job runs on a scheduler thread with no request
    behind it, and borrowing a request's session is how a background write ends
    up committed — or rolled back — by somebody else's transaction.
    """
    own = session is None
    session = session or SessionLocal()
    try:
        result = PollResult()
        _deliver_due_reminders(session, result)
        _sweep_finished_visits(session, result)
        return result
    finally:
        if own:
            session.close()


# --- reminders -----------------------------------------------------------


def _deliver_due_reminders(session: Session, result: PollResult) -> None:
    """Deliver everything due, one row at a time and one transaction at a time.

    **Per-row isolation is the point.** A batch that aborts on its first bad
    row delivers nothing, and from the outside that is indistinguishable from a
    batch with nothing due — the failure mode that hides itself. So each row
    gets its own try/except and its own commit, and a poisoned one costs only
    itself.
    """
    settings = get_settings()
    moment = clock.now()
    due = (
        session.query(Reminder)
        .filter(
            Reminder.status == ReminderStatus.PENDING,
            Reminder.scheduled_at <= moment,
        )
        .order_by(Reminder.scheduled_at, Reminder.id)
        .limit(settings.reminder_batch_size)
        .all()
    )

    for reminder in due:
        reminder_id = reminder.id
        # **Counted before the attempt, and committed before it.** This is the
        # only ordering that survives a crash *inside* the attempt: a row that
        # reliably kills the process would otherwise never accumulate an
        # attempt and would be retried forever, every tick, at everyone else's
        # expense. The cost of this ordering is that a delivery which fails
        # after succeeding still burns an attempt, which is the cheap direction.
        reminder.attempts += 1
        attempts = reminder.attempts
        session.commit()

        try:
            deliver(session, reminder)
            reminder.status = ReminderStatus.SENT
            reminder.sent_at = clock.now()
            write_system_audit(
                session,
                action="reminder_delivered",
                entity_type="Reminder",
                entity_id=reminder_id,
                metadata={"attempts": attempts},
            )
            session.commit()
            result.delivered.append(reminder_id)
        except Exception as exc:  # noqa: BLE001 - one row must not stop the batch
            session.rollback()
            logger.warning("reminder %s failed to deliver: %s", reminder_id, exc)
            _record_attempt_failure(
                session, reminder_id=reminder_id, attempts=attempts, error=exc,
                cap=settings.max_reminder_attempts, result=result,
            )


def _record_attempt_failure(
    session: Session,
    *,
    reminder_id: int,
    attempts: int,
    error: Exception,
    cap: int,
    result: PollResult,
) -> None:
    """Retire the row once it has burnt the cap. Runs after the rollback.

    It does **not** re-apply the attempt, and that is worth stating because the
    obvious version of this function does: the increment was committed before
    the delivery was attempted, so the rollback never reached it. A line here
    setting ``attempts`` again would be a line that cannot fail — it was
    written, checked by deleting it, and deleted.

    ``attempts`` is still passed in rather than re-read, because what the cap is
    compared against must be the number this tick burnt, not whatever the row
    says by the time the failure is being recorded.
    """
    reminder = session.get(Reminder, reminder_id)
    if reminder is None:  # pragma: no cover - deleted mid-tick
        return

    if attempts >= cap:
        reminder.status = ReminderStatus.FAILED
        write_system_audit(
            session,
            action="reminder_failed",
            entity_type="Reminder",
            entity_id=reminder_id,
            metadata={"attempts": attempts, "cap": cap, "error": str(error)[:200]},
        )
        result.failed.append(reminder_id)
    session.commit()


# --- the visit-completion sweep -------------------------------------------


def _sweep_finished_visits(session: Session, result: PollResult) -> None:
    """Confirmed appointments whose end time has passed become ``completed``.

    Keyed on the **status**, not on the time alone — which is what bounds it.
    A sweep that asked only "is the end time in the past?" would re-fire every
    tick for the rest of the appointment's life; only one tick can move a row
    out of ``confirmed``, so only one tick audits it and only one opens the
    follow-up task.

    Each appointment commits on its own, for the same reason reminders do.
    """
    moment = clock.now()
    finished = (
        session.query(Appointment)
        .join(Appointment.slot)
        .filter(Appointment.status == AppointmentStatus.CONFIRMED)
        .filter(Appointment.slot.has())
        .all()
    )

    for appointment in finished:
        if appointment.slot is None or appointment.slot.end_time > moment:
            continue
        appointment_id = appointment.id
        try:
            appointment.status = AppointmentStatus.COMPLETED
            write_system_audit(
                session,
                action="appointment_completed",
                entity_type="Appointment",
                entity_id=appointment_id,
                metadata={"reference_code": appointment.reference_code},
            )
            # PRD story 32: the follow-up is triggered by the sweep rather than
            # by somebody remembering. `upsert_followup_task` keeps it to one
            # open task per appointment, so a re-run cannot pile them up even
            # if the status guard above were ever loosened.
            upsert_followup_task(
                session,
                patient_id=appointment.patient_id,
                task_type=FollowUpTaskType.POST_VISIT,
                appointment_id=appointment_id,
                details={
                    "reference_code": appointment.reference_code,
                    "department": appointment.department.name
                    if appointment.department
                    else None,
                },
            )
            session.commit()
            result.completed.append(appointment_id)
        except Exception as exc:  # noqa: BLE001 - one row must not stop the sweep
            session.rollback()
            logger.warning("visit sweep failed for appointment %s: %s", appointment_id, exc)


__all__ = ["PollResult", "poll_once"]
