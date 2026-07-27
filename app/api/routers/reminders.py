"""Reminders, follow-up tasks, and in-app notifications.

All three are *derived* rows: a reminder comes from an appointment, a task from
a document shortfall or a completed visit, a notification from a staff decision
or a delivered reminder. Nothing here creates any of them, which is the point —
a derived row created by hand is a row with no update rule.

The one write is marking a notification read, which is the reader's own fact
about their own row and derives nothing.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.dependencies import CurrentUser, PatientUser
from app.auth.ownership import get_owned_or_404, patient_profile_for
from app.db import get_session
from app.models import Notification
from app.tools import list_open_tasks, list_patient_reminders

router = APIRouter(tags=["reminders"])

DbSession = Annotated[Session, Depends(get_session)]


def _notification_out(row: Notification) -> dict[str, Any]:
    return {
        "notification_id": row.id,
        "kind": row.kind.value,
        "title": row.title,
        "body": row.body,
        "read": row.read,
        "reminder_id": row.reminder_id,
        "workflow_run_id": row.workflow_run_id,
        "created_at": row.created_at.isoformat(),
    }


@router.get("/reminders")
def list_reminders(
    user: PatientUser, session: DbSession, include_inactive: bool = False
) -> list[dict[str, Any]]:
    """Pending reminders by default; ``include_inactive`` adds sent and cancelled."""
    profile = patient_profile_for(session, user)
    return list_patient_reminders(
        session, patient_id=profile.id, include_inactive=include_inactive
    )


@router.get("/tasks")
def list_tasks(user: PatientUser, session: DbSession) -> list[dict[str, Any]]:
    """Open follow-up tasks — missing documents, post-visit, missed visit."""
    profile = patient_profile_for(session, user)
    return list_open_tasks(session, patient_id=profile.id)


@router.get("/notifications")
def list_notifications(user: PatientUser, session: DbSession) -> list[dict[str, Any]]:
    """The caller's in-app inbox, newest first."""
    profile = patient_profile_for(session, user)
    rows = (
        session.query(Notification)
        .filter(Notification.patient_id == profile.id)
        .order_by(Notification.id.desc())
        .all()
    )
    return [_notification_out(row) for row in rows]


@router.post("/notifications/{notification_id}/read")
def mark_read(
    notification_id: int, user: CurrentUser, session: DbSession
) -> dict[str, Any]:
    """Mark one notification read. Ownership first, then the write."""
    notification = get_owned_or_404(session, Notification, notification_id, user)
    notification.read = True
    session.commit()
    session.refresh(notification)
    return _notification_out(notification)
