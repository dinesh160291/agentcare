"""Follow-up tasks and escalations — both upserts, never blind inserts.

These are the two record types an agent can be provoked into creating over and
over. A frightened patient typing "chest pain" five times must produce one
queue item with five recorded triggers, not five queue items; and re-running a
required-documents diff must update the open task rather than stack another
copy beside it.

That is the boundedness invariant applied to record creation: every automated
writer needs a growth rule, and here the rule is *one open record per subject*.
Repeats attach to it.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from app.audit import write_audit
from app.models import (
    Escalation,
    EscalationKind,
    EscalationStatus,
    FollowUpTask,
    FollowUpTaskStatus,
    FollowUpTaskType,
    User,
)

#: Escalation states that still belong to a human.
OPEN_ESCALATION_STATUSES = (EscalationStatus.OPEN, EscalationStatus.ACKNOWLEDGED)


def _serialise_task(task: FollowUpTask) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "patient_id": task.patient_id,
        "appointment_id": task.appointment_id,
        "task_type": task.task_type.value,
        "status": task.status.value,
        "details": task.details,
        "due_date": task.due_date.isoformat() if task.due_date else None,
    }


def _serialise_escalation(escalation: Escalation) -> dict[str, Any]:
    return {
        "escalation_id": escalation.id,
        "workflow_run_id": escalation.workflow_run_id,
        "kind": escalation.kind.value,
        "status": escalation.status.value,
        "reason": escalation.reason,
        "occurrence_count": escalation.occurrence_count,
        "latest_message": escalation.latest_message,
    }


def find_open_task(
    session: Session,
    *,
    patient_id: int,
    task_type: FollowUpTaskType,
    appointment_id: int | None = None,
) -> FollowUpTask | None:
    """The single open task for this subject, if one exists."""
    return (
        session.query(FollowUpTask)
        .filter(
            FollowUpTask.patient_id == patient_id,
            FollowUpTask.task_type == task_type,
            FollowUpTask.appointment_id == appointment_id,
            FollowUpTask.status == FollowUpTaskStatus.OPEN,
        )
        .order_by(FollowUpTask.id)
        .first()
    )


def upsert_followup_task(
    session: Session,
    *,
    patient_id: int,
    task_type: FollowUpTaskType,
    details: dict[str, Any],
    appointment_id: int | None = None,
    due_date: date | None = None,
    close_when_empty_key: str | None = None,
    actor: User | None = None,
) -> dict[str, Any]:
    """Create or update the one open task for this subject.

    ``close_when_empty_key`` names an entry in ``details`` whose emptiness means
    the task is done — for a missing-documents task, the list of what is still
    missing. When the diff comes back clean the task closes itself, so nothing
    has to remember to go back and tidy up. Re-running the diff updates the
    open task in place; it never spawns a sibling.
    """
    existing = find_open_task(
        session, patient_id=patient_id, task_type=task_type, appointment_id=appointment_id
    )

    should_close = (
        close_when_empty_key is not None and not details.get(close_when_empty_key)
    )

    if existing is None:
        if should_close:
            # Nothing outstanding and nothing open: creating a task purely to
            # close it would be noise in the patient's list.
            return {"created": False, "updated": False, "closed": False, "task": None}

        task = FollowUpTask(
            patient_id=patient_id,
            appointment_id=appointment_id,
            task_type=task_type,
            status=FollowUpTaskStatus.OPEN,
            details=details,
            due_date=due_date,
        )
        session.add(task)
        session.flush()
        write_audit(
            session,
            action="followup_task_opened",
            entity_type="FollowUpTask",
            entity_id=task.id,
            actor=actor,
            actor_kind="user" if actor else "system",
            metadata={"task_type": task_type.value},
        )
        return {"created": True, "updated": False, "closed": False,
                "task": _serialise_task(task)}

    existing.details = details
    if due_date is not None:
        existing.due_date = due_date

    if should_close:
        existing.status = FollowUpTaskStatus.CLOSED
        write_audit(
            session,
            action="followup_task_closed",
            entity_type="FollowUpTask",
            entity_id=existing.id,
            actor=actor,
            actor_kind="user" if actor else "system",
            metadata={"task_type": task_type.value},
        )
        session.flush()
        return {"created": False, "updated": True, "closed": True,
                "task": _serialise_task(existing)}

    session.flush()
    return {"created": False, "updated": True, "closed": False,
            "task": _serialise_task(existing)}


def close_followup_tasks(
    session: Session,
    *,
    patient_id: int,
    task_type: FollowUpTaskType,
    appointment_id: int | None = None,
    actor: User | None = None,
) -> int:
    """Close every open task for a subject. Returns how many were closed."""
    tasks = (
        session.query(FollowUpTask)
        .filter(
            FollowUpTask.patient_id == patient_id,
            FollowUpTask.task_type == task_type,
            FollowUpTask.appointment_id == appointment_id,
            FollowUpTask.status == FollowUpTaskStatus.OPEN,
        )
        .all()
    )
    for task in tasks:
        task.status = FollowUpTaskStatus.CLOSED
        write_audit(
            session,
            action="followup_task_closed",
            entity_type="FollowUpTask",
            entity_id=task.id,
            actor=actor,
            actor_kind="user" if actor else "system",
            metadata={"task_type": task_type.value},
        )
    session.flush()
    return len(tasks)


def list_open_tasks(session: Session, *, patient_id: int) -> list[dict[str, Any]]:
    """Every open task for a patient, oldest first."""
    tasks = (
        session.query(FollowUpTask)
        .filter(
            FollowUpTask.patient_id == patient_id,
            FollowUpTask.status == FollowUpTaskStatus.OPEN,
        )
        .order_by(FollowUpTask.id)
        .all()
    )
    return [_serialise_task(task) for task in tasks]


def create_escalation(
    session: Session,
    *,
    workflow_run_id: int,
    kind: EscalationKind,
    reason: str,
    message: str | None = None,
    actor: User | None = None,
) -> dict[str, Any]:
    """Open an escalation for a run, or attach to the one already open.

    Attaching rather than multiplying keeps the staff queue readable. Five
    triggers on one run become one item with ``occurrence_count`` five — and
    the repetition is still visible, as urgency context on a single record
    rather than five near-identical rows a human has to reconcile.

    Every trigger is audited separately, so nothing is lost by not creating a
    row for it.
    """
    existing = (
        session.query(Escalation)
        .filter(
            Escalation.workflow_run_id == workflow_run_id,
            Escalation.status.in_(OPEN_ESCALATION_STATUSES),
        )
        .order_by(Escalation.id)
        .first()
    )

    if existing is not None:
        existing.occurrence_count += 1
        if message is not None:
            existing.latest_message = message
        session.flush()
        write_audit(
            session,
            action="escalation_retriggered",
            entity_type="Escalation",
            entity_id=existing.id,
            actor=actor,
            actor_kind="user" if actor else "system",
            metadata={"kind": kind.value, "occurrence_count": existing.occurrence_count},
        )
        return {"created": False, "attached": True,
                "escalation": _serialise_escalation(existing)}

    escalation = Escalation(
        workflow_run_id=workflow_run_id,
        kind=kind,
        reason=reason,
        status=EscalationStatus.OPEN,
        occurrence_count=1,
        latest_message=message,
    )
    session.add(escalation)
    session.flush()
    write_audit(
        session,
        action="escalation_opened",
        entity_type="Escalation",
        entity_id=escalation.id,
        actor=actor,
        actor_kind="user" if actor else "system",
        metadata={"kind": kind.value, "reason": reason},
    )
    return {"created": True, "attached": False,
            "escalation": _serialise_escalation(escalation)}


def list_open_escalations(session: Session) -> list[dict[str, Any]]:
    """The staff queue: every escalation still awaiting a human, oldest first."""
    escalations = (
        session.query(Escalation)
        .filter(Escalation.status.in_(OPEN_ESCALATION_STATUSES))
        .order_by(Escalation.id)
        .all()
    )
    return [_serialise_escalation(e) for e in escalations]
