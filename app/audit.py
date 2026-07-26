"""Audit logging.

Every state transition writes an ``AuditEvent``, and so does every *denied*
attempt — a denial that leaves no trace is indistinguishable from a request
that never happened, which is exactly the record you want when someone probes
another patient's ids.

The scheduler also writes here. It acts with no run and no session, so it has
nothing to trace against; the audit log is its channel. That is the trace/audit
split: traces explain a conversation turn, audit explains a row's history.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.models import AuditEvent, User


def write_audit(
    session: Session,
    *,
    action: str,
    entity_type: str,
    entity_id: int | None = None,
    actor: User | None = None,
    actor_kind: str = "user",
    metadata: dict[str, Any] | None = None,
    flush: bool = True,
) -> AuditEvent:
    """Record an action. Caller owns the transaction.

    Deliberately does not commit: an audit row must land in the same
    transaction as the change it describes, or a rollback leaves the log
    claiming something happened that did not.
    """
    event = AuditEvent(
        actor_id=actor.id if actor is not None else None,
        actor_kind=actor_kind,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        event_metadata=metadata or {},
    )
    session.add(event)
    if flush:
        session.flush()
    return event


def write_system_audit(
    session: Session,
    *,
    action: str,
    entity_type: str,
    entity_id: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditEvent:
    """Audit an action with no human actor (the scheduler's sweep, retries)."""
    return write_audit(
        session,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        actor=None,
        actor_kind="system",
        metadata=metadata,
    )


def audit_denied_access(
    session: Session,
    *,
    actor: User,
    entity_type: str,
    entity_id: object,
    reason: str = "ownership",
) -> AuditEvent:
    """Record a denied access attempt.

    Called on the ownership path before raising ``RecordNotFound``. The caller
    sees a 404 and learns nothing; the audit log records precisely what was
    probed and by whom.
    """
    return write_audit(
        session,
        action="access_denied",
        entity_type=entity_type,
        entity_id=entity_id if isinstance(entity_id, int) else None,
        actor=actor,
        metadata={"reason": reason, "requested_id": str(entity_id)},
    )
