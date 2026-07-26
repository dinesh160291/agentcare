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
    """Record a denied access attempt, durably.

    Called on the ownership path before raising ``RecordNotFound``. The caller
    sees a 404 and learns nothing; the audit log records precisely what was
    probed and by whom.

    Unlike every other audit write, this one **commits**. The rest of the
    module deliberately does not, because an audit row describing a change must
    roll back with that change. A denial describes no change: it happens on a
    request that is about to fail, and a request-scoped session is closed
    without committing — so a denial left in that transaction is rolled back,
    and the system audits nothing while appearing to audit everything.

    Committing on a *separate* session would isolate this better, but SQLite
    permits one writer at a time: the request session may already hold the
    write lock, and the second connection would block until it timed out.

    Consequence for callers: run ownership guards **before** mutations, which
    is where a guard belongs anyway. Pending writes in the same session would
    be committed by this call.
    """
    event = write_audit(
        session,
        action="access_denied",
        entity_type=entity_type,
        entity_id=entity_id if isinstance(entity_id, int) else None,
        actor=actor,
        metadata={"reason": reason, "requested_id": str(entity_id)},
    )
    session.commit()
    return event
