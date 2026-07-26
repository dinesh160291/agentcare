"""The workflow state machine — the deterministic machine everything hangs off.

``WorkflowRun.status`` drives pause/resume, escalation resume, and
confirm-before-commit. The model never moves it; only this module does, and
only along an edge in :data:`LEGAL_TRANSITIONS`.

Three rules are enforced here rather than trusted:

**Every transition is a compare-and-swap.** The new status is set only where
the row still holds the expected old one. Zero affected rows means another
request got there first, and the loser treats it as a no-op — never a crash and
never a double-fire. The motivating trace is a double-clicked Confirm: two
requests both read ``pending_confirmation``, and while the booking transaction
would catch the resulting double-book, the CAS stops it a layer earlier and
does so on every edge rather than only the dangerous one.

**Every transition writes both ledgers.** An ``AuditEvent`` for the row's
history and a ``TraceEvent`` for the turn's story. Neither is optional, which
is why ``writer`` has no default: a transition that appears in one ledger and
not the other is a transition somebody will later fail to explain.

**Cancellation names its reason.** The staff queue separates "withdrawn while
pending" from "superseded" with a query over terminal rows, so the reason has
to be recorded at the single moment it is known.

Callers must never assign ``run.status`` directly. Doing so bypasses the CAS,
both ledgers, and the table.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.audit import write_audit
from app.errors import InvalidTransition, ValidationFailed
from app.models import CancellationReason, User, WorkflowRun, WorkflowStatus
from app.trace import TraceWriter

_S = WorkflowStatus

#: The PRD's pinned transition table. Any state change not in here raises and
#: is audited as an illegal-transition attempt. A message that triggers no
#: transition is legal — the table governs *changes*, so it has no self-edges.
LEGAL_TRANSITIONS: dict[WorkflowStatus, frozenset[WorkflowStatus]] = {
    _S.IN_PROGRESS: frozenset(
        {
            _S.PENDING_CONFIRMATION,  # Appointment agent proposes book/reschedule/cancel
            _S.PENDING_REVIEW,        # low-confidence or unsupported routing
            _S.CANCELLED,             # patient withdraws
            _S.ESCALATED,             # safety verdict
            _S.COMPLETED,             # all plan steps done
            _S.FAILED,                # unrecoverable after bounded retries
        }
    ),
    _S.PENDING_CONFIRMATION: frozenset(
        {
            _S.IN_PROGRESS,  # patient confirms (commit + continue) or declines
            _S.CANCELLED,    # patient withdraws instead of answering
            _S.ESCALATED,    # safety verdict
        }
    ),
    _S.PENDING_REVIEW: frozenset(
        {
            _S.IN_PROGRESS,  # staff approve or redirect (typed action)
            _S.REJECTED,     # staff reject (typed action)
            _S.CANCELLED,    # patient withdraws while awaiting review
            _S.ESCALATED,    # safety verdict
        }
    ),
    # Terminal for automation. `escalated` in particular hands ownership to the
    # Escalation record: a withdrawal must not close a run that is in front of
    # a human because the safety screen fired.
    _S.COMPLETED: frozenset(),
    _S.REJECTED: frozenset(),
    _S.FAILED: frozenset(),
    _S.CANCELLED: frozenset(),
    _S.ESCALATED: frozenset(),
}

#: Runs are created directly in one of these two states — never anywhere else.
#: ``escalated`` at birth exists so an emergency on a session's opening message
#: still has a run to key its Escalation to, keeping that FK non-nullable.
INITIAL_STATUSES = frozenset({_S.IN_PROGRESS, _S.ESCALATED})


@dataclass(frozen=True)
class TransitionResult:
    """What happened. ``applied`` is False only when the CAS lost its race."""

    applied: bool
    from_status: WorkflowStatus
    to_status: WorkflowStatus
    run_id: int


def is_legal(from_status: WorkflowStatus, to_status: WorkflowStatus) -> bool:
    return to_status in LEGAL_TRANSITIONS.get(from_status, frozenset())


def _check_reason(
    to_status: WorkflowStatus, reason: CancellationReason | None
) -> None:
    if to_status is _S.CANCELLED and reason is None:
        raise ValidationFailed(
            "Cancelling a run requires a CancellationReason: the staff queue "
            "distinguishes withdrawal from supersede by query, not by state."
        )
    if to_status is not _S.CANCELLED and reason is not None:
        raise ValidationFailed(
            f"A cancellation reason is meaningless on a transition to {to_status}."
        )


def transition(
    session: Session,
    run: WorkflowRun,
    *,
    to: WorkflowStatus,
    trigger: str,
    writer: TraceWriter,
    reason: CancellationReason | None = None,
    actor: User | None = None,
    detail: dict | None = None,
) -> TransitionResult:
    """Move a run along one edge of the machine.

    Does not commit — the transition, its audit row, and whatever change
    prompted it belong in one transaction, so the caller owns it.

    :raises InvalidTransition: the edge is not in the table (audited first).
    :raises ValidationFailed: a cancellation without a reason, or a reason
        attached to an edge that is not a cancellation.
    """
    from_status = run.status

    # Attribute this event to the run when the turn opened before the run was
    # known. An existing binding is never stomped: during a supersede the
    # writer is bound to the run being cancelled, and that cancellation belongs
    # to *that* run, not to the one about to replace it.
    if writer.workflow_run_id is None:
        writer.bind_run(run.id)

    # Legality first. Whether the edge exists is the more fundamental question;
    # the cancellation reason validates the payload of an edge already accepted.
    if not is_legal(from_status, to):
        writer.validation(
            "workflow_transition",
            accepted=False,
            detail={"from": from_status.value, "to": to.value, "trigger": trigger},
        )
        write_audit(
            session,
            action="workflow_transition_illegal",
            entity_type="workflow_run",
            entity_id=run.id,
            actor=actor,
            metadata={"from": from_status.value, "to": to.value, "trigger": trigger},
        )
        raise InvalidTransition(from_status, to)

    _check_reason(to, reason)

    values: dict[str, object] = {"status": to}
    if reason is not None:
        values["cancellation_reason"] = reason.value

    # The compare-and-swap. `fetch` makes the database the authority on whether
    # the row still holds the expected status — an in-Python evaluation would
    # happily match a status the caller had already changed in memory.
    result = session.execute(
        update(WorkflowRun)
        .where(WorkflowRun.id == run.id, WorkflowRun.status == from_status)
        .values(**values)
        .execution_options(synchronize_session="fetch")
    )
    applied = result.rowcount == 1

    writer.transition(
        from_status=from_status.value,
        to_status=to.value,
        reason=reason.value if reason is not None else "",
        trigger=trigger,
        applied=applied,
        detail=detail or {},
    )
    write_audit(
        session,
        action="workflow_transition" if applied else "workflow_transition_lost_race",
        entity_type="workflow_run",
        entity_id=run.id,
        actor=actor,
        metadata={
            "from": from_status.value,
            "to": to.value,
            "trigger": trigger,
            "reason": reason.value if reason is not None else None,
            **(detail or {}),
        },
    )

    if applied:
        # `fetch` expired the updated columns; re-read so the caller's object
        # agrees with the row it just changed.
        session.refresh(run)

    return TransitionResult(
        applied=applied, from_status=from_status, to_status=to, run_id=run.id
    )


def create_run(
    session: Session,
    *,
    patient_id: int,
    status: WorkflowStatus,
    trigger: str,
    writer: TraceWriter,
    request_text: str = "",
    plan: list[str] | None = None,
    session_id: str | None = None,
    actor: User | None = None,
) -> WorkflowRun:
    """Create a run in one of the two permitted initial states.

    Binds ``writer`` to the new run: a turn opens before its run exists (the
    inbound event and the guard verdicts come first), and everything after this
    point has to be attributable to the run or the timeline splits in half.
    """
    if status not in INITIAL_STATUSES:
        raise ValidationFailed(
            f"A run cannot be created in {status}. Runs are born in_progress, "
            "or escalated when the safety screen fires on an opening message."
        )

    run = WorkflowRun(
        patient_id=patient_id,
        status=status,
        request_text=request_text,
        plan=plan or [],
        completed_steps=[],
        state={},
        session_id=session_id or writer.session_id,
    )
    session.add(run)
    session.flush()

    writer.bind_run(run.id)
    writer.transition(
        from_status=None, to_status=status.value, reason="", trigger=trigger, applied=True
    )
    write_audit(
        session,
        action="workflow_run_created",
        entity_type="workflow_run",
        entity_id=run.id,
        actor=actor,
        metadata={"status": status.value, "trigger": trigger},
    )
    return run
