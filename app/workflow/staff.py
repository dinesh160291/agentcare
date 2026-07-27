"""Staff decisions — typed actions, applied by code, with no model involved.

**The staff decision path is LLM-free.** An approve, reject, or redirect is
structured input that code applies directly: no model reads it, interprets it,
or rephrases it. A human decision enters the system the same way every other
consequential state change does — validated, typed, and audited — because the
one thing worse than a model inventing a booking is a model reinterpreting a
person's decision about one.

Three rules that are easy to state and easy to omit:

**A resume is a fresh proposal, not a replay.** Before acting, the preconditions
are re-checked against *current* database state: is this still the patient's
latest run, and is the department still open? A failed precondition never acts
silently — it comes back to staff with the changed context, because staff
approving a dead request is exactly the zombie the withdrawal edge exists to
prevent, arriving by the other door.

**Approval is lazy-continue.** It changes state and writes a notification, and
then it stops. No agent executes, no outbound chat turn is written into an
empty room, and no non-answer clock starts against a patient who does not yet
know a question exists. The patient's return is the inbound that wakes the run
— which also means the slot search happens while they are present, so the times
they are offered are current by construction.

**The two escalation lifecycles are kept apart in code, not just in the UI
copy.** Review escalations are approved or rejected; safety and emergency
escalations are acknowledged and resolved. Nobody "approves" an emergency, and
the way to guarantee that phrase never reaches a screen is to make the
transition impossible rather than merely discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.audit import write_audit
from app.errors import PermissionDenied, RecordNotFound, ValidationFailed
from app.models import (
    Appointment,
    AppointmentStatus,
    Department,
    DocumentStatus,
    Escalation,
    EscalationKind,
    EscalationStatus,
    FollowUpTaskType,
    Notification,
    NotificationKind,
    PatientDocument,
    PlanStep,
    TERMINAL_WORKFLOW_STATUSES,
    TraceAuthor,
    User,
    UserRole,
    WorkflowRun,
    WorkflowStatus,
)
from app.tools.documents import diff_required_documents
from app.tools.tasks import OPEN_ESCALATION_STATUSES, upsert_followup_task
from app.trace import TraceWriter
from app.workflow.plan import advance_plan
from app.workflow.state_machine import transition

#: The closed set of review decisions. Freeform staff input never reaches state.
STAFF_DECISIONS = ("approve", "reject", "redirect")

#: How a review escalation ends, by the decision that ended it.
_ESCALATION_OUTCOME = {
    "approve": EscalationStatus.APPROVED,
    "redirect": EscalationStatus.APPROVED,
    "reject": EscalationStatus.REJECTED,
}

#: Safety escalations have their own vocabulary and it never overlaps.
SAFETY_OUTCOMES = (EscalationStatus.ACKNOWLEDGED, EscalationStatus.RESOLVED)
REVIEW_OUTCOMES = (EscalationStatus.APPROVED, EscalationStatus.REJECTED)


@dataclass(frozen=True)
class StaffDecision:
    """What a decision did. The API and the tests both read this."""

    run_id: int
    action: str
    status: str
    department_name: str | None = None
    notification_id: int | None = None
    escalation_id: int | None = None


def _require_staff(user: User) -> None:
    if user.role is not UserRole.STAFF:
        raise PermissionDenied("Only staff may decide a paused workflow.")


def _notify(
    session: Session,
    *,
    run: WorkflowRun,
    title: str,
    body: str,
) -> Notification:
    """Tell the patient something happened while they were not looking.

    Staff-caused terminal transitions need this and patient-caused ones do not:
    a withdrawal is acknowledged in the reply the patient is already reading,
    while a rejection lands in an empty room.
    """
    notification = Notification(
        patient_id=run.patient_id,
        kind=NotificationKind.STAFF_DECISION,
        title=title,
        body=body,
        workflow_run_id=run.id,
    )
    session.add(notification)
    session.flush()
    return notification


def _open_escalation(session: Session, run_id: int) -> Escalation | None:
    return (
        session.query(Escalation)
        .filter(
            Escalation.workflow_run_id == run_id,
            Escalation.status.in_(OPEN_ESCALATION_STATUSES),
        )
        .order_by(Escalation.id)
        .first()
    )


def _revalidate(session: Session, run: WorkflowRun, department: Department | None) -> None:
    """Re-check the world against what the decision assumes about it.

    :raises ValidationFailed: with what changed, so staff see the new context
        rather than a decision quietly landing on stale ground.
    """
    newer = (
        session.query(WorkflowRun)
        .filter(
            WorkflowRun.patient_id == run.patient_id,
            WorkflowRun.id > run.id,
            WorkflowRun.status.notin_(list(TERMINAL_WORKFLOW_STATUSES)),
        )
        .order_by(WorkflowRun.id.desc())
        .first()
    )
    if newer is not None:
        raise ValidationFailed(
            f"Run {run.id} is no longer this patient's latest request — run "
            f"{newer.id} has replaced it. Review that one instead."
        )

    if department is not None and not department.active:
        raise ValidationFailed(
            f"{department.name} is no longer accepting appointments. "
            "Redirect this request to another department."
        )


def _department_for(
    session: Session, run: WorkflowRun, action: str, department_name: str | None
) -> Department:
    """The department this decision routes to, validated against the table.

    A redirect names one; an approval accepts the candidate the routing step
    recorded on the run. Either way the name is checked against the
    ``Department`` table rather than trusted — staff typing is input too.
    """
    if action == "redirect":
        if not department_name:
            raise ValidationFailed("A redirect must name the department to route to.")
        department = (
            session.query(Department)
            .filter(Department.name == department_name)
            .one_or_none()
        )
        if department is None:
            raise ValidationFailed(f"{department_name!r} is not a department.")
        return department

    proposed = (run.state or {}).get("proposed_department_id")
    if proposed is None:
        raise ValidationFailed(
            "No department was proposed for this request, so there is nothing "
            "to approve. Use redirect and name the department."
        )
    department = session.get(Department, int(proposed))
    if department is None:  # pragma: no cover - a deleted department
        raise ValidationFailed("The proposed department no longer exists.")
    return department


def apply_staff_decision(
    session: Session,
    *,
    staff: User,
    run_id: int,
    action: str,
    writer: TraceWriter,
    department_name: str | None = None,
    note: str = "",
) -> StaffDecision:
    """Apply one typed staff decision to a paused run.

    Does not commit — the transition, the escalation's close, the notification,
    and the audit rows belong to one transaction.
    """
    _require_staff(staff)

    chosen = str(action or "").strip().lower()
    if chosen not in STAFF_DECISIONS:
        raise ValidationFailed(
            f"{action!r} is not a staff decision. Use one of: "
            f"{', '.join(STAFF_DECISIONS)}."
        )

    run = session.get(WorkflowRun, run_id)
    if run is None:
        raise RecordNotFound("WorkflowRun", run_id)

    writer.bind_run(run.id)
    writer.inbound(chosen, author=TraceAuthor.STAFF_ACTION)

    # Everything that can refuse this decision runs before anything applies it,
    # and a refusal still closes the turn: a staff action that vanished from
    # the timeline is the same hole as a patient turn that did.
    try:
        if run.status is not WorkflowStatus.PENDING_REVIEW:
            raise ValidationFailed(
                f"Run {run_id} is {run.status.value}, not awaiting review. "
                "Nothing is pending a decision on it."
            )

        escalation = _open_escalation(session, run.id)
        if escalation is not None and escalation.kind is EscalationKind.SAFETY:
            raise ValidationFailed(
                "A safety escalation is acknowledged and resolved, never "
                "approved or rejected. Use the safety queue."
            )

        department = (
            None
            if chosen == "reject"
            else _department_for(session, run, chosen, department_name)
        )
        _revalidate(session, run, department)
    except ValidationFailed as refused:
        writer.guard_verdict(
            "staff_decision_precondition",
            passed=False,
            detail={"action": chosen, "problem": str(refused)},
        )
        write_audit(
            session,
            action="staff_decision_refused",
            entity_type="workflow_run",
            entity_id=run.id,
            actor=staff,
            metadata={"action": chosen, "problem": str(refused)},
        )
        writer.outbound(str(refused), author=TraceAuthor.GUARD)
        raise

    writer.guard_verdict(
        "staff_decision_precondition", passed=True, detail={"action": chosen}
    )

    if chosen == "reject":
        result = _reject(session, run=run, staff=staff, writer=writer, note=note)
    else:
        result = _approve(
            session,
            run=run,
            staff=staff,
            writer=writer,
            department=department,
            action=chosen,
        )

    if escalation is not None:
        escalation.status = _ESCALATION_OUTCOME[chosen]
        escalation.reviewed_by = staff.id
        escalation.resolution_note = note
        session.flush()
        write_audit(
            session,
            action="escalation_decided",
            entity_type="Escalation",
            entity_id=escalation.id,
            actor=staff,
            metadata={"decision": chosen, "status": escalation.status.value},
        )

    return StaffDecision(
        run_id=run.id,
        action=chosen,
        status=run.status.value,
        department_name=department.name if department is not None else None,
        notification_id=result,
        escalation_id=escalation.id if escalation is not None else None,
    )


def _approve(
    session: Session,
    *,
    run: WorkflowRun,
    staff: User,
    writer: TraceWriter,
    department: Department,
    action: str,
) -> int:
    """Approve or redirect: the routing decision becomes the human's, and stops.

    The route step is marked done here on purpose. Leaving it open would send
    the resumed run straight back through routing, which would reach the same
    ambiguity, escalate again, and put the request back in the queue a human
    has just taken it out of.
    """
    state = dict(run.state or {})
    state["department_id"] = department.id
    state["department_name"] = department.name
    state["routed_by"] = "staff"
    run.state = state

    if PlanStep.ROUTE.value in (run.plan or []):
        advance_plan(run, PlanStep.ROUTE)

    transition(
        session,
        run,
        to=WorkflowStatus.IN_PROGRESS,
        trigger=f"staff_{action}",
        writer=writer,
        actor=staff,
        detail={"department": department.name},
    )

    notification = _notify(
        session,
        run=run,
        title=f"Your request has been routed to {department.name}",
        body=(
            f"A member of staff has reviewed your request and routed it to "
            f"{department.name}. Open the chat to carry on and choose a time."
        ),
    )
    # Lazy-continue: state changed, patient told, and nothing else runs.
    # Humans resume conversations; state changes do not.
    writer.outbound(notification.title, author=TraceAuthor.TEMPLATE)
    return notification.id


def _reject(
    session: Session,
    *,
    run: WorkflowRun,
    staff: User,
    writer: TraceWriter,
    note: str,
) -> int:
    transition(
        session,
        run,
        to=WorkflowStatus.REJECTED,
        trigger="staff_reject",
        writer=writer,
        actor=staff,
        detail={"note": note},
    )
    notification = _notify(
        session,
        run=run,
        title="A member of staff has closed your request",
        body=(
            note
            or "A member of staff has reviewed your request and closed it. "
            "Please get in touch if you would like to discuss it."
        ),
    )
    writer.outbound(notification.title, author=TraceAuthor.TEMPLATE)
    return notification.id


def resolve_escalation(
    session: Session,
    *,
    staff: User,
    escalation_id: int,
    status: EscalationStatus,
    note: str = "",
) -> dict:
    """Acknowledge or resolve a **safety** escalation.

    Deliberately a separate function with a separate vocabulary. Sharing one
    with :func:`apply_staff_decision` would put "approve" one enum member away
    from an emergency, and the whole reason these lifecycles are named apart is
    that no interface should ever be able to say a person's chest pain was
    approved.
    """
    _require_staff(staff)

    escalation = session.get(Escalation, escalation_id)
    if escalation is None:
        raise RecordNotFound("Escalation", escalation_id)

    permitted = (
        SAFETY_OUTCOMES if escalation.kind is EscalationKind.SAFETY else REVIEW_OUTCOMES
    )
    if status not in permitted:
        raise ValidationFailed(
            f"A {escalation.kind.value} escalation cannot be set to "
            f"{status.value}. Permitted: {', '.join(s.value for s in permitted)}."
        )

    escalation.status = status
    escalation.reviewed_by = staff.id
    escalation.resolution_note = note
    session.flush()
    write_audit(
        session,
        action="escalation_resolved",
        entity_type="Escalation",
        entity_id=escalation.id,
        actor=staff,
        metadata={"status": status.value, "kind": escalation.kind.value},
    )
    return {
        "escalation_id": escalation.id,
        "status": escalation.status.value,
        "kind": escalation.kind.value,
    }


#: What staff may do with a flagged document.
DOCUMENT_RESOLUTIONS = ("accept", "reclassify", "reject")


def resolve_document(
    session: Session,
    *,
    staff: User,
    document_id: int,
    action: str,
    corrected_type: str | None = None,
    note: str = "",
) -> dict:
    """Close a flagged document, and re-derive everything that depended on it.

    The second half is the point. A document's type decides which requirements
    it satisfies, so changing it changes the required-documents diff — and the
    **derivation invariant** says a derived row gets its update rule applied in
    the source's transaction, not eventually. Without that, a patient whose
    misfiled X-ray is reclassified as an ECG keeps a follow-up task telling
    them to supply the ECG that is now on file.

    The diff is re-run rather than adjusted, and the task upserted rather than
    inserted, so this stays correct however many times staff change their mind.
    """
    _require_staff(staff)

    chosen = str(action or "").strip().lower()
    if chosen not in DOCUMENT_RESOLUTIONS:
        raise ValidationFailed(
            f"{action!r} is not a document resolution. Use one of: "
            f"{', '.join(DOCUMENT_RESOLUTIONS)}."
        )

    document = session.get(PatientDocument, document_id)
    if document is None:
        raise RecordNotFound("PatientDocument", document_id)

    if chosen == "accept":
        # The declared type stands: the model read it wrong, or the content is
        # thinner than the label. Either way a human has looked.
        document.status = DocumentStatus.VERIFIED
    elif chosen == "reclassify":
        if not (corrected_type or "").strip():
            raise ValidationFailed("A reclassification must name the correct type.")
        document.document_type = corrected_type.strip()
        document.status = DocumentStatus.VERIFIED
    else:
        document.status = DocumentStatus.REJECTED

    document.verification_note = (note or "").strip()[:500] or document.verification_note
    session.flush()

    write_audit(
        session,
        action="document_resolved",
        entity_type="PatientDocument",
        entity_id=document.id,
        actor=staff,
        metadata={
            "resolution": chosen,
            "document_type": document.document_type,
            "status": document.status.value,
        },
    )

    rediffed = _rediff_for(session, document=document, staff=staff)
    return {
        "document_id": document.id,
        "resolution": chosen,
        "status": document.status.value,
        "document_type": document.document_type,
        "tasks_updated": rediffed,
    }


def _rediff_for(session: Session, *, document: PatientDocument, staff: User) -> list[dict]:
    """Re-run the required-docs diff for every department this patient is
    booked into, and upsert the open task from *its* result.

    Scoped to departments the patient actually has an appointment in: a diff
    against a department nobody booked would open a task for paperwork nobody
    asked for.
    """
    appointments = (
        session.query(Appointment)
        .filter(
            Appointment.patient_id == document.patient_id,
            Appointment.status.in_(
                (AppointmentStatus.PENDING, AppointmentStatus.CONFIRMED)
            ),
        )
        .all()
    )

    updated: list[dict] = []
    for appointment in appointments:
        diff = diff_required_documents(
            session,
            patient_id=document.patient_id,
            department_id=appointment.department_id,
        )
        outcome = upsert_followup_task(
            session,
            patient_id=document.patient_id,
            task_type=FollowUpTaskType.MISSING_DOCUMENTS,
            details={"missing": diff["missing_mandatory"]},
            appointment_id=appointment.id,
            close_when_empty_key="missing",
            actor=staff,
        )
        updated.append(
            {
                "appointment_id": appointment.id,
                "missing": diff["missing_mandatory"],
                "closed": outcome["closed"],
            }
        )
    return updated


__all__ = [
    "DOCUMENT_RESOLUTIONS",
    "REVIEW_OUTCOMES",
    "SAFETY_OUTCOMES",
    "STAFF_DECISIONS",
    "StaffDecision",
    "apply_staff_decision",
    "resolve_document",
    "resolve_escalation",
]
