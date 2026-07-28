"""Staff operations: the queues, the typed decisions, and oversight.

**This router owns transactions that its callees deliberately do not.**
``apply_staff_decision``, ``resolve_escalation`` and ``resolve_document`` each
write a state change, its audit row, and (where there is a turn) its trace rows,
and none of them commits — because those writes belong in one transaction with
whatever the caller is doing. The caller is here. A handler that forgets the
commit returns a cheerful 200 and changes nothing.

**Two refusals, handled differently, and the difference is the point.**

``apply_staff_decision`` writes its own audit and trace rows *before* raising
``ValidationFailed``: a refused decision is a thing a human did, and it belongs
in the timeline. So that call site catches the exception, **commits**, and
re-raises as a 422. It cannot be left to the exception handler in
:mod:`app.api.errors`, because FastAPI unwinds the dependency stack — closing
the session — before a handler runs.

``resolve_document`` refuses *before* writing anything, so its
``ValidationFailed`` has nothing to preserve and is left to that handler, where
the uncommitted session is exactly the right outcome.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.routers.workflow import run_out
from app.api.schemas import (
    ActiveToggle,
    AuditEventOut,
    DocumentResolution,
    EscalationResolution,
    RunOut,
    SlotRequest,
    StaffDecisionRequest,
    TraceEventOut,
    VisitDecisionRequest,
)
from app.audit import write_audit
from app.auth.dependencies import StaffUser
from app.db import get_session
from app.errors import RecordNotFound, ValidationFailed
from app.models import (
    Appointment,
    AppointmentSlot,
    AppointmentStatus,
    AuditEvent,
    Department,
    Doctor,
    EscalationStatus,
    SlotStatus,
    TraceEvent,
    WorkflowRun,
    WorkflowStatus,
)
from app.tools import (
    describe_appointment,
    get_patient_context,
    list_departments,
    list_flagged_documents,
    list_open_escalations,
)
from app.trace import TraceWriter
from app.workflow.staff import (
    apply_staff_decision,
    apply_visit_decision,
    resolve_document,
    resolve_escalation,
)

router = APIRouter(prefix="/staff", tags=["staff"])

DbSession = Annotated[Session, Depends(get_session)]


# --- queues -------------------------------------------------------------


@router.get("/queue", response_model=list[RunOut])
def request_queue(
    staff: StaffUser,
    session: DbSession,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[RunOut]:
    """Every patient request and the state it is in, newest first.

    A misspelled ``status`` is a 422 rather than an empty list. An empty queue
    reads as "nothing to review", which is the one wrong answer a monitoring
    view can give while looking like it worked.
    """
    query = session.query(WorkflowRun)
    if status is not None:
        try:
            query = query.filter(WorkflowRun.status == WorkflowStatus(status))
        except ValueError as exc:
            raise ValidationFailed(
                f"{status!r} is not a workflow status. Known: "
                f"{', '.join(s.value for s in WorkflowStatus)}."
            ) from exc

    runs = query.order_by(WorkflowRun.id.desc()).limit(limit).all()
    return [run_out(run) for run in runs]


@router.get("/patients/{patient_id}")
def patient_view(patient_id: int, staff: StaffUser, session: DbSession) -> dict[str, Any]:
    """One patient's full administrative context, for review."""
    return get_patient_context(session, staff, patient_id=patient_id)


@router.get("/escalations")
def escalation_queue(staff: StaffUser, session: DbSession) -> list[dict[str, Any]]:
    """Everything still awaiting a human, oldest first."""
    return list_open_escalations(session)


@router.get("/documents/flagged")
def flagged_documents(staff: StaffUser, session: DbSession) -> list[dict[str, Any]]:
    """Documents whose content did not match the type they were filed under."""
    return list_flagged_documents(session)


# --- typed decisions ----------------------------------------------------


@router.post("/runs/{run_id}/decision")
def decide(
    run_id: int,
    payload: StaffDecisionRequest,
    staff: StaffUser,
    session: DbSession,
) -> dict[str, Any]:
    """Approve, reject, or redirect a paused run. No model is involved."""
    run = session.get(WorkflowRun, run_id)
    if run is None:
        raise RecordNotFound("WorkflowRun", run_id)

    # The decision is a turn, and a turn belongs to the run's conversation —
    # otherwise the staff action lands in the timeline under a session id
    # nobody can look up.
    writer = TraceWriter(session, session_id=run.session_id)

    try:
        decision = apply_staff_decision(
            session,
            staff=staff,
            run_id=run_id,
            action=payload.action,
            writer=writer,
            department_name=payload.department_name,
            note=payload.note,
        )
    except ValidationFailed as refused:
        # Commit *before* the 422: the refusal's own audit and trace rows are
        # the record of a decision a human made, and they are only in memory
        # until this line runs.
        session.commit()
        raise HTTPException(status_code=422, detail=str(refused)) from refused

    session.commit()
    return {
        "run_id": decision.run_id,
        "action": decision.action,
        "status": decision.status,
        "department_name": decision.department_name,
        "notification_id": decision.notification_id,
        "escalation_id": decision.escalation_id,
    }


@router.get("/visits")
def swept_visits(
    staff: StaffUser,
    session: DbSession,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Visits the sweep has closed, newest first, for staff to correct.

    Both statuses on purpose: a screen that listed only ``completed`` would let
    staff mark a no-show and then lose the row they had just acted on, with no
    way back. ``missed`` is a correction, not a verdict, and the flip has to be
    reversible from the same list it was made in.
    """
    rows = (
        session.query(Appointment)
        .filter(
            Appointment.status.in_(
                [AppointmentStatus.COMPLETED, AppointmentStatus.MISSED]
            )
        )
        .order_by(Appointment.id.desc())
        .limit(limit)
        .all()
    )
    return [describe_appointment(session, row) for row in rows]


@router.post("/appointments/{appointment_id}/visit")
def correct_visit(
    appointment_id: int,
    payload: VisitDecisionRequest,
    staff: StaffUser,
    session: DbSession,
) -> dict[str, Any]:
    """Mark a swept appointment completed or missed. No model is involved.

    The poll job's sweep can only see that an end time has passed; whether the
    patient attended is not something a clock knows. This is the correction,
    and it is the only thing that opens a missed-visit follow-up.

    No ``TraceWriter`` here, deliberately: a visit correction has no workflow
    run and therefore no turn, so it belongs in the audit ledger alone — the
    same split the scheduler follows.
    """
    decision = apply_visit_decision(
        session, staff=staff, appointment_id=appointment_id, action=payload.action
    )
    session.commit()
    return decision


@router.post("/escalations/{escalation_id}/resolve")
def close_escalation(
    escalation_id: int,
    payload: EscalationResolution,
    staff: StaffUser,
    session: DbSession,
) -> dict[str, Any]:
    """Acknowledge or resolve an escalation.

    The vocabulary check lives in ``workflow.staff``: a safety escalation
    cannot be set to ``approved``, whatever this endpoint is asked for.
    """
    result = resolve_escalation(
        session,
        staff=staff,
        escalation_id=escalation_id,
        status=EscalationStatus(payload.status),
        note=payload.note,
    )
    session.commit()
    return result


@router.post("/documents/{document_id}/resolve")
def close_document(
    document_id: int,
    payload: DocumentResolution,
    staff: StaffUser,
    session: DbSession,
) -> dict[str, Any]:
    """Accept, reclassify, or reject a flagged document.

    The re-diff happens inside ``resolve_document``, in this transaction — that
    is the derivation invariant, and committing here is what makes it one.
    """
    result = resolve_document(
        session,
        staff=staff,
        document_id=document_id,
        action=payload.action,
        corrected_type=payload.corrected_type,
        note=payload.note,
    )
    session.commit()
    return result


# --- oversight ----------------------------------------------------------


@router.get("/runs/{run_id}/trace", response_model=list[TraceEventOut])
def read_trace(run_id: int, staff: StaffUser, session: DbSession) -> list[TraceEventOut]:
    """The full timeline for one run, in the order it was written.

    **Whole turns, not just the events bound to the run.** A turn opens before
    its run exists — the inbound event, the safety screen and the classification
    all precede the moment intent is accepted and ``bind_run`` is called — so
    those rows carry a null ``workflow_run_id`` on purpose. Filtering on the run
    id alone returns a timeline that begins in the middle, with no inbound event
    and no guard verdicts: the part a reviewer most needs is exactly the part it
    drops. So the run's turns are found first, and then every event of those
    turns is returned.

    Ordered by id rather than by ``seq``: ``seq`` restarts at 1 for every turn,
    so sorting by it would interleave a three-turn conversation into nonsense.
    """
    if session.get(WorkflowRun, run_id) is None:
        raise RecordNotFound("WorkflowRun", run_id)

    turn_ids = [
        turn_id
        for (turn_id,) in session.query(TraceEvent.turn_id)
        .filter(TraceEvent.workflow_run_id == run_id)
        .distinct()
    ]
    events = (
        session.query(TraceEvent)
        .filter(TraceEvent.turn_id.in_(turn_ids))
        .order_by(TraceEvent.id)
        .all()
        if turn_ids
        else []
    )
    return [
        TraceEventOut(
            seq=event.seq,
            turn_id=event.turn_id,
            event_type=event.event_type.value,
            author=event.author.value if event.author else None,
            agent_name=event.agent_name,
            correlation_id=event.correlation_id,
            payload=event.payload or {},
            created_at=event.created_at.isoformat(),
        )
        for event in events
    ]


@router.get("/audit", response_model=list[AuditEventOut])
def read_audit(
    staff: StaffUser,
    session: DbSession,
    action: str | None = None,
    entity_type: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[AuditEventOut]:
    """Who did what, to which entity, when. Newest first."""
    query = session.query(AuditEvent)
    if action is not None:
        query = query.filter(AuditEvent.action == action)
    if entity_type is not None:
        query = query.filter(AuditEvent.entity_type == entity_type)

    events = query.order_by(AuditEvent.id.desc()).limit(limit).all()
    return [
        AuditEventOut(
            id=event.id,
            actor_id=event.actor_id,
            actor_kind=event.actor_kind,
            action=event.action,
            entity_type=event.entity_type,
            entity_id=event.entity_id,
            metadata=event.event_metadata or {},
            created_at=event.created_at.isoformat(),
        )
        for event in events
    ]


# --- capacity -----------------------------------------------------------


@router.get("/departments")
def departments(staff: StaffUser, session: DbSession) -> list[dict[str, Any]]:
    """Every department, open or closed.

    ``active_only=False`` deliberately: this is the listing an operator manages
    capacity from, and a closed department missing from it could never be
    re-opened.
    """
    return list_departments(session, active_only=False)


@router.patch("/departments/{department_id}")
def set_department_active(
    department_id: int, payload: ActiveToggle, staff: StaffUser, session: DbSession
) -> dict[str, Any]:
    """Open or close a department. Creation comes from the seed script."""
    department = session.get(Department, department_id)
    if department is None:
        raise RecordNotFound("Department", department_id)

    department.active = payload.active
    write_audit(
        session,
        action="department_active_set",
        entity_type="Department",
        entity_id=department.id,
        actor=staff,
        metadata={"active": payload.active},
    )
    session.commit()
    return {"department_id": department.id, "name": department.name, "active": department.active}


@router.patch("/doctors/{doctor_id}")
def set_doctor_active(
    doctor_id: int, payload: ActiveToggle, staff: StaffUser, session: DbSession
) -> dict[str, Any]:
    """Take a doctor off the roster. Their existing slots stop being offered."""
    doctor = session.get(Doctor, doctor_id)
    if doctor is None:
        raise RecordNotFound("Doctor", doctor_id)

    doctor.active = payload.active
    write_audit(
        session,
        action="doctor_active_set",
        entity_type="Doctor",
        entity_id=doctor.id,
        actor=staff,
        metadata={"active": payload.active},
    )
    session.commit()
    return {"doctor_id": doctor.id, "name": doctor.name, "active": doctor.active}


@router.post("/doctors/{doctor_id}/slots", status_code=201)
def add_slots(
    doctor_id: int, payload: SlotRequest, staff: StaffUser, session: DbSession
) -> dict[str, Any]:
    """Add capacity for one doctor.

    Every timestamp is parsed **before** any row is created, so a batch with
    one bad entry adds nothing rather than half of itself. A start time the
    doctor already has is skipped and counted, not duplicated — running the
    same request twice must not double-book the calendar.
    """
    doctor = session.get(Doctor, doctor_id)
    if doctor is None:
        raise RecordNotFound("Doctor", doctor_id)

    starts: list[datetime] = []
    for raw in payload.start_times:
        try:
            starts.append(datetime.fromisoformat(raw))
        except ValueError as exc:
            raise ValidationFailed(
                f"{raw!r} is not an ISO 8601 date-time. Nothing was created."
            ) from exc

    duration = timedelta(minutes=payload.duration_minutes)
    existing = {
        start
        for (start,) in session.query(AppointmentSlot.start_time).filter(
            AppointmentSlot.doctor_id == doctor_id,
            AppointmentSlot.start_time.in_(starts),
        )
    }

    created = 0
    for start in starts:
        if start in existing:
            continue
        session.add(
            AppointmentSlot(
                doctor_id=doctor_id,
                start_time=start,
                end_time=start + duration,
                status=SlotStatus.AVAILABLE,
            )
        )
        existing.add(start)
        created += 1

    write_audit(
        session,
        action="slots_added",
        entity_type="Doctor",
        entity_id=doctor_id,
        actor=staff,
        metadata={"created": created, "requested": len(starts)},
    )
    session.commit()
    return {
        "doctor_id": doctor_id,
        "created": created,
        "skipped": len(starts) - created,
    }
