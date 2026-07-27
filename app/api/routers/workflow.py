"""The two front doors, and the run they act on.

**Both doors are separate endpoints because they are separate kinds of input.**
A chat message is language and gets the whole classification apparatus; a
Confirm click is a typed action out of a closed set and gets none of it. Posting
the word "confirm" to ``/workflow/messages`` is a message that happens to say
confirm — it is read by the exact-token reader like any other text. Pressing the
button posts to ``/workflow/actions``, where there is nothing to read.

Neither handler owns a transaction. ``run_workflow`` and ``apply_patient_action``
open their own session because the turn *is* the transaction — its state change,
its audit rows and its trace rows commit together or not at all — so a router
that passed its own session in would be splitting a bracket that exists
precisely to be unsplittable.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.schemas import ActionRequest, MessageRequest, RunOut, TurnOut
from app.auth.dependencies import CurrentUser, PatientUser
from app.auth.ownership import get_owned_or_404, patient_profile_for
from app.db import get_session
from app.models import AppointmentSlot, WorkflowRun
from app.orchestrator import apply_patient_action, run_workflow
from app.workflow.replies import when_words

router = APIRouter(prefix="/workflow", tags=["workflow"])

DbSession = Annotated[Session, Depends(get_session)]


def run_out(run: WorkflowRun, session: Session | None = None) -> RunOut:
    """Serialize a run. Shared with the staff queue, which shows the same row.

    The proposal's doctor, day and time are resolved here rather than on the
    screen: what the patient is agreeing to is a doctor at a time, and a client
    that derived those from a slot id could show something the row does not
    say. Only looked up when there is a proposal and a session to look it up
    with — the staff queue lists runs in bulk and needs none of it.
    """
    state = run.state or {}
    doctor = day = time_of_day = None
    if session is not None and run.proposed_slot_id:
        slot = session.get(AppointmentSlot, run.proposed_slot_id)
        if slot is not None:
            day, time_of_day = when_words(slot.start_time)
            doctor = slot.doctor.name if slot.doctor else None
    return RunOut(
        proposed_doctor_name=doctor,
        proposed_day=day,
        proposed_time=time_of_day,
        run_id=run.id,
        patient_id=run.patient_id,
        status=run.status.value,
        current_step=run.current_step,
        plan=list(run.plan or []),
        completed_steps=list(run.completed_steps or []),
        request_text=run.request_text or "",
        department_name=state.get("department_name"),
        proposed_action=run.proposed_action.value if run.proposed_action else None,
        proposed_slot_id=run.proposed_slot_id,
        session_id=run.session_id,
        created_at=run.created_at.isoformat(),
        updated_at=run.updated_at.isoformat(),
    )




@router.post("/messages", response_model=TurnOut)
async def submit_message(payload: MessageRequest, user: PatientUser) -> TurnOut:
    """One conversational turn, start to finish."""
    message = payload.message.strip()
    if not message:
        # Refused here rather than inside the envelope: an all-whitespace
        # message is not a turn, and opening one would leave an inbound event
        # with nothing to be inbound about.
        raise HTTPException(status_code=422, detail="A message cannot be empty.")
    return TurnOut.of(await run_workflow(user, message, payload.session_id))


@router.post("/actions", response_model=TurnOut)
async def submit_action(payload: ActionRequest, user: PatientUser) -> TurnOut:
    """✅ Confirm / ❌ Decline. No model is consulted about what it meant."""
    return TurnOut.of(
        await apply_patient_action(user, payload.action, payload.session_id)
    )


@router.get("/runs", response_model=list[RunOut])
def list_runs(user: PatientUser, session: DbSession) -> list[RunOut]:
    """The caller's own runs, newest first."""
    profile = patient_profile_for(session, user)
    runs = (
        session.query(WorkflowRun)
        .filter(WorkflowRun.patient_id == profile.id)
        .order_by(WorkflowRun.id.desc())
        .all()
    )
    return [run_out(run) for run in runs]


@router.get("/runs/{run_id}", response_model=RunOut)
def read_run(run_id: int, user: CurrentUser, session: DbSession) -> RunOut:
    """One run — the caller's own, or any run if the caller is staff.

    ``get_owned_or_404`` audits the denied attempt before it raises, and the
    404 is byte-identical to the one for an id that was never issued.
    """
    return run_out(get_owned_or_404(session, WorkflowRun, run_id, user), session)
