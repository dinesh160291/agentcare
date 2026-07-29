"""Turning a safety verdict into state a human owns.

Four rules live here, and each one exists because the obvious implementation
gets it wrong.

**The run is born escalated when there is nothing else to key to.** An
emergency on a session's opening message has no workflow run — and every
``Escalation`` points at one, with a non-nullable foreign key, so that the
staff queue and the trace timeline have no orphan special-case. Rather than
making the column nullable for one path, the run is created directly in
``escalated``. This is the only door into that initial state, and it opens for
safety alone.

**Repeats attach; they never multiply.** A frightened patient types "chest
pain" five times. The naive path creates five runs, because the first one is
already terminal and so no longer the *active* run — five queue items, five
things for a human to reconcile, for one person in trouble. So before creating
anything, this module looks for the escalated run this session already
produced, and attaches to it. Five triggers become one queue item with an
occurrence count of five, each trigger separately audited.

**A request waiting on a human is not the one to escalate.** Attaching to the
active run is right while the system holds it, and wrong the moment staff do: a
``pending_review`` run is a queue item, ``escalated`` is terminal, and folding a
new scare into it destroys a request a person was about to decide on. That one
state gets a run of its own.

**A withdrawal cannot close it.** That is enforced by the state machine —
``escalated`` is terminal for automation — but the reason belongs here: the
patient saying "actually forget it" after a chest-pain message is exactly the
moment the system must *not* be helpful.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.audit import write_audit
from app.models import (
    Escalation,
    EscalationKind,
    TraceAuthor,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.models.enums import PlanStep
from app.safety.screen import SafetyCategory, SafetyVerdict
from app.tools.tasks import OPEN_ESCALATION_STATUSES, create_escalation
from app.trace import TraceWriter
from app.workflow.mapping import names_appointment_verbs
from app.workflow.state_machine import create_run, transition

#: Code-authored, identical in mock and live. It directs the patient to urgent
#: care and says nothing about what may be wrong with them: naming a cause
#: would be the clinical claim this whole layer exists to prevent.
EMERGENCY_REPLY = (
    "This needs urgent help rather than an appointment booking. Please call your "
    "local emergency number now, or go to your nearest emergency department. "
    "I've flagged this to our staff straight away."
)

CLINICAL_REPLY = (
    "I'm sorry — I can't help with that one. I handle hospital administration "
    "only: appointments, documents, reminders, and follow-ups. I've passed this "
    "to our staff so that someone who can answer it will get back to you."
)


#: Appended when an escalated message *also* asked for something this system
#: does. Deterministic, and one sentence: the refusal is the important part and
#: stays exactly as it is.
BOOKING_HINT = (
    "If you'd like to book an appointment, just tell me the department or the "
    "reason for the visit."
)


def reply_for(verdict: SafetyVerdict, message: str = "") -> str:
    """The template that answers a fired verdict.

    A clinical refusal gains a signpost when the same message named a booking.
    Live, "Need help to book an appointment for vision test as lately I'm
    feeling little bit blurry on my right eye" escalated — correctly; the
    screen erring conservative is the trade this project chose — and the reply
    said only that it could not help. What recovered the booking was the
    patient guessing a wording with no symptom in it. That guess is now a
    signpost, and the screen is untouched.

    **Never on an emergency.** That reply says this needs urgent help *rather
    than* an appointment, and appending an invitation to book one would argue
    with it in front of a frightened patient. The hint exists for the case
    where administration was genuinely part of what was asked; an emergency is
    the case where it is beside the point.

    Two newlines, because the chat renders CommonMark and one would weld the
    signpost onto the refusal as a single paragraph.
    """
    if verdict.category is SafetyCategory.EMERGENCY:
        return EMERGENCY_REPLY
    if names_appointment_verbs(message) == {PlanStep.BOOK}:
        return f"{CLINICAL_REPLY}\n\n{BOOKING_HINT}"
    return CLINICAL_REPLY


def _open_escalated_run(
    session: Session, *, patient_id: int, session_id: str | None
) -> WorkflowRun | None:
    """This session's already-escalated run, if it still has an open escalation.

    Scoped to the session because a new conversation is a new request. Scoped
    to *open* escalations because one a human has resolved should not silently
    collect a sixth trigger under a closed record.
    """
    return (
        session.query(WorkflowRun)
        .join(Escalation, Escalation.workflow_run_id == WorkflowRun.id)
        .filter(
            WorkflowRun.patient_id == patient_id,
            WorkflowRun.session_id == session_id,
            WorkflowRun.status == WorkflowStatus.ESCALATED,
            Escalation.status.in_(OPEN_ESCALATION_STATUSES),
        )
        .order_by(WorkflowRun.id.desc())
        .first()
    )


def escalate(
    session: Session,
    *,
    verdict: SafetyVerdict,
    user: User,
    patient_id: int,
    message: str,
    writer: TraceWriter,
    session_id: str,
    run: WorkflowRun | None,
) -> tuple[WorkflowRun, str, TraceAuthor]:
    """Put a run in front of a human and keep it there.

    Returns the run the escalation is keyed to, the reply, and its author.
    Does not commit — the transition, the escalation row, the audit rows, and
    the trace rows all belong to the turn's one transaction.
    """
    category = verdict.category or SafetyCategory.CLINICAL_ADVICE
    reason = (
        f"Safety screen ({verdict.source}) matched {verdict.rule!r}: "
        f"{category.value.replace('_', ' ')}."
    )

    if run is not None and run.status is WorkflowStatus.PENDING_REVIEW:
        # A run in front of staff is a queue item as well as a conversation, and
        # escalating it consumes both. Live: "my kid has ear pain" was routed
        # ambiguously and queued for a human; two messages later an unrelated
        # scare arrived, the active run was the queued one, and it went
        # `pending_review -> escalated` — which is terminal, so the ear-pain
        # request died without a staff decision ever being made on it and
        # without the patient being told.
        #
        # The scare is not that request, so it does not get that request's row.
        # Handing it a run of its own leaves the queue item where a human left
        # it, and the escalated run says what it is about. Only `pending_review`
        # is spared: at `in_progress` or `pending_confirmation` the system holds
        # the run, nothing is waiting on a person, and folding the scare into
        # the conversation it interrupted is the right reading.
        writer.guard_verdict(
            "escalation_target",
            passed=False,
            detail={
                "run_id": run.id,
                "status": run.status.value,
                "problem": "the active run is waiting for staff; it is not this one",
            },
        )
        run = None

    if run is None:
        run = _open_escalated_run(
            session, patient_id=patient_id, session_id=session_id
        )

    if run is None:
        # Nothing to key to. The one case where a run is born terminal.
        run = create_run(
            session,
            patient_id=patient_id,
            status=WorkflowStatus.ESCALATED,
            trigger=f"safety_screen_{category.value}",
            writer=writer,
            request_text=message,
            plan=[],
            session_id=session_id,
            actor=user,
        )
    elif run.status is not WorkflowStatus.ESCALATED:
        writer.bind_run(run.id)
        transition(
            session,
            run,
            to=WorkflowStatus.ESCALATED,
            trigger=f"safety_screen_{category.value}",
            writer=writer,
            actor=user,
            detail=verdict.as_trace_detail(),
        )
    else:
        # Already escalated in this session: the repeat attaches below.
        writer.bind_run(run.id)

    create_escalation(
        session,
        workflow_run_id=run.id,
        kind=EscalationKind.SAFETY,
        reason=reason,
        message=message,
        actor=user,
    )
    write_audit(
        session,
        action="safety_screen_fired",
        entity_type="workflow_run",
        entity_id=run.id,
        actor=user,
        metadata=verdict.as_trace_detail(),
    )

    return run, reply_for(verdict, message), TraceAuthor.GUARD


__all__ = ["BOOKING_HINT", "CLINICAL_REPLY", "EMERGENCY_REPLY", "escalate", "reply_for"]
