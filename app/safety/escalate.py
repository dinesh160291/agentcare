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

**A live request is not the one to escalate.** ``escalated`` is terminal, so
folding a scare into whatever run happens to be open destroys that run — and it
is a *different subject*, which is the whole reason it fired. This started as a
carve-out for ``pending_review`` alone, on the reasoning that at ``in_progress``
nothing is waiting on a person and the scare is part of the same conversation.
Live, that reasoning cost a booking: "I get bad migraines every morning",
arriving during a General Medicine checkup request, took the run from
``in_progress`` to ``escalated`` and the patient was never told the request they
had spent four messages on was gone. A remark about how someone feels is not a
decision to abandon what they asked for, so every live run is spared now and the
scare gets a run of its own. The reply says what is still open, because a run
nobody mentions is a run the patient has to guess is there.

**A withdrawal cannot close it.** That is enforced by the state machine —
``escalated`` is terminal for automation — but the reason belongs here: the
patient saying "actually forget it" after a chest-pain message is exactly the
moment the system must *not* be helpful.
"""

from __future__ import annotations

import re

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
from app.safety.screen import SafetyCategory, SafetyVerdict
from app.tools.tasks import OPEN_ESCALATION_STATUSES, create_escalation
from app.trace import TraceWriter
from app.workflow.mapping import mentions_domain_subject
from app.workflow.state_machine import create_run

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

#: "I need to see someone", "can I get checked", "I'd like to see a doctor".
#: A patient asking to be *seen* is asking for an appointment without using any
#: of the words this system files under booking — and all three of the live
#: escalations this round came in that shape: "stomach upset with acidity, need
#: to see someone", "ringing in my ears", "I get bad migraines every morning".
#: The first names no verb at all by :func:`names_appointment_verbs`' rule,
#: which wants a verb *and* an appointment noun.
_ASKS_TO_BE_SEEN = re.compile(
    r"\b(?:see\s+(?:someone|somebody|a\s+doctor|a\s+specialist|a\s+consultant)"
    r"|get\s+(?:seen|checked)|be\s+seen|checked\s+out)\b",
    re.IGNORECASE,
)


def wants_administration(message: str) -> bool:
    """Does this message also ask for something this system can actually do?

    The gate on :data:`BOOKING_HINT`, widened in round 11b. It used to demand
    that the message name exactly ``{BOOK}``, which is a verb plus an
    appointment noun — and the three refusals that produced this item named
    neither, so all three gave the patient no path back at all.

    Veto-free and one-directional, like :func:`mentions_domain_subject`: the
    only thing it can do is add a sentence to a refusal that has already been
    decided. It cannot unblock a screen, cannot change a class, and cannot
    reach an emergency reply. That is what lets it read generously, and it is
    the opposite trade from the screen it sits under.
    """
    return mentions_domain_subject(message) or bool(_ASKS_TO_BE_SEEN.search(message))


def still_open_note(run: WorkflowRun | None) -> str:
    """One sentence naming the request a scare interrupted, or "".

    The other half of sparing the active run. A run left standing that nobody
    mentions is a run the patient has to guess is still there — and the reply
    they are reading says their message went to staff, which reads like the end
    of the conversation.
    """
    if run is None:
        return ""
    department = (run.state or {}).get("department_name")
    what = f"{department} request" if department else "earlier request"
    return f"Your {what} is still open — say the word and we'll carry on with it."


def reply_for(
    verdict: SafetyVerdict, message: str = "", *, still_open: str = ""
) -> str:
    """The template that answers a fired verdict.

    A clinical refusal gains a signpost when the same message also asked for
    administration. Live, "Need help to book an appointment for vision test as
    lately I'm feeling little bit blurry on my right eye" escalated —
    correctly; the screen erring conservative is the trade this project chose —
    and the reply said only that it could not help. What recovered the booking
    was the patient guessing a wording with no symptom in it. That guess is now
    a signpost, and the screen is untouched.

    **Never on an emergency.** That reply says this needs urgent help *rather
    than* an appointment, and appending an invitation to book one — or a note
    about a booking that is still open — would argue with it in front of a
    frightened patient. Both tails exist for the case where administration was
    genuinely part of what was asked; an emergency is the case where it is
    beside the point. So the emergency template goes out byte-identical, always.

    Two newlines between parts, because the chat renders CommonMark and one
    would weld a signpost onto the refusal as a single paragraph.
    """
    if verdict.category is SafetyCategory.EMERGENCY:
        return EMERGENCY_REPLY
    parts = [CLINICAL_REPLY]
    if wants_administration(message):
        parts.append(BOOKING_HINT)
    if still_open:
        parts.append(still_open)
    return "\n\n".join(parts)


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

    spared: WorkflowRun | None = None
    if run is not None:
        # A live run is somebody's work in progress, and `escalated` is
        # terminal — so folding a scare into it ends that work. Two live
        # failures, one round apart, are the same failure at two states. At
        # `pending_review`: "my kid has ear pain" was queued for a human, an
        # unrelated scare arrived two messages later, and the request died
        # without any staff decision ever being made on it. At `in_progress`:
        # "I get bad migraines every morning", said during a General Medicine
        # checkup request, consumed it — and nothing in the reply said so.
        #
        # The scare is a different subject; that is why it fired. So it gets a
        # run of its own, the queue item or the conversation stays where the
        # patient left it, and `still_open_note` says which. Repeats are
        # unaffected: the dedup below looks for this session's *escalated* run,
        # which is never the active one.
        writer.guard_verdict(
            "escalation_target",
            passed=False,
            detail={
                "run_id": run.id,
                "status": run.status.value,
                "problem": "the active run is live work; the scare is not that request",
            },
        )
        spared, run = run, None

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
    else:
        # This session's escalated run, and the repeat attaches to it below.
        # There is no third case: sparing a live run above means the only run
        # that survives to here is one ``_open_escalated_run`` found, and that
        # query filters on ``escalated``. The transition-an-existing-run branch
        # that used to sit here had no reachable input left, and a line that
        # cannot fail vouches for nothing.
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

    reply = reply_for(verdict, message, still_open=still_open_note(spared))
    return run, reply, TraceAuthor.GUARD


__all__ = [
    "BOOKING_HINT",
    "CLINICAL_REPLY",
    "EMERGENCY_REPLY",
    "escalate",
    "reply_for",
    "still_open_note",
    "wants_administration",
]
