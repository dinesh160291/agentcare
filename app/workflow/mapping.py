"""Message→run mapping — one active run per patient, and six ways to relate.

A patient with a live run keeps talking. Every message they send has to be
placed against that run before anything happens, and the placement is where the
system is most exposed, because each class costs something different when it is
wrong:

* a wrongly superseded **review** costs the patient a re-ask;
* a **zombie resume** — staff acting on a request the patient abandoned — costs
  an incident;
* a wrongly superseded **cooperation** ("also, here's my old ECG", read as a
  brand-new request) costs the booking the patient actually wanted.

The last one is why classification precedes consequence rather than supersede
being the default for everything. The model proposes the class; code validates
it and applies the consequence.

Code re-checks the one rule the model is most likely to get wrong: **a message
carrying the same intent type as the active run can never be complementary** —
it is conflicting by definition. Complementary is reserved for compatible
intent types that serve the active goal.

The scope gate runs inside the mapping too, not only at run creation. An
off-topic message that fell through to continuation would be appended to the
run's stored request text, contaminating what routing and slot matching read
later — so ``off_topic`` is a self-stay that touches nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from sqlalchemy.orm import Session

from app.errors import ClassRejected, ValidationFailed
from app.models import (
    APPOINTMENT_VERBS,
    CancellationReason,
    MessageClass,
    PlanStep,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.tools.departments import resolve_department
from app.tools.tasks import close_escalations_for_run
from app.trace import TraceWriter
from app.workflow.plan import CANONICAL_ORDER, append_step
from app.workflow.state_machine import transition

#: Evaluation order within a turn. Withdrawal outranks everything — a patient
#: abandoning a request must never be read as continuing it — and off-topic
#: sits second so it cannot fall through and contaminate the request text.
CLASSIFICATION_ORDER: tuple[MessageClass, ...] = (
    MessageClass.WITHDRAWAL,
    MessageClass.OFF_TOPIC,
    MessageClass.SIDE_QUESTION,
    MessageClass.COMPLEMENTARY,
    MessageClass.CONFLICTING,
    MessageClass.CONTINUATION,
)

#: Which cancellation reason a withdrawal carries, by the state it interrupts.
#: The staff queue's "withdrawn while pending" section is a query over this.
_WITHDRAWAL_REASON = {
    WorkflowStatus.PENDING_REVIEW: CancellationReason.WITHDRAWN_DURING_REVIEW,
    WorkflowStatus.IN_PROGRESS: CancellationReason.WITHDRAWN,
    WorkflowStatus.PENDING_CONFIRMATION: CancellationReason.WITHDRAWN,
}


class Consequence(str, Enum):
    """What a class is permitted to do. One entry per row of the PRD's table."""

    FEED_RUN = "feed_run"                # continuation
    APPEND_STEP = "append_step"          # complementary
    SUPERSEDE = "supersede"              # conflicting
    ANSWER_AND_STAY = "answer_and_stay"  # side question
    SCOPE_REPLY = "scope_reply"          # off-topic
    WITHDRAW = "withdraw"                # withdrawal
    #: Not a class of its own — what a *conflicting* message becomes when the
    #: run it would replace is waiting for staff and the message shows no
    #: difference. See :func:`_supersede_needs_difference`.
    STATUS_REPLY = "status_reply"


CONSEQUENCE_FOR: dict[MessageClass, Consequence] = {
    MessageClass.CONTINUATION: Consequence.FEED_RUN,
    MessageClass.COMPLEMENTARY: Consequence.APPEND_STEP,
    MessageClass.CONFLICTING: Consequence.SUPERSEDE,
    MessageClass.SIDE_QUESTION: Consequence.ANSWER_AND_STAY,
    MessageClass.OFF_TOPIC: Consequence.SCOPE_REPLY,
    MessageClass.WITHDRAWAL: Consequence.WITHDRAW,
}


@dataclass(frozen=True)
class ClassVerdict:
    """The class that will actually be applied, and what was proposed."""

    message_class: MessageClass
    proposed: MessageClass | None
    adjusted: bool
    reason: str


@dataclass(frozen=True)
class MappingOutcome:
    """What happened to the active run, and what the caller still owes.

    ``spawns_new_run`` is the orchestrator's cue: the mapping owns the fate of
    the *existing* run, and creating its replacement needs the plan for the new
    message, which the mapping never sees.
    """

    consequence: Consequence
    message_class: MessageClass
    spawns_new_run: bool
    run_id: int


#: Widest first. Every appointment verb outranks the supporting steps, because
#: a plan closes over the ones it needs: ``reschedule`` drags ``follow_up`` in
#: with it, and a ranking that missed the verb would call the whole request a
#: follow-up.
#:
#: Leaving ``reschedule`` and ``cancel`` out of this tuple was not a gap with a
#: small consequence. "Lets reschedule my ENT appointment", sent while a
#: *booking* sat in review, ranked as ``follow_up`` against the booking's
#: ``book`` — different, so no re-check fired, so it was taken as a
#: continuation of the booking. Its text was appended to that run, routing
#: re-ran on the two requests welded together, and the Appointment agent
#: proposed a reschedule under the ``book`` step: a ``pending_review`` run
#: carrying ``proposed_action = reschedule``, and a reply answering two
#: requests as though they were one.
_INTENT_RANK: tuple[PlanStep, ...] = (
    PlanStep.BOOK,
    PlanStep.RESCHEDULE,
    PlanStep.CANCEL,
    PlanStep.DOCUMENTS,
    PlanStep.ROUTE,
    PlanStep.FOLLOW_UP,
)


def primary_intent(plan: Sequence[str | PlanStep]) -> PlanStep | None:
    """The widest step in a plan — what the run is fundamentally *for*.

    Two messages share an intent when their primary steps match, which is the
    test that separates a cooperation from a rephrase.
    """
    present = set()
    for value in plan or []:
        present.add(value if isinstance(value, PlanStep) else PlanStep(value))
    for step in _INTENT_RANK:
        if step in present:
            return step
    return None


def validate_class(
    proposed: object,
    *,
    run: WorkflowRun,
    incoming_steps: Sequence[PlanStep] | None,
    writer: TraceWriter,
) -> ClassVerdict:
    """Check the model's proposed class, adjusting it where code knows better.

    :raises ClassRejected: the proposal is not one of the six classes.
    """
    if not isinstance(proposed, (str, MessageClass)):
        writer.validation(
            "message_class",
            accepted=False,
            detail={"proposed": repr(proposed), "problem": "not a class name"},
        )
        raise ClassRejected(
            f"A message class must be one of {[c.value for c in CLASSIFICATION_ORDER]}, "
            f"got {type(proposed).__name__}."
        )

    try:
        message_class = MessageClass(proposed)
    except ValueError:
        writer.validation(
            "message_class",
            accepted=False,
            detail={"proposed": str(proposed), "problem": "unknown class"},
        )
        raise ClassRejected(
            f"Unknown message class {proposed!r}. The set is closed: "
            f"{[c.value for c in CLASSIFICATION_ORDER]}."
        ) from None

    # A *different* appointment verb is a different request, whatever the model
    # called it. Booking, moving and cancelling are three things one run cannot
    # be doing at once — `validate_plan` refuses a plan naming two of them for
    # exactly this reason — so a message asking for one while the run pursues
    # another is conflicting, and neither a continuation of it nor a
    # cooperation with it.
    #
    # This is checked ahead of the complementary rule below because the live
    # failure arrived as a **continuation**: "lets reschedule my ENT
    # appointment", sent while a booking waited for staff, was fed into the
    # booking's own request text. Guarding only `complementary` left the class
    # the model actually chose unguarded, and the run that a human was about to
    # review grew a reschedule proposal.
    #
    # Deliberately narrow. Both intents must be appointment verbs: "here's my
    # old ECG" during a booking is `documents` against `book` and stays exactly
    # as cooperative as it was.
    if incoming_steps and message_class in (
        MessageClass.CONTINUATION,
        MessageClass.COMPLEMENTARY,
    ):
        active = primary_intent(run.plan or [])
        incoming = primary_intent(list(incoming_steps))
        if (
            active in APPOINTMENT_VERBS
            and incoming in APPOINTMENT_VERBS
            and active is not incoming
        ):
            writer.validation(
                "message_class",
                accepted=False,
                detail={
                    "proposed": message_class.value,
                    "applied": MessageClass.CONFLICTING.value,
                    "problem": "a different appointment verb from the active run",
                    "intent": incoming.value,
                    "active_intent": active.value,
                },
            )
            return ClassVerdict(
                message_class=MessageClass.CONFLICTING,
                proposed=message_class,
                adjusted=True,
                reason=(
                    f"asks to {incoming.value} while the run is a "
                    f"{active.value} request"
                ),
            )

    # The one rule code re-checks. Complementary is reserved for compatible
    # intent types; a message carrying the run's own intent is a rephrase, and
    # a rephrase is conflicting by definition.
    if message_class is MessageClass.COMPLEMENTARY and incoming_steps:
        active = primary_intent(run.plan or [])
        incoming = primary_intent(list(incoming_steps))
        if active is not None and active is incoming:
            writer.validation(
                "message_class",
                accepted=False,
                detail={
                    "proposed": message_class.value,
                    "applied": MessageClass.CONFLICTING.value,
                    "problem": "same intent as the active run",
                    "intent": active.value,
                },
            )
            return ClassVerdict(
                message_class=MessageClass.CONFLICTING,
                proposed=message_class,
                adjusted=True,
                reason=f"same intent as the active run ({active.value})",
            )

    writer.validation(
        "message_class", accepted=True, detail={"class": message_class.value}
    )
    return ClassVerdict(
        message_class=message_class, proposed=message_class, adjusted=False, reason=""
    )


#: Things this system owns. A message naming one of them is asking about this
#: system's business, whatever else it does or fails to do.
#:
#: Matched at a word start, so plurals and inflections come free
#: ("appointments", "documents", "rescheduled", "cancellation") without the
#: substring trap that put "erm" inside *derm*atology. They are all long words,
#: which is what makes prefix matching safe here.
_DOMAIN_SUBJECTS: tuple[str, ...] = (
    "appointment",
    "document",
    "reminder",
    "booking",
    "reschedul",
    "cancel",
)

_DOMAIN_PATTERN = re.compile(
    r"\b(?:" + "|".join(_DOMAIN_SUBJECTS) + r")", re.IGNORECASE
)


def mentions_domain_subject(text: str) -> bool:
    """Whether a message names something this system administers.

    A veto, not a classifier. It can only ever *widen* what counts as in scope,
    and it decides nothing about what happens next — the message still has to
    be planned or classified like any other.

    The live failure it answers: "can you tell me my appointments" was
    scope-refused twice, while a differently-worded ask for the same thing went
    through. That is not a judgement call the model gets to make. Whether a
    message is about appointments is a fact about the message, and a request
    that names the system's own subject matter is never out of scope — at
    worst it is one the Coordinator failed to plan for, which is a different
    answer ("tell me more") from a refusal ("I don't do that").

    The false-positive direction is a clarifying question and never an action,
    which is why "cancel my streaming subscription" landing here is a cost
    worth paying. Compare the safety screen, whose false positives fill a queue
    a human has to read: the two guards look alike and their trades are
    opposite.
    """
    return bool(_DOMAIN_PATTERN.search(text or ""))


def _supersede_needs_difference(
    session: Session,
    run: WorkflowRun,
    *,
    message: str,
    incoming_steps: Sequence[PlanStep] | None,
    writer: TraceWriter,
) -> bool:
    """Would superseding this run throw away a review nobody replaced?

    **At `pending_review`, conflicting requires difference.** A run waiting on
    staff is a queue item as well as a conversation, and cancelling it destroys
    both — so a message that carries the run's own intent and names nothing new
    must not be able to do it. The live shape: a low-confidence route queues for
    review, the patient says "looks good, lets book that time", the assent
    classifies as conflicting because it carries the same intent, and the
    request a human was about to look at is gone. The patient watches a fresh
    search start from nothing and is never told.

    Difference means one of two things, and both are decided here rather than
    proposed:

    * **a different intent** — cancelling is not agreeing to book;
    * **a different subject** — the message resolves, against the Department
      table, to a department that is not this run's.

    The subject test is ``resolve_department`` on the message text, which is
    the same resolution routing uses everywhere else. No model argument, no
    keyword list: "a different subject" means what the table says it means, and
    a phrasing nobody anticipated cannot slip past a list that does not exist.

    Unknown intent supersedes. If the Coordinator proposed no steps there is
    nothing to compare, and the conservative reading of "requires difference"
    is that difference has not been *dis*proved — the patient keeps the ability
    to replace a request they may have meant to replace.
    """
    incoming = primary_intent(list(incoming_steps or []))
    if incoming is None or incoming is not primary_intent(run.plan or []):
        return False

    named = resolve_department(session, message or "")
    if named.get("status") == "resolved":
        current = (run.state or {}).get("department_id")
        if named["department"]["id"] != current:
            return False

    writer.validation(
        "supersede_at_review",
        accepted=False,
        detail={
            "intent": incoming.value,
            "problem": "same intent and no new subject while awaiting staff review",
            "subject": named.get("status"),
        },
    )
    return True


def _retire(
    session: Session, run: WorkflowRun, *, note: str, actor: User | None
) -> None:
    """Everything derived from a run that has just died, updated with it.

    The derivation invariant, in the transaction that killed the run — the same
    rule that moves a reminder when its appointment moves. Two rows outlive a
    cancelled run if nobody retires them, and both were found live on run 8:

    * **an open escalation**, which is a queue item a human would have picked
      up and worked on for a request the patient had already replaced;
    * **a stale proposal**, which is a held slot and an appointment id on a row
      whose status says it is over. Nothing can confirm it — the confirmation
      path reads the *active* run — but it is a lie in the record, and the
      staff viewer renders it.

    Safety escalations are never closed here; see
    :func:`~app.tools.tasks.close_escalations_for_run`.
    """
    close_escalations_for_run(
        session, workflow_run_id=run.id, note=note, actor=actor
    )
    run.clear_proposal()


def apply_consequence(
    session: Session,
    run: WorkflowRun,
    verdict: ClassVerdict,
    *,
    writer: TraceWriter,
    message: str,
    incoming_steps: Sequence[PlanStep] | None = None,
    actor: User | None = None,
) -> MappingOutcome:
    """Apply a validated class to the active run.

    Does not commit — the caller owns the transaction, as everywhere else.

    :raises ValidationFailed: the run is terminal. Mapping applies to the
        *active* run; a terminal one has none of the state these consequences
        assume, and ``escalated`` in particular must never be reopened —
        "actually forget it" after a safety trigger stays in front of humans.
    """
    if run.is_terminal:
        raise ValidationFailed(
            f"Run {run.id} is {run.status.value} and is no longer the active run. "
            "Terminal runs are owned by staff or already closed."
        )

    message_class = verdict.message_class
    consequence = CONSEQUENCE_FOR[message_class]

    # The class stays what it was — the message really is conflicting — and
    # only what it is *allowed to do* changes. Recording it as a side question
    # instead would make the trace describe a different message.
    # A message naming an appointment, a document or a reminder is about this
    # system's business, and calling it off-topic is a refusal the patient has
    # to work around by guessing a better phrasing. Live, "can you tell me my
    # appointments" was refused twice. The class stays `off_topic` in the trace
    # — recording it as something else would make the trace describe a
    # different message — and only what it may *do* changes: it becomes the
    # read-only class, which touches exactly as little.
    if consequence is Consequence.SCOPE_REPLY and mentions_domain_subject(message):
        writer.validation(
            "off_topic_vetoed",
            accepted=False,
            detail={"problem": "the message names a subject this system administers"},
        )
        consequence = Consequence.ANSWER_AND_STAY

    if (
        consequence is Consequence.SUPERSEDE
        and run.status is WorkflowStatus.PENDING_REVIEW
        and _supersede_needs_difference(
            session,
            run,
            message=message,
            incoming_steps=incoming_steps,
            writer=writer,
        )
    ):
        consequence = Consequence.STATUS_REPLY

    if consequence is Consequence.WITHDRAW:
        transition(
            session,
            run,
            to=WorkflowStatus.CANCELLED,
            trigger="patient_withdrawal",
            writer=writer,
            reason=_WITHDRAWAL_REASON[run.status],
            actor=actor,
        )
        _retire(session, run, note="Withdrawn by the patient.", actor=actor)

    elif consequence is Consequence.SUPERSEDE:
        transition(
            session,
            run,
            to=WorkflowStatus.CANCELLED,
            trigger="superseded_by_new_request",
            writer=writer,
            reason=CancellationReason.SUPERSEDED,
            actor=actor,
        )
        _retire(session, run, note="Superseded by a later request.", actor=actor)

    elif consequence is Consequence.APPEND_STEP:
        for step in incoming_steps or []:
            append_step(run, step)

    elif consequence is Consequence.FEED_RUN:
        # Part of the request, so routing and slot matching should read it.
        run.request_text = f"{run.request_text}\n{message}".strip()

    # ANSWER_AND_STAY, SCOPE_REPLY and STATUS_REPLY deliberately do nothing. A
    # side question is read-only, an off-topic message must leave the run —
    # including its request text — byte-identical, and a status reply is a
    # supersede that was refused: touching anything would be the very write the
    # refusal exists to prevent.

    return MappingOutcome(
        consequence=consequence,
        message_class=message_class,
        spawns_new_run=consequence is Consequence.SUPERSEDE,
        run_id=run.id,
    )


__all__ = [
    "CANONICAL_ORDER",
    "CLASSIFICATION_ORDER",
    "CONSEQUENCE_FOR",
    "ClassVerdict",
    "Consequence",
    "MappingOutcome",
    "apply_consequence",
    "mentions_domain_subject",
    "primary_intent",
    "validate_class",
]
