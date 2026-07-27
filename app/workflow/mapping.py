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

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from sqlalchemy.orm import Session

from app.errors import ClassRejected, ValidationFailed
from app.models import (
    CancellationReason,
    MessageClass,
    PlanStep,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.tools.departments import resolve_department
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


def primary_intent(plan: Sequence[str | PlanStep]) -> PlanStep | None:
    """The widest step in a plan — what the run is fundamentally *for*.

    Two messages share an intent when their primary steps match, which is the
    test that separates a cooperation from a rephrase.
    """
    present = set()
    for value in plan or []:
        present.add(value if isinstance(value, PlanStep) else PlanStep(value))
    for step in (PlanStep.BOOK, PlanStep.DOCUMENTS, PlanStep.ROUTE, PlanStep.FOLLOW_UP):
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
    "primary_intent",
    "validate_class",
]
