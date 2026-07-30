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
from app.tools.dates import MONTHS, PARTS_OF_DAY, WEEKDAYS
from app.tools.departments import resolve_department
from app.tools.tasks import close_escalations_for_run
from app.trace import TraceWriter
from app.workflow.confirmation import normalise
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
    #: difference. See :func:`_shows_no_difference`.
    STATUS_REPLY = "status_reply"
    #: The same refusal, one state earlier. A message that carries the run's own
    #: intent and names no new subject is *refining* the request it arrived in,
    #: not replacing it — so the timing question rides to the slot search and
    #: the run is left exactly where it was. Read-only, like a side question.
    REFINE = "refine"


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

    ``overruled`` says code disagreed with the model about this message — the
    class was adjusted, or the consequence the class maps to was not the one
    applied. It is the same set of facts the ``validation`` events above record;
    what it adds is a single thing for the reply to key on, so that a verdict
    code rejected cannot ship the prose the model wrote to go with it.
    """

    consequence: Consequence
    message_class: MessageClass
    spawns_new_run: bool
    run_id: int
    overruled: bool = False


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


#: What a patient abandoning a request actually says. Code-owned, and the
#: authority on whether a withdrawal may be *applied* — the model still
#: proposes the class, as it does for every other class, but this is the one
#: consequence that destroys the patient's work, so code checks the words
#: before executing it.
#:
#: The live case is as small as it gets. The agent asked "which appointment, 1
#: or 2?", the patient answered **"2"**, and gpt-4o-mini classified that as a
#: withdrawal. Code validated the class *name* — a real member of the enum, so
#: nothing objected — cancelled the run, and replied "I've closed that
#: request". The patient had said the opposite of that.
#:
#: The cost asymmetry decides which way the doubtful case falls: a withdrawal
#: applied wrongly destroys a request and tells the patient it was their idea;
#: a withdrawal missed costs one more message. So the rule is *cue required*,
#: and a message with no cue is fed to the run instead.
#:
#: Deliberately a twin of the mock's ``WITHDRAWAL_CUES`` rather than an import.
#: The mock's list is how the understudy *proposes* a class; this one is what
#: code will *apply*, and a provider sharing the guard that checks it would be
#: marking its own work.
WITHDRAWAL_CUES: tuple[str, ...] = (
    "never mind",
    "nevermind",
    "no mind",
    "forget it",
    "forget about it",
    "forget the whole",
    "don't bother",
    "dont bother",
    "no longer need",
    "not interested",
    "leave it",
    "drop it",
    "cancel that request",
    "cancel the request",
    "cancel my request",
    "cancel this request",
    "close that request",
    "close the request",
    "close my request",
    "close this request",
    "close the previous request",
    "withdraw",
    "call the whole thing off",
    "changed my mind",
)


def says_withdrawal(text: str) -> bool:
    """Whether the message actually contains a withdrawal cue."""
    lowered = (text or "").lower()
    return any(cue in lowered for cue in WITHDRAWAL_CUES)


def says_only_withdrawal(text: str) -> bool:
    """Whether the message is a withdrawal cue and *nothing else*.

    :func:`says_withdrawal` asks whether a withdrawal may be applied — a
    necessary condition on a class the model proposed. This asks the stronger
    question of whether code may apply one **on its own**, and the two have to
    be different tests because containment is not a safe basis for the
    consequence that destroys the patient's work. "I changed my mind, I'd like
    Tuesday instead" contains a cue and is a refinement; acting on the cue there
    would delete a live request and tell the patient it was their idea.

    So the rule is the confirmation reader's rule, for the confirmation
    reader's reason: **exact match after normalisation, never containment.**
    A message that is only the phrase is only the act. Anything with content
    around it goes back to the model, and costs at most one more message.

    The live case is "close the previous request", sent to a run that had been
    answering it with availability lists for six turns. The Coordinator did
    call it a withdrawal — and then called it something else in the same turn,
    which is the lottery this replaces rather than argues with.
    """
    token = normalise(text)
    return bool(token) and token in {normalise(cue) for cue in WITHDRAWAL_CUES}


#: What turning down an offered time actually sounds like. The decline half of
#: the confirmation read, and the exact counterpart of ``WITHDRAWAL_CUES``:
#: **the model proposes the verdict, and this is what code will apply it on.**
#:
#: The asymmetry that forced it. A withdrawal verdict has had a cue guard since
#: round 6 because applying one destroys the patient's work — and a *decline*
#: clears a held proposal, which is the same kind of destruction one step
#: smaller, and had no guard at all. Live, at ``pending_confirmation`` holding a
#: 2:00 PM slot: "yes lets confirm it" reached the model, came back
#: ``decline`` with the affirmative sentence quoted in its own ``reason`` field,
#: and code applied it unread. The proposal was cleared, the reply said "that's
#: fine, nothing has been booked", and the patient's reschedule died silently.
#:
#: Word-boundary matched, never containment: "no" lives inside "now" and
#: "know", and this list is exactly the kind of short vocabulary that put "erm"
#: inside *derm*atology.
_DECLINE_CUES = (
    r"no",
    r"nope",
    r"nah",
    r"not",
    r"don[’']?t",
    r"won[’']?t",
    r"can[’']?t make",
    r"cannot make",
    r"rather not",
    r"different (?:time|day|slot)",
    r"another (?:time|day|slot)",
    r"something else",
    r"cancel",
)

_DECLINE_PATTERN = re.compile(
    r"\b(?:" + "|".join(_DECLINE_CUES) + r")\b", re.IGNORECASE
)

#: What agreement sounds like. Not a confirmation reader — that is
#: :func:`~app.workflow.confirmation.read_confirmation`, it is exact-match only,
#: and nothing here may ever commit anything. This is a **veto**: a message
#: carrying one of these words is not a message code may treat as a refusal,
#: whatever else it contains and whatever the model called it.
#:
#: The direction is what makes it safe. Refusing to apply a decline costs the
#: patient one more message — the re-ask states what is held and what would
#: settle it. Applying one wrongly throws away a time they had just accepted.
_AFFIRMATIVE_CUES = (
    r"yes",
    r"yeah",
    r"yep",
    r"yup",
    r"sure",
    r"ok",
    r"okay",
    r"confirm(?:ed|ing|s)?",
    r"sounds good",
    r"go ahead",
    r"book it",
    r"do it",
    r"lets do",
    r"let[’']?s do",
)

_AFFIRMATIVE_PATTERN = re.compile(
    r"\b(?:" + "|".join(_AFFIRMATIVE_CUES) + r")\b", re.IGNORECASE
)


def says_decline(text: str) -> bool:
    """Whether the message actually contains a cue for turning an offer down.

    A necessary condition on a verdict the model proposed, in the same family as
    :func:`says_withdrawal` and read the same way: it decides nothing on its
    own, and the caller may only use it to check a decline the model already
    submitted.
    """
    return bool(_DECLINE_PATTERN.search(text or ""))


def says_affirmative(text: str) -> bool:
    """Whether the message contains a word of agreement.

    Only ever consulted to *refuse* a decline. "Yes lets confirm it" is not a
    confirmation — the tokens are exact and this is not one of them — but it is
    certainly not a refusal, and the honest answer to a yes the reader cannot
    read is to ask again rather than to guess in the expensive direction.
    """
    return bool(_AFFIRMATIVE_PATTERN.search(text or ""))


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


#: What a follow-up question is about, beyond the subjects every other part of
#: the system shares. Kept separate from ``_DOMAIN_SUBJECTS`` rather than folded
#: into it: that tuple feeds the scope-gate veto, whose behaviour on "nvidia
#: stock" and "who won the fifa final" is pinned byte for byte, and widening it
#: with words as ordinary as "pending" and "upcoming" would move a guard nobody
#: asked to move.
_FOLLOW_UP_SUBJECTS: tuple[str, ...] = (
    r"\btasks?\b",
    r"\bfollow[- ]?ups?\b",
    r"\bupcoming\b",
    r"\bpending\b",
    r"\boutstanding\b",
)

_FOLLOW_UP_PATTERN = re.compile("|".join(_FOLLOW_UP_SUBJECTS), re.IGNORECASE)


def names_followup_subject(text: str) -> bool:
    """Whether a message is about the patient's own outstanding business.

    The narrow companion to :func:`mentions_domain_subject`, and it exists
    because ``follow_up`` is the one plan step that names no subject. Every
    other step is about something — a department, an appointment, a document —
    so a plan containing it says what the message was about. A plan of
    ``["follow_up"]`` alone says nothing, and the gate had nothing to check it
    against: live, "what is the capital city of India?" was planned that way,
    accepted, and answered with a list of the patient's reminders.

    "task" is here rather than in ``_DOMAIN_SUBJECTS`` because it belongs to
    exactly this question. It is also not optional: "do I have any pending
    tasks?" is in neither the domain vocabulary nor the listing detector, so a
    gate assembled from those two alone would refuse the one follow-up question
    the live session actually got right.
    """
    return bool(_FOLLOW_UP_PATTERN.search(text or ""))


#: The verbs that act on an appointment the patient already has, each with the
#: plan step it names. Deliberately short, and deliberately *not* including
#: "change": every live phrasing used "reschedule", and a bare "change" beside
#: an appointment noun ("an appointment to change my dressing") would turn a
#: booking into a reschedule, which is the expensive direction.
_CHANGE_VERBS: tuple[tuple[PlanStep, str], ...] = (
    (PlanStep.CANCEL, r"\bcancel(?:s|led|ling)?\b"),
    (
        PlanStep.RESCHEDULE,
        r"\b(?:reschedul\w*|re-schedul\w*|rearrange\w*|move|moving|postpone\w*"
        r"|push back|bring forward)\b",
    ),
)

#: Verb **plus** appointment noun, never the verb alone — the same rule the
#: withdrawal/cancel collision already forced. "Cancel" on its own declines a
#: proposal; "cancel that request" withdraws a run; only "cancel my
#: appointment" acts on an appointment.
_APPOINTMENT_NOUN = re.compile(
    r"\b(?:appointments?|bookings?|visits?)\b", re.IGNORECASE
)


def names_change_verb(text: str) -> PlanStep | None:
    """The appointment verb this message plainly states, if it states one.

    A fact about the message, in the same family as
    :func:`mentions_domain_subject` and :func:`says_withdrawal`: it reads
    words, decides nothing about consequence, and the caller may only use it to
    check a plan the model already proposed.

    Withdrawal is checked first and wins, because "cancel that request" and
    "cancel my appointment" are different acts and reading the first as the
    second leaves an appointment standing while the reply says it was dealt
    with.
    """
    message = text or ""
    if says_withdrawal(message):
        return None
    if not _APPOINTMENT_NOUN.search(message):
        return None
    for step, pattern in _CHANGE_VERBS:
        if re.search(pattern, message, re.IGNORECASE):
            return step
    return None


#: The words a patient scopes a timing question with. Weekdays and months come
#: from ``resolve_date``'s own tables, so the two cannot drift: anything this
#: reads as a timing phrase is a phrase that module can turn into a window.
_TIMING_WORDS: tuple[str, ...] = (
    tuple(WEEKDAYS)
    + tuple(MONTHS)
    + tuple(PARTS_OF_DAY)
    + ("today", "tomorrow", "week", "weekend", "month", "earlier", "later", "sooner")
)

_TIMING_PATTERN = re.compile(
    r"\b(?:" + "|".join(sorted(_TIMING_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def names_timing(text: str) -> bool:
    """Whether a message scopes itself to a day, a date or a part of a day.

    A fact about the message, like :func:`mentions_domain_subject`, and used the
    same way: it may only widen what a turn *answers*, never what it does.

    The live failure. At ``pending_confirmation``, "do you have any appointments
    at the end of the same week like thursday or friday preferably in the
    afternoon?" was classified a side question — defensibly — and then answered
    with nothing. The Coordinator wrote prose about availability without running
    a search, the grounding guard correctly suppressed it, and what reached the
    patient was the bare re-ask, counted as a stall. The identical sentence one
    turn later, at ``in_progress``, got a real Thursday list. A question that
    names a day deserves the day's times in either state.

    False positives cost a slot list where a re-ask would have gone, which is
    the cheap direction — the opposite trade from the safety screen, and the
    same one :func:`mentions_domain_subject` makes.
    """
    return bool(_TIMING_PATTERN.search(text or ""))


#: "Book me something", "schedule a new one", "set up an appointment". Kept out
#: of ``_CHANGE_VERBS`` because it names no existing appointment and so belongs
#: to no plan correction — this is only ever counted, never applied.
_BOOK_VERB = re.compile(r"\b(?:book|booking|schedule|set up|make)\b", re.IGNORECASE)


def names_appointment_verbs(text: str) -> set[PlanStep]:
    """Every appointment verb this message plainly states.

    One request confirms one thing — ``validate_plan`` refuses a plan naming two
    appointment actions, and that rule is right and stays. What was missing is
    that the patient is never told which one survived. Live: "okay lets cancel
    that appointment and book a new one for skin rash" produced a booking, and
    the cancellation was neither done nor mentioned. The model had already
    dropped it at classification, so the plan validator never saw two verbs and
    had nothing to report; only the words show that two things were asked for.

    Same construction as :func:`names_change_verb` and the same two rules —
    withdrawal wins, and a verb needs an appointment noun beside it — so "cancel"
    answering a proposal and "cancel that request" closing a run are still not
    appointment verbs here.
    """
    message = text or ""
    if says_withdrawal(message) or not _APPOINTMENT_NOUN.search(message):
        return set()
    found = {
        step
        for step, pattern in _CHANGE_VERBS
        if re.search(pattern, message, re.IGNORECASE)
    }
    if _BOOK_VERB.search(message):
        found.add(PlanStep.BOOK)
    return found


def _shows_no_difference(
    session: Session,
    run: WorkflowRun,
    *,
    message: str,
    incoming_steps: Sequence[PlanStep] | None,
    writer: TraceWriter,
) -> bool:
    """Is this "conflicting" message actually a refinement of the run it arrived in?

    **Same intent, no new subject — so superseding it destroys the request the
    patient is still making.** The rule started life scoped to
    ``pending_review``, where the cost was loudest: a run waiting on staff is a
    queue item as well as a conversation, and "looks good, lets book that time"
    classified as conflicting, cancelled the request a human was about to look
    at, and started a fresh search the patient was never told about.

    It is the same mistake one state earlier, and round 5 found all three of
    its shapes in one live transcript:

    * "can you give me slots for next week?", asked during an Orthopedics
      proposal, **superseded the booking** — a question about times destroyed
      the request it was asking about;
    * "do you have any more appointment for the next week?" at
      ``pending_confirmation`` became a new run, routed nowhere in particular,
      queued for review, and answered *"a member of staff will assist you with
      that"* — a referral to a human for a question the slot table answers.

    One cause, so one rule, and what changes with the state is only what the
    refusal *becomes*: a status reply while staff hold the run, a slot answer
    while the system does.

    Difference means one of two things, and both are decided here rather than
    proposed:

    * **a different intent** — cancelling is not agreeing to book;
    * **a different subject** — the message resolves, against the Department
      table, to a department that is not this run's.

    The subject test is ``resolve_department`` on the message text, which is
    the same resolution routing uses everywhere else. No model argument, no
    keyword list: "a different subject" means what the table says it means, and
    a phrasing nobody anticipated cannot slip past a list that does not exist.
    "Book me a dermatology appointment instead" names a department that is not
    this run's and still supersedes, exactly as before.

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
        "supersede_refused",
        accepted=False,
        detail={
            "intent": incoming.value,
            "problem": "same intent and no new subject as the active run",
            "state": run.status.value,
            "subject": named.get("status"),
        },
    )
    return True


def _names_another_department(
    session: Session,
    run: WorkflowRun,
    *,
    message: str,
    writer: TraceWriter,
) -> bool:
    """Does this message name a desk that is not the one the run already routed to?

    The other half of :func:`_shows_no_difference`, and it exists because the
    subject check that function performs was only ever reachable through the
    *supersede* path. Live: a General Medicine proposal was declined, and the
    patient wrote "let me clarify — appointment for vision issues". The model
    called that a **continuation** with no steps of its own, which is a fair
    reading of the words — so nothing re-resolved anything, the completed
    routing step stayed completed, and the run answered an eye request with
    General Medicine slots. Clarifying again could not escape it: every
    rewording was another continuation fed into the same routed run, and only
    the word "instead" — which reaches the conflicting path — would have got
    the patient out.

    A refinement adds detail to a subject. A message naming a *different*
    department has changed it, whatever class fits the grammar, so this is the
    same shape as the rule that makes a different appointment verb conflicting
    however the model classified it.

    Three conditions, and each one is a direction this must not fire in:

    * **the run must already have a department.** A run still being routed has
      nothing to differ from, and superseding it would destroy the request on
      the patient's first clarification.
    * **the resolver must be certain.** ``ambiguous`` is the safety valve: "my
      kid has ear pain" during a run names two desks and settles nothing, and
      a supersede on a maybe is a supersede on noise.
    * **the message alone is read, never the run's accumulated text.** By the
      time this runs the continuation has not been appended yet — and it must
      not be, because the combined text says "general medicine" *and* "vision"
      and resolves to nothing but ambiguity.

    Only ``continuation`` is upgraded. A *complementary* message — "also,
    here's my old ECG" during a General Medicine booking — names Cardiology and
    is cooperating, and reading that as a new request is the most expensive
    mistake this module can make. It keeps its own consequence.
    """
    current = (run.state or {}).get("department_id")
    if not current:
        return False

    named = resolve_department(session, message or "")
    if named.get("status") != "resolved":
        return False

    different = int(named["department"]["id"]) != int(current)
    writer.validation(
        "new_subject",
        accepted=different,
        detail={
            "run_department_id": int(current),
            "named_department": named["department"]["name"],
            "matched_terms": named["matched_terms"],
            "applied": (
                Consequence.SUPERSEDE.value if different else Consequence.FEED_RUN.value
            ),
        },
    )
    return different


def _awaiting_selection(run: WorkflowRun) -> bool:
    """Has this run shown a numbered list and not yet had an answer to it?

    ``listed_appointment_ids`` is written by ``render_appointment_choice`` at
    the moment the list is rendered, and it outlives the answer — so "still
    waiting" also needs nothing to have been proposed or committed since.
    """
    state = run.state or {}
    if not state.get("listed_appointment_ids"):
        return False
    return run.proposed_action is None and not state.get("committed_action")


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
    # The consequence that destroys work needs the patient to have asked for
    # it. "2", answering "which appointment, 1 or 2?", was classified as a
    # withdrawal and cancelled the run — the class name was a real enum member,
    # so validation had nothing to object to, and the reply said "I've closed
    # that request" to a patient who was in the middle of making one.
    #
    # Downgraded to continuation rather than refused outright: the message was
    # still the patient talking to their own run, and feeding it in is what
    # answers "2". The class stays `withdrawal` in the trace — it is what the
    # model said — and only what it may do changes.
    if consequence is Consequence.WITHDRAW and not says_withdrawal(message):
        writer.validation(
            "withdrawal_cue",
            accepted=False,
            detail={
                "problem": "no withdrawal cue in the message",
                "applied": Consequence.FEED_RUN.value,
            },
        )
        consequence = Consequence.FEED_RUN

    # A run that has just asked the patient a question cannot then rule their
    # answer out of the conversation. "2", answering "which one, 1 or 2?",
    # carries no administrative noun only because the question already supplied
    # one — the same reasoning the confirmation state has always used for "hmm,
    # maybe". Two providers reached this two different ways: gpt-4o-mini called
    # it a withdrawal (caught above), the mock calls it off-topic.
    #
    # Only off-topic is downgraded here. A withdrawal mid-choice is handled by
    # the cue check above, which lets a genuine "forget the whole thing" land:
    # this guard is about answers being heard, not about trapping a patient in
    # a listing they no longer want.
    if consequence is Consequence.SCOPE_REPLY and _awaiting_selection(run):
        writer.validation(
            "selection_pending",
            accepted=False,
            detail={
                "proposed": consequence.value,
                "applied": Consequence.FEED_RUN.value,
                "problem": "the run is waiting for the patient to choose",
            },
        )
        consequence = Consequence.FEED_RUN

    if consequence is Consequence.SCOPE_REPLY and mentions_domain_subject(message):
        writer.validation(
            "off_topic_vetoed",
            accepted=False,
            detail={"problem": "the message names a subject this system administers"},
        )
        consequence = Consequence.ANSWER_AND_STAY

    # One rule, three states, two landings. Where staff hold the run the patient
    # is told its position and nothing moves; where the system holds it the
    # message is a refinement and gets answered with times. Superseding is
    # refused either way, and the refusal is traced either way.
    if consequence is Consequence.SUPERSEDE and _shows_no_difference(
        session,
        run,
        message=message,
        incoming_steps=incoming_steps,
        writer=writer,
    ):
        consequence = (
            Consequence.STATUS_REPLY
            if run.status is WorkflowStatus.PENDING_REVIEW
            else Consequence.REFINE
        )

    # And the mirror image: a *continuation* that names a different desk is not
    # one. Placed after the refusal above so the two rules cannot chase each
    # other — this one only ever reads a consequence the other never produces.
    if consequence is Consequence.FEED_RUN and _names_another_department(
        session, run, message=message, writer=writer
    ):
        consequence = Consequence.SUPERSEDE

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

    # ANSWER_AND_STAY, SCOPE_REPLY, STATUS_REPLY and REFINE deliberately do
    # nothing. A side question is read-only, an off-topic message must leave the
    # run — including its request text — byte-identical, and a status reply and
    # a refinement are both supersedes that were refused: touching anything
    # would be the very write the refusal exists to prevent.

    return MappingOutcome(
        consequence=consequence,
        message_class=message_class,
        spawns_new_run=consequence is Consequence.SUPERSEDE,
        run_id=run.id,
        # Either half counts: `validate_class` may have changed the class before
        # this function saw it, and every downgrade above changes what the class
        # is allowed to do. Both mean the same thing to a reply — the model's
        # account of this message was not the one that was acted on.
        overruled=verdict.adjusted or consequence is not CONSEQUENCE_FOR[message_class],
    )


__all__ = [
    "CANONICAL_ORDER",
    "CLASSIFICATION_ORDER",
    "CONSEQUENCE_FOR",
    "ClassVerdict",
    "Consequence",
    "MappingOutcome",
    "apply_consequence",
    "WITHDRAWAL_CUES",
    "mentions_domain_subject",
    "names_followup_subject",
    "names_appointment_verbs",
    "names_change_verb",
    "names_timing",
    "primary_intent",
    "says_affirmative",
    "says_decline",
    "says_only_withdrawal",
    "says_withdrawal",
    "validate_class",
]
