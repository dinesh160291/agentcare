"""``run_workflow`` — the one entry point, and the only place ADK is driven.

Everything above this line (the API, the UI, the evals) knows about a function
that takes a user and a message and returns a reply. Everything below it — the
runner, the session services, the callbacks — is confined here and in
``app/agents/``. That is what keeps the LangGraph fallback a fallback rather
than a rewrite.

The shape of a turn:

1. the inbound event opens it, before anything can fail;
2. if the run is waiting on a confirmation, the answer is read **in code** —
   an exact token, or nothing;
3. otherwise the Coordinator classifies the message against the active run, or
   plans a new one, and code validates what it proposed;
4. the validated plan is dispatched step by step to the specialist that owns
   it, each getting a typed task and no history;
5. the outbound event closes the turn, naming its author.

Two things this file refuses to do. It never lets the model apply a
consequence — every state change goes through the machine, every booking
through a tool, every date through ``resolve_date``. And it never lets a turn
end unrecorded: an unexpected exception writes its trace and commits *before*
propagating, because a turn that vanished is worse than a turn that failed.
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import clock
from app.agents import SPECIALIST_FOR_STEP, Toolbelt, TurnCallbacks, coordinator, memory
from app.agents.base import run_agent
from app.audit import write_audit
from app.config import get_settings
from app.db import SessionLocal
from app.errors import BudgetExceeded, ProviderError, ValidationFailed
from app.models import (
    APPOINTMENT_VERBS,
    AppointmentSlot,
    CancellationReason,
    EscalationKind,
    MessageClass,
    PatientProfile,
    PlanStep,
    ProposedAction,
    TERMINAL_WORKFLOW_STATUSES,
    TraceAuthor,
    TraceEvent,
    TraceEventType,
    User,
    UserRole,
    WorkflowRun,
    WorkflowStatus,
)
from app.safety import SafetyVerdict, escalate, keyword_screen, llm_screen
from app.tools import (
    book_appointment,
    cancel_appointment,
    create_escalation,
    list_patient_appointments,
    reschedule_appointment,
)
from app.tools.tasks import close_escalations_for_run
from app.trace import TraceWriter
from app.workflow.confirmation import ConfirmationAnswer, read_confirmation
from app.workflow.mapping import (
    Consequence,
    apply_consequence,
    mentions_domain_subject,
    names_appointment_verbs,
    names_change_verb,
    names_followup_subject,
    names_timing,
    primary_intent,
    refers_back,
    says_affirmative,
    says_decline,
    says_only_withdrawal,
    validate_class,
)
from app.workflow.recall import (
    clear_offer,
    mark_offer,
    pending_offer,
    recall_request,
    remember_dropped_verb,
)
from app.workflow.selection import Offer, read_selection
from app.workflow.queries import (
    QueryKind,
    answer_query,
    detect_query,
    names_document_subject,
)
from app.workflow.plan import (
    advance_plan,
    is_plan_complete,
    next_step,
    record_replan,
    validate_plan,
)
from app.workflow.replies import (
    ONE_VERB_AT_A_TIME,
    VERB_WORDS,
    claims_availability,
    offered_slot_ids,
    promises_action,
    render_outstanding,
    listed_appointment_ids,
    shortlist_slot_ids,
    render_alternatives,
    render_appointment_choice,
    window_heading,
    window_note,
    render_change_proposal,
    render_proposal,
    render_reask,
    render_receipt,
)
from app.workflow.state_machine import create_run, is_legal, transition
from app.workflow.targets import read_choice

# Code-authored replies. They are templates on purpose: a guard's output and a
# failure notice are the system's most deterministic moments, and they must
# read identically under `mock` and under a live provider.
SCOPE_REPLY = (
    "I can help with hospital administration — booking or changing appointments, "
    "documents, reminders, and follow-ups. What would you like to do?"
)
FAILED_REPLY = (
    "I'm sorry — I couldn't complete this request. A member of staff can help; "
    "I've flagged it for them."
)
WITHDRAWN_REPLY = "No problem — I've closed that request. Just ask if you need it again."
DECLINED_REPLY = (
    "That's fine, nothing has been booked. Tell me what time would suit you better."
)
NO_PLAN_REPLY = (
    "I want to make sure I get this right — could you tell me a little more about "
    "what you need help with?"
)
#: Said when the model planned something for a message that named nothing this
#: system administers. Distinct from ``SCOPE_REPLY``, which answers a message
#: the model itself declined to plan: there the invitation is the whole point,
#: because the patient may simply not have said enough yet. Here they said
#: plenty and none of it was ours, so the honest reply closes rather than asks.
#: No escalation sentence, because nothing is being sent anywhere — there is
#: nothing for a human to review about the capital of India.
UNSUPPORTED_TOPIC_REPLY = (
    "I handle hospital administration only: appointments, documents, reminders, "
    "and follow-ups — I can't help with that one."
)
#: Said when a message would have replaced a run that is waiting for staff, and
#: showed no difference to justify it. It states the run's *position*, never
#: anything about what staff will decide.
AWAITING_REVIEW_REPLY = (
    "Your request is with our staff for review — nothing has been lost, and I "
    "haven't started it again. They'll come back on it shortly. If you'd like "
    "something different instead, tell me and I'll take it from there."
)
NOTHING_TO_CONFIRM_REPLY = (
    "There's nothing waiting for your confirmation just now. Tell me what you'd "
    "like to do and I'll pick it up from there."
)
#: The bounded end of the re-ask loop. Explicit framing, code-authored, and
#: identical in mock and live: once the wording has failed three times, more
#: wording is not the answer.
STALLED_REASK_REPLY = (
    "Just to be clear about the time I offered: press ✅ Confirm or reply 'yes' "
    "and I'll book it — reply 'no' and I'll look for another. Nothing is booked "
    "until you say so."
)


@dataclass
class TurnResult:
    """What one turn produced. The API and the evals both read this."""

    reply: str
    author: TraceAuthor
    turn_id: str
    session_id: str
    run_id: int | None = None
    status: str | None = None
    message_class: MessageClass | None = None
    plan: list[str] = field(default_factory=list)
    steps_run: list[str] = field(default_factory=list)
    budget_exhausted: bool = False


# --- helpers ------------------------------------------------------------


def _patient_profile(session: Session, user: User) -> PatientProfile:
    profile = (
        session.query(PatientProfile).filter(PatientProfile.user_id == user.id).one_or_none()
    )
    if profile is None:
        raise ValueError(
            f"User {user.id} has no patient profile; run_workflow is a patient path."
        )
    return profile


def active_run(session: Session, patient_id: int) -> WorkflowRun | None:
    """The patient's one live run, if there is one.

    Newest first: a superseded run is terminal, so the newest non-terminal row
    is the only candidate. "One active run per patient" is a rule the mapping
    enforces; this query is how it is read back.
    """
    return (
        session.query(WorkflowRun)
        .filter(
            WorkflowRun.patient_id == patient_id,
            WorkflowRun.status.notin_(list(TERMINAL_WORKFLOW_STATUSES)),
        )
        .order_by(WorkflowRun.id.desc())
        .first()
    )


def _task_for(step: PlanStep, run: WorkflowRun, *, message: str, extra: dict) -> str:
    """The typed task a specialist receives — and the whole of what it receives.

    No transcript. Routing gets the request text because classifying the
    patient's words is its job; the others get state, not language.
    """
    task: dict[str, object] = {
        "step": step.value,
        "run_id": run.id,
        "today": clock.today().isoformat(),
        "department": (run.state or {}).get("department_name"),
        "department_id": (run.state or {}).get("department_id"),
    }
    if step is PlanStep.ROUTE:
        task["request_text"] = run.request_text or message
    if step in APPOINTMENT_VERBS:
        task["request_text"] = run.request_text or message
        task["proposed_slot_id"] = run.proposed_slot_id
        task["appointment_id"] = (run.state or {}).get("appointment_id")
        task["committed"] = (run.state or {}).get("committed_action")
    if step in (PlanStep.RESCHEDULE, PlanStep.CANCEL):
        # Which appointments the patient actually has. The specialist gets no
        # history, so "my appointment" means nothing to it without this — and
        # guessing the referent is the one thing story 20 forbids.
        #
        # Once the patient has answered the numbered list, that is the whole
        # list: the choice has been made, and handing a specialist two rows to
        # pick between is inviting it to make it again. `_settle_target` still
        # stands under this, so a model that proposes the other id anyway is
        # corrected rather than believed.
        chosen = (run.state or {}).get("chosen_appointment_id")
        rows = _live_appointments(run)
        if chosen is not None:
            rows = [row for row in rows if row["appointment_id"] == int(chosen)] or rows
        task["appointments"] = rows
        # And, once a numbered list has been shown, which ids that list held —
        # in the order the patient saw. Without it "2" is a number the model
        # has to map against a listing it wrote itself, which is how a bare
        # selection reply became unanswerable.
        task["listed_appointment_ids"] = listed_appointment_ids(run) or None
    task.update(extra)
    return json.dumps({k: v for k, v in task.items() if v is not None}, sort_keys=True)


def _live_appointments(run: WorkflowRun) -> list[dict]:
    """The patient's changeable appointments, for the two verbs that need one."""
    session = Session.object_session(run)
    if session is None:  # pragma: no cover - a detached run is a caller bug
        return []
    return list_patient_appointments(session, patient_id=run.patient_id, live_only=True)


async def _screen(
    message: str,
    *,
    callbacks: TurnCallbacks,
    writer: TraceWriter,
    user_id: str,
    provider: str | None,
) -> SafetyVerdict:
    """Both safety layers, in the one order they are allowed to run in.

    The deterministic screen goes first and always. Only if it passes does the
    model get asked — so nothing the model says can unblock a message the
    phrase list already stopped, which is what makes prompt injection a
    non-event here rather than a debate.

    The second layer is skipped for an answer the confirmation reader can read
    outright. "yes" and "no" are the two most common messages this system
    receives and neither can be a subtle emergency; spending a model call on
    them would be a cost with no possible finding. The skip is *recorded* —
    "the screen passed" and "the screen did not run" have to stay different
    facts in the trace.
    """
    verdict = keyword_screen(message)
    writer.guard_verdict(
        "safety_keyword_screen",
        passed=not verdict.fired,
        detail=verdict.as_trace_detail(),
    )
    if verdict.fired:
        return verdict

    if read_confirmation(message) is not ConfirmationAnswer.UNREAD:
        writer.guard_verdict(
            "safety_llm_screen",
            passed=True,
            detail={"skipped": "exact_token_answer", "source": "llm"},
        )
        return verdict

    verdict = await llm_screen(
        message,
        callbacks=callbacks,
        writer=writer,
        user_id=user_id,
        provider=provider,
    )
    writer.guard_verdict(
        "safety_llm_screen",
        passed=not verdict.fired,
        detail=verdict.as_trace_detail(),
    )
    return verdict


def _decline_proposal(
    session: Session,
    *,
    run: WorkflowRun,
    writer: TraceWriter,
    user: User,
    trigger: str,
    base: dict,
) -> TurnResult:
    """The patient said no. Back to slot selection, proposal cleared.

    One function for all three ways of declining — an exact token, a button, or
    the model's read of "not that one" — because a proposal left confirmable
    after a decline is the same bug however the decline arrived.
    """
    run.clear_proposal()
    transition(
        session,
        run,
        to=WorkflowStatus.IN_PROGRESS,
        trigger=trigger,
        writer=writer,
        actor=user,
    )
    return TurnResult(
        reply=DECLINED_REPLY,
        author=TraceAuthor.TEMPLATE,
        run_id=run.id,
        status=run.status.value,
        message_class=MessageClass.CONTINUATION,
        **base,
    )


def _offers(session: Session, slot_ids: list[int]) -> list[Offer]:
    """Slot ids as (id, start) pairs, in the order given, skipping any gone."""
    if not slot_ids:
        return []
    rows = {
        row.id: row
        for row in session.query(AppointmentSlot)
        .filter(AppointmentSlot.id.in_(slot_ids))
        .all()
    }
    return [
        Offer(slot_id=slot_id, start=rows[slot_id].start_time)
        for slot_id in slot_ids
        if slot_id in rows
    ]


def _settle_selection(
    session: Session,
    *,
    run: WorkflowRun,
    belt: Toolbelt,
    writer: TraceWriter,
    user: User,
    message: str,
    base: dict,
) -> TurnResult | None:
    """A choice among times already shown, read in code before any model call.

    ``None`` when the message is neither, which leaves the turn exactly as it
    was — so every sentence this cannot read is handled by whatever handled it
    before.

    **Why it runs here.** A run at ``in_progress`` that has shown times had no
    way to hear an answer to them. The Coordinator called "3pm will work for me"
    a new request, the mapping correctly refused to supersede a run with itself,
    and the refusal's consequence is to answer with more times — so the reply
    asked for a time, a time arrived, and the reply asked again. Seven messages
    in a row, live, and a second run in the same session died identically. The
    selection machinery existed and worked at ``pending_confirmation``; it was
    simply unreachable from one state earlier.

    Before the Coordinator rather than after it, on the same reasoning the
    listing questions use: a fallback for when classification goes wrong is
    still a lottery, because the wrong class is as likely as no class. Reading a
    number against a list this run rendered needs no model, and a turn that
    needs no model should not spend one.

    **It cannot commit.** The run lands in ``pending_confirmation`` holding the
    slot, and only an exact "yes" or the ✅ button books it. That gate is what
    makes reading generously safe here: a misread selection costs one decline,
    where a misread confirmation would book against the patient's word.

    **Both states, and originally only one.** The first version ran at
    ``in_progress`` alone, on the reasoning that a held slot meant the model's
    ``slot_question`` path already worked. It does not work often enough: with a
    three-slot list on screen, "option 3" was classified *conflicting*, refused
    a supersede, and answered with a fresh morning list; the next message, "I
    want the 4pm appointment on august 3", came back from the confirmation
    reader as a **decline** with the reason "Patient specified a different
    time". That closed enum has no selection in it, so a choice among the times
    on screen had no verdict it could be expressed as. The reader runs ahead of
    the exact tokens now, and the tokens are untouched underneath: "yes" and
    "no" name no time and no position, so they never reach it.

    **And both verbs that hold a slot**, for the reason given at the guard
    below: a reschedule's re-hold moves the time and nothing else.
    """
    # A cancellation holds no slot, so a time cannot be an answer to it: the
    # patient naming one is asking for something else and the model should
    # hear it. A *reschedule* does hold one, and this used to stand aside from
    # that too — because the hold went through `_propose_appointment`, which
    # sets `proposed_action` to BOOK and would have converted one kind of
    # proposal into another. The conversion is what was unsafe, not the
    # reading: `hold_offered_slot` now moves a reschedule's slot as a
    # reschedule, so the verb cannot flip and only the time moves. Standing
    # aside cost a live run — "3", answering three alternatives rendered under
    # a held reschedule, was matched by this reader, suppressed here, and then
    # classified a withdrawal by the model.
    if run.proposed_action is ProposedAction.CANCEL:
        return None

    # A message that is *only* a withdrawal phrase is the one thing that must
    # not be read as a choice. Live, "close the previous request" drew the same
    # availability list as the six selections before it — a patient asking to
    # be let go, answered with more of what they were trying to leave.
    if says_only_withdrawal(message):
        writer.validation(
            "withdrawal_cue",
            accepted=True,
            detail={"scope": "whole_message", "state": run.status.value},
        )
        transition(
            session,
            run,
            to=WorkflowStatus.CANCELLED,
            trigger="patient_withdrawal",
            writer=writer,
            reason=CancellationReason.WITHDRAWN,
            actor=user,
        )
        close_escalations_for_run(
            session, workflow_run_id=run.id, note="Withdrawn by the patient.", actor=user
        )
        run.clear_proposal()
        return TurnResult(
            reply=WITHDRAWN_REPLY,
            author=TraceAuthor.TEMPLATE,
            run_id=run.id,
            status=run.status.value,
            message_class=MessageClass.WITHDRAWAL,
            **base,
        )

    shortlist = _offers(session, shortlist_slot_ids(run))
    offered = _offers(session, offered_slot_ids(run))
    if not offered:
        return None

    slot_id = read_selection(message, shortlist=shortlist, offered=offered)
    writer.guard_verdict(
        "slot_selection",
        passed=slot_id is not None,
        detail={
            "slot_id": slot_id,
            "shortlist": [offer.slot_id for offer in shortlist],
            "offered": len(offered),
        },
    )
    if slot_id is None:
        return None

    held = belt.hold_offered_slot(slot_id)
    if not held.get("accepted"):
        # Taken between being shown and being chosen. The refusal carries fresh
        # alternatives, so the patient gets something to choose from rather than
        # a dead end — the same trade `_propose_another_slot` already makes for
        # the model's version of this move.
        return TurnResult(
            reply=render_alternatives(session, run, held.get("slots") or []),
            author=TraceAuthor.TEMPLATE,
            run_id=run.id,
            status=run.status.value,
            message_class=MessageClass.CONTINUATION,
            **base,
        )

    return TurnResult(
        reply=render_reask(session, run),
        author=TraceAuthor.TEMPLATE,
        run_id=run.id,
        status=run.status.value,
        message_class=MessageClass.CONTINUATION,
        **base,
    )


def _pending_change_verb(run: WorkflowRun) -> str | None:
    """The reschedule or cancel this run is still waiting to make.

    ``None`` once something has been proposed or committed: at that point the
    run is waiting on an answer to the proposal, not on which appointment.
    """
    if run.proposed_action is not None or (run.state or {}).get("committed_action"):
        return None
    return next(
        (step for step in (run.plan or []) if step in ("reschedule", "cancel")), None
    )


def _awaiting_choice(run: WorkflowRun) -> bool:
    """Is this run about to ask which appointment, and has it not been told?

    The question is a halt, not a failure — the same shape as a proposal
    waiting on a "yes". What makes the distinction load-bearing is the re-plan
    budget: see the call site.
    """
    if _pending_change_verb(run) is None:
        return False
    if (run.state or {}).get("chosen_appointment_id"):
        return False
    return len(_live_appointments(run)) > 1


def _choice_reask(session: Session, run: WorkflowRun) -> str:
    """The numbered list again, when this turn has nothing else to offer.

    "" unless the run is a change run still waiting to learn which appointment
    it is about — where there is a proposal, or a commit, or only one
    appointment, ``render_appointment_choice`` has nothing to ask.
    """
    verb = _pending_change_verb(run)
    if verb is None:
        return ""
    return render_appointment_choice(session, run, _live_appointments(run), verb=verb)


async def _settle_choice(
    session: Session,
    *,
    run: WorkflowRun,
    belt: Toolbelt,
    writer: TraceWriter,
    callbacks: TurnCallbacks,
    user: User,
    message: str,
    provider: str | None,
    settings,
    base: dict,
) -> TurnResult | None:
    """The answer to "which appointment?", read in code before any model call.

    ``None`` when the message answers no list, which leaves the turn as it was.

    ``render_appointment_choice`` numbers the patient's appointments and records
    the ids *so that* "2" can mean something. Nothing read it. The number went
    to the Coordinator like any other message, and live it came back classified
    a fresh cancellation request — the supersede was refused for naming no new
    subject, the refusal's recovery searched for slots on a run with no
    department and could only be told so, the re-plan budget went, and the run
    **failed**. The patient had answered the question they were asked.

    So the same placement the slot reader has, and for the same reason: a
    numbered answer to a list this run drew is not a judgement call, and a turn
    that needs no model should not spend one on a decision it might get wrong.

    What the answer does depends on the verb the run is already carrying, and
    it can never introduce one: a **cancel** run proposes cancelling that
    appointment — a proposal, so nothing is cancelled until the patient says the
    word — and a **reschedule** run carries on into its own plan with the target
    settled, which is where the slot search lives. Neither commits anything.
    """
    verb = _pending_change_verb(run)
    listed = listed_appointment_ids(run)
    if verb is None or not listed:
        return None

    appointment_id = read_choice(message, listed=listed)
    writer.guard_verdict(
        "appointment_selection",
        passed=appointment_id is not None,
        detail={"appointment_id": appointment_id, "listed": listed, "verb": verb},
    )
    if appointment_id is None:
        # Out of range, or not a position at all. The model gets it, exactly as
        # it did before this existed.
        return None

    def record() -> None:
        """The choice, and the words that carried it, onto the run.

        Written only once this turn is certainly ours. A path that recorded
        first and then handed the turn back to the model would append the
        message twice — once here and once when ``FEED_RUN`` appends it — and
        the duplicate is read later by routing and by slot matching.
        """
        run.request_text = f"{run.request_text}\n{message}".strip()
        state = dict(run.state or {})
        state["chosen_appointment_id"] = appointment_id
        run.state = state
        session.flush()

    if verb == "cancel":
        proposed = belt.propose_cancellation_for(appointment_id)
        if not proposed.get("accepted"):
            # Gone, or not theirs. The refusal is already traced; handing the
            # turn to the model is a better answer than a dead end.
            return None
        record()
        return TurnResult(
            reply=render_change_proposal(session, run),
            author=TraceAuthor.TEMPLATE,
            run_id=run.id,
            status=run.status.value,
            message_class=MessageClass.CONTINUATION,
            **base,
        )

    record()
    result = await _execute_plan(
        session,
        run=run,
        belt=belt,
        writer=writer,
        callbacks=callbacks,
        user=user,
        message=message,
        fallback_reply="",
        provider=provider,
        settings=settings,
        base=base,
    )
    result.message_class = MessageClass.CONTINUATION
    return result


def _settle_confirmation(
    session: Session,
    *,
    run: WorkflowRun,
    belt: Toolbelt,
    writer: TraceWriter,
    user: User,
    message: str,
    coordinator_reply: str,
    outcome,
    base: dict,
) -> TurnResult:
    """The model's half of the confirmation read: decline, or re-ask.

    It reaches here only for text the exact tokens could not read. Its verdict
    is one of two, and code enforces which two — see
    :data:`app.agents.toolbelt.CONFIRMATION_VERDICTS`.

    A stall never resolves implicitly in either direction. There is no
    auto-commit and no auto-decline: the run stays in ``pending_confirmation``
    until an explicit answer, a withdrawal, or a supersede. What the cap bounds
    is the *wording* spend, not the patient's time.
    """
    settings = get_settings()

    if belt.proposals.confirmation_verdict == "decline":
        # **A decline needs a decline cue.** The class name was in the enum and
        # the argument was well-formed, so nothing had anything to object to:
        # live, "yes lets confirm it" came back as ``decline`` — with that
        # sentence quoted in the verdict's own ``reason`` — and code cleared the
        # held slot and answered "that's fine, nothing has been booked". The
        # patient's reschedule died and they were told they had ended it.
        #
        # The same guard withdrawal has had since round 6, for the same reason
        # and with the same cost asymmetry: applying this destroys work the
        # patient did, refusing it costs one more message. Two conditions, and
        # together they cover the three cases — a cue is required, and a word of
        # agreement vetoes regardless of what sits beside it, so "both cues" and
        # "neither cue" both land on the re-ask below.
        cue = says_decline(message)
        agreement = says_affirmative(message)
        permitted = cue and not agreement
        writer.validation(
            "decline_cue",
            accepted=permitted,
            detail={
                "decline_cue": cue,
                "affirmative_cue": agreement,
                "applied": "decline" if permitted else "non_answer",
                **(
                    {}
                    if permitted
                    else {
                        "problem": (
                            "the message shows agreement"
                            if agreement
                            else "no decline cue in the message"
                        )
                    }
                ),
            },
        )
        if permitted:
            return _decline_proposal(
                session, run=run, writer=writer, user=user,
                trigger="model_read_decline", base=base,
            )
        # An overruled verdict takes its prose with it — the round-8 rule, which
        # lives in ``_continue_run`` and reads ``MappingOutcome.overruled``. It
        # cannot fire for this: the confirmation verdict is a different axis from
        # the message class, and the class here ("continuation", usually) was
        # applied exactly as the model proposed. So the drop happens here, at the
        # point where the reply is still the model's to lose. The sentence was
        # written believing the proposal was about to be cleared, and it is not.
        coordinator_reply = ""

    cap = settings.max_confirmation_non_answers
    stalled = run.non_answer_count >= cap
    writer.guard_verdict(
        "confirmation_stall",
        passed=not stalled,
        detail={"non_answers": run.non_answer_count, "cap": cap},
    )

    # The facts are code's either way. What the cap governs is whether the
    # model gets to put a sentence in front of them.
    facts = render_reask(session, run)

    # A re-ask that announces an action is describing a turn that is already
    # over: nothing runs after the reply, so the patient waits for something
    # that will not arrive and then repeats themselves — which the counter
    # above then reads as a stall. Live: "looks good. lets book that time" was
    # answered with "I will proceed to find a suitable time for you. Please
    # hold on", on a turn that was already holding one.
    lead = (
        ""
        if stalled
        else _guarded(
            coordinator_reply,
            writer=writer,
            searched_slots=belt.proposals.searched_slots,
        )[0]
    )
    return TurnResult(
        reply=f"{lead}\n{facts}".strip(),
        author=TraceAuthor.LLM if lead else TraceAuthor.TEMPLATE,
        run_id=run.id,
        status=run.status.value,
        message_class=outcome.message_class,
        **base,
    )


def _review_wall(
    session: Session,
    *,
    run: WorkflowRun,
    writer: TraceWriter,
    message: str,
    outcome,
    base: dict,
) -> TurnResult:
    """What a request waiting on a person may say back, and nothing else.

    A run at ``pending_review`` is a queue item as well as a conversation, and
    it is the one state where **the system is not the thing holding the work**.
    Everything the run could otherwise do to itself — route again, propose a
    time, search the schedule, let the model write a sentence — is a promise
    only a person can keep, and the state cannot honour any of it.

    Live, run 6 generated half a transcript out of that gap. "Book me a
    cardiology appointment" was classified a *continuation*, so the plan carried
    on **inside the queued run**: routing re-ran and was accepted, a slot was
    proposed and accepted, and the model shipped "shall I book it?" — an offer
    nothing could honour. The exact "yes" that followed then had no proposal
    state to land in, so it fell to the classifier and leaked "it seems that
    your response was not a confirmation…", twice. "The earliest the better"
    looped ``list_other_slots`` nine times into the iteration budget and ended
    in the apology template.

    So code answers here, before any dispatch. Three things still get through,
    and each is something only the patient can decide:

    * a **genuine supersede** — a different intent or a different subject; it
      cancels the run and starts another, which is the patient's right and is
      already gated by ``_shows_no_difference``;
    * a **withdrawal** — likewise theirs, and likewise already gated by the cue
      rule;
    * a **read-only status question** — "show my upcoming appointments" reads
      rows and touches nothing, and it was answered correctly mid-review live.

    The first two have already happened by the time this runs: both transition
    the run off ``pending_review``, so this function never sees them. The third
    is checked here rather than left to the class, because at this state the
    class is exactly what went wrong — the same reasoning that keyed the timing
    answer on the *destination* instead of the label.
    """
    query = detect_query(message)
    if query is not None:
        writer.guard_verdict(
            "review_wall",
            passed=True,
            detail={"answered": "query", "kind": query.value},
        )
        return TurnResult(
            reply=answer_query(
                session, patient_id=run.patient_id, kind=query, message=message
            ),
            author=TraceAuthor.TEMPLATE,
            run_id=run.id,
            status=run.status.value,
            message_class=outcome.message_class,
            **base,
        )

    writer.guard_verdict(
        "review_wall",
        passed=False,
        detail={
            "answered": "wall",
            "class": outcome.message_class.value,
            "consequence": outcome.consequence.value,
            "problem": "a run in front of staff may not route, propose or search",
        },
    )
    return TurnResult(
        reply=AWAITING_REVIEW_REPLY,
        author=TraceAuthor.TEMPLATE,
        run_id=run.id,
        status=run.status.value,
        message_class=outcome.message_class,
        **base,
    )


def _guarded(
    text: str,
    *,
    writer: TraceWriter,
    fallback: str = "",
    searched_slots: bool = False,
) -> tuple[str, bool]:
    """Model prose, unless it announces an errand or invents availability.

    Returns (text, was_model).

    **The errand.** A reply that says the assistant is off doing something
    describes a turn that is already over: nothing runs after the reply is
    written. Live, at ``in_progress``, a side question got "noted as a side
    question. I will continue... Please hold on for a moment" — twice, with
    nothing following either time. The patient waits, then repeats themselves,
    and the stall counter reads their patience as a stall.

    **The invented schedule.** A reply that states what is or is not free has
    to have looked. Live: "It appears that there are currently no available
    appointment slots in the ENT department for next week" — on a turn with no
    ``resolve_date`` and no slot search at all, and contradicted two turns
    later by a search returning **72** matching slots in that same window. The
    model had two ``list_other_slots`` results saying *"No department has been
    decided yet"* and reported a refusal-to-search as an empty schedule.

    ``searched_slots`` is set by the tools that actually return slots, so this
    is grounding rather than phrasing: a claim with a search behind it passes
    whatever words it uses, and a claim with nothing behind it fails whatever
    words it uses. Which is why the wording list can afford to be short.

    Applied to every model-authored reply rather than only to the re-ask,
    because neither failure is a property of the confirmation state — both are
    properties of a model being asked to say something it has no basis for.
    """
    if not text:
        return fallback, False
    if promises_action(text):
        writer.guard_verdict(
            "reply_promises_action", passed=False, detail={"reply": text[:200]}
        )
        return fallback, False
    if not searched_slots and claims_availability(text):
        writer.guard_verdict(
            "reply_claims_availability",
            passed=False,
            detail={"reply": text[:200], "problem": "no slot search ran this turn"},
        )
        return fallback, False
    return text, True


def _answer_while_holding(
    session: Session,
    *,
    run: WorkflowRun,
    belt: Toolbelt,
    outcome,
    base: dict,
) -> TurnResult | None:
    """The answer half of answer-and-stay, wherever the question was asked.

    ``None`` when this turn did neither of the two things, which leaves the
    confirmation path exactly as it was.

    The mapping table has always said a side question is answered and the run
    stays put. At ``pending_confirmation`` the code went straight to
    decline-or-re-ask, so the answering never happened: live, "show me more
    available slots" got the re-ask nag, twice. Both branches below leave the
    run in ``pending_confirmation``, and neither can commit — a re-proposal is
    still a proposal, and the patient still has to say the word.

    The reply is assembled rather than delegated because the facts are slot
    ids, doctors and times, and a paraphrase of those is how a patient ends up
    at the wrong clinic on the wrong day.
    """
    if belt.proposals.reproposed:
        # The shortlist is only there if this turn also listed times. "The 2pm
        # one", answering a list from three exchanges ago, moves the proposal
        # with nothing fresh to show — so the fallback states what is now held,
        # which is the fact that actually changed.
        return TurnResult(
            reply=(
                render_proposal(session, run, belt.proposals.offered_slots)
                or render_reask(session, run)
            ),
            author=TraceAuthor.TEMPLATE,
            run_id=run.id,
            status=run.status.value,
            message_class=outcome.message_class,
            **base,
        )

    if belt.proposals.answered_with_slots:
        # A time the patient asked for that the search withheld because they
        # are already busy at it. Without the sentence the list reads as an
        # answer to a question nobody asked: "how about 11am?" came back as 9,
        # 10 and 2, with the patient's own 11:00 in another department the only
        # reason 11 was missing. Prefixed with a blank line between, because
        # the chat renders CommonMark and a sentence one newline above a list
        # is read as part of it.
        note = belt.proposals.clash_note
        # A constraint that was stated and not honoured is said out loud, and it
        # takes the list's heading rather than sitting above it. Live, "Thursday
        # next week or after August 6th" produced three Monday slots under
        # "Other times that are free:" — a silent fallback the patient reads as
        # an answer.
        # Three things a heading can be, and the order is a precedence. The two
        # failures come first — an unread constraint and an honoured-but-empty
        # window are both about the list not being the answer asked for. Only
        # when neither applies is there room to say whose window it *was*, and
        # that is set solely by the model-proposed path.
        heading = window_note(
            unreadable=belt.proposals.window_unreadable,
            empty_label=belt.proposals.window_empty_label,
            part_of_day=belt.proposals.window_part_of_day,
        )
        if not heading and belt.proposals.window_provenance:
            heading = window_heading(belt.proposals.window_provenance)
        times = render_alternatives(
            session,
            run,
            belt.proposals.offered_slots,
            **({"heading": heading} if heading else {}),
        )
        return TurnResult(
            reply=f"{note}\n\n{times}" if note else times,
            author=TraceAuthor.TEMPLATE,
            run_id=run.id,
            status=run.status.value,
            message_class=outcome.message_class,
            **base,
        )

    return None


def _corrected_change_plan(
    session: Session,
    plan: list[PlanStep] | None,
    *,
    message: str,
    patient_id: int,
    writer: TraceWriter,
) -> list[PlanStep] | None:
    """A verb the patient stated is not the model's to reinterpret.

    "Please reschedule my appointment to next week" reached routing and queued
    for a staff decision on which department should own it — a question already
    answered by the appointment being moved. The same sentence had worked hours
    earlier, and nothing in between touched planning: two live replays of one
    phrasing produced a correct plan, a ``[route, book]`` plan, and *no plan at
    all*. It was never deterministic. A patient cannot be asked to keep
    rephrasing until the dice land.

    So where the message plainly names an appointment verb — verb **plus** an
    appointment noun, withdrawal excluded, and only when the patient actually
    has an appointment to act on — the plan is that verb's closure, and a plan
    proposing anything else is corrected. Both directions are traced: an
    agreement is worth as much as an override when the question is whether this
    ran at all.

    Deliberately narrow, and narrow in the direction that costs least. The
    verb list holds no bare "change"; a patient with nothing booked is left
    entirely to the model, because "reschedule my appointment" from someone who
    has none is a booking request badly worded, and only the model can tell.
    """
    verb = names_change_verb(message)
    if verb is None:
        return plan
    if not list_patient_appointments(session, patient_id=patient_id, live_only=True):
        return plan

    if plan is not None and primary_intent(plan) is verb:
        writer.validation(
            "plan_change_verb",
            accepted=True,
            detail={"verb": verb.value, "plan": [step.value for step in plan]},
        )
        return plan

    corrected = validate_plan([verb.value])
    writer.validation(
        "plan_change_verb",
        accepted=False,
        detail={
            "verb": verb.value,
            "proposed": [step.value for step in plan] if plan else None,
            "applied": [step.value for step in corrected],
            "problem": "the message names an appointment verb the plan does not",
        },
    )
    return corrected


def _plan_floor(
    session: Session,
    *,
    message: str,
    belt: Toolbelt,
    writer: TraceWriter,
    user: User,
) -> list[PlanStep] | None:
    """A canonical plan for a booking the patient stated and nobody planned.

    The second of the two places code may supply a plan, and the twin of
    :func:`_corrected_change_plan` — which already covers the change verbs, for
    a patient who has something to change. What had no equivalent is ``book``,
    because it closes over nothing that exists yet, and that is the verb the
    live failure was about: "I wanted to book an appointment for knee pain"
    reached ``submit_plan`` as ``["conflicting"]`` four times, as
    ``["cancel", "book"]`` once, and then as prose. Every guard did its job and
    the patient met a clarifying question they had already answered in the
    sentence. Rephrasing produced the identical fixation.

    Three conditions, and each of them narrows it deliberately:

    * **Nothing was accepted, and something was attempted.** A model that
      planned is never second-guessed here — this fills a hole, it does not
      arbitrate. And where nothing was even proposed, the clarify is still the
      honest answer.
    * **The words name exactly one verb, and it is ``book``.** Reusing
      :func:`names_appointment_verbs`, so the vocabulary is the one the dropped
      verb line and the plan correction already read. Two verbs, or a change
      verb, and this stands aside: one request confirms one thing, and the
      change verbs have an owner already.
    * **The plan goes through ``validate_plan`` like any other.** The floor
      cannot express anything a Coordinator could not, and every downstream
      guard runs unchanged — routing still resolves the department from the
      Department table, which is why the fallback never has to guess one.
    """
    if belt.proposals.plan is not None or not belt.proposals.rejections:
        return None
    if names_appointment_verbs(message) != {PlanStep.BOOK}:
        return None

    plan = validate_plan([PlanStep.BOOK.value])
    writer.guard_verdict(
        "plan_floor",
        passed=True,
        detail={
            "verb": PlanStep.BOOK.value,
            "plan": [step.value for step in plan],
            "rejected_attempts": len(belt.proposals.rejections),
        },
    )
    write_audit(
        session,
        action="plan_floor_applied",
        entity_type="workflow_run",
        actor=user,
        metadata={"verb": PlanStep.BOOK.value},
    )
    return plan


#: How many fresh-turn clarifies in a row before a human is asked to look.
#: Two, so the third message is answered by a person rather than by the same
#: question a third time.
MAX_CONSECUTIVE_CLARIFIES = 2

CLARIFY_ESCALATED_REPLY = (
    "I'm sorry — I haven't been able to pin down what you need, so I've asked a "
    "member of staff to pick this up. They'll be in touch."
)


def _consecutive_clarifies(session: Session, *, session_id: str, turn_id: str) -> int:
    """How many fresh-turn clarifies run unbroken up to this turn.

    The clarify is bounded like every other automated writer here: a reply that
    invites a rephrase, given to a rephrase, is a loop, and live it ran until
    the patient gave up. What makes it a *loop* rather than a bad answer is that
    nothing about the turn changes between iterations.

    Counted from the trace because a fresh turn writes nothing else — no run,
    no proposal, nowhere on a row to keep a counter. Consecutive means
    consecutive: any turn in between that did something else resets it, so a
    patient who books successfully and is puzzling over something new later
    starts from zero rather than inheriting an old tally.
    """
    rows = (
        session.query(TraceEvent.turn_id, TraceEvent.event_type, TraceEvent.payload)
        .filter(TraceEvent.session_id == session_id)
        .order_by(TraceEvent.id)
        .all()
    )
    order: dict[str, None] = {}
    clarified: set[str] = set()
    for row_turn, event_type, payload in rows:
        if row_turn is None:
            continue
        order.setdefault(row_turn, None)
        if (
            event_type is TraceEventType.GUARD_VERDICT
            and (payload or {}).get("guard") == "clarify_repeat"
        ):
            clarified.add(row_turn)

    count = 0
    for previous in reversed([t for t in order if t != turn_id]):
        if previous not in clarified:
            break
        count += 1
    return count


#: The plan steps that name no subject of their own. Every other step is about
#: something the message had to mention — a department, an appointment, a time —
#: so a plan containing one of those *is* evidence about the request. These two
#: are not: the follow-up agent can run on any patient at any time, and the
#: documents agent reads whatever is on file.
_SUBJECTLESS_STEPS = frozenset({PlanStep.DOCUMENTS, PlanStep.FOLLOW_UP})


def _earns_plan(message: str) -> bool:
    """Whether a message has said enough to deserve a plan that names no verb.

    Four ways, and they are all existing readers rather than a new opinion: the
    domain subjects every guard here shares, the follow-up nouns that name a
    patient's own outstanding business, the document words the listing detector
    already owns, and the detector itself.

    The false-positive direction costs a run that reads the patient's own
    reminders back to them — mildly silly. The false-negative direction refuses
    a real question about their own tasks or paperwork, which is the expensive
    one, so the test is deliberately generous and every one of its four limbs is
    a reader that already existed.
    """
    return (
        mentions_domain_subject(message)
        or names_followup_subject(message)
        or names_document_subject(message)
        or detect_query(message) is not None
    )


def _escalate_unplannable(
    session: Session,
    *,
    writer: TraceWriter,
    user: User,
    message: str,
    base: dict,
) -> TurnResult:
    """Hand a request nobody could plan to a human.

    A run is created to key the escalation to, for the reason
    :func:`_budget_failure` gives: every escalation points at a run through a
    non-nullable column, and one row is a better trade than one nullable key
    for one path. It is born ``in_progress`` and escalated a line later,
    because the transition is what the trace needs.

    The kind is ``unsupported_request`` rather than ``system_failure``: nothing
    broke. The request is real, it names something this system administers, and
    the planner could not turn it into steps — which is exactly the case a
    human is better at than another clarifying question.
    """
    profile = _patient_profile(session, user)
    run = create_run(
        session,
        patient_id=profile.id,
        status=WorkflowStatus.IN_PROGRESS,
        trigger="clarify_loop_escalated",
        writer=writer,
        request_text=message,
        plan=[],
        session_id=base.get("session_id"),
        actor=user,
    )
    transition(
        session,
        run,
        to=WorkflowStatus.ESCALATED,
        trigger="clarify_loop_escalated",
        writer=writer,
        actor=user,
        detail={"consecutive_clarifies": MAX_CONSECUTIVE_CLARIFIES + 1},
    )
    create_escalation(
        session,
        workflow_run_id=run.id,
        kind=EscalationKind.UNSUPPORTED_REQUEST,
        reason=(
            "The patient rephrased a request naming an administrative subject "
            "three times and no plan could be produced."
        ),
        message=message,
        actor=user,
    )
    write_audit(
        session,
        action="clarify_loop_escalated",
        entity_type="workflow_run",
        entity_id=run.id,
        actor=user,
        metadata={"consecutive_clarifies": MAX_CONSECUTIVE_CLARIFIES + 1},
    )
    return TurnResult(
        reply=CLARIFY_ESCALATED_REPLY,
        author=TraceAuthor.TEMPLATE,
        run_id=run.id,
        status=run.status.value,
        **base,
    )


def _offer_remembered(
    session: Session,
    *,
    writer: TraceWriter,
    user: User,
    patient_id: int,
    message: str,
    conversation_id: str,
    base: dict,
) -> TurnResult | None:
    """A refusal that names the request it could pick up instead.

    ``None`` when there is nothing to name, which leaves the generic reply
    exactly as it was — the negative control, and the thing that keeps this
    falsifiable.

    Two conditions, and both are about the *message* rather than about the
    memory. It must point backwards (:func:`refers_back`), because a memory
    consulted on every refusal would answer "who won the fifa final" with an
    offer to restart a booking. And it must have reached a refusal at all: a
    message the Coordinator planned for is planned for, and code supplying
    something else here would be the planning bin all over again.

    Nothing is started. The offer is recorded on the run it refers to, and the
    only thing that can act on it is the patient's own exact word on the very
    next turn.
    """
    if not refers_back(message):
        return None

    remembered = recall_request(
        session,
        patient_id=patient_id,
        session_id=conversation_id,
        turn_id=writer.turn_id,
    )
    writer.guard_verdict(
        "request_recalled",
        passed=remembered is not None,
        detail=(
            {
                "kind": remembered.kind,
                "verb": remembered.verb.value,
                "run_id": remembered.run_id,
            }
            if remembered is not None
            else {"problem": "no unfinished request in this conversation"}
        ),
    )
    if remembered is None:
        return None

    mark_offer(session, remembered, turn_id=writer.turn_id)
    write_audit(
        session,
        action="request_recall_offered",
        entity_type="workflow_run",
        entity_id=remembered.run_id,
        actor=user,
        metadata={"kind": remembered.kind, "verb": remembered.verb.value},
    )
    return TurnResult(
        reply=remembered.reply, author=TraceAuthor.TEMPLATE, **base
    )


async def _restart_remembered(
    session: Session,
    *,
    writer: TraceWriter,
    callbacks: TurnCallbacks,
    belt: Toolbelt,
    user: User,
    patient_id: int,
    message: str,
    conversation_id: str,
    provider: str | None,
    settings,
    base: dict,
) -> TurnResult | None:
    """The "yes" that starts a request the patient was offered.

    ``None`` unless the previous turn made an offer, so a bare "ok" with nothing
    pending falls through to the ordinary clarify — the memory must never turn a
    stray token into a run.

    The plan is the verb's own closure through ``validate_plan``, exactly as
    :func:`_plan_floor` and :func:`_corrected_change_plan` build theirs: this
    cannot express anything a Coordinator could not, and every guard downstream
    runs unchanged. The request text is the **remembered subject** rather than
    the word "yes" — the department the patient's own sentence resolved to, which
    is what makes the restarted run route with the confidence that sentence
    earned instead of handing routing a pronoun.
    """
    remembered = pending_offer(
        session,
        patient_id=patient_id,
        session_id=conversation_id,
        turn_id=writer.turn_id,
    )
    if remembered is None:
        return None

    plan = validate_plan([remembered.verb.value])
    writer.guard_verdict(
        "request_restarted",
        passed=True,
        detail={
            "kind": remembered.kind,
            "verb": remembered.verb.value,
            "from_run_id": remembered.run_id,
            "plan": [step.value for step in plan],
        },
    )
    clear_offer(session, remembered)
    run = create_run(
        session,
        patient_id=patient_id,
        status=WorkflowStatus.IN_PROGRESS,
        trigger="remembered_request_restarted",
        writer=writer,
        request_text=remembered.subject or message,
        plan=[step.value for step in plan],
        session_id=conversation_id,
        actor=user,
    )
    write_audit(
        session,
        action="request_recall_accepted",
        entity_type="workflow_run",
        entity_id=run.id,
        actor=user,
        metadata={"kind": remembered.kind, "verb": remembered.verb.value},
    )
    belt.run = run
    return await _execute_plan(
        session,
        run=run,
        belt=belt,
        writer=writer,
        callbacks=callbacks,
        user=user,
        message=remembered.subject or message,
        fallback_reply="",
        provider=provider,
        settings=settings,
        base=base,
    )


def _budget_failure(
    session: Session,
    *,
    run: WorkflowRun | None,
    writer: TraceWriter,
    user: User,
    base: dict,
    stage: str,
) -> TurnResult:
    """A budget blew. Report it — never pass the stub's last words off as an answer.

    It can blow before a run exists (in the safety screen, or in the
    Coordinator on a first message), and there is then nothing to transition to
    ``failed``. The turn still has to say so — and, since both branches answer
    with ``FAILED_REPLY`` and that template promises staff have been told, both
    branches have to make it true. So a run is created here purely to have
    something to fail and to key the escalation to, on the same reasoning the
    safety module gives for its born-escalated run: every escalation points at
    a run through a non-nullable key, and one nullable column for one path is a
    worse trade than one row.
    """
    if run is None:
        write_audit(
            session,
            action="turn_budget_exhausted",
            entity_type="workflow_run",
            actor=user,
            metadata={"stage": stage},
        )
        # Born ``in_progress`` and failed a line later, rather than born
        # ``failed``: the state machine refuses the latter, and it is right to
        # — "runs are born in_progress, or escalated" is a pinned rule, and the
        # failure edge it would bypass is the one that writes the transition
        # this turn needs in its trace.
        profile = _patient_profile(session, user)
        run = create_run(
            session,
            patient_id=profile.id,
            status=WorkflowStatus.IN_PROGRESS,
            trigger=f"turn_budget_exhausted_{stage}",
            writer=writer,
            request_text="",
            plan=[],
            session_id=base.get("session_id"),
            actor=user,
        )
    return _fail_run(
        session,
        run=run,
        writer=writer,
        user=user,
        reason="tool_iteration_budget",
        base=base,
        plan=list(run.plan or []),
        steps_run=[],
    )


# --- the turn ------------------------------------------------------------


@dataclass
class _Envelope:
    """What a turn is handed, and where it puts what it produced."""

    session: Session
    writer: TraceWriter
    callbacks: TurnCallbacks
    user: User
    conversation_id: str
    result: TurnResult | None = None


@asynccontextmanager
async def _turn_envelope(
    user: User,
    *,
    content: str,
    author: TraceAuthor,
    session_id: str | None,
):
    """The bracket every turn shares, whichever front door it came through.

    **The system has two front doors** — a chat message and a typed action —
    and the turn grammar recognises both. Sharing this envelope is what keeps
    that true: one place opens a turn with an inbound event, one place closes
    it with an outbound event, and one place commits. A second copy would be a
    second place for a typed-action turn to end up unbracketed, which is
    exactly the shape of hole the well-formedness checker exists to catch and
    the easiest one to introduce by writing the button path separately.

    The turn *is* the transaction: the state change, its audit row, and its
    trace rows commit together or not at all. An unexpected exception writes
    its trace and **commits** before propagating — a turn that vanished is
    worse than a turn that failed.
    """
    settings = get_settings()
    session = SessionLocal()
    conversation_id = session_id or f"conv-{uuid.uuid4().hex[:10]}"

    # Re-read the acting user into this session: the caller's instance belongs
    # to a session we do not own and may already be detached.
    acting = session.get(User, user.id)
    if acting is None:
        session.close()
        raise ValueError(f"No such user: {user.id}")
    if acting.role is not UserRole.PATIENT:
        session.close()
        raise ValueError("This is the patient path; staff act by typed action.")

    writer = TraceWriter(session, session_id=conversation_id)
    callbacks = TurnCallbacks(
        writer,
        max_tool_iterations=settings.max_tool_iterations,
        history_window_turns=settings.history_window_turns,
    )
    writer.inbound(content, author=author)

    envelope = _Envelope(
        session=session,
        writer=writer,
        callbacks=callbacks,
        user=acting,
        conversation_id=conversation_id,
    )
    try:
        yield envelope
        if envelope.result is None:  # pragma: no cover - a caller bug
            raise RuntimeError("A turn ended without producing a result.")
        writer.outbound(envelope.result.reply, author=envelope.result.author)
        session.commit()
    except Exception as exc:  # noqa: BLE001 - recorded and committed, then re-raised
        callbacks.fail_pending_request(f"{type(exc).__name__}: {exc}")
        writer.outbound(FAILED_REPLY, author=TraceAuthor.TEMPLATE, error=str(exc))
        write_audit(
            session,
            action="turn_failed",
            entity_type="workflow_run",
            entity_id=writer.workflow_run_id,
            actor=acting,
            metadata={"error": f"{type(exc).__name__}: {exc}"},
        )
        session.commit()
        # The trace now says the patient was answered with a template. Carry
        # that answer out with the exception so the layer that replies can send
        # the same words: a turn recorded one way and delivered another is a
        # trace that describes a system nobody used.
        if isinstance(exc, ProviderError):
            exc.turn = TurnResult(
                reply=FAILED_REPLY,
                author=TraceAuthor.TEMPLATE,
                turn_id=writer.turn_id,
                session_id=conversation_id,
                run_id=writer.workflow_run_id,
            )
        raise
    finally:
        session.close()


def _one_verb_note(session: Session, result: TurnResult, *, message: str) -> str:
    """The line owed to a message that asked for two appointment actions.

    ``validate_plan`` refuses a plan naming two — one request confirms one thing
    — and that rule is not in question. What was missing is that the patient
    never heard about it. Live: "okay lets cancel that appointment and book a
    new one for skin rash" produced a booking, no cancellation, and no mention
    of one. The model had dropped the verb at *classification*, so the plan
    validator never saw two and had nothing to complain about; the only record
    that two things were asked for is the sentence itself.

    Applied once, here, because the turn ends in a dozen different branches and
    a note appended in some of them is a note that reads as an inconsistency.
    It states what is being done and invites the rest — it promises nothing
    about the half that was dropped, because nothing was done about it.
    """
    if result.run_id is None:
        return ""
    verbs = names_appointment_verbs(message)
    if len(verbs) < 2:
        return ""
    run = session.get(WorkflowRun, result.run_id)
    if run is None:  # pragma: no cover - a run named by a result always exists
        return ""
    doing = next(
        (step for step in (run.plan or []) if step in VERB_WORDS),
        None,
    )
    if doing is None:
        return ""
    # And the half that is not being done is written down. This line promises
    # the patient they may "ask me about the other one right after", and until
    # now nothing recorded what the other one was — so "now do the other thing I
    # asked" named no subject, met the generic refusal, and the rephrase gave
    # routing "booking request" to resolve, which is not a department. A promise
    # is kept by the thing that made it.
    remember_dropped_verb(
        session,
        run,
        dropped=verbs - {PlanStep(doing)},
        message=message,
        turn_id=result.turn_id,
    )
    return ONE_VERB_AT_A_TIME.format(verb=VERB_WORDS[doing])


async def run_workflow(
    user: User,
    message: str,
    session_id: str | None = None,
    *,
    provider: str | None = None,
) -> TurnResult:
    """Run one conversational turn to completion.

    The seam. Everything above it — the API, the UI, the evals — knows about a
    function that takes a user and a message and returns a reply.
    """
    async with _turn_envelope(
        user,
        content=message,
        author=TraceAuthor.PATIENT_MESSAGE,
        session_id=session_id,
    ) as envelope:
        envelope.result = await _turn(
            envelope.session,
            writer=envelope.writer,
            callbacks=envelope.callbacks,
            user=envelope.user,
            message=message,
            conversation_id=envelope.conversation_id,
            provider=provider,
        )
        # Inside the envelope, so the note is part of the reply the trace's
        # outbound event records. A sentence added after the turn was written
        # down is a sentence the trace does not vouch for.
        note = _one_verb_note(envelope.session, envelope.result, message=message)
        if note:
            envelope.result.reply = f"{envelope.result.reply}\n\n{note}".strip()
    return envelope.result


async def apply_patient_action(
    user: User,
    action: str,
    session_id: str,
    *,
    provider: str | None = None,
) -> TurnResult:
    """A ✅ Confirm or ❌ Decline click. **Zero interpretation.**

    The buttons render with the *first* proposal, not as a fallback after the
    patient has already been misunderstood once — so this is the ordinary path
    to a booking rather than the exceptional one. A click carries no free text,
    so nothing here reads language: the action is matched against a closed set
    and applied. No model is consulted about what the patient meant, because
    there is nothing to be uncertain about.

    A stale click is a no-op, not an error. The motivating trace is the
    double-clicked Confirm: the second request finds the proposal already
    committed and must say so calmly.
    """
    chosen = str(action or "").strip().lower()

    async with _turn_envelope(
        user,
        content=chosen,
        author=TraceAuthor.PATIENT_ACTION,
        session_id=session_id,
    ) as envelope:
        session, writer, user_row = envelope.session, envelope.writer, envelope.user
        base = dict(turn_id=writer.turn_id, session_id=envelope.conversation_id)

        # A button press has no free text, so there is nothing for the safety
        # screen to read. Recorded rather than silently absent: "did not run"
        # and "was not applicable" must stay different facts.
        writer.guard_verdict(
            "safety_keyword_screen",
            passed=True,
            detail={"skipped": "typed_action", "source": "keyword"},
        )

        if chosen not in ("confirm", "decline"):
            raise ValidationFailed(
                f"{action!r} is not a patient action. Use 'confirm' or 'decline'."
            )

        profile = _patient_profile(session, user_row)
        run = active_run(session, profile.id)
        if run is None or run.status is not WorkflowStatus.PENDING_CONFIRMATION:
            writer.guard_verdict(
                "typed_action_applicable",
                passed=False,
                detail={"action": chosen, "status": run.status.value if run else None},
            )
            write_audit(
                session,
                action="patient_action_stale",
                entity_type="workflow_run",
                entity_id=run.id if run else None,
                actor=user_row,
                metadata={"action": chosen},
            )
            envelope.result = TurnResult(
                reply=NOTHING_TO_CONFIRM_REPLY,
                author=TraceAuthor.TEMPLATE,
                run_id=run.id if run else None,
                status=run.status.value if run else None,
                **base,
            )
            return envelope.result

        writer.bind_run(run.id)
        writer.guard_verdict(
            "typed_action_applicable", passed=True, detail={"action": chosen}
        )

        if chosen == "decline":
            envelope.result = _decline_proposal(
                session, run=run, writer=writer, user=user_row,
                trigger="patient_declined_action", base=base,
            )
            return envelope.result

        belt = Toolbelt(
            session, user=user_row, patient_id=profile.id, writer=writer, run=run
        )
        envelope.result = await _commit_proposal(
            session,
            run=run,
            belt=belt,
            writer=writer,
            callbacks=envelope.callbacks,
            user=user_row,
            conversation_id=envelope.conversation_id,
            provider=provider,
            base=base,
        )
    return envelope.result


async def _turn(
    session: Session,
    *,
    writer: TraceWriter,
    callbacks: TurnCallbacks,
    user: User,
    message: str,
    conversation_id: str,
    provider: str | None,
) -> TurnResult:
    settings = get_settings()
    profile = _patient_profile(session, user)
    run = active_run(session, profile.id)
    if run is not None:
        writer.bind_run(run.id)

    belt = Toolbelt(
        session, user=user, patient_id=profile.id, writer=writer, run=run, message=message
    )
    base = dict(turn_id=writer.turn_id, session_id=conversation_id)

    # --- 0. the safety screen: first, always, whatever the run's state ---
    verdict = await _screen(
        message,
        callbacks=callbacks,
        writer=writer,
        user_id=str(user.id),
        provider=provider,
    )
    if callbacks.budget_exhausted:
        return _budget_failure(
            session, run=run, writer=writer, user=user, base=base, stage="safety_screen"
        )
    if verdict.fired:
        escalated, reply, author = escalate(
            session,
            verdict=verdict,
            user=user,
            patient_id=profile.id,
            message=message,
            writer=writer,
            session_id=conversation_id,
            run=run,
        )
        return TurnResult(
            reply=reply,
            author=author,
            run_id=escalated.id,
            status=escalated.status.value,
            **base,
        )

    # --- 0b. a listing question is answered from the rows, before planning ---
    # "What do I have?" has no consequence to weigh, no slot to hold and no
    # state to move, so routing it through plan-then-delegate makes the answer
    # depend on whether the Coordinator felt like emitting a plan for that
    # particular sentence. Live, three phrasings dead-ended in one session
    # while a fourth worked.
    #
    # Before the Coordinator rather than after it, and that ordering is the
    # whole point: a *fallback* for when planning fails would still be a
    # lottery, because a wrong plan is as likely as no plan — the mock happily
    # plans "show my appointments" as a booking and asks which department. This
    # runs after the safety screen, which is first always, and only where there
    # is no active run: with one, the same question arrives as a side question
    # and is answered on that path, with the mapping's record intact.
    #
    # Documents are the exception, and the reason is worth stating: the
    # documents *step* verifies one pending upload per turn, so pre-empting it
    # would answer the question and quietly stop the work the question used to
    # trigger. That answer is templated further down instead — the specialist
    # still runs, and only what the patient is told is code's, which is the
    # same division the receipt already uses.
    if run is None:
        query = detect_query(message)
        if query is QueryKind.DOCUMENTS:
            query = None
        if query is not None:
            writer.guard_verdict(
                "listing_query", passed=True, detail={"kind": query.value}
            )
            write_audit(
                session,
                action="query_answered",
                entity_type="workflow_run",
                actor=user,
                metadata={"kind": query.value},
            )
            return TurnResult(
                reply=answer_query(
                    session, patient_id=profile.id, kind=query, message=message
                ),
                author=TraceAuthor.TEMPLATE,
                **base,
            )

    # --- 0c. an exact yes to an offer the last turn made -------------------
    # A fresh turn, so nothing else in this system is waiting for a "yes": the
    # confirmation reader needs a proposal and there is no run at all. What there
    # may be is a request this system offered to pick up one message ago, and the
    # patient's own exact word is the only thing that may start it.
    #
    # Placed here rather than at the scope gate because a bare "yes" reaches
    # neither branch of that gate usefully — it names no subject, so it is
    # refused as off-topic, which is the right answer when nothing was offered
    # and the wrong one when something was.
    if run is None and read_confirmation(message) is ConfirmationAnswer.CONFIRM:
        restarted = await _restart_remembered(
            session,
            writer=writer,
            callbacks=callbacks,
            belt=belt,
            user=user,
            patient_id=profile.id,
            message=message,
            conversation_id=conversation_id,
            provider=provider,
            settings=settings,
            base=base,
        )
        if restarted is not None:
            return restarted

    # --- 1. a choice among times already shown, read in code ---------------
    # The state one step earlier had no reader at all: a run holding nothing but
    # a list of times it had shown could not hear an answer to that list, so
    # "3pm will work for me" went to the Coordinator, came back a new request,
    # was correctly refused a supersede, and was answered with the same list
    # again. Seven times in a row, live.
    #
    # And the state *with* a slot held could not hear one either, for a
    # different reason: the exact-token reader has no selection verdict, so
    # "option 3" fell through to the model, which called it conflicting, and "I
    # want the 4pm appointment on august 3" came back a **decline** — a choice
    # among the times on screen recorded as a refusal of them. So this runs
    # ahead of the tokens rather than after them. Nothing about the tokens
    # changes: "yes" and "no" name no time and no position, so they pass
    # straight through to the reader below.
    # The appointment-side twin runs first, and only one of them can apply: a
    # numbered list of appointments is outstanding exactly while no proposal
    # exists, and a list of *times* only ever exists once one does. "2" against
    # a choice of appointments and "2" against a choice of slots are the same
    # word answering two different questions, and the run's own state is what
    # says which was asked.
    if run is not None and run.status is WorkflowStatus.IN_PROGRESS:
        chosen = await _settle_choice(
            session,
            run=run,
            belt=belt,
            writer=writer,
            callbacks=callbacks,
            user=user,
            message=message,
            provider=provider,
            settings=settings,
            base=base,
        )
        if chosen is not None:
            return chosen

    if run is not None and run.status in (
        WorkflowStatus.IN_PROGRESS,
        WorkflowStatus.PENDING_CONFIRMATION,
    ):
        selected = _settle_selection(
            session,
            run=run,
            belt=belt,
            writer=writer,
            user=user,
            message=message,
            base=base,
        )
        if selected is not None:
            return selected

    # --- 1b. a pending confirmation is read in code, before any model call ---
    if run is not None and run.status is WorkflowStatus.PENDING_CONFIRMATION:
        answer = read_confirmation(message)
        writer.guard_verdict(
            "confirmation_reader",
            passed=answer is not ConfirmationAnswer.UNREAD,
            detail={"answer": answer.value},
        )
        if answer is ConfirmationAnswer.CONFIRM:
            return await _commit_proposal(
                session,
                run=run,
                belt=belt,
                writer=writer,
                callbacks=callbacks,
                user=user,
                conversation_id=conversation_id,
                provider=provider,
                base=base,
            )
        if answer is ConfirmationAnswer.DECLINE:
            return _decline_proposal(
                session, run=run, writer=writer, user=user,
                trigger="patient_declined", base=base,
            )
        # UNREAD falls through: the model may re-ask or decline, never confirm.

    # --- 2. the Coordinator: classify against a live run, or plan a new one ---
    # One service for the whole turn — each instance builds its own engine, and
    # the conversation is the only thing that needs a durable one.
    conversation = memory.conversation_service()
    existing = await conversation.get_session(
        app_name=memory.APP_NAME, user_id=str(user.id), session_id=conversation_id
    )
    # With no live run the Coordinator is a planner, and a planner is answering
    # a question about *this* message: what does it need, given who is asking.
    # The transcript cannot help with that and demonstrably hurt — see
    # ``plan_context_only``. With a run, the same agent is a classifier whose
    # entire question is how this message relates to what came before, so the
    # window stays exactly as it was.
    callbacks.plan_context_only = run is None
    try:
        coordinator_reply = await run_agent(
            coordinator.build_agent(belt, callbacks, provider=provider),
            task_text=message,
            callbacks=callbacks,
            session_service=conversation,
            session_id=conversation_id,
            user_id=str(user.id),
            create=existing is None,
        )
    finally:
        # Specialists run later in this same turn against their own throwaway
        # sessions; leaving the flag set would be harmless there and wrong to
        # rely on, and the turn envelope is the only place that knows.
        callbacks.plan_context_only = False

    if callbacks.budget_exhausted:
        return _budget_failure(
            session, run=run, writer=writer, user=user, base=base, stage="coordinator"
        )

    if run is not None:
        return await _continue_run(
            session,
            run=run,
            belt=belt,
            writer=writer,
            callbacks=callbacks,
            user=user,
            profile=profile,
            message=message,
            coordinator_reply=coordinator_reply,
            conversation_id=conversation_id,
            provider=provider,
            base=base,
        )

    # --- 3. the scope gate ------------------------------------------------
    # The Coordinator classifies three ways: supported intent, unsafe, or
    # off-topic. Safety has already had its turn, so a message that produced no
    # plan is off-topic — and only a supported intent may spawn a workflow.
    plan = _corrected_change_plan(
        session,
        belt.proposals.plan,
        message=message,
        patient_id=profile.id,
        writer=writer,
    )
    if plan is None:
        plan = _plan_floor(
            session, message=message, belt=belt, writer=writer, user=user
        )
    # A plan made entirely of subjectless steps has to be earned by the message.
    # Round 9 closed this for ``["follow_up"]`` alone, and the hole was one door
    # wide: live, "now do the other thing I asked" was planned as
    # ``["documents", "follow_up"]`` — two valid steps, so the gate passed with
    # ``domain_subject: false`` — and the documents agent ran departmentless,
    # shipping its own internal refusal ("no department has been decided") to the
    # patient as an answer, followed by a dump of their open tasks. The recall
    # memory that exists for exactly that sentence never fired, because it is
    # consulted only when the gate *refuses* and the model had planned around it.
    unearned = (
        plan is not None
        and set(plan) <= _SUBJECTLESS_STEPS
        and not _earns_plan(message)
    )

    # Before any of the three refusals below, and only for a message that points
    # backwards: the patient may be asking for something they have already asked
    # for once. This changes what the refusal *offers* and nothing else — no run
    # is created here, and the offer is answerable only by an exact word.
    if plan is None or unearned:
        offered = _offer_remembered(
            session,
            writer=writer,
            user=user,
            patient_id=profile.id,
            message=message,
            conversation_id=conversation_id,
            base=base,
        )
        if offered is not None:
            return offered

    if unearned:
        # A plan of subjectless steps asserts nothing about the message. Every
        # other step names something the request had to be about; these do not,
        # so the gate — which asks for a domain subject only when there is *no*
        # plan — had nothing to disagree with, and three off-topic questions
        # became three completed runs that listed the patient's reminders back
        # at them.
        writer.guard_verdict(
            "plan_earned",
            passed=False,
            detail={
                "steps": [step.value for step in plan],
                "problem": "no appointment verb, and the message names no subject",
            },
        )
        write_audit(
            session,
            action="scope_gate_refused",
            entity_type="workflow_run",
            actor=user,
            metadata={"reason": "subjectless plan not earned by the message"},
        )
        return TurnResult(
            reply=UNSUPPORTED_TOPIC_REPLY, author=TraceAuthor.GUARD, **base
        )
    names_a_subject = mentions_domain_subject(message)
    writer.guard_verdict(
        "scope_gate",
        passed=plan is not None,
        detail={
            "steps": [step.value for step in plan] if plan else [],
            "domain_subject": names_a_subject,
        },
    )
    if plan is None and names_a_subject:
        # No plan, but the message names an appointment, a document or a
        # reminder — so it is not out of scope, and refusing it as though it
        # were tells the patient to go away and rephrase. Live, "can you tell
        # me my appointments" was refused twice while a different wording of
        # the same question worked.
        #
        # Code cannot invent the plan the Coordinator did not produce — except
        # where the patient named the verb outright and has something to apply
        # it to, which `_corrected_change_plan` has already handled above. Every
        # other unplanned message is genuinely undecided, and picking steps for
        # it here would put the deterministic layer in the planning bin. So the
        # honest answer is the one that asks for more, and the difference
        # between that and ``SCOPE_REPLY`` is the difference between "tell me
        # more" and "I don't do that".
        write_audit(
            session,
            action="scope_gate_vetoed",
            entity_type="workflow_run",
            actor=user,
            metadata={"reason": "message names a subject this system administers"},
        )
        # Asking again is only an answer the first couple of times. Live, the
        # same clarify met the same rephrasing until the patient stopped
        # trying — the reply was polite, the run was correct, and nothing was
        # happening. A third time, a person looks at it instead.
        prior = _consecutive_clarifies(
            session,
            session_id=base.get("session_id") or "",
            turn_id=writer.turn_id,
        )
        if prior >= MAX_CONSECUTIVE_CLARIFIES:
            return _escalate_unplannable(
                session, writer=writer, user=user, message=message, base=base
            )
        writer.guard_verdict(
            "clarify_repeat", passed=True, detail={"consecutive": prior + 1}
        )
        return TurnResult(reply=NO_PLAN_REPLY, author=TraceAuthor.TEMPLATE, **base)

    if plan is None:
        # No run, no tools fired, no escalation. Off-topic is noise, not a
        # human-review case, and a queue full of noise is a queue nobody reads.
        #
        # The refusal is a **template**, not the model's words. It must read
        # identically under mock and under a live provider, and it must state
        # nothing about the patient — which is exactly what freeform prose
        # cannot promise.
        write_audit(
            session,
            action="scope_gate_refused",
            entity_type="workflow_run",
            actor=user,
            metadata={"reason": "no supported administrative intent"},
        )
        return TurnResult(reply=SCOPE_REPLY, author=TraceAuthor.GUARD, **base)

    run = create_run(
        session,
        patient_id=profile.id,
        status=WorkflowStatus.IN_PROGRESS,
        trigger="intent_accepted",
        writer=writer,
        request_text=message,
        plan=[step.value for step in plan],
        session_id=conversation_id,
        actor=user,
    )
    belt.run = run
    return await _execute_plan(
        session,
        run=run,
        belt=belt,
        writer=writer,
        callbacks=callbacks,
        user=user,
        message=message,
        fallback_reply=coordinator_reply,
        provider=provider,
        settings=settings,
        base=base,
    )


async def _continue_run(
    session: Session,
    *,
    run: WorkflowRun,
    belt: Toolbelt,
    writer: TraceWriter,
    callbacks: TurnCallbacks,
    user: User,
    profile: PatientProfile,
    message: str,
    coordinator_reply: str,
    conversation_id: str,
    provider: str | None,
    base: dict,
) -> TurnResult:
    """Apply the message→run class, then carry on where it leaves the run."""
    settings = get_settings()
    #: Read before the mapping, which may move the run off this state.
    awaiting_confirmation = run.status is WorkflowStatus.PENDING_CONFIRMATION
    verdict = belt.proposals.class_verdict
    if verdict is None:
        # The Coordinator did not classify. Default to the class that changes
        # nothing: a side question is read-only, spawns nothing, supersedes
        # nothing, and cannot contaminate the run's request text.
        verdict = validate_class(
            MessageClass.SIDE_QUESTION.value,
            run=run,
            incoming_steps=None,
            writer=writer,
        )
        writer.guard_verdict(
            "message_class_defaulted", passed=False, detail={"applied": "side_question"}
        )

    outcome = apply_consequence(
        session,
        run,
        verdict,
        writer=writer,
        message=message,
        incoming_steps=belt.proposals.incoming_steps,
        actor=user,
    )

    # A classification code overruled takes its prose with it. The model wrote
    # its sentence *believing* the verdict code has just rejected, so the two
    # cannot both be right — and shipping the sentence anyway is how a patient
    # who chose option 3 was told "it seems you've decided to withdraw your
    # request again", above a re-ask offering them the time they had just
    # picked. The action was blocked; only the words got through.
    #
    # One rule, one place, at the point where the reply is still the model's to
    # lose. Everything downstream — the confirmation re-ask, the answer-and-stay
    # branch, `_execute_plan`'s fallback — reads this same variable, and the
    # specialists' own replies are untouched: they run *after* the correction
    # and speak about the work, not about the class.
    if outcome.overruled:
        writer.guard_verdict(
            "overruled_prose",
            passed=False,
            detail={
                "class": outcome.message_class.value,
                "applied": outcome.consequence.value,
                "reply": coordinator_reply[:200],
            },
        )
        coordinator_reply = ""

    # The scope gate runs inside the mapping too, not only at run creation.
    # An off-topic message that fell through to continuation would be appended
    # to the run's stored request text, contaminating what routing and slot
    # matching read later.
    writer.guard_verdict(
        "scope_gate",
        passed=outcome.consequence is not Consequence.SCOPE_REPLY,
        detail={"class": outcome.message_class.value},
    )

    # A question that names a day is answered with that day's times, in either
    # state. Live, at `pending_confirmation`: "do you have any appointments at
    # the end of the same week like thursday or friday preferably in the
    # afternoon?" was classified a side question, and the answer half of
    # answer-and-stay then had nothing to render — because at this state nothing
    # searches. The model holds no `list_other_slots` here, so it wrote prose
    # about availability instead, which the grounding guard rightly threw away
    # for having no search behind it, and what reached the patient was the bare
    # re-ask with a stall counted against it. The identical sentence one turn
    # later, at `in_progress`, produced a real Thursday list.
    #
    # Keyed on the *outcome* rather than on the class, and that is the whole
    # difference between this working and not. The live model called it a side
    # question; the mock calls it a continuation whose confirmation verdict is
    # `non_answer`. Two classes, one destination — the re-ask — so the re-ask is
    # what this stands in front of. `scope_reply` is deliberately not in the
    # list: "what's the weather like today" names a timing word, and answering
    # an off-topic message with a slot list is the failure this would otherwise
    # introduce while fixing another.
    #
    # Code runs the search itself, exactly as it does for a refused supersede,
    # with the patient's own words as the phrase. It refuses itself when the run
    # has no department, leaving `answered_with_slots` false — which is what
    # keeps "answered" and "tried" different facts below.
    answered_timing = False
    if (
        awaiting_confirmation
        and run.status is WorkflowStatus.PENDING_CONFIRMATION
        and outcome.consequence
        in (Consequence.ANSWER_AND_STAY, Consequence.FEED_RUN, Consequence.APPEND_STEP)
        and belt.proposals.confirmation_verdict != "decline"
        and not belt.proposals.reproposed
        and not belt.proposals.answered_with_slots
        and names_timing(message)
    ):
        belt.answer_with_other_slots(message)
        answered_timing = belt.proposals.answered_with_slots
        writer.guard_verdict(
            "timing_question", passed=answered_timing, detail={"message": message[:200]}
        )

    # Stall containment, counted **independently of the class**. Deliberately
    # so: a message that leaves the run waiting is a non-answer whatever it was
    # classified as, and tying the counter to one class would let a
    # misclassification — an "hmm, maybe" read as off-topic — quietly make the
    # re-ask loop unbounded again. The bound must not be reachable only through
    # the paths that were classified correctly.
    #
    # A timing question that got its answer is the one exception, and it is not
    # a hole in the rule: the counter bounds a re-ask loop, and a turn that
    # rendered times is not a re-ask. The message that *was* counted here — the
    # Thursday-or-Friday question above — was a patient being told nothing and
    # then charged for it.
    #
    # Read from ``answered_with_slots``, not from ``answered_timing``. The
    # second means "*code* ran the search", and the two were the same thing
    # only for as long as code was the only thing that could render times at
    # this state. Round 9 gave the model ``propose_search_window``, which sets
    # ``answered_with_slots`` and therefore *skips* the code-driven branch
    # above — leaving ``answered_timing`` false on a turn that had just shown
    # the patient a list. They asked about timing, were answered, and were
    # charged a non-answer for it. The sentence this comment already made is
    # the correct rule; the condition simply did not implement it.
    if (
        awaiting_confirmation
        and run.status is WorkflowStatus.PENDING_CONFIRMATION
        and not belt.proposals.answered_with_slots
    ):
        run.non_answer_count += 1
        session.flush()

    if outcome.consequence is Consequence.WITHDRAW:
        return TurnResult(
            reply=WITHDRAWN_REPLY,
            author=TraceAuthor.TEMPLATE,
            run_id=run.id,
            status=run.status.value,
            message_class=outcome.message_class,
            **base,
        )

    if outcome.consequence is Consequence.STATUS_REPLY:
        # A supersede the mapping refused. The patient has to be told *why*
        # nothing appeared to happen, or the silence reads as the message
        # having been ignored — and a templated answer is the only kind that
        # can promise it states nothing about the review's outcome.
        return TurnResult(
            reply=AWAITING_REVIEW_REPLY,
            author=TraceAuthor.TEMPLATE,
            run_id=run.id,
            status=run.status.value,
            message_class=outcome.message_class,
            **base,
        )

    if outcome.consequence is Consequence.SCOPE_REPLY:
        return TurnResult(
            reply=SCOPE_REPLY,
            author=TraceAuthor.GUARD,
            run_id=run.id,
            status=run.status.value,
            message_class=outcome.message_class,
            **base,
        )

    # Everything else, while a person holds the run. Placed after the three
    # branches above rather than before them so each keeps its own answer: a
    # withdrawal has already moved the run off this state, a refused supersede
    # is told its position by the branch that refused it, and an off-topic
    # message still gets the off-topic reply — walling that one would answer
    # "who won the fifa final" with a note about a booking.
    if run.status is WorkflowStatus.PENDING_REVIEW:
        return _review_wall(
            session,
            run=run,
            writer=writer,
            message=message,
            outcome=outcome,
            base=base,
        )

    # A supersede the mapping refused because the message was refining the run
    # rather than replacing it. The model called it a new request, so it never
    # asked for the times — code runs the search itself, with the patient's own
    # words as the phrase, and `resolve_date` turns them into a window as it
    # does everywhere else. Live, this message destroyed an Orthopedics booking
    # in progress; the refusal alone would have left it un-destroyed and
    # unanswered, which is half a fix.
    #
    # If nothing comes back — no department decided yet, nothing free in that
    # window — this falls through deliberately. The re-ask below states what is
    # being held, and at `in_progress` the plan carries on, so the turn ends on
    # something the patient can act on either way.
    if outcome.consequence is Consequence.REFINE:
        belt.answer_with_other_slots(message)
        answered = _answer_while_holding(
            session, run=run, belt=belt, outcome=outcome, base=base
        )
        if answered is not None:
            return answered

        # Nothing came back, and on a run that has not decided which appointment
        # it is about, nothing could: the search needs a department and the
        # department lives on the appointment nobody has picked yet. Falling
        # through from here re-dispatched the specialist, which asked "which
        # one?" again, failed to propose, and spent the run's last re-plan —
        # the patient's answer to a question, answered with "I couldn't complete
        # this request". Ask again instead; a re-ask is not progress, but it is
        # a turn the patient can act on, and the list it draws is the one the
        # reader above can read.
        if not belt.proposals.answered_with_slots:
            choice = _choice_reask(session, run)
            if choice:
                return TurnResult(
                    reply=choice,
                    author=TraceAuthor.TEMPLATE,
                    run_id=run.id,
                    status=run.status.value,
                    message_class=outcome.message_class,
                    **base,
                )

    # Text the exact tokens could not read, on a run that is still waiting.
    # This is the third and last step of the reading order, and the only one
    # the model takes part in.
    if awaiting_confirmation and run.status is WorkflowStatus.PENDING_CONFIRMATION:
        answered = _answer_while_holding(
            session, run=run, belt=belt, outcome=outcome, base=base
        )
        if answered is not None:
            return answered
        return _settle_confirmation(
            session,
            run=run,
            belt=belt,
            writer=writer,
            user=user,
            message=message,
            coordinator_reply=coordinator_reply,
            outcome=outcome,
            base=base,
        )

    if outcome.consequence is Consequence.ANSWER_AND_STAY:
        # An availability question is answered with availability, in whatever
        # state it was asked. Only the confirmation state had this path before,
        # so the same question at `in_progress` got prose and no times.
        answered = _answer_while_holding(
            session, run=run, belt=belt, outcome=outcome, base=base
        )
        if answered is not None:
            return answered

        # And a listing question is answered with the listing. Read-only, so
        # it disturbs a live run exactly as little as any other side question.
        query = detect_query(message)
        if query is not None:
            return TurnResult(
                reply=answer_query(
                    session, patient_id=run.patient_id, kind=query, message=message
                ),
                author=TraceAuthor.TEMPLATE,
                run_id=run.id,
                status=run.status.value,
                message_class=outcome.message_class,
                **base,
            )

        text, from_model = _guarded(
            coordinator_reply,
            writer=writer,
            fallback=NO_PLAN_REPLY,
            searched_slots=belt.proposals.searched_slots,
        )
        return TurnResult(
            reply=text,
            author=TraceAuthor.LLM if from_model else TraceAuthor.TEMPLATE,
            run_id=run.id,
            status=run.status.value,
            message_class=outcome.message_class,
            **base,
        )

    if outcome.consequence is Consequence.SUPERSEDE:
        # The replacement run gets the same plan check a first-message run gets.
        # It did not, and the gap is a whole live failure: "lets reschedule my
        # Ophthalmology appointment", arriving over a live run, superseded it —
        # and the replacement's plan was built from `incoming_steps` verbatim,
        # which gpt-4o-mini returned as `[route, book, ...]`. A plainly stated
        # reschedule became a booking run, and the reschedule the Appointment
        # agent then proposed committed under a `book` step it could never
        # settle. A correction that runs on one door into `create_run` and not
        # the other is not a correction; it is a coin flip on which door the
        # message came through.
        #
        # Invisible to the mock, which proposes `[reschedule, follow_up]` here
        # and always did — so this guard has to be falsified against steps a
        # model actually got wrong, never against the understudy's.
        steps = belt.proposals.incoming_steps or [PlanStep.ROUTE]
        corrected = _corrected_change_plan(
            session,
            validate_plan([s.value for s in steps]),
            message=message,
            patient_id=profile.id,
            writer=writer,
        )
        replacement = create_run(
            session,
            patient_id=profile.id,
            status=WorkflowStatus.IN_PROGRESS,
            trigger="superseded_previous_request",
            writer=writer,
            request_text=message,
            plan=[step.value for step in corrected],
            session_id=conversation_id,
            actor=user,
        )
        belt.run = replacement
        result = await _execute_plan(
            session,
            run=replacement,
            belt=belt,
            writer=writer,
            callbacks=callbacks,
            user=user,
            message=message,
            fallback_reply=coordinator_reply,
            provider=provider,
            settings=settings,
            base=base,
        )
        return TurnResult(
            reply=(
                "I've closed your earlier request and started this one instead. "
                + result.reply
            ),
            author=result.author,
            run_id=result.run_id,
            status=result.status,
            message_class=outcome.message_class,
            plan=result.plan,
            steps_run=result.steps_run,
            budget_exhausted=result.budget_exhausted,
            **base,
        )

    # FEED_RUN and APPEND_STEP both carry on with the run they already had.
    result = await _execute_plan(
        session,
        run=run,
        belt=belt,
        writer=writer,
        callbacks=callbacks,
        user=user,
        message=message,
        fallback_reply=coordinator_reply,
        provider=provider,
        settings=settings,
        base=base,
    )
    result.message_class = outcome.message_class
    return result


async def _execute_plan(
    session: Session,
    *,
    run: WorkflowRun,
    belt: Toolbelt,
    writer: TraceWriter,
    callbacks: TurnCallbacks,
    user: User,
    message: str,
    fallback_reply: str,
    provider: str | None,
    settings,
    base: dict,
) -> TurnResult:
    """Dispatch planned steps to specialists until the run halts or finishes.

    Halting is normal and is not failure: a proposal waiting on the patient
    stops the plan mid-way on purpose, and the run resumes when they answer.
    """
    steps_run: list[str] = []
    #: Set when code replaces what the specialists said. The author on the
    #: outbound event has to name whoever actually wrote the words.
    code_authored = False
    #: A documents question's templated answer, built while the step runs and
    #: applied after the loop. Applied at the end rather than mid-loop because
    #: replacing `said` while later steps are still to speak is how a guard over
    #: a joined reply drops the good sentence with the bad one.
    documents_answer = ""

    # Each specialist's reply is kept, not overwritten. The booking receipt and
    # the missing-documents list are both things the patient needs; letting the
    # last step win would silently drop whichever mattered most.
    said: list[str] = []

    while True:
        step = next_step(run)
        if step is None:
            break

        # A verb that has committed is done, and re-entering it is not a retry:
        # it is a *second proposal* for a request the patient has already
        # settled. Live, the reschedule step ran again immediately after its own
        # commit, called `find_slots_for_reschedule` with no window, proposed the
        # earliest free slot, and put the run back into `pending_confirmation` —
        # so the patient's second Confirm click moved a 4 August 11:00
        # appointment they had chosen to a 28 July 09:00 one they were never
        # shown. Two `appointment_rescheduled` audits, eighteen seconds apart.
        #
        # `_settle_step` has always known this, but it runs *after* the
        # specialist: by the time it read `committed_action` the proposal had
        # already been made and accepted. The check belongs here, before the
        # dispatch, where re-entry is still preventable rather than merely
        # detectable.
        #
        # **Any** committed verb settles the step, not only a matching one, and
        # that difference is the whole of a live defect. A run's plan and the
        # action it commits can name different verbs: run 6 was created by the
        # supersede path with the plan `[route, book, ...]`, and the Appointment
        # agent — correctly reading "lets reschedule my Ophthalmology
        # appointment" — proposed a *reschedule* under the `book` step. The
        # patient typed "yes", `_commit_proposal` moved the appointment and
        # transitioned the run back to `in_progress`, and then this loop found
        # `book` incomplete and `committed_action == "reschedule"`, saw no
        # match, and dispatched the specialist again. It proposed a second time,
        # the run went back to `pending_confirmation`, and the receipt for the
        # commit that *had* happened was rendered beside a fresh proposal card.
        # The Decline pressed next was applicable — there really was an open
        # proposal — and answered "nothing has been booked" about an
        # appointment that had already moved.
        #
        # One run commits at most one appointment action: `validate_plan`
        # refuses a plan naming two verbs, precisely because one
        # `ProposedAction` can only dispatch one. So a committed verb settles
        # this run's single appointment step whatever the plan happens to call
        # it, and both names are traced so a mismatch stays visible rather than
        # becoming invisible by being handled.
        committed = (run.state or {}).get("committed_action")
        if step in APPOINTMENT_VERBS and committed:
            writer.validation(
                "step_already_committed",
                accepted=True,
                detail={"step": step.value, "committed": committed, "run_id": run.id},
            )
            # Still recorded as run. The plan advanced through it this turn —
            # what changed is that code settled it instead of a specialist, and
            # a turn that reports having skipped its own booking step describes
            # something that did not happen.
            steps_run.append(step.value)
            advance_plan(run, step)
            continue

        specialist = SPECIALIST_FOR_STEP[step.value]
        task = _task_for(step, run, message=message, extra={})
        step_reply = await run_agent(
            specialist.build_agent(belt, callbacks, provider=provider),
            task_text=task,
            callbacks=callbacks,
            session_service=memory.task_service(),
            session_id=f"{writer.turn_id}-{step.value}",
            user_id=str(user.id),
            create=True,
        )
        steps_run.append(step.value)
        # Guarded per specialist rather than over the join: one agent's framing
        # ("let me find you a time") must not take the next agent's proposal
        # down with it. Dropping the whole reply left a turn that had proposed
        # a time saying only "could you tell me more about what you need".
        kept, _ = _guarded(
            step_reply, writer=writer, searched_slots=belt.proposals.searched_slots
        )
        if kept:
            said.append(kept)

        # The "what is still needed" half of a documents answer is code's, from
        # the open task rows. A listing run that carries no department has
        # nothing to diff against, and the model filled that gap with "no
        # documents required" while an open task for a missing MRI report sat
        # in the database — contradicted one turn later by the follow-up
        # screen. Nothing to compare is not the same fact as nothing required.
        if step is PlanStep.DOCUMENTS:
            # A documents *question* gets the templated answer, replacing the
            # model's inventory prose entirely. The step still ran — the
            # verification it performs is why this is a replacement here rather
            # than a pre-empt higher up — and only the saying is code's.
            #
            # Live, the model's version dumped a field-by-field inventory and
            # closed with "since no department has been decided for this
            # request, there are no additional documents required", to a patient
            # who had named ENT in the question and had an ENT appointment on
            # the books. Both halves are now read: the department from the
            # message or the appointment, the requirements from the diff.
            if detect_query(message) is QueryKind.DOCUMENTS:
                documents_answer = answer_query(
                    session,
                    patient_id=run.patient_id,
                    kind=QueryKind.DOCUMENTS,
                    message=message,
                )
            outstanding = render_outstanding(session, patient_id=run.patient_id)
            if outstanding:
                said.append(outstanding)

        if callbacks.budget_exhausted:
            return _fail_run(
                session, run=run, writer=writer, user=user,
                reason="tool_iteration_budget", base=base,
                plan=list(run.plan or []), steps_run=steps_run,
            )

        halted, completed = _settle_step(
            session, step=step, run=run, belt=belt, writer=writer, user=user
        )
        if completed:
            advance_plan(run, step)
        if halted:
            break
        if not completed:
            # Asking which appointment is not getting nowhere. The step is
            # waiting on the patient, exactly as a proposal waits on a "yes",
            # and the turn ends either way — so charging it a re-plan spends
            # the run's only retry on asking a legitimate question, and every
            # change run that has to ask then has **none left for the answer**.
            # Live: the patient answered "1. and please make it sometime the
            # week after", the reader took the 1, and the specialist read "the
            # week after" as unparseable and stopped without searching. One
            # fumble, no budget, and the reply to a correct answer was "I'm
            # sorry — I couldn't complete this request."
            #
            # The bound is unchanged where it bounds something: a re-plan loop
            # is automated and this is not, because nothing continues until the
            # patient sends another message.
            if step in APPOINTMENT_VERBS and _awaiting_choice(run):
                writer.guard_verdict(
                    "awaiting_choice",
                    passed=True,
                    detail={"step": step.value, "run_id": run.id},
                )
                break

            # The step ran and got nowhere. Ask for a new plan once — the
            # budget is what stops "widen the search and try again" becoming
            # a loop of perfectly successful calls.
            try:
                record_replan(run, max_replans=settings.max_replans_per_run)
            except BudgetExceeded:
                return _fail_run(
                    session, run=run, writer=writer, user=user,
                    reason="replan_budget", base=base,
                    plan=list(run.plan or []), steps_run=steps_run,
                )
            write_audit(
                session,
                action="workflow_replanned",
                entity_type="workflow_run",
                entity_id=run.id,
                actor=user,
                metadata={"step": step.value, "attempt": run.replan_count},
            )
            break

    if is_plan_complete(run) and run.status is WorkflowStatus.IN_PROGRESS:
        transition(
            session,
            run,
            to=WorkflowStatus.COMPLETED,
            trigger="plan_complete",
            writer=writer,
            actor=user,
        )

    # A proposal now goes out with the shortlist it was drawn from. The typed
    # proposal is still exactly one slot and the state machine has not moved:
    # this only tells the patient that there were others and that asking for
    # one is a thing they may do. Appended by code, from the slot payload, so
    # it says the same thing under mock and under a live provider — which the
    # specialist's own prose cannot promise.
    if (
        run.status is WorkflowStatus.PENDING_CONFIRMATION
        and run.proposed_action is ProposedAction.BOOK
        and belt.proposals.offered_slots
    ):
        shortlist = render_proposal(session, run, belt.proposals.offered_slots)
        if shortlist:
            said.append(shortlist)

    # A reschedule proposal holds two times of the same shape — the one the
    # patient has and the one being offered — and the model welded them: "I
    # found your appointment ... at 9:00 AM. Would you like me to reschedule
    # it?", where 9:00 was the *new* slot and the appointment was at 10:00. The
    # sentence named no new time at all, so it read as a request to approve
    # moving an appointment to the hour it already occupied.
    #
    # Replaced rather than appended, unlike the booking shortlist above: there
    # the model's sentence and the code's list say different things, and here
    # they say the same thing twice with different numbers in it.
    if run.status is WorkflowStatus.PENDING_CONFIRMATION and run.proposed_action in (
        ProposedAction.RESCHEDULE,
        ProposedAction.CANCEL,
    ):
        change = render_change_proposal(session, run)
        if change:
            said = [change]
            # And the trace has to say who wrote it. Code-authored replies are
            # not invisible: leaving the author at `llm` because a specialist
            # happened to speak earlier in the turn would vouch for a sentence
            # the model did not produce.
            code_authored = True

    # Nothing proposed, and more than one appointment it could have been: the
    # specialist is asking which one. That listing is code's, because the
    # patient's answer to it — "2" — is only meaningful against a list somebody
    # recorded. Live the model wrote its own numbering, in prose, so when the
    # patient answered "2" there was nothing anywhere that knew what they meant;
    # it was classified a withdrawal and cancelled the run.
    #
    # `render_appointment_choice` numbers the rows and records the ids in the
    # same breath, exactly as `render_proposal` does for slots, and returns ""
    # when there is only one candidate — the auto-target path, where asking
    # would be a worse answer than acting.
    if documents_answer:
        said = [documents_answer]
        code_authored = True

    change_step = next(
        (value for value in steps_run if value in ("reschedule", "cancel")), None
    )
    if (
        change_step is not None
        and run.status is WorkflowStatus.IN_PROGRESS
        and run.proposed_action is None
        # A patient who has already answered this must not be asked again. The
        # answering turn only reaches here at all because asking no longer
        # spends the re-plan above — before that it died of the budget first,
        # which is why this line was briefly deleted for being unfalsifiable.
        and not (run.state or {}).get("chosen_appointment_id")
        and not (run.state or {}).get("committed_action")
    ):
        choice = render_appointment_choice(
            session, run, _live_appointments(run), verb=change_step
        )
        if choice:
            said = [choice]
            code_authored = True

    # The Coordinator's acknowledgement is only used when no specialist spoke:
    # "I'll find you an appointment" adds nothing next to "here is the time".
    if said:
        reply, from_model = " ".join(said), not code_authored
    else:
        reply, from_model = _guarded(
            fallback_reply,
            writer=writer,
            fallback=NO_PLAN_REPLY,
            searched_slots=belt.proposals.searched_slots,
        )
    return TurnResult(
        reply=reply,
        author=TraceAuthor.LLM if from_model else TraceAuthor.TEMPLATE,
        run_id=run.id,
        status=run.status.value,
        plan=list(run.plan or []),
        steps_run=steps_run,
        **base,
    )


def _settle_step(
    session: Session,
    *,
    step: PlanStep,
    run: WorkflowRun,
    belt: Toolbelt,
    writer: TraceWriter,
    user: User,
) -> tuple[bool, bool]:
    """Apply the deterministic consequence of a step. Returns (halted, completed).

    The specialist proposed; this is where it becomes true. A step is complete
    only when the database says so — not when the agent said something that
    sounded like completion.
    """
    if step is PlanStep.ROUTE:
        proposals = belt.proposals
        if proposals.department_id is None:
            return False, False
        if proposals.routing_confidence == "low":
            if run.status is WorkflowStatus.PENDING_REVIEW:
                # Already queued, and routing has just re-run on a run that was
                # waiting. `pending_review -> pending_review` is not an edge, so
                # driving it raised through the turn envelope and the patient
                # got an HTTP 500 — seen live when a message during a review
                # classified as a continuation and the plan carried on. The
                # queue item exists, the run is where it belongs, and the only
                # thing to do is stop.
                writer.guard_verdict(
                    "routing_rerun_while_queued",
                    passed=False,
                    detail={"department_id": proposals.department_id},
                )
                return True, False
            # Low-confidence routing is a human's decision, not a confident
            # guess dressed up as one.
            #
            # The candidate is recorded on the run, not only in the escalation's
            # prose: staff approving this decision need a department *id* to
            # apply, and re-deriving one from a sentence would put the model's
            # words back in the consequential path at the exact moment a human
            # was brought in to keep them out of it.
            state = dict(run.state or {})
            state["proposed_department_id"] = proposals.department_id
            state["proposed_department_name"] = proposals.department_name
            run.state = state

            create_escalation(
                session,
                workflow_run_id=run.id,
                kind=EscalationKind.LOW_CONFIDENCE_ROUTING,
                reason=(
                    "Routing was ambiguous; best candidate "
                    f"{proposals.department_name!r}. A person should decide."
                ),
                message=run.request_text or "",
                actor=user,
            )
            transition(
                session,
                run,
                to=WorkflowStatus.PENDING_REVIEW,
                trigger="low_confidence_routing",
                writer=writer,
                actor=user,
            )
            return True, False

        state = dict(run.state or {})
        state["department_id"] = proposals.department_id
        state["department_name"] = proposals.department_name
        run.state = state
        return False, True

    if step in APPOINTMENT_VERBS:
        # Done when *this* verb committed. Checking only for an appointment id
        # would report a reschedule complete before it ran, because the
        # appointment it moves already existed.
        if (run.state or {}).get("committed_action") == step.value:
            return False, True
        if run.proposed_action is not None:
            # Waiting on the patient. Not a failure — the point of the step.
            return True, False
        return False, False

    # Document and follow-up steps have no further deterministic consequence
    # here: their tools already wrote what they had to write.
    return False, True


def _fail_run(
    session: Session,
    *,
    run: WorkflowRun,
    writer: TraceWriter,
    user: User,
    reason: str,
    base: dict,
    plan: list[str],
    steps_run: list[str],
) -> TurnResult:
    """Exhausted budget: the run fails, loudly and on the record.

    And in front of a human. ``FAILED_REPLY`` says "I've flagged it for them",
    and until this escalation existed that was a sentence about nothing: a live
    run failed on its re-plan budget and left **zero** escalation rows, so the
    patient was told help was coming and the staff queue never heard of it. A
    template that makes a promise has to be the thing that keeps it.

    Deduped per run and kind like every other escalation, so a run that fails
    twice is one item with two occurrences rather than two items.

    **The run is failed only from a state the table lets it fail from.** A
    budget can blow on a turn that does not own the run's state at all — the
    safety screen or the Coordinator, while the run sits at
    ``pending_confirmation`` holding a slot the patient has not answered yet, or
    at ``pending_review`` in front of staff. ``pending_confirmation -> failed``
    is not an edge, and driving it raised straight through the turn envelope:
    the patient got an HTTP 500 where a failure notice was intended, and the
    live sweep caught it twice on one run.

    Not moving the run is also the right answer rather than merely the legal
    one. The proposal is still valid and still confirmable; discarding a held
    time because one turn misbehaved would cost the patient the booking they
    had already made a decision about. What the turn owes is the notice and the
    queue item, and both are written either way.
    """
    if is_legal(run.status, WorkflowStatus.FAILED):
        transition(
            session,
            run,
            to=WorkflowStatus.FAILED,
            trigger=reason,
            writer=writer,
            actor=user,
            detail={"budget": reason},
        )
    else:
        # The state the run is in has no edge to `failed`, and that is the
        # table saying the run is not this turn's to end. Recorded rather than
        # silently skipped: "the run failed" and "a turn failed while the run
        # stood" are different facts and the queue item below is the same
        # either way.
        writer.guard_verdict(
            "run_failable",
            passed=False,
            detail={"status": run.status.value, "reason": reason},
        )
    create_escalation(
        session,
        workflow_run_id=run.id,
        kind=EscalationKind.SYSTEM_FAILURE,
        reason=(
            f"The request could not be completed automatically ({reason}). "
            "A person should pick it up."
        ),
        message=run.request_text or "",
        actor=user,
    )
    return TurnResult(
        reply=FAILED_REPLY,
        author=TraceAuthor.TEMPLATE,
        run_id=run.id,
        status=run.status.value,
        plan=plan,
        steps_run=steps_run,
        budget_exhausted=True,
        **base,
    )


async def _commit_proposal(
    session: Session,
    *,
    run: WorkflowRun,
    belt: Toolbelt,
    writer: TraceWriter,
    callbacks: TurnCallbacks,
    user: User,
    conversation_id: str,
    provider: str | None,
    base: dict,
) -> TurnResult:
    """The patient confirmed. Code commits; the agent only words the receipt.

    **The commit dispatches on what was proposed.** ``run.proposed_action`` is
    typed state written at proposal time, and reading it here is the whole
    difference between a system with three appointment verbs and one with one:
    hard-coding ``book_appointment`` meant a cancellation, once confirmed,
    booked something. The three tools all return the same shape, so everything
    downstream of the call is common to all of them.

    A resume is a fresh proposal, not a replay: the target is re-checked inside
    the mutating transaction, and a slot taken in the meantime returns the
    patient to selection rather than failing silently — for every verb, not
    just the one that happened to be built first.
    """
    settings = get_settings()
    action = run.proposed_action
    slot_id = run.proposed_slot_id
    appointment_id = run.proposed_appointment_id

    if action is ProposedAction.RESCHEDULE:
        tool_name = "reschedule_appointment"
        args = {"appointment_id": appointment_id, "new_slot_id": slot_id}
        call = lambda: reschedule_appointment(  # noqa: E731
            session, user, appointment_id, new_slot_id=slot_id, run=run
        )
    elif action is ProposedAction.CANCEL:
        tool_name = "cancel_appointment"
        args = {"appointment_id": appointment_id}
        call = lambda: cancel_appointment(  # noqa: E731
            session, user, appointment_id, run=run
        )
    else:
        tool_name = "book_appointment"
        args = {"slot_id": slot_id}
        call = lambda: book_appointment(  # noqa: E731
            session, user, slot_id=slot_id, reason=run.request_text or "", run=run
        )

    correlation = writer.tool_call(tool_name, args=args, agent_name="orchestrator")
    booked = call()
    writer.tool_result(
        correlation, name=tool_name, result=booked, agent_name="orchestrator"
    )

    if not booked.get("ok"):
        # The proposal was cleared by the tool. Back to selection with fresh
        # alternatives — a dead proposal must never stay confirmable.
        if run.status is WorkflowStatus.PENDING_CONFIRMATION:
            transition(
                session,
                run,
                to=WorkflowStatus.IN_PROGRESS,
                trigger="commit_failed",
                writer=writer,
                actor=user,
                detail={"reason": booked.get("reason")},
            )
        return TurnResult(
            reply=booked.get("message") or DECLINED_REPLY,
            author=TraceAuthor.TEMPLATE,
            run_id=run.id,
            status=run.status.value,
            message_class=MessageClass.CONTINUATION,
            **base,
        )

    appointment_id = (booked.get("appointment") or {}).get("appointment_id")
    state = dict(run.state or {})
    state["appointment_id"] = appointment_id
    # Which verb actually landed. The step bookkeeping reads this rather than
    # the mere presence of an appointment id: a reschedule and a cancel both
    # act on an appointment that already existed, so "there is an id" says
    # nothing about whether this run's step is done.
    state["committed_action"] = (action or ProposedAction.BOOK).value
    run.state = state
    run.clear_proposal()

    transition(
        session,
        run,
        to=WorkflowStatus.IN_PROGRESS,
        trigger="patient_confirmed",
        writer=writer,
        actor=user,
        detail={"appointment_id": appointment_id},
    )

    # The plan carries on from here, and `_execute_plan` settles the committed
    # verb without dispatching its specialist again — `committed_action`, just
    # written above, is what tells it so. The steps that have not run yet (the
    # required-documents diff, the follow-up sweep) still do.
    #
    # This used to be the other way round: the verb was left unmarked on purpose,
    # so that the Appointment agent would run once more and call
    # `render_confirmation` — the seam that re-reads the persisted row. That
    # reason is gone. The receipt is assembled below from rows, so re-running the
    # specialist bought nothing and cost a second proposal.
    belt.run = run
    result = await _execute_plan(
        session,
        run=run,
        belt=belt,
        writer=writer,
        callbacks=callbacks,
        user=user,
        message="",
        fallback_reply="",
        provider=provider,
        settings=settings,
        base=base,
    )

    # The receipt is code's, not the model's. The specialists still ran — the
    # missing-documents task, the follow-up sweep and the verification all
    # happen in the calls above — but what the patient is *told* is assembled
    # from rows here.
    #
    # One live receipt carried four defects at once: a mandatory document and
    # an optional one welded into "still needed, which is optional"; "recorded
    # for follow-up" beside "no outstanding follow-up tasks"; the reminder's
    # fire date presented as the appointment's, which sends a patient to an
    # empty clinic a day early; and a context dump nobody asked for. Every one
    # of them is a fact surviving a paraphrase, so the paraphrase goes.
    receipt = render_receipt(session, run)
    if receipt:
        result.reply = receipt
        result.author = TraceAuthor.TEMPLATE

    result.message_class = MessageClass.CONTINUATION
    return result


__all__ = ["TurnResult", "active_run", "run_workflow"]
