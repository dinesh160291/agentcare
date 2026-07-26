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
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app import clock
from app.agents import SPECIALIST_FOR_STEP, Toolbelt, TurnCallbacks, coordinator, memory
from app.agents.base import run_agent
from app.audit import write_audit
from app.config import get_settings
from app.db import SessionLocal
from app.errors import BudgetExceeded
from app.models import (
    EscalationKind,
    MessageClass,
    PatientProfile,
    PlanStep,
    TERMINAL_WORKFLOW_STATUSES,
    TraceAuthor,
    User,
    UserRole,
    WorkflowRun,
    WorkflowStatus,
)
from app.safety import SafetyVerdict, escalate, keyword_screen, llm_screen
from app.tools import book_appointment, create_escalation
from app.trace import TraceWriter
from app.workflow.confirmation import ConfirmationAnswer, read_confirmation
from app.workflow.mapping import Consequence, apply_consequence, validate_class
from app.workflow.plan import (
    advance_plan,
    is_plan_complete,
    next_step,
    record_replan,
    validate_plan,
)
from app.workflow.state_machine import create_run, transition

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
    if step is PlanStep.BOOK:
        task["request_text"] = run.request_text or message
        task["proposed_slot_id"] = run.proposed_slot_id
        task["appointment_id"] = (run.state or {}).get("appointment_id")
    task.update(extra)
    return json.dumps({k: v for k, v in task.items() if v is not None}, sort_keys=True)


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
    ``failed``. The turn still has to say so.
    """
    if run is None:
        write_audit(
            session,
            action="turn_budget_exhausted",
            entity_type="workflow_run",
            actor=user,
            metadata={"stage": stage},
        )
        return TurnResult(
            reply=FAILED_REPLY,
            author=TraceAuthor.TEMPLATE,
            budget_exhausted=True,
            **base,
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


async def run_workflow(
    user: User,
    message: str,
    session_id: str | None = None,
    *,
    provider: str | None = None,
) -> TurnResult:
    """Run one conversational turn to completion.

    Owns its own database session, because the turn *is* the transaction: the
    state change, its audit row, and its trace rows commit together or not at
    all.
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
        raise ValueError("run_workflow is the patient path; staff act by typed action.")

    writer = TraceWriter(session, session_id=conversation_id)
    callbacks = TurnCallbacks(
        writer,
        max_tool_iterations=settings.max_tool_iterations,
        history_window_turns=settings.history_window_turns,
    )
    writer.inbound(message, author=TraceAuthor.PATIENT_MESSAGE)

    try:
        result = await _turn(
            session,
            writer=writer,
            callbacks=callbacks,
            user=acting,
            message=message,
            conversation_id=conversation_id,
            provider=provider,
        )
        writer.outbound(result.reply, author=result.author)
        session.commit()
        return result
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
        raise
    finally:
        session.close()


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
        session, user=user, patient_id=profile.id, writer=writer, run=run
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

    # --- 1. a pending confirmation is read in code, before any model call ---
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
            run.clear_proposal()
            transition(
                session,
                run,
                to=WorkflowStatus.IN_PROGRESS,
                trigger="patient_declined",
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
        # UNREAD falls through: the model may re-ask or decline, never confirm.

    # --- 2. the Coordinator: classify against a live run, or plan a new one ---
    # One service for the whole turn — each instance builds its own engine, and
    # the conversation is the only thing that needs a durable one.
    conversation = memory.conversation_service()
    existing = await conversation.get_session(
        app_name=memory.APP_NAME, user_id=str(user.id), session_id=conversation_id
    )
    coordinator_reply = await run_agent(
        coordinator.build_agent(belt, callbacks, provider=provider),
        task_text=message,
        callbacks=callbacks,
        session_service=conversation,
        session_id=conversation_id,
        user_id=str(user.id),
        create=existing is None,
    )

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
    plan = belt.proposals.plan
    writer.guard_verdict(
        "scope_gate",
        passed=plan is not None,
        detail={"steps": [step.value for step in plan] if plan else []},
    )
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

    # The scope gate runs inside the mapping too, not only at run creation.
    # An off-topic message that fell through to continuation would be appended
    # to the run's stored request text, contaminating what routing and slot
    # matching read later.
    writer.guard_verdict(
        "scope_gate",
        passed=outcome.consequence is not Consequence.SCOPE_REPLY,
        detail={"class": outcome.message_class.value},
    )

    if outcome.consequence is Consequence.WITHDRAW:
        return TurnResult(
            reply=WITHDRAWN_REPLY,
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

    if outcome.consequence is Consequence.ANSWER_AND_STAY:
        return TurnResult(
            reply=coordinator_reply or NO_PLAN_REPLY,
            author=TraceAuthor.LLM if coordinator_reply else TraceAuthor.TEMPLATE,
            run_id=run.id,
            status=run.status.value,
            message_class=outcome.message_class,
            **base,
        )

    if outcome.consequence is Consequence.SUPERSEDE:
        steps = belt.proposals.incoming_steps or [PlanStep.ROUTE]
        replacement = create_run(
            session,
            patient_id=profile.id,
            status=WorkflowStatus.IN_PROGRESS,
            trigger="superseded_previous_request",
            writer=writer,
            request_text=message,
            plan=[step.value for step in validate_plan([s.value for s in steps])],
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
    # Each specialist's reply is kept, not overwritten. The booking receipt and
    # the missing-documents list are both things the patient needs; letting the
    # last step win would silently drop whichever mattered most.
    said: list[str] = []

    while True:
        step = next_step(run)
        if step is None:
            break

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
        if step_reply:
            said.append(step_reply)

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

    # The Coordinator's acknowledgement is only used when no specialist spoke:
    # "I'll find you an appointment" adds nothing next to "here is the time".
    reply = " ".join(said) if said else fallback_reply
    return TurnResult(
        reply=reply or NO_PLAN_REPLY,
        author=TraceAuthor.LLM if reply else TraceAuthor.TEMPLATE,
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
            # Low-confidence routing is a human's decision, not a confident
            # guess dressed up as one.
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

    if step is PlanStep.BOOK:
        if (run.state or {}).get("appointment_id"):
            return False, True
        if run.proposed_slot_id is not None:
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
    """Exhausted budget: the run fails, loudly and on the record."""
    transition(
        session,
        run,
        to=WorkflowStatus.FAILED,
        trigger=reason,
        writer=writer,
        actor=user,
        detail={"budget": reason},
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

    A resume is a fresh proposal, not a replay: the slot is re-checked inside
    the booking transaction, and a slot taken in the meantime returns the
    patient to selection rather than failing silently.
    """
    slot_id = run.proposed_slot_id
    settings = get_settings()

    correlation = writer.tool_call(
        "book_appointment", args={"slot_id": slot_id}, agent_name="orchestrator"
    )
    booked = book_appointment(
        session, user, slot_id=slot_id, reason=run.request_text or "", run=run
    )
    writer.tool_result(
        correlation, name="book_appointment", result=booked, agent_name="orchestrator"
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

    # The `book` step is deliberately *not* marked done here. Letting the plan
    # run it is what makes the Appointment agent call `render_confirmation` —
    # the seam that re-reads the persisted row — so the receipt states facts
    # from the database rather than from what the booking call returned.
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
    result.message_class = MessageClass.CONTINUATION
    return result


__all__ = ["TurnResult", "active_run", "run_workflow"]
