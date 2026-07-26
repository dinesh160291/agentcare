"""Binding plain tool functions to a turn.

``app/tools/`` holds framework-agnostic functions that take a session, an
acting user, and primitives. An agent cannot call those directly: it has no
session, no user, and no business being handed either. This module closes over
them for the duration of one turn and hands the agent a set of no-secrets-in-
the-signature callables.

That binding is a guard as much as a convenience. A tool whose ``patient_id``
is bound from the authenticated user cannot be pointed at another patient by a
model that got creative, and a booking tool whose department comes from the
run's state cannot file a Cardiology slot under Dermatology.

**Proposal tools return their rejection instead of raising.** A model that
submits a malformed plan gets told what was wrong and can correct it — that
*is* the retry ladder. An exception here would abort the turn instead, losing a
recoverable situation and taking the trace with it.

ADK calls sync tool functions directly on the event loop, not in a worker
thread, so every tool here shares the turn's transaction: the audit row, the
trace rows, and the change they describe commit or roll back together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from sqlalchemy.orm import Session

from app import clock
from app.errors import ClassRejected, PlanRejected
from app.models import (
    FollowUpTaskType,
    PlanStep,
    ProposedAction,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.trace import TraceWriter
from app.tools import (
    diff_required_documents,
    find_available_slots,
    get_patient_context,
    get_slot,
    list_departments,
    list_open_tasks,
    list_patient_documents,
    list_patient_reminders,
    render_confirmation,
    resolve_date,
    resolve_department,
    upsert_followup_task,
    validate_department,
)
from app.workflow.mapping import ClassVerdict, validate_class
from app.workflow.plan import validate_plan
from app.workflow.state_machine import transition

#: The closed set the model may answer a proposal with. ``confirm`` is
#: deliberately absent: commitment requires a click or an exact token, and the
#: model's only permitted verdicts are decline or non-answer.
CONFIRMATION_VERDICTS = ("decline", "non_answer")


@dataclass
class TurnProposals:
    """What the agents proposed this turn, after code validated it.

    The orchestrator reads this rather than parsing replies: a proposal that
    only exists in prose is a proposal nobody can enforce an ordering on.
    """

    plan: list[PlanStep] | None = None
    class_verdict: ClassVerdict | None = None
    #: "decline" or "non_answer" — never "confirm". See ``CONFIRMATION_VERDICTS``.
    confirmation_verdict: str | None = None
    incoming_steps: list[PlanStep] = field(default_factory=list)
    department_id: int | None = None
    department_name: str | None = None
    routing_confidence: str | None = None
    proposed_slot_id: int | None = None
    rejections: list[str] = field(default_factory=list)


class Toolbelt:
    """One turn's worth of bound tools."""

    def __init__(
        self,
        session: Session,
        *,
        user: User,
        patient_id: int,
        writer: TraceWriter,
        run: WorkflowRun | None = None,
    ) -> None:
        self.session = session
        self.user = user
        self.patient_id = patient_id
        self.writer = writer
        self.run = run
        self.proposals = TurnProposals()

    # --- helpers ---------------------------------------------------------

    def _department_id(self) -> int | None:
        """The department this run is about, from the run's state.

        The run row is authoritative; session state is the model's scratchpad
        and may never supply a consequential value when a row exists.
        """
        if self.proposals.department_id is not None:
            return self.proposals.department_id
        if self.run is not None:
            value = (self.run.state or {}).get("department_id")
            return int(value) if value is not None else None
        return None

    @staticmethod
    def _as_date(value: str) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None

    # --- Coordinator -----------------------------------------------------

    def coordinator_tools(self) -> list:
        def load_patient_context() -> dict:
            """Load what is already known about this patient: their profile,
            upcoming appointments, and documents on file."""
            return get_patient_context(
                self.session, self.user, patient_id=self.patient_id
            )

        def submit_plan(steps: list[str]) -> dict:
            """Submit the plan for this request as a list of steps. Valid steps
            are: route, book, documents, follow_up."""
            try:
                plan = validate_plan(steps)
            except PlanRejected as rejected:
                self.writer.validation(
                    "coordinator_plan",
                    accepted=False,
                    detail={"proposed": steps, "problem": str(rejected)},
                )
                self.proposals.rejections.append(str(rejected))
                return {"accepted": False, "problem": str(rejected)}

            self.writer.validation(
                "coordinator_plan",
                accepted=True,
                detail={"proposed": steps, "plan": [s.value for s in plan]},
            )
            self.proposals.plan = plan
            return {"accepted": True, "plan": [step.value for step in plan]}

        def classify_message(message_class: str, incoming_steps: list[str]) -> dict:
            """Say how this message relates to the patient's active request.
            One of: withdrawal, off_topic, side_question, complementary,
            conflicting, continuation. `incoming_steps` is what this new
            message by itself would need (may be empty)."""
            if self.run is None:
                return {
                    "accepted": False,
                    "problem": "There is no active request to classify against.",
                }

            steps: list[PlanStep] = []
            for value in incoming_steps or []:
                try:
                    steps.append(PlanStep(value))
                except ValueError:
                    # An unknown step is not worth failing the classification
                    # over; the class is the consequential part.
                    continue

            try:
                verdict = validate_class(
                    message_class,
                    run=self.run,
                    incoming_steps=steps or None,
                    writer=self.writer,
                )
            except ClassRejected as rejected:
                self.proposals.rejections.append(str(rejected))
                return {"accepted": False, "problem": str(rejected)}

            self.proposals.class_verdict = verdict
            self.proposals.incoming_steps = steps
            return {
                "accepted": True,
                "applied_class": verdict.message_class.value,
                "adjusted": verdict.adjusted,
                "reason": verdict.reason,
            }

        def submit_confirmation_verdict(verdict: str, reason: str = "") -> dict:
            """Say how the patient answered the time you offered them.
            `verdict` must be "decline" or "non_answer". There is no third
            option: you may never confirm a booking. Only the patient's own
            exact word, or the Confirm button, can do that."""
            return self._submit_confirmation_verdict(verdict, reason)

        # The toolset itself says which decision is wanted. A Coordinator with
        # no active run cannot classify against one, and a Coordinator with a
        # live run must not quietly start a second — so the wrong tool is
        # absent rather than merely discouraged.
        decision = classify_message if self.run is not None else submit_plan
        tools = [load_patient_context, decision]

        # The model's half of the confirmation read, and only where it applies.
        # Handing it out permanently would invite it to answer a question
        # nobody asked.
        if (
            self.run is not None
            and self.run.status is WorkflowStatus.PENDING_CONFIRMATION
        ):
            tools.append(submit_confirmation_verdict)
        return tools

    def _submit_confirmation_verdict(self, verdict: str, reason: str) -> dict:
        """Validate the model's read of a confirmation answer.

        The asymmetry, restated as code: a wrongly re-asked "yes" costs one
        tap; a wrongly committed "no" books an appointment against the
        patient's word at the exact step built to prevent that. So the enum has
        two members and ``confirm`` is not one of them — the refusal is
        structural rather than a matter of the prompt holding.
        """
        proposed = str(verdict or "").strip().lower()

        if proposed not in CONFIRMATION_VERDICTS:
            problem = (
                "You cannot confirm a booking. Only the patient's exact word or "
                "the Confirm button can. Use 'decline' or 'non_answer'."
                if proposed in ("confirm", "confirmed", "yes")
                else f"Not a verdict. Use one of: {', '.join(CONFIRMATION_VERDICTS)}."
            )
            self.writer.validation(
                "confirmation_verdict",
                accepted=False,
                detail={"proposed": verdict, "problem": problem},
            )
            self.proposals.rejections.append(problem)
            return {"accepted": False, "problem": problem}

        self.writer.validation(
            "confirmation_verdict", accepted=True, detail={"verdict": proposed}
        )
        self.proposals.confirmation_verdict = proposed
        return {"accepted": True, "verdict": proposed, "reason": reason}

    # --- Department Routing ----------------------------------------------

    def routing_tools(self) -> list:
        def resolve_department_tool(text: str) -> dict:
            """Work out which department handles a request. Pass the part of
            the patient's message describing what the appointment is for."""
            return resolve_department(self.session, text)

        def list_departments_tool() -> dict:
            """List every department that accepts appointments."""
            return {"departments": list_departments(self.session)}

        def submit_routing(department_name: str, confidence: str) -> dict:
            """Submit the department this request should be routed to.
            `confidence` is "high" or "low"; use "low" when the match is
            ambiguous or unsupported, so a person can decide."""
            checked = validate_department(self.session, department_name)
            accepted = bool(checked.get("valid"))

            self.writer.validation(
                "routing_department",
                accepted=accepted,
                detail={
                    "proposed": department_name,
                    "confidence": confidence,
                    "problem": None if accepted else "not a department",
                },
            )
            if not accepted:
                self.proposals.rejections.append(
                    f"{department_name!r} is not a department"
                )
                return {
                    "accepted": False,
                    "problem": (
                        f"{department_name!r} is not one of this hospital's "
                        "departments. Call list_departments to see them."
                    ),
                }

            self.proposals.department_id = checked["department"]["id"]
            self.proposals.department_name = checked["department"]["name"]
            self.proposals.routing_confidence = (
                "low" if str(confidence).lower().startswith("low") else "high"
            )
            return {
                "accepted": True,
                "department": checked["department"],
                "confidence": self.proposals.routing_confidence,
            }

        resolve_department_tool.__name__ = "resolve_department"
        list_departments_tool.__name__ = "list_departments"
        return [resolve_department_tool, list_departments_tool, submit_routing]

    # --- Appointment ------------------------------------------------------

    def appointment_tools(self) -> list:
        def resolve_date_tool(phrase: str) -> dict:
            """Turn a phrase like "next Tuesday" or "next week" into real
            dates. Always use this; never work a date out yourself."""
            return resolve_date(phrase, today=clock.today())

        def find_available_slots_tool(
            start: str = "", end: str = "", part_of_day: str = ""
        ) -> dict:
            """Find free appointment times in this request's department.
            Dates are ISO (YYYY-MM-DD); part_of_day is morning or afternoon."""
            department_id = self._department_id()
            if department_id is None:
                return {
                    "slots": [],
                    "total_matching": 0,
                    "problem": "No department has been decided for this request yet.",
                }
            return find_available_slots(
                self.session,
                department_id=department_id,
                start=self._as_date(start),
                end=self._as_date(end),
                part_of_day=part_of_day or None,
            )

        def propose_appointment(slot_id: int) -> dict:
            """Propose a specific time to the patient. This books nothing: it
            records what they are being asked to confirm."""
            return self._propose_appointment(slot_id)

        def render_confirmation_tool(appointment_id: int) -> dict:
            """Read a booked appointment back from the database so you can
            state its details. Every fact in your reply must come from here."""
            return render_confirmation(self.session, appointment_id)

        resolve_date_tool.__name__ = "resolve_date"
        find_available_slots_tool.__name__ = "find_available_slots"
        render_confirmation_tool.__name__ = "render_confirmation"
        return [
            resolve_date_tool,
            find_available_slots_tool,
            propose_appointment,
            render_confirmation_tool,
        ]

    def _propose_appointment(self, slot_id: int) -> dict:
        """Record a typed proposal on the run and pause for confirmation.

        The proposal is state on the row, not a sentence in the transcript:
        the patient's yes/no is resolved against this, so it has to survive
        history windowing, session expiry, and restarts.
        """
        run = self.run
        if run is None:
            return {"accepted": False, "problem": "There is no active request."}

        found = get_slot(self.session, slot_id)
        if not found.get("found"):
            return {
                "accepted": False,
                "problem": f"Slot {slot_id} does not exist.",
            }

        # `available` is the tool's own verdict, and it is stricter than a
        # status check: it also rules out an inactive doctor, a closed
        # department, and a slot that is simply in the past.
        if not found.get("available"):
            return {
                "accepted": False,
                "problem": "That time is no longer available. Offer another.",
            }

        department_id = self._department_id()
        if department_id is not None and found["department_id"] != department_id:
            self.writer.validation(
                "appointment_proposal",
                accepted=False,
                detail={
                    "slot_id": slot_id,
                    "problem": "slot is in a different department to the run",
                },
            )
            return {
                "accepted": False,
                "problem": "That time is in a different department to this request.",
            }

        self.writer.validation(
            "appointment_proposal", accepted=True, detail={"slot_id": slot_id}
        )

        run.proposed_action = ProposedAction.BOOK
        run.proposed_slot_id = slot_id
        self.proposals.proposed_slot_id = slot_id

        if run.status is WorkflowStatus.IN_PROGRESS:
            transition(
                self.session,
                run,
                to=WorkflowStatus.PENDING_CONFIRMATION,
                trigger="appointment_proposed",
                writer=self.writer,
                actor=self.user,
                detail={"slot_id": slot_id},
            )

        return {"accepted": True, "proposed": found}

    # --- Document ---------------------------------------------------------

    def document_tools(self) -> list:
        def list_patient_documents_tool() -> dict:
            """List the documents this patient already has on file."""
            return {
                "documents": list_patient_documents(
                    self.session, patient_id=self.patient_id
                )
            }

        def diff_required_documents_tool() -> dict:
            """Compare what this request's department requires against what the
            patient has filed, and report what is missing."""
            department_id = self._department_id()
            if department_id is None:
                return {
                    "missing": [],
                    "optional_missing": [],
                    "problem": "No department has been decided for this request yet.",
                }
            return diff_required_documents(
                self.session, patient_id=self.patient_id, department_id=department_id
            )

        def record_missing_documents() -> dict:
            """Record what is still missing as a follow-up task, so it is not
            lost when this conversation ends."""
            return self._record_missing_documents()

        list_patient_documents_tool.__name__ = "list_patient_documents"
        diff_required_documents_tool.__name__ = "diff_required_documents"
        return [
            list_patient_documents_tool,
            diff_required_documents_tool,
            record_missing_documents,
        ]

    def _record_missing_documents(self) -> dict:
        """Re-run the diff and upsert the task from *its* result.

        Deliberately takes no arguments. A model-supplied list of missing
        documents is a list a model can invent; re-reading makes the task and
        the database agree by construction, and the upsert keeps it to one open
        task per appointment however often this runs.
        """
        department_id = self._department_id()
        if department_id is None:
            return {"recorded": False, "problem": "No department decided yet."}

        diff = diff_required_documents(
            self.session, patient_id=self.patient_id, department_id=department_id
        )
        missing = diff.get("missing") or []
        appointment_id = (
            (self.run.state or {}).get("appointment_id") if self.run else None
        )

        if not missing:
            from app.tools import close_followup_tasks

            closed = close_followup_tasks(
                self.session,
                patient_id=self.patient_id,
                task_type=FollowUpTaskType.MISSING_DOCUMENTS,
                appointment_id=appointment_id,
            )
            return {"recorded": True, "missing": [], "closed": closed}

        task = upsert_followup_task(
            self.session,
            patient_id=self.patient_id,
            task_type=FollowUpTaskType.MISSING_DOCUMENTS,
            details={"missing": missing},
            appointment_id=appointment_id,
        )
        return {"recorded": True, "missing": missing, "task": task["task"]}

    # --- Follow-up --------------------------------------------------------

    def followup_tools(self) -> list:
        def list_patient_reminders_tool() -> dict:
            """List reminders scheduled for this patient."""
            return {
                "reminders": list_patient_reminders(
                    self.session, patient_id=self.patient_id
                )
            }

        def list_open_tasks_tool() -> dict:
            """List this patient's outstanding follow-up tasks."""
            return {"tasks": list_open_tasks(self.session, patient_id=self.patient_id)}

        list_patient_reminders_tool.__name__ = "list_patient_reminders"
        list_open_tasks_tool.__name__ = "list_open_tasks"
        return [list_patient_reminders_tool, list_open_tasks_tool]


__all__ = ["CONFIRMATION_VERDICTS", "Toolbelt", "TurnProposals"]
