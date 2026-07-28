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
from typing import Literal

from sqlalchemy.orm import Session

from app import clock
from app.errors import ClassRejected, PlanRejected
from app.models import (
    Appointment,
    AppointmentSlot,
    FollowUpTaskType,
    PatientDocument,
    PlanStep,
    ProposedAction,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.tools.appointments import LIVE_STATUSES
from app.trace import TraceWriter
from app.workflow.replies import (
    listed_appointment_ids,
    offered_slot_ids,
    record_offered,
    was_offered,
)
from app.tools import (
    apply_verification,
    diff_required_documents,
    extract_document_text,
    find_available_slots,
    get_patient_context,
    get_slot,
    list_departments,
    list_open_tasks,
    list_patient_appointments,
    list_patient_documents,
    list_patient_reminders,
    list_unverified_documents,
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
#: The model's permitted reads of an answer to a proposal. ``confirm`` is
#: deliberately absent — commitment requires a click or an exact token.
#: ``slot_question`` is the third because a question is a third kind of answer,
#: and without it four availability questions in one live conversation were all
#: filed as non-answers and met the same frozen re-ask.
CONFIRMATION_VERDICTS = ("decline", "non_answer", "slot_question")


@dataclass
class TurnProposals:
    """What the agents proposed this turn, after code validated it.

    The orchestrator reads this rather than parsing replies: a proposal that
    only exists in prose is a proposal nobody can enforce an ordering on.
    """

    plan: list[PlanStep] | None = None
    class_verdict: ClassVerdict | None = None
    #: "decline", "non_answer" or "slot_question" — never "confirm". See
    #: ``CONFIRMATION_VERDICTS``.
    confirmation_verdict: str | None = None
    incoming_steps: list[PlanStep] = field(default_factory=list)
    department_id: int | None = None
    department_name: str | None = None
    routing_confidence: str | None = None
    proposed_slot_id: int | None = None
    #: Set only by a reschedule or cancellation proposal — the appointment the
    #: patient is being asked about, never one the model picked for them.
    proposed_appointment_id: int | None = None
    #: The slot payload the last availability search returned, verbatim. The
    #: reply's shortlist is built from this rather than from the specialist's
    #: prose, so what the patient is shown and what the re-proposal guard will
    #: accept are the same list by construction.
    offered_slots: list[dict] = field(default_factory=list)
    #: This turn answered a question about other times while a proposal stood.
    answered_with_slots: bool = False
    #: A tool that returns slots actually ran and answered this turn. What the
    #: reply guard checks an availability claim against: live, a turn whose only
    #: tool results were two "No department has been decided yet" refusals told
    #: the patient there were "no available appointment slots in ENT for next
    #: week", and a search two turns later found 72. A refusal to search is not
    #: an empty schedule. Set only where slots came back — a refusal leaves it
    #: alone, which is the whole distinction.
    searched_slots: bool = False
    #: The window ``resolve_date`` made of the patient's words, when they
    #: scoped their question ("between the 5th and the 10th"). Kept so the
    #: reply can say which period it is answering about.
    slot_window: dict | None = None
    #: This turn moved the proposal to a different slot the patient had been
    #: shown. Never a commit — the run is still waiting on an exact answer.
    reproposed: bool = False
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

        A reschedule or cancellation run never went through routing, so its
        state holds no department — and the department is not missing, it just
        lives somewhere else: on the appointment being changed. Without this
        fallback, "some time next week?" inside a reschedule run reached
        ``list_other_slots``, got "No department has been decided yet", and the
        patient was asked to name a department the system already knew — then
        offered a *cancelled* Dermatology one as the likely answer.
        """
        if self.proposals.department_id is not None:
            return self.proposals.department_id
        if self.run is None:
            return None
        value = (self.run.state or {}).get("department_id")
        if value is not None:
            return int(value)
        target = self._target_appointment()
        return target.department_id if target is not None else None

    def _target_appointment(self) -> Appointment | None:
        """The appointment this run is changing, when that is already settled.

        Settled means one of three things, in order: a proposal names it, a
        commit recorded it, or the patient has exactly one live appointment and
        the run is about changing an appointment — the auto-target case, where
        there was never anything to choose between.

        Never a guess among several. Choosing the referent is language and it
        belongs to the model, with ``listed_appointment_ids`` standing under
        it; this is only for reading a department off a decision already made.
        Ownership is re-checked here because an id on the run is still an id.
        """
        run = self.run
        if run is None:
            return None

        for candidate_id in (
            run.proposed_appointment_id,
            (run.state or {}).get("appointment_id"),
        ):
            if candidate_id is None:
                continue
            appointment = self.session.get(Appointment, int(candidate_id))
            if appointment is not None and appointment.patient_id == self.patient_id:
                return appointment

        planned = set(run.plan or [])
        if not planned & {PlanStep.RESCHEDULE.value, PlanStep.CANCEL.value}:
            return None
        live = list_patient_appointments(
            self.session, patient_id=self.patient_id, live_only=True
        )
        if len(live) != 1:
            return None
        return self.session.get(Appointment, live[0]["appointment_id"])

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

        def submit_plan(
            steps: list[
                Literal[
                    "route", "book", "reschedule", "cancel", "documents", "follow_up"
                ]
            ],
        ) -> dict:
            """Submit the plan for this request: a list of step names, e.g.
            ["route", "book"]. Each item is one of the words below, on its own —
            never an object. Valid steps are: route, book, reschedule, cancel,
            documents, follow_up."""
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

        def submit_confirmation_verdict(
            verdict: str, reason: str = "", phrase: str = ""
        ) -> dict:
            """Say how the patient answered the time you offered them.
            "decline" - they turned it down. "slot_question" - they asked to
            see other times; put their own words about when into `phrase`
            ("between the 5th and the 10th", "week after next"). "non_answer" -
            anything else. You may never confirm a booking: only the patient's
            own exact word, or the Confirm button, can do that."""
            return self._submit_confirmation_verdict(verdict, reason, phrase)

        def list_other_slots(phrase: str = "") -> dict:
            """Show the patient free appointment times. Use this whenever they
            ask what is available. Put their own words about when into `phrase`
            ("next week", "after the 10th") and leave it empty if they said
            nothing about timing. This books nothing and changes nothing."""
            return self._list_other_slots(phrase)

        def propose_another_slot(slot_id: int) -> dict:
            """Move the offer to a different time the patient has already been
            shown — use it when they pick one of the other options. `slot_id`
            must come from a list you have shown them in this conversation.
            This books nothing: they still have to confirm."""
            return self._propose_another_slot(slot_id)

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
            # Moving the offer is only meaningful where there is one to move.
            # Listing times is *not* handed out here: at this state it is
            # reachable through the confirmation verdict instead, so there is
            # exactly one way to answer "what else is there?" and exactly one
            # place the proposal could be disturbed.
            if self.run.proposed_action is ProposedAction.BOOK:
                tools.append(propose_another_slot)
        elif self.run is not None and not self.run.is_terminal:
            # Answering "what else is free?" is the other half of the mapping
            # table's answer-and-stay, and it needs data. Live, at
            # `in_progress`, the same question got "noted as a side question. I
            # will continue... Please hold on for a moment" — twice, with
            # nothing following. Read-only by construction, and it refuses
            # itself when no department has been decided.
            tools.append(list_other_slots)
        return tools

    def answer_with_other_slots(self, phrase: str = "") -> dict:
        """The same search, run by code rather than asked for by the model.

        The orchestrator needs this when it has decided *deterministically*
        that a message is a refinement — "can you give me slots for next week?"
        while a proposal stands. The model had already called that message a
        new request, so it never asked for the times; if code refuses the
        supersede and then waits for a tool call that is not coming, the
        patient gets a re-ask for an answer that was one query away.

        Read-only and identical to the bound tool, deliberately: two ways of
        producing a slot list would be two places for the proposal to be
        disturbed.
        """
        return self._list_other_slots(phrase)

    def _list_other_slots(self, phrase: str = "") -> dict:
        """Free times other than the one being held. Read-only, always.

        The proposal is not touched here — not cleared, not moved, not
        re-timed. That is the whole point of the tool existing: the mapping
        table has always said a side question is answer-and-stay, and the
        confirmation path could only decline or re-ask, so the *answer* half
        had nowhere to happen and a patient asking "what else is there?" got a
        nag instead of an answer, twice.

        ``phrase`` is the patient's own words about timing, and it goes through
        ``resolve_date`` like every other date in this system — never parsed
        here. Without it, "between august 5th and 10th" and "week after next"
        are both answered with the same list starting from today, which is how
        four different questions in one live conversation got one answer.

        Refusing is not answering. When there is no department yet — a run
        still waiting for staff to route it — this returns the problem and
        leaves ``answered_with_slots`` alone, so the turn falls through to
        whatever it would otherwise have said instead of rendering an empty
        list as though it were an answer.
        """
        run = self.run
        if run is None:
            return {"slots": [], "problem": "There is no active request."}

        department_id = self._department_id()
        if department_id is None:
            return {"slots": [], "problem": "No department has been decided yet."}

        window = resolve_date(phrase, today=clock.today()) if phrase else {}
        start = self._as_date(window.get("start") or "") if window.get("resolved") else None
        end = self._as_date(window.get("end") or "") if window.get("resolved") else None

        # With no window, anchored on the day being held rather than on today:
        # the alternatives to an offer are the times near it, and a list
        # starting from now answers a question nobody asked. It is also what
        # makes "the 10am one" resolvable — the times they can name are the
        # times they were shown.
        held = (
            self.session.get(AppointmentSlot, run.proposed_slot_id)
            if run.proposed_slot_id
            else None
        )
        if start is None and held is not None:
            start = held.start_time.date()

        found = find_available_slots(
            self.session,
            department_id=department_id,
            start=start,
            end=end,
            part_of_day=window.get("part_of_day") or None,
        )
        others = [
            slot
            for slot in found.get("slots") or []
            if slot.get("slot_id") != run.proposed_slot_id
        ]
        self.proposals.offered_slots = others
        self.proposals.answered_with_slots = True
        self.proposals.searched_slots = True
        self.proposals.slot_window = window if window.get("resolved") else None
        # Everything the search returned is answerable, not merely the three
        # the reply lists. "Lets go with 4pm slot" named a time that was in
        # this payload and below the fold, and recording only the rendered
        # shortlist made the re-proposal guard refuse a slot the tool had just
        # produced — the patient was shown alternatives again instead of being
        # given the one they asked for.
        #
        # The guarantee is unchanged and is the one that matters: an id has to
        # come from a tool result rather than from the model's memory of the
        # conversation. A slot in another department, or one no search
        # returned, is still refused.
        record_offered(run, others)
        return {"slots": others, "total_matching": len(others), "window": window or None}

    def _propose_another_slot(self, slot_id: int) -> dict:
        """Move the offer to a slot the patient has actually been shown.

        Two checks, and the order matters.

        **Was it offered?** The id must be in the set this run built from
        ``find_available_slots`` payloads. A model naming an id it recalls from
        its context window is indistinguishable from one inventing an id —
        both arrive as an integer — so the only safe source is a tool result.
        The rejection is written to the trace, because a refused invention is
        exactly the event a reviewer needs to see and it leaves no other mark.

        **Is it still free?** A slot on the shown list can be taken between
        being shown and being chosen. This is the commit-time slot-taken
        discipline applied one step earlier: ``_propose_appointment`` runs the
        same liveness check the original proposal ran, and a dead slot returns
        fresh options rather than a proposal nobody can honour. The proposal
        already held is left standing throughout — a failed swap must never
        leave the patient holding nothing.
        """
        run = self.run
        if run is None:
            return {"accepted": False, "problem": "There is no active request."}

        if not was_offered(run, slot_id):
            self.writer.validation(
                "reproposal_slot_offered",
                accepted=False,
                detail={
                    "slot_id": slot_id,
                    "offered": offered_slot_ids(run),
                    "problem": "that slot was never shown to this patient",
                },
            )
            return {
                "accepted": False,
                "problem": (
                    "You may only offer a time this patient has already been "
                    "shown. Ask for the times first, and use an id from that list."
                ),
            }

        self.writer.validation(
            "reproposal_slot_offered", accepted=True, detail={"slot_id": slot_id}
        )

        result = self._propose_appointment(slot_id)
        if not result.get("accepted"):
            # Gone since it was shown. Say so with something to choose from —
            # a refusal with no alternatives is where a conversation stops.
            return {**result, **self._list_other_slots()}

        self.proposals.reproposed = True
        return result

    def _submit_confirmation_verdict(
        self, verdict: str, reason: str, phrase: str = ""
    ) -> dict:
        """Validate the model's read of a confirmation answer.

        The asymmetry, restated as code: a wrongly re-asked "yes" costs one
        tap; a wrongly committed "no" books an appointment against the
        patient's word at the exact step built to prevent that. So ``confirm``
        is not a member of the enum — the refusal is structural rather than a
        matter of the prompt holding.

        **Three members, because a question is a third kind of answer.** Live,
        four availability questions in a row — "between august 5th and 10th",
        "week after next", "after august 3rd", "more slots after august 10th" —
        were all read as non-answers and met the same frozen re-ask, because
        the enum had nowhere else to put them. ``slot_question`` carries the
        patient's own words about timing and routes them through
        ``resolve_date``, which is the only thing in this system permitted to
        turn a phrase into dates.

        Classification stays the model's job and the phrase stays the
        patient's: there is no keyword list here, so a phrasing nobody
        anticipated is read by the same thing that reads every other one.
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

        if proposed == "slot_question":
            # Answered here rather than by a second tool: the confirmation read
            # is the one place that says what the patient's reply *was*, and
            # two entry points for "show me other times" would be two places
            # for the proposal to be disturbed.
            return {
                "accepted": True,
                "verdict": proposed,
                **self._list_other_slots(phrase),
            }
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
            found = find_available_slots(
                self.session,
                department_id=department_id,
                start=self._as_date(start),
                end=self._as_date(end),
                part_of_day=part_of_day or None,
            )
            self.proposals.offered_slots = list(found.get("slots") or [])
            self.proposals.searched_slots = True
            return found

        def propose_appointment(slot_id: int) -> dict:
            """Propose a specific time to the patient. This books nothing: it
            records what they are being asked to confirm."""
            return self._propose_appointment(slot_id)

        def render_confirmation_tool(appointment_id: int) -> dict:
            """Read a booked appointment back from the database so you can
            state its details. Every fact in your reply must come from here."""
            return render_confirmation(self.session, appointment_id)

        def list_my_appointments() -> dict:
            """List this patient's changeable appointments. Use this before
            proposing a reschedule or a cancellation: you must name exactly
            which appointment, and you may never guess which one they mean."""
            return {
                "appointments": list_patient_appointments(
                    self.session, patient_id=self.patient_id, live_only=True
                )
            }

        def find_slots_for_reschedule(
            appointment_id: int, start: str = "", end: str = "", part_of_day: str = ""
        ) -> dict:
            """Find free times to move an existing appointment to.

            Separate from find_available_slots because a reschedule keeps the
            appointment's own department, and a reschedule run never went
            through routing so the run does not carry one."""
            appointment = self.session.get(Appointment, appointment_id)
            if appointment is None or appointment.patient_id != self.patient_id:
                return {
                    "slots": [],
                    "total_matching": 0,
                    "problem": f"Appointment {appointment_id} is not this patient's.",
                }
            found = find_available_slots(
                self.session,
                department_id=appointment.department_id,
                start=self._as_date(start),
                end=self._as_date(end),
                part_of_day=part_of_day or None,
            )
            self.proposals.searched_slots = True
            return found

        def propose_reschedule(appointment_id: int, slot_id: int) -> dict:
            """Propose moving an existing appointment to a new time. This
            changes nothing: it records what the patient is being asked to
            confirm."""
            return self._propose_change(
                ProposedAction.RESCHEDULE,
                appointment_id=appointment_id,
                slot_id=slot_id,
            )

        def propose_cancellation(appointment_id: int) -> dict:
            """Propose cancelling one specific appointment. This cancels
            nothing: it records what the patient is being asked to confirm."""
            return self._propose_change(
                ProposedAction.CANCEL, appointment_id=appointment_id, slot_id=None
            )

        resolve_date_tool.__name__ = "resolve_date"
        find_available_slots_tool.__name__ = "find_available_slots"
        render_confirmation_tool.__name__ = "render_confirmation"
        return [
            resolve_date_tool,
            find_available_slots_tool,
            propose_appointment,
            render_confirmation_tool,
            list_my_appointments,
            find_slots_for_reschedule,
            propose_reschedule,
            propose_cancellation,
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

    def _propose_change(
        self,
        action: ProposedAction,
        *,
        appointment_id: int,
        slot_id: int | None,
    ) -> dict:
        """Record a typed reschedule or cancellation proposal on the run.

        Every check here is the deterministic half of "name exactly which
        appointment". Code cannot know which appointment the patient *meant* —
        that is language, and it is the model's job — but it can refuse to
        record a proposal against an appointment that is not this patient's, is
        not changeable, or does not exist. What the model is left free to get
        wrong is the choice among the patient's own live appointments, and the
        confirmation step is what stands under that.
        """
        run = self.run
        if run is None:
            return {"accepted": False, "problem": "There is no active request."}

        appointment = self.session.get(Appointment, appointment_id)
        # Ownership is checked as a refusal rather than a raise: this is a
        # model slip, not an attack, and it is recoverable. A patient's token
        # never reaches here — `patient_id` is bound from the session.
        if appointment is None or appointment.patient_id != self.patient_id:
            self.writer.validation(
                "appointment_change_proposal",
                accepted=False,
                detail={"appointment_id": appointment_id, "problem": "not this patient's"},
            )
            return {
                "accepted": False,
                "problem": (
                    f"Appointment {appointment_id} is not one of this patient's. "
                    "Call list_my_appointments and use an id from it."
                ),
            }

        if appointment.status not in LIVE_STATUSES:
            return {
                "accepted": False,
                "problem": (
                    f"That appointment is {appointment.status.value} and can no "
                    "longer be changed."
                ),
            }

        # If this run has *asked* which appointment, the answer has to be one of
        # the ones it asked about. The twin of the offered-slot check, and it
        # exists for the identical reason: an appointment id the model recalls
        # from its context is indistinguishable from one it invented, because
        # both arrive as an integer.
        #
        # Empty means no listing has been shown — the single-appointment
        # auto-target path — and imposes nothing.
        listed = listed_appointment_ids(run)
        if listed and appointment_id not in listed:
            self.writer.validation(
                "appointment_choice",
                accepted=False,
                detail={
                    "appointment_id": appointment_id,
                    "listed": listed,
                    "problem": "not one of the appointments offered as a choice",
                },
            )
            return {
                "accepted": False,
                "problem": (
                    f"Appointment {appointment_id} was not one of the choices "
                    "offered. Use one of the ids that was listed."
                ),
            }
        if listed:
            self.writer.validation(
                "appointment_choice",
                accepted=True,
                detail={"appointment_id": appointment_id, "listed": listed},
            )

        if action is ProposedAction.RESCHEDULE:
            found = get_slot(self.session, slot_id) if slot_id else {"found": False}
            if not found.get("found"):
                return {"accepted": False, "problem": f"Slot {slot_id} does not exist."}
            if not found.get("available"):
                return {
                    "accepted": False,
                    "problem": "That time is no longer available. Offer another.",
                }
            # A reschedule moves the time, not the department: the required
            # documents and the routing decision were settled when it was
            # booked, and its plan closes over neither of them.
            if found["department_id"] != appointment.department_id:
                self.writer.validation(
                    "appointment_change_proposal",
                    accepted=False,
                    detail={
                        "appointment_id": appointment_id,
                        "problem": "slot is in a different department",
                    },
                )
                return {
                    "accepted": False,
                    "problem": (
                        "That time is in a different department. Rescheduling "
                        "keeps the same department; book a new appointment "
                        "instead."
                    ),
                }

        self.writer.validation(
            "appointment_change_proposal",
            accepted=True,
            detail={"action": action.value, "appointment_id": appointment_id,
                    "slot_id": slot_id},
        )

        new_slot = get_slot(self.session, slot_id) if slot_id else None

        run.proposed_action = action
        run.proposed_appointment_id = appointment_id
        run.proposed_slot_id = slot_id
        self.proposals.proposed_slot_id = slot_id
        self.proposals.proposed_appointment_id = appointment_id

        if run.status is WorkflowStatus.IN_PROGRESS:
            transition(
                self.session,
                run,
                to=WorkflowStatus.PENDING_CONFIRMATION,
                trigger=f"{action.value}_proposed",
                writer=self.writer,
                actor=self.user,
                detail={"appointment_id": appointment_id, "slot_id": slot_id},
            )

        return {
            # What the patient holds now, re-read from the row — so the
            # sentence naming "exactly which appointment" is not assembled
            # from anything the model remembered.
            "accepted": True,
            "proposed": render_confirmation(self.session, appointment_id)["facts"],
            # And, for a reschedule, what they are being moved to.
            "new_slot": new_slot,
        }

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
                # Shape-stable with the real result. A refusal that invents its
                # own key names is how a consumer ends up reading a key nobody
                # writes and quietly seeing nothing missing, forever.
                return {
                    "required": [],
                    "satisfied": [],
                    "missing_mandatory": [],
                    "missing_optional": [],
                    "complete": False,
                    "problem": "No department has been decided for this request yet.",
                }
            return diff_required_documents(
                self.session, patient_id=self.patient_id, department_id=department_id
            )

        def record_missing_documents() -> dict:
            """Record what is still missing as a follow-up task, so it is not
            lost when this conversation ends."""
            return self._record_missing_documents()

        def list_unverified_documents_tool() -> dict:
            """List documents that have been uploaded but not yet checked
            against the type the patient declared them as."""
            return {
                "documents": list_unverified_documents(
                    self.session, patient_id=self.patient_id
                )
            }

        def read_document_text(document_id: int) -> dict:
            """Read the text of an uploaded PDF so you can see what it is.
            Images return no text: classify those by their declared type."""
            return self._read_document_text(document_id)

        def submit_document_verification(
            document_id: int, detected_type: str, matches: bool
        ) -> dict:
            """Say whether a document's content matches the type the patient
            declared it as. `detected_type` is what the content looks like.
            You propose; the status is set by code."""
            return self._submit_document_verification(
                document_id, detected_type, matches
            )

        list_patient_documents_tool.__name__ = "list_patient_documents"
        diff_required_documents_tool.__name__ = "diff_required_documents"
        list_unverified_documents_tool.__name__ = "list_unverified_documents"
        return [
            list_patient_documents_tool,
            list_unverified_documents_tool,
            read_document_text,
            submit_document_verification,
            diff_required_documents_tool,
            record_missing_documents,
        ]

    def _read_document_text(self, document_id: int) -> dict:
        """Extract text, but only from a document this patient owns.

        The ownership check is the whole reason this is bound rather than
        handed over raw: a document id is an integer, and a model that got
        creative with one would otherwise read another patient's file.
        """
        document = self.session.get(PatientDocument, document_id)
        if document is None or document.patient_id != self.patient_id:
            # Indistinguishable from a document that never existed — the same
            # rule the HTTP layer follows, for the same reason.
            return {"extracted": False, "reason": "not_found", "text": "", "pages": 0}
        return extract_document_text(self.session, document_id)

    def _submit_document_verification(
        self, document_id: int, detected_type: str, matches: bool
    ) -> dict:
        document = self.session.get(PatientDocument, document_id)
        if document is None or document.patient_id != self.patient_id:
            self.writer.validation(
                "document_verification",
                accepted=False,
                detail={"document_id": document_id, "problem": "not this patient's"},
            )
            return {"accepted": False, "problem": f"No document {document_id}."}

        self.writer.validation(
            "document_verification",
            accepted=True,
            detail={
                "document_id": document_id,
                "declared_type": document.declared_type,
                "detected_type": detected_type,
                "matches": bool(matches),
            },
        )
        verified = apply_verification(
            self.session,
            document_id=document_id,
            matches=bool(matches),
            detected_type=detected_type,
            note=(
                ""
                if matches
                else (
                    f"Content reads as {detected_type!r}, filed as "
                    f"{document.declared_type!r}."
                )
            ),
            actor=self.user,
        )
        return {"accepted": True, "document": verified}

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
        missing = diff.get("missing_mandatory") or []
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
            close_when_empty_key="missing",
            actor=self.user,
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
