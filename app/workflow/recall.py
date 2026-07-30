"""What the patient asked for that this system did not do.

A short, bounded memory of unfinished business, consulted at exactly one
moment: when a fresh-turn message has failed the scope gate and the honest
answer is otherwise "tell me more" about something the patient has already
told us.

**The live failure.** The one-verb rule tells a patient "I'll start with the
cancellation; ask me about the other one right after" — and nothing anywhere
recorded what the other one *was*. So "now do the other thing I asked" named no
domain subject, met the generic refusal, and the rephrase that followed ("the
other booking request") gave routing nothing to grip: "booking request" is not a
department, so it resolved to General Medicine with low confidence and queued
for a human. The patient had said the words once and was asked to say them
again, worse, twice.

Two suppliers, one mechanism:

* a **dropped verb**, written where the one-verb line is written, because that
  is the moment this system knowingly declines half a request;
* a **closed run**, which needs no writing at all — the run row already holds
  the request text, the plan and the department, and a fresh turn only ever
  happens when every run of the patient's is terminal. Deriving it beats storing
  it twice.

Three rules keep this in the deterministic bin:

1. **It only ever changes what a refusal offers.** No plan is built from memory
   and no run is started from it without an exact "yes" — the same token reader
   that guards a booking, for the same reason.
2. **It is consulted only for a message that points backwards.** The cue list is
   what keeps "who won the fifa final" answered as off-topic while a completed
   booking sits in the same conversation: a memory that answered every refusal
   would turn one closed run into a standing offer.
3. **It expires.** A dropped verb from twenty turns ago is not a pending
   request, and an offer is answerable on the very next turn only — "yes" five
   messages later is an answer to something else.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models import Appointment, PlanStep, TraceEvent, WorkflowRun, WorkflowStatus
from app.tools import resolve_department
from app.workflow.mapping import primary_intent

#: How far back a memory reaches, counted in turns of this conversation. Ten is
#: the same order as the transcript window the Coordinator gets: far enough that
#: "the other thing I asked" still means something, short enough that it cannot
#: resurface after the patient has moved on.
MEMORY_WINDOW_TURNS = 10

#: The verb as an errand, for the sentence that offers it.
_AS_ERRAND = {
    PlanStep.BOOK: "booking",
    PlanStep.RESCHEDULE: "rescheduling",
    PlanStep.CANCEL: "cancelling",
}

#: The verb as a thing, for the sentence that offers to restart it.
_AS_THING = {
    PlanStep.BOOK: "booking",
    PlanStep.RESCHEDULE: "reschedule",
    PlanStep.CANCEL: "cancellation",
}

#: Runs worth offering to start again. A failed, rejected or escalated run is
#: **not** here: the first two are this system's problem to explain rather than
#: the patient's to re-request, and an escalated one is with a person — offering
#: to restart it from memory is the second door into the thing that makes
#: ``escalated`` terminal.
_RESTARTABLE = (WorkflowStatus.COMPLETED, WorkflowStatus.CANCELLED)

#: What a refusal becomes when there is something to offer. Two sentences,
#: because the two facts are different: one request was never started, the other
#: was started and closed, and telling a patient their live request is waiting
#: when it was cancelled would be worse than the generic line it replaces.
DROPPED_OFFER = (
    'The other request was {label} — shall I start it? Say "yes" and I will.'
)
RESTART_OFFER = (
    'Your last request was {label}, which I closed — shall I restart it? '
    'Say "yes" and I will.'
)


@dataclass(frozen=True)
class Remembered:
    """A request this system was told about and did not carry out."""

    #: The plan step to build a canonical plan from, if the patient says yes.
    verb: PlanStep
    #: What routing should read. The **department name** where the patient's own
    #: words resolved to one, and the words themselves where they did not — so
    #: the restarted run routes with the confidence the original sentence
    #: earned, rather than handing the router "the other booking request".
    subject: str
    #: How the offer names it.
    label: str
    #: ``"dropped"`` or ``"closed"``. Chooses the sentence, and nothing else.
    kind: str
    #: The run the memory hangs off, and the row the offer is marked on.
    run_id: int

    @property
    def reply(self) -> str:
        template = DROPPED_OFFER if self.kind == "dropped" else RESTART_OFFER
        return template.format(label=self.label)


def _department_named(session: Session, text: str, *, besides: int | None = None) -> str:
    """The department the patient's own words resolve to, or "".

    ``resolve_department`` and nothing else, so the Department table decides
    what a subject is here exactly as it does in routing. There is no second
    keyword list to drift from the first.

    ``besides`` is the department the *kept* half of the request already used, and
    it is what makes a two-verb sentence readable. "Book me a cardiology
    appointment and cancel my dermatology one" names two desks, so the resolver
    correctly reports ambiguity and nothing was stored — the offer said "booking
    an appointment", and the "yes" that accepted it routed the whole two-verb
    sentence, which resolves to nothing but ambiguity again and queued for a
    human at low confidence.

    Subtracting is not guessing. The cancel run *acted on* Dermatology, so that
    candidate is accounted for; if exactly one is left, the sentence named it and
    the dropped half is about it. Two left and this still stores nothing, because
    a three-department message genuinely does not say.
    """
    named = resolve_department(session, text or "")
    if named.get("status") == "resolved":
        return str(named["department"]["name"])
    if named.get("status") != "ambiguous" or besides is None:
        return ""
    remaining = [
        candidate
        for candidate in named.get("candidates") or []
        if int(candidate["id"]) != int(besides)
    ]
    if len(remaining) == 1:
        return str(remaining[0]["name"])
    return ""


def _run_department_id(session: Session, run: WorkflowRun) -> int | None:
    """Which desk the run that *did* happen was about.

    Read in the same order :meth:`~app.agents.toolbelt.Toolbelt._target_appointment`
    reads it, and for the same reason: a booking run keeps its department in
    state, while a reschedule or cancel run never routed and its department is a
    fact about the appointment being changed. Ownership is re-checked because an
    id on a run is still an id.
    """
    state = run.state or {}
    if state.get("department_id"):
        return int(state["department_id"])
    for candidate in (
        run.proposed_appointment_id,
        state.get("chosen_appointment_id"),
        state.get("appointment_id"),
    ):
        if not candidate:
            continue
        appointment = session.get(Appointment, int(candidate))
        if appointment is not None and appointment.patient_id == run.patient_id:
            return appointment.department_id
    return None


def _session_turns(session: Session, *, session_id: str) -> list[tuple[str, set[int]]]:
    """This conversation's turns in order, each with the runs it touched.

    Read from the trace because a fresh turn writes nothing else — the same
    reason ``_consecutive_clarifies`` counts from there. One pass answers both
    questions this module asks: whether a stamp is still inside the window, and
    which turn a closed run was last part of.
    """
    rows = (
        session.query(TraceEvent.turn_id, TraceEvent.workflow_run_id)
        .filter(TraceEvent.session_id == session_id)
        .order_by(TraceEvent.id)
        .all()
    )
    ordered: dict[str, set[int]] = {}
    for turn_id, run_id in rows:
        if turn_id is None:
            continue
        touched = ordered.setdefault(turn_id, set())
        if run_id is not None:
            touched.add(int(run_id))
    return list(ordered.items())


def remember_dropped_verb(
    session: Session,
    run: WorkflowRun,
    *,
    dropped: set[PlanStep],
    message: str,
    turn_id: str,
) -> None:
    """Record the half of a two-verb request that was not done.

    Written on the run that *did* happen, which is where the patient's sentence
    lives and where the one-verb line was owed. Nothing reads it except a
    refusal, and only inside the window.
    """
    verb = next((step for step in _AS_ERRAND if step in dropped), None)
    if verb is None:
        return

    department = _department_named(
        session, message, besides=_run_department_id(session, run)
    )
    state = dict(run.state or {})
    state["dropped_request"] = {
        "verb": verb.value,
        "department": department,
        "text": message,
        "turn_id": turn_id,
    }
    run.state = state
    session.flush()


def _from_dropped(run: WorkflowRun, stored: dict) -> Remembered | None:
    try:
        verb = PlanStep(stored.get("verb"))
    except ValueError:  # pragma: no cover - written from the enum
        return None
    department = stored.get("department") or ""
    errand = _AS_ERRAND[verb]
    label = (
        f"{errand} a {department} appointment"
        if department
        else f"{errand} an appointment"
    )
    return Remembered(
        verb=verb,
        subject=department or str(stored.get("text") or ""),
        label=label,
        kind="dropped",
        run_id=run.id,
    )


def _from_closed(session: Session, run: WorkflowRun) -> Remembered | None:
    verb = primary_intent(list(run.plan or []))
    if verb is None or verb not in _AS_THING:
        return None
    department = str((run.state or {}).get("department_name") or "") or (
        _department_named(session, run.request_text or "")
    )
    thing = _AS_THING[verb]
    label = f"the {department} {thing}" if department else f"the {thing}"
    return Remembered(
        verb=verb,
        subject=department or (run.request_text or ""),
        label=label,
        kind="closed",
        run_id=run.id,
    )


def recall_request(
    session: Session, *, patient_id: int, session_id: str, turn_id: str
) -> Remembered | None:
    """The most recent unfinished request in this conversation, or ``None``.

    Most recent first at run granularity, and within one run a dropped verb
    outranks the run itself: the dropped half was never attempted, so it is the
    more specific answer to "the other thing I asked".

    ``None`` is the whole of the old behaviour. A conversation with no memory
    gets exactly the reply it got before this existed, which is what keeps the
    generic refusal falsifiable.
    """
    turns = _session_turns(session, session_id=session_id)
    order = [previous for previous, _ in turns if previous != turn_id]
    recent = set(order[-MEMORY_WINDOW_TURNS:])
    last_turn_of: dict[int, str] = {}
    for previous, touched in turns:
        for run_id in touched:
            last_turn_of[run_id] = previous

    runs = (
        session.query(WorkflowRun)
        .filter(WorkflowRun.patient_id == patient_id)
        .filter(WorkflowRun.session_id == session_id)
        .order_by(WorkflowRun.id.desc())
        .all()
    )
    for run in runs:
        stored = (run.state or {}).get("dropped_request")
        if stored and stored.get("turn_id") in recent:
            remembered = _from_dropped(run, stored)
            if remembered is not None:
                return remembered
        if run.status in _RESTARTABLE and last_turn_of.get(run.id) in recent:
            remembered = _from_closed(session, run)
            if remembered is not None:
                return remembered
    return None


def mark_offer(
    session: Session, remembered: Remembered, *, turn_id: str
) -> None:
    """Record that this exact request was offered, on the run it belongs to.

    On a row rather than in the trace, and carrying the payload rather than a
    flag: the turn that reads a "yes" must start the thing that was *offered*,
    not whatever the memory would produce a second time.
    """
    run = session.get(WorkflowRun, remembered.run_id)
    if run is None:  # pragma: no cover - the run was just read
        return
    state = dict(run.state or {})
    state["restart_offer"] = {
        "turn_id": turn_id,
        "verb": remembered.verb.value,
        "subject": remembered.subject,
        "label": remembered.label,
        "kind": remembered.kind,
    }
    run.state = state
    session.flush()


def pending_offer(
    session: Session, *, patient_id: int, session_id: str, turn_id: str
) -> Remembered | None:
    """The offer this message could be answering, or ``None``.

    Answerable on the **very next turn** and no later. A wider rule would let a
    "yes" meant for something else start a request the patient may already have
    declined — and "yes" is the one word in this system that must never mean
    something the patient did not just agree to.
    """
    turns = _session_turns(session, session_id=session_id)
    order = [previous for previous, _ in turns if previous != turn_id]
    if not order:
        return None
    previous_turn = order[-1]

    runs = (
        session.query(WorkflowRun)
        .filter(WorkflowRun.patient_id == patient_id)
        .filter(WorkflowRun.session_id == session_id)
        .order_by(WorkflowRun.id.desc())
        .all()
    )
    for run in runs:
        stored = (run.state or {}).get("restart_offer")
        if not stored or stored.get("turn_id") != previous_turn:
            continue
        try:
            verb = PlanStep(stored.get("verb"))
        except ValueError:  # pragma: no cover - written from the enum
            continue
        return Remembered(
            verb=verb,
            subject=str(stored.get("subject") or ""),
            label=str(stored.get("label") or ""),
            kind=str(stored.get("kind") or "closed"),
            run_id=run.id,
        )
    return None


def clear_offer(session: Session, remembered: Remembered) -> None:
    """Spend the memory the moment it is acted on.

    Both keys go: the offer, so one "yes" cannot start two runs, and the dropped
    verb, because it has now been done. Boundedness — an automated writer with
    no growth rule is the one thing this project does not ship.
    """
    run = session.get(WorkflowRun, remembered.run_id)
    if run is None:  # pragma: no cover - the run was just read
        return
    state = dict(run.state or {})
    state.pop("restart_offer", None)
    state.pop("dropped_request", None)
    run.state = state
    session.flush()


__all__ = [
    "DROPPED_OFFER",
    "MEMORY_WINDOW_TURNS",
    "RESTART_OFFER",
    "Remembered",
    "clear_offer",
    "mark_offer",
    "pending_offer",
    "recall_request",
    "remember_dropped_verb",
]
