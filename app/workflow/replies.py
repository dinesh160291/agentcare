"""What code says to the patient about a proposal, and after a commit.

Every other module here decides *what happens*. This one decides *what is
said* about the two moments where being wrong is expensive, and it exists
because two live transcripts showed the same failure in four shapes.

One receipt, one turn, four defects: a mandatory document and an optional one
mashed into "still needed, which is optional"; "recorded for follow-up"
standing beside "no outstanding follow-up tasks"; the *reminder's* fire date
presented as the appointment's, which sends a patient to an empty clinic a day
early; and a context dump nobody asked for. None of these is the model failing
to follow an instruction it could have followed. They are what happens when
facts pass through a paraphrase.

So the division is sharper here than elsewhere: around a proposal and after a
commit, **the model contributes no facts at all.** It may still speak — the
Coordinator's re-ask, the specialist's framing — but everything the patient
could act on is assembled below, from rows, at the moment of writing.

What this does *not* touch is the pinned rule. Nothing here reads a
confirmation, and nothing here commits. A better-worded re-ask is still a
re-ask: only an exact token or the button books anything.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Iterable, Sequence

from sqlalchemy.orm import Session

from app import clock
from app.models import (
    Appointment,
    AppointmentSlot,
    FollowUpTask,
    FollowUpTaskStatus,
    FollowUpTaskType,
    Reminder,
    ReminderStatus,
    WorkflowRun,
)
from app.tools.availability import get_slot
from app.tools.confirmations import render_confirmation
from app.tools.documents import diff_required_documents

#: The shortlist's length. The typed proposal is always exactly one slot — the
#: state machine is untouched by any of this — and the list beside it is a
#: choice, not a timetable. Twenty options is a wall.
MAX_OPTIONS = 3

#: Said once, after the document groups, whenever anything is missing. Item 6's
#: whole content: a patient told something is outstanding and not told where to
#: put it has been given a chore, not an instruction.
UPLOAD_POINTER = "You can upload documents on the Documents page."

REMINDER_LINE = "We'll remind you the day before."

#: What a patient is owed when their message asked for two appointment actions
#: and the system did one. "One request confirms one thing" is a rule with a
#: cost, and the cost has to be visible: live, "okay lets cancel that
#: appointment and book a new one for skin rash" booked, did not cancel, and
#: said nothing about the half it dropped.
ONE_VERB_AT_A_TIME = (
    "One change at a time — I'll start with {verb}; ask me about the other one "
    "right after."
)

#: The plan step in the patient's own words, for the line above.
VERB_WORDS = {
    "book": "the booking",
    "reschedule": "the reschedule",
    "cancel": "the cancellation",
}

#: Phrases where the assistant announces it is going off to do something. The
#: live case: "I will proceed to find a suitable time for you. Please hold on"
#: — said on a turn where a time was already being held, and followed by
#: nothing at all, because there was nothing left to do.
#:
#: Tuned to catch *announcements of departure*, not statements of the rule:
#: "I'll book it as soon as you say yes" is the rule and must survive. The
#: false-positive direction is cheap here — the fallback carries the same facts
#: and asks the same question — which is the opposite of the safety screen's
#: trade and worth saying out loud, because the two guards look alike.
_ACTION_PROMISES = (
    "hold on",
    "please hold",
    "one moment",
    "just a moment",
    "bear with",
    "i will proceed",
    "i'll proceed",
    "let me check",
    "let me find",
    "let me look",
    "let me see",
    "i will check",
    "i'll check",
    "i will find",
    "i'll find",
    "i will look",
    "i'll look",
    "i am looking",
    "i'm looking",
    "i am checking",
    "i'm checking",
    "i will get back",
    "i'll get back",
    "searching for",
)

#: Phrases that assert something about the schedule — in either direction. A
#: reply containing one of these is making a claim only a slot search can
#: support, and :func:`claims_availability` is checked *only* on turns where no
#: search ran, so a grounded reply never meets it whatever words it chose.
#:
#: The live sentence: "It appears that there are currently no available
#: appointment slots in the ENT department for next week" — written on a turn
#: whose only tool results were two refusals reading "No department has been
#: decided yet", and contradicted two turns later by a search that returned 72
#: slots in that exact window. A refusal to search is not an empty schedule,
#: and those are the two things this stops a paraphrase from merging.
_AVAILABILITY_CLAIMS = (
    "no available",
    "no availability",
    "not available",
    "no appointment slots",
    "no slots",
    "no free",
    "no times",
    "no openings",
    "nothing available",
    "fully booked",
    "are available",
    "is available",
    "slots available",
    "times available",
    "availability for",
    "have availability",
    "there are slots",
    "there is a slot",
)


# --- the offered set ------------------------------------------------------


def offered_slot_ids(run: WorkflowRun) -> list[int]:
    """Every slot id this run has actually shown the patient, in order."""
    return list((run.state or {}).get("offered_slot_ids") or [])


def record_offered(run: WorkflowRun, slots: Iterable[dict[str, Any]]) -> None:
    """Remember which slots were shown, from the tool payload that showed them.

    The ids are taken from ``find_available_slots``' own result and from
    nowhere else. That is the point: a model naming a slot id it recalls from
    its context window is indistinguishable from a model inventing one, since
    both arrive as an integer — so the set a re-proposal is checked against has
    to be born from a tool result, never from prose.

    It is the union over the whole run, never the last page: patients say
    "actually, the first one you showed me" three exchanges later, and a set
    that shrank would answer them with a refusal. Slots going stale in the
    meantime is a *liveness* problem, handled where a proposal is made, not by
    forgetting what was offered.
    """
    seen = offered_slot_ids(run)
    for slot in slots or []:
        slot_id = slot.get("slot_id")
        if slot_id is not None and slot_id not in seen:
            seen.append(int(slot_id))
    # Assigned, not mutated: SQLAlchemy's JSON column only notices assignment,
    # and a set that silently failed to persist would read from here as the
    # model inventing ids.
    state = dict(run.state or {})
    state["offered_slot_ids"] = seen
    run.state = state


def was_offered(run: WorkflowRun, slot_id: int) -> bool:
    """Has this run shown the patient this slot?"""
    return slot_id in offered_slot_ids(run)


def shortlist_slot_ids(run: WorkflowRun) -> list[int]:
    """The last numbered list, in the order the patient read it."""
    return list((run.state or {}).get("shortlist_slot_ids") or [])


def record_shortlist(run: WorkflowRun, slots: Iterable[dict[str, Any]]) -> None:
    """Remember the numbering, not just the membership.

    :func:`record_offered` is a *union over the run* and deliberately so: "the
    first one you showed me" arrives three exchanges later and a set that shrank
    would refuse it. That makes it the wrong thing to answer "option 2" with —
    a union has no second element, only a twentieth-of-twenty.

    So the ordered list is kept beside it, replaced each time a list is drawn.
    Rendering and recording are one step here for the same reason they are in
    :func:`render_appointment_choice`: a patient answering "2" is answering the
    list they were shown, and if the list is not code's the answer means
    nothing. The union stays the authority on *whether* a slot may be proposed;
    this only decides *which* slot a number refers to.

    Replaced rather than accumulated, because a number always refers to the most
    recent listing — the patient is reading the message in front of them.
    """
    state = dict(run.state or {})
    state["shortlist_slot_ids"] = [
        int(slot["slot_id"]) for slot in slots or [] if slot.get("slot_id") is not None
    ]
    run.state = state


# --- rendering ------------------------------------------------------------


def clock_time(moment: datetime) -> str:
    """A time as a patient says it: 9:00 AM, 3:00 PM.

    24-hour is how the rows are stored and how staff read a schedule; it is not
    how anyone answers "what time is my appointment". The card said 15:00 while
    the chat beside it said 3:00 PM, about the same slot, which reads as two
    appointments rather than as two notations. So there is one formatter and
    everything patient-facing goes through it — card, option list, re-ask,
    receipt.

    ``%-I`` is not portable to Windows and ``%#I`` is not portable off it, so
    the hour is done by hand.
    """
    hour = moment.hour % 12 or 12
    return f"{hour}:{moment:%M} {'AM' if moment.hour < 12 else 'PM'}"


def when_words(start: str | datetime) -> tuple[str, str]:
    """(day, time) as the patient reads them.

    One formatter, so the proposal card in the UI and the sentence in the chat
    cannot disagree about the same slot — which is the sort of difference a
    patient reads as two different appointments.
    """
    moment = start if isinstance(start, datetime) else datetime.fromisoformat(start)
    return f"{moment:%A} {moment.day} {moment:%B}", clock_time(moment)


_when = when_words


def render_options(
    slots: Sequence[dict[str, Any]], *, proposed_slot_id: int | None
) -> str:
    """The shortlist, numbered, soonest first, with the held one marked.

    Returns "" for no slots so a caller can concatenate unconditionally: an
    empty offer is the absence of a sentence, not an empty heading.
    """
    shortlist = list(slots or [])[:MAX_OPTIONS]
    if not shortlist:
        return ""

    lines = []
    previous: str | None = None
    for index, slot in enumerate(shortlist, start=1):
        day, time = _when(slot["start"])
        when = f"{day} at {time}"
        # Two doctors free at the same moment is ordinary — Neurology has two —
        # but a list repeating the identical day and time on consecutive lines
        # reads as a data error rather than as a choice of clinician. Saying so
        # explicitly is the difference between "the system is broken" and "you
        # can see either of them".
        if when == previous:
            lines.append(f"{index}. same time with {slot['doctor_name']}")
        else:
            lines.append(f"{index}. {when} with {slot['doctor_name']}")
        previous = when
        if slot.get("slot_id") == proposed_slot_id:
            lines[-1] += " — I'm holding this one for you"
    return "\n".join(lines)


def _shortlist(slots: Sequence[dict], *, held: int | None, first: bool) -> list[dict]:
    """The three that will be shown, held one first or excluded entirely.

    One slot per start time. A department with three doctors free at 09:00
    returns three slots at 09:00, and offering those as options one to three is
    not a choice — it is the same appointment three times, and it hides the
    09:30 the patient would actually have picked. What is being chosen here is
    a *time*; which doctor holds it is the schedule's business.
    """
    mine = [slot for slot in slots or [] if slot.get("slot_id") == held]
    others = [slot for slot in slots or [] if slot.get("slot_id") != held]
    ordered = (mine + others) if first else others

    shortlist: list[dict] = []
    duplicates: list[dict] = []
    seen_times: set[str] = set()
    for slot in ordered:
        start = str(slot.get("start"))
        if start in seen_times:
            duplicates.append(slot)
            continue
        seen_times.add(start)
        shortlist.append(slot)
        if len(shortlist) == MAX_OPTIONS:
            return shortlist

    # Distinct times are the choice worth offering, and they come first. Only
    # when the department has fewer than three of them is a same-moment
    # alternative worth a line — at that point it is a choice of clinician,
    # which is a real thing to offer, rather than the same appointment padding
    # the list out. ``render_options`` labels those "same time with Dr X" so a
    # repeated day and time reads as two doctors and not as a broken query.
    for slot in duplicates:
        shortlist.append(slot)
        if len(shortlist) == MAX_OPTIONS:
            break
    return sorted(shortlist, key=lambda s: (s.get("slot_id") != held, str(s["start"])))


def render_proposal(session: Session, run: WorkflowRun, slots: Sequence[dict]) -> str:
    """The reply that offers a time: the shortlist, then how to answer.

    The typed proposal is still exactly one slot and the state machine has not
    moved an inch. This changes only what the patient is *told* — that there
    were others, and that asking for one is a thing they may do — which is what
    turns "no" from a dead end into a choice.

    Records what it shows, because showing is what makes an id answerable
    later. Rendering and recording are one step on purpose: a path that
    rendered without recording would offer the patient a slot the re-proposal
    guard would then refuse, which is worse than not offering it.
    """
    shortlist = _shortlist(slots, held=run.proposed_slot_id, first=True)
    options = render_options(shortlist, proposed_slot_id=run.proposed_slot_id)
    if not options:
        return ""
    record_offered(run, shortlist)
    record_shortlist(run, shortlist)
    return (
        "Here are the earliest times I can offer:\n"
        f"{options}\n\n"
        'Say "yes" to take the one I\'m holding, or tell me which of the '
        "others you'd like."
    )


def render_alternatives(
    session: Session, run: WorkflowRun, slots: Sequence[dict]
) -> str:
    """Answer "what else is there?" without disturbing what is held.

    The whole of item 1 in one function: the patient asked a question, so they
    get an answer — and the proposal they have not answered yet is still
    theirs. The re-ask follows so the turn ends where it began, on a question
    the patient can settle.
    """
    shortlist = _shortlist(slots, held=run.proposed_slot_id, first=False)
    options = render_options(shortlist, proposed_slot_id=None)
    if not options:
        return (
            "I couldn't find any other free times in that department. "
            + render_reask(session, run)
        )
    record_offered(run, shortlist)
    record_shortlist(run, shortlist)
    return (
        f"Other times that are free:\n{options}\n\n" + render_reask(session, run)
    )


def listed_appointment_ids(run: WorkflowRun) -> list[int]:
    """Every appointment this run has actually listed for the patient to choose."""
    return list((run.state or {}).get("listed_appointment_ids") or [])


def was_listed(run: WorkflowRun, appointment_id: int) -> bool:
    """Has this run offered the patient this appointment as a choice?"""
    return appointment_id in listed_appointment_ids(run)


def render_appointment_choice(
    session: Session, run: WorkflowRun, appointments: Sequence[dict[str, Any]], *, verb: str
) -> str:
    """Ask which appointment, numbering them — and record what was numbered.

    The twin of :func:`render_proposal`, for the same reason and with the same
    rule: **rendering and recording are one step.** A patient who answers "2"
    is answering the list they were shown, so the list has to be code's or the
    answer means nothing. Live it was the model's — it wrote its own numbered
    listing in prose, and nothing anywhere knew what "2" referred to.

    ``listed_appointment_ids`` is the appointment-side ``offered_slot_ids``,
    born from the rows this function renders and from nowhere else, because an
    appointment id the model recalls is indistinguishable from one it invented:
    both arrive as an integer.

    Returns "" for fewer than two candidates — with one appointment there is
    nothing to choose between, and asking would be a worse answer than acting.
    """
    rows = list(appointments or [])
    if len(rows) < 2:
        return ""

    lines = []
    for index, row in enumerate(rows, start=1):
        when = ""
        if row.get("start"):
            day, time = _when(row["start"])
            when = f" on {day} at {time}"
        lines.append(
            f"{index}. {row.get('department_name')} with {row.get('doctor_name')}"
            f"{when} ({row.get('reference_code')})"
        )

    state = dict(run.state or {})
    state["listed_appointment_ids"] = [int(row["appointment_id"]) for row in rows]
    run.state = state

    action = "cancel" if verb == "cancel" else "move"
    return (
        f"You have more than one upcoming appointment, so I want to be sure "
        f"which one to {action}:\n"
        + "\n".join(lines)
        + "\n\nTell me the number, or name the department. Nothing has changed yet."
    )


def render_change_proposal(session: Session, run: WorkflowRun) -> str:
    """What the patient is asked before a reschedule or a cancellation.

    A reschedule proposal has **two** times in it — the one they have and the
    one being offered — and they are the whole content of the question. The
    model was given both, in one tool payload, and welded them: "I found your
    appointment with Dr. Deepa Krishnan in the ENT department on Monday, 3
    August 2026, at 9:00 AM. Would you like me to reschedule it?" The 9:00 was
    the *new* slot. The appointment was at 10:00, and the sentence names no new
    time at all — so a patient reading it is being asked to approve a move to
    the time they already have.

    That is not a wording that could have been better. Two facts of the same
    shape, in one sentence, is exactly what a paraphrase loses — the same
    failure as the receipt printing the reminder's date as the visit's. So the
    facts are assembled here, from the appointment row and the slot row, and
    the specialist's sentence is not used.

    Returns "" when there is no change proposal on the run, so callers can use
    it as a test as well as a renderer.
    """
    action = run.proposed_action
    if action is None or action.value not in ("reschedule", "cancel"):
        return ""
    if not run.proposed_appointment_id:
        return ""

    appointment = session.get(Appointment, run.proposed_appointment_id)
    if appointment is None or appointment.slot is None:
        return ""

    facts = render_confirmation(session, appointment.id)["facts"]
    current_day, current_time = _when(appointment.slot.start_time)
    # The reference code is not decoration. Story 20: a destructive action is
    # never ambiguous, and a patient with two appointments has to be able to
    # tell from the sentence alone which one is about to move or disappear.
    who = (
        f"{facts['department_name']} appointment with {facts['doctor_name']} "
        f"({facts['reference_code']})"
    )

    if action.value == "cancel":
        return (
            f"Your {who} is on {current_day} at {current_time}. "
            'Say "yes" and I\'ll cancel it — nothing changes until you do.'
        )

    new_slot = (
        session.get(AppointmentSlot, run.proposed_slot_id)
        if run.proposed_slot_id
        else None
    )
    if new_slot is None:
        return ""
    new_day, new_time = _when(new_slot.start_time)
    moved = (
        f"the same day at {new_time}"
        if new_day == current_day
        else f"{new_day} at {new_time}"
    )
    return (
        f"Your {who} is currently {current_day} at {current_time}. "
        f"I can move it to {moved}. "
        'Say "yes" and I\'ll make the change — nothing moves until you do.'
    )


def _proposed_facts(session: Session, run: WorkflowRun) -> str | None:
    """The held slot, as a sentence. ``None`` if there is nothing held."""
    if not run.proposed_slot_id:
        return None
    found = get_slot(session, run.proposed_slot_id)
    if not found.get("found"):
        return None
    slot = session.get(AppointmentSlot, run.proposed_slot_id)
    day, time = _when(slot.start_time)
    return f"{found['doctor_name']}, {found['department_name']}, {day} at {time}"


def render_reask(session: Session, run: WorkflowRun) -> str:
    """Ask again, carrying the facts and claiming nothing.

    Two failures at once. The live re-ask promised an action it never took
    ("I will proceed to find a suitable time... please hold on") *and* restated
    none of the proposal, so a patient who had lost the thread was asked to
    confirm something the reply did not name.

    It deliberately does not open with "that sounds like a yes". Reading assent
    is exactly the judgement this state exists to refuse, and the detector that
    would be needed fails on the sentence that prompted the whole item: "looks
    good. lets book that time" contains no confirmation token at all. So the
    re-ask states what is held and what would settle it, for every non-answer
    alike, and reads the patient's intent nowhere.
    """
    facts = _proposed_facts(session, run)
    action = (run.proposed_action.value if run.proposed_action else "book")
    verb = {"reschedule": "move it", "cancel": "cancel it"}.get(action, "book it")

    if facts is None:
        return (
            "Nothing is booked yet. Tell me a time that suits you and I'll "
            "check what's free."
        )
    return (
        f"The time I'm holding is {facts}. "
        f'I can only {verb} on an exact "yes" or the ✅ Confirm button — '
        f"shall I {verb}?"
    )


def promises_action(text: str) -> bool:
    """Does this text announce that the assistant is off doing something?

    A re-ask that says so is describing a turn that is already over: nothing
    runs after the reply. The patient waits for something that will not arrive
    and then repeats themselves, which is the stall the counter then punishes.
    """
    lowered = re.sub(r"[’']", "'", (text or "").lower())
    return any(phrase in lowered for phrase in _ACTION_PROMISES)


def claims_availability(text: str) -> bool:
    """Does this text assert what is or is not free?

    The companion to :func:`promises_action`, and the same shape of guard with
    the same cheap false-positive direction — a rejected reply falls back to a
    template carrying the facts. The difference is that this one is only ever
    consulted when **no slot search ran this turn**, so it is not really a
    judgement about wording: a claim about the schedule with nothing behind it
    is wrong however it is phrased.
    """
    lowered = re.sub(r"[’']", "'", (text or "").lower())
    return any(phrase in lowered for phrase in _AVAILABILITY_CLAIMS)


# --- the receipt ----------------------------------------------------------


def _document_lines(session: Session, run: WorkflowRun) -> list[str]:
    """The three groups, each named for what the patient must do about it.

    Separate lines, never one clause. The live receipt's "still needed, which
    is optional" is two different obligations welded together, and a patient
    cannot act on it in either direction.
    """
    department_id = (run.state or {}).get("department_id")
    if department_id is None:
        return []

    diff = diff_required_documents(
        session, patient_id=run.patient_id, department_id=int(department_id)
    )
    if not diff["required"]:
        # General Medicine asks for nothing, so the receipt said nothing, and a
        # patient read the silence as an omission — then went looking for the
        # paperwork section that was never coming. This is the *opposite* case
        # from the documents listing's careful silence: there, no department had
        # been resolved and "nothing is required" would have been a claim about
        # a hospital nobody had consulted. Here the department is known and the
        # rules table is empty, so the sentence is a fact about this visit.
        return ["No documents are needed for this visit."]

    lines = []
    if diff["satisfied"]:
        lines.append(f"Already on file: {', '.join(diff['satisfied'])}.")
    if diff["missing_mandatory"]:
        lines.append(
            f"Required before your visit: {', '.join(diff['missing_mandatory'])}."
        )
    if diff["missing_optional"]:
        lines.append(f"Optional but helpful: {', '.join(diff['missing_optional'])}.")
    if diff["missing_mandatory"] or diff["missing_optional"]:
        lines.append(UPLOAD_POINTER)
    return lines


def render_outstanding(session: Session, *, patient_id: int) -> str:
    """What is still outstanding, read from the task rows themselves.

    The document listing answers "what do I have on file?", and the half of
    that answer about what is still *needed* was the model's. Live, a query run
    that carried no department was answered with "no documents required" while
    an open task for a missing MRI report sat in the database — and the next
    turn said so, contradicting it.

    The rows are the authority precisely because the diff is not available in
    that situation: a listing with no department has nothing to diff against,
    and "nothing to compare" is not the same fact as "nothing is required".
    The open task is what survives across turns and across runs, so it is what
    the sentence is built from.

    Returns "" when nothing is outstanding — the absence of a requirement is
    not a sentence, and asserting it is how the contradiction happened.
    """
    tasks = (
        session.query(FollowUpTask)
        .filter(
            FollowUpTask.patient_id == patient_id,
            FollowUpTask.task_type == FollowUpTaskType.MISSING_DOCUMENTS,
            FollowUpTask.status == FollowUpTaskStatus.OPEN,
        )
        .order_by(FollowUpTask.id)
        .all()
    )
    items: list[str] = []
    for task in tasks:
        for item in (task.details or {}).get("missing", []):
            if item not in items:
                items.append(item)
    if not items:
        return ""
    return f"Still needed for your visit: {', '.join(items)}. {UPLOAD_POINTER}"


def _has_pending_reminder(session: Session, appointment_id: int) -> bool:
    return (
        session.query(Reminder)
        .filter(
            Reminder.appointment_id == appointment_id,
            Reminder.status == ReminderStatus.PENDING,
        )
        .count()
        > 0
    )


def render_receipt(session: Session, run: WorkflowRun) -> str:
    """What the patient is told after a commit. Assembled, never paraphrased.

    Three parts, each omitted when it has nothing to say:

    (a) the appointment itself, from ``render_confirmation``, which re-reads
        the row — so a reschedule states the new time because the row holds it,
        not because anything remembered it;
    (b) the documents, in three separately-named groups;
    (c) one sentence about reminders, and only if a reminder row exists.

    (c) is the smallest part and the one that has already been wrong twice. The
    old receipt promised a reminder unconditionally, so a *cancellation* — which
    retires the reminder in the same transaction — promised one that had just
    been deleted; and the live receipt printed the reminder's fire date as
    though it were the visit. Hence: never a date from that table, and never
    the sentence without the row. Every fact re-read, including the ones that
    look like boilerplate.
    """
    state = run.state or {}
    appointment_id = state.get("appointment_id")
    if appointment_id is None:
        return ""

    appointment = session.get(Appointment, int(appointment_id))
    if appointment is None:
        return ""

    parts = [render_confirmation(session, appointment.id)["sentence"]]

    # A cancelled visit is not one to prepare for, so the document groups are
    # about nothing. Read from what actually committed rather than from the
    # plan: the plan says what was intended, the row says what happened.
    if state.get("committed_action") in (None, "book", "reschedule"):
        parts.extend(_document_lines(session, run))

    if _has_pending_reminder(session, appointment.id):
        parts.append(REMINDER_LINE)
    elif appointment.slot is not None and appointment.slot.start_time.date() == clock.today():
        # A same-day booking has no day before to be reminded on. Saying so is
        # not a smaller version of the reminder promise — it is the fact the
        # patient actually needs, and the old line promised a message that
        # could never arrive for a visit only hours away.
        parts.append(
            f"That's today at {clock_time(appointment.slot.start_time)}, so there's "
            "no reminder to send — please come along at that time."
        )

    # Blank-line separated, because these are the facts the patient acts on and
    # markdown collapses a single newline. ``_document_lines`` goes to some
    # trouble to keep a mandatory document and an optional one in separate
    # sentences; joining them with one newline handed them back to the renderer
    # to weld into a paragraph, which is most of the way back to the "still
    # needed, which is optional" clause that started all of this.
    return "\n\n".join(part for part in parts if part)


__all__ = [
    "MAX_OPTIONS",
    "ONE_VERB_AT_A_TIME",
    "REMINDER_LINE",
    "UPLOAD_POINTER",
    "VERB_WORDS",
    "listed_appointment_ids",
    "offered_slot_ids",
    "promises_action",
    "record_offered",
    "record_shortlist",
    "shortlist_slot_ids",
    "render_alternatives",
    "render_appointment_choice",
    "render_change_proposal",
    "render_options",
    "render_outstanding",
    "render_proposal",
    "render_reask",
    "render_receipt",
    "was_listed",
    "was_offered",
    "clock_time",
    "when_words",
]
