"""Which appointment the patient meant — decided from their words, in code.

The appointment-side twin of :mod:`app.workflow.selection`, and it exists for a
worse reason. A selection that reads wrongly costs one decline. A *target* that
reads wrongly moves the wrong appointment, and the patient is told the right one
moved.

**The live failure.** With a Monday Dermatology appointment and a Thursday
Orthopedics one on file, "I wanted to reschedule the appointment on Monday to
Wednesday" reached ``propose_reschedule`` as ``appointment_id: 3`` — the
Thursday one. Every deterministic guard around it did its job: the id was this
patient's, the appointment was live, the new slot was free and in the same
department, and the proposal was recorded correctly. Nothing anywhere compared
the model's pick against the word "Monday". The patient confirmed, the Thursday
Orthopedics appointment moved, and two turns later a booking failed with "you
already have an appointment at that time" — about a Monday morning the patient
still believed they had given up.

So the model's argument stops being the last word. Code reads the cues the
patient actually gave, matches them against the rows, and one of three things is
true:

* **exactly one appointment matches** — that is the target, and a differing
  argument is overridden rather than argued with;
* **nothing in the message points at an appointment, or the cues point at
  several** — nobody knows which, so the run asks, which is what
  ``render_appointment_choice`` is for;
* **there is only one live appointment** — there was never a choice to get
  wrong, and this decides nothing.

**Why matching is over rows, not over cues.** "Reschedule the appointment on
Monday to Wednesday" names two weekdays. Only one of them is an appointment the
patient has; the other is where they want to go. Counting *matched appointments*
rather than *matched phrases* dissolves that without a rule about destinations:
Wednesday matches nothing, so it contributes nothing. Two appointments on the
same Monday genuinely is ambiguous, and there the count is two and the run asks.

The cues are deliberately the four a patient actually uses to point at an
appointment they already have — the day of the week, the date, the department,
and the reference code we gave them. A time of day is not among them: "the 9am
one" names a slot far more often than an appointment, and the slot reader owns
that sentence.

**And when the run has already asked.** :func:`read_choice` is the other half:
once ``render_appointment_choice`` has numbered the rows, "2" is an answer to
that list and needs no cues at all. It had no reader for a whole round — the
list was drawn, the number came back, and the turn went to the Coordinator,
which called it a new request. Both readers answer the same question from
different evidence, so they live together; which one applies is decided by
whether a list is outstanding.

:func:`narrow_by_department` is the third, and it exists because the list's own
closing sentence offers the patient two ways to answer and only one of them
could be heard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Sequence

from sqlalchemy.orm import Session

from app import clock
from app.tools.dates import MONTHS, WEEKDAYS
from app.tools.departments import resolve_department
from app.workflow.mapping import names_appointment_verbs, says_withdrawal
from app.workflow.selection import position_named

_MONTH_NAMES = "|".join(MONTHS)
_WEEKDAY_NAMES = "|".join(sorted(WEEKDAYS, key=len, reverse=True))

_WEEKDAY = re.compile(rf"\b({_WEEKDAY_NAMES})\b", re.IGNORECASE)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAMES})\b", re.IGNORECASE)
_MONTH_DAY = re.compile(rf"\b({_MONTH_NAMES})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.IGNORECASE)

#: "AC-000002", and the same code typed without its punctuation.
_REFERENCE = re.compile(r"\b([A-Za-z]{1,4})[- ]?(\d{3,})\b")


@dataclass(frozen=True)
class TargetVerdict:
    """Which appointment the message points at, and how that was decided."""

    #: The appointment to act on, or ``None`` when the message does not say.
    appointment_id: int | None
    #: ``only_one`` | ``matched`` | ``no_cue`` | ``ambiguous`` — the trace's word
    #: for why, so a reviewer can tell a silent pass from a real decision.
    reason: str
    #: The cue phrases that were read, for the same reason.
    cues: list[str]
    #: Every appointment the cues pointed at. One entry means a decision.
    candidates: list[int]


def _normalise_code(value: str) -> str:
    return re.sub(r"[^0-9a-z]", "", (value or "").lower()).lstrip("0")


def _dates_named(text: str, *, today: date) -> set[date]:
    """Every calendar date this message states outright.

    All of them, not the first: a message may name the date it is moving *from*
    and the date it is moving *to*, and which is which is decided by what the
    patient actually has booked, not by word order.

    Numeric ``3/8`` forms stay unsupported here for the reason ``resolve_date``
    refuses them: they mean two different days on either side of the Atlantic.
    """
    found: set[date] = set()
    for year, month, day in _ISO_DATE.findall(text or ""):
        try:
            found.add(date(int(year), int(month), int(day)))
        except ValueError:
            continue

    pairs: list[tuple[int, int]] = [
        (int(day), MONTHS[month.lower()]) for day, month in _DAY_MONTH.findall(text or "")
    ]
    pairs += [
        (int(day), MONTHS[month.lower()]) for month, day in _MONTH_DAY.findall(text or "")
    ]
    for day, month in pairs:
        # A bare day and month has no year on it. Try this one and the next, and
        # take the first that has not already passed — "3 January" said in
        # August means the January that is coming.
        for year in (today.year, today.year + 1):
            try:
                candidate = date(year, month, day)
            except ValueError:
                break
            if candidate >= today:
                found.add(candidate)
                break
    return found


def _weekdays_named(text: str) -> set[int]:
    return {WEEKDAYS[match.lower()] for match in _WEEKDAY.findall(text or "")}


def _codes_named(text: str) -> set[str]:
    return {
        _normalise_code(f"{prefix}{digits}") for prefix, digits in _REFERENCE.findall(text or "")
    }


def resolve_target(
    session: Session,
    *,
    message: str,
    appointments: Sequence[dict[str, Any]],
    today: date | None = None,
) -> TargetVerdict:
    """The appointment this message points at, among the ones the patient has.

    ``appointments`` is ``list_patient_appointments(live_only=True)``' own
    payload, so the rows are already this patient's and already changeable —
    this reads words and nothing else, and it never widens what may be acted on.
    """
    rows = list(appointments or [])
    if not rows:
        return TargetVerdict(None, "no_cue", [], [])
    if len(rows) == 1:
        # There was never a choice to get wrong. Saying so explicitly keeps the
        # single-appointment path out of the ambiguity branch below, where a
        # message with no cue in it would otherwise start asking "which one?"
        # about a list of one.
        return TargetVerdict(int(rows[0]["appointment_id"]), "only_one", [], [])

    text = message or ""
    weekdays = _weekdays_named(text)
    dates = _dates_named(text, today=today or clock.today())
    codes = _codes_named(text)

    # The department table decides what a department word is, exactly as it does
    # for routing and for the new-subject rule — so there is no keyword list
    # here to drift from the one the rest of the system uses.
    department = resolve_department(session, text)
    department_id = (
        department["department"]["id"] if department.get("status") == "resolved" else None
    )

    cues: list[str] = []
    cues += [f"weekday:{day}" for day in sorted(weekdays)]
    cues += [f"date:{day.isoformat()}" for day in sorted(dates)]
    cues += [f"code:{code}" for code in sorted(codes)]
    if department_id is not None:
        cues.append(f"department:{department_id}")

    candidates: list[int] = []
    for row in rows:
        start = row.get("start")
        moment = datetime.fromisoformat(start) if start else None
        hit = (
            (moment is not None and moment.weekday() in weekdays)
            or (moment is not None and moment.date() in dates)
            or _normalise_code(row.get("reference_code") or "") in codes
            or (department_id is not None and row.get("department_id") == department_id)
        )
        if hit:
            candidates.append(int(row["appointment_id"]))

    if len(candidates) == 1:
        return TargetVerdict(candidates[0], "matched", cues, candidates)
    if candidates:
        return TargetVerdict(None, "ambiguous", cues, candidates)
    return TargetVerdict(None, "no_cue", cues, [])


#: A position that opens the message: "1. Move it to after my other one". The
#: dot or bracket is required and so is the space after it — "1.30pm" is a time
#: and "2)" is an answer. Kept apart from
#: :func:`~app.workflow.selection.position_named`, which reads a bare number
#: only when it is the *whole* message, because here the rest of the sentence
#: is the patient elaborating on the choice they just made.
_LEADING_POSITION = re.compile(r"^\s*(\d{1,2})\s*[.)]\s+\S")


def read_choice(text: str, *, listed: Sequence[int]) -> int | None:
    """The appointment id this message picks out of a numbered list.

    ``None`` when it picks none, which leaves the turn exactly as it was.

    **The hole this fills.** ``render_appointment_choice`` numbers the rows and
    records the ids precisely so that "2" means something — and then nothing in
    code read the answer. ``_settle_target`` stands aside once a list has been
    shown, on the reasoning that the list's own answer would be read; it was
    not. Live, a cancel run asked "which one? 1..., 2..." and the patient
    answered "2". The Coordinator called that a *new cancellation request*, the
    supersede was correctly refused, the refusal's recovery searched for slots
    on a run that had no department, and the run failed: "I'm sorry — I couldn't
    complete this request."

    **A leading position outranks a date here, and only here.** "1. Move it to
    after I finish my General Medicine appointment which is on 6th August at
    10am" is a choice followed by an instruction. The referent list is
    appointments, so the numeral at the front is the answer and the date at the
    back is what to do next — the opposite of
    :func:`~app.workflow.selection.read_selection`, where a list of *times*
    makes "the 2pm one" a time and never row two.

    Out of range falls through to the model. A number that does not name a row
    is not a near miss to be rounded into one.
    """
    rows = list(listed or [])
    if not rows:
        return None

    message = text or ""
    # "Forget it" is not a choice, checked first for the reason it is checked
    # first everywhere else here: the withdrawal reading has to win outright or
    # a patient asking to be let go is answered with a cancellation proposal.
    if says_withdrawal(message):
        return None

    match = _LEADING_POSITION.match(message)
    position = int(match.group(1)) if match else position_named(message)
    if position is None or not 1 <= position <= len(rows):
        return None
    return int(rows[position - 1])


def narrow_by_department(
    session: Session,
    text: str,
    *,
    listed: Sequence[int],
    appointments: Sequence[dict[str, Any]],
) -> list[int]:
    """Which of the numbered rows a department word in this message points at.

    ``[]`` when the message names no department, or names one that matches none
    of them — both of which leave the turn exactly as it was.

    **The template asks for this answer.** ``render_appointment_choice`` closes
    with "Tell me the number, or name the department", and only the number had
    a reader. Live, with two Dermatology appointments listed, the patient
    answered "Dermatology": :func:`read_choice` found no position, the
    Coordinator called the message a *new* request, and the run was superseded —
    the template's own suggested answer swapped the request out from under the
    patient. A question that suggests an answer has to be able to hear it.

    One match is a decision and the caller acts on it. Two or more is half an
    answer: the patient has narrowed the list and the run asks again over
    exactly those rows, which is a shorter question than the one before it and
    never a supersede. That is the same shape :func:`resolve_target` uses and
    for the same reason — ambiguity refuses rather than picks.

    Matching is over the *listed* rows only. A department the patient has an
    appointment in that this run did not offer as a choice is not an answer to
    the question that was asked, and reading it as one would act on a row the
    numbering never mentioned.

    **A department inside a request is not an answer to a question.** "Book me
    a cardiology appointment", sent to a cancel run that had asked which of two
    appointments to cancel, names a department the patient does hold — and
    reading that as the answer would propose cancelling it. So a message
    carrying any appointment verb is left entirely alone: it is a new request,
    or a restatement, and both already have paths. The word this reader exists
    for is the bare department name the template asked for, and a bare
    department name carries no verb.
    """
    rows = list(listed or [])
    if not rows:
        return []

    message = text or ""
    # "Forget it" names no department, and it is checked here for the reason it
    # is checked first in every reader in this module: a withdrawal that reaches
    # a matching rule is a patient asking to be let go and being answered with a
    # proposal.
    if says_withdrawal(message) or names_appointment_verbs(message):
        return []

    department = resolve_department(session, message)
    if department.get("status") != "resolved":
        return []
    department_id = department["department"]["id"]

    by_id = {int(row["appointment_id"]): row for row in appointments or []}
    return [
        appointment_id
        for appointment_id in rows
        if by_id.get(appointment_id, {}).get("department_id") == department_id
    ]


__all__ = ["TargetVerdict", "narrow_by_department", "read_choice", "resolve_target"]
