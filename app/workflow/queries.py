"""Answering "what do I have?" from the rows, without asking the model.

Three phrasings in one live session — "show my appointments", "show my
upcoming appointments?", "what documents I have on file?" — reached a
clarification dead end, while "what appointments I have coming up?" worked. The
scope gate was not the problem by then; the *planning* was. A listing question
has no consequence to weigh, no slot to hold and no state to move, and routing
it through plan-then-delegate makes whether the patient gets an answer depend
on whether the Coordinator felt like emitting a plan for that particular
sentence.

So it does not go through planning at all. This is the same architectural move
the slot question already made: a read-only question is answered by code, from
the read tools, in a template. The model still classifies — a question during a
run is a side question like any other — but what comes back is assembled here.

**Detection is deliberately narrow, and the asymmetry is the point.** A false
positive answers a question nobody asked with the patient's own data: annoying,
read-only, harmless. A false negative falls through to exactly today's
behaviour. What must never happen is an *action* being read as a query, so a
message carrying an appointment verb is never a listing question however much
it sounds like one — "book me a cardiology appointment" contains "appointment"
and is not a request to see a list.
"""

from __future__ import annotations

import re
from enum import Enum

from sqlalchemy.orm import Session

from app.models import Appointment
from app.tools.appointments import list_patient_appointments
from app.tools.departments import resolve_department
from app.tools.documents import diff_required_documents, list_patient_documents
from app.tools.reminders import list_patient_reminders
from app.workflow.replies import UPLOAD_POINTER, when_words


class QueryKind(str, Enum):
    """What a listing question is about. One renderer each."""

    APPOINTMENTS = "appointments"
    DOCUMENTS = "documents"
    REMINDERS = "reminders"


#: The shapes an ask takes. Either an imperative ("show me...", "list...") or a
#: question about possession ("what ... do I have", "which ... are booked").
#: Matched at word starts for the same reason everything else here is — a
#: substring rule is how "erm" got found inside "dermatology".
_ASKING = (
    r"\bshow\b",
    r"\blist\b",
    r"\btell me\b",
    r"\bwhat\b",
    r"\bwhich\b",
    r"\bdo i have\b",
    r"\bhave i got\b",
    r"\bon file\b",
    r"\bcoming up\b",
    r"\bupcoming\b",
    r"\bany\b",
)

#: What the question is about. First match wins, and the order is the order of
#: specificity: "what documents do I need for my appointment" is about
#: documents, and reading it as an appointment listing would answer the wrong
#: half of a sentence that named both.
#:
#: Documents outranking appointments was already true, and it was not enough.
#: Live, "do I need to submit any documentations for this visit?" came back as
#: the appointments list — not because the precedence was wrong but because the
#: document half never matched at all: ``\bdocuments?\b`` stops at "document"
#: and "documents", and "documentations" is neither. The only term left
#: standing was "visit". So the stems are open-ended here, and the words a
#: patient uses for *handing something over* — upload, submit, bring — count as
#: naming the subject, which is what makes the precedence do any work.
_SUBJECTS: tuple[tuple[QueryKind, tuple[str, ...]], ...] = (
    (QueryKind.DOCUMENTS, (r"\bdocument\w*\b", r"\breports?\b", r"\bpaperwork\b",
                           r"\brecords?\b", r"\bfiles?\b", r"\bupload\w*\b",
                           r"\bsubmit\w*\b", r"\bbring\b")),
    (QueryKind.REMINDERS, (r"\breminders?\b",)),
    (QueryKind.APPOINTMENTS, (r"\bappointments?\b", r"\bbookings?\b", r"\bvisits?\b")),
)

#: An action verb makes a message a request, not a question, whatever else it
#: contains. This is the guard that keeps "book me an appointment next week"
#: and "cancel my appointment" out of here — the cost of getting *that* wrong
#: is a booking silently replaced by a list.
_ACTIONS = (
    r"\bbook\b", r"\bbooking an\b", r"\bschedule\b", r"\breschedul", r"\bcancel\b",
    r"\bmove\b", r"\bchange\b", r"\bpostpone\b", r"\barrange\b", r"\bmake an?\b",
    r"\bneed an?\b", r"\bwant an?\b", r"\bupload\b", r"\bsend\b",
)

#: The exception to ``_ACTIONS``. "What documents do I *need* to bring" is a
#: question about requirements, not a request to do anything — and it was one
#: of the live dead ends, so the carve-out is named rather than accidental.
_NEEDS_FOR_VISIT = re.compile(
    r"\b(need|require|bring|prepare)\b.*\b(bring|visit|appointment|for my|beforehand)\b"
    r"|\b(what|which)\b.*\b(need|require|bring)\b",
    re.IGNORECASE,
)


def _matches(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def detect_query(text: str) -> QueryKind | None:
    """Which listing question this is, or ``None`` if it is not one.

    ``None`` is the common answer and the safe one: everything that is not
    plainly a question about the patient's own records goes on to be planned
    exactly as it is today.
    """
    message = (text or "").strip()
    if not message:
        return None

    subject = next(
        (kind for kind, patterns in _SUBJECTS if _matches(patterns, message)), None
    )
    if subject is None:
        return None
    if not _matches(_ASKING, message):
        return None

    if _matches(_ACTIONS, message):
        # A requirements question survives the action filter, and nothing else
        # does. "What documents do I need to bring for my ENT visit" contains
        # "need"; it is still a question.
        if not (subject is QueryKind.DOCUMENTS and _NEEDS_FOR_VISIT.search(message)):
            return None

    return subject


# --- rendering ------------------------------------------------------------


def _appointments(session: Session, patient_id: int) -> str:
    rows = list_patient_appointments(session, patient_id=patient_id, live_only=True)
    if not rows:
        return (
            "You have no upcoming appointments at the moment. "
            "Tell me what you need and I'll find you a time."
        )
    lines = []
    for row in rows:
        when = ""
        if row.get("start"):
            day, time = when_words(row["start"])
            when = f" on {day} at {time}"
        lines.append(
            f"- {row.get('department_name')} with {row.get('doctor_name')}"
            f"{when} ({row.get('reference_code')})"
        )
    heading = "Your upcoming appointment:" if len(lines) == 1 else "Your upcoming appointments:"
    return heading + "\n" + "\n".join(lines)


def _reminders(session: Session, patient_id: int) -> str:
    rows = list_patient_reminders(session, patient_id=patient_id)
    if not rows:
        return "You have no reminders scheduled at the moment."
    lines = []
    for row in rows:
        day, time = when_words(row["scheduled_at"])
        lines.append(f"- {day} at {time}")
    return "Reminders we have scheduled for you:\n" + "\n".join(lines)


def _department_for_documents(
    session: Session, *, patient_id: int, message: str
) -> tuple[int, str] | None:
    """Which department a documents question is about.

    In order: the department the patient named, then the one their upcoming
    appointment is in. Live, "what documents do I need to bring for my ENT
    visit?" was answered with *"no department has been decided for this
    request"* — with ENT written in the message and an ENT appointment on the
    books. Neither source was consulted, because the only one the code knew
    about was the *run's* department, and a documents query has no run that
    ever routed.

    ``resolve_department`` decides what the message named, so the Department
    table is the authority and there is no keyword list to slip past.
    """
    named = resolve_department(session, message or "")
    if named.get("status") == "resolved":
        return named["department"]["id"], named["department"]["name"]

    upcoming = list_patient_appointments(session, patient_id=patient_id, live_only=True)
    if len(upcoming) == 1:
        appointment = session.get(Appointment, upcoming[0]["appointment_id"])
        if appointment is not None and appointment.department is not None:
            return appointment.department_id, appointment.department.name
    return None


def _documents(session: Session, patient_id: int, message: str) -> str:
    """What is on file, and what the visit still wants.

    Two sections, each omitted when it has nothing to say — and the second is
    omitted *loudly* rather than replaced by a reassurance. Silence about
    requirements means "there is nothing to compare against here"; saying "no
    additional documents are required" would mean "the hospital wants nothing",
    and the live reply said the second while meaning the first.
    """
    parts: list[str] = []

    held = list_patient_documents(session, patient_id=patient_id)
    if held:
        lines = [
            f"- {row['document_type']} ({row['status']})"
            for row in held
        ]
        parts.append("Documents on file:\n" + "\n".join(lines))
    else:
        parts.append("You have no documents on file yet.")

    where = _department_for_documents(
        session, patient_id=patient_id, message=message
    )
    if where is not None:
        department_id, department_name = where
        diff = diff_required_documents(
            session, patient_id=patient_id, department_id=department_id
        )
        if diff["missing_mandatory"]:
            parts.append(
                f"Required before your {department_name} visit: "
                f"{', '.join(diff['missing_mandatory'])}."
            )
        if diff["missing_optional"]:
            parts.append(
                f"Optional but helpful for {department_name}: "
                f"{', '.join(diff['missing_optional'])}."
            )
        if diff["missing_mandatory"] or diff["missing_optional"]:
            parts.append(UPLOAD_POINTER)
        elif diff["required"]:
            # Everything the department asks for is already filed. That *is* a
            # complete answer, and it is only sayable because there was a
            # department to diff against — which is exactly the claim the live
            # reply made without one.
            parts.append(
                f"Nothing else is needed for your {department_name} visit — "
                "everything they ask for is already on file."
            )

    # Blank lines, not single newlines: the first part is a bullet list, and a
    # sentence one newline below a list item is read as part of that item —
    # "Required before your ENT visit" rendered as a continuation of the last
    # document on file.
    return "\n\n".join(parts)


def answer_query(
    session: Session, *, patient_id: int, kind: QueryKind, message: str
) -> str:
    """The answer, assembled from rows. Never empty."""
    if kind is QueryKind.APPOINTMENTS:
        return _appointments(session, patient_id)
    if kind is QueryKind.REMINDERS:
        return _reminders(session, patient_id)
    return _documents(session, patient_id, message)


__all__ = ["QueryKind", "answer_query", "detect_query"]
