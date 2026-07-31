"""What a sentence about an appointment must state, read from the row.

This is the half of Layer 2 that is not simply Layer 1 with the bill attached.
Plumbing keys — status, plan, counts — are graded identically at both layers,
because they are code's either way. **Facts are different**: a doctor's name, a
weekday, a clock time and a reference code are the four things a patient acts
on, and getting one wrong sends them to the wrong place at the wrong hour with
a receipt that looks perfect.

Two design notes, both learned from live failures.

**Facts are compared against the row, never against a literal in the scenario
file.** A scenario that hard-codes "Monday 3 August at 9:00 AM" is pinned to
the day it was written, and a scenario that hard-codes the reference code
cannot tell a *correct* appointment from a lucky one. So the expected values are
re-read from the database after the turn, and the reply is checked against
those.

**The negative half is the point.** "Names the right appointment" and "does not
name the wrong one" are different claims, and only the second catches the defect
that motivated this: a reschedule that moved the Thursday appointment while the
receipt described the Monday one. Every ownership, liveness and availability
guard passed — none of them asks whether the id is the one the patient *named*.
So a fact case may carry a ``forbid`` subject whose reference code must be
absent from the reply.

Note honestly what this can and cannot falsify. Since the paraphrase failures of
round 4, receipts and change proposals are assembled in ``app.workflow.replies``
from the rows, so the *rendering* of these facts is code's and will not drift
under a live model. What remains genuinely probabilistic — and what these cases
exist to grade — is **which row the model picked**, and whether the assembled
sentence reached the patient at all rather than the turn dead-ending short of
it.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models import Appointment, PatientProfile, User
from app.workflow.replies import when_words

#: The four fact fields, plus the department a patient reads as the fifth.
#: Named individually because not every sentence owes all of them — a
#: cancellation receipt states which appointment ended, not which doctor was
#: going to be there.
FACT_FIELDS = ("reference", "department", "day", "time", "doctor")


@dataclass(frozen=True)
class AppointmentFacts:
    reference: str
    department: str
    day: str
    time: str
    doctor: str

    def get(self, field: str) -> str:
        return getattr(self, field)


def appointment_facts(appointment: Appointment) -> AppointmentFacts:
    """The five facts, formatted exactly as a patient-facing sentence must.

    ``when_words`` is the application's own formatter rather than a second one
    written here. A grader with its own idea of how a time is spelled would
    pass a reply saying 15:00 and fail one saying 3:00 PM, which is precisely
    backwards — patient-facing times are 12-hour everywhere, through one
    formatter, and this asserts *that* rule as much as the value.
    """
    slot = appointment.slot
    # ``slot_id`` is nullable, and a case that grades only ``reference`` and
    # ``department`` must not blow up on a row that has released its slot. An
    # empty string is graded as "the row has no value to state", which is a
    # failure if the case asked for the field and silent if it did not.
    day, time = when_words(slot.start_time) if slot is not None else ("", "")
    return AppointmentFacts(
        reference=appointment.reference_code or "",
        department=appointment.department.name,
        day=day,
        time=time,
        doctor=appointment.doctor.name,
    )


def patient_appointments(session: Session, email: str) -> list[Appointment]:
    """This patient's appointments, oldest id first — the order a scenario counts in."""
    user = session.query(User).filter(User.email == email).one()
    profile = (
        session.query(PatientProfile).filter(PatientProfile.user_id == user.id).one()
    )
    return (
        session.query(Appointment)
        .filter(Appointment.patient_id == profile.id)
        .order_by(Appointment.id)
        .all()
    )


def find_appointment(
    session: Session,
    email: str,
    *,
    reference: str | None = None,
    department: str | None = None,
    newest: bool = False,
) -> Appointment | None:
    """Locate the appointment a fact case is about. Exactly one selector.

    ``reference`` where the scenario inherits one — ``AC-000001`` is the seed's
    pre-existing Cardiology visit. ``newest`` where the scenario *creates* the
    subject, since a reference code is generated at commit and cannot be written
    down in advance. ``department`` only where it is unambiguous.

    The three are separate rather than tried in turn because the first version
    fell back from one to another and quietly graded the wrong row: the booking
    scenario books into Cardiology, which is also the department of the
    appointment the seed ships, so "the Cardiology one" named the row that was
    already there and the case failed for looking at an appointment the turn was
    never about. A selector that can match two rows has to say which it meant.
    """
    chosen = [
        selector
        for selector in (reference, department, "newest" if newest else None)
        if selector is not None
    ]
    if len(chosen) != 1:
        raise ValueError(
            f"exactly one of reference/department/newest, got {len(chosen)}: {chosen}"
        )

    appointments = patient_appointments(session, email)
    if newest:
        return appointments[-1] if appointments else None
    for appointment in appointments:
        if reference is not None and appointment.reference_code == reference:
            return appointment
        if department is not None and appointment.department.name == department:
            return appointment
    return None


def grade(reply: str, facts: AppointmentFacts, fields: tuple[str, ...]) -> list[str]:
    """Every named field that is missing from the reply. Empty means clean."""
    misses = []
    for field in fields:
        expected = facts.get(field)
        if not expected:
            misses.append(f"{field}: the row has no value to state")
        elif expected not in reply:
            misses.append(f"{field}: expected {expected!r} in the reply")
    return misses


def grade_absent(reply: str, facts: AppointmentFacts, fields: tuple[str, ...]) -> list[str]:
    """Every named field that is present and should not be.

    Used for the appointment a turn must *not* be about. Reference code is the
    honest field here: a department name legitimately appears in a sentence for
    other reasons, and two appointments in the same department share it.
    """
    return [
        f"{field}: {facts.get(field)!r} must not appear — that is the other appointment"
        for field in fields
        if facts.get(field) and facts.get(field) in reply
    ]


__all__ = [
    "AppointmentFacts",
    "FACT_FIELDS",
    "appointment_facts",
    "find_appointment",
    "grade",
    "grade_absent",
    "patient_appointments",
]
