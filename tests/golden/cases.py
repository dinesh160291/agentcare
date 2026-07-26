"""Golden cases: one entry per retrieval surface.

Each case is a function that takes a freshly seeded session and returns a
JSON-serialisable result. The runner freezes the clock to :data:`ANCHOR`,
seeds, calls the **real tool functions** — never raw SQL, or the goldens would
pin the database rather than the behaviour — and diffs against the blessed
JSON in ``expected/``.

No LLM is involved anywhere here. These run in CI on every push and must never
be flaky.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.models import User
from app.tools.appointments import book_appointment, reschedule_appointment
from app.tools.availability import find_available_slots
from app.tools.dates import resolve_date
from app.tools.departments import list_departments, resolve_department, validate_department
from app.tools.documents import (
    diff_required_documents,
    find_duplicate,
    list_patient_documents,
)
from app.tools.reminders import list_due_reminders

#: Monday. Every golden expectation is relative to this date.
ANCHOR = date(2026, 8, 3)
#: Eight in the morning, before the first slot of the day at nine.
ANCHOR_MOMENT = datetime(2026, 8, 3, 8, 0)

CARDIOLOGY = 1
ENT = 7

GoldenCase = Callable[[Session], Any]


# --- departments --------------------------------------------------------


def departments_all(session: Session) -> Any:
    return list_departments(session)


def department_resolve_by_name(session: Session) -> Any:
    return resolve_department(session, "I need a Cardiology appointment next week")


def department_resolve_by_synonym(session: Session) -> Any:
    return resolve_department(session, "something to do with my heart")


def department_resolve_ambiguous(session: Session) -> Any:
    """The seeded ambiguous case that feeds the staff-review path."""
    return resolve_department(session, "my kid has ear pain")


def department_resolve_unsupported(session: Session) -> Any:
    return resolve_department(session, "can you tell me the weather forecast")


def department_validate_invented(session: Session) -> Any:
    """Plausible, well-formed, and not a department this hospital has."""
    return validate_department(session, "Cardiovascular Medicine")


# --- dates --------------------------------------------------------------


def dates_next_week(session: Session) -> Any:
    return resolve_date("next week")


def dates_tomorrow_morning(session: Session) -> Any:
    return resolve_date("tomorrow morning")


def dates_weekday(session: Session) -> Any:
    return resolve_date("next friday")


def dates_explicit(session: Session) -> Any:
    return resolve_date("20 August")


def dates_refused(session: Session) -> Any:
    return resolve_date("sometime soonish maybe")


def dates_in_the_past(session: Session) -> Any:
    return resolve_date("2026-07-01")


# --- availability -------------------------------------------------------


def slots_cardiology_next_week(session: Session) -> Any:
    window = resolve_date("next week")
    return find_available_slots(
        session,
        department_id=CARDIOLOGY,
        start=date.fromisoformat(window["start"]),
        end=date.fromisoformat(window["end"]),
        limit=5,
    )


def slots_morning_only(session: Session) -> Any:
    return find_available_slots(
        session, department_id=CARDIOLOGY, part_of_day="morning", limit=4
    )


def slots_none_available(session: Session) -> Any:
    far = ANCHOR + timedelta(days=400)
    return find_available_slots(session, department_id=CARDIOLOGY, start=far, end=far)


# --- documents ----------------------------------------------------------


def required_docs_missing_everything(session: Session) -> Any:
    """Patient 2 holds nothing; Cardiology wants two documents."""
    return diff_required_documents(session, patient_id=2, department_id=CARDIOLOGY)


def required_docs_partially_supplied(session: Session) -> Any:
    """Patient 1 is seeded with an ECG report and a blood test report."""
    return diff_required_documents(session, patient_id=1, department_id=CARDIOLOGY)


def required_docs_optional_only(session: Session) -> Any:
    """An optional shortfall must not read as incomplete."""
    return diff_required_documents(session, patient_id=2, department_id=ENT)


def documents_of_patient_one(session: Session) -> Any:
    return list_patient_documents(session, patient_id=1)


def duplicate_detected(session: Session) -> Any:
    existing = list_patient_documents(session, patient_id=1)[0]
    return find_duplicate(session, patient_id=1, checksum=existing["checksum"])


def duplicate_not_detected(session: Session) -> Any:
    return find_duplicate(session, patient_id=1, checksum="00" * 32)


# --- the derivation invariant -------------------------------------------


def reminders_due_after_reschedule(session: Session) -> Any:
    """Book, reschedule, then ask what is due at the *original* reminder time.

    The old reminder must be absent and the new one present. A reminder left
    behind would fire for an appointment that moved, telling the patient to
    attend on a day nothing is happening.
    """
    patient = session.query(User).filter(User.id == 2).one()

    first = find_available_slots(session, department_id=CARDIOLOGY, limit=1)["slots"][0]
    # Move it into next week, so the golden shows a move a reader can see.
    window = resolve_date("next week")
    later = find_available_slots(
        session,
        department_id=CARDIOLOGY,
        start=date.fromisoformat(window["start"]),
        end=date.fromisoformat(window["end"]),
        limit=1,
    )["slots"][0]

    booked = book_appointment(session, patient, slot_id=first["slot_id"], reason="follow-up")
    appointment_id = booked["appointment"]["appointment_id"]
    original_reminder_time = datetime.fromisoformat(first["start"]) - timedelta(hours=24)

    reschedule_appointment(session, patient, appointment_id, new_slot_id=later["slot_id"])

    return {
        "due_at_original_reminder_time": list_due_reminders(session, at=original_reminder_time),
        "due_after_new_reminder_time": list_due_reminders(
            session, at=datetime.fromisoformat(later["start"]) - timedelta(hours=23)
        ),
    }


def patient_context_seeded(session: Session) -> Any:
    """Everything the Coordinator knows about the seeded patient."""
    from app.tools.patients import get_patient_context

    patient = session.query(User).filter(User.id == 1).one()
    return get_patient_context(session, patient)


def confirmation_for_seeded_appointment(session: Session) -> Any:
    """The consequential-wording seam, pinned exactly.

    Live evals grade these fact fields against the database; this golden is
    what "correct" means for them.
    """
    from app.tools.confirmations import render_confirmation

    return render_confirmation(session, 1)


CASES: dict[str, GoldenCase] = {
    "patient_context_seeded": patient_context_seeded,
    "confirmation_for_seeded_appointment": confirmation_for_seeded_appointment,
    "departments_all": departments_all,
    "department_resolve_by_name": department_resolve_by_name,
    "department_resolve_by_synonym": department_resolve_by_synonym,
    "department_resolve_ambiguous": department_resolve_ambiguous,
    "department_resolve_unsupported": department_resolve_unsupported,
    "department_validate_invented": department_validate_invented,
    "dates_next_week": dates_next_week,
    "dates_tomorrow_morning": dates_tomorrow_morning,
    "dates_weekday": dates_weekday,
    "dates_explicit": dates_explicit,
    "dates_refused": dates_refused,
    "dates_in_the_past": dates_in_the_past,
    "slots_cardiology_next_week": slots_cardiology_next_week,
    "slots_morning_only": slots_morning_only,
    "slots_none_available": slots_none_available,
    "required_docs_missing_everything": required_docs_missing_everything,
    "required_docs_partially_supplied": required_docs_partially_supplied,
    "required_docs_optional_only": required_docs_optional_only,
    "documents_of_patient_one": documents_of_patient_one,
    "duplicate_detected": duplicate_detected,
    "duplicate_not_detected": duplicate_not_detected,
    "reminders_due_after_reschedule": reminders_due_after_reschedule,
}
