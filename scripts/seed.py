"""Reset and seed the AgentCare database, then verify what was written.

Run:  python scripts/seed.py            (reset + seed + self-check)
      python scripts/seed.py --check    (self-check only, no writes)

Together with the SQLAlchemy models and ``create_all``, this file is the
submission's "database models and initialization files".

**Date robustness.** The seed must produce equivalent state on any boot day. A
relative date that lands on a slotless Sunday would strand a seeded run
mid-flight, and an active run swallows the demo patient's first real message
via the message-to-run mapping. So: slots are generated for every non-Sunday in
the window, and every seeded appointment is pinned to a slot that provably
exists. The self-check enforces this rather than trusting it.

All data here is obviously synthetic. No real person, no real contact detail,
no real credential.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

# Allow "python scripts/seed.py" from a clean checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session  # noqa: E402

from app import clock  # noqa: E402
from app.auth.security import hash_password  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.workflow.replies import clock_time  # noqa: E402
from app.db import SessionLocal, create_all, drop_all, engine  # noqa: E402
from app.models import (  # noqa: E402
    Appointment,
    AppointmentSlot,
    AppointmentStatus,
    Department,
    DepartmentRequiredDocument,
    DepartmentSynonym,
    Doctor,
    DocumentStatus,
    PatientDocument,
    PatientProfile,
    Reminder,
    ReminderStatus,
    ReminderType,
    SlotStatus,
    User,
    UserRole,
    WorkflowRun,
)
from app.models.enums import TERMINAL_WORKFLOW_STATUSES  # noqa: E402
from scripts.sample_pdf import build_pdf  # noqa: E402

# --- shape of the seeded world ------------------------------------------
SLOT_WINDOW_DAYS = 14
SLOT_TIMES = [time(9, 0), time(10, 0), time(11, 0), time(14, 0), time(15, 0), time(16, 0)]
SLOT_DURATION = timedelta(minutes=30)
DEMO_PASSWORD = "Demo123!pass"  # synthetic, documented in the README

# Departments: (id, name, description, synonyms, [(document type, mandatory)])
# Descriptions are administrative only — which department handles which kind of
# request. Nothing here describes, implies, or names a medical condition.
#
# The synonym list is routing vocabulary — the words a patient uses for the
# desk they need, not a symptom dictionary. It is deliberately long, and it is
# **data**: a 25-phrase probe of ordinary phrasings resolved 15, and the other
# ten ("back pain", "headache", "fever", "diarrhea", "hair loss", "fractured",
# "red eyes", "pregnant", "my son") each took the clarify-or-review path. None
# of those was a misroute — unresolved fails safe — but a queue that fills with
# questions the table could have answered is a queue nobody reads.
#
# Three rules hold this list together:
#
# * **Plurals are rows, not code.** Matching is on word boundaries, so "ears"
#   is not "ear" and "knees" is not "knee". Teaching the matcher to pluralise
#   would be a change to the deterministic bin for something thirty rows do
#   exactly and visibly.
# * **A term belongs to one department.** ``uq_department_synonym_term`` is a
#   global unique constraint, so the same string cannot be filed under two
#   desks. Ambiguity is produced the way "my kid has ear pain" produces it —
#   two *different* terms owned by two departments, both present in one
#   sentence. Hence "blood pressure" under Cardiology and the bare word
#   "pressure" under General Medicine: either desk is a defensible answer, so
#   the pair resolves ``ambiguous`` and the patient is asked, which is the
#   behaviour that was wanted and the constraint would otherwise have refused.
# * **A word that is ordinary English stays out.** Bare "back" was dropped for
#   "back pain"/"backache"/"lower back": "push back my appointment" is one of
#   the reschedule phrasings this project already reads, and a synonym that
#   turns it into an Orthopedics request would hand the refinement rule a new
#   subject that the patient never named.
DEPARTMENTS: list[tuple[int, str, str, list[str], list[tuple[str, bool]]]] = [
    (
        1,
        "Cardiology",
        "Handles heart and circulatory appointments and follow-ups.",
        ["heart", "cardiac", "cardiology", "ecg", "ekg", "echo",
         "blood pressure", "bp", "palpitations"],
        [("ECG report", True), ("Blood test report", True)],
    ),
    (
        2,
        "Orthopedics",
        "Handles bone, joint, and musculoskeletal appointments.",
        ["bone", "joint", "orthopedic", "orthopaedic", "fracture", "knee", "shoulder",
         "bones", "joints", "knees", "shoulders",
         "back pain", "backache", "lower back", "spine",
         "fractured", "broken", "sprain", "sprained",
         "ankle", "hip", "wrist", "elbow"],
        [("X-ray report", True)],
    ),
    (
        3,
        "Dermatology",
        "Handles skin, hair, and nail appointments.",
        ["skin", "derma", "dermatology", "rash", "acne",
         "hair", "hair loss", "itching", "allergy", "eczema"],
        [("Previous treatment summary", False)],
    ),
    (
        4,
        "General Medicine",
        "Handles general adult consultations and routine check-ups.",
        ["general", "physician", "gp", "general medicine", "check-up", "checkup",
         "fever", "cough", "cold", "flu", "body ache",
         "physical", "annual", "weakness", "pressure"],
        [],
    ),
    (
        5,
        "Pediatrics",
        "Handles appointments for children and adolescents.",
        ["child", "children", "kid", "paediatric", "pediatric", "infant", "baby",
         "kids", "babies", "son", "daughter", "toddler",
         "vaccination", "immunization"],
        [("Immunization record", True)],
    ),
    (
        6,
        "Neurology",
        "Handles nervous-system appointments and follow-ups.",
        ["neuro", "neurology", "nerve", "migraine", "seizure",
         "headache", "headaches", "migraines", "dizziness", "numbness"],
        [("Prior MRI or CT report", True)],
    ),
    (
        7,
        "ENT",
        "Handles ear, nose, and throat appointments.",
        ["ent", "ear", "nose", "throat", "hearing", "sinus", "tonsil",
         "ears", "earache", "snoring", "hoarse"],
        [("Previous audiometry report", False)],
    ),
    (
        8,
        "Ophthalmology",
        "Handles eye and vision appointments.",
        ["eye", "vision", "ophthalmology", "optical", "cataract",
         "eyes", "eyesight", "sight", "blurry", "eyelid",
         "glasses", "lens", "lenses"],
        [("Previous eye test report", True)],
    ),
    (
        9,
        "Gynecology & Obstetrics",
        "Handles women's health, pregnancy, and maternity appointments.",
        ["gynecology", "gynaecology", "obstetrics", "pregnancy", "maternity", "prenatal",
         "pregnant", "period", "periods", "cramps", "menstrual"],
        [("Previous ultrasound report", True)],
    ),
    (
        10,
        "Gastroenterology",
        "Handles digestive-system appointments and follow-ups.",
        ["gastro", "gastroenterology", "digestive", "stomach", "liver", "endoscopy",
         "gastric", "diarrhea", "acidity", "indigestion", "heartburn",
         "vomiting", "nausea", "tummy", "belly", "constipation"],
        [("Previous endoscopy report", False)],
    ),
]

# (id, department id, name)
DOCTORS: list[tuple[int, int, str]] = [
    (1, 1, "Dr. Anita Rao"), (2, 1, "Dr. Vikram Nair"), (3, 1, "Dr. Leena Fernandes"),
    (4, 2, "Dr. Sanjay Menon"), (5, 2, "Dr. Priya Iyer"),
    (6, 3, "Dr. Rahul Bose"), (7, 3, "Dr. Meera Kulkarni"),
    (8, 4, "Dr. Arjun Pillai"), (9, 4, "Dr. Sneha Reddy"), (10, 4, "Dr. Imran Sheikh"),
    (11, 5, "Dr. Kavita Joshi"), (12, 5, "Dr. Nikhil Verma"),
    (13, 6, "Dr. Farida Qureshi"), (14, 6, "Dr. Alok Chandra"),
    (15, 7, "Dr. Deepa Krishnan"), (16, 7, "Dr. Rohit Malhotra"),
    (17, 8, "Dr. Sunita Grover"), (18, 8, "Dr. Karthik Subramanian"),
    (19, 9, "Dr. Nandini Shah"), (20, 9, "Dr. Ruchi Agarwal"),
    (21, 10, "Dr. Manish Dutta"), (22, 10, "Dr. Ayesha Siddiqui"),
]

# (id, name, email, role)
USERS: list[tuple[int, str, str, UserRole]] = [
    (1, "Asha Menon", "asha.patient@example.invalid", UserRole.PATIENT),
    (2, "Rohan Gupta", "rohan.patient@example.invalid", UserRole.PATIENT),
    (3, "Fatima Noor", "fatima.patient@example.invalid", UserRole.PATIENT),
    (4, "Daniel Osei", "daniel.patient@example.invalid", UserRole.PATIENT),
    (5, "Priya Desk", "staff@example.invalid", UserRole.STAFF),
]

# (profile id, user id, birth year-month-day, phone, language, emergency contact)
PROFILES: list[tuple[int, int, date, str, str, str]] = [
    (1, 1, date(1986, 4, 12), "+1-555-0100", "English", "Ravi Menon +1-555-0101"),
    (2, 2, date(1994, 9, 3), "+1-555-0102", "English", "Sunita Gupta +1-555-0103"),
    (3, 3, date(1979, 1, 27), "+1-555-0104", "English", "Imran Noor +1-555-0105"),
    (4, 4, date(2015, 6, 8), "+1-555-0106", "English", "Grace Osei +1-555-0107"),
]


def _first_non_sunday(start: date) -> date:
    """Slots are not generated on Sundays; skip forward to a day that has them."""
    day = start
    while day.weekday() == 6:
        day += timedelta(days=1)
    return day


def _slot_days(anchor: date) -> list[date]:
    """Every slotted day in the seeding window."""
    return [
        anchor + timedelta(days=offset)
        for offset in range(SLOT_WINDOW_DAYS)
        if (anchor + timedelta(days=offset)).weekday() != 6
    ]


def _document_fixtures() -> list[tuple[str, str, list[str]]]:
    """(document type, title, body lines) for the synthetic sample documents.

    The last entry deliberately mismatches the type it will be declared as, so
    the document-verification path has a genuine failure case to catch.
    """
    return [
        (
            "ECG report",
            "SYNTHETIC ECG REPORT - SAMPLE DATA",
            [
                "Facility: AgentCare Demo Hospital (synthetic)",
                "Patient: SYNTHETIC RECORD - not a real person",
                "Report type: Electrocardiogram summary",
                "This document contains no real patient information.",
                "Generated for demonstration of document coordination only.",
            ],
        ),
        (
            "Blood test report",
            "SYNTHETIC BLOOD TEST REPORT - SAMPLE DATA",
            [
                "Facility: AgentCare Demo Hospital (synthetic)",
                "Patient: SYNTHETIC RECORD - not a real person",
                "Report type: Routine blood panel summary",
                "This document contains no real patient information.",
                "Values omitted: this is an administrative fixture.",
            ],
        ),
        (
            "X-ray report",
            "SYNTHETIC X-RAY REPORT - SAMPLE DATA",
            [
                "Facility: AgentCare Demo Hospital (synthetic)",
                "Patient: SYNTHETIC RECORD - not a real person",
                "Report type: Radiology summary",
                "This document contains no real patient information.",
            ],
        ),
    ]


def seed(session: Session, *, anchor: date | None = None) -> None:
    """Populate an empty database. Caller owns the transaction."""
    anchor = anchor or clock.today()
    settings = get_settings()

    # --- departments, synonyms, required-document rules -----------------
    for dept_id, name, description, synonyms, required in DEPARTMENTS:
        session.add(
            Department(id=dept_id, name=name, description=description, active=True)
        )
        for term in synonyms:
            session.add(DepartmentSynonym(department_id=dept_id, term=term.lower()))
        for document_type, mandatory in required:
            session.add(
                DepartmentRequiredDocument(
                    department_id=dept_id, document_type=document_type, mandatory=mandatory
                )
            )

    for doctor_id, dept_id, name in DOCTORS:
        session.add(Doctor(id=doctor_id, department_id=dept_id, name=name, active=True))
    session.flush()

    # --- slots ----------------------------------------------------------
    slot_id = 1
    for doctor_id, _dept_id, _name in DOCTORS:
        for day in _slot_days(anchor):
            for slot_time in SLOT_TIMES:
                start = datetime.combine(day, slot_time)
                session.add(
                    AppointmentSlot(
                        id=slot_id,
                        doctor_id=doctor_id,
                        start_time=start,
                        end_time=start + SLOT_DURATION,
                        status=SlotStatus.AVAILABLE,
                    )
                )
                slot_id += 1
    session.flush()

    # --- users and patient profiles -------------------------------------
    password_hash = hash_password(DEMO_PASSWORD)
    for user_id, name, email, role in USERS:
        session.add(
            User(id=user_id, name=name, email=email, password_hash=password_hash, role=role)
        )
    session.flush()

    for profile_id, user_id, dob, phone, language, emergency in PROFILES:
        session.add(
            PatientProfile(
                id=profile_id,
                user_id=user_id,
                date_of_birth=dob,
                phone=phone,
                preferred_language=language,
                emergency_contact=emergency,
            )
        )
    session.flush()

    # --- one already-booked appointment, pinned to a real slot ----------
    # Three days out, skipped forward past any Sunday, so this never points at
    # a day the seed generated no slots for.
    target_day = _first_non_sunday(anchor + timedelta(days=3))
    booked_slot = (
        session.query(AppointmentSlot)
        .filter(
            AppointmentSlot.doctor_id == 1,
            AppointmentSlot.start_time >= datetime.combine(target_day, time.min),
            AppointmentSlot.start_time <= datetime.combine(target_day, time.max),
            AppointmentSlot.status == SlotStatus.AVAILABLE,
        )
        .order_by(AppointmentSlot.start_time)
        .first()
    )
    if booked_slot is None:  # pragma: no cover - the window guarantees one
        raise RuntimeError("Seed could not find a slot to book; slot generation is broken")

    booked_slot.status = SlotStatus.BOOKED
    appointment = Appointment(
        id=1,
        patient_id=1,
        doctor_id=booked_slot.doctor_id,
        slot_id=booked_slot.id,
        department_id=1,
        status=AppointmentStatus.CONFIRMED,
        reason="Follow-up appointment requested by patient",
        reference_code="AC-000001",
    )
    session.add(appointment)
    session.flush()

    # Derived from the appointment: 24 hours before it starts.
    session.add(
        Reminder(
            patient_id=1,
            appointment_id=appointment.id,
            reminder_type=ReminderType.APPOINTMENT,
            scheduled_at=booked_slot.start_time - timedelta(hours=24),
            status=ReminderStatus.PENDING,
            message=(
                # Through the one patient-facing formatter, like every other
                # time this system says out loud. Phase 8 is what made this
                # matter: the poll job now *delivers* this string, so a seeded
                # "09:00" would reach a patient beside a chat saying 9:00 AM.
                f"Reminder: appointment with {session.get(Doctor, booked_slot.doctor_id).name} "
                f"on {booked_slot.start_time:%A %d %B} at "
                f"{clock_time(booked_slot.start_time)}."
            ),
        )
    )

    # --- synthetic documents on disk ------------------------------------
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    for index, (document_type, title, lines) in enumerate(_document_fixtures(), start=1):
        payload = build_pdf(title, lines)
        checksum = hashlib.sha256(payload).hexdigest()
        stored_name = f"seed-{index:03d}-{checksum[:12]}.pdf"
        (upload_dir / stored_name).write_bytes(payload)

        # The third fixture is filed under a deliberately wrong declared type
        # so verification has something real to flag.
        declared = document_type if index < 3 else "ECG report"
        session.add(
            PatientDocument(
                patient_id=1,
                declared_type=declared,
                document_type=declared,
                detected_type=None,
                storage_path=str(Path(settings.upload_dir) / stored_name),
                original_filename=f"{document_type.lower().replace(' ', '_')}.pdf",
                mime_type="application/pdf",
                size_bytes=len(payload),
                document_date=anchor - timedelta(days=30 * index),
                checksum=checksum,
                status=DocumentStatus.PENDING_VERIFICATION,
            )
        )

    session.flush()


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def self_check(session: Session, *, anchor: date | None = None) -> list[str]:
    """Assert the seeded database is demo-ready. Returns a list of problems."""
    anchor = anchor or clock.today()
    problems: list[str] = []

    settings = get_settings()
    if ":memory:" in settings.database_url:
        problems.append("DATABASE_URL is in-memory; persistent state is required")

    # No seeded patient may hold a live run. An active run would capture the
    # demo patient's first real message instead of starting a fresh one.
    live_runs = (
        session.query(WorkflowRun)
        .filter(WorkflowRun.status.notin_([s.value for s in TERMINAL_WORKFLOW_STATUSES]))
        .count()
    )
    if live_runs:
        problems.append(f"{live_runs} seeded workflow run(s) are non-terminal")

    if session.query(Department).count() != len(DEPARTMENTS):
        problems.append("department count does not match the seed definition")

    for dept_id, name, _desc, _syn, _req in DEPARTMENTS:
        doctors = session.query(Doctor).filter(Doctor.department_id == dept_id).count()
        if doctors < 2:
            problems.append(f"{name} has {doctors} doctor(s); expected at least 2")

    # Every department must have bookable capacity inside the window, on any
    # boot day — this is the check that makes "clone and demo on a Sunday" safe.
    window_end = datetime.combine(anchor + timedelta(days=SLOT_WINDOW_DAYS), time.max)
    for dept_id, name, _desc, _syn, _req in DEPARTMENTS:
        available = (
            session.query(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .filter(
                Doctor.department_id == dept_id,
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                AppointmentSlot.start_time >= datetime.combine(anchor, time.min),
                AppointmentSlot.start_time <= window_end,
            )
            .count()
        )
        if available < len(SLOT_TIMES):
            problems.append(f"{name} has only {available} available slot(s) in the window")

    # Slots must never be seeded into the past.
    stale = (
        session.query(AppointmentSlot)
        .filter(AppointmentSlot.start_time < datetime.combine(anchor, time.min))
        .count()
    )
    if stale:
        problems.append(f"{stale} slot(s) are dated before today")

    # Every booked slot has exactly one appointment, and vice versa.
    for appointment in session.query(Appointment).all():
        slot = session.get(AppointmentSlot, appointment.slot_id)
        if slot is None:
            problems.append(f"appointment {appointment.id} references a missing slot")
        elif slot.status != SlotStatus.BOOKED:
            problems.append(f"appointment {appointment.id} holds a slot marked available")

    booked = session.query(AppointmentSlot).filter(
        AppointmentSlot.status == SlotStatus.BOOKED
    ).count()
    if booked != session.query(Appointment).count():
        problems.append("booked slot count does not match appointment count")

    # Reminders are derived rows: each must point at a live appointment.
    for reminder in session.query(Reminder).all():
        if reminder.appointment_id is None:
            continue
        appointment = session.get(Appointment, reminder.appointment_id)
        if appointment is None:
            problems.append(f"reminder {reminder.id} references a missing appointment")
        elif appointment.status == AppointmentStatus.CANCELLED:
            problems.append(f"reminder {reminder.id} survives a cancelled appointment")

    # Document bytes must match their stored checksum, or duplicate detection
    # is testing something that is not on disk.
    for document in session.query(PatientDocument).all():
        path = Path(document.storage_path)
        if not path.exists():
            problems.append(f"document {document.id} file is missing: {path}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != document.checksum:
            problems.append(f"document {document.id} checksum does not match its file")

    if session.query(User).filter(User.role == UserRole.STAFF).count() < 1:
        problems.append("no staff account was seeded")

    return problems


def run(reset: bool = True, check_only: bool = False) -> int:
    """Entry point. Returns a process exit code."""
    anchor = clock.today()

    if not check_only:
        if reset:
            drop_all()
        create_all()

    session = SessionLocal()
    try:
        if not check_only:
            seed(session, anchor=anchor)
            session.commit()

        problems = self_check(session, anchor=anchor)
    finally:
        session.close()

    if problems:
        print("Seed self-check FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    if not check_only:
        session = SessionLocal()
        try:
            counts = {
                "departments": session.query(Department).count(),
                "doctors": session.query(Doctor).count(),
                "slots": session.query(AppointmentSlot).count(),
                "users": session.query(User).count(),
                "appointments": session.query(Appointment).count(),
                "documents": session.query(PatientDocument).count(),
                "reminders": session.query(Reminder).count(),
            }
        finally:
            session.close()
        print(f"Seeded {engine.url.render_as_string(hide_password=True)} (anchor {anchor})")
        for label, value in counts.items():
            print(f"  {label:14s} {value}")
        print(f"\n  Demo password for every seeded account: {DEMO_PASSWORD}")
        print(f"  Staff login: {USERS[-1][2]}")
        print(f"  Patient login: {USERS[0][2]}")

    print("Seed self-check passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset and seed the AgentCare database.")
    parser.add_argument(
        "--check", action="store_true", help="run the self-check only, without writing"
    )
    parser.add_argument(
        "--no-reset", action="store_true", help="seed without dropping existing tables"
    )
    args = parser.parse_args()
    return run(reset=not args.no_reset, check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
