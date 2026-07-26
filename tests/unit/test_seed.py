"""The seed script and its self-check.

The centrepiece is :class:`TestDateRobustness`. A judge clones this repo and
demos it on whatever day of the week that happens to be, so the seed has to
produce a working, fully-terminal database on all seven — including the Sunday
that generates no slots.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest

from app import clock
from app.db import SessionLocal, create_all, drop_all
from app.models import (
    Appointment,
    AppointmentSlot,
    Department,
    DepartmentRequiredDocument,
    DepartmentSynonym,
    Doctor,
    PatientDocument,
    Reminder,
    SlotStatus,
    User,
    UserRole,
    WorkflowRun,
)
from scripts.sample_pdf import build_pdf
from scripts.seed import DEPARTMENTS, SLOT_WINDOW_DAYS, seed, self_check


class TestSeedContents:
    def test_ten_departments_are_seeded(self, seeded_db):
        assert seeded_db.query(Department).count() == 10

    def test_every_department_has_at_least_two_doctors(self, seeded_db):
        for dept in seeded_db.query(Department).all():
            count = seeded_db.query(Doctor).filter(Doctor.department_id == dept.id).count()
            assert count >= 2, f"{dept.name} has {count} doctors"

    def test_routing_synonyms_are_rows_and_lowercase(self, seeded_db):
        """Deterministic department resolution reads these rows; mixed case
        would make the lookup depend on how the patient typed."""
        synonyms = seeded_db.query(DepartmentSynonym).all()
        assert len(synonyms) > 40
        assert all(s.term == s.term.lower() for s in synonyms)

    def test_required_documents_are_rules_rows_with_a_mandatory_flag(self, seeded_db):
        cardiology = seeded_db.query(Department).filter_by(name="Cardiology").one()
        rules = (
            seeded_db.query(DepartmentRequiredDocument)
            .filter_by(department_id=cardiology.id)
            .all()
        )
        assert {r.document_type for r in rules} == {"ECG report", "Blood test report"}
        assert all(r.mandatory for r in rules)

    def test_a_department_may_require_nothing(self, seeded_db):
        general = seeded_db.query(Department).filter_by(name="General Medicine").one()
        assert (
            seeded_db.query(DepartmentRequiredDocument)
            .filter_by(department_id=general.id)
            .count()
            == 0
        )

    def test_optional_documents_are_distinguishable_from_mandatory_ones(self, seeded_db):
        """The diff has to say 'missing' rather than 'nice to have'."""
        ent = seeded_db.query(Department).filter_by(name="ENT").one()
        rule = (
            seeded_db.query(DepartmentRequiredDocument).filter_by(department_id=ent.id).one()
        )
        assert rule.mandatory is False

    def test_a_staff_account_exists(self, seeded_db):
        assert seeded_db.query(User).filter(User.role == UserRole.STAFF).count() == 1

    def test_seeded_credentials_are_obviously_synthetic(self, seeded_db):
        """RULE-6: no real data. ``.invalid`` is reserved by RFC 2606 and can
        never resolve to a real mailbox."""
        for user in seeded_db.query(User).all():
            assert user.email.endswith("@example.invalid")

    def test_no_seeded_workflow_run_is_live(self, seeded_db):
        """An active run would swallow the demo patient's first real message
        through the message-to-run mapping."""
        assert seeded_db.query(WorkflowRun).count() == 0

    def test_the_booked_appointment_holds_a_real_slot(self, seeded_db):
        appointment = seeded_db.query(Appointment).one()
        slot = seeded_db.get(AppointmentSlot, appointment.slot_id)
        assert slot is not None
        assert slot.status == SlotStatus.BOOKED

    def test_the_appointment_has_a_derived_reminder(self, seeded_db):
        appointment = seeded_db.query(Appointment).one()
        reminder = seeded_db.query(Reminder).one()
        assert reminder.appointment_id == appointment.id
        slot = seeded_db.get(AppointmentSlot, appointment.slot_id)
        assert reminder.scheduled_at == slot.start_time - timedelta(hours=24)

    def test_document_checksums_match_the_bytes_on_disk(self, seeded_db):
        documents = seeded_db.query(PatientDocument).all()
        assert len(documents) == 3
        for document in documents:
            payload = Path(document.storage_path).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == document.checksum

    def test_stored_filenames_are_server_generated(self, seeded_db):
        """The client-supplied filename is a path-traversal vector and is kept
        only as a display label."""
        for document in seeded_db.query(PatientDocument).all():
            assert Path(document.storage_path).name.startswith("seed-")
            assert document.original_filename is not None

    def test_one_document_is_filed_under_a_wrong_declared_type(self, seeded_db):
        """Verification needs a genuine mismatch to catch, not a synthetic one
        invented at test time."""
        declared = [d.declared_type for d in seeded_db.query(PatientDocument).all()]
        assert declared.count("ECG report") == 2


class TestDateRobustness:
    """The seed must behave identically on any boot day."""

    @pytest.mark.parametrize(
        "anchor",
        [
            pytest.param(date(2026, 3, 2), id="monday"),
            pytest.param(date(2026, 3, 3), id="tuesday"),
            pytest.param(date(2026, 3, 4), id="wednesday"),
            pytest.param(date(2026, 3, 5), id="thursday"),
            pytest.param(date(2026, 3, 6), id="friday"),
            pytest.param(date(2026, 3, 7), id="saturday"),
            pytest.param(date(2026, 3, 8), id="sunday"),
        ],
    )
    def test_self_check_passes_on_every_weekday(self, anchor):
        """Seeded on a Sunday — the day that generates no slots — the database
        must still be demo-ready."""
        clock.freeze(anchor)
        drop_all()
        create_all()
        session = SessionLocal()
        try:
            seed(session, anchor=anchor)
            session.commit()
            problems = self_check(session, anchor=anchor)
        finally:
            session.close()

        assert problems == [], f"seeded on {anchor:%A}: {problems}"

    @pytest.mark.parametrize("offset", [0, 5, 6])
    def test_the_booked_slot_never_lands_on_a_slotless_day(self, offset):
        """A relative date resolving onto a Sunday would strand the seeded run
        mid-flight, which is the failure this rule exists to prevent."""
        anchor = date(2026, 3, 2) + timedelta(days=offset)
        clock.freeze(anchor)
        drop_all()
        create_all()
        session = SessionLocal()
        try:
            seed(session, anchor=anchor)
            session.commit()
            appointment = session.query(Appointment).one()
            slot = session.get(AppointmentSlot, appointment.slot_id)
            assert slot.start_time.weekday() != 6
            assert slot.start_time.date() >= anchor
        finally:
            session.close()

    def test_no_slot_is_seeded_into_the_past(self, seeded_db):
        anchor = clock.today()
        earliest = (
            seeded_db.query(AppointmentSlot).order_by(AppointmentSlot.start_time).first()
        )
        assert earliest.start_time >= datetime.combine(anchor, time.min)

    def test_every_department_has_capacity_inside_the_window(self, seeded_db):
        anchor = clock.today()
        window_end = datetime.combine(anchor + timedelta(days=SLOT_WINDOW_DAYS), time.max)
        for dept_id, name, *_ in DEPARTMENTS:
            available = (
                seeded_db.query(AppointmentSlot)
                .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
                .filter(
                    Doctor.department_id == dept_id,
                    AppointmentSlot.status == SlotStatus.AVAILABLE,
                    AppointmentSlot.start_time <= window_end,
                )
                .count()
            )
            assert available > 0, f"{name} has no bookable capacity"


class TestSelfCheckCatchesProblems:
    """An alarm you have never tripped is a decoration."""

    def test_it_notices_a_live_workflow_run(self, seeded_db):
        from app.models import WorkflowStatus

        seeded_db.add(WorkflowRun(patient_id=1, status=WorkflowStatus.IN_PROGRESS))
        seeded_db.flush()
        problems = self_check(seeded_db, anchor=clock.today())
        assert any("non-terminal" in p for p in problems)

    def test_it_notices_a_booked_slot_released_behind_its_appointment(self, seeded_db):
        appointment = seeded_db.query(Appointment).one()
        seeded_db.get(AppointmentSlot, appointment.slot_id).status = SlotStatus.AVAILABLE
        seeded_db.flush()
        problems = self_check(seeded_db, anchor=clock.today())
        assert any("slot marked available" in p for p in problems)

    def test_it_notices_a_checksum_that_stopped_matching(self, seeded_db):
        document = seeded_db.query(PatientDocument).first()
        document.checksum = "0" * 64
        seeded_db.flush()
        problems = self_check(seeded_db, anchor=clock.today())
        assert any("checksum" in p for p in problems)

    def test_it_notices_a_missing_document_file(self, seeded_db):
        document = seeded_db.query(PatientDocument).first()
        document.storage_path = "/nonexistent/definitely-not-here.pdf"
        seeded_db.flush()
        problems = self_check(seeded_db, anchor=clock.today())
        assert any("file is missing" in p for p in problems)


class TestSeedCommandLine:
    """The script a judge actually runs."""

    def test_run_seeds_and_reports_success(self, capsys):
        from scripts.seed import run

        assert run() == 0
        output = capsys.readouterr().out
        assert "Seed self-check passed." in output
        assert "departments    10" in output

    def test_run_prints_the_demo_credentials(self, capsys):
        """A judge needs to be able to log in without reading the source."""
        from scripts.seed import DEMO_PASSWORD, run

        run()
        output = capsys.readouterr().out
        assert DEMO_PASSWORD in output
        assert "staff@example.invalid" in output

    def test_check_only_mode_makes_no_changes(self, capsys):
        from scripts.seed import run

        run()
        before = SessionLocal()
        try:
            slots = before.query(AppointmentSlot).count()
        finally:
            before.close()

        assert run(check_only=True) == 0

        after = SessionLocal()
        try:
            assert after.query(AppointmentSlot).count() == slots
        finally:
            after.close()

    def test_run_reports_failure_when_the_check_finds_a_problem(self, capsys):
        """The exit code is what CI reads, so it has to be honest."""
        from app.models import WorkflowStatus
        from scripts.seed import run

        run()
        session = SessionLocal()
        try:
            session.add(WorkflowRun(patient_id=1, status=WorkflowStatus.IN_PROGRESS))
            session.commit()
        finally:
            session.close()

        assert run(check_only=True) == 1
        assert "Seed self-check FAILED" in capsys.readouterr().out


class TestSamplePdf:
    def test_output_is_a_valid_looking_pdf(self):
        payload = build_pdf("Title", ["line one", "line two"])
        assert payload.startswith(b"%PDF-1.4")
        assert payload.rstrip().endswith(b"%%EOF")
        assert b"/Type /Catalog" in payload

    def test_output_is_byte_for_byte_deterministic(self):
        """Seed checksums are asserted on directly; a timestamp in the output
        would make every seeded run produce different ones."""
        assert build_pdf("T", ["a"]) == build_pdf("T", ["a"])

    def test_different_content_produces_different_bytes(self):
        assert build_pdf("T", ["a"]) != build_pdf("T", ["b"])

    def test_parentheses_in_text_are_escaped(self):
        """Unescaped parens terminate a PDF string and corrupt the file."""
        payload = build_pdf("Report (draft)", ["value (mg/dL)"])
        assert rb"\(draft\)" in payload
