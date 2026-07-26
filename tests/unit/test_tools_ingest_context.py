"""Document ingest, patient context, and confirmation rendering."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from app import clock
from app.errors import RecordNotFound
from app.models import PatientDocument, User
from app.tools.confirmations import render_confirmation
from app.tools.documents import ingest_document
from app.tools.patients import get_patient_context
from scripts.sample_pdf import build_pdf

MONDAY = date(2026, 8, 3)


@pytest.fixture(autouse=True)
def world(db):
    clock.freeze(datetime(2026, 8, 3, 8, 0))
    from scripts.seed import seed

    seed(db, anchor=MONDAY)
    db.commit()
    return db


@pytest.fixture
def patient(db):
    return db.query(User).filter(User.id == 1).one()


@pytest.fixture
def patient_b(db):
    return db.query(User).filter(User.id == 2).one()


@pytest.fixture
def staff(db):
    return db.query(User).filter(User.id == 5).one()


PDF = build_pdf("SYNTHETIC REPORT", ["nothing real here"])


class TestDocumentIngest:
    def test_a_valid_pdf_is_stored(self, db, patient_b):
        result = ingest_document(db, patient_b, content=PDF, declared_type="ECG report")
        assert result["ok"] is True
        assert result["document"]["mime_type"] == "application/pdf"

    def test_the_file_lands_on_disk(self, db, patient_b):
        result = ingest_document(db, patient_b, content=PDF, declared_type="ECG report")
        document = db.get(PatientDocument, result["document"]["document_id"])
        assert Path(document.storage_path).read_bytes() == PDF

    def test_the_stored_filename_is_server_generated(self, db, patient_b):
        """The client filename reaches a filesystem call; "../../etc/passwd" is
        a perfectly ordinary filename as far as an upload form is concerned."""
        result = ingest_document(
            db, patient_b, content=PDF, declared_type="ECG report",
            original_filename="../../../etc/passwd",
        )
        document = db.get(PatientDocument, result["document"]["document_id"])
        name = Path(document.storage_path).name
        assert name.startswith("doc-")
        assert ".." not in document.storage_path

    def test_the_original_filename_is_kept_only_as_a_label(self, db, patient_b):
        result = ingest_document(
            db, patient_b, content=PDF, declared_type="ECG report",
            original_filename="my ecg.pdf",
        )
        document = db.get(PatientDocument, result["document"]["document_id"])
        assert document.original_filename == "my ecg.pdf"

    def test_an_oversized_file_is_refused(self, db, patient_b, monkeypatch):
        from app.config import get_settings

        monkeypatch.setattr(get_settings(), "max_upload_bytes", 10, raising=False)
        result = ingest_document(db, patient_b, content=PDF, declared_type="ECG report")
        assert result["ok"] is False
        assert result["reason"] == "too_large"

    def test_a_disguised_executable_is_refused(self, db, patient_b):
        """Magic bytes, not the extension: naming a PNG "report.pdf" must not
        get it past the allowlist, and neither must the reverse."""
        result = ingest_document(
            db, patient_b, content=b"MZ\x90\x00" + b"\x00" * 200,
            declared_type="ECG report", original_filename="report.pdf",
        )
        assert result["ok"] is False
        assert result["reason"] == "unsupported_type"

    def test_a_refused_file_is_not_written_to_disk(self, db, patient_b):
        before = db.query(PatientDocument).count()
        ingest_document(db, patient_b, content=b"MZ\x90\x00" + b"\x00" * 200,
                        declared_type="ECG report")
        assert db.query(PatientDocument).count() == before

    def test_an_empty_file_is_refused(self, db, patient_b):
        assert ingest_document(db, patient_b, content=b"",
                               declared_type="ECG report")["reason"] == "empty_file"

    def test_a_missing_declared_type_is_refused(self, db, patient_b):
        """The patient declares the type; verification checks it later."""
        assert ingest_document(db, patient_b, content=PDF,
                               declared_type="  ")["reason"] == "missing_declared_type"

    def test_re_uploading_the_same_file_is_a_duplicate(self, db, patient_b):
        first = ingest_document(db, patient_b, content=PDF, declared_type="ECG report")
        second = ingest_document(db, patient_b, content=PDF, declared_type="ECG report")
        assert second["ok"] is False
        assert second["reason"] == "duplicate"
        assert second["duplicate_of"] == first["document"]["document_id"]

    def test_a_duplicate_leaves_no_second_copy(self, db, patient_b):
        ingest_document(db, patient_b, content=PDF, declared_type="ECG report")
        before = db.query(PatientDocument).count()
        ingest_document(db, patient_b, content=PDF, declared_type="ECG report")
        assert db.query(PatientDocument).count() == before

    def test_two_patients_may_hold_the_same_file(self, db, patient, patient_b):
        assert ingest_document(db, patient_b, content=PDF, declared_type="ECG report")["ok"]
        assert ingest_document(db, patient, content=PDF, declared_type="ECG report")["ok"]

    def test_upload_is_audited(self, db, patient_b):
        from app.models import AuditEvent

        ingest_document(db, patient_b, content=PDF, declared_type="ECG report")
        assert "document_uploaded" in {e.action for e in db.query(AuditEvent).all()}

    def test_success_and_refusal_share_a_shape(self, db, patient_b):
        ok = ingest_document(db, patient_b, content=PDF, declared_type="ECG report")
        bad = ingest_document(db, patient_b, content=b"", declared_type="ECG report")
        assert set(ok) == set(bad)


class TestPatientContext:
    def test_a_patient_sees_their_own_record(self, db, patient):
        context = get_patient_context(db, patient)
        assert context["patient_id"] == 1
        assert context["name"] == "Asha Menon"

    def test_seeded_documents_and_appointments_are_included(self, db, patient):
        context = get_patient_context(db, patient)
        assert len(context["documents"]) == 3
        assert len(context["upcoming_appointments"]) == 1

    def test_appointments_carry_what_a_reply_needs(self, db, patient):
        appointment = get_patient_context(db, patient)["upcoming_appointments"][0]
        assert set(appointment) >= {"department_name", "doctor_name", "start", "reference_code"}

    def test_a_patient_with_nothing_gets_empty_collections(self, db, patient_b):
        context = get_patient_context(db, patient_b)
        assert context["upcoming_appointments"] == []
        assert context["documents"] == []
        assert context["active_run"] is None

    def test_a_patient_cannot_read_another_patients_context(self, db, patient_b):
        with pytest.raises(RecordNotFound):
            get_patient_context(db, patient_b, patient_id=1)

    def test_staff_may_read_any_patients_context(self, db, staff):
        assert get_patient_context(db, staff, patient_id=1)["patient_id"] == 1

    def test_staff_must_name_a_patient(self, db, staff):
        """Staff have no profile of their own to fall back on."""
        with pytest.raises(ValueError):
            get_patient_context(db, staff)

    def test_an_active_run_is_reported(self, db, patient_b):
        from app.models import ProposedAction, WorkflowRun, WorkflowStatus

        db.add(
            WorkflowRun(
                patient_id=2,
                status=WorkflowStatus.PENDING_CONFIRMATION,
                proposed_action=ProposedAction.BOOK,
            )
        )
        db.flush()
        active = get_patient_context(db, patient_b)["active_run"]
        assert active["status"] == "pending_confirmation"
        assert active["has_pending_proposal"] is True

    def test_a_terminal_run_is_not_reported_as_active(self, db, patient_b):
        from app.models import WorkflowRun, WorkflowStatus

        db.add(WorkflowRun(patient_id=2, status=WorkflowStatus.COMPLETED))
        db.flush()
        assert get_patient_context(db, patient_b)["active_run"] is None

    def test_context_is_json_serialisable(self, db, patient):
        import json

        json.dumps(get_patient_context(db, patient))


class TestRenderConfirmation:
    def test_facts_are_read_back_from_the_persisted_row(self, db):
        result = render_confirmation(db, 1)
        assert result["facts"]["doctor_name"] == "Dr. Anita Rao"
        assert result["facts"]["department_name"] == "Cardiology"
        assert result["facts"]["reference_code"] == "AC-000001"

    def test_the_weekday_matches_the_stored_date(self, db):
        """The drift that matters: a fluent sentence naming the wrong day sends
        the patient to the hospital on the wrong morning."""
        from app.models import Appointment, AppointmentSlot

        slot = db.get(AppointmentSlot, db.get(Appointment, 1).slot_id)
        assert result_weekday(db) == f"{slot.start_time:%A}"

    def test_the_sentence_contains_every_fact(self, db):
        result = render_confirmation(db, 1)
        for value in ("Dr. Anita Rao", "Cardiology", "AC-000001"):
            assert value in result["sentence"]

    def test_the_sentence_carries_no_clinical_language(self, db):
        sentence = render_confirmation(db, 1)["sentence"].lower()
        for word in ("diagnos", "prescrib", "dose", "dosage", "condition", "symptom"):
            assert word not in sentence

    def test_a_missing_appointment_raises(self, db):
        with pytest.raises(RecordNotFound):
            render_confirmation(db, 999_999)

    def test_the_result_is_json_serialisable(self, db):
        import json

        json.dumps(render_confirmation(db, 1))


def result_weekday(db) -> str:
    return render_confirmation(db, 1)["facts"]["weekday"]
