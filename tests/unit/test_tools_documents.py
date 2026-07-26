"""The required-documents diff and duplicate detection."""

from __future__ import annotations

import pytest

from app.models import DocumentStatus, PatientDocument
from app.tools.documents import (
    checksum_bytes,
    diff_required_documents,
    find_duplicate,
    list_patient_documents,
)

CARDIOLOGY = 1
GENERAL_MEDICINE = 4
ENT = 7


def _add(db, patient_id: int, document_type: str, checksum: str, status=DocumentStatus.VERIFIED):
    db.add(
        PatientDocument(
            patient_id=patient_id,
            declared_type=document_type,
            document_type=document_type,
            storage_path=f"/synthetic/{checksum}.pdf",
            checksum=checksum,
            status=status,
        )
    )
    db.flush()


class TestRequiredDocumentsDiff:
    def test_a_patient_with_nothing_is_missing_everything_mandatory(self, seeded_db):
        result = diff_required_documents(seeded_db, patient_id=2, department_id=CARDIOLOGY)
        assert set(result["missing_mandatory"]) == {"ECG report", "Blood test report"}
        assert result["complete"] is False

    def test_supplying_one_leaves_the_other_missing(self, seeded_db):
        _add(seeded_db, 2, "ECG report", "aa" * 32)
        result = diff_required_documents(seeded_db, patient_id=2, department_id=CARDIOLOGY)
        assert result["satisfied"] == ["ECG report"]
        assert result["missing_mandatory"] == ["Blood test report"]

    def test_supplying_all_mandatory_documents_completes_the_diff(self, seeded_db):
        _add(seeded_db, 2, "ECG report", "aa" * 32)
        _add(seeded_db, 2, "Blood test report", "bb" * 32)
        result = diff_required_documents(seeded_db, patient_id=2, department_id=CARDIOLOGY)
        assert result["missing_mandatory"] == []
        assert result["complete"] is True

    def test_a_department_requiring_nothing_is_complete_immediately(self, seeded_db):
        result = diff_required_documents(
            seeded_db, patient_id=2, department_id=GENERAL_MEDICINE
        )
        assert result["required"] == []
        assert result["complete"] is True

    def test_an_optional_shortfall_does_not_block_completeness(self, seeded_db):
        """This is why the rules are a table with a mandatory flag: nagging a
        patient for optional paperwork is a different reply, and a different
        follow-up task, from telling them something is genuinely missing."""
        result = diff_required_documents(seeded_db, patient_id=2, department_id=ENT)
        assert result["missing_optional"] == ["Previous audiometry report"]
        assert result["missing_mandatory"] == []
        assert result["complete"] is True

    def test_a_flagged_document_does_not_satisfy_a_requirement(self, seeded_db):
        """The patient supplied something the hospital cannot use. Counting it
        would close a follow-up task that needs to stay open."""
        _add(seeded_db, 2, "ECG report", "cc" * 32, status=DocumentStatus.FLAGGED)
        result = diff_required_documents(seeded_db, patient_id=2, department_id=CARDIOLOGY)
        assert "ECG report" in result["missing_mandatory"]

    def test_a_rejected_document_does_not_satisfy_a_requirement(self, seeded_db):
        _add(seeded_db, 2, "ECG report", "dd" * 32, status=DocumentStatus.REJECTED)
        result = diff_required_documents(seeded_db, patient_id=2, department_id=CARDIOLOGY)
        assert "ECG report" in result["missing_mandatory"]

    def test_a_document_awaiting_verification_does_satisfy_it(self, seeded_db):
        """Verification is asynchronous; the patient has done their part."""
        _add(seeded_db, 2, "ECG report", "ee" * 32, status=DocumentStatus.PENDING_VERIFICATION)
        result = diff_required_documents(seeded_db, patient_id=2, department_id=CARDIOLOGY)
        assert "ECG report" in result["satisfied"]

    def test_matching_ignores_case_and_padding(self, seeded_db):
        _add(seeded_db, 2, "  ecg REPORT  ", "ff" * 32)
        result = diff_required_documents(seeded_db, patient_id=2, department_id=CARDIOLOGY)
        assert "ECG report" in result["satisfied"]

    def test_another_patients_documents_are_not_counted(self, seeded_db):
        """Patient 1 is seeded with an ECG report; patient 2 must not benefit."""
        result = diff_required_documents(seeded_db, patient_id=2, department_id=CARDIOLOGY)
        assert "ECG report" in result["missing_mandatory"]

    def test_output_ordering_is_deterministic(self, seeded_db):
        first = diff_required_documents(seeded_db, patient_id=2, department_id=CARDIOLOGY)
        second = diff_required_documents(seeded_db, patient_id=2, department_id=CARDIOLOGY)
        assert first == second


class TestDuplicateDetection:
    def test_an_identical_checksum_is_a_duplicate(self, seeded_db):
        _add(seeded_db, 2, "ECG report", "11" * 32)
        result = find_duplicate(seeded_db, patient_id=2, checksum="11" * 32)
        assert result["is_duplicate"] is True
        assert result["existing_document_type"] == "ECG report"

    def test_a_new_checksum_is_not_a_duplicate(self, seeded_db):
        assert find_duplicate(seeded_db, patient_id=2, checksum="22" * 32)["is_duplicate"] is False

    def test_duplicate_detection_is_scoped_to_one_patient(self, seeded_db):
        """Two patients holding byte-identical paperwork is not a duplicate —
        and reporting it as one would leak that another patient's file exists."""
        _add(seeded_db, 1, "ECG report", "33" * 32)
        result = find_duplicate(seeded_db, patient_id=2, checksum="33" * 32)
        assert result["is_duplicate"] is False

    def test_checksum_of_identical_bytes_matches(self):
        assert checksum_bytes(b"same") == checksum_bytes(b"same")

    def test_checksum_of_different_bytes_differs(self):
        """The documented limit: a re-exported file with the same content but
        different bytes reads as new. Catching that needs content comparison."""
        assert checksum_bytes(b"report v1") != checksum_bytes(b"report v1 ")


class TestListing:
    def test_seeded_documents_are_listed_for_their_owner(self, seeded_db):
        documents = list_patient_documents(seeded_db, patient_id=1)
        assert len(documents) == 3

    def test_a_patient_with_no_documents_gets_an_empty_list(self, seeded_db):
        assert list_patient_documents(seeded_db, patient_id=3) == []

    def test_results_are_json_serialisable(self, seeded_db):
        import json

        json.dumps(list_patient_documents(seeded_db, patient_id=1))
        json.dumps(diff_required_documents(seeded_db, patient_id=1, department_id=CARDIOLOGY))
