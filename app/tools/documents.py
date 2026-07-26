"""Document queries: the required-documents diff and duplicate detection.

The diff distinguishes **mandatory** from **optional** shortfalls, which is why
the required-docs rules are a table rather than a delimited column. "Missing"
and "nice to have" produce different replies and different follow-up tasks, and
a diff that cannot tell them apart nags patients about paperwork nobody needs.

Duplicate detection is exact-checksum only, and that limit is deliberate and
documented: a re-exported PDF with identical content but different bytes reads
as a new document. Catching that needs content comparison, which is out of
scope here.
"""

from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy.orm import Session

from app.models import DepartmentRequiredDocument, PatientDocument
from app.models.enums import DocumentStatus

#: Document statuses that count as "the patient has supplied this".
SATISFYING_STATUSES = (
    DocumentStatus.PENDING_VERIFICATION,
    DocumentStatus.VERIFIED,
)


def checksum_bytes(payload: bytes) -> str:
    """SHA-256 of a file's contents — the duplicate-detection key."""
    return hashlib.sha256(payload).hexdigest()


def diff_required_documents(
    session: Session, *, patient_id: int, department_id: int
) -> dict[str, Any]:
    """Compare what a department requires against what the patient has filed.

    A ``flagged`` or ``rejected`` document does not satisfy a requirement: the
    patient supplied *something*, but not something the hospital can use, and
    reporting it as satisfied would close a follow-up task that should stay
    open.
    """
    rules = (
        session.query(DepartmentRequiredDocument)
        .filter(DepartmentRequiredDocument.department_id == department_id)
        .order_by(DepartmentRequiredDocument.document_type)
        .all()
    )

    held = (
        session.query(PatientDocument)
        .filter(
            PatientDocument.patient_id == patient_id,
            PatientDocument.status.in_(SATISFYING_STATUSES),
        )
        .all()
    )
    held_types = {doc.document_type.strip().lower() for doc in held}

    required = [{"document_type": r.document_type, "mandatory": r.mandatory} for r in rules]
    missing_mandatory = [
        r.document_type
        for r in rules
        if r.mandatory and r.document_type.strip().lower() not in held_types
    ]
    missing_optional = [
        r.document_type
        for r in rules
        if not r.mandatory and r.document_type.strip().lower() not in held_types
    ]
    satisfied = [
        r.document_type for r in rules if r.document_type.strip().lower() in held_types
    ]

    return {
        "patient_id": patient_id,
        "department_id": department_id,
        "required": required,
        "satisfied": satisfied,
        "missing_mandatory": missing_mandatory,
        "missing_optional": missing_optional,
        "complete": not missing_mandatory,
    }


def find_duplicate(
    session: Session, *, patient_id: int, checksum: str
) -> dict[str, Any]:
    """Look for an existing document with the same checksum for this patient.

    Scoped to the patient: two patients legitimately holding byte-identical
    paperwork is not a duplicate, and treating it as one would leak the
    existence of another patient's file.
    """
    existing = (
        session.query(PatientDocument)
        .filter(
            PatientDocument.patient_id == patient_id,
            PatientDocument.checksum == checksum,
        )
        .order_by(PatientDocument.id)
        .first()
    )
    return {
        "checksum": checksum,
        "is_duplicate": existing is not None,
        "existing_document_id": existing.id if existing else None,
        "existing_document_type": existing.document_type if existing else None,
    }


def list_patient_documents(session: Session, *, patient_id: int) -> list[dict[str, Any]]:
    """Every document on file for a patient, newest first."""
    documents = (
        session.query(PatientDocument)
        .filter(PatientDocument.patient_id == patient_id)
        .order_by(PatientDocument.created_at.desc(), PatientDocument.id.desc())
        .all()
    )
    return [
        {
            "document_id": doc.id,
            "document_type": doc.document_type,
            "declared_type": doc.declared_type,
            "detected_type": doc.detected_type,
            "status": doc.status.value,
            "document_date": doc.document_date.isoformat() if doc.document_date else None,
            "checksum": doc.checksum,
            "original_filename": doc.original_filename,
        }
        for doc in documents
    ]
