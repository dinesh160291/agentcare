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
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import filetype
from sqlalchemy.orm import Session

from app.audit import write_audit
from app.auth.ownership import patient_profile_for
from app.config import get_settings
from app.errors import RecordNotFound
from app.models import DepartmentRequiredDocument, PatientDocument, User
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


def _stored_filename(checksum: str, extension: str) -> str:
    """Build a filename the server chose.

    The client-supplied name is never used on disk. It is an attacker-controlled
    string that reaches a filesystem call, and "../../etc/passwd" is a filename
    as far as an upload form is concerned. The original is kept in the database
    as a display label only.
    """
    return f"doc-{uuid.uuid4().hex}-{checksum[:12]}{extension}"


def ingest_document(
    session: Session,
    user: User,
    *,
    content: bytes,
    declared_type: str,
    original_filename: str | None = None,
    document_date: date | None = None,
) -> dict[str, Any]:
    """Store an uploaded document for the acting patient.

    Hardened on three axes, each closing a way an upload endpoint gets abused:
    a size cap, a MIME allowlist checked against **magic bytes** rather than the
    declared content type or the file extension (both of which the client
    chooses), and a server-generated filename.

    Duplicates are detected before anything is written to disk, so re-uploading
    the same file does not leave orphaned copies accumulating.
    """
    settings = get_settings()
    profile = patient_profile_for(session, user)

    def refuse(reason: str, message: str) -> dict[str, Any]:
        return {
            "ok": False,
            "reason": reason,
            "message": message,
            "document": None,
            "duplicate_of": None,
        }

    if not content:
        return refuse("empty_file", "That file is empty.")

    if len(content) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes / (1024 * 1024)
        return refuse("too_large", f"Files must be smaller than {limit_mb:.0f} MB.")

    if not (declared_type or "").strip():
        return refuse("missing_declared_type", "Please say what kind of document this is.")

    # Magic bytes, not the extension and not the client's content-type header.
    kind = filetype.guess(content)
    mime = kind.mime if kind else None
    if mime not in settings.allowed_upload_mime_list:
        allowed = ", ".join(settings.allowed_upload_mime_list)
        return refuse("unsupported_type", f"Only these file types are accepted: {allowed}.")

    checksum = checksum_bytes(content)
    duplicate = find_duplicate(session, patient_id=profile.id, checksum=checksum)
    if duplicate["is_duplicate"]:
        return {
            "ok": False,
            "reason": "duplicate",
            "message": "You have already uploaded this file.",
            "document": None,
            "duplicate_of": duplicate["existing_document_id"],
        }

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_name = _stored_filename(checksum, f".{kind.extension}")
    (upload_dir / stored_name).write_bytes(content)

    document = PatientDocument(
        patient_id=profile.id,
        declared_type=declared_type.strip(),
        document_type=declared_type.strip(),
        detected_type=None,
        storage_path=str(upload_dir / stored_name),
        original_filename=original_filename,
        mime_type=mime,
        size_bytes=len(content),
        document_date=document_date,
        checksum=checksum,
        # Verification is Phase 5's job; the file is on record either way.
        status=DocumentStatus.PENDING_VERIFICATION,
    )
    session.add(document)
    session.flush()

    write_audit(
        session,
        action="document_uploaded",
        entity_type="PatientDocument",
        entity_id=document.id,
        actor=user,
        metadata={"declared_type": document.declared_type, "mime_type": mime},
    )

    return {
        "ok": True,
        "reason": None,
        "message": "Document received.",
        "document": {
            "document_id": document.id,
            "document_type": document.document_type,
            "declared_type": document.declared_type,
            "status": document.status.value,
            "checksum": checksum,
            "size_bytes": len(content),
            "mime_type": mime,
        },
        "duplicate_of": None,
    }


#: Text extraction is bounded like everything else. A model does not need a
#: whole discharge summary to tell an ECG from an X-ray report, and an
#: unbounded extract is an unbounded prompt.
MAX_EXTRACT_CHARS = 4000


def extract_document_text(session: Session, document_id: int) -> dict[str, Any]:
    """Pull readable text out of a stored document.

    PDFs go through pypdf. **Images return no text and say so** — there is no
    OCR here, and that is a scope decision rather than an oversight: an image
    is classified by its declared type alone, so the verification step must be
    able to tell "this says nothing" apart from "this says nothing relevant".
    Silently returning an empty string for both would make every uploaded photo
    look like a mismatch.
    """
    document = session.get(PatientDocument, document_id)
    if document is None:
        return {"extracted": False, "reason": "not_found", "text": "", "pages": 0}

    if (document.mime_type or "") != "application/pdf":
        return {
            "extracted": False,
            "reason": "not_extractable",
            "text": "",
            "pages": 0,
            "mime_type": document.mime_type,
        }

    path = Path(document.storage_path)
    if not path.exists():
        return {"extracted": False, "reason": "file_missing", "text": "", "pages": 0}

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return {
        "extracted": True,
        "reason": None,
        "text": text.strip()[:MAX_EXTRACT_CHARS],
        "pages": len(reader.pages),
        "mime_type": document.mime_type,
    }


def apply_verification(
    session: Session,
    *,
    document_id: int,
    matches: bool,
    detected_type: str | None = None,
    note: str = "",
    actor: User | None = None,
) -> dict[str, Any]:
    """Record a verification verdict. **Code sets the status, not the model.**

    This is the decision-bin split at its clearest: the model proposes that
    content does not match its declared type — a reading task — and code turns
    that proposal into a ``flagged`` row and a human review. The model never
    writes a status, never rejects a document, and never decides what happens
    next.

    A flagged document stops satisfying a requirement, which is the point:
    the patient supplied *something*, but not something the hospital can use,
    and the follow-up task must stay open.
    """
    document = session.get(PatientDocument, document_id)
    if document is None:
        raise RecordNotFound("PatientDocument", document_id)

    document.detected_type = (detected_type or "").strip() or None
    document.status = (
        DocumentStatus.VERIFIED if matches else DocumentStatus.FLAGGED
    )
    document.verification_note = note.strip()[:500] or None
    session.flush()

    write_audit(
        session,
        action="document_verified" if matches else "document_flagged",
        entity_type="PatientDocument",
        entity_id=document.id,
        actor=actor,
        actor_kind="user" if actor else "system",
        metadata={
            "declared_type": document.declared_type,
            "detected_type": document.detected_type,
            "status": document.status.value,
        },
    )
    return _serialise_document(document)


def list_unverified_documents(
    session: Session, *, patient_id: int
) -> list[dict[str, Any]]:
    """Documents still awaiting a verification verdict, oldest first."""
    documents = (
        session.query(PatientDocument)
        .filter(
            PatientDocument.patient_id == patient_id,
            PatientDocument.status == DocumentStatus.PENDING_VERIFICATION,
        )
        .order_by(PatientDocument.id)
        .all()
    )
    return [_serialise_document(doc) for doc in documents]


def list_flagged_documents(session: Session) -> list[dict[str, Any]]:
    """The staff review queue: every document whose content did not match."""
    documents = (
        session.query(PatientDocument)
        .filter(PatientDocument.status == DocumentStatus.FLAGGED)
        .order_by(PatientDocument.id)
        .all()
    )
    return [_serialise_document(doc) for doc in documents]


def _serialise_document(doc: PatientDocument) -> dict[str, Any]:
    return {
        "document_id": doc.id,
        "patient_id": doc.patient_id,
        "document_type": doc.document_type,
        "declared_type": doc.declared_type,
        "detected_type": doc.detected_type,
        "status": doc.status.value,
        "verification_note": doc.verification_note,
        "document_date": doc.document_date.isoformat() if doc.document_date else None,
        "checksum": doc.checksum,
        "original_filename": doc.original_filename,
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
