"""Uploads and document reads.

The hardening lives in ``ingest_document`` — size cap, MIME allowlist checked
against magic bytes, server-generated filename — and this router's whole job on
the write path is to hand it the bytes and own the transaction. Two things it
must not do, both tempting and both fatal:

* **never touch ``UploadFile.filename`` except to store it as a label.** The
  client chooses that string, and a router that builds a path from it hands the
  caller the filesystem.
* **never trust ``content_type``.** It is also the client's, which is why the
  tool sniffs the bytes instead.

A refusal comes back as a shape-stable dict, so the status code is chosen from
its ``reason`` and the body is passed through unchanged. The alternative — a
bare ``detail`` string — would drop ``duplicate_of``, which is the one piece of
a duplicate refusal the patient can act on.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.auth.dependencies import CurrentUser, PatientUser
from app.auth.ownership import get_owned_or_404, patient_profile_for
from app.db import get_session
from app.models import PatientDocument
from app.tools import describe_document, ingest_document, list_patient_documents

router = APIRouter(tags=["documents"])

DbSession = Annotated[Session, Depends(get_session)]

#: Why the upload was refused → what the client should be told. Each is a
#: different remedy: shrink it, convert it, name it, or you already have it.
REFUSAL_STATUS: dict[str, int] = {
    "empty_file": 422,
    "missing_declared_type": 422,
    "too_large": 413,
    "unsupported_type": 415,
    "duplicate": 409,
}


@router.post("/documents", status_code=201)
async def upload_document(
    user: PatientUser,
    session: DbSession,
    file: Annotated[UploadFile, File()],
    declared_type: Annotated[str, Form()],
    document_date: Annotated[date | None, Form()] = None,
) -> Any:
    """Store a document the patient has declared the type of."""
    content = await file.read()
    result = ingest_document(
        session,
        user,
        content=content,
        declared_type=declared_type,
        # A label for the patient's benefit. It never reaches a path.
        original_filename=file.filename,
        document_date=document_date,
    )

    if not result["ok"]:
        # Nothing was written, so there is nothing to commit — and the tool's
        # own dict is the body, keys and all.
        return JSONResponse(
            status_code=REFUSAL_STATUS.get(result["reason"], 422), content=result
        )

    session.commit()
    return result


@router.get("/documents")
def list_documents(user: PatientUser, session: DbSession) -> list[dict[str, Any]]:
    profile = patient_profile_for(session, user)
    return list_patient_documents(session, patient_id=profile.id)


@router.get("/documents/{document_id}")
def read_document(
    document_id: int, user: CurrentUser, session: DbSession
) -> dict[str, Any]:
    document = get_owned_or_404(session, PatientDocument, document_id, user)
    return describe_document(document)
