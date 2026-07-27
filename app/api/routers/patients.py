"""The patient's own record.

There is no ``/patients/{id}`` here on purpose. An endpoint that takes a
patient id invites the one-digit edit, and the only patient this router can
ever mean is the one holding the token — so the id never enters the URL and
there is nothing to probe. Staff read a patient through the staff router,
where the role check is the thing granting access.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.schemas import ProfileOut, ProfileUpdate
from app.audit import write_audit
from app.auth.dependencies import PatientUser
from app.auth.ownership import patient_profile_for
from app.db import get_session
from app.models import PatientProfile, User
from app.tools import get_patient_context

router = APIRouter(prefix="/patients", tags=["patients"])

DbSession = Annotated[Session, Depends(get_session)]


def _out(profile: PatientProfile, user: User) -> ProfileOut:
    return ProfileOut(
        patient_id=profile.id,
        name=user.name,
        email=user.email,
        date_of_birth=profile.date_of_birth,
        phone=profile.phone,
        preferred_language=profile.preferred_language,
        emergency_contact=profile.emergency_contact,
    )


@router.get("/me", response_model=ProfileOut)
def read_profile(user: PatientUser, session: DbSession) -> ProfileOut:
    return _out(patient_profile_for(session, user), user)


@router.patch("/me", response_model=ProfileOut)
def update_profile(
    payload: ProfileUpdate, user: PatientUser, session: DbSession
) -> ProfileOut:
    """Patch the caller's own profile.

    ``exclude_unset`` is what makes this a patch rather than a replace: a field
    the client did not send keeps its value, and a field sent explicitly as
    ``null`` is cleared. Collapsing those two would let a form that renders
    three of four fields silently erase the fourth.
    """
    profile = patient_profile_for(session, user)
    changes = payload.model_dump(exclude_unset=True)

    for field, value in changes.items():
        # The language column is non-nullable and has a default; a null there
        # means "back to the default", not "no language".
        if field == "preferred_language" and value is None:
            continue
        setattr(profile, field, value)

    if changes:
        write_audit(
            session,
            action="profile_updated",
            entity_type="PatientProfile",
            entity_id=profile.id,
            actor=user,
            # Field *names* only. The values are the PII this endpoint exists
            # to store, and an audit log is read by a wider audience than the
            # row is.
            metadata={"fields": sorted(changes)},
        )
    session.commit()
    session.refresh(profile)
    return _out(profile, user)


@router.get("/me/context")
def read_context(user: PatientUser, session: DbSession) -> dict[str, Any]:
    """Everything already known about the caller — the tool's own shape.

    Re-describing this as a response model would be a second definition of a
    dict the agents already consume, and the two would drift.
    """
    return get_patient_context(session, user)
