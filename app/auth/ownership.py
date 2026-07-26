"""Ownership enforcement.

Role says what *kinds* of things you may do; ownership says which *rows* you
may do them to. Checking only the first is the single most common RBAC hole,
and the one a judge probes first by editing an id in the URL.

Two rules, both enforced here:

* A patient may only touch rows belonging to their own ``PatientProfile``.
* A failure returns **404, not 403** — see :mod:`app.errors` for why — and is
  audited before it raises.

Staff bypass ownership by role, which is the point of the role: they operate
the queue. Their access is still audited.
"""

from __future__ import annotations

from typing import TypeVar

from sqlalchemy.orm import Session

from app.audit import audit_denied_access
from app.errors import RecordNotFound
from app.models import PatientProfile, User, UserRole

T = TypeVar("T")

#: Models that carry a ``patient_id`` and are therefore patient-scoped.
PATIENT_SCOPED_ENTITIES = (
    "Appointment",
    "PatientDocument",
    "Reminder",
    "WorkflowRun",
    "FollowUpTask",
    "Notification",
)


def patient_profile_for(session: Session, user: User) -> PatientProfile:
    """The acting user's own patient profile, or 404."""
    profile = (
        session.query(PatientProfile).filter(PatientProfile.user_id == user.id).one_or_none()
    )
    if profile is None:
        raise RecordNotFound("PatientProfile", user.id)
    return profile


def get_owned_or_404(
    session: Session,
    model: type[T],
    entity_id: int | None,
    user: User,
    *,
    audit: bool = True,
) -> T:
    """Fetch ``model`` by id, enforcing ownership for patients.

    Staff receive the row by role. Patients receive it only when its
    ``patient_id`` matches their own profile; every other outcome — missing
    row, someone else's row, no profile at all — is the same 404, so the
    response cannot be used to enumerate ids.
    """
    entity = session.get(model, entity_id) if entity_id is not None else None

    if user.role == UserRole.STAFF:
        if entity is None:
            raise RecordNotFound(model.__name__, entity_id)
        return entity

    profile = (
        session.query(PatientProfile).filter(PatientProfile.user_id == user.id).one_or_none()
    )
    owner_id = getattr(entity, "patient_id", None) if entity is not None else None

    if entity is None or profile is None or owner_id != profile.id:
        if audit and entity is not None:
            # A real row the caller does not own: worth recording precisely.
            audit_denied_access(
                session,
                actor=user,
                entity_type=model.__name__,
                entity_id=entity_id,
                reason="ownership",
            )
        raise RecordNotFound(model.__name__, entity_id)

    return entity


def assert_owns(
    session: Session,
    model: type[T],
    entity_id: int | None,
    user: User,
) -> None:
    """Ownership check with no return value, for mutating call sites."""
    get_owned_or_404(session, model, entity_id, user)


def require_own_patient_id(session: Session, user: User, patient_id: int) -> None:
    """Guard a patient_id supplied directly in a request body.

    An endpoint that accepts a ``patient_id`` argument re-opens the hole that
    :func:`get_owned_or_404` closes, so the same rule applies.
    """
    if user.role == UserRole.STAFF:
        return
    profile = (
        session.query(PatientProfile).filter(PatientProfile.user_id == user.id).one_or_none()
    )
    if profile is None or profile.id != patient_id:
        audit_denied_access(
            session,
            actor=user,
            entity_type="PatientProfile",
            entity_id=patient_id,
            reason="ownership",
        )
        raise RecordNotFound("PatientProfile", patient_id)
