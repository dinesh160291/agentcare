"""Registration, login, and "who am I".

Two rules this router exists to hold:

**Registration creates patients only.** Staff accounts come from the seed
script, which is where a hospital's own onboarding would sit. The request
schema has no ``role`` field and forbids extras, so the attempt is refused
rather than ignored.

**A failed login says the same thing however it failed.** Wrong password and
unknown address return one status and one body; only the audit log records
which of the two happened, and it records it by user id rather than by the
address that was typed — an audit table is read by more people than the user
row is, and an email address in its metadata is a PII leak arriving through
the one door the trace redactor does not watch.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.schemas import LoginRequest, RegisterRequest, TokenOut, UserOut
from app.audit import write_audit
from app.auth.dependencies import CurrentUser
from app.auth.security import create_access_token, hash_password, verify_password
from app.db import get_session
from app.models import PatientProfile, User, UserRole

router = APIRouter(prefix="/auth", tags=["auth"])

DbSession = Annotated[Session, Depends(get_session)]

#: One message for every way a login can fail.
_LOGIN_FAILED = "Email or password is incorrect"


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, session: DbSession) -> UserOut:
    """Create a patient account and its profile, in one transaction."""
    existing = session.query(User).filter(User.email == payload.email).one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists",
        )

    user = User(
        name=payload.name.strip(),
        email=payload.email,
        password_hash=hash_password(payload.password),
        role=UserRole.PATIENT,
    )
    session.add(user)
    session.flush()

    # A patient without a profile is a patient no tool can act for: every
    # patient-scoped row hangs off the profile id, not the user id.
    session.add(PatientProfile(user_id=user.id))
    write_audit(
        session,
        action="user_registered",
        entity_type="User",
        entity_id=user.id,
        actor=user,
        metadata={"role": user.role.value},
    )
    session.commit()

    return UserOut(user_id=user.id, name=user.name, email=user.email, role=user.role.value)


@router.post("/login", response_model=TokenOut)
def login(payload: LoginRequest, session: DbSession) -> TokenOut:
    """Exchange credentials for a bearer token. Both outcomes are audited."""
    user = session.query(User).filter(User.email == payload.email).one_or_none()

    if user is None or not verify_password(payload.password, user.password_hash):
        write_audit(
            session,
            action="login_failed",
            entity_type="User",
            entity_id=user.id if user is not None else None,
            actor=user,
            metadata={"reason": "bad_password" if user is not None else "unknown_email"},
        )
        session.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_LOGIN_FAILED,
            headers={"WWW-Authenticate": "Bearer"},
        )

    write_audit(
        session,
        action="login_succeeded",
        entity_type="User",
        entity_id=user.id,
        actor=user,
        metadata={"role": user.role.value},
    )
    session.commit()

    return TokenOut(
        access_token=create_access_token(user_id=user.id, role=user.role.value),
        user_id=user.id,
        name=user.name,
        role=user.role.value,
    )


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> UserOut:
    """The acting user, read back from the row the token resolved to."""
    return UserOut(user_id=user.id, name=user.name, email=user.email, role=user.role.value)
