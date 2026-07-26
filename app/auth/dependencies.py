"""FastAPI dependencies for authentication and role enforcement.

RBAC lives in the backend, not in whether the UI drew a button. Every protected
route resolves the acting user from the token and re-reads the user row, so a
token cannot outlive a role change.
"""

from __future__ import annotations

from typing import Annotated, Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth.security import user_id_from_token
from app.db import get_session
from app.errors import AuthenticationError
from app.models import User, UserRole

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    session: Annotated[Session, Depends(get_session)],
) -> User:
    """Resolve the acting user from the Authorization header."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        user_id = user_id_from_token(credentials.credentials)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    # Re-read the row: the token's role claim is a convenience, not authority.
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account no longer exists",
        )
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_role(*roles: UserRole) -> Callable[[User], User]:
    """Dependency factory enforcing one of ``roles``."""

    def _dependency(user: CurrentUser) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this operation",
            )
        return user

    return _dependency


require_patient = require_role(UserRole.PATIENT)
require_staff = require_role(UserRole.STAFF)

PatientUser = Annotated[User, Depends(require_patient)]
StaffUser = Annotated[User, Depends(require_staff)]
