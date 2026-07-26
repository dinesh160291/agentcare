"""Password hashing and JWT issue/verify.

bcrypt for passwords, PyJWT for tokens. No password ever reaches the database,
the logs, or a trace row.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import bcrypt
import jwt

from app import clock
from app.config import get_settings
from app.errors import AuthenticationError, ValidationFailed

#: bcrypt hashes at most 72 bytes and silently ignores the rest. Rejecting
#: longer input is safer than truncating it, which would make two different
#: long passwords interchangeable.
MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    """Hash a password with a per-password salt."""
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValidationFailed(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes when UTF-8 encoded"
        )
    if not password:
        raise ValidationFailed("Password must not be empty")
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Check a password against a stored hash. Never raises on bad input."""
    try:
        encoded = password.encode("utf-8")[:MAX_PASSWORD_BYTES]
        return bcrypt.checkpw(encoded, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(
    *,
    user_id: int,
    role: str,
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Issue a signed JWT for ``user_id``.

    The role is stamped into the token for convenience only — every request
    re-reads the user row, so a token claiming ``staff`` for a demoted account
    grants nothing.
    """
    settings = get_settings()
    issued_at = clock.now()
    expires_at = issued_at + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Verify and decode a JWT, or raise ``AuthenticationError``.

    Expiry is checked against :func:`app.clock.now`, not PyJWT's own clock.
    PyJWT compares ``exp`` to real wall-clock time, which would bypass the
    clock seam entirely: with ``APP_TODAY`` pinned to a past date, every token
    the application issued would be born already expired and nobody could log
    in. Signature verification stays with PyJWT; only the time comparison moves.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"verify_exp": False},
        )
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Invalid authentication token") from exc

    expires_at = payload.get("exp")
    if expires_at is None:
        raise AuthenticationError("Token is missing its expiry claim")
    if clock.now().timestamp() >= float(expires_at):
        raise AuthenticationError("Session has expired; please log in again")

    return payload


def user_id_from_token(token: str) -> int:
    """Extract the subject claim as an int, or raise ``AuthenticationError``."""
    payload = decode_access_token(token)
    subject = payload.get("sub")
    if subject is None:
        raise AuthenticationError("Token is missing its subject claim")
    try:
        return int(subject)
    except (TypeError, ValueError) as exc:
        raise AuthenticationError("Token subject is not a valid user id") from exc
