"""Request and response models for the HTTP layer.

Request models set ``extra="forbid"``. That is a deliberate choice about how a
refusal reads: with extras ignored, a client posting ``{"role": "staff"}`` to
the register endpoint gets a 201 and no staff account, which looks exactly like
a bug that has not been noticed yet. Forbidding extras turns the same attempt
into a 422 — the refusal becomes visible to the caller, and to the next person
reading the tests.

Responses are plain models rather than ORM serializers: most of what this API
returns already exists as a tool result dict, and re-describing those shapes
here would be a second definition to keep in step with the first.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Deliberately loose. Address validity is the mail server's job; this only
#: rejects input that is obviously not an address, without a new dependency
#: (``EmailStr`` requires ``email-validator``).
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

#: Long enough to be worth hashing, short enough not to trip bcrypt's 72-byte
#: input limit — over which bcrypt silently truncates rather than failing.
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 72


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _validate_email(value: str) -> str:
    address = value.strip().lower()
    if not _EMAIL.match(address):
        raise ValueError("That does not look like an email address.")
    return address


# --- auth ---------------------------------------------------------------


class RegisterRequest(_Request):
    """Patient self-registration. There is no ``role`` field, on purpose."""

    name: str = Field(min_length=1, max_length=120)
    email: str = Field(max_length=255)
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)

    _normalise_email = field_validator("email")(_validate_email)


class LoginRequest(_Request):
    email: str = Field(max_length=255)
    password: str = Field(max_length=MAX_PASSWORD_LENGTH)

    _normalise_email = field_validator("email")(_validate_email)


class UserOut(BaseModel):
    user_id: int
    name: str
    email: str
    role: str


class TokenOut(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user_id: int
    name: str
    role: str


# --- patient profile ----------------------------------------------------


class ProfileUpdate(_Request):
    """Every field optional: this is a patch, and an absent field means
    "leave it alone" rather than "clear it"."""

    date_of_birth: date | None = None
    phone: str | None = Field(default=None, max_length=40)
    preferred_language: str | None = Field(default=None, max_length=40)
    emergency_contact: str | None = Field(default=None, max_length=160)


class ProfileOut(BaseModel):
    patient_id: int
    name: str
    email: str
    date_of_birth: date | None
    phone: str | None
    preferred_language: str
    emergency_contact: str | None


# --- the workflow front doors -------------------------------------------


class MessageRequest(_Request):
    """A free-text turn. ``session_id`` continues an existing conversation."""

    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=80)


class ActionRequest(_Request):
    """A Confirm/Decline click. Closed set, zero interpretation.

    ``session_id`` is required rather than optional: a typed action always
    belongs to a conversation the patient is already in, and inventing one
    would bracket the turn under a session nobody can find again.
    """

    action: Literal["confirm", "decline"]
    session_id: str = Field(min_length=1, max_length=80)


class TurnOut(BaseModel):
    reply: str
    author: str
    turn_id: str
    session_id: str
    run_id: int | None = None
    status: str | None = None
    message_class: str | None = None
    plan: list[str] = Field(default_factory=list)
    steps_run: list[str] = Field(default_factory=list)

    @classmethod
    def of(cls, result: Any) -> "TurnOut":
        """Serialize a ``TurnResult``.

        A classmethod rather than a helper in the router because a turn now
        leaves the system through two doors: the router returns one, and the
        provider-failure handler in ``app.api.errors`` serves one for a turn
        that raised. Two serializers would be two shapes, and the one nobody
        looks at is the one that drifts.
        """
        return cls(
            reply=result.reply,
            author=result.author.value,
            turn_id=result.turn_id,
            session_id=result.session_id,
            run_id=result.run_id,
            status=result.status,
            message_class=result.message_class.value if result.message_class else None,
            plan=result.plan,
            steps_run=result.steps_run,
        )


class RunOut(BaseModel):
    run_id: int
    patient_id: int
    status: str
    current_step: str | None
    plan: list[str]
    completed_steps: list[str]
    request_text: str
    department_name: str | None
    proposed_action: str | None
    proposed_slot_id: int | None
    #: The proposal as a patient reads it. A slot id and an action enum say
    #: nothing to the person being asked to agree to them, and the screen must
    #: not derive these itself — a card showing a doctor and a time it worked
    #: out locally is a card that can disagree with the row.
    proposed_doctor_name: str | None = None
    proposed_day: str | None = None
    proposed_time: str | None = None
    session_id: str | None
    created_at: str
    updated_at: str


# --- documents ----------------------------------------------------------


class DocumentResolution(_Request):
    action: Literal["accept", "reclassify", "reject"]
    corrected_type: str | None = Field(default=None, max_length=80)
    note: str = Field(default="", max_length=500)


# --- staff decisions ----------------------------------------------------


class StaffDecisionRequest(_Request):
    action: Literal["approve", "reject", "redirect"]
    department_name: str | None = Field(default=None, max_length=120)
    note: str = Field(default="", max_length=500)


class EscalationResolution(_Request):
    """Safety escalations are acknowledged and resolved — never approved.

    The vocabulary split is enforced in ``workflow.staff``; repeating it in the
    schema means the wrong word is refused before it reaches a session, and
    means this file cannot be read as though the two lifecycles were one.
    """

    status: Literal["acknowledged", "resolved", "approved", "rejected"]
    note: str = Field(default="", max_length=500)


class ActiveToggle(_Request):
    active: bool


class SlotRequest(_Request):
    """Staff adding capacity for one doctor."""

    start_times: list[str] = Field(min_length=1, max_length=48)
    duration_minutes: int = Field(default=30, ge=5, le=240)


class TraceEventOut(BaseModel):
    seq: int
    turn_id: str
    event_type: str
    author: str | None
    agent_name: str | None
    correlation_id: str | None
    payload: dict[str, Any]
    created_at: str


class AuditEventOut(BaseModel):
    id: int
    actor_id: int | None
    actor_kind: str
    action: str
    entity_type: str
    entity_id: int | None
    metadata: dict[str, Any]
    created_at: str
