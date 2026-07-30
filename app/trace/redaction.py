"""PII and secret redaction.

Applied at the choke points before anything is persisted or logged. Two
separate jobs live here:

* **Secrets** — API keys, tokens, passwords, authorization headers. These must
  never reach a trace row, a log line, or a staff timeline.
* **Contact PII** — phone numbers and email addresses. Traces are read by staff
  during review and by anyone debugging, which is a wider audience than the
  record itself has.

Redaction is by key name *and* by value pattern, because the same phone number
arrives sometimes as ``{"phone": ...}`` and sometimes buried in free text.

Key matching comes in two kinds, and the difference is load-bearing. Secrets are
matched on **substrings**, so ``GROQ_API_KEY`` and ``user_password_hash`` are
both caught. Identity fields are matched on **whole key names**, because the
obvious substring version of that rule would treat ``id`` as sensitive and
redact every foreign key in the system — a trace with no slot ids explains
nothing at all.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[redacted]"

#: Substring match against lower-cased keys — a key containing any of these is
#: replaced wholesale, whatever its value looks like.
SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "jwt",
)

#: Whole key names, never substrings. These are the PRD's "phone, email, DOB,
#: IDs": fields whose *value* is the patient's identity rather than the
#: system's own bookkeeping.
SENSITIVE_KEY_NAMES = (
    "dob",
    "date_of_birth",
    "phone",
    "phone_number",
    "mobile",
    "email",
    "email_address",
    "national_id",
    "insurance_number",
    "mrn",
)

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
#: An ISO date is ten characters of digits and hyphens, which is exactly what a
#: loose phone pattern looks for. Left alone it ate every search window in the
#: trace — ``find_available_slots(start="[redacted]", end="[redacted]")``, on the
#: one question a reviewer asks first. Dates are excluded here and a date of
#: birth is caught by its **key** instead, which is where the guarantee belongs:
#: a DOB typed into the chat is already stored raw in the run's request text and
#: the model's history by design, so masking it in the trace alone was never the
#: protection it looked like.
_ISO_DATE = r"(?!\d{4}-\d{2}-\d{2}(?!\d))"
#: Deliberately loose otherwise: catches +1-555-0100, (555) 010 0100, 555.010.0100.
PHONE = re.compile(r"(?<!\w)" + _ISO_DATE + r"(\+?\d[\d\s().-]{7,}\d)(?!\w)")


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in SENSITIVE_KEY_NAMES:
        return True
    return any(part in lowered for part in SENSITIVE_KEY_PARTS)


def redact_text(text: str) -> str:
    """Mask contact details inside a free-text string."""
    masked = EMAIL.sub(REDACTED, text)
    return PHONE.sub(REDACTED, masked)


def redact(value: Any) -> Any:
    """Recursively redact a JSON-able structure.

    Returns a new structure; the caller's object is never mutated, because a
    redactor that edits its input in place would quietly corrupt the row the
    application is still working with.
    """
    if isinstance(value, dict):
        return {
            key: (REDACTED if _is_sensitive_key(str(key)) else redact(val))
            for key, val in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value
