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

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
#: Deliberately loose: catches +1-555-0100, (555) 010 0100, 555.010.0100.
PHONE = re.compile(r"(?<!\w)(\+?\d[\d\s().-]{7,}\d)(?!\w)")


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
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
