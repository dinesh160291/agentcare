"""Reading the patient's answer to a proposal — in code, never in the model.

The rule, stated as an asymmetry: **a wrongly re-asked "yes" costs one tap; a
wrongly committed "no" books an appointment against the patient's word at the
exact step built to prevent that.** So commitment requires a click or an exact
token. Anything else is not a confirmation, however confident it sounds.

Reading order (the first two live here; the third is the model's, and its only
permitted verdicts are *decline* or *non-answer* — never confirm):

1. Confirm / Decline buttons, which arrive as a typed action and need no
   interpretation at all. They render with the first proposal, not as a
   fallback after the patient has already been misunderstood once.
2. Exact-token match on typed text, below.
3. Text the tokens cannot read.

The token sets are deliberately small and exact. "Yes, but can we do Tuesday
instead?" is *not* in them, and must not be: it is a question wearing a yes.
"""

from __future__ import annotations

import re

from app.models.enums import StrEnum


class ConfirmationAnswer(StrEnum):
    CONFIRM = "confirm"
    DECLINE = "decline"
    #: The tokens could not read it. Not a rejection — a handover.
    UNREAD = "unread"


#: Exact matches only, after normalisation. Every addition here widens what
#: counts as consent, so each one is a decision rather than a convenience.
CONFIRM_TOKENS = frozenset(
    {"yes", "y", "yeah", "yep", "confirm", "confirmed", "ok", "okay",
     "book it", "go ahead", "sounds good", "that works"}
)
DECLINE_TOKENS = frozenset(
    {"no", "n", "nope", "cancel", "decline", "stop", "not that one"}
)


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Punctuation only — nothing that could turn a longer sentence into a token.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())).strip()


def read_confirmation(text: str) -> ConfirmationAnswer:
    """Read a typed answer. Exact tokens only.

    A message that merely *contains* "yes" is not a confirmation: "yes, but
    actually no" contains it, and so does "yesterday".
    """
    token = normalise(text)
    if token in CONFIRM_TOKENS:
        return ConfirmationAnswer.CONFIRM
    if token in DECLINE_TOKENS:
        return ConfirmationAnswer.DECLINE
    return ConfirmationAnswer.UNREAD


__all__ = [
    "CONFIRM_TOKENS",
    "DECLINE_TOKENS",
    "ConfirmationAnswer",
    "normalise",
    "read_confirmation",
]
