"""Reading which of the times already shown the patient picked — in code.

The twin of :mod:`app.workflow.confirmation`, one step earlier in the same
conversation, and it exists because that step had no reader at all.

**The hole.** A run at ``pending_confirmation`` holds a slot, and "lets book the
4pm one" moves the offer to 4pm — that machinery works and is not touched here.
A run at ``in_progress`` holds nothing, and there the same sentence had nowhere
to land: the Coordinator classified it as a new request, ``_shows_no_difference``
correctly refused to supersede the run with itself, and the refusal's consequence
is to answer with *more times*. So the reply asked for a time, a time arrived,
and the reply asked for a time again. Live, seven consecutive messages — "okay
lets book at 3pm then", "lets do 3pm and book it", "option 2", "3pm will work for
me", "2pm will work for me", "confirm 2pm slot" — each drew the identical
availability list. The patient could not complete a booking from that state by
any wording, and a second run died the same way in the same session.

Nothing was wrong with the refusal. What was missing is that a patient naming a
time they were just shown is *selecting*, not asking, and selection is a
deterministic act: the candidates are rows this run rendered, and matching a
clock time against them needs no judgement.

**What this may and may not do.** It holds a slot and asks. It cannot commit —
the run lands in ``pending_confirmation`` and the pinned rule is untouched: only
an exact token or the ✅ button books anything. That is what makes the trade
here the opposite of the confirmation reader's. A wrongly read *confirmation*
books against the patient's word; a wrongly read *selection* holds a time they
then decline, costing one message. So this reads more freely than
:func:`~app.workflow.confirmation.read_confirmation` does, and it is still
bounded by the same gate.

**Three ways a patient names a slot**, in the order they are tried:

1. **A list number** — "option 2", "2", "the second one". Answerable only
   against the ordered list this run last rendered, which is why
   :func:`~app.workflow.replies.record_shortlist` exists: ``offered_slot_ids``
   is a union over the whole run and a union has no second element.
2. **A clock time in the numbered list** — "3pm will work for me". The list in
   front of the patient is the referent, so it is searched first and a match
   there ends the read.
3. **A clock time anywhere the run has offered** — "actually the 4pm one you
   showed me earlier". Wider, so it requires a *unique* match: two offered slots
   at 3pm on different days is a question, not an answer, and guessing which day
   is exactly the kind of choice that belongs to language.

A message carrying a withdrawal cue is never a selection, checked first, on the
same reasoning that puts withdrawal ahead of the appointment verbs in
:func:`~app.workflow.mapping.names_change_verb`: "forget it" and "book it" must
not be read by the same rule in the same breath.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from app.workflow.confirmation import normalise
from app.workflow.mapping import says_withdrawal


@dataclass(frozen=True)
class Offer:
    """One slot the patient has been shown: an id and when it starts."""

    slot_id: int
    start: datetime


#: "3pm", "3 pm", "3:30pm", "at 10 am". The meridiem is required — a bare "3"
#: is a list number, and reading it as 3 o'clock would silently answer a
#: different question. ``\b`` on the left keeps "AC-000003pm" and the tail of a
#: reference code out of it.
_MERIDIEM = re.compile(
    r"\b(1[0-2]|0?[1-9])(?::([0-5][0-9]))?\s*([ap])\.?\s?m\.?\b", re.IGNORECASE
)

#: "15:00", "09:30". Unambiguous without a meridiem because of the colon.
_TWENTY_FOUR = re.compile(r"\b([01]?[0-9]|2[0-3]):([0-5][0-9])\b")

#: "option 2", "number 2", "#2", "no. 2" — a digit that announces itself as a
#: position. Kept separate from the bare-digit case below because this one may
#: appear in a longer sentence and that one may not.
_NUMBERED = re.compile(r"\b(?:option|number|no\.?|choice|#)\s*(\d{1,2})\b", re.IGNORECASE)

#: "the 2nd one", "the 3rd please".
_ORDINAL_DIGIT = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)

_ORDINAL_WORDS = {"first": 1, "second": 2, "third": 3}

#: A message that is *only* a number. "2" answering a numbered list is the whole
#: message; a "2" inside a sentence is far more likely to be a date, a count or
#: a reference than a position, so it is not read here.
_BARE_NUMBER = re.compile(r"^(\d{1,2})$")


def times_named(text: str) -> set[tuple[int, int]]:
    """Every (hour, minute) this message states, as 24-hour pairs.

    The meridiem pattern runs first and its matches are *removed* before the
    24-hour one looks. Both patterns match the "3:00" inside "3:00 PM", and
    leaving them both to fire yielded ``{(3, 0), (15, 0)}`` — two times, which
    this module reads as a range and refuses. A sentence naming one time would
    have been silently unreadable, which is the failure mode that looks like
    nothing being wrong at all.
    """
    message = text or ""
    found: set[tuple[int, int]] = set()
    for match in _MERIDIEM.finditer(message):
        hour, minute, meridiem = match.groups()
        value = int(hour) % 12
        if meridiem.lower() == "p":
            value += 12
        found.add((value, int(minute or 0)))
    remainder = _MERIDIEM.sub(" ", message)
    for hour, minute in _TWENTY_FOUR.findall(remainder):
        found.add((int(hour), int(minute)))
    return found


def position_named(text: str) -> int | None:
    """The 1-based list position this message states, if it states one.

    Public because slots are not the only numbered list a patient answers:
    :mod:`app.workflow.targets` reads the same three notations against the
    appointment choice. What differs between the two readers is what a position
    *means* and what may outrank it, never how "option 2" is spelled.
    """
    message = text or ""
    match = _NUMBERED.search(message)
    if match:
        return int(match.group(1))

    normalised = normalise(message)
    for word, position in _ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", normalised):
            return position

    match = _ORDINAL_DIGIT.search(message)
    if match:
        return int(match.group(1))

    match = _BARE_NUMBER.match(normalised)
    if match:
        return int(match.group(1))
    return None


def read_selection(
    text: str,
    *,
    shortlist: list[Offer],
    offered: list[Offer],
) -> int | None:
    """The slot id this message picks, or ``None`` if it picks none.

    ``None`` is the ordinary answer and costs nothing: the turn carries on to
    the Coordinator exactly as it did before, so every message this cannot read
    is handled by whatever handled it yesterday.
    """
    message = text or ""
    if not message.strip():
        return None

    # "Forget it" names no slot, and a rule that read it as one would turn an
    # abandonment into a booking. Checked before anything else for the same
    # reason `names_change_verb` checks it before the appointment verbs.
    if says_withdrawal(message):
        return None

    times = times_named(message)

    # A position, but only when the message is not also naming a time — "book
    # the 2pm one" contains a 2 that is an hour and not a row number, and
    # `position_named` would happily read it as one.
    if not times:
        position = position_named(message)
        if position is not None and 1 <= position <= len(shortlist):
            return shortlist[position - 1].slot_id
        # A number outside the list is not a near miss to be rounded into
        # range. Falling through hands it to the model, which is where a
        # message nobody anticipated belongs.

    if len(times) != 1:
        # Two times is a range or a comparison ("anything between 2pm and 4pm"),
        # which is a question about availability rather than a choice among
        # answers. It already has a path; this one would answer it wrongly.
        return None
    wanted = next(iter(times))

    # The list in front of the patient first. It is the referent of "the 3pm
    # one" whatever else the run has shown over its life.
    for offer in shortlist:
        if (offer.start.hour, offer.start.minute) == wanted:
            return offer.slot_id

    # Then everything this run ever offered — but only if the time names exactly
    # one of them. "3pm" against a Tuesday 3pm and a Wednesday 3pm is a question
    # the patient has not answered, and picking the earlier one would book a day
    # they did not choose.
    matches = [
        offer.slot_id
        for offer in offered
        if (offer.start.hour, offer.start.minute) == wanted
    ]
    if len(matches) == 1:
        return matches[0]
    return None


__all__ = ["Offer", "position_named", "read_selection", "times_named"]
