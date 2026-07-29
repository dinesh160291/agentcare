"""Deterministic date resolution.

The model never resolves a date. It passes the patient's phrase through
unchanged and this module decides what the phrase means, against
:func:`app.clock.today`. An LLM given "next week" with no anchor will resolve
it differently across runs while remaining schema-valid and plausible, and a
plausible wrong date survives every check except the patient reading it back.

Refusal is a first-class outcome. An unparseable phrase returns
``resolved: False`` and no dates at all — guessing one is the exact failure
this module exists to prevent.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from typing import Any

from app import clock

WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "a": 1, "an": 1, "couple": 2, "few": 3,
}

PARTS_OF_DAY = {
    "morning": "morning",
    "afternoon": "afternoon",
    "evening": "evening",
    "night": "evening",
}

_MONTH_NAMES = "|".join(MONTHS)

#: Weekday names as patients write them: plural ("Tuesdays"), possessive
#: ("Tuesday's"), abbreviated ("tue", "tues"). Longest-first so "tuesday" wins
#: the alternation ahead of "tue" and the suffix group has something to match.
#:
#: Exported because ``workflow/targets.py`` reads weekday cues out of a
#: patient's message too, and two regexes over one vocabulary is how the two
#: drift: a plural this module learns and that one does not is a reschedule
#: that stops recognising the day it was told.
WEEKDAY_PATTERN = re.compile(
    rf"\b({'|'.join(sorted(WEEKDAYS, key=len, reverse=True))})(?:'s|’s|s)?\b",
    re.IGNORECASE,
)

#: How far an open-ended window runs. "After the 6th" names no end and a search
#: needs one; two weeks matches the horizon the seed lays slots across, so the
#: answer covers everything actually bookable rather than trailing into empty
#: calendar.
OPEN_ENDED_DAYS = 13


def _result(
    *,
    phrase: str,
    resolved: bool,
    start: date | None = None,
    end: date | None = None,
    kind: str | None = None,
    part_of_day: str | None = None,
    label: str = "",
    reason: str | None = None,
) -> dict[str, Any]:
    """Build the shape-stable result dict every path returns."""
    return {
        "phrase": phrase,
        "resolved": resolved,
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "kind": kind,
        "part_of_day": part_of_day,
        "label": label,
        "reason": reason,
    }


def _format(day: date) -> str:
    return f"{day.day} {day:%B} {day.year}"


def _label_for(start: date, end: date) -> str:
    if start == end:
        return f"{start:%A} {_format(start)}"
    return f"{_format(start)} to {_format(end)}"


def _extract_part_of_day(text: str) -> tuple[str, str | None]:
    """Pull a time-of-day preference out, returning the remaining text."""
    for word, value in PARTS_OF_DAY.items():
        pattern = rf"\b{word}\b"
        if re.search(pattern, text):
            return re.sub(pattern, " ", text).strip(), value
    return text, None


def _count_in(text: str, unit: str) -> int | None:
    """Read "in 3 days" / "in three weeks" — digits or spelled out."""
    match = re.search(rf"\bin\s+(\d+)\s+{unit}s?\b", text)
    if match:
        return int(match.group(1))
    words = "|".join(NUMBER_WORDS)
    match = re.search(rf"\bin\s+({words})\s+{unit}s?\b", text)
    if match:
        return NUMBER_WORDS[match.group(1)]
    return None


def _explicit_date(text: str, today: date) -> date | None:
    """Parse an unambiguous written date, or return None.

    Numeric day/month forms such as "3/8" are deliberately unsupported: they
    mean different dates on either side of the Atlantic, and a tool whose job
    is to stop guessing should not guess here either.
    """
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None

    day_first = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAMES})\b", text)
    month_first = re.search(rf"\b({_MONTH_NAMES})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", text)

    if day_first:
        day, month = int(day_first.group(1)), MONTHS[day_first.group(2)]
    elif month_first:
        month, day = MONTHS[month_first.group(1)], int(month_first.group(2))
    else:
        return None

    for year in (today.year, today.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            return None
        # "3 January" said in August means the January that is coming.
        if candidate >= today:
            return candidate
    return None


#: "after the 6th", "from Tuesday", "before August 10th". The verb decides
#: whether the named day is included, and *after* excluding it is the whole
#: point: live, "after August 6th" answered with August 6th.
_BOUNDARY = re.compile(r"\b(after|from|starting|before|until|till|by)\b")


def _bounded_by_a_date(text: str, today: date) -> tuple[date, date, str] | None:
    """A window opened or closed by a named day, or ``None``.

    Only fires when a boundary word sits beside something this module can
    already resolve to a day, so it adds a *direction* to the existing
    vocabulary rather than a second parser. An open end runs to
    :data:`OPEN_ENDED_DAYS`, because a search needs one and the bookable
    horizon is the honest place to stop.
    """
    boundary = _BOUNDARY.search(text)
    if boundary is None:
        return None

    anchor = _explicit_date(text, today)
    if anchor is None:
        named = WEEKDAY_PATTERN.search(text)
        if named is None:
            return None
        index = WEEKDAYS[named.group(1).lower()]
        anchor = today + timedelta(days=(index - today.weekday()) % 7 or 7)

    word = boundary.group(1)
    if word in ("after",):
        start = anchor + timedelta(days=1)
        return start, start + timedelta(days=OPEN_ENDED_DAYS), "range"
    if word in ("from", "starting"):
        return anchor, anchor + timedelta(days=OPEN_ENDED_DAYS), "range"
    if word in ("before",):
        return today, anchor - timedelta(days=1), "range"
    # "until"/"till"/"by" include the day named.
    return today, anchor, "range"


def _weekday_in_a_week(text: str, today: date) -> date | None:
    """"Tuesday next week", "Thursday the week after next" — or ``None``.

    Both halves of these phrases match branches of their own, and neither
    branch is right on its own: the weekday loop answers with *this* week's
    Tuesday, and the period branch answers with the whole week. The compound
    has to be read before either.
    """
    named = WEEKDAY_PATTERN.search(text)
    if named is None:
        return None
    index = WEEKDAYS[named.group(1).lower()]

    if re.search(r"\bweek after next\b", text):
        weeks = 2
    elif re.search(r"\bnext week\b", text):
        weeks = 1
    else:
        return None

    monday = today + timedelta(days=(0 - today.weekday()) % 7 or 7)
    return monday + timedelta(days=index, weeks=weeks - 1)


def resolve_date(phrase: str, *, today: date | None = None) -> dict[str, Any]:
    """Resolve a natural-language date phrase to a concrete date or range.

    Returns a shape-stable dict. ``kind`` is ``"exact"`` for a single day and
    ``"range"`` when the patient named a period rather than a day — the
    distinction matters, because offering slots across a range is honest where
    picking one day out of it is not.
    """
    today = today or clock.today()
    original = phrase or ""
    text = original.strip().lower()

    if not text:
        return _result(phrase=original, resolved=False, reason="unparseable")

    text, part_of_day = _extract_part_of_day(text)

    def unreadable(reason: str) -> dict[str, Any]:
        """A refusal that still carries what *was* read.

        The part of day is extracted before the date is parsed, so a phrase
        whose date fails had its "afternoon" understood and thrown away with
        it. Live: "more slots in the afternoon?" searched unfiltered and came
        back with 10 and 11 AM. No dates are returned — that part of the
        refusal is unchanged and is the whole point of the module.
        """
        return _result(
            phrase=original, resolved=False, reason=reason, part_of_day=part_of_day
        )

    def ok(start: date, end: date, kind: str) -> dict[str, Any]:
        if end < today:
            return _result(phrase=original, resolved=False, reason="in_the_past")
        # A range that has already started is trimmed to what is still bookable.
        start = max(start, today)
        return _result(
            phrase=original,
            resolved=True,
            start=start,
            end=end,
            kind=kind,
            part_of_day=part_of_day,
            label=_label_for(start, end),
        )

    # --- plain offsets ---------------------------------------------------
    if re.search(r"\btoday\b", text):
        return ok(today, today, "exact")
    if re.search(r"\bday after tomorrow\b", text):
        return ok(today + timedelta(days=2), today + timedelta(days=2), "exact")
    if re.search(r"\btomorrow\b", text):
        return ok(today + timedelta(days=1), today + timedelta(days=1), "exact")
    if re.search(r"\byesterday\b", text):
        return _result(phrase=original, resolved=False, reason="in_the_past")

    days = _count_in(text, "day")
    if days is not None:
        target = today + timedelta(days=days)
        return ok(target, target, "exact")

    weeks = _count_in(text, "week")
    if weeks is not None:
        start = today + timedelta(weeks=weeks)
        return ok(start, start + timedelta(days=6), "range")

    # --- compounds -------------------------------------------------------
    # "Tuesday next week" names a day *and* the week it falls in, and both
    # halves match branches below — the weekday loop would answer with this
    # week's Tuesday and "next week" with the whole week. Neither is what was
    # asked, and the whole-week answer is the one that looks right.
    compound = _weekday_in_a_week(text, today)
    if compound is not None:
        return ok(compound, compound, "exact")

    # --- named periods ---------------------------------------------------
    # "The week after next" is checked ahead of "next week", which it contains.
    if re.search(r"\bweek after next\b", text):
        next_monday = today + timedelta(days=7 - today.weekday())
        start = next_monday + timedelta(days=7)
        return ok(start, start + timedelta(days=6), "range")

    if re.search(r"\bweekends?\b", text):
        # The coming Saturday and Sunday. Saturday is 5.
        ahead = (5 - today.weekday()) % 7
        saturday = today + timedelta(days=ahead)
        return ok(saturday, saturday + timedelta(days=1), "range")

    if re.search(r"\bweekdays?\b", text):
        # Monday to Friday of the working week the patient is asking about:
        # this one while it still has days left, otherwise the next.
        ahead = (0 - today.weekday()) % 7 if today.weekday() > 4 else 0
        monday = today + timedelta(days=ahead)
        return ok(max(monday, today), monday + timedelta(days=4), "range")

    if re.search(r"\bnext week\b", text):
        # Monday of the week after the one containing today, whatever day it is.
        next_monday = today + timedelta(days=7 - today.weekday())
        return ok(next_monday, next_monday + timedelta(days=6), "range")

    if re.search(r"\bthis week\b", text):
        # From today, not from Monday: the earlier half is not bookable.
        sunday = today + timedelta(days=6 - today.weekday())
        return ok(today, sunday, "range")

    if re.search(r"\bnext month\b", text):
        year = today.year + (1 if today.month == 12 else 0)
        month = 1 if today.month == 12 else today.month + 1
        last = calendar.monthrange(year, month)[1]
        return ok(date(year, month, 1), date(year, month, last), "range")

    # --- boundaries ------------------------------------------------------
    # After the periods and before the plain-date branch. Before, because
    # "after august 6th" contains a date that branch reads as *the 6th itself*
    # — live, that answered a question excluding the 6th by showing it. After,
    # because "week after next" contains the word "after" and this branch would
    # otherwise read it as a boundary on the Tuesday beside it.
    boundary = _bounded_by_a_date(text, today)
    if boundary is not None:
        start, end, kind = boundary
        return ok(start, end, kind)

    # --- explicit calendar dates ----------------------------------------
    explicit = _explicit_date(text, today)
    if explicit is not None:
        return ok(explicit, explicit, "exact")
    if re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", text):
        # Looked like a date and was not one (e.g. 30 February).
        return unreadable("unparseable")

    # --- weekday names ---------------------------------------------------
    named = WEEKDAY_PATTERN.search(text)
    if named:
        name = named.group(1).lower()
        index = WEEKDAYS[name]
        # A bare weekday means the next one coming, never today: today's slots
        # are already partly gone.
        ahead = (index - today.weekday()) % 7 or 7
        if re.search(rf"\bnext\s+{name}", text):
            ahead += 7
        target = today + timedelta(days=ahead)
        return ok(target, target, "exact")

    return unreadable("unparseable")
