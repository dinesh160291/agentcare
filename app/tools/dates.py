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

    # --- named periods ---------------------------------------------------
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

    # --- explicit calendar dates ----------------------------------------
    explicit = _explicit_date(text, today)
    if explicit is not None:
        return ok(explicit, explicit, "exact")
    if re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", text):
        # Looked like a date and was not one (e.g. 30 February).
        return _result(phrase=original, resolved=False, reason="unparseable")

    # --- weekday names ---------------------------------------------------
    for name, index in WEEKDAYS.items():
        if not re.search(rf"\b{name}\b", text):
            continue
        # A bare weekday means the next one coming, never today: today's slots
        # are already partly gone.
        ahead = (index - today.weekday()) % 7 or 7
        if re.search(rf"\bnext\s+{name}\b", text):
            ahead += 7
        target = today + timedelta(days=ahead)
        return ok(target, target, "exact")

    return _result(phrase=original, resolved=False, reason="unparseable")
