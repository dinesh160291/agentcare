"""The clock seam.

Nothing in application code calls ``date.today()`` or ``datetime.now()``
directly — everything goes through here. Golden tests and scenario evals need
a frozen clock, and a single stray direct call is enough to make a suite that
passes in July fail in August.

Time is **naive local time** throughout the application. Appointments are wall
clock events at a hospital; mixing naive and aware datetimes is a reliable
source of comparison bugs, so the codebase commits to naive and says so here.

Two override mechanisms, both for tests and demos only — the live app always
reports the real current date:

* ``APP_TODAY`` (environment) — pins the date. Used by the seed script and by
  reproducible demos.
* :func:`freeze` / :func:`unfreeze` (programmatic) — pins a full datetime.
  Used by tests that need time to advance deterministically.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from typing import Iterator

from app.config import get_settings

#: When ``APP_TODAY`` pins the date but no full datetime is supplied, the time
#: of day is anchored here so that "now" is deterministic rather than drifting
#: with the wall clock.
ANCHOR_TIME = time(9, 0)

_frozen_now: datetime | None = None


def freeze(moment: datetime | date) -> None:
    """Pin :func:`now` and :func:`today` to ``moment`` (tests only)."""
    global _frozen_now
    if isinstance(moment, datetime):
        _frozen_now = moment
    else:
        _frozen_now = datetime.combine(moment, ANCHOR_TIME)


def unfreeze() -> None:
    """Release a :func:`freeze`, returning to real (or ``APP_TODAY``) time."""
    global _frozen_now
    _frozen_now = None


def advance(delta: timedelta) -> datetime:
    """Move a frozen clock forward. Requires :func:`freeze` first."""
    global _frozen_now
    if _frozen_now is None:
        raise RuntimeError("advance() requires a frozen clock; call freeze() first")
    _frozen_now = _frozen_now + delta
    return _frozen_now


@contextmanager
def frozen_at(moment: datetime | date) -> Iterator[datetime]:
    """Freeze the clock for the duration of a ``with`` block."""
    previous = _frozen_now
    freeze(moment)
    try:
        yield now()
    finally:
        globals()["_frozen_now"] = previous


def now() -> datetime:
    """Current local time, honouring any freeze or ``APP_TODAY`` override."""
    if _frozen_now is not None:
        return _frozen_now
    override = get_settings().app_today
    if override is not None:
        return datetime.combine(override, ANCHOR_TIME)
    return datetime.now()


def today() -> date:
    """Current local date, honouring any freeze or ``APP_TODAY`` override."""
    return now().date()


def is_overridden() -> bool:
    """True when the clock is not reporting real time.

    Surfaced in the UI and in traces so a demo running on a pinned date is
    never mistaken for a live one.
    """
    return _frozen_now is not None or get_settings().app_today is not None
