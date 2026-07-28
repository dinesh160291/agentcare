"""The poll job — the only part of the system with no actor behind it.

A patient presses Confirm, staff press Approve, and nobody presses "the visit
is over". Anything triggered by the passage of time rather than by a message or
a click needs a process that watches the clock, and this is it: one periodic
job, two queries, both answered from the database rather than from anything the
scheduler remembers.
"""

from __future__ import annotations

from app.scheduler.delivery import deliver
from app.scheduler.poll import PollResult, poll_once
from app.scheduler.service import shutdown, start

__all__ = ["PollResult", "deliver", "poll_once", "shutdown", "start"]
