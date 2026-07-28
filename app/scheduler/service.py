"""Running :func:`~app.scheduler.poll.poll_once` on a timer.

Everything interesting is in ``poll.py``; this is the six lines that make it
happen periodically, kept apart so that nothing worth testing is trapped behind
a background thread.

One job, ``max_instances=1``, ``coalesce=True``. A tick that overruns must not
start a second copy of itself beside the first — two pollers racing the same
due rows is exactly the situation the unique constraint on
``Notification.reminder_id`` exists to survive, and surviving it is not a reason
to arrange it.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from app.config import get_settings
from app.scheduler.poll import poll_once

logger = logging.getLogger("agentcare.scheduler")

_scheduler: BackgroundScheduler | None = None

JOB_ID = "agentcare-poll"


def _tick() -> None:
    """One tick, with its own exception boundary.

    APScheduler swallows an unhandled job exception into its own logger and
    carries on, which is survivable but silent. A job that has been failing
    every minute for an hour should say so in this application's log, under
    this application's name.
    """
    try:
        result = poll_once()
    except Exception:  # noqa: BLE001 - logged, and the schedule continues
        logger.exception("poll tick failed")
        return
    if result.delivered or result.failed or result.completed:
        logger.info(
            "poll tick: %d delivered, %d failed, %d visits completed",
            len(result.delivered),
            len(result.failed),
            len(result.completed),
        )


def start() -> BackgroundScheduler | None:
    """Start the poll job, unless configuration says not to.

    ``None`` when disabled. Off in tests — they drive ``poll_once`` directly, so
    a thread ticking underneath them would be a source of writes no test asked
    for and every "nothing happened" assertion would be racing it.
    """
    global _scheduler
    settings = get_settings()
    if not settings.scheduler_enabled:
        logger.info("scheduler disabled by configuration")
        return None
    if _scheduler is not None:  # pragma: no cover - idempotent start
        return _scheduler

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        _tick,
        "interval",
        seconds=settings.reminder_poll_seconds,
        id=JOB_ID,
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    logger.info("scheduler started: every %ds", settings.reminder_poll_seconds)
    return _scheduler


def shutdown() -> None:
    """Stop the poll job if it is running. Safe to call when it is not."""
    global _scheduler
    if _scheduler is None:
        return
    _scheduler.shutdown(wait=False)
    _scheduler = None
    logger.info("scheduler stopped")


__all__ = ["JOB_ID", "shutdown", "start"]
