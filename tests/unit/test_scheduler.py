"""The poll job: reminders that actually fire, and visits that end by themselves.

Written before the implementation, like everything else in the deterministic
bin. There is no model here at all — the scheduler is pure code reacting to the
clock, which is precisely why it can be pinned this hard.

Two duties in one tick, and they are together on purpose. Both are triggered by
**the passage of time rather than by a message or a click**, and that is the
only thing in this system with no actor: a patient presses Confirm, staff press
Approve, and nobody presses "the visit is over". A process that watches the
clock is the answer to both.

Three properties get the most attention here, because each one is a way the job
could look healthy while being useless:

* **Restart safety.** The table is the source of truth, never scheduler memory.
  A reminder written before a crash still fires after it, which is what DB
  polling buys and what per-reminder in-memory jobs would lose.
* **Poisoned rows.** Restart safety cuts both ways: a row whose delivery throws
  would otherwise be retried every tick forever and could take its batchmates
  down with it. The attempt counter is incremented *before* the attempt, so a
  crash mid-delivery still costs an attempt.
* **Ordering.** Deliver, then mark. A crash in between costs a bounded
  re-attempt that the unique constraint on ``Notification.reminder_id`` makes
  invisible — never a duplicate, and never an unbounded re-delivery.
"""

from __future__ import annotations

import re
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app import clock
from app import main as app_module
from app.db import SessionLocal
from app.models import (
    Appointment,
    AppointmentStatus,
    AuditEvent,
    FollowUpTask,
    FollowUpTaskStatus,
    FollowUpTaskType,
    Notification,
    Reminder,
    ReminderStatus,
    ReminderType,
    TraceEvent,
    User,
)
from app.config import get_settings
from app.errors import PermissionDenied, ValidationFailed
from app.scheduler import poll_once
from app.scheduler import service
from app.workflow.staff import apply_visit_decision

SEEDED_APPOINTMENT_ID = 1
ASHA_PROFILE_ID = 1


def fresh():
    return SessionLocal()


def _reminder(session, *, at, patient_id=ASHA_PROFILE_ID, message="Reminder.") -> Reminder:
    reminder = Reminder(
        patient_id=patient_id,
        appointment_id=SEEDED_APPOINTMENT_ID,
        reminder_type=ReminderType.APPOINTMENT,
        scheduled_at=at,
        status=ReminderStatus.PENDING,
        message=message,
    )
    session.add(reminder)
    session.flush()
    return reminder


def _audits(session, action: str) -> int:
    return session.query(AuditEvent).filter(AuditEvent.action == action).count()


@pytest.fixture
def staff_user(seeded_db):
    return seeded_db.query(User).filter(User.email == "staff@example.invalid").one()


@pytest.fixture
def patient_user(seeded_db):
    return seeded_db.query(User).filter(
        User.email == "asha.patient@example.invalid"
    ).one()


@pytest.fixture
def world(seeded_db):
    """A seeded database with the clock frozen at a known moment.

    Frozen rather than real: every assertion below is about *when* something
    happens, and a test that reports on the wall clock reports on the wrong
    thing. The seed's own reminder is retired first so each test states its
    own due rows and nothing arrives from the scenery.
    """
    for reminder in seeded_db.query(Reminder).all():
        reminder.status = ReminderStatus.CANCELLED
    seeded_db.commit()
    with clock.frozen_at(clock.now()) as moment:
        yield moment


class TestDeliveringDueReminders:
    def test_a_due_reminder_is_delivered_and_marked(self, seeded_db, world):
        reminder = _reminder(seeded_db, at=world - timedelta(minutes=1))
        seeded_db.commit()
        reminder_id = reminder.id

        result = poll_once()

        assert result.delivered == [reminder_id]
        session = fresh()
        try:
            row = session.get(Reminder, reminder_id)
            assert row.status is ReminderStatus.SENT
            assert row.sent_at is not None
            assert row.attempts == 1
            notification = (
                session.query(Notification)
                .filter(Notification.reminder_id == reminder_id)
                .one()
            )
            assert notification.patient_id == ASHA_PROFILE_ID
        finally:
            session.close()

    def test_a_reminder_that_is_not_due_yet_is_left_alone(self, seeded_db, world):
        reminder = _reminder(seeded_db, at=world + timedelta(hours=1))
        seeded_db.commit()

        result = poll_once()

        assert result.delivered == []
        session = fresh()
        try:
            assert session.get(Reminder, reminder.id).status is ReminderStatus.PENDING
        finally:
            session.close()

    def test_a_cancelled_reminder_is_never_delivered(self, seeded_db, world):
        """The consumer the derivation invariant was written for. Cancelling an
        appointment cancels its reminders in the same transaction; if the poll
        job delivered them anyway, that whole discipline would buy nothing."""
        reminder = _reminder(seeded_db, at=world - timedelta(hours=2))
        reminder.status = ReminderStatus.CANCELLED
        seeded_db.commit()

        assert poll_once().delivered == []

    def test_delivery_is_audited_and_leaves_no_trace_rows(self, seeded_db, world):
        """The trace/audit split, stated as a test. The poll job acts with no
        run and no session, so its writes belong in the row's history and not
        in any turn's story — the timeline does not "miss" reminder sends, they
        are in the other ledger by design."""
        _reminder(seeded_db, at=world - timedelta(minutes=1))
        seeded_db.commit()
        before = fresh()
        try:
            traces = before.query(TraceEvent).count()
        finally:
            before.close()

        poll_once()

        session = fresh()
        try:
            assert _audits(session, "reminder_delivered") == 1
            assert session.query(TraceEvent).count() == traces
        finally:
            session.close()

    def test_what_reaches_the_patient_says_the_time_the_way_people_do(
        self, seeded_db, world
    ):
        """Phase 8 is what turned this from a latent inconsistency into a
        patient-facing one. The message was composed at booking time and, until
        a poll job existed, read by nobody — the seed wrote ``09:00`` while
        every chat reply beside it said 9:00 AM about the same slot, which
        reads as two appointments rather than two notations.

        Asserted on the **delivered notification**, not on the row: the row is
        where the string is written and the notification is where a patient
        meets it, and only the second is the promise.
        """
        for reminder in seeded_db.query(Reminder).all():
            reminder.status = ReminderStatus.PENDING
            reminder.scheduled_at = world - timedelta(minutes=1)
        seeded_db.commit()

        poll_once()

        session = fresh()
        try:
            bodies = [row.body for row in session.query(Notification).all()]
        finally:
            session.close()

        assert bodies
        for body in bodies:
            assert "AM" in body or "PM" in body, body
            assert not re.search(r"\d{2}:\d{2}", body), body

    def test_a_second_tick_does_not_deliver_it_again(self, seeded_db, world):
        _reminder(seeded_db, at=world - timedelta(minutes=1))
        seeded_db.commit()

        poll_once()
        assert poll_once().delivered == []


class TestOrderingIsDeliverThenMark:
    def test_a_crash_after_delivery_costs_one_invisible_re_attempt(
        self, seeded_db, world, monkeypatch
    ):
        """The pinned ordering. Delivery lands, the process dies before the row
        is marked, and the next tick tries again — the unique constraint on
        ``Notification.reminder_id`` makes the second delivery a no-op instead
        of a second message to the patient.

        Marking first would be the other failure and the worse one: a reminder
        recorded as sent that nobody ever received.
        """
        reminder = _reminder(seeded_db, at=world - timedelta(minutes=1))
        seeded_db.commit()
        reminder_id = reminder.id

        real = __import__("app.scheduler.poll", fromlist=["deliver"]).deliver

        def deliver_then_die(session, reminder):  # noqa: ANN001
            real(session, reminder)
            raise RuntimeError("process died between delivering and marking")

        monkeypatch.setattr("app.scheduler.poll.deliver", deliver_then_die)
        poll_once()

        monkeypatch.setattr("app.scheduler.poll.deliver", real)
        poll_once()

        session = fresh()
        try:
            assert (
                session.query(Notification)
                .filter(Notification.reminder_id == reminder_id)
                .count()
                == 1
            )
            assert session.get(Reminder, reminder_id).status is ReminderStatus.SENT
        finally:
            session.close()


class TestPoisonedRows:
    """One bad row must not become an infinite retry or a blocked batch."""

    @pytest.fixture
    def poison(self, monkeypatch):
        """Delivery that always throws, for one nominated reminder."""

        def poisoned(target_id: int) -> None:
            real = __import__("app.scheduler.poll", fromlist=["deliver"]).deliver

            def deliver(session, reminder):  # noqa: ANN001
                if reminder.id == target_id:
                    raise RuntimeError("simulated delivery failure")
                return real(session, reminder)

            monkeypatch.setattr("app.scheduler.poll.deliver", deliver)

        return poisoned

    def test_the_attempt_is_counted_even_though_delivery_threw(
        self, seeded_db, world, poison
    ):
        """Counted *before* the attempt, which is the only ordering that
        survives a crash inside the attempt itself. Counting afterwards would
        make a row that reliably kills the process immortal."""
        reminder = _reminder(seeded_db, at=world - timedelta(minutes=1))
        seeded_db.commit()
        poison(reminder.id)

        poll_once()

        session = fresh()
        try:
            row = session.get(Reminder, reminder.id)
            assert row.attempts == 1
            assert row.status is ReminderStatus.PENDING
        finally:
            session.close()

    def test_it_reaches_failed_at_exactly_the_cap(self, seeded_db, world, poison, settings):
        reminder = _reminder(seeded_db, at=world - timedelta(minutes=1))
        seeded_db.commit()
        poison(reminder.id)

        for tick in range(settings.max_reminder_attempts):
            poll_once()
            session = fresh()
            try:
                row = session.get(Reminder, reminder.id)
                expected = (
                    ReminderStatus.FAILED
                    if tick == settings.max_reminder_attempts - 1
                    else ReminderStatus.PENDING
                )
                assert row.status is expected, f"after tick {tick + 1}"
            finally:
                session.close()

    def test_the_failure_is_audited(self, seeded_db, world, poison, settings):
        reminder = _reminder(seeded_db, at=world - timedelta(minutes=1))
        seeded_db.commit()
        poison(reminder.id)

        for _ in range(settings.max_reminder_attempts):
            poll_once()

        session = fresh()
        try:
            assert _audits(session, "reminder_failed") == 1
        finally:
            session.close()

    def test_a_failed_row_is_never_polled_again(self, seeded_db, world, poison, settings):
        reminder = _reminder(seeded_db, at=world - timedelta(minutes=1))
        seeded_db.commit()
        poison(reminder.id)
        for _ in range(settings.max_reminder_attempts):
            poll_once()

        poll_once()

        session = fresh()
        try:
            assert session.get(Reminder, reminder.id).attempts == (
                settings.max_reminder_attempts
            )
        finally:
            session.close()

    def test_its_batchmates_still_deliver(self, seeded_db, world, poison):
        """The property that makes per-row isolation worth having. A batch that
        aborts on its first bad row delivers nothing at all, and looks from the
        outside exactly like a batch with nothing due."""
        bad = _reminder(seeded_db, at=world - timedelta(minutes=5), message="bad")
        good = _reminder(seeded_db, at=world - timedelta(minutes=4), message="good")
        seeded_db.commit()
        poison(bad.id)

        result = poll_once()

        assert result.delivered == [good.id]
        session = fresh()
        try:
            assert session.get(Reminder, good.id).status is ReminderStatus.SENT
        finally:
            session.close()


class TestTheVisitCompletionSweep:
    """"The visit is over" is the one transition with no actor.

    Every other appointment edge has someone behind it — the patient confirms,
    the patient or the agent cancels, staff correct a mistake. This one is the
    passage of time, so the process that watches the clock owns it.
    """

    def _end_of(self, session) -> object:
        return session.get(Appointment, SEEDED_APPOINTMENT_ID).slot.end_time

    def test_a_past_appointment_is_completed(self, seeded_db, world):
        end = self._end_of(seeded_db)
        with clock.frozen_at(end + timedelta(minutes=1)):
            result = poll_once()

        assert SEEDED_APPOINTMENT_ID in result.completed
        session = fresh()
        try:
            assert (
                session.get(Appointment, SEEDED_APPOINTMENT_ID).status
                is AppointmentStatus.COMPLETED
            )
        finally:
            session.close()

    def test_an_appointment_still_in_the_future_is_untouched(self, seeded_db, world):
        assert poll_once().completed == []
        session = fresh()
        try:
            assert (
                session.get(Appointment, SEEDED_APPOINTMENT_ID).status
                is AppointmentStatus.CONFIRMED
            )
        finally:
            session.close()

    def test_a_cancelled_appointment_is_never_swept(self, seeded_db, world):
        """Only ``confirmed`` visits happen. Sweeping a cancellation into
        ``completed`` would invent an attendance record."""
        appointment = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID)
        end = appointment.slot.end_time
        appointment.status = AppointmentStatus.CANCELLED
        seeded_db.commit()

        with clock.frozen_at(end + timedelta(minutes=1)):
            assert poll_once().completed == []

    def test_the_sweep_is_audited(self, seeded_db, world):
        end = self._end_of(seeded_db)
        with clock.frozen_at(end + timedelta(minutes=1)):
            poll_once()

        session = fresh()
        try:
            assert _audits(session, "appointment_completed") == 1
        finally:
            session.close()

    def test_it_opens_a_post_visit_follow_up_task(self, seeded_db, world):
        """PRD story 32: the follow-up is triggered by the sweep, not by a
        person remembering. The task is the trigger's only durable trace."""
        end = self._end_of(seeded_db)
        with clock.frozen_at(end + timedelta(minutes=1)):
            poll_once()

        session = fresh()
        try:
            tasks = (
                session.query(FollowUpTask)
                .filter(
                    FollowUpTask.appointment_id == SEEDED_APPOINTMENT_ID,
                    FollowUpTask.task_type == FollowUpTaskType.POST_VISIT,
                )
                .all()
            )
        finally:
            session.close()

        assert len(tasks) == 1
        assert tasks[0].status is FollowUpTaskStatus.OPEN

    def test_ticking_twice_sweeps_once(self, seeded_db, world):
        """The boundedness rule for this writer. A sweep keyed on time alone
        would re-fire every minute for the rest of the appointment's life; it
        is keyed on the *status*, which only one tick can change."""
        end = self._end_of(seeded_db)
        with clock.frozen_at(end + timedelta(minutes=1)):
            poll_once()
            second = poll_once()

        assert second.completed == []
        session = fresh()
        try:
            assert _audits(session, "appointment_completed") == 1
            assert (
                session.query(FollowUpTask)
                .filter(FollowUpTask.task_type == FollowUpTaskType.POST_VISIT)
                .count()
                == 1
            )
        finally:
            session.close()

    def test_a_tick_does_both_duties(self, seeded_db, world):
        """One job, two queries. Splitting them into two schedules would be two
        things to configure, two things to forget to start, and no benefit."""
        end = self._end_of(seeded_db)
        reminder = _reminder(seeded_db, at=world - timedelta(minutes=1))
        seeded_db.commit()

        with clock.frozen_at(end + timedelta(minutes=1)):
            result = poll_once()

        assert result.delivered == [reminder.id]
        assert result.completed == [SEEDED_APPOINTMENT_ID]


class TestCorrectingASweptVisit:
    """PRD 32a. The sweep's optimism, corrected by someone who was there.

    The poll job can only see that an end time has passed. Whether the patient
    turned up is not a fact the clock has, so ``completed`` is a default rather
    than an observation — and the correction has to move the derived row in
    *both* directions, or one mis-click leaves a patient permanently recorded
    as a no-show with a follow-up task nobody can clear.
    """

    def _swept(self, seeded_db):
        appointment = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID)
        end = appointment.slot.end_time
        with clock.frozen_at(end + timedelta(minutes=1)):
            poll_once()
        seeded_db.expire_all()
        return appointment

    def test_a_swept_visit_can_be_marked_missed(self, seeded_db, world, staff_user):
        self._swept(seeded_db)

        apply_visit_decision(
            seeded_db,
            staff=staff_user,
            appointment_id=SEEDED_APPOINTMENT_ID,
            action="missed",
        )
        seeded_db.commit()

        session = fresh()
        try:
            assert (
                session.get(Appointment, SEEDED_APPOINTMENT_ID).status
                is AppointmentStatus.MISSED
            )
            assert _audits(session, "appointment_marked_missed") == 1
        finally:
            session.close()

    def test_marking_it_missed_opens_a_missed_visit_task(
        self, seeded_db, world, staff_user
    ):
        self._swept(seeded_db)
        apply_visit_decision(
            seeded_db, staff=staff_user, appointment_id=SEEDED_APPOINTMENT_ID,
            action="missed",
        )
        seeded_db.commit()

        session = fresh()
        try:
            task = (
                session.query(FollowUpTask)
                .filter(
                    FollowUpTask.appointment_id == SEEDED_APPOINTMENT_ID,
                    FollowUpTask.task_type == FollowUpTaskType.MISSED_VISIT,
                )
                .one()
            )
            assert task.status is FollowUpTaskStatus.OPEN
        finally:
            session.close()

    def test_flipping_back_closes_it(self, seeded_db, world, staff_user):
        """The half that is easy to leave out, and the derivation invariant's
        own failure mode if it is: a task that only ever opens."""
        self._swept(seeded_db)
        for action in ("missed", "completed"):
            apply_visit_decision(
                seeded_db, staff=staff_user,
                appointment_id=SEEDED_APPOINTMENT_ID, action=action,
            )
        seeded_db.commit()

        session = fresh()
        try:
            task = (
                session.query(FollowUpTask)
                .filter(FollowUpTask.task_type == FollowUpTaskType.MISSED_VISIT)
                .one()
            )
            assert task.status is FollowUpTaskStatus.CLOSED
            assert (
                session.get(Appointment, SEEDED_APPOINTMENT_ID).status
                is AppointmentStatus.COMPLETED
            )
        finally:
            session.close()

    def test_a_future_visit_cannot_be_marked_missed(self, seeded_db, world, staff_user):
        """The appointment is still `confirmed` — it has not happened. Recording
        a no-show for an afternoon that has not arrived would be a fact about
        the future."""
        with pytest.raises(ValidationFailed):
            apply_visit_decision(
                seeded_db, staff=staff_user,
                appointment_id=SEEDED_APPOINTMENT_ID, action="missed",
            )

    def test_the_action_is_a_closed_set(self, seeded_db, world, staff_user):
        self._swept(seeded_db)
        with pytest.raises(ValidationFailed):
            apply_visit_decision(
                seeded_db, staff=staff_user,
                appointment_id=SEEDED_APPOINTMENT_ID, action="cancelled",
            )

    def test_a_patient_cannot_correct_a_visit(self, seeded_db, world, patient_user):
        """Role, at the service layer rather than at the screen."""
        self._swept(seeded_db)
        with pytest.raises(PermissionDenied):
            apply_visit_decision(
                seeded_db, staff=patient_user,
                appointment_id=SEEDED_APPOINTMENT_ID, action="missed",
            )

    def test_it_writes_no_trace_rows(self, seeded_db, world, staff_user):
        """A staff action with no run is not a turn. The split holds for the
        human path as well as for the scheduler's."""
        self._swept(seeded_db)
        before = fresh()
        try:
            traces = before.query(TraceEvent).count()
        finally:
            before.close()

        apply_visit_decision(
            seeded_db, staff=staff_user,
            appointment_id=SEEDED_APPOINTMENT_ID, action="missed",
        )
        seeded_db.commit()

        session = fresh()
        try:
            assert session.query(TraceEvent).count() == traces
        finally:
            session.close()


class TestTheServiceWiring:
    """The six lines that make the tick periodic.

    Small, but not nothing: a poll job that is never started is a poll job that
    passes every test in this file and delivers no reminder ever. What is
    checked here is that starting is real, that stopping is real, and that the
    switch the suite relies on to keep the thread off actually keeps it off.
    """

    def test_it_does_not_start_when_configuration_says_not_to(self):
        """The suite's own guarantee. Every test above drives ``poll_once``
        directly, and a thread ticking underneath them would deliver reminders
        nobody asked about and race every "nothing happened" assertion."""
        assert get_settings().scheduler_enabled is False
        assert service.start() is None

    def test_it_starts_one_job_at_the_configured_interval(self, monkeypatch):
        monkeypatch.setattr(
            "app.scheduler.service.get_settings",
            lambda: get_settings().model_copy(
                update={"scheduler_enabled": True, "reminder_poll_seconds": 42}
            ),
        )
        started = service.start()
        try:
            assert started is not None
            jobs = started.get_jobs()
            assert len(jobs) == 1
            assert jobs[0].id == service.JOB_ID
            assert jobs[0].trigger.interval.total_seconds() == 42
        finally:
            service.shutdown()

    def test_shutting_down_is_safe_when_nothing_is_running(self):
        service.shutdown()
        service.shutdown()

    def test_a_failing_tick_does_not_kill_the_schedule(self, monkeypatch):
        """APScheduler swallows a job exception into its own logger and carries
        on, which is survivable but silent. A tick that has been failing every
        minute for an hour should say so under this application's name — and,
        more to the point, must not stop the next tick from running."""
        calls = []

        def explode():
            calls.append(1)
            raise RuntimeError("tick failed")

        monkeypatch.setattr("app.scheduler.service.poll_once", explode)
        service._tick()
        service._tick()

        assert len(calls) == 2

    def test_the_app_starts_and_stops_it_with_the_process(self, monkeypatch):
        """In ``lifespan`` rather than at import: a module that started a
        thread merely by being imported would start one in every collection."""
        events = []
        monkeypatch.setattr("app.scheduler.start", lambda: events.append("start"))
        monkeypatch.setattr("app.scheduler.shutdown", lambda: events.append("stop"))

        with TestClient(app_module.app):
            pass

        assert events == ["start", "stop"]
