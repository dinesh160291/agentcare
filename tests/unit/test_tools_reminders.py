"""Reminder queries — the read side of the reminder lifecycle."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from app import clock
from app.models import Reminder, ReminderStatus
from app.tools.reminders import list_due_reminders, list_patient_reminders

MONDAY = date(2026, 8, 3)


@pytest.fixture(autouse=True)
def world(db):
    clock.freeze(datetime(2026, 8, 3, 8, 0))
    from scripts.seed import seed

    seed(db, anchor=MONDAY)
    db.commit()
    return db


class TestDueReminders:
    def test_nothing_is_due_before_its_time(self, db):
        assert list_due_reminders(db, at=datetime(2026, 8, 1, 0, 0)) == []

    def test_a_reminder_is_due_once_its_time_arrives(self, db):
        seeded = db.query(Reminder).one()
        due = list_due_reminders(db, at=seeded.scheduled_at)
        assert [r["reminder_id"] for r in due] == [seeded.id]

    def test_a_cancelled_reminder_is_never_due(self, db):
        """This is what cancelling reminders inside the appointment's
        transaction buys: they cannot resurface here afterwards."""
        seeded = db.query(Reminder).one()
        seeded.status = ReminderStatus.CANCELLED
        db.flush()
        assert list_due_reminders(db, at=seeded.scheduled_at + timedelta(days=1)) == []

    def test_a_sent_reminder_is_not_due_again(self, db):
        seeded = db.query(Reminder).one()
        seeded.status = ReminderStatus.SENT
        db.flush()
        assert list_due_reminders(db, at=seeded.scheduled_at) == []

    def test_due_reminders_are_ordered_oldest_first(self, db):
        seeded = db.query(Reminder).one()
        db.add(
            Reminder(
                patient_id=1,
                appointment_id=seeded.appointment_id,
                reminder_type=seeded.reminder_type,
                scheduled_at=seeded.scheduled_at - timedelta(hours=1),
            )
        )
        db.flush()
        due = list_due_reminders(db, at=seeded.scheduled_at)
        assert [r["scheduled_at"] for r in due] == sorted(r["scheduled_at"] for r in due)

    def test_the_default_moment_follows_the_clock_seam(self, db):
        seeded = db.query(Reminder).one()
        clock.freeze(seeded.scheduled_at)
        assert [r["reminder_id"] for r in list_due_reminders(db)] == [seeded.id]


class TestPatientReminders:
    def test_a_patients_pending_reminders_are_listed(self, db):
        assert len(list_patient_reminders(db, patient_id=1)) == 1

    def test_another_patients_reminders_are_not_included(self, db):
        assert list_patient_reminders(db, patient_id=2) == []

    def test_inactive_reminders_are_hidden_by_default(self, db):
        db.query(Reminder).one().status = ReminderStatus.CANCELLED
        db.flush()
        assert list_patient_reminders(db, patient_id=1) == []

    def test_inactive_reminders_can_be_requested_explicitly(self, db):
        """Staff reviewing a history need to see what was cancelled."""
        db.query(Reminder).one().status = ReminderStatus.CANCELLED
        db.flush()
        assert len(list_patient_reminders(db, patient_id=1, include_inactive=True)) == 1

    def test_results_are_json_serialisable(self, db):
        import json

        json.dumps(list_patient_reminders(db, patient_id=1))
        json.dumps(list_due_reminders(db, at=datetime(2026, 8, 9, 0, 0)))
