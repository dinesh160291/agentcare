"""Engine configuration and the persistence requirement."""

from __future__ import annotations

import pytest
from sqlalchemy import inspect

from app.db import create_all, create_db_engine, engine, journal_mode
from app.models import Department, User, UserRole


def test_wal_mode_is_enabled():
    """FastAPI handlers and the scheduler share one file; without WAL they
    meet 'database is locked' under ordinary concurrency."""
    assert journal_mode() == "wal"


def test_foreign_keys_are_enforced():
    """SQLite ignores foreign keys unless asked, per connection."""
    with engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1


def test_in_memory_database_is_refused():
    """An in-memory database passes every test that never restarts the
    process, and fails the one requirement that matters."""
    with pytest.raises(ValueError, match="In-memory"):
        create_db_engine("sqlite:///:memory:")


def test_schema_contains_every_expected_table():
    create_all()
    tables = set(inspect(engine).get_table_names())
    expected = {
        "users",
        "patient_profiles",
        "departments",
        "department_synonyms",
        "department_required_documents",
        "doctors",
        "appointment_slots",
        "appointments",
        "patient_documents",
        "workflow_runs",
        "reminders",
        "escalations",
        "audit_events",
        "trace_events",
        "notifications",
        "follow_up_tasks",
    }
    assert expected <= tables


def test_data_survives_a_restart(tmp_root):
    """The hard requirement: state outlives the process that wrote it.

    Simulated by disposing the engine entirely and opening a new one against
    the same file, which is what a restart does.
    """
    url = f"sqlite:///{(tmp_root / 'restart-check.db').as_posix()}"

    first = create_db_engine(url)
    create_all(first)
    from sqlalchemy.orm import Session

    with Session(first) as session:
        session.add(Department(id=99, name="Persistence Check", description="", active=True))
        session.add(
            User(
                id=99,
                name="Restart Probe",
                email="restart.probe@example.invalid",
                password_hash="not-a-real-hash",
                role=UserRole.PATIENT,
            )
        )
        session.commit()
    first.dispose()

    second = create_db_engine(url)
    with Session(second) as session:
        assert session.get(Department, 99).name == "Persistence Check"
        assert session.get(User, 99).role == UserRole.PATIENT
    second.dispose()
