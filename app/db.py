"""Database engine, session factory, and schema creation.

Two things here are load-bearing:

* **WAL mode at engine creation.** FastAPI request handlers and the APScheduler
  poll job write to the same SQLite file. Without write-ahead logging they meet
  "database is locked" under perfectly ordinary concurrency. The pragma is
  SQLite-only and a no-op elsewhere.
* **The database is a file, never ``:memory:``.** Persistence across restarts
  is a hard requirement, and an in-memory database would satisfy every test
  that never restarts the process while failing the one that matters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""


def _sqlite_file_path(url: str) -> Path | None:
    """Filesystem path behind a SQLite URL, or None for other backends."""
    if not url.startswith("sqlite"):
        return None
    _, _, tail = url.partition(":///")
    if not tail or tail == ":memory:":
        return None
    return Path(tail)


def _apply_sqlite_pragmas(engine: Engine) -> None:
    """Enable WAL and enforce foreign keys on every SQLite connection."""

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        try:
            # Concurrent readers alongside one writer — the FastAPI/scheduler case.
            cursor.execute("PRAGMA journal_mode=WAL")
            # SQLite does not enforce foreign keys unless asked, per connection.
            cursor.execute("PRAGMA foreign_keys=ON")
            # Wait rather than fail immediately if a write lock is briefly held.
            cursor.execute("PRAGMA busy_timeout=5000")
        finally:
            cursor.close()


def create_db_engine(url: str | None = None) -> Engine:
    """Build an engine with the right pragmas for its backend."""
    settings = get_settings()
    url = url or settings.database_url

    if ":memory:" in url:
        raise ValueError(
            "In-memory SQLite is not permitted: persistent state is a hard "
            "requirement. Point DATABASE_URL at a file."
        )

    path = _sqlite_file_path(url)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)

    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        # The scheduler thread and request handlers share the engine.
        connect_args["check_same_thread"] = False

    engine = create_engine(url, connect_args=connect_args, future=True)
    if url.startswith("sqlite"):
        _apply_sqlite_pragmas(engine)
    return engine


engine: Engine = create_db_engine()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a session that always closes."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def create_all(target_engine: Engine | None = None) -> None:
    """Create every table. This is the project's "migrations" step."""
    from app import models  # noqa: F401  (import registers the mappers)

    Base.metadata.create_all(bind=target_engine or engine)


def drop_all(target_engine: Engine | None = None) -> None:
    """Drop every table — used by the seed script's reset path and by tests."""
    from app import models  # noqa: F401

    Base.metadata.drop_all(bind=target_engine or engine)


def database_file() -> Path | None:
    """Path of the SQLite file in use, if the backend is SQLite."""
    return _sqlite_file_path(get_settings().database_url)


def journal_mode(target_engine: Engine | None = None) -> str | None:
    """Report the active journal mode (a test asserts this is 'wal')."""
    eng = target_engine or engine
    if not eng.url.get_backend_name().startswith("sqlite"):
        return None
    with eng.connect() as conn:
        return conn.exec_driver_sql("PRAGMA journal_mode").scalar()


__all__ = [
    "Base",
    "SessionLocal",
    "create_all",
    "create_db_engine",
    "database_file",
    "drop_all",
    "engine",
    "get_session",
    "journal_mode",
]
