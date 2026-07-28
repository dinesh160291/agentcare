"""Shared test configuration and fixtures.

The environment is set **at module scope, before any application import**, and
that ordering is load-bearing. ``app.config`` reads the environment when its
settings object is first constructed, and ``app.db`` builds an engine at import
time. Setting these inside a fixture would run too late: a module imported
during collection would already have captured the developer's real ``.env``.

The specific danger is ``LLM_PROVIDER``. Real environment variables take
priority over ``.env``, so forcing ``mock`` here is what guarantees the suite
never makes a billed API call — and a version of this file that set it late
would still pass while quietly calling a live provider.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

# --- environment, before app imports ------------------------------------
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="agentcare-tests-"))

os.environ["LLM_PROVIDER"] = "mock"
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP_ROOT / 'test.db').as_posix()}"
os.environ["ADK_SESSION_DB_URL"] = f"sqlite+aiosqlite:///{(_TMP_ROOT / 'adk.db').as_posix()}"
os.environ["UPLOAD_DIR"] = str(_TMP_ROOT / "uploads")
os.environ["JWT_SECRET_KEY"] = "test-only-secret-not-a-real-credential"
# A stray APP_TODAY in the developer's shell would silently freeze the suite.
os.environ.pop("APP_TODAY", None)
# The poll job is driven tick-by-tick from the scheduler tests. A background
# thread ticking underneath the suite would deliver reminders no test asked
# about and race every "nothing happened" assertion in the file.
os.environ["SCHEDULER_ENABLED"] = "false"

import pytest  # noqa: E402

from app import clock  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import SessionLocal, create_all, drop_all  # noqa: E402


def pytest_addoption(parser) -> None:  # noqa: ANN001
    """``--update-golden`` re-blesses golden files instead of asserting.

    Deliberately a flag rather than an environment variable or an auto-update:
    re-blessing has to be a decision someone makes, and its result has to land
    as a reviewable git diff.
    """
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="rewrite golden expectation files from current behaviour",
    )


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ANN001, ARG001
    """Remove the temporary database and uploads at the end of the run."""
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)


@pytest.fixture(autouse=True)
def _release_clock():
    """No test may leak a frozen clock into the next one."""
    yield
    clock.unfreeze()


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def db():
    """A fresh, empty database for one test."""
    drop_all()
    create_all()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def seeded_db(db):
    """A database populated by the real seed script."""
    from scripts.seed import seed

    seed(db, anchor=clock.today())
    db.commit()
    return db


@pytest.fixture
def tmp_root() -> Path:
    return _TMP_ROOT
