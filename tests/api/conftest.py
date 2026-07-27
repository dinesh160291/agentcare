"""Fixtures for the HTTP layer.

These tests drive the **real** app — ``app.main.app``, its real dependencies,
its real routers — over ``httpx``'s ``TestClient``. Nothing is stubbed. A
router that routed around a tool's ownership check would pass a unit test of
that tool and fail here, which is the whole reason this directory exists.

The session dependency is left alone deliberately: ``get_session`` builds from
the same ``SessionLocal`` the ``db`` fixture uses, both pointed at the one
temporary SQLite file that ``tests/conftest.py`` configured before any app
import. Overriding it with the test's own session would hide exactly the bug
this layer is prone to — a router that forgot to commit.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.auth.security import create_access_token
from app.main import app


@pytest.fixture
def client(db):
    """The real app against a fresh, empty database."""
    return TestClient(app)


@pytest.fixture
def seeded_client(seeded_db):
    """The real app against the seeded demo database."""
    return TestClient(app)


def auth_header(user_id: int, role: str = "patient") -> dict[str, str]:
    """A bearer header for an existing user row.

    The role in the token is a convenience only — every protected route
    re-reads the user row — so tests may pass whatever the row actually says.
    """
    return {"Authorization": f"Bearer {create_access_token(user_id=user_id, role=role)}"}
