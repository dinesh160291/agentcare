"""Fixtures for the frontend↔backend wiring tests.

**Nothing is mocked.** ``api_client`` is pointed at the real FastAPI app over an
ASGI transport, so every call in these tests goes through the real routers, the
real auth dependencies, the real ownership guards, and the real orchestrator.

That is the whole point of this directory. A test suite that stubbed the API
would prove the UI could render a dictionary, which nobody doubts; what is in
question is whether the UI is wired to the thing that enforces the rules.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from ui import api_client


def _bind_real_app():
    """An ``ApiClient`` speaking to the real app, installed as the UI's client.

    ``TestClient`` rather than a bare ``httpx.ASGITransport``: that transport is
    async-only in httpx 0.28, and ``api_client`` is deliberately synchronous
    because Streamlit is. ``TestClient`` *is* an ``httpx.Client`` — it drives
    the ASGI app from a sync caller — so the client under test is the real one,
    unmodified, with only its transport swapped.
    """
    client = api_client.ApiClient(TestClient(app, base_url="http://agentcare.test"))
    api_client.set_client(client)
    return client


@pytest.fixture
def wired(db):
    """``ui.api_client`` bound to the real app, on an empty database."""
    client = _bind_real_app()
    try:
        yield client
    finally:
        api_client.set_client(None)


@pytest.fixture
def wired_seeded(seeded_db):
    """The same, against the seeded demo database."""
    client = _bind_real_app()
    try:
        yield client
    finally:
        api_client.set_client(None)


@pytest.fixture
def patient_token(wired_seeded) -> str:
    return wired_seeded.login(
        email="asha.patient@example.invalid", password="Demo123!pass"
    )["access_token"]


@pytest.fixture
def other_patient_token(wired_seeded) -> str:
    return wired_seeded.login(
        email="rohan.patient@example.invalid", password="Demo123!pass"
    )["access_token"]


@pytest.fixture
def staff_token(wired_seeded) -> str:
    return wired_seeded.login(
        email="staff@example.invalid", password="Demo123!pass"
    )["access_token"]
