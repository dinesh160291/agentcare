"""Role and ownership enforcement through the HTTP seam.

``app/main.py`` does not exist until Phase 6, so these tests mount the real
dependencies on a throwaway FastAPI app. That is deliberate: the dependencies
are the place RBAC is actually enforced, and enforcement that is only proven
through a UI that hides buttons is not enforcement at all.

The routes here are stand-ins. The dependencies, the ownership helper, and the
error mapping are the real thing.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.auth.dependencies import CurrentUser, PatientUser, StaffUser
from app.auth.ownership import get_owned_or_404
from app.auth.security import create_access_token, hash_password
from app.db import get_session
from app.errors import RecordNotFound
from app.models import (
    Appointment,
    AppointmentStatus,
    AuditEvent,
    Department,
    Doctor,
    PatientProfile,
    User,
    UserRole,
)


@pytest.fixture
def api(db):
    """A minimal app exposing the dependencies under test."""
    app = FastAPI()

    @app.get("/me")
    def read_me(user: CurrentUser) -> dict:
        return {"id": user.id, "role": user.role.value}

    @app.get("/staff/queue")
    def staff_only(user: StaffUser) -> dict:
        return {"ok": True, "actor": user.id}

    @app.get("/patient/home")
    def patient_only(user: PatientUser) -> dict:
        return {"ok": True, "actor": user.id}

    @app.get("/appointments/{appointment_id}")
    def read_appointment(
        appointment_id: int,
        user: CurrentUser,
        session: Session = Depends(get_session),
    ) -> dict:
        try:
            appointment = get_owned_or_404(session, Appointment, appointment_id, user)
        except RecordNotFound as exc:
            raise HTTPException(status_code=404, detail="Not found") from exc
        return {"id": appointment.id}

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def people(db):
    """Patient A (id 1), patient B (id 2) who owns appointment 10, and staff."""
    db.add_all(
        [
            User(id=1, name="Patient A", email="a@example.invalid",
                 password_hash=hash_password("pw-a"), role=UserRole.PATIENT),
            User(id=2, name="Patient B", email="b@example.invalid",
                 password_hash=hash_password("pw-b"), role=UserRole.PATIENT),
            User(id=3, name="Staff", email="s@example.invalid",
                 password_hash=hash_password("pw-s"), role=UserRole.STAFF),
        ]
    )
    db.flush()
    db.add_all([PatientProfile(id=1, user_id=1), PatientProfile(id=2, user_id=2)])
    db.add(Department(id=1, name="Cardiology", description="", active=True))
    db.add(Doctor(id=1, department_id=1, name="Dr. Synthetic", active=True))
    db.flush()
    db.add(
        Appointment(id=10, patient_id=2, doctor_id=1, slot_id=None, department_id=1,
                    status=AppointmentStatus.CONFIRMED, reason="synthetic")
    )
    db.commit()
    return db


def auth(user_id: int, role: str, **kwargs) -> dict[str, str]:
    token = create_access_token(user_id=user_id, role=role, **kwargs)
    return {"Authorization": f"Bearer {token}"}


class TestAuthentication:
    def test_missing_header_is_401(self, api, people):
        assert api.get("/me").status_code == 401

    def test_garbage_token_is_401(self, api, people):
        response = api.get("/me", headers={"Authorization": "Bearer not-a-jwt"})
        assert response.status_code == 401

    def test_expired_token_is_401(self, api, people):
        headers = auth(1, "patient", expires_delta=timedelta(seconds=-1))
        assert api.get("/me", headers=headers).status_code == 401

    def test_token_for_a_deleted_account_is_401(self, api, people):
        """A signed token must not outlive the account it names."""
        headers = auth(999, "patient")
        assert api.get("/me", headers=headers).status_code == 401

    def test_valid_token_resolves_the_acting_user(self, api, people):
        response = api.get("/me", headers=auth(1, "patient"))
        assert response.status_code == 200
        assert response.json() == {"id": 1, "role": "patient"}


class TestRoleEnforcement:
    def test_patient_cannot_reach_a_staff_route(self, api, people):
        """403, not 404: the endpoint's existence is not a secret. Only
        *ownership* failures hide behind a 404."""
        assert api.get("/staff/queue", headers=auth(1, "patient")).status_code == 403

    def test_staff_can_reach_a_staff_route(self, api, people):
        assert api.get("/staff/queue", headers=auth(3, "staff")).status_code == 200

    def test_staff_cannot_reach_a_patient_route(self, api, people):
        assert api.get("/patient/home", headers=auth(3, "staff")).status_code == 403

    def test_a_forged_role_claim_grants_nothing(self, api, people):
        """The token's role is a convenience; the user row is the authority.

        This is the attack the re-read exists to stop: a patient mints a token
        claiming staff. The signature is valid — the claim is not.
        """
        headers = auth(1, "staff")  # user 1 is a patient in the database
        assert api.get("/staff/queue", headers=headers).status_code == 403


class TestOwnershipThroughHttp:
    def test_owner_gets_their_appointment(self, api, people):
        response = api.get("/appointments/10", headers=auth(2, "patient"))
        assert response.status_code == 200
        assert response.json()["id"] == 10

    def test_cross_patient_probe_is_404(self, api, people):
        """The one-digit edit, over HTTP this time."""
        assert api.get("/appointments/10", headers=auth(1, "patient")).status_code == 404

    def test_probing_a_real_row_looks_exactly_like_probing_a_missing_one(self, api, people):
        """Status code and body must be identical, or the difference between
        them enumerates which ids exist."""
        forbidden = api.get("/appointments/10", headers=auth(1, "patient"))
        missing = api.get("/appointments/9999", headers=auth(1, "patient"))
        assert forbidden.status_code == missing.status_code == 404
        assert forbidden.json() == missing.json()

    def test_staff_may_read_any_appointment(self, api, people):
        assert api.get("/appointments/10", headers=auth(3, "staff")).status_code == 200

    def test_the_denied_probe_survives_the_failed_request(self, api, people, db):
        """The denial must be *durable*, not merely written.

        A denial is recorded on a request that then fails, and a request-scoped
        session is closed without committing. If the audit row rides in that
        transaction it is rolled back — leaving the system claiming to audit
        denied attempts while quietly recording none of them.
        """
        assert api.get("/appointments/10", headers=auth(1, "patient")).status_code == 404

        db.expire_all()
        denials = db.query(AuditEvent).filter(AuditEvent.action == "access_denied").all()
        assert len(denials) == 1
        assert denials[0].actor_id == 1
        assert denials[0].entity_type == "Appointment"
        assert denials[0].event_metadata["requested_id"] == "10"
