"""Registration, login, and the shape of an error.

Written before the routers exist. Everything asserted here is deterministic-bin
behaviour — who may create what, which status code a failure carries, and what
the audit log has to say afterwards — so it is transcribed from the PRD first
and implemented second.

The two properties worth naming, because both are easy to lose and neither
announces itself when lost:

* **A registration cannot promote itself.** The request schema has no ``role``
  field and forbids extras, so ``{"role": "staff"}`` is a 422 rather than a
  field that is silently dropped — the difference between a refusal and a
  coincidence.
* **A failed login must not enumerate accounts.** Wrong password and unknown
  address return the same status *and the same body*; only the audit row knows
  which happened.
"""

from __future__ import annotations

from tests.api.conftest import auth_header

from app.auth.security import verify_password
from app.models import AuditEvent, PatientProfile, User, UserRole

GOOD = {
    "name": "New Patient",
    "email": "new.patient@example.invalid",
    "password": "Demo123!pass",
}


class TestHealth:
    def test_health_is_open(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestRegistration:
    def test_registering_creates_a_patient_and_a_profile(self, client, db):
        response = client.post("/auth/register", json=GOOD)
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["email"] == GOOD["email"]
        assert body["role"] == "patient"

        db.expire_all()
        user = db.query(User).filter(User.email == GOOD["email"]).one()
        assert user.role is UserRole.PATIENT
        # The row survived the request, so the router committed.
        assert db.query(PatientProfile).filter(PatientProfile.user_id == user.id).count() == 1

    def test_the_password_is_hashed_not_stored(self, client, db):
        client.post("/auth/register", json=GOOD)
        db.expire_all()
        user = db.query(User).filter(User.email == GOOD["email"]).one()
        assert user.password_hash != GOOD["password"]
        assert verify_password(GOOD["password"], user.password_hash)

    def test_the_response_never_carries_the_password(self, client, db):
        response = client.post("/auth/register", json=GOOD)
        assert GOOD["password"] not in response.text

    def test_registration_is_audited(self, client, db):
        client.post("/auth/register", json=GOOD)
        db.expire_all()
        events = db.query(AuditEvent).filter(AuditEvent.action == "user_registered").all()
        assert len(events) == 1

    def test_a_duplicate_email_is_409(self, client, db):
        assert client.post("/auth/register", json=GOOD).status_code == 201
        assert client.post("/auth/register", json=GOOD).status_code == 409

    def test_a_duplicate_email_creates_no_second_row(self, client, db):
        client.post("/auth/register", json=GOOD)
        client.post("/auth/register", json=GOOD)
        db.expire_all()
        assert db.query(User).filter(User.email == GOOD["email"]).count() == 1

    def test_nobody_registers_themselves_as_staff(self, client, db):
        """The escalation a self-service register endpoint invites.

        A 422 and not a 201: an ignored field looks identical to an accepted
        one from the caller's side, and the next reader cannot tell whether the
        refusal was designed or accidental.
        """
        response = client.post("/auth/register", json={**GOOD, "role": "staff"})
        assert response.status_code == 422
        db.expire_all()
        assert db.query(User).filter(User.role == UserRole.STAFF).count() == 0

    def test_a_short_password_is_refused(self, client, db):
        response = client.post("/auth/register", json={**GOOD, "password": "short"})
        assert response.status_code == 422

    def test_a_malformed_email_is_refused(self, client, db):
        response = client.post("/auth/register", json={**GOOD, "email": "not-an-address"})
        assert response.status_code == 422


class TestLogin:
    def test_a_seeded_account_can_log_in(self, seeded_client):
        response = seeded_client.post(
            "/auth/login",
            json={"email": "asha.patient@example.invalid", "password": "Demo123!pass"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["role"] == "patient"
        assert body["user_id"] == 1
        assert body["access_token"]

    def test_the_issued_token_authenticates(self, seeded_client):
        token = seeded_client.post(
            "/auth/login",
            json={"email": "staff@example.invalid", "password": "Demo123!pass"},
        ).json()["access_token"]

        me = seeded_client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json() == {
            "user_id": 5,
            "name": "Priya Desk",
            "email": "staff@example.invalid",
            "role": "staff",
        }

    def test_a_wrong_password_is_401(self, seeded_client):
        response = seeded_client.post(
            "/auth/login",
            json={"email": "asha.patient@example.invalid", "password": "wrong-password"},
        )
        assert response.status_code == 401

    def test_a_wrong_password_looks_exactly_like_an_unknown_address(self, seeded_client):
        """Otherwise the login form is an account-enumeration oracle."""
        wrong = seeded_client.post(
            "/auth/login",
            json={"email": "asha.patient@example.invalid", "password": "wrong-password"},
        )
        unknown = seeded_client.post(
            "/auth/login",
            json={"email": "nobody@example.invalid", "password": "wrong-password"},
        )
        assert wrong.status_code == unknown.status_code == 401
        assert wrong.json() == unknown.json()

    def test_both_outcomes_are_audited(self, seeded_client, seeded_db):
        seeded_client.post(
            "/auth/login",
            json={"email": "asha.patient@example.invalid", "password": "Demo123!pass"},
        )
        seeded_client.post(
            "/auth/login",
            json={"email": "asha.patient@example.invalid", "password": "wrong-password"},
        )
        seeded_client.post(
            "/auth/login",
            json={"email": "nobody@example.invalid", "password": "wrong-password"},
        )

        seeded_db.expire_all()
        actions = [
            (e.action, e.actor_id, e.event_metadata.get("reason"))
            for e in seeded_db.query(AuditEvent)
            .filter(AuditEvent.action.in_(("login_succeeded", "login_failed")))
            .order_by(AuditEvent.id)
            .all()
        ]
        assert actions == [
            ("login_succeeded", 1, None),
            ("login_failed", 1, "bad_password"),
            ("login_failed", None, "unknown_email"),
        ]

    def test_no_audit_row_records_the_address_that_was_tried(self, seeded_client, seeded_db):
        """The audit log identifies people by id, not by contact detail.

        An audit table is read by staff and copied into support tickets; an
        email address in its metadata is the same PII leak the trace redactor
        exists to prevent, arriving through the door nobody redacts.
        """
        seeded_client.post(
            "/auth/login",
            json={"email": "asha.patient@example.invalid", "password": "wrong-password"},
        )
        seeded_db.expire_all()
        events = seeded_db.query(AuditEvent).filter(AuditEvent.action == "login_failed").all()
        assert events
        for event in events:
            assert "@" not in str(event.event_metadata)


class TestWhoAmI:
    def test_me_requires_a_token(self, client):
        assert client.get("/auth/me").status_code == 401

    def test_me_rejects_a_token_for_a_deleted_account(self, seeded_client):
        assert seeded_client.get("/auth/me", headers=auth_header(999)).status_code == 401
