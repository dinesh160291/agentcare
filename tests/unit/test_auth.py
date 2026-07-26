"""Password hashing, tokens, and the role + ownership rules."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app import clock
from app.auth.ownership import get_owned_or_404, patient_profile_for, require_own_patient_id
from app.auth.security import (
    MAX_PASSWORD_BYTES,
    create_access_token,
    decode_access_token,
    hash_password,
    user_id_from_token,
    verify_password,
)
from app.errors import AuthenticationError, RecordNotFound, ValidationFailed
from app.models import (
    Appointment,
    AppointmentStatus,
    AuditEvent,
    Department,
    Doctor,
    PatientDocument,
    PatientProfile,
    Reminder,
    ReminderType,
    User,
    UserRole,
    WorkflowRun,
    WorkflowStatus,
)


class TestPasswords:
    def test_hash_then_verify_round_trip(self):
        digest = hash_password("correct horse battery staple")
        assert verify_password("correct horse battery staple", digest) is True

    def test_wrong_password_is_rejected(self):
        digest = hash_password("correct horse battery staple")
        assert verify_password("Correct horse battery staple", digest) is False

    def test_the_plaintext_never_appears_in_the_hash(self):
        digest = hash_password("hunter2-is-a-terrible-password")
        assert "hunter2" not in digest

    def test_the_same_password_hashes_differently_each_time(self):
        """Per-password salt: identical passwords must not share a digest."""
        assert hash_password("same input") != hash_password("same input")

    def test_overlong_passwords_are_rejected_not_truncated(self):
        """bcrypt ignores everything past 72 bytes. Truncating silently would
        make two different long passwords interchangeable."""
        with pytest.raises(ValidationFailed):
            hash_password("a" * (MAX_PASSWORD_BYTES + 1))

    def test_empty_password_is_rejected(self):
        with pytest.raises(ValidationFailed):
            hash_password("")

    def test_verify_survives_a_malformed_hash(self):
        assert verify_password("anything", "not-a-bcrypt-hash") is False


class TestTokens:
    def test_round_trip_carries_the_subject(self):
        token = create_access_token(user_id=7, role="patient")
        assert user_id_from_token(token) == 7
        assert decode_access_token(token)["role"] == "patient"

    def test_expired_token_is_rejected(self):
        token = create_access_token(user_id=7, role="patient", expires_delta=timedelta(seconds=-1))
        with pytest.raises(AuthenticationError, match="expired"):
            decode_access_token(token)

    def test_tampered_token_is_rejected(self):
        token = create_access_token(user_id=7, role="patient")
        head, payload, signature = token.split(".")
        forged = f"{head}.{payload}.{signature[:-4]}AAAA"
        with pytest.raises(AuthenticationError):
            decode_access_token(forged)

    def test_token_issued_against_a_frozen_clock_expires_on_schedule(self):
        """Token lifetime is measured by the clock seam, not the wall clock."""
        from datetime import datetime

        clock.freeze(datetime(2026, 3, 1, 9, 0))
        token = create_access_token(user_id=1, role="patient", expires_delta=timedelta(minutes=30))
        clock.freeze(datetime(2026, 3, 1, 9, 15))
        assert user_id_from_token(token) == 1

        clock.freeze(datetime(2026, 3, 1, 10, 0))
        with pytest.raises(AuthenticationError):
            decode_access_token(token)

    def test_tokens_work_when_the_clock_is_pinned_to_the_past(self):
        """Regression: APP_TODAY must not make login impossible.

        PyJWT compares ``exp`` against real wall-clock time. Left to its own
        devices, a demo running with the clock pinned to a past date issues
        tokens that are already expired the instant they are created — the app
        boots and nobody can log in. Expiry is therefore checked against the
        clock seam.
        """
        from datetime import datetime

        clock.freeze(datetime(2020, 1, 1, 9, 0))
        token = create_access_token(user_id=42, role="patient")
        assert user_id_from_token(token) == 42


@pytest.fixture
def two_patients(db):
    """Two patients with one record each, plus a staff account."""
    users = [
        User(id=1, name="Patient A", email="a@example.invalid",
             password_hash=hash_password("pw-a"), role=UserRole.PATIENT),
        User(id=2, name="Patient B", email="b@example.invalid",
             password_hash=hash_password("pw-b"), role=UserRole.PATIENT),
        User(id=3, name="Staff", email="s@example.invalid",
             password_hash=hash_password("pw-s"), role=UserRole.STAFF),
    ]
    db.add_all(users)
    db.flush()
    db.add_all([PatientProfile(id=1, user_id=1), PatientProfile(id=2, user_id=2)])
    db.add(Department(id=1, name="Cardiology", description="", active=True))
    db.add(Doctor(id=1, department_id=1, name="Dr. Synthetic", active=True))
    db.flush()

    # One record of each patient-scoped type, owned by patient B.
    db.add(Appointment(id=10, patient_id=2, doctor_id=1, slot_id=None, department_id=1,
                       status=AppointmentStatus.CONFIRMED, reason="synthetic"))
    db.add(PatientDocument(id=10, patient_id=2, declared_type="ECG report",
                           document_type="ECG report", storage_path="/tmp/x.pdf",
                           checksum="deadbeef"))
    db.add(Reminder(id=10, patient_id=2, appointment_id=10,
                    reminder_type=ReminderType.APPOINTMENT,
                    scheduled_at=clock.now()))
    db.add(WorkflowRun(id=10, patient_id=2, status=WorkflowStatus.IN_PROGRESS))
    db.flush()
    return db


class TestOwnership:
    @pytest.mark.parametrize(
        "model", [Appointment, PatientDocument, Reminder, WorkflowRun]
    )
    def test_cross_patient_probe_is_a_404_on_every_entity(self, two_patients, model):
        """The one-digit edit a judge tries first.

        Patient A holds a perfectly valid token and asks for patient B's row.
        The answer must be indistinguishable from asking for an id that was
        never issued — a 403 would confirm the record exists.
        """
        db = two_patients
        patient_a = db.get(User, 1)
        with pytest.raises(RecordNotFound):
            get_owned_or_404(db, model, 10, patient_a)

    def test_a_denied_probe_is_audited(self, two_patients):
        db = two_patients
        patient_a = db.get(User, 1)
        with pytest.raises(RecordNotFound):
            get_owned_or_404(db, Appointment, 10, patient_a)

        denials = db.query(AuditEvent).filter(AuditEvent.action == "access_denied").all()
        assert len(denials) == 1
        assert denials[0].actor_id == 1
        assert denials[0].entity_type == "Appointment"
        assert denials[0].event_metadata["reason"] == "ownership"

    def test_missing_and_forbidden_are_the_same_answer(self, two_patients):
        """Both raise the same error type with the same message, so response
        shape cannot be used to enumerate which ids exist."""
        db = two_patients
        patient_a = db.get(User, 1)

        with pytest.raises(RecordNotFound) as forbidden:
            get_owned_or_404(db, Appointment, 10, patient_a)
        with pytest.raises(RecordNotFound) as missing:
            get_owned_or_404(db, Appointment, 9999, patient_a)

        assert str(forbidden.value) == str(missing.value)

    def test_owner_can_read_their_own_record(self, two_patients):
        db = two_patients
        patient_b = db.get(User, 2)
        assert get_owned_or_404(db, Appointment, 10, patient_b).id == 10

    def test_staff_read_by_role(self, two_patients):
        """Staff operate the queue; that is what the role is for."""
        db = two_patients
        staff = db.get(User, 3)
        assert get_owned_or_404(db, Appointment, 10, staff).id == 10

    def test_staff_still_get_404_for_a_row_that_does_not_exist(self, two_patients):
        db = two_patients
        staff = db.get(User, 3)
        with pytest.raises(RecordNotFound):
            get_owned_or_404(db, Appointment, 9999, staff)

    def test_patient_id_supplied_in_a_request_body_is_guarded(self, two_patients):
        """An endpoint taking patient_id directly re-opens the same hole."""
        db = two_patients
        patient_a = db.get(User, 1)
        require_own_patient_id(db, patient_a, 1)  # own id: fine
        with pytest.raises(RecordNotFound):
            require_own_patient_id(db, patient_a, 2)

    def test_profile_lookup_for_a_user_without_one(self, two_patients):
        db = two_patients
        staff = db.get(User, 3)
        with pytest.raises(RecordNotFound):
            patient_profile_for(db, staff)
