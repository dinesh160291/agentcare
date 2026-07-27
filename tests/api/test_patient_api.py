"""The patient's own surface: profile, context, and the two front doors.

The workflow tests here are end-to-end through HTTP under ``LLM_PROVIDER=mock``
— a real Coordinator turn, real tools, real rows. They are slower than a unit
test and that is the point: the thing being proven is that the router reached
the seam rather than reimplementing a piece of it.

What is pinned:

* a PATCH leaves absent fields alone, because "not sent" and "sent as null" are
  different requests and only one of them means *clear this*;
* the two front doors are separate endpoints — a Confirm is a typed action, not
  a chat message containing the word "yes";
* a run belongs to its patient, and reading someone else's is a 404.
"""

from __future__ import annotations

from tests.api.conftest import auth_header

from app.models import Appointment, AuditEvent, PatientProfile, WorkflowRun

BOOKING = "I need a cardiology appointment next week"
ASHA, ROHAN, STAFF = 1, 2, 5
SEEDED_APPOINTMENT_ID = 1


class TestProfile:
    def test_a_patient_reads_their_own_profile(self, seeded_client):
        response = seeded_client.get("/patients/me", headers=auth_header(ASHA))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["patient_id"] == 1
        assert body["name"] == "Asha Menon"
        assert body["date_of_birth"] == "1986-04-12"

    def test_staff_have_no_patient_profile_surface(self, seeded_client):
        """403 and not 404: the endpoint is patient-only by *role*, and role
        failures are not the thing we hide."""
        response = seeded_client.get("/patients/me", headers=auth_header(STAFF, "staff"))
        assert response.status_code == 403

    def test_a_patch_updates_only_what_it_names(self, seeded_client, seeded_db):
        response = seeded_client.patch(
            "/patients/me",
            headers=auth_header(ASHA),
            json={"phone": "+1-555-0999"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["phone"] == "+1-555-0999"
        # Untouched by a patch that did not mention them.
        assert body["preferred_language"] == "English"
        assert body["emergency_contact"] == "Ravi Menon +1-555-0101"

    def test_the_update_is_durable(self, seeded_client, seeded_db):
        """A router that forgot to commit would still return 200 with the new
        value — the object in memory is updated either way."""
        seeded_client.patch(
            "/patients/me", headers=auth_header(ASHA), json={"preferred_language": "Tamil"}
        )
        seeded_db.expire_all()
        assert seeded_db.get(PatientProfile, 1).preferred_language == "Tamil"

    def test_an_explicit_null_clears_the_field(self, seeded_client, seeded_db):
        seeded_client.patch(
            "/patients/me", headers=auth_header(ASHA), json={"emergency_contact": None}
        )
        seeded_db.expire_all()
        assert seeded_db.get(PatientProfile, 1).emergency_contact is None

    def test_the_update_is_audited(self, seeded_client, seeded_db):
        seeded_client.patch(
            "/patients/me", headers=auth_header(ASHA), json={"phone": "+1-555-0999"}
        )
        seeded_db.expire_all()
        events = seeded_db.query(AuditEvent).filter(
            AuditEvent.action == "profile_updated"
        ).all()
        assert len(events) == 1
        assert events[0].actor_id == ASHA
        # The audit says which fields moved, never what they moved to.
        assert events[0].event_metadata["fields"] == ["phone"]
        assert "555" not in str(events[0].event_metadata)


class TestContext:
    def test_context_comes_from_the_tool(self, seeded_client):
        response = seeded_client.get("/patients/me/context", headers=auth_header(ASHA))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["patient_id"] == 1
        assert body["name"] == "Asha Menon"
        # The seed ships one booked appointment and three documents for Asha.
        assert len(body["upcoming_appointments"]) == 1
        assert len(body["documents"]) == 3

    def test_a_patient_with_no_history_gets_empty_lists_not_an_error(self, seeded_client):
        response = seeded_client.get("/patients/me/context", headers=auth_header(ROHAN))
        assert response.status_code == 200
        assert response.json()["documents"] == []


class TestTheChatFrontDoor:
    def test_a_message_runs_a_turn_and_returns_its_reply(self, seeded_client):
        response = seeded_client.post(
            "/workflow/messages",
            headers=auth_header(ASHA),
            json={"message": BOOKING, "session_id": "api-chat-1"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["session_id"] == "api-chat-1"
        assert body["run_id"] is not None
        assert body["status"] == "pending_confirmation"
        assert body["reply"]

    def test_a_message_with_no_session_id_starts_one(self, seeded_client):
        response = seeded_client.post(
            "/workflow/messages", headers=auth_header(ASHA), json={"message": BOOKING}
        )
        assert response.status_code == 200, response.text
        assert response.json()["session_id"]

    def test_staff_cannot_use_the_patient_chat(self, seeded_client):
        response = seeded_client.post(
            "/workflow/messages",
            headers=auth_header(STAFF, "staff"),
            json={"message": BOOKING, "session_id": "api-chat-staff"},
        )
        assert response.status_code == 403

    def test_an_empty_message_is_refused_before_a_turn_opens(self, seeded_client, seeded_db):
        before = seeded_db.query(WorkflowRun).count()
        response = seeded_client.post(
            "/workflow/messages",
            headers=auth_header(ASHA),
            json={"message": "   ", "session_id": "api-chat-empty"},
        )
        assert response.status_code == 422
        seeded_db.expire_all()
        assert seeded_db.query(WorkflowRun).count() == before


class TestTheButtonFrontDoor:
    def test_confirm_commits_the_booking(self, seeded_client, seeded_db):
        seeded_client.post(
            "/workflow/messages",
            headers=auth_header(ASHA),
            json={"message": BOOKING, "session_id": "api-btn-1"},
        )
        response = seeded_client.post(
            "/workflow/actions",
            headers=auth_header(ASHA),
            json={"action": "confirm", "session_id": "api-btn-1"},
        )
        assert response.status_code == 200, response.text

        seeded_db.expire_all()
        booked = (
            seeded_db.query(Appointment)
            .filter(Appointment.patient_id == 1, Appointment.id != SEEDED_APPOINTMENT_ID)
            .all()
        )
        assert len(booked) == 1

    def test_a_word_that_is_not_an_action_never_reaches_the_seam(self, seeded_client):
        """The closed set is enforced by the schema, so 'maybe' is a 422 rather
        than something the orchestrator has to have an opinion about."""
        response = seeded_client.post(
            "/workflow/actions",
            headers=auth_header(ASHA),
            json={"action": "maybe", "session_id": "api-btn-2"},
        )
        assert response.status_code == 422

    def test_a_stale_click_is_a_calm_no_op(self, seeded_client, seeded_db):
        """Nothing is pending, so nothing happens — and it is not an error."""
        before = seeded_db.query(Appointment).count()
        response = seeded_client.post(
            "/workflow/actions",
            headers=auth_header(ROHAN),
            json={"action": "confirm", "session_id": "api-btn-stale"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["run_id"] is None
        seeded_db.expire_all()
        assert seeded_db.query(Appointment).count() == before


class TestRunVisibility:
    def test_a_patient_lists_their_own_runs(self, seeded_client):
        seeded_client.post(
            "/workflow/messages",
            headers=auth_header(ASHA),
            json={"message": BOOKING, "session_id": "api-runs-1"},
        )
        response = seeded_client.get("/workflow/runs", headers=auth_header(ASHA))
        assert response.status_code == 200, response.text
        runs = response.json()
        assert len(runs) == 1
        assert runs[0]["patient_id"] == 1
        assert runs[0]["department_name"] == "Cardiology"

    def test_another_patients_run_is_not_in_the_list(self, seeded_client):
        seeded_client.post(
            "/workflow/messages",
            headers=auth_header(ASHA),
            json={"message": BOOKING, "session_id": "api-runs-2"},
        )
        assert seeded_client.get("/workflow/runs", headers=auth_header(ROHAN)).json() == []

    def test_reading_another_patients_run_is_404(self, seeded_client, seeded_db):
        """The one-digit edit, on the run id."""
        run_id = seeded_client.post(
            "/workflow/messages",
            headers=auth_header(ASHA),
            json={"message": BOOKING, "session_id": "api-runs-3"},
        ).json()["run_id"]

        assert (
            seeded_client.get(f"/workflow/runs/{run_id}", headers=auth_header(ASHA))
        ).status_code == 200
        assert (
            seeded_client.get(f"/workflow/runs/{run_id}", headers=auth_header(ROHAN))
        ).status_code == 404

    def test_the_denied_probe_is_audited(self, seeded_client, seeded_db):
        run_id = seeded_client.post(
            "/workflow/messages",
            headers=auth_header(ASHA),
            json={"message": BOOKING, "session_id": "api-runs-4"},
        ).json()["run_id"]
        seeded_client.get(f"/workflow/runs/{run_id}", headers=auth_header(ROHAN))

        seeded_db.expire_all()
        denials = (
            seeded_db.query(AuditEvent)
            .filter(AuditEvent.action == "access_denied", AuditEvent.actor_id == ROHAN)
            .all()
        )
        assert len(denials) == 1
        assert denials[0].entity_type == "WorkflowRun"
