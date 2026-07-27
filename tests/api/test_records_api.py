"""Appointments, documents, reminders, tasks, notifications.

Mostly reads — the mutations these records have all belong to the workflow —
plus the one write the patient does directly: an upload.

The upload tests are the load-bearing ones. ``ingest_document`` is already
hardened, so what is being proven here is that the *router* did not route
around it: that the file is sniffed by content rather than trusted by name,
that a refusal carries a status code a client can act on, and that a refused
upload leaves nothing behind. A router that read ``UploadFile.filename`` and
built a path from it would pass every unit test the tool has.
"""

from __future__ import annotations

from tests.api.conftest import auth_header

from app.models import Notification, PatientDocument

ASHA, ROHAN, STAFF = 1, 2, 5

#: A minimal, valid PDF. Magic bytes first, which is the only thing the
#: allowlist looks at.
PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"


def upload(client, user_id, *, content=PDF, filename="report.pdf", declared="ECG report"):
    return client.post(
        "/documents",
        headers=auth_header(user_id),
        files={"file": (filename, content, "application/pdf")},
        data={"declared_type": declared},
    )


class TestAppointments:
    def test_a_patient_sees_their_own_appointments(self, seeded_client):
        response = seeded_client.get("/appointments", headers=auth_header(ASHA))
        assert response.status_code == 200, response.text
        rows = response.json()
        assert len(rows) == 1
        assert rows[0]["appointment_id"] == 1
        assert rows[0]["department_name"] == "Cardiology"
        assert rows[0]["status"] == "confirmed"

    def test_another_patient_sees_none_of_them(self, seeded_client):
        assert seeded_client.get("/appointments", headers=auth_header(ROHAN)).json() == []

    def test_reading_another_patients_appointment_is_404(self, seeded_client):
        assert (
            seeded_client.get("/appointments/1", headers=auth_header(ASHA)).status_code == 200
        )
        assert (
            seeded_client.get("/appointments/1", headers=auth_header(ROHAN)).status_code == 404
        )

    def test_a_real_row_and_a_missing_one_are_indistinguishable(self, seeded_client):
        forbidden = seeded_client.get("/appointments/1", headers=auth_header(ROHAN))
        missing = seeded_client.get("/appointments/9999", headers=auth_header(ROHAN))
        assert forbidden.status_code == missing.status_code == 404
        assert forbidden.json() == missing.json()

    def test_there_is_no_direct_way_to_book(self, seeded_client):
        """Booking goes through the workflow, which confirms before it commits.

        A POST here would be a second path to the same state change with none
        of the guarantees — so the route does not exist.
        """
        response = seeded_client.post(
            "/appointments", headers=auth_header(ASHA), json={"slot_id": 1}
        )
        assert response.status_code == 405


class TestUpload:
    def test_a_pdf_is_accepted_and_stored(self, seeded_client, seeded_db):
        response = upload(seeded_client, ROHAN)
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["ok"] is True
        assert body["document"]["declared_type"] == "ECG report"

        seeded_db.expire_all()
        stored = (
            seeded_db.query(PatientDocument)
            .filter(PatientDocument.patient_id == 2)
            .one()
        )
        assert stored.id == body["document"]["document_id"]

    def test_the_client_filename_never_becomes_the_stored_path(self, seeded_client, seeded_db):
        """The traversal vector. The name is kept as a label and nothing else."""
        response = upload(seeded_client, ROHAN, filename="../../../../etc/passwd.pdf")
        assert response.status_code == 201, response.text

        seeded_db.expire_all()
        stored = seeded_db.query(PatientDocument).filter(
            PatientDocument.patient_id == 2
        ).one()
        assert ".." not in stored.storage_path
        assert "passwd" not in stored.storage_path
        assert stored.original_filename == "../../../../etc/passwd.pdf"

    def test_content_decides_the_type_not_the_extension(self, seeded_client, seeded_db):
        """An executable renamed .pdf, declared application/pdf. Both lie; the
        magic bytes do not."""
        response = seeded_client.post(
            "/documents",
            headers=auth_header(ROHAN),
            files={"file": ("innocent.pdf", b"MZ\x90\x00\x03" + b"\x00" * 200, "application/pdf")},
            data={"declared_type": "ECG report"},
        )
        assert response.status_code == 415
        assert response.json()["reason"] == "unsupported_type"

        seeded_db.expire_all()
        assert seeded_db.query(PatientDocument).filter(
            PatientDocument.patient_id == 2
        ).count() == 0

    def test_an_oversized_file_is_413(self, seeded_client, settings):
        big = PDF + b"\x00" * (settings.max_upload_bytes + 1)
        response = upload(seeded_client, ROHAN, content=big)
        assert response.status_code == 413
        assert response.json()["reason"] == "too_large"

    def test_an_empty_file_is_422(self, seeded_client):
        response = upload(seeded_client, ROHAN, content=b"")
        assert response.status_code == 422
        assert response.json()["reason"] == "empty_file"

    def test_a_missing_declared_type_is_422(self, seeded_client):
        response = upload(seeded_client, ROHAN, declared="   ")
        assert response.status_code == 422
        assert response.json()["reason"] == "missing_declared_type"

    def test_the_same_file_twice_is_409_and_names_the_original(self, seeded_client, seeded_db):
        first = upload(seeded_client, ROHAN)
        assert first.status_code == 201
        second = upload(seeded_client, ROHAN)
        assert second.status_code == 409
        assert second.json()["reason"] == "duplicate"
        assert second.json()["duplicate_of"] == first.json()["document"]["document_id"]

        seeded_db.expire_all()
        assert seeded_db.query(PatientDocument).filter(
            PatientDocument.patient_id == 2
        ).count() == 1

    def test_two_patients_may_hold_the_same_file(self, seeded_client):
        """Dedup is scoped per patient — two people can both have the same
        standard form."""
        assert upload(seeded_client, ROHAN).status_code == 201
        assert upload(seeded_client, 3).status_code == 201

    def test_staff_do_not_upload_as_a_patient(self, seeded_client):
        assert upload(seeded_client, STAFF).status_code == 403


class TestDocumentReads:
    def test_a_patient_lists_their_own_documents(self, seeded_client):
        response = seeded_client.get("/documents", headers=auth_header(ASHA))
        assert response.status_code == 200
        assert len(response.json()) == 3

    def test_reading_another_patients_document_is_404(self, seeded_client):
        assert seeded_client.get("/documents/1", headers=auth_header(ASHA)).status_code == 200
        assert seeded_client.get("/documents/1", headers=auth_header(ROHAN)).status_code == 404


class TestRemindersTasksNotifications:
    def test_a_patient_sees_their_pending_reminder(self, seeded_client):
        response = seeded_client.get("/reminders", headers=auth_header(ASHA))
        assert response.status_code == 200, response.text
        rows = response.json()
        assert len(rows) == 1
        assert rows[0]["status"] == "pending"

    def test_another_patient_sees_no_reminders(self, seeded_client):
        assert seeded_client.get("/reminders", headers=auth_header(ROHAN)).json() == []

    def test_open_tasks_start_empty(self, seeded_client):
        assert seeded_client.get("/tasks", headers=auth_header(ASHA)).json() == []

    def test_notifications_are_listed_newest_first(self, seeded_client, seeded_db):
        seeded_db.add_all(
            [
                Notification(patient_id=1, kind="workflow_update", title="First", body=""),
                Notification(patient_id=1, kind="workflow_update", title="Second", body=""),
            ]
        )
        seeded_db.commit()

        rows = seeded_client.get("/notifications", headers=auth_header(ASHA)).json()
        assert [r["title"] for r in rows] == ["Second", "First"]
        assert rows[0]["read"] is False

    def test_marking_one_read_is_durable(self, seeded_client, seeded_db):
        seeded_db.add(
            Notification(id=77, patient_id=1, kind="workflow_update", title="Hi", body="")
        )
        seeded_db.commit()

        response = seeded_client.post("/notifications/77/read", headers=auth_header(ASHA))
        assert response.status_code == 200, response.text
        seeded_db.expire_all()
        assert seeded_db.get(Notification, 77).read is True

    def test_marking_another_patients_notification_read_is_404(self, seeded_client, seeded_db):
        seeded_db.add(
            Notification(id=78, patient_id=1, kind="workflow_update", title="Hi", body="")
        )
        seeded_db.commit()

        assert (
            seeded_client.post("/notifications/78/read", headers=auth_header(ROHAN))
        ).status_code == 404
        seeded_db.expire_all()
        assert seeded_db.get(Notification, 78).read is False
