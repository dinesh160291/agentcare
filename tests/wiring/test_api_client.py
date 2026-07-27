"""``api_client`` against the real backend — the contract surface.

If Streamlit is a thin client, then this module is where "thin" is checked:
every method here goes through the real routers, so a client that quietly
reshaped a payload, swallowed a refusal, or reached past the API would fail
here rather than in a screenshot.

Two properties are worth naming:

* **A refusal survives the round trip as a refusal.** The client raises
  ``ApiError`` carrying the backend's own status and message; it never turns a
  403 into an empty list, which is how a UI comes to show "nothing to review"
  to someone who is simply not allowed to review.
* **The guards still apply through this door.** Cross-patient ids answer 404
  and role failures answer 403 exactly as they do over HTTP, because it *is*
  HTTP — the transport is ASGI, the app is the real one.
"""

from __future__ import annotations

import pytest

from ui.api_client import ApiError

BOOKING = "I need a cardiology appointment next week"
PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"


class TestAuth:
    def test_health_reports_the_provider(self, wired):
        assert wired.health()["provider"] == "mock"

    def test_register_then_log_in(self, wired):
        wired.register(
            name="New Patient", email="new@example.invalid", password="Demo123!pass"
        )
        token = wired.login(email="new@example.invalid", password="Demo123!pass")
        assert token["role"] == "patient"
        assert wired.me(token["access_token"])["email"] == "new@example.invalid"

    def test_a_bad_password_raises_rather_than_returning_none(self, wired_seeded):
        """A client that returned None here would let a page render a logged-out
        state as though login had succeeded."""
        with pytest.raises(ApiError) as caught:
            wired_seeded.login(
                email="asha.patient@example.invalid", password="wrong-password"
            )
        assert caught.value.status_code == 401
        assert caught.value.is_auth

    def test_an_expired_or_absent_token_is_an_auth_error(self, wired_seeded):
        with pytest.raises(ApiError) as caught:
            wired_seeded.me("not-a-jwt")
        assert caught.value.is_auth


class TestTheChatFrontDoor:
    def test_a_message_returns_a_turn(self, wired_seeded, patient_token):
        turn = wired_seeded.send_message(
            patient_token, message=BOOKING, session_id="wire-1"
        )
        assert turn["session_id"] == "wire-1"
        assert turn["status"] == "pending_confirmation"
        assert turn["reply"]

    def test_confirm_is_a_different_call_from_a_message(
        self, wired_seeded, patient_token
    ):
        """The whole guarantee is that a commitment is a typed action. If the
        client sent the word "confirm" as a message, nothing would break
        visibly — and the exact-token reader would be doing the work the button
        is supposed to make unnecessary."""
        wired_seeded.send_message(patient_token, message=BOOKING, session_id="wire-2")
        turn = wired_seeded.send_action(
            patient_token, action="confirm", session_id="wire-2"
        )
        assert turn["run_id"]

        appointments = wired_seeded.appointments(patient_token)
        assert any(a["status"] == "confirmed" for a in appointments)

    def test_declining_books_nothing(self, wired_seeded, patient_token):
        before = len(wired_seeded.appointments(patient_token))
        wired_seeded.send_message(patient_token, message=BOOKING, session_id="wire-3")
        wired_seeded.send_action(
            patient_token, action="decline", session_id="wire-3"
        )
        assert len(wired_seeded.appointments(patient_token)) == before

    def test_a_run_is_readable_after_the_turn(self, wired_seeded, patient_token):
        turn = wired_seeded.send_message(
            patient_token, message=BOOKING, session_id="wire-4"
        )
        run = wired_seeded.run(patient_token, turn["run_id"])
        assert run["proposed_action"] == "book"
        assert run["department_name"] == "Cardiology"


class TestOwnershipThroughTheClient:
    def test_another_patients_run_is_a_404(self, wired_seeded, patient_token,
                                           other_patient_token):
        turn = wired_seeded.send_message(
            patient_token, message=BOOKING, session_id="wire-own"
        )
        with pytest.raises(ApiError) as caught:
            wired_seeded.run(other_patient_token, turn["run_id"])
        assert caught.value.is_missing

    def test_a_patient_cannot_reach_the_staff_queue(self, wired_seeded, patient_token):
        with pytest.raises(ApiError) as caught:
            wired_seeded.queue(patient_token)
        assert caught.value.is_forbidden

    def test_a_forbidden_read_is_not_an_empty_list(self, wired_seeded, patient_token):
        """The failure mode this pins: a client that caught the 403 and returned
        [] would render an empty queue, which reads as "nothing to review"."""
        with pytest.raises(ApiError):
            wired_seeded.escalations(patient_token)


class TestRecords:
    def test_documents_come_back_for_the_owner(self, wired_seeded, patient_token):
        assert len(wired_seeded.documents(patient_token)) == 3

    def test_an_upload_round_trips(self, wired_seeded, other_patient_token):
        result = wired_seeded.upload_document(
            other_patient_token,
            filename="report.pdf",
            content=PDF,
            declared_type="ECG report",
            content_type="application/pdf",
        )
        assert result["ok"] is True
        assert len(wired_seeded.documents(other_patient_token)) == 1

    def test_a_rejected_upload_carries_the_reason(self, wired_seeded, other_patient_token):
        """The backend's shape-stable refusal must survive as a refusal — a
        client that returned it as a success would have the page announce that
        an executable had been filed as an ECG."""
        with pytest.raises(ApiError) as caught:
            wired_seeded.upload_document(
                other_patient_token,
                filename="innocent.pdf",
                content=b"MZ\x90\x00\x03" + b"\x00" * 200,
                declared_type="ECG report",
                content_type="application/pdf",
            )
        assert caught.value.status_code == 415
        assert caught.value.payload["reason"] == "unsupported_type"

    def test_a_duplicate_upload_names_the_original(self, wired_seeded, other_patient_token):
        first = wired_seeded.upload_document(
            other_patient_token, filename="a.pdf", content=PDF,
            declared_type="ECG report", content_type="application/pdf",
        )
        with pytest.raises(ApiError) as caught:
            wired_seeded.upload_document(
                other_patient_token, filename="b.pdf", content=PDF,
                declared_type="ECG report", content_type="application/pdf",
            )
        assert caught.value.status_code == 409
        assert (
            caught.value.payload["duplicate_of"]
            == first["document"]["document_id"]
        )

    def test_profile_patch_round_trips(self, wired_seeded, patient_token):
        updated = wired_seeded.update_profile(patient_token, {"phone": "+1-555-0999"})
        assert updated["phone"] == "+1-555-0999"
        assert wired_seeded.profile(patient_token)["phone"] == "+1-555-0999"

    def test_marking_a_notification_read_needs_one_that_exists(
        self, wired_seeded, patient_token
    ):
        with pytest.raises(ApiError) as caught:
            wired_seeded.mark_notification_read(patient_token, 9999)
        assert caught.value.is_missing


class TestStaffThroughTheClient:
    def test_the_queue_and_a_decision(self, wired_seeded, patient_token, staff_token):
        wired_seeded.send_message(
            patient_token,
            message="book an appointment, my kid has ear pain",
            session_id="wire-staff",
        )
        queue = wired_seeded.queue(staff_token, status="pending_review")
        assert len(queue) == 1

        decision = wired_seeded.decide(
            staff_token, queue[0]["run_id"], action="approve"
        )
        assert decision["status"] == "in_progress"

    def test_a_refused_decision_raises_422_with_the_reason(
        self, wired_seeded, patient_token, staff_token
    ):
        wired_seeded.send_message(
            patient_token,
            message="book an appointment, my kid has ear pain",
            session_id="wire-refuse",
        )
        run_id = wired_seeded.queue(staff_token, status="pending_review")[0]["run_id"]
        with pytest.raises(ApiError) as caught:
            wired_seeded.decide(staff_token, run_id, action="redirect")
        assert caught.value.status_code == 422
        assert "department" in caught.value.detail.lower()

    def test_a_safety_escalation_cannot_be_approved(
        self, wired_seeded, patient_token, staff_token
    ):
        wired_seeded.send_message(
            patient_token,
            message="I have chest pain and my left arm hurts",
            session_id="wire-safety",
        )
        escalations = wired_seeded.escalations(staff_token)
        assert escalations and escalations[0]["kind"] == "safety"

        with pytest.raises(ApiError) as caught:
            wired_seeded.resolve_escalation(
                staff_token, escalations[0]["escalation_id"], status="approved"
            )
        assert caught.value.status_code == 422

    def test_a_safety_escalation_is_acknowledged(
        self, wired_seeded, patient_token, staff_token
    ):
        wired_seeded.send_message(
            patient_token,
            message="I have chest pain and my left arm hurts",
            session_id="wire-ack",
        )
        escalation_id = wired_seeded.escalations(staff_token)[0]["escalation_id"]
        result = wired_seeded.resolve_escalation(
            staff_token, escalation_id, status="acknowledged"
        )
        assert result["status"] == "acknowledged"

    def test_the_trace_covers_whole_turns(self, wired_seeded, patient_token, staff_token):
        turn = wired_seeded.send_message(
            patient_token, message=BOOKING, session_id="wire-trace"
        )
        events = wired_seeded.trace(staff_token, turn["run_id"])
        assert events[0]["event_type"] == "inbound"
        assert events[-1]["event_type"] == "outbound"

    def test_the_audit_log_reads(self, wired_seeded, staff_token):
        rows = wired_seeded.audit(staff_token, action="login_succeeded")
        assert rows and all(r["action"] == "login_succeeded" for r in rows)

    def test_capacity_toggles(self, wired_seeded, staff_token):
        result = wired_seeded.set_department_active(staff_token, 1, active=False)
        assert result["active"] is False
        assert wired_seeded.set_department_active(staff_token, 1, active=True)["active"]

    def test_adding_slots_reports_what_it_did(self, wired_seeded, staff_token):
        result = wired_seeded.add_slots(
            staff_token, 1, start_times=["2099-03-01T09:00:00"]
        )
        assert result["created"] == 1
        repeat = wired_seeded.add_slots(
            staff_token, 1, start_times=["2099-03-01T09:00:00"]
        )
        assert repeat["created"] == 0 and repeat["skipped"] == 1
