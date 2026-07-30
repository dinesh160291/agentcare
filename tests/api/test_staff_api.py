"""The staff surface: queues, typed decisions, oversight, capacity.

Four properties are load-bearing here, and each has a way of looking fine while
being broken:

* **The router owns the transaction.** ``apply_staff_decision`` and
  ``resolve_document`` deliberately do not commit — their writes belong in the
  caller's transaction. A router that forgets returns a cheerful 200 and
  changes nothing, so every decision test re-reads the row from a different
  session.
* **A refusal keeps its rows.** ``apply_staff_decision`` writes its own audit
  and trace rows *before* raising, and those rows are the interesting part: a
  refused decision is a thing a human did. The 422 must not take them with it.
* **Every staff decision is LLM-free.** The trace for a decision turn contains
  no LLM request, and that is asserted rather than assumed.
* **Oversight is staff-only in the backend.** Not by hiding a button.
"""

from __future__ import annotations

from tests.api.conftest import auth_header

from app.db import SessionLocal
from app.models import (
    AppointmentSlot,
    AuditEvent,
    Department,
    Doctor,
    DocumentStatus,
    Escalation,
    EscalationStatus,
    Notification,
    PatientDocument,
    TraceEvent,
    TraceEventType,
    WorkflowRun,
    WorkflowStatus,
)

ASHA, ROHAN, STAFF = 1, 2, 5
#: The seed's deliberately ambiguous case — ENT or Pediatrics, a human decides.
AMBIGUOUS = "book an appointment, my kid has ear pain"
EMERGENCY = "I have chest pain and my left arm hurts"


def paused_run(client, session_id="api-staff-1") -> int:
    """Drive a real turn until it parks in ``pending_review``."""
    body = client.post(
        "/workflow/messages",
        headers=auth_header(ASHA),
        json={"message": AMBIGUOUS, "session_id": session_id},
    ).json()
    assert body["status"] == WorkflowStatus.PENDING_REVIEW.value, body
    return body["run_id"]


def escalated_run(client, session_id="api-staff-esc") -> int:
    body = client.post(
        "/workflow/messages",
        headers=auth_header(ASHA),
        json={"message": EMERGENCY, "session_id": session_id},
    ).json()
    assert body["status"] == WorkflowStatus.ESCALATED.value, body
    return body["run_id"]


class TestTheQueue:
    def test_staff_see_every_run(self, seeded_client):
        run_id = paused_run(seeded_client)
        response = seeded_client.get("/staff/queue", headers=auth_header(STAFF, "staff"))
        assert response.status_code == 200, response.text
        assert [r["run_id"] for r in response.json()] == [run_id]

    def test_the_queue_filters_by_status(self, seeded_client):
        paused_run(seeded_client)
        response = seeded_client.get(
            "/staff/queue?status=completed", headers=auth_header(STAFF, "staff")
        )
        assert response.json() == []

    def test_an_invented_status_is_refused_rather_than_silently_empty(self, seeded_client):
        """An empty list for a typo'd filter reads as 'nothing to review'."""
        response = seeded_client.get(
            "/staff/queue?status=pending_reveiw", headers=auth_header(STAFF, "staff")
        )
        assert response.status_code == 422

    def test_a_patient_cannot_read_the_queue(self, seeded_client):
        assert seeded_client.get("/staff/queue", headers=auth_header(ASHA)).status_code == 403

    def test_staff_may_read_any_patients_run(self, seeded_client):
        run_id = paused_run(seeded_client)
        response = seeded_client.get(
            f"/workflow/runs/{run_id}", headers=auth_header(STAFF, "staff")
        )
        assert response.status_code == 200

    def test_staff_may_read_a_patients_context(self, seeded_client):
        response = seeded_client.get(
            "/staff/patients/1", headers=auth_header(STAFF, "staff")
        )
        assert response.status_code == 200
        assert response.json()["patient_id"] == 1

    def test_a_patient_cannot_read_another_patients_context(self, seeded_client):
        assert (
            seeded_client.get("/staff/patients/1", headers=auth_header(ROHAN)).status_code
            == 403
        )


class TestDecisions:
    def test_approving_moves_the_run_and_the_change_is_durable(
        self, seeded_client, seeded_db
    ):
        run_id = paused_run(seeded_client, "api-staff-approve")
        response = seeded_client.post(
            f"/staff/runs/{run_id}/decision",
            headers=auth_header(STAFF, "staff"),
            json={"action": "approve"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == WorkflowStatus.IN_PROGRESS.value

        seeded_db.expire_all()
        assert seeded_db.get(WorkflowRun, run_id).status is WorkflowStatus.IN_PROGRESS

    def test_approval_notifies_the_patient(self, seeded_client, seeded_db):
        run_id = paused_run(seeded_client, "api-staff-notify")
        seeded_client.post(
            f"/staff/runs/{run_id}/decision",
            headers=auth_header(STAFF, "staff"),
            json={"action": "approve"},
        )
        seeded_db.expire_all()
        assert (
            seeded_db.query(Notification)
            .filter(Notification.workflow_run_id == run_id)
            .count()
            == 1
        )

    def test_rejecting_closes_the_run(self, seeded_client, seeded_db):
        run_id = paused_run(seeded_client, "api-staff-reject")
        response = seeded_client.post(
            f"/staff/runs/{run_id}/decision",
            headers=auth_header(STAFF, "staff"),
            json={"action": "reject", "note": "Please call the clinic."},
        )
        assert response.status_code == 200, response.text
        seeded_db.expire_all()
        assert seeded_db.get(WorkflowRun, run_id).status is WorkflowStatus.REJECTED

    def test_redirecting_routes_to_the_named_department(self, seeded_client, seeded_db):
        run_id = paused_run(seeded_client, "api-staff-redirect")
        response = seeded_client.post(
            f"/staff/runs/{run_id}/decision",
            headers=auth_header(STAFF, "staff"),
            json={"action": "redirect", "department_name": "General Medicine"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["department_name"] == "General Medicine"

        seeded_db.expire_all()
        assert seeded_db.get(WorkflowRun, run_id).state["department_name"] == "General Medicine"

    def test_a_decision_costs_no_model_call(self, seeded_client, seeded_db):
        """LLM-free, and the trace is what proves it rather than the docstring."""
        run_id = paused_run(seeded_client, "api-staff-llmfree")
        before = seeded_db.query(TraceEvent).count()
        seeded_client.post(
            f"/staff/runs/{run_id}/decision",
            headers=auth_header(STAFF, "staff"),
            json={"action": "approve"},
        )
        seeded_db.expire_all()
        added = (
            seeded_db.query(TraceEvent)
            .filter(TraceEvent.id > before)
            .order_by(TraceEvent.id)
            .all()
        )
        assert added, "the decision wrote no trace at all"
        assert not [
            e
            for e in added
            if e.event_type in (TraceEventType.LLM_REQUEST, TraceEventType.LLM_RESPONSE)
        ]

    def test_a_patient_cannot_decide(self, seeded_client, seeded_db):
        run_id = paused_run(seeded_client, "api-staff-role")
        response = seeded_client.post(
            f"/staff/runs/{run_id}/decision",
            headers=auth_header(ASHA),
            json={"action": "approve"},
        )
        assert response.status_code == 403
        seeded_db.expire_all()
        assert seeded_db.get(WorkflowRun, run_id).status is WorkflowStatus.PENDING_REVIEW

    def test_a_decision_on_a_missing_run_is_404(self, seeded_client):
        response = seeded_client.post(
            "/staff/runs/9999/decision",
            headers=auth_header(STAFF, "staff"),
            json={"action": "approve"},
        )
        assert response.status_code == 404

    def test_an_invented_action_never_reaches_the_seam(self, seeded_client):
        run_id = paused_run(seeded_client, "api-staff-bogus")
        response = seeded_client.post(
            f"/staff/runs/{run_id}/decision",
            headers=auth_header(STAFF, "staff"),
            json={"action": "escalate"},
        )
        assert response.status_code == 422


class TestARefusalKeepsItsRows:
    """The refusal *is* the event. ``apply_staff_decision`` writes its audit and
    trace rows and then raises, so a router that let the 422 unwind without
    committing would erase the record of a decision a human actually made."""

    def _refuse(self, client):
        run_id = paused_run(client, "api-staff-refuse")
        response = client.post(
            f"/staff/runs/{run_id}/decision",
            headers=auth_header(STAFF, "staff"),
            json={"action": "redirect"},  # a redirect must name a department
        )
        return run_id, response

    def test_the_refusal_is_422(self, seeded_client):
        _, response = self._refuse(seeded_client)
        assert response.status_code == 422
        assert "department" in response.json()["detail"].lower()

    def test_the_run_did_not_move(self, seeded_client, seeded_db):
        run_id, _ = self._refuse(seeded_client)
        seeded_db.expire_all()
        assert seeded_db.get(WorkflowRun, run_id).status is WorkflowStatus.PENDING_REVIEW

    def test_the_audit_row_survives_the_422(self, seeded_client, seeded_db):
        run_id, _ = self._refuse(seeded_client)
        seeded_db.expire_all()
        refusals = (
            seeded_db.query(AuditEvent)
            .filter(
                AuditEvent.action == "staff_decision_refused",
                AuditEvent.entity_id == run_id,
            )
            .all()
        )
        assert len(refusals) == 1
        assert refusals[0].actor_id == STAFF

    def test_the_trace_rows_survive_the_422(self, seeded_client, seeded_db):
        run_id, _ = self._refuse(seeded_client)
        seeded_db.expire_all()
        events = (
            seeded_db.query(TraceEvent)
            .filter(TraceEvent.workflow_run_id == run_id)
            .order_by(TraceEvent.id)
            .all()
        )
        guards = [
            e.payload
            for e in events
            if e.event_type is TraceEventType.GUARD_VERDICT
            and (e.payload or {}).get("guard") == "staff_decision_precondition"
        ]
        assert guards and guards[-1]["passed"] is False

    def test_deciding_a_run_that_is_not_awaiting_review_is_also_422(
        self, seeded_client, seeded_db
    ):
        run_id = paused_run(seeded_client, "api-staff-twice")
        first = seeded_client.post(
            f"/staff/runs/{run_id}/decision",
            headers=auth_header(STAFF, "staff"),
            json={"action": "approve"},
        )
        assert first.status_code == 200
        second = seeded_client.post(
            f"/staff/runs/{run_id}/decision",
            headers=auth_header(STAFF, "staff"),
            json={"action": "approve"},
        )
        assert second.status_code == 422


class TestEscalations:
    def test_the_safety_queue_lists_the_open_escalation(self, seeded_client):
        escalated_run(seeded_client)
        response = seeded_client.get(
            "/staff/escalations", headers=auth_header(STAFF, "staff")
        )
        assert response.status_code == 200, response.text
        rows = response.json()
        assert len(rows) == 1
        assert rows[0]["kind"] == "safety"

    def test_a_patient_cannot_read_the_safety_queue(self, seeded_client):
        assert (
            seeded_client.get("/staff/escalations", headers=auth_header(ASHA)).status_code
            == 403
        )

    def test_acknowledging_is_durable(self, seeded_client, seeded_db):
        escalated_run(seeded_client, "api-esc-ack")
        seeded_db.expire_all()
        escalation_id = seeded_db.query(Escalation).one().id

        response = seeded_client.post(
            f"/staff/escalations/{escalation_id}/resolve",
            headers=auth_header(STAFF, "staff"),
            json={"status": "acknowledged", "note": "Called the patient."},
        )
        assert response.status_code == 200, response.text

        seeded_db.expire_all()
        escalation = seeded_db.get(Escalation, escalation_id)
        assert escalation.status is EscalationStatus.ACKNOWLEDGED
        assert escalation.reviewed_by == STAFF

    def test_nobody_approves_an_emergency(self, seeded_client, seeded_db):
        """The two lifecycles are kept apart in code, and the HTTP layer must
        not be the place they quietly meet again."""
        escalated_run(seeded_client, "api-esc-approve")
        seeded_db.expire_all()
        escalation_id = seeded_db.query(Escalation).one().id

        response = seeded_client.post(
            f"/staff/escalations/{escalation_id}/resolve",
            headers=auth_header(STAFF, "staff"),
            json={"status": "approved"},
        )
        assert response.status_code == 422

        seeded_db.expire_all()
        assert seeded_db.get(Escalation, escalation_id).status is EscalationStatus.OPEN

    def test_an_invented_status_is_refused_by_the_schema(self, seeded_client, seeded_db):
        escalated_run(seeded_client, "api-esc-bogus")
        seeded_db.expire_all()
        escalation_id = seeded_db.query(Escalation).one().id
        response = seeded_client.post(
            f"/staff/escalations/{escalation_id}/resolve",
            headers=auth_header(STAFF, "staff"),
            json={"status": "closed"},
        )
        assert response.status_code == 422

    def test_resolving_a_missing_escalation_is_404(self, seeded_client):
        response = seeded_client.post(
            "/staff/escalations/9999/resolve",
            headers=auth_header(STAFF, "staff"),
            json={"status": "resolved"},
        )
        assert response.status_code == 404


class TestFlaggedDocuments:
    """The seed's third document is an X-ray filed as an ECG. Phase 5 proves the
    pipeline flags it; what is proved here is the staff resolve path over HTTP,
    so the flag is set directly rather than driven through six chat turns."""

    def flag(self, session):
        document = (
            session.query(PatientDocument)
            .filter(PatientDocument.original_filename == "x-ray_report.pdf")
            .one()
        )
        document.status = DocumentStatus.FLAGGED
        document.detected_type = "X-ray report"
        session.commit()
        return document.id

    def test_the_review_queue_lists_it(self, seeded_client, seeded_db):
        document_id = self.flag(seeded_db)
        response = seeded_client.get(
            "/staff/documents/flagged", headers=auth_header(STAFF, "staff")
        )
        assert response.status_code == 200, response.text
        rows = response.json()
        assert [r["document_id"] for r in rows] == [document_id]
        assert rows[0]["declared_type"] == "ECG report"
        assert rows[0]["detected_type"] == "X-ray report"

    def test_a_patient_cannot_read_the_review_queue(self, seeded_client):
        assert (
            seeded_client.get(
                "/staff/documents/flagged", headers=auth_header(ASHA)
            ).status_code
            == 403
        )

    def test_reclassifying_is_durable(self, seeded_client, seeded_db):
        document_id = self.flag(seeded_db)
        response = seeded_client.post(
            f"/staff/documents/{document_id}/resolve",
            headers=auth_header(STAFF, "staff"),
            json={"action": "reclassify", "corrected_type": "X-ray report"},
        )
        assert response.status_code == 200, response.text

        seeded_db.expire_all()
        document = seeded_db.get(PatientDocument, document_id)
        assert document.status is DocumentStatus.VERIFIED
        assert document.document_type == "X-ray report"

    def test_a_reclassification_must_name_the_type(self, seeded_client, seeded_db):
        document_id = self.flag(seeded_db)
        response = seeded_client.post(
            f"/staff/documents/{document_id}/resolve",
            headers=auth_header(STAFF, "staff"),
            json={"action": "reclassify"},
        )
        assert response.status_code == 422
        seeded_db.expire_all()
        assert seeded_db.get(PatientDocument, document_id).status is DocumentStatus.FLAGGED

    def test_resolving_a_missing_document_is_404(self, seeded_client):
        response = seeded_client.post(
            "/staff/documents/9999/resolve",
            headers=auth_header(STAFF, "staff"),
            json={"action": "accept"},
        )
        assert response.status_code == 404


class TestOversight:
    def test_the_trace_timeline_is_ordered_and_complete(self, seeded_client, seeded_db):
        run_id = paused_run(seeded_client, "api-trace-1")
        response = seeded_client.get(
            f"/staff/runs/{run_id}/trace", headers=auth_header(STAFF, "staff")
        )
        assert response.status_code == 200, response.text
        events = response.json()
        assert events
        # One message, so one turn: seq is 1..n with no gaps, and the turn is
        # bracketed. Ordering by ``seq`` across several turns would interleave
        # them, which is why the endpoint orders by insertion instead.
        assert len({e["turn_id"] for e in events}) == 1
        assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
        assert events[0]["event_type"] == "inbound"
        assert events[-1]["event_type"] == "outbound"

    def test_the_timeline_returns_whole_turns_including_the_prefix(
        self, seeded_client, seeded_db
    ):
        """The seq-23 defect, stated as a test.

        A turn opens *before* its run exists — inbound, safety screen,
        classification — so those rows carry a null ``workflow_run_id`` by
        design and ``bind_run`` only attaches what comes after. A timeline
        selected with ``WHERE workflow_run_id = ?`` therefore drops exactly the
        part a reviewer needs, and it looks complete because it is ordered and
        non-empty. Two things are asserted together because either alone is
        satisfiable by the broken version: that rows the run does not own are
        present, and that every turn still starts at its own inbound.
        """
        run_id = paused_run(seeded_client, "api-trace-3")
        # A second turn on the same run, so the "ordered by id, not by seq"
        # rule has something to be wrong about: seq restarts at 1 per turn.
        seeded_client.post(
            "/workflow/messages",
            headers=auth_header(ASHA),
            json={"message": "any update on that?", "session_id": "api-trace-3"},
        )

        events = seeded_client.get(
            f"/staff/runs/{run_id}/trace", headers=auth_header(STAFF, "staff")
        ).json()
        turns = {event["turn_id"] for event in events}
        assert len(turns) >= 2, "the second turn is missing from the timeline"

        unowned = (
            SessionLocal()
            .query(TraceEvent)
            .filter(
                TraceEvent.turn_id.in_(turns),
                TraceEvent.workflow_run_id.is_(None),
            )
            .count()
        )
        assert unowned, "no turn has a pre-run prefix, so this proves nothing"
        assert len(events) > (
            SessionLocal()
            .query(TraceEvent)
            .filter(TraceEvent.workflow_run_id == run_id)
            .count()
        ), "the timeline returned only the events bound to the run"

        for turn_id in turns:
            in_turn = [event for event in events if event["turn_id"] == turn_id]
            assert in_turn[0]["seq"] == 1
            assert in_turn[0]["event_type"] == "inbound"
            assert [event["seq"] for event in in_turn] == list(
                range(1, len(in_turn) + 1)
            )

    def test_a_patient_cannot_read_a_trace(self, seeded_client):
        run_id = paused_run(seeded_client, "api-trace-2")
        assert (
            seeded_client.get(
                f"/staff/runs/{run_id}/trace", headers=auth_header(ASHA)
            ).status_code
            == 403
        )

    def test_the_audit_log_is_readable_newest_first(self, seeded_client, seeded_db):
        paused_run(seeded_client, "api-audit-1")
        response = seeded_client.get("/staff/audit", headers=auth_header(STAFF, "staff"))
        assert response.status_code == 200, response.text
        rows = response.json()
        assert rows
        assert [r["id"] for r in rows] == sorted((r["id"] for r in rows), reverse=True)

    def test_the_audit_log_filters_by_action(self, seeded_client, seeded_db):
        paused_run(seeded_client, "api-audit-2")
        rows = seeded_client.get(
            "/staff/audit?action=workflow_transition", headers=auth_header(STAFF, "staff")
        ).json()
        assert rows
        assert {r["action"] for r in rows} == {"workflow_transition"}

    def test_a_patient_cannot_read_the_audit_log(self, seeded_client):
        assert seeded_client.get("/staff/audit", headers=auth_header(ASHA)).status_code == 403


class TestCapacity:
    def test_departments_are_listed(self, seeded_client):
        response = seeded_client.get(
            "/staff/departments", headers=auth_header(STAFF, "staff")
        )
        assert response.status_code == 200
        assert len(response.json()) == 10

    def test_deactivating_a_department_is_durable(self, seeded_client, seeded_db):
        response = seeded_client.patch(
            "/staff/departments/1",
            headers=auth_header(STAFF, "staff"),
            json={"active": False},
        )
        assert response.status_code == 200, response.text
        seeded_db.expire_all()
        assert seeded_db.get(Department, 1).active is False

    def test_deactivating_a_doctor_is_durable(self, seeded_client, seeded_db):
        response = seeded_client.patch(
            "/staff/doctors/1", headers=auth_header(STAFF, "staff"), json={"active": False}
        )
        assert response.status_code == 200, response.text
        seeded_db.expire_all()
        assert seeded_db.get(Doctor, 1).active is False

    def test_a_patient_cannot_change_capacity(self, seeded_client, seeded_db):
        response = seeded_client.patch(
            "/staff/departments/1", headers=auth_header(ASHA), json={"active": False}
        )
        assert response.status_code == 403
        seeded_db.expire_all()
        assert seeded_db.get(Department, 1).active is True

    def test_adding_slots_creates_them(self, seeded_client, seeded_db):
        before = seeded_db.query(AppointmentSlot).filter(
            AppointmentSlot.doctor_id == 1
        ).count()
        response = seeded_client.post(
            "/staff/doctors/1/slots",
            headers=auth_header(STAFF, "staff"),
            json={"start_times": ["2099-01-05T09:00:00", "2099-01-05T09:30:00"]},
        )
        assert response.status_code == 201, response.text
        assert response.json()["created"] == 2

        seeded_db.expire_all()
        after = seeded_db.query(AppointmentSlot).filter(
            AppointmentSlot.doctor_id == 1
        ).count()
        assert after == before + 2

    def test_the_same_slot_twice_is_not_created_twice(self, seeded_client, seeded_db):
        seeded_client.post(
            "/staff/doctors/1/slots",
            headers=auth_header(STAFF, "staff"),
            json={"start_times": ["2099-01-06T09:00:00"]},
        )
        response = seeded_client.post(
            "/staff/doctors/1/slots",
            headers=auth_header(STAFF, "staff"),
            json={"start_times": ["2099-01-06T09:00:00"]},
        )
        assert response.status_code == 201
        assert response.json()["created"] == 0
        assert response.json()["skipped"] == 1

    def test_an_unparseable_time_is_refused_and_creates_nothing(
        self, seeded_client, seeded_db
    ):
        before = seeded_db.query(AppointmentSlot).count()
        response = seeded_client.post(
            "/staff/doctors/1/slots",
            headers=auth_header(STAFF, "staff"),
            json={"start_times": ["next tuesday", "2099-01-07T09:00:00"]},
        )
        assert response.status_code == 422
        seeded_db.expire_all()
        assert seeded_db.query(AppointmentSlot).count() == before

    def test_adding_slots_for_a_missing_doctor_is_404(self, seeded_client):
        response = seeded_client.post(
            "/staff/doctors/9999/slots",
            headers=auth_header(STAFF, "staff"),
            json={"start_times": ["2099-01-08T09:00:00"]},
        )
        assert response.status_code == 404


class TestTheCapacityListingIsNotAOneWayDoor:
    """Closing a department must not remove it from the page that re-opens it.

    Found by rendering the capacity view for the first time: the endpoint was
    serving the agents' active-only listing, so a closed department disappeared
    and could never be switched back on.
    """

    def test_a_closed_department_is_still_listed(self, seeded_client, seeded_db):
        seeded_client.patch(
            "/staff/departments/1",
            headers=auth_header(STAFF, "staff"),
            json={"active": False},
        )
        listed = seeded_client.get(
            "/staff/departments", headers=auth_header(STAFF, "staff")
        ).json()

        cardiology = next(d for d in listed if d["name"] == "Cardiology")
        assert cardiology["active"] is False

    def test_it_can_therefore_be_reopened(self, seeded_client, seeded_db):
        headers = auth_header(STAFF, "staff")
        seeded_client.patch("/staff/departments/1", headers=headers,
                            json={"active": False})
        seeded_client.patch("/staff/departments/1", headers=headers,
                            json={"active": True})

        seeded_db.expire_all()
        assert seeded_db.get(Department, 1).active is True


class TestCorrectingASweptVisit:
    """PRD 32a over HTTP: the sweep's default, corrected by a person.

    The router owns the transaction here as everywhere else in this file, so
    every assertion re-reads the row from a different session — a handler that
    forgot to commit would otherwise return a cheerful 200 and change nothing.
    """

    def _swept(self, seeded_db):
        """Run the sweep against a past-dated appointment, as the job would."""
        from datetime import timedelta

        from app import clock
        from app.models import Appointment
        from app.scheduler import poll_once

        end = seeded_db.get(Appointment, 1).slot.end_time
        with clock.frozen_at(end + timedelta(minutes=1)):
            poll_once()
        seeded_db.expire_all()

    def test_staff_can_mark_a_swept_visit_missed(self, seeded_client, seeded_db):
        self._swept(seeded_db)

        response = seeded_client.post(
            "/staff/appointments/1/visit",
            json={"action": "missed"},
            headers=auth_header(STAFF, "staff"),
        )

        assert response.status_code == 200
        assert response.json()["status"] == "missed"

        session = SessionLocal()
        try:
            from app.models import Appointment, AppointmentStatus

            assert session.get(Appointment, 1).status is AppointmentStatus.MISSED
        finally:
            session.close()

    def test_a_patient_is_refused(self, seeded_client, seeded_db):
        self._swept(seeded_db)

        response = seeded_client.post(
            "/staff/appointments/1/visit",
            json={"action": "missed"},
            headers=auth_header(ASHA),
        )

        assert response.status_code == 403

    def test_an_unswept_visit_is_refused(self, seeded_client, seeded_db):
        """The appointment is still `confirmed` — it has not happened yet, and
        a no-show recorded in advance is a claim about the future."""
        response = seeded_client.post(
            "/staff/appointments/1/visit",
            json={"action": "missed"},
            headers=auth_header(STAFF, "staff"),
        )

        assert response.status_code == 422

    def test_the_action_vocabulary_is_closed_at_the_schema(
        self, seeded_client, seeded_db
    ):
        """Refused before it reaches a session: a typo must never be one
        `str.lower()` away from a state change."""
        self._swept(seeded_db)

        response = seeded_client.post(
            "/staff/appointments/1/visit",
            json={"action": "cancelled"},
            headers=auth_header(STAFF, "staff"),
        )

        assert response.status_code == 422

    def test_the_list_shows_swept_visits_and_hides_the_rest(
        self, seeded_client, seeded_db
    ):
        """The list the correction is made from. It was broken — a wrong
        argument count in the handler — and the whole API suite stayed green,
        because nothing here read it; the Streamlit wiring test found it. So
        this exists, and it asserts both directions."""
        empty = seeded_client.get("/staff/visits", headers=auth_header(STAFF, "staff"))
        assert empty.status_code == 200
        assert empty.json() == []

        self._swept(seeded_db)

        listed = seeded_client.get("/staff/visits", headers=auth_header(STAFF, "staff"))
        assert listed.status_code == 200
        codes = [row["reference_code"] for row in listed.json()]
        assert codes == ["AC-000001"]

    def test_a_patient_cannot_read_the_visit_list(self, seeded_client, seeded_db):
        self._swept(seeded_db)
        response = seeded_client.get("/staff/visits", headers=auth_header(ASHA))
        assert response.status_code == 403
