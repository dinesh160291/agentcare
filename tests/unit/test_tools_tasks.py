"""Follow-up tasks and escalations — the boundedness invariant for records.

Both of these are things an agent can be provoked into creating repeatedly.
The rule is one open record per subject; repeats attach to it.
"""

from __future__ import annotations

import pytest

from app.models import (
    Escalation,
    EscalationKind,
    EscalationStatus,
    FollowUpTask,
    FollowUpTaskStatus,
    FollowUpTaskType,
    WorkflowRun,
    WorkflowStatus,
)
from app.tools.tasks import (
    close_followup_tasks,
    create_escalation,
    list_open_escalations,
    list_open_tasks,
    upsert_followup_task,
)

MISSING_DOCS = FollowUpTaskType.MISSING_DOCUMENTS


@pytest.fixture
def run(seeded_db):
    run = WorkflowRun(patient_id=1, status=WorkflowStatus.IN_PROGRESS)
    seeded_db.add(run)
    seeded_db.flush()
    return run


class TestFollowUpTaskUpsert:
    def test_the_first_call_opens_a_task(self, seeded_db):
        result = upsert_followup_task(
            seeded_db, patient_id=1, task_type=MISSING_DOCS,
            details={"missing": ["ECG report"]}, appointment_id=1,
        )
        assert result["created"] is True
        assert result["task"]["details"]["missing"] == ["ECG report"]

    def test_a_second_call_updates_rather_than_duplicates(self, seeded_db):
        """Re-running the diff must not stack a second copy beside the first."""
        for missing in (["ECG report", "Blood test report"], ["Blood test report"]):
            upsert_followup_task(
                seeded_db, patient_id=1, task_type=MISSING_DOCS,
                details={"missing": missing}, appointment_id=1,
            )

        tasks = seeded_db.query(FollowUpTask).all()
        assert len(tasks) == 1
        assert tasks[0].details["missing"] == ["Blood test report"]

    def test_an_emptied_list_closes_the_task_by_itself(self, seeded_db):
        """Nothing has to remember to come back and tidy up."""
        upsert_followup_task(
            seeded_db, patient_id=1, task_type=MISSING_DOCS,
            details={"missing": ["ECG report"]}, appointment_id=1,
            close_when_empty_key="missing",
        )
        result = upsert_followup_task(
            seeded_db, patient_id=1, task_type=MISSING_DOCS,
            details={"missing": []}, appointment_id=1,
            close_when_empty_key="missing",
        )
        assert result["closed"] is True
        assert seeded_db.query(FollowUpTask).one().status == FollowUpTaskStatus.CLOSED

    def test_nothing_outstanding_and_nothing_open_creates_no_task(self, seeded_db):
        """Opening a task purely to close it is noise in the patient's list."""
        result = upsert_followup_task(
            seeded_db, patient_id=1, task_type=MISSING_DOCS,
            details={"missing": []}, appointment_id=1, close_when_empty_key="missing",
        )
        assert result == {"created": False, "updated": False, "closed": False, "task": None}
        assert seeded_db.query(FollowUpTask).count() == 0

    def test_a_closed_task_does_not_block_a_later_one(self, seeded_db):
        """A document withdrawn after the task closed must reopen the loop."""
        upsert_followup_task(
            seeded_db, patient_id=1, task_type=MISSING_DOCS,
            details={"missing": ["ECG report"]}, appointment_id=1,
            close_when_empty_key="missing",
        )
        upsert_followup_task(
            seeded_db, patient_id=1, task_type=MISSING_DOCS,
            details={"missing": []}, appointment_id=1, close_when_empty_key="missing",
        )
        result = upsert_followup_task(
            seeded_db, patient_id=1, task_type=MISSING_DOCS,
            details={"missing": ["ECG report"]}, appointment_id=1,
            close_when_empty_key="missing",
        )
        assert result["created"] is True
        assert seeded_db.query(FollowUpTask).count() == 2

    def test_tasks_for_different_appointments_are_separate(self, seeded_db):
        for appointment_id in (1, None):
            upsert_followup_task(
                seeded_db, patient_id=1, task_type=MISSING_DOCS,
                details={"missing": ["ECG report"]}, appointment_id=appointment_id,
            )
        assert seeded_db.query(FollowUpTask).count() == 2

    def test_tasks_of_different_types_are_separate(self, seeded_db):
        upsert_followup_task(seeded_db, patient_id=1, task_type=MISSING_DOCS,
                             details={}, appointment_id=1)
        upsert_followup_task(seeded_db, patient_id=1, task_type=FollowUpTaskType.POST_VISIT,
                             details={}, appointment_id=1)
        assert seeded_db.query(FollowUpTask).count() == 2

    def test_another_patients_task_is_never_reused(self, seeded_db):
        for patient_id in (1, 2):
            upsert_followup_task(seeded_db, patient_id=patient_id, task_type=MISSING_DOCS,
                                 details={"missing": ["ECG report"]}, appointment_id=None)
        assert seeded_db.query(FollowUpTask).count() == 2

    def test_opening_a_task_is_audited(self, seeded_db):
        from app.models import AuditEvent

        upsert_followup_task(seeded_db, patient_id=1, task_type=MISSING_DOCS,
                             details={"missing": ["x"]}, appointment_id=1)
        assert "followup_task_opened" in {e.action for e in seeded_db.query(AuditEvent).all()}

    def test_closing_explicitly_reports_the_count(self, seeded_db):
        upsert_followup_task(seeded_db, patient_id=1, task_type=MISSING_DOCS,
                             details={"missing": ["x"]}, appointment_id=1)
        assert close_followup_tasks(seeded_db, patient_id=1, task_type=MISSING_DOCS,
                                    appointment_id=1) == 1
        assert close_followup_tasks(seeded_db, patient_id=1, task_type=MISSING_DOCS,
                                    appointment_id=1) == 0

    def test_listing_shows_only_open_tasks(self, seeded_db):
        upsert_followup_task(seeded_db, patient_id=1, task_type=MISSING_DOCS,
                             details={"missing": ["x"]}, appointment_id=1)
        assert len(list_open_tasks(seeded_db, patient_id=1)) == 1
        close_followup_tasks(seeded_db, patient_id=1, task_type=MISSING_DOCS, appointment_id=1)
        assert list_open_tasks(seeded_db, patient_id=1) == []


class TestEscalationDedup:
    def test_the_first_trigger_opens_an_escalation(self, seeded_db, run):
        result = create_escalation(
            seeded_db, workflow_run_id=run.id, kind=EscalationKind.SAFETY,
            reason="emergency language detected", message="chest pain",
        )
        assert result["created"] is True
        assert result["escalation"]["occurrence_count"] == 1

    def test_repeat_triggers_attach_to_the_open_record(self, seeded_db, run):
        """A frightened patient typing the same thing five times is one queue
        item with five triggers, not five queue items."""
        for _ in range(5):
            create_escalation(
                seeded_db, workflow_run_id=run.id, kind=EscalationKind.SAFETY,
                reason="emergency language detected", message="chest pain",
            )
        escalations = seeded_db.query(Escalation).all()
        assert len(escalations) == 1
        assert escalations[0].occurrence_count == 5

    def test_the_latest_message_is_kept_as_context(self, seeded_db, run):
        """Repetition stays visible as urgency, on the one record."""
        create_escalation(seeded_db, workflow_run_id=run.id, kind=EscalationKind.SAFETY,
                          reason="r", message="first")
        create_escalation(seeded_db, workflow_run_id=run.id, kind=EscalationKind.SAFETY,
                          reason="r", message="second")
        assert seeded_db.query(Escalation).one().latest_message == "second"

    def test_every_trigger_is_audited_even_when_no_row_is_created(self, seeded_db, run):
        """Nothing is lost by not creating a row — the trail is in the audit log."""
        from app.models import AuditEvent

        for _ in range(3):
            create_escalation(seeded_db, workflow_run_id=run.id,
                              kind=EscalationKind.SAFETY, reason="r", message="m")

        actions = [e.action for e in seeded_db.query(AuditEvent).all()]
        assert actions.count("escalation_opened") == 1
        assert actions.count("escalation_retriggered") == 2

    def test_an_acknowledged_escalation_still_absorbs_repeats(self, seeded_db, run):
        """Acknowledged means a human has seen it, not that it is finished."""
        create_escalation(seeded_db, workflow_run_id=run.id, kind=EscalationKind.SAFETY,
                          reason="r", message="m")
        seeded_db.query(Escalation).one().status = EscalationStatus.ACKNOWLEDGED
        seeded_db.flush()

        result = create_escalation(seeded_db, workflow_run_id=run.id,
                                   kind=EscalationKind.SAFETY, reason="r", message="m")
        assert result["attached"] is True

    def test_a_resolved_escalation_does_not_absorb_a_new_trigger(self, seeded_db, run):
        """Once closed, a fresh emergency deserves a fresh queue item."""
        create_escalation(seeded_db, workflow_run_id=run.id, kind=EscalationKind.SAFETY,
                          reason="r", message="m")
        seeded_db.query(Escalation).one().status = EscalationStatus.RESOLVED
        seeded_db.flush()

        result = create_escalation(seeded_db, workflow_run_id=run.id,
                                   kind=EscalationKind.SAFETY, reason="r", message="m2")
        assert result["created"] is True
        assert seeded_db.query(Escalation).count() == 2

    def test_escalations_on_different_runs_are_separate(self, seeded_db, run):
        other = WorkflowRun(patient_id=2, status=WorkflowStatus.IN_PROGRESS)
        seeded_db.add(other)
        seeded_db.flush()

        for run_id in (run.id, other.id):
            create_escalation(seeded_db, workflow_run_id=run_id,
                              kind=EscalationKind.SAFETY, reason="r", message="m")
        assert seeded_db.query(Escalation).count() == 2

    def test_the_staff_queue_lists_open_escalations(self, seeded_db, run):
        create_escalation(seeded_db, workflow_run_id=run.id, kind=EscalationKind.SAFETY,
                          reason="r", message="m")
        assert len(list_open_escalations(seeded_db)) == 1

    def test_resolved_escalations_leave_the_queue(self, seeded_db, run):
        create_escalation(seeded_db, workflow_run_id=run.id, kind=EscalationKind.SAFETY,
                          reason="r", message="m")
        seeded_db.query(Escalation).one().status = EscalationStatus.RESOLVED
        seeded_db.flush()
        assert list_open_escalations(seeded_db) == []

    def test_results_are_json_serialisable(self, seeded_db, run):
        import json

        json.dumps(create_escalation(seeded_db, workflow_run_id=run.id,
                                     kind=EscalationKind.SAFETY, reason="r", message="m"))
        json.dumps(list_open_escalations(seeded_db))
        json.dumps(list_open_tasks(seeded_db, patient_id=1))
