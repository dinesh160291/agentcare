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
    close_escalations_for_run,
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

    def test_a_shortfall_that_returns_reopens_the_same_row(self, seeded_db):
        """A document withdrawn after the task closed must reopen the loop —
        in place.

        The growth rule is one row per subject, not one *open* row. A
        reappearing shortfall that created a sibling would leave a trail of
        closed rows behind a patient who changed nothing, and the row count is
        the thing the boundedness invariant actually bounds.
        """
        upsert_followup_task(
            seeded_db, patient_id=1, task_type=MISSING_DOCS,
            details={"missing": ["ECG report"]}, appointment_id=1,
            close_when_empty_key="missing",
        )
        closed = upsert_followup_task(
            seeded_db, patient_id=1, task_type=MISSING_DOCS,
            details={"missing": []}, appointment_id=1, close_when_empty_key="missing",
        )
        assert closed["closed"] is True

        result = upsert_followup_task(
            seeded_db, patient_id=1, task_type=MISSING_DOCS,
            details={"missing": ["ECG report"]}, appointment_id=1,
            close_when_empty_key="missing",
        )
        assert result["created"] is False
        assert result["closed"] is False
        assert result["task"]["status"] == "open"
        assert seeded_db.query(FollowUpTask).count() == 1

    def test_the_row_count_never_grows_past_one_per_subject(self, seeded_db):
        """Whatever the diff does, however often it runs."""
        for missing in (["ECG report"], [], ["ECG report", "Blood test report"], [], []):
            upsert_followup_task(
                seeded_db, patient_id=1, task_type=MISSING_DOCS,
                details={"missing": missing}, appointment_id=1,
                close_when_empty_key="missing",
            )
        assert seeded_db.query(FollowUpTask).count() == 1

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


class TestEscalationKindIsPartOfTheKey:
    """One open record per *subject*, and a kind is a subject.

    The live failure: a run queued for review with a ``low_confidence_routing``
    escalation, then the patient sent two messages the safety screen fired on.
    Keyed on the run alone, both were absorbed as occurrences of the routing
    question — one queue item, ``low_confidence_routing``, count 3, whose reason
    still described a department choice. Nothing in the queue said "safety", so
    triage would never have treated it as one.
    """

    def test_a_different_kind_opens_its_own_record(self, seeded_db, run):
        create_escalation(
            seeded_db, workflow_run_id=run.id,
            kind=EscalationKind.LOW_CONFIDENCE_ROUTING, reason="r", message="m",
        )
        result = create_escalation(
            seeded_db, workflow_run_id=run.id,
            kind=EscalationKind.SAFETY, reason="r2", message="m2",
        )

        assert result["created"] is True
        assert seeded_db.query(Escalation).count() == 2

    def test_the_safety_case_is_visible_as_safety_in_the_queue(self, seeded_db, run):
        """The assertion that would have caught it. A count of two is not
        enough — what triage needs is the *kind* on a row of its own."""
        create_escalation(
            seeded_db, workflow_run_id=run.id,
            kind=EscalationKind.LOW_CONFIDENCE_ROUTING, reason="r", message="m",
        )
        create_escalation(
            seeded_db, workflow_run_id=run.id,
            kind=EscalationKind.SAFETY, reason="r2", message="chest pain",
        )

        queued = list_open_escalations(seeded_db)
        safety = [item for item in queued if item["kind"] == "safety"]
        assert len(safety) == 1
        assert safety[0]["latest_message"] == "chest pain"

    def test_repeats_of_the_same_kind_still_attach(self, seeded_db, run):
        """The bound the dedup exists for is untouched: it now holds *within* a
        kind, which is where the repetition actually happens."""
        create_escalation(
            seeded_db, workflow_run_id=run.id,
            kind=EscalationKind.SAFETY, reason="r", message="chest pain",
        )
        for _ in range(4):
            create_escalation(
                seeded_db, workflow_run_id=run.id,
                kind=EscalationKind.SAFETY, reason="r", message="chest pain",
            )

        rows = seeded_db.query(Escalation).all()
        assert len(rows) == 1
        assert rows[0].occurrence_count == 5


class TestClosingADeadRunsEscalations:
    """The derivation invariant, applied to the staff queue."""

    def test_an_open_review_is_retired(self, seeded_db, run):
        create_escalation(
            seeded_db, workflow_run_id=run.id,
            kind=EscalationKind.LOW_CONFIDENCE_ROUTING, reason="r", message="m",
        )

        closed = close_escalations_for_run(
            seeded_db, workflow_run_id=run.id, note="Superseded by a later request."
        )

        assert closed == 1
        row = seeded_db.query(Escalation).one()
        assert row.status is EscalationStatus.RESOLVED
        assert row.resolution_note == "Superseded by a later request."
        assert list_open_escalations(seeded_db) == []

    def test_a_safety_escalation_is_never_closed_this_way(self, seeded_db, run):
        """"Actually, forget it" after a chest-pain message is exactly the
        moment the system must not be helpful. The state machine already
        refuses to move an escalated run; this is the second lock, on the queue
        item rather than on the run."""
        create_escalation(
            seeded_db, workflow_run_id=run.id,
            kind=EscalationKind.SAFETY, reason="r", message="chest pain",
        )

        closed = close_escalations_for_run(
            seeded_db, workflow_run_id=run.id, note="Withdrawn by the patient."
        )

        assert closed == 0
        assert seeded_db.query(Escalation).one().status is EscalationStatus.OPEN
        assert len(list_open_escalations(seeded_db)) == 1

    def test_another_runs_queue_item_is_untouched(self, seeded_db, run):
        other = WorkflowRun(patient_id=2, status=WorkflowStatus.IN_PROGRESS)
        seeded_db.add(other)
        seeded_db.flush()
        for run_id in (run.id, other.id):
            create_escalation(
                seeded_db, workflow_run_id=run_id,
                kind=EscalationKind.LOW_CONFIDENCE_ROUTING, reason="r", message="m",
            )

        close_escalations_for_run(seeded_db, workflow_run_id=run.id, note="n")

        assert len(list_open_escalations(seeded_db)) == 1
