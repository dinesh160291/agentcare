"""The workflow state machine — written before the implementation exists.

This file transcribes the PRD's pinned transition table (§ Workflow state
machine). It is deliberately a transcription rather than a derivation: if the
implementation and the table ever disagree, the failure should point at the
table, not at a clever restatement of it.

Four properties are pinned here, and each one exists because of a specific
failure it prevents:

* **Exhaustive legality.** Every legal edge succeeds and a sample of illegal
  ones raise. A state machine nobody enumerated is a set of ``if`` statements
  wearing a hat.
* **Compare-and-swap.** A double-clicked Confirm sends two requests that both
  read ``pending_confirmation``. The booking transaction would catch the
  double-book one layer down; the CAS stops it here, and applies to every edge
  rather than only the one that happened to be dangerous.
* **Both ledgers.** Every transition writes an ``AuditEvent`` *and* a
  ``TraceEvent``. A transition visible in only one is a transition somebody
  will later fail to explain.
* **Cancellation carries a reason.** The staff queue distinguishes "withdrawn
  while pending" from "superseded" by query, not by live state — which only
  works if the reason was recorded at the moment it was known.
"""

from __future__ import annotations

import pytest

from app.db import SessionLocal
from app.errors import InvalidTransition, ValidationFailed
from app.models import (
    AuditEvent,
    CancellationReason,
    TraceEvent,
    TraceEventType,
    WorkflowRun,
    WorkflowStatus,
)
from app.trace import TraceWriter
from app.workflow.state_machine import (
    INITIAL_STATUSES,
    LEGAL_TRANSITIONS,
    create_run,
    is_legal,
    transition,
)

S = WorkflowStatus

#: The PRD's table, transcribed. (from, to) pairs only — triggers are prose in
#: the PRD and are carried into the audit metadata, not into the machine.
PINNED_EDGES = {
    (S.IN_PROGRESS, S.PENDING_CONFIRMATION),
    (S.PENDING_CONFIRMATION, S.IN_PROGRESS),
    (S.IN_PROGRESS, S.PENDING_REVIEW),
    (S.PENDING_REVIEW, S.IN_PROGRESS),
    (S.PENDING_REVIEW, S.REJECTED),
    (S.IN_PROGRESS, S.CANCELLED),
    (S.PENDING_CONFIRMATION, S.CANCELLED),
    (S.PENDING_REVIEW, S.CANCELLED),
    # "any non-terminal -> escalated": the screen runs on every message,
    # whatever the current state.
    (S.IN_PROGRESS, S.ESCALATED),
    (S.PENDING_CONFIRMATION, S.ESCALATED),
    (S.PENDING_REVIEW, S.ESCALATED),
    (S.IN_PROGRESS, S.COMPLETED),
    (S.IN_PROGRESS, S.FAILED),
}

TERMINAL = {S.COMPLETED, S.REJECTED, S.FAILED, S.CANCELLED, S.ESCALATED}

#: Cancellation must name its reason, so every edge to CANCELLED needs one.
REASON_FOR = {
    (S.IN_PROGRESS, S.CANCELLED): CancellationReason.WITHDRAWN,
    (S.PENDING_CONFIRMATION, S.CANCELLED): CancellationReason.WITHDRAWN,
    (S.PENDING_REVIEW, S.CANCELLED): CancellationReason.WITHDRAWN_DURING_REVIEW,
}


@pytest.fixture
def writer(seeded_db):
    return TraceWriter(seeded_db, session_id="test-session")


def make_run(session, status: WorkflowStatus) -> WorkflowRun:
    """A run parked in a given state, without going through the machine.

    Tests need arbitrary starting states; the machine is what is under test, so
    reaching them *through* it would make every test depend on every edge.
    """
    run = WorkflowRun(patient_id=1, status=status, request_text="pinned test run")
    session.add(run)
    session.flush()
    return run


class TestTheTableMatchesThePRD:
    """The machine's own declaration, checked against the transcription above."""

    def test_the_declared_edges_are_exactly_the_pinned_edges(self):
        declared = {
            (source, target)
            for source, targets in LEGAL_TRANSITIONS.items()
            for target in targets
        }
        assert declared == PINNED_EDGES

    def test_terminal_states_have_no_outgoing_edges(self):
        """Terminal *for automation*: ownership passes to an Escalation record
        or to the patient starting again."""
        for status in TERMINAL:
            assert not LEGAL_TRANSITIONS.get(status), f"{status} should be terminal"

    def test_only_two_initial_states_exist(self):
        """Runs are born ``in_progress``, or born ``escalated`` when the safety
        screen fires on a session's opening message — the one case where the
        highest-priority path would otherwise have no run to key an Escalation
        to."""
        assert INITIAL_STATUSES == frozenset({S.IN_PROGRESS, S.ESCALATED})

    def test_cancelled_is_reachable_from_every_non_terminal_state(self):
        """Withdrawal is typed state. A withdrawal that lived only in the
        transcript would leave a ``pending_review`` row reading "waiting to
        resume" for a request the patient abandoned — the zombie approval."""
        for status in (S.IN_PROGRESS, S.PENDING_CONFIRMATION, S.PENDING_REVIEW):
            assert is_legal(status, S.CANCELLED)

    def test_withdrawal_cannot_close_an_escalated_run(self):
        """The one state withdrawal must not close. "Actually forget it" after
        a safety trigger stays in front of humans."""
        assert not is_legal(S.ESCALATED, S.CANCELLED)


class TestEveryLegalEdge:
    @pytest.mark.parametrize("source,target", sorted(PINNED_EDGES, key=str))
    def test_the_edge_applies(self, seeded_db, writer, source, target):
        run = make_run(seeded_db, source)
        result = transition(
            seeded_db,
            run,
            to=target,
            trigger="pinned-edge-test",
            writer=writer,
            reason=REASON_FOR.get((source, target)),
        )

        assert result.applied is True
        assert result.from_status is source
        assert result.to_status is target
        assert run.status is target


class TestIllegalEdges:
    ILLEGAL = [
        (S.IN_PROGRESS, S.REJECTED),        # only staff, only from review
        (S.PENDING_CONFIRMATION, S.COMPLETED),  # a confirmation is not the plan
        (S.PENDING_CONFIRMATION, S.PENDING_REVIEW),
        (S.COMPLETED, S.IN_PROGRESS),       # terminal
        (S.CANCELLED, S.IN_PROGRESS),       # terminal
        (S.ESCALATED, S.CANCELLED),         # withdrawal must not close it
        (S.REJECTED, S.ESCALATED),          # terminal
    ]

    @pytest.mark.parametrize("source,target", ILLEGAL)
    def test_the_edge_raises(self, seeded_db, writer, source, target):
        run = make_run(seeded_db, source)
        with pytest.raises(InvalidTransition):
            transition(
                seeded_db, run, to=target, trigger="illegal-edge-test", writer=writer
            )
        assert run.status is source

    def test_an_illegal_attempt_is_audited(self, seeded_db, writer):
        run = make_run(seeded_db, S.COMPLETED)
        with pytest.raises(InvalidTransition):
            transition(
                seeded_db, run, to=S.IN_PROGRESS, trigger="resume", writer=writer
            )

        audits = (
            seeded_db.query(AuditEvent)
            .filter(AuditEvent.action == "workflow_transition_illegal")
            .all()
        )
        assert len(audits) == 1
        assert audits[0].event_metadata["from"] == "completed"
        assert audits[0].event_metadata["to"] == "in_progress"

    def test_an_illegal_attempt_is_traced_as_a_rejection(self, seeded_db, writer):
        """Every rejection recorded — the trace-completeness invariant does not
        exempt the ones that raised."""
        run = make_run(seeded_db, S.CANCELLED)
        with pytest.raises(InvalidTransition):
            transition(
                seeded_db, run, to=S.IN_PROGRESS, trigger="resume", writer=writer
            )

        rejections = [
            event
            for event in seeded_db.query(TraceEvent).all()
            if event.event_type is TraceEventType.VALIDATION
            and event.payload["accepted"] is False
        ]
        assert len(rejections) == 1
        assert rejections[0].payload["what"] == "workflow_transition"

    def test_a_self_edge_raises(self, seeded_db, writer):
        """A self-stay is expressed by *not calling* the machine. Answering a
        side question in ``pending_confirmation`` changes no state, so silently
        accepting a no-change call would hide the case where code meant to move
        and computed the same status by mistake."""
        run = make_run(seeded_db, S.IN_PROGRESS)
        with pytest.raises(InvalidTransition):
            transition(
                seeded_db, run, to=S.IN_PROGRESS, trigger="self", writer=writer
            )


class TestCompareAndSwap:
    def test_the_loser_of_a_race_no_ops_rather_than_crashing(self, seeded_db, writer):
        """Two requests both read ``pending_confirmation`` — the double-clicked
        Confirm. One wins; the other must not double-fire and must not raise."""
        run = make_run(seeded_db, S.PENDING_CONFIRMATION)
        seeded_db.commit()
        run_id = run.id

        # A second session that loaded the row *before* the winner committed:
        # its in-memory copy still says pending_confirmation.
        loser_session = SessionLocal()
        try:
            loser_run = loser_session.get(WorkflowRun, run_id)
            assert loser_run.status is S.PENDING_CONFIRMATION

            winner = transition(
                seeded_db, run, to=S.IN_PROGRESS, trigger="confirm", writer=writer
            )
            seeded_db.commit()
            assert winner.applied is True

            loser_writer = TraceWriter(loser_session, session_id="loser-session")
            loser = transition(
                loser_session,
                loser_run,
                to=S.IN_PROGRESS,
                trigger="confirm",
                writer=loser_writer,
            )
            loser_session.commit()
        finally:
            loser_session.close()

        assert loser.applied is False, "the losing attempt must not report success"

    def test_the_loser_leaves_the_row_alone(self, seeded_db, writer):
        """The dangerous version of losing is applying anyway. Here the winner
        moves to ``cancelled``; the loser still believes it can confirm."""
        run = make_run(seeded_db, S.PENDING_CONFIRMATION)
        seeded_db.commit()
        run_id = run.id

        loser_session = SessionLocal()
        try:
            loser_run = loser_session.get(WorkflowRun, run_id)

            transition(
                seeded_db,
                run,
                to=S.CANCELLED,
                trigger="withdraw",
                writer=writer,
                reason=CancellationReason.WITHDRAWN,
            )
            seeded_db.commit()

            loser = transition(
                loser_session,
                loser_run,
                to=S.IN_PROGRESS,
                trigger="confirm",
                writer=TraceWriter(loser_session, session_id="loser-session"),
            )
            loser_session.commit()
        finally:
            loser_session.close()

        assert loser.applied is False
        seeded_db.expire_all()
        assert seeded_db.get(WorkflowRun, run_id).status is S.CANCELLED

    def test_a_lost_race_is_still_recorded(self, seeded_db, writer):
        """Losing is an event. A no-op nobody can see is indistinguishable from
        a request that never arrived."""
        run = make_run(seeded_db, S.PENDING_CONFIRMATION)
        seeded_db.commit()
        run_id = run.id

        loser_session = SessionLocal()
        try:
            loser_run = loser_session.get(WorkflowRun, run_id)
            transition(
                seeded_db, run, to=S.IN_PROGRESS, trigger="confirm", writer=writer
            )
            seeded_db.commit()

            transition(
                loser_session,
                loser_run,
                to=S.IN_PROGRESS,
                trigger="confirm",
                writer=TraceWriter(loser_session, session_id="loser-session"),
            )
            loser_session.commit()

            lost = (
                loser_session.query(AuditEvent)
                .filter(AuditEvent.action == "workflow_transition_lost_race")
                .all()
            )
            assert len(lost) == 1
        finally:
            loser_session.close()


class TestBothLedgers:
    def test_a_transition_writes_an_audit_event(self, seeded_db, writer):
        run = make_run(seeded_db, S.IN_PROGRESS)
        transition(
            seeded_db,
            run,
            to=S.PENDING_CONFIRMATION,
            trigger="appointment_proposal",
            writer=writer,
        )

        audit = (
            seeded_db.query(AuditEvent)
            .filter(AuditEvent.action == "workflow_transition")
            .one()
        )
        assert audit.entity_type == "workflow_run"
        assert audit.entity_id == run.id
        assert audit.event_metadata["from"] == "in_progress"
        assert audit.event_metadata["to"] == "pending_confirmation"
        assert audit.event_metadata["trigger"] == "appointment_proposal"

    def test_a_transition_writes_a_trace_event(self, seeded_db, writer):
        run = make_run(seeded_db, S.IN_PROGRESS)
        transition(
            seeded_db, run, to=S.COMPLETED, trigger="plan_complete", writer=writer
        )

        events = [
            event
            for event in seeded_db.query(TraceEvent).all()
            if event.event_type is TraceEventType.TRANSITION
        ]
        assert len(events) == 1
        assert events[0].payload["to"] == "completed"
        assert events[0].payload["applied"] is True

    def test_the_trace_event_is_bound_to_the_run(self, seeded_db, writer):
        run = make_run(seeded_db, S.IN_PROGRESS)
        transition(
            seeded_db, run, to=S.FAILED, trigger="budget_exhausted", writer=writer
        )
        event = (
            seeded_db.query(TraceEvent)
            .filter(TraceEvent.event_type == TraceEventType.TRANSITION)
            .one()
        )
        assert event.workflow_run_id == run.id


class TestCancellationReason:
    def test_cancelling_without_a_reason_is_refused(self, seeded_db, writer):
        """The staff queue reads terminal states by query. "Cancelled" with no
        reason cannot be sorted into "withdrawn while pending" or "superseded",
        so the reason is required at the one moment it is known."""
        run = make_run(seeded_db, S.IN_PROGRESS)
        with pytest.raises(ValidationFailed):
            transition(seeded_db, run, to=S.CANCELLED, trigger="withdraw", writer=writer)
        assert run.status is S.IN_PROGRESS

    def test_the_reason_is_persisted_on_the_row(self, seeded_db, writer):
        run = make_run(seeded_db, S.PENDING_REVIEW)
        transition(
            seeded_db,
            run,
            to=S.CANCELLED,
            trigger="withdraw",
            writer=writer,
            reason=CancellationReason.WITHDRAWN_DURING_REVIEW,
        )
        assert run.cancellation_reason == "withdrawn_during_review"

    def test_supersede_is_recorded_as_its_own_reason(self, seeded_db, writer):
        run = make_run(seeded_db, S.PENDING_REVIEW)
        transition(
            seeded_db,
            run,
            to=S.CANCELLED,
            trigger="superseded_by_new_request",
            writer=writer,
            reason=CancellationReason.SUPERSEDED,
        )
        assert run.cancellation_reason == "superseded"

    def test_a_reason_is_not_accepted_on_a_non_cancelling_edge(self, seeded_db, writer):
        """A cancellation reason on a completed run would misreport why it
        ended, and the queue query reads that column."""
        run = make_run(seeded_db, S.IN_PROGRESS)
        with pytest.raises(ValidationFailed):
            transition(
                seeded_db,
                run,
                to=S.COMPLETED,
                trigger="plan_complete",
                writer=writer,
                reason=CancellationReason.WITHDRAWN,
            )


class TestRunCreation:
    def test_a_run_is_born_in_progress(self, seeded_db, writer):
        run = create_run(
            seeded_db,
            patient_id=1,
            status=S.IN_PROGRESS,
            trigger="intent_accepted",
            writer=writer,
            request_text="I need a cardiology appointment",
        )
        assert run.id is not None
        assert run.status is S.IN_PROGRESS
        assert run.request_text == "I need a cardiology appointment"

    def test_a_run_can_be_born_escalated(self, seeded_db, writer):
        """The safety door: an emergency on a session's opening message needs a
        run to key its Escalation to, or the FK becomes nullable and every
        trace query grows an orphan special case."""
        run = create_run(
            seeded_db,
            patient_id=1,
            status=S.ESCALATED,
            trigger="safety_screen_opening_message",
            writer=writer,
            request_text="redacted",
        )
        assert run.status is S.ESCALATED

    @pytest.mark.parametrize(
        "status", [S.PENDING_CONFIRMATION, S.PENDING_REVIEW, S.COMPLETED, S.CANCELLED]
    )
    def test_no_other_initial_state_is_permitted(self, seeded_db, writer, status):
        with pytest.raises(ValidationFailed):
            create_run(
                seeded_db,
                patient_id=1,
                status=status,
                trigger="invented",
                writer=writer,
            )

    def test_creation_writes_both_ledgers(self, seeded_db, writer):
        run = create_run(
            seeded_db,
            patient_id=1,
            status=S.IN_PROGRESS,
            trigger="intent_accepted",
            writer=writer,
        )

        audit = (
            seeded_db.query(AuditEvent)
            .filter(AuditEvent.action == "workflow_run_created")
            .one()
        )
        assert audit.entity_id == run.id

        event = (
            seeded_db.query(TraceEvent)
            .filter(TraceEvent.event_type == TraceEventType.TRANSITION)
            .one()
        )
        assert event.payload["from"] is None
        assert event.payload["to"] == "in_progress"

    def test_the_writer_is_bound_to_the_new_run(self, seeded_db, writer):
        """A turn opens before its run exists. Everything after creation must
        be attributable to the run, or the timeline splits in half."""
        run = create_run(
            seeded_db,
            patient_id=1,
            status=S.IN_PROGRESS,
            trigger="intent_accepted",
            writer=writer,
        )
        assert writer.workflow_run_id == run.id
