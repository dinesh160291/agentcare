"""Message→run mapping: what a message arriving during an active run may do.

Written before the implementation. One active run per patient, and every
subsequent message is classified into one of six classes — the model proposes,
code validates the proposal, and only then does anything happen to the run.

The classification is where this system is most exposed, because each class has
a different cost when it is wrong:

* A wrongly superseded **review** costs the patient a re-ask.
* A zombie resume — acting on a request the patient abandoned — costs an
  incident.
* A wrongly superseded **cooperation** ("also, here's my ECG" read as a new
  request) costs the booking the patient actually wanted.

That last one is why classification precedes consequence instead of supersede
being the default for everything, and why code re-checks the one rule the model
is most likely to get wrong: a message carrying the same intent as the active
run can never be complementary.
"""

from __future__ import annotations

import pytest

from app.errors import ClassRejected, ValidationFailed
from app.models import (
    AuditEvent,
    MessageClass,
    PlanStep,
    TraceEvent,
    TraceEventType,
    WorkflowRun,
    WorkflowStatus,
)
from app.trace import TraceWriter
from app.workflow.mapping import (
    CLASSIFICATION_ORDER,
    Consequence,
    apply_consequence,
    primary_intent,
    validate_class,
)

S = WorkflowStatus
M = MessageClass


@pytest.fixture
def writer(seeded_db):
    return TraceWriter(seeded_db, session_id="test-session")


def make_run(session, status=S.IN_PROGRESS, plan=("route", "book"), text="book cardiology"):
    run = WorkflowRun(
        patient_id=1,
        status=status,
        plan=list(plan),
        completed_steps=[],
        request_text=text,
    )
    session.add(run)
    session.flush()
    return run


class TestClassificationOrder:
    def test_the_order_is_the_pinned_one(self):
        """Withdrawal outranks everything: a patient abandoning a request must
        not be read as a continuation of it. Off-topic sits second because an
        off-topic message that fell through to continuation would be appended
        to the run's request text and contaminate what routing reads later."""
        assert CLASSIFICATION_ORDER == (
            M.WITHDRAWAL,
            M.OFF_TOPIC,
            M.SIDE_QUESTION,
            M.COMPLEMENTARY,
            M.CONFLICTING,
            M.CONTINUATION,
        )

    def test_every_class_appears_exactly_once(self):
        assert len(CLASSIFICATION_ORDER) == len(set(CLASSIFICATION_ORDER))
        assert set(CLASSIFICATION_ORDER) == set(MessageClass)


class TestClassValidation:
    def test_a_known_class_is_accepted(self, seeded_db, writer):
        run = make_run(seeded_db)
        verdict = validate_class(
            "continuation", run=run, incoming_steps=None, writer=writer
        )
        assert verdict.message_class is M.CONTINUATION
        assert verdict.adjusted is False

    def test_an_unknown_class_is_rejected(self, seeded_db, writer):
        run = make_run(seeded_db)
        with pytest.raises(ClassRejected):
            validate_class("escalate_it", run=run, incoming_steps=None, writer=writer)

    def test_prose_is_rejected(self, seeded_db, writer):
        run = make_run(seeded_db)
        with pytest.raises(ClassRejected):
            validate_class(
                "the patient seems to be continuing",
                run=run,
                incoming_steps=None,
                writer=writer,
            )

    def test_every_validation_is_traced(self, seeded_db, writer):
        """The validation verdict is a capture point in its own right — without
        it a rejected class followed by a retry reads as the model
        inexplicably classifying twice."""
        run = make_run(seeded_db)
        validate_class("side_question", run=run, incoming_steps=None, writer=writer)

        events = [
            event
            for event in seeded_db.query(TraceEvent).all()
            if event.event_type is TraceEventType.VALIDATION
        ]
        assert len(events) == 1
        assert events[0].payload["what"] == "message_class"
        assert events[0].payload["accepted"] is True

    def test_a_rejection_is_traced_too(self, seeded_db, writer):
        run = make_run(seeded_db)
        with pytest.raises(ClassRejected):
            validate_class("nonsense", run=run, incoming_steps=None, writer=writer)

        rejected = [
            event
            for event in seeded_db.query(TraceEvent).all()
            if event.event_type is TraceEventType.VALIDATION
            and event.payload["accepted"] is False
        ]
        assert len(rejected) == 1


class TestComplementaryCannotCarryTheSameIntent:
    def test_a_second_booking_request_is_downgraded_to_conflicting(
        self, seeded_db, writer
    ):
        """"Book me a cardiology appointment" while a booking run is live is a
        rephrase, not a cooperation. Complementary is reserved for compatible
        intent types that serve the active goal."""
        run = make_run(seeded_db, plan=("route", "book"))
        verdict = validate_class(
            "complementary", run=run, incoming_steps=[PlanStep.BOOK], writer=writer
        )

        assert verdict.message_class is M.CONFLICTING
        assert verdict.adjusted is True
        assert verdict.proposed is M.COMPLEMENTARY

    def test_a_document_upload_during_a_booking_stays_complementary(
        self, seeded_db, writer
    ):
        """The ECG-during-confirmation trace: the booking must survive."""
        run = make_run(seeded_db, plan=("route", "book"))
        verdict = validate_class(
            "complementary",
            run=run,
            incoming_steps=[PlanStep.DOCUMENTS],
            writer=writer,
        )
        assert verdict.message_class is M.COMPLEMENTARY
        assert verdict.adjusted is False

    def test_the_downgrade_is_traced_as_an_adjustment(self, seeded_db, writer):
        run = make_run(seeded_db, plan=("route", "book"))
        validate_class(
            "complementary", run=run, incoming_steps=[PlanStep.BOOK], writer=writer
        )
        event = (
            seeded_db.query(TraceEvent)
            .filter(TraceEvent.event_type == TraceEventType.VALIDATION)
            .one()
        )
        assert event.payload["accepted"] is False
        assert event.payload["detail"]["applied"] == "conflicting"

    def test_a_document_upload_during_a_document_run_is_conflicting(
        self, seeded_db, writer
    ):
        """The rule is about intent identity, not about booking specifically."""
        run = make_run(seeded_db, plan=("documents",))
        verdict = validate_class(
            "complementary",
            run=run,
            incoming_steps=[PlanStep.DOCUMENTS],
            writer=writer,
        )
        assert verdict.message_class is M.CONFLICTING

    def test_no_incoming_steps_leaves_complementary_alone(self, seeded_db, writer):
        """Nothing to compare against is not evidence of a clash."""
        run = make_run(seeded_db, plan=("route", "book"))
        verdict = validate_class(
            "complementary", run=run, incoming_steps=None, writer=writer
        )
        assert verdict.message_class is M.COMPLEMENTARY

    def test_only_complementary_is_ever_downgraded(self, seeded_db, writer):
        """A continuation carrying the same intent is exactly what a
        continuation is — the downgrade must not spread to other classes."""
        run = make_run(seeded_db, plan=("route", "book"))
        verdict = validate_class(
            "continuation", run=run, incoming_steps=[PlanStep.BOOK], writer=writer
        )
        assert verdict.message_class is M.CONTINUATION


class TestPrimaryIntent:
    def test_booking_is_the_widest_intent(self):
        assert primary_intent(["route", "book", "documents"]) is PlanStep.BOOK

    def test_a_document_plan_reports_documents(self):
        assert primary_intent(["documents"]) is PlanStep.DOCUMENTS

    def test_routing_alone_reports_routing(self):
        assert primary_intent(["route"]) is PlanStep.ROUTE

    def test_an_empty_plan_has_no_intent(self):
        assert primary_intent([]) is None


class TestOffTopicConsequence:
    def test_off_topic_changes_no_state(self, seeded_db, writer):
        """The PRD's named Layer-1 scenario: off-topic during an in_progress
        lull leaves run state and request text byte-identical."""
        run = make_run(seeded_db, text="I need a cardiology appointment")
        before_status, before_text = run.status, run.request_text

        outcome = apply_consequence(
            seeded_db, run, _verdict(M.OFF_TOPIC), writer=writer, message="what's the weather?"
        )

        assert outcome.consequence is Consequence.SCOPE_REPLY
        assert run.status is before_status
        assert run.request_text == before_text

    def test_off_topic_never_touches_the_request_text(self, seeded_db, writer):
        """The contamination this class exists to prevent: an off-topic message
        appended to the request text is read later by routing and by slot
        matching."""
        run = make_run(seeded_db, text="I need a cardiology appointment")
        apply_consequence(
            seeded_db, run, _verdict(M.OFF_TOPIC), writer=writer, message="who won the match"
        )
        assert "match" not in run.request_text

    def test_off_topic_does_not_supersede(self, seeded_db, writer):
        run = make_run(seeded_db)
        apply_consequence(
            seeded_db, run, _verdict(M.OFF_TOPIC), writer=writer, message="hello there"
        )
        assert run.status is S.IN_PROGRESS
        assert run.cancellation_reason is None

    def test_off_topic_spawns_no_run(self, seeded_db, writer):
        run = make_run(seeded_db)
        outcome = apply_consequence(
            seeded_db, run, _verdict(M.OFF_TOPIC), writer=writer, message="hello"
        )
        assert outcome.spawns_new_run is False


class TestSideQuestionConsequence:
    def test_a_side_question_leaves_the_run_where_it_was(self, seeded_db, writer):
        run = make_run(seeded_db, status=S.PENDING_CONFIRMATION)
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.SIDE_QUESTION),
            writer=writer,
            message="what docs do I have on file?",
        )
        assert outcome.consequence is Consequence.ANSWER_AND_STAY
        assert run.status is S.PENDING_CONFIRMATION

    def test_a_side_question_spawns_no_run(self, seeded_db, writer):
        run = make_run(seeded_db)
        outcome = apply_consequence(
            seeded_db, run, _verdict(M.SIDE_QUESTION), writer=writer, message="what docs?"
        )
        assert outcome.spawns_new_run is False


class TestComplementaryConsequence:
    def test_the_step_is_appended_to_the_live_plan(self, seeded_db, writer):
        run = make_run(seeded_db, plan=("route", "book"))
        apply_consequence(
            seeded_db,
            run,
            _verdict(M.COMPLEMENTARY),
            writer=writer,
            message="also here's my ECG",
            incoming_steps=[PlanStep.DOCUMENTS],
        )
        assert "documents" in run.plan

    def test_the_booking_survives(self, seeded_db, writer):
        """Never supersede. The whole point of the class."""
        run = make_run(seeded_db, status=S.PENDING_CONFIRMATION, plan=("route", "book"))
        apply_consequence(
            seeded_db,
            run,
            _verdict(M.COMPLEMENTARY),
            writer=writer,
            message="also here's my ECG",
            incoming_steps=[PlanStep.DOCUMENTS],
        )
        assert run.status is S.PENDING_CONFIRMATION
        assert "book" in run.plan

    def test_it_does_not_consume_the_replan_budget(self, seeded_db, writer):
        """A deterministic append is not a re-plan — no model was asked."""
        run = make_run(seeded_db, plan=("route", "book"))
        apply_consequence(
            seeded_db,
            run,
            _verdict(M.COMPLEMENTARY),
            writer=writer,
            message="also my ECG",
            incoming_steps=[PlanStep.DOCUMENTS],
        )
        assert run.replan_count == 0


class TestConflictingConsequence:
    def test_the_old_run_is_cancelled_as_superseded(self, seeded_db, writer):
        run = make_run(seeded_db, status=S.PENDING_REVIEW)
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONFLICTING),
            writer=writer,
            message="actually make it dermatology",
        )
        assert outcome.consequence is Consequence.SUPERSEDE
        assert run.status is S.CANCELLED
        assert run.cancellation_reason == "superseded"

    def test_a_replacement_run_is_called_for(self, seeded_db, writer):
        run = make_run(seeded_db)
        outcome = apply_consequence(
            seeded_db, run, _verdict(M.CONFLICTING), writer=writer, message="make it derm"
        )
        assert outcome.spawns_new_run is True

    def test_the_supersede_is_audited(self, seeded_db, writer):
        run = make_run(seeded_db)
        apply_consequence(
            seeded_db, run, _verdict(M.CONFLICTING), writer=writer, message="make it derm"
        )
        audits = [
            event
            for event in seeded_db.query(AuditEvent).all()
            if event.action == "workflow_transition"
        ]
        assert audits and audits[-1].event_metadata["reason"] == "superseded"


class TestWithdrawalConsequence:
    def test_withdrawal_cancels_an_in_progress_run(self, seeded_db, writer):
        run = make_run(seeded_db, status=S.IN_PROGRESS)
        outcome = apply_consequence(
            seeded_db, run, _verdict(M.WITHDRAWAL), writer=writer, message="never mind"
        )
        assert outcome.consequence is Consequence.WITHDRAW
        assert run.status is S.CANCELLED
        assert run.cancellation_reason == "withdrawn"

    def test_withdrawal_during_review_records_its_own_reason(self, seeded_db, writer):
        """The staff queue's "withdrawn while pending" section is a query over
        this column."""
        run = make_run(seeded_db, status=S.PENDING_REVIEW)
        apply_consequence(
            seeded_db, run, _verdict(M.WITHDRAWAL), writer=writer, message="forget it"
        )
        assert run.cancellation_reason == "withdrawn_during_review"

    def test_withdrawal_from_pending_confirmation_is_a_plain_withdrawal(
        self, seeded_db, writer
    ):
        run = make_run(seeded_db, status=S.PENDING_CONFIRMATION)
        apply_consequence(
            seeded_db, run, _verdict(M.WITHDRAWAL), writer=writer, message="forget it"
        )
        assert run.cancellation_reason == "withdrawn"

    def test_withdrawal_spawns_no_replacement(self, seeded_db, writer):
        run = make_run(seeded_db)
        outcome = apply_consequence(
            seeded_db, run, _verdict(M.WITHDRAWAL), writer=writer, message="never mind"
        )
        assert outcome.spawns_new_run is False


class TestContinuationConsequence:
    def test_continuation_feeds_the_run_without_moving_it(self, seeded_db, writer):
        run = make_run(seeded_db, status=S.PENDING_CONFIRMATION)
        outcome = apply_consequence(
            seeded_db, run, _verdict(M.CONTINUATION), writer=writer, message="yes"
        )
        assert outcome.consequence is Consequence.FEED_RUN
        assert run.status is S.PENDING_CONFIRMATION

    def test_continuation_extends_the_request_text(self, seeded_db, writer):
        """Unlike off-topic, a continuation is part of the request — routing
        and slot matching are supposed to read it."""
        run = make_run(seeded_db, text="I need a cardiology appointment")
        apply_consequence(
            seeded_db, run, _verdict(M.CONTINUATION), writer=writer, message="Tuesday works"
        )
        assert "Tuesday works" in run.request_text


class TestTerminalRunsAreNotMappable:
    @pytest.mark.parametrize("status", [S.ESCALATED, S.COMPLETED, S.CANCELLED])
    def test_a_terminal_run_refuses_every_consequence(self, seeded_db, writer, status):
        """Mapping applies to the *active* run. A terminal run has none of the
        state these consequences assume, and one of them must never be
        reopened at all."""
        run = make_run(seeded_db, status=status)
        with pytest.raises(ValidationFailed):
            apply_consequence(
                seeded_db, run, _verdict(M.CONTINUATION), writer=writer, message="hello"
            )

    def test_withdrawal_cannot_close_an_escalated_run(self, seeded_db, writer):
        """"Actually forget it" after a chest-pain trigger stays in front of
        humans. This is the one state withdrawal must not close."""
        run = make_run(seeded_db, status=S.ESCALATED)
        with pytest.raises(ValidationFailed):
            apply_consequence(
                seeded_db, run, _verdict(M.WITHDRAWAL), writer=writer, message="forget it"
            )
        assert run.status is S.ESCALATED


def _verdict(message_class: MessageClass):
    from app.workflow.mapping import ClassVerdict

    return ClassVerdict(
        message_class=message_class, proposed=message_class, adjusted=False, reason=""
    )
