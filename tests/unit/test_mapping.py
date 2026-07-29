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
    Department,
    MessageClass,
    PlanStep,
    ProposedAction,
    TraceEvent,
    TraceEventType,
    WorkflowRun,
    WorkflowStatus,
)
from app.trace import TraceWriter
from app.tools.dates import WEEKDAYS
from app.workflow.mapping import (
    CLASSIFICATION_ORDER,
    Consequence,
    apply_consequence,
    mentions_domain_subject,
    names_appointment_verbs,
    names_timing,
    primary_intent,
    says_withdrawal,
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


class TestADifferentAppointmentVerbIsAlwaysConflicting:
    """Item 5. Booking, moving and cancelling are three things one run cannot
    be doing at once — ``validate_plan`` refuses a plan naming two of them —
    so a message asking for one while the run pursues another is a different
    request whatever the model called it.

    The class the live model chose was **continuation**, which is why this is
    checked for continuation and not only for complementary: guarding the class
    the model was least likely to pick left the one it actually picked
    unguarded, and the reschedule was fed into a booking run that was waiting
    for staff.
    """

    def test_a_reschedule_during_a_booking_run_is_downgraded(self, seeded_db, writer):
        """The live message, the live class, the live run shape."""
        run = make_run(seeded_db, status=S.PENDING_REVIEW, plan=("route", "book"))
        verdict = validate_class(
            "continuation",
            run=run,
            incoming_steps=[PlanStep.RESCHEDULE],
            writer=writer,
        )

        assert verdict.message_class is M.CONFLICTING
        assert verdict.adjusted is True
        assert verdict.proposed is M.CONTINUATION

    def test_it_applies_to_complementary_too(self, seeded_db, writer):
        run = make_run(seeded_db, plan=("route", "book"))
        verdict = validate_class(
            "complementary", run=run, incoming_steps=[PlanStep.CANCEL], writer=writer
        )
        assert verdict.message_class is M.CONFLICTING

    def test_the_same_verb_is_left_alone(self, seeded_db, writer):
        """The rule is *difference*. A continuation carrying the run's own verb
        is exactly what a continuation is, and turning that into a supersede
        would cancel every run its patient talked to."""
        run = make_run(seeded_db, plan=("reschedule", "follow_up"))
        verdict = validate_class(
            "continuation",
            run=run,
            incoming_steps=[PlanStep.RESCHEDULE],
            writer=writer,
        )
        assert verdict.message_class is M.CONTINUATION
        assert verdict.adjusted is False

    def test_a_document_upload_during_a_booking_is_untouched(self, seeded_db, writer):
        """Narrow on purpose: *both* intents must be appointment verbs. "Here's
        my old ECG" during a booking is `documents` against `book`, and stays
        as cooperative as it ever was — that cooperation is the expensive one
        to get wrong."""
        run = make_run(seeded_db, plan=("route", "book"))
        for proposed in ("continuation", "complementary"):
            verdict = validate_class(
                proposed,
                run=run,
                incoming_steps=[PlanStep.DOCUMENTS],
                writer=writer,
            )
            assert verdict.message_class is MessageClass(proposed)

    def test_the_adjustment_is_traced(self, seeded_db, writer):
        run = make_run(seeded_db, plan=("route", "book"))
        validate_class(
            "continuation",
            run=run,
            incoming_steps=[PlanStep.RESCHEDULE],
            writer=writer,
        )
        event = (
            seeded_db.query(TraceEvent)
            .filter(TraceEvent.event_type == TraceEventType.VALIDATION)
            .one()
        )
        assert event.payload["accepted"] is False
        assert event.payload["detail"]["applied"] == "conflicting"
        assert event.payload["detail"]["active_intent"] == "book"
        assert event.payload["detail"]["intent"] == "reschedule"


class TestPrimaryIntent:
    def test_booking_is_the_widest_intent(self):
        assert primary_intent(["route", "book", "documents"]) is PlanStep.BOOK

    def test_a_document_plan_reports_documents(self):
        assert primary_intent(["documents"]) is PlanStep.DOCUMENTS

    def test_routing_alone_reports_routing(self):
        assert primary_intent(["route"]) is PlanStep.ROUTE

    def test_an_empty_plan_has_no_intent(self):
        assert primary_intent([]) is None

    def test_a_reschedule_plan_reports_reschedule_not_its_closure(self):
        """``reschedule`` closes over ``follow_up``, so a ranking that did not
        know the verb called the whole request a follow-up — and "different
        from `book`" was then true for the wrong reason, which is how a
        reschedule got fed into a booking run as a continuation."""
        assert primary_intent(["reschedule", "follow_up"]) is PlanStep.RESCHEDULE

    def test_a_cancel_plan_reports_cancel(self):
        assert primary_intent(["cancel"]) is PlanStep.CANCEL


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


class TestConflictingAtReviewRequiresDifference:
    """A run waiting on staff cannot be destroyed by agreement.

    Found by repro rather than in the wild: "my kid has ear pain" routes with
    low confidence and queues for review; "looks good. lets book that time"
    carries the run's own intent, so it classifies as conflicting — and
    superseding cancelled the queued review a human was about to look at. The
    patient sees a fresh search start from nothing and never learns their
    request was thrown away.

    The asymmetry is the same one that governs complementary: a wrongly
    superseded review costs a re-ask *and* a staff queue item, while a
    genuinely new request that has to be repeated costs a sentence. So
    difference has to be shown, not assumed.

    **Difference is decided in code, from the Department table.** The message's
    own text is resolved against it — no model input, no new tool argument, and
    no keyword list — so "a different subject" means exactly what routing means
    by it everywhere else.
    """

    def test_the_same_intent_with_no_new_subject_does_not_supersede(
        self, seeded_db, writer
    ):
        run = make_run(seeded_db, status=S.PENDING_REVIEW)
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONFLICTING),
            writer=writer,
            message="looks good. lets book that time",
            incoming_steps=[PlanStep.BOOK],
        )

        assert outcome.consequence is Consequence.STATUS_REPLY
        assert run.status is S.PENDING_REVIEW
        assert outcome.spawns_new_run is False

    def test_the_queued_run_is_untouched(self, seeded_db, writer):
        """Not merely uncancelled: the request text must not pick up the
        assent either, or the staff member reads a request nobody made."""
        run = make_run(seeded_db, status=S.PENDING_REVIEW)
        before = (run.request_text, run.cancellation_reason)

        apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONFLICTING),
            writer=writer,
            message="looks good. lets book that time",
            incoming_steps=[PlanStep.BOOK],
        )

        assert (run.request_text, run.cancellation_reason) == before

    def test_a_different_department_still_supersedes(self, seeded_db, writer):
        """The other direction, and the one that makes the guard a rule rather
        than a blanket refusal. A patient who has changed their mind about what
        they need is making a new request, review or no review."""
        run = make_run(seeded_db, status=S.PENDING_REVIEW, text="book ent")
        ent = seeded_db.query(Department).filter(Department.name == "ENT").one()
        run.state = {"department_name": ent.name, "department_id": ent.id}
        seeded_db.flush()

        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONFLICTING),
            writer=writer,
            message="actually I need a dermatology appointment instead",
            incoming_steps=[PlanStep.BOOK],
        )

        assert outcome.consequence is Consequence.SUPERSEDE
        assert run.status is S.CANCELLED

    def test_a_different_intent_still_supersedes(self, seeded_db, writer):
        """Same subject, different verb: cancelling something is not agreeing
        to book it."""
        run = make_run(seeded_db, status=S.PENDING_REVIEW)
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONFLICTING),
            writer=writer,
            message="cancel my appointment",
            incoming_steps=[PlanStep.CANCEL],
        )

        assert outcome.consequence is Consequence.SUPERSEDE

    def test_withdrawal_is_unaffected(self, seeded_db, writer):
        """The patient can always leave. A guard that protected the run from
        its own owner would be a trap, not a safeguard."""
        run = make_run(seeded_db, status=S.PENDING_REVIEW)
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.WITHDRAWAL),
            writer=writer,
            message="never mind, forget it",
        )

        assert outcome.consequence is Consequence.WITHDRAW
        assert run.status is S.CANCELLED

    def test_the_refusal_is_traced(self, seeded_db, writer):
        """A supersede that did not happen leaves no other mark. Without the
        event, a reviewer cannot tell this run from one nobody wrote to."""
        run = make_run(seeded_db, status=S.PENDING_REVIEW)
        apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONFLICTING),
            writer=writer,
            message="looks good. lets book that time",
            incoming_steps=[PlanStep.BOOK],
        )
        seeded_db.flush()

        refusals = [
            event.payload
            for event in seeded_db.query(TraceEvent).all()
            if event.event_type is TraceEventType.VALIDATION
            and event.payload.get("what") == "supersede_refused"
        ]
        assert refusals and refusals[0]["accepted"] is False
        assert refusals[0]["detail"]["state"] == S.PENDING_REVIEW.value


class TestTheSameRefusalAtTheLiveStates:
    """Round 5 item 2. The rule was always about the *message*, not the state.

    It shipped scoped to ``pending_review`` because that is where it was found,
    and the scoping was pinned as a falsification: a version that blocked every
    same-intent supersede everywhere would have passed every test above. One
    transcript later, the wider case is the one doing damage — "can you give me
    slots for next week?", asked during an Orthopedics proposal, superseded the
    booking it was asking about — so the scoping is gone and what varies with
    the state is only what the refusal *becomes*.

    The falsification moves with it. Two counterexamples below hold the rule to
    a rule: a different department and a different verb still supersede at
    ``in_progress``, which is the rephrase path the original scoping protected.
    """

    def test_a_timing_question_during_a_proposal_refines_rather_than_replaces(
        self, seeded_db, writer
    ):
        run = make_run(seeded_db, status=S.PENDING_CONFIRMATION)
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONFLICTING),
            writer=writer,
            message="can you give me slots for next week?",
            incoming_steps=[PlanStep.BOOK],
        )

        assert outcome.consequence is Consequence.REFINE
        assert outcome.spawns_new_run is False
        assert run.status is S.PENDING_CONFIRMATION

    def test_the_same_message_at_in_progress_also_refines(self, seeded_db, writer):
        run = make_run(seeded_db, status=S.IN_PROGRESS)
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONFLICTING),
            writer=writer,
            message="do you have any more appointment for the next week?",
            incoming_steps=[PlanStep.BOOK],
        )

        assert outcome.consequence is Consequence.REFINE
        assert run.status is S.IN_PROGRESS

    def test_a_refinement_writes_nothing(self, seeded_db, writer):
        """A refused supersede that then edited the request text would be the
        write the refusal exists to prevent, arriving by another door."""
        run = make_run(seeded_db, status=S.PENDING_CONFIRMATION)
        before = (run.request_text, run.plan, run.cancellation_reason)

        apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONFLICTING),
            writer=writer,
            message="can you give me slots for next week?",
            incoming_steps=[PlanStep.BOOK],
        )

        assert (run.request_text, run.plan, run.cancellation_reason) == before

    def test_a_different_department_still_supersedes_at_in_progress(
        self, seeded_db, writer
    ):
        """The pinned counterexample. "Instead" means a new request, and a
        refinement rule that swallowed it would have broken the one path this
        whole guard is scoped around."""
        run = make_run(seeded_db, status=S.IN_PROGRESS)
        cardiology = (
            seeded_db.query(Department).filter(Department.name == "Cardiology").one()
        )
        run.state = {"department_name": cardiology.name, "department_id": cardiology.id}
        seeded_db.flush()

        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONFLICTING),
            writer=writer,
            message="book me a dermatology appointment instead",
            incoming_steps=[PlanStep.BOOK],
        )

        assert outcome.consequence is Consequence.SUPERSEDE
        assert run.status is S.CANCELLED

    def test_a_different_verb_still_supersedes_at_in_progress(self, seeded_db, writer):
        run = make_run(seeded_db, status=S.IN_PROGRESS)
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONFLICTING),
            writer=writer,
            message="actually cancel my appointment instead",
            incoming_steps=[PlanStep.CANCEL],
        )

        assert outcome.consequence is Consequence.SUPERSEDE

    def test_the_refusal_names_the_state_it_happened_in(self, seeded_db, writer):
        """One rule, three states — so the trace has to say which one, or a
        reviewer cannot tell a status reply from a refinement afterwards."""
        run = make_run(seeded_db, status=S.PENDING_CONFIRMATION)
        apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONFLICTING),
            writer=writer,
            message="can you give me slots for next week?",
            incoming_steps=[PlanStep.BOOK],
        )
        seeded_db.flush()

        refusals = [
            event.payload
            for event in seeded_db.query(TraceEvent).all()
            if event.event_type is TraceEventType.VALIDATION
            and event.payload.get("what") == "supersede_refused"
        ]
        assert refusals
        assert refusals[0]["detail"]["state"] == S.PENDING_CONFIRMATION.value


class TestTheVerbAMessageStates:
    """``names_change_verb``: a fact about the words, deciding nothing alone.

    It exists because the plan for "please reschedule my appointment to next
    week" was a lottery — three live replays of two phrasings produced a
    correct plan, a ``[route, book]`` plan, and no plan at all. The dangerous
    direction is a false positive: reading a *booking* as a change would send a
    patient's new request at an appointment they already have. So the verb has
    to be stated, beside an appointment noun, and withdrawal wins first.
    """

    @pytest.mark.parametrize(
        "message, step",
        [
            ("please reschedule my appointment to next week", PlanStep.RESCHEDULE),
            ("lets reschedule my appointment", PlanStep.RESCHEDULE),
            ("can we move my appointment to Friday", PlanStep.RESCHEDULE),
            ("I need to postpone my visit", PlanStep.RESCHEDULE),
            ("i want to cancel my upcoming appointment", PlanStep.CANCEL),
            ("I want to cancel my appointment", PlanStep.CANCEL),
        ],
    )
    def test_a_stated_verb_is_read(self, message, step):
        from app.workflow.mapping import names_change_verb

        assert names_change_verb(message) is step

    @pytest.mark.parametrize(
        "message",
        [
            # A withdrawal, not an appointment verb — the collision that leaves
            # an appointment standing while the reply says it was dealt with.
            "cancel that request",
            "actually cancel my request",
            # The bare token that declines a proposal.
            "cancel",
            # A booking, and the one phrasing that must keep superseding.
            "book me a dermatology appointment instead",
            "I need a cardiology appointment next week",
            # A question about times, which changes nothing by itself.
            "can you give me slots for next week?",
            "show my appointments",
            # No appointment noun anywhere.
            "please move my phone number to the new one",
            "what's the weather like today?",
        ],
    )
    def test_anything_less_is_not_a_stated_verb(self, message):
        from app.workflow.mapping import names_change_verb

        assert names_change_verb(message) is None


class TestDomainNounsVetoTheOffTopicVerdict:
    """Item 6, at the seam where the rule lives.

    Whether a message is about appointments is a fact about the message, not a
    judgement the model gets to make twice differently. Live, "can you tell me
    my appointments" was scope-refused twice while another wording of the same
    question went straight through.

    Both directions are pinned here, and the second matters as much as the
    first: a veto that also swallowed "nvidia stock" would have replaced a
    working refusal with a system that never refuses anything.
    """

    def test_the_live_message_is_not_off_topic(self):
        assert mentions_domain_subject("can you tell me my appointments") is True

    def test_the_named_vocabulary_all_counts(self):
        for message in (
            "where do I upload my document",
            "did I get a reminder for that",
            "what's my booking reference",
            "I want to reschedule",
            "cancel it please",
            "show me my appointments",
        ):
            assert mentions_domain_subject(message) is True, message

    def test_genuinely_off_topic_messages_stay_off_topic(self):
        """Byte-identical behaviour for the messages the gate exists for."""
        for message in (
            "what's the weather like today?",
            "how is nvidia stock doing",
            "who won the fifa world cup",
            "tell me a joke",
            "what do you think of my haircut",
        ):
            assert mentions_domain_subject(message) is False, message

    def test_it_matches_at_word_starts_not_anywhere(self):
        """The `erm`-inside-`dermatology` trap, avoided by construction. A
        substring match would make any word ending in one of these a domain
        subject."""
        assert mentions_domain_subject("discancellation") is False
        assert mentions_domain_subject("a cancellation") is True

    def test_empty_input_is_not_a_subject(self):
        assert mentions_domain_subject("") is False
        assert mentions_domain_subject(None) is False

    def test_the_consequence_is_downgraded_not_the_class(self, seeded_db, writer):
        """The trace has to keep describing the message that arrived. The class
        stays `off_topic` — it is what the model said — and only what it may do
        changes, to the read-only class that touches exactly as little."""
        run = make_run(seeded_db, text="I need a cardiology appointment")
        before_text, before_state = run.request_text, dict(run.state or {})

        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.OFF_TOPIC),
            writer=writer,
            message="can you tell me my appointments",
        )

        assert outcome.message_class is M.OFF_TOPIC
        assert outcome.consequence is Consequence.ANSWER_AND_STAY
        assert outcome.spawns_new_run is False
        assert run.request_text == before_text
        assert dict(run.state or {}) == before_state

    def test_a_real_off_topic_message_still_gets_the_scope_reply(
        self, seeded_db, writer
    ):
        run = make_run(seeded_db, text="I need a cardiology appointment")
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.OFF_TOPIC),
            writer=writer,
            message="how is nvidia stock doing",
        )
        assert outcome.consequence is Consequence.SCOPE_REPLY

    def test_the_veto_is_traced(self, seeded_db, writer):
        run = make_run(seeded_db)
        apply_consequence(
            seeded_db,
            run,
            _verdict(M.OFF_TOPIC),
            writer=writer,
            message="show me my appointments",
        )
        events = [
            event
            for event in seeded_db.query(TraceEvent).all()
            if event.event_type is TraceEventType.VALIDATION
            and event.payload["what"] == "off_topic_vetoed"
        ]
        assert len(events) == 1


class TestWithdrawalNeedsTheWordsForIt:
    """Item 1(a). The one consequence that destroys the patient's work.

    Live: the agent asked "which appointment, 1 or 2?", the patient answered
    **"2"**, and gpt-4o-mini classified that as a withdrawal. Validation had
    nothing to object to — `withdrawal` is a real member of the enum — so the
    run was cancelled and the patient was told "I've closed that request",
    which is the opposite of what they had just said.

    The cost asymmetry picks the direction: a wrong withdrawal destroys a
    request and blames the patient for it; a wrong continuation costs one more
    message. So a withdrawal is applied only when the message says so.
    """

    def test_a_bare_number_does_not_withdraw(self, seeded_db, writer):
        run = make_run(seeded_db, status=S.IN_PROGRESS)
        outcome = apply_consequence(
            seeded_db, run, _verdict(M.WITHDRAWAL), writer=writer, message="2"
        )

        assert outcome.consequence is Consequence.FEED_RUN
        assert run.status is S.IN_PROGRESS
        assert run.cancellation_reason is None

    def test_the_class_is_still_recorded_as_withdrawal(self, seeded_db, writer):
        """The trace has to keep describing the message that arrived. The model
        really did say withdrawal; only what it may do changed."""
        run = make_run(seeded_db, status=S.IN_PROGRESS)
        outcome = apply_consequence(
            seeded_db, run, _verdict(M.WITHDRAWAL), writer=writer, message="2"
        )
        assert outcome.message_class is M.WITHDRAWAL

    def test_a_real_withdrawal_still_withdraws(self, seeded_db, writer):
        """The direction that must not move. Every phrasing the cue list owns
        has to keep working, or this guard has traded one broken path for
        another."""
        for phrase in (
            "never mind",
            "actually never mind, forget it",
            "forget it",
            "don't bother",
            "I no longer need this",
            "cancel that request",
            "I've changed my mind",
        ):
            run = make_run(seeded_db, status=S.IN_PROGRESS)
            outcome = apply_consequence(
                seeded_db, run, _verdict(M.WITHDRAWAL), writer=writer, message=phrase
            )
            assert outcome.consequence is Consequence.WITHDRAW, phrase
            assert run.status is S.CANCELLED, phrase

    def test_the_downgrade_is_traced(self, seeded_db, writer):
        run = make_run(seeded_db, status=S.IN_PROGRESS)
        apply_consequence(
            seeded_db, run, _verdict(M.WITHDRAWAL), writer=writer, message="2"
        )
        events = [
            event
            for event in seeded_db.query(TraceEvent).all()
            if event.event_type is TraceEventType.VALIDATION
            and event.payload["what"] == "withdrawal_cue"
        ]
        assert len(events) == 1
        assert events[0].payload["accepted"] is False

    def test_says_withdrawal_reads_the_words_not_the_intent(self):
        assert says_withdrawal("actually never mind") is True
        assert says_withdrawal("2") is False
        assert says_withdrawal("the second one") is False
        assert says_withdrawal("") is False
        assert says_withdrawal(None) is False


class TestARunWaitingForAChoiceKeepsTheAnswer:
    """Item 1, the other half of the same failure.

    Two providers got this wrong in two different ways: gpt-4o-mini called "2"
    a withdrawal, the mock calls it off-topic. Both are the run refusing to
    hear an answer to the question it just asked — and "2" carries no
    administrative noun only because the question already supplied one, which
    is the reasoning the confirmation state has always used for "hmm, maybe".
    """

    def _listing_run(self, session):
        run = make_run(session, status=S.IN_PROGRESS, plan=("cancel",))
        run.state = {"listed_appointment_ids": [1, 2]}
        session.flush()
        return run

    def test_off_topic_becomes_a_continuation(self, seeded_db, writer):
        run = self._listing_run(seeded_db)
        outcome = apply_consequence(
            seeded_db, run, _verdict(M.OFF_TOPIC), writer=writer, message="2"
        )
        assert outcome.consequence is Consequence.FEED_RUN

    def test_withdrawal_becomes_a_continuation(self, seeded_db, writer):
        run = self._listing_run(seeded_db)
        outcome = apply_consequence(
            seeded_db, run, _verdict(M.WITHDRAWAL), writer=writer, message="2"
        )
        assert outcome.consequence is Consequence.FEED_RUN
        assert run.status is S.IN_PROGRESS

    def test_a_genuine_withdrawal_still_lands_while_choosing(self, seeded_db, writer):
        """A patient may abandon a request mid-choice, and saying so must
        work. The guard is about answers being heard, not about trapping
        anyone in a listing."""
        run = self._listing_run(seeded_db)
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.WITHDRAWAL),
            writer=writer,
            message="actually never mind, forget the whole thing",
        )
        assert outcome.consequence is Consequence.WITHDRAW
        assert run.status is S.CANCELLED

    def test_a_run_with_no_listing_is_unaffected(self, seeded_db, writer):
        """Scoping control: off-topic during an ordinary run still gets the
        scope reply, so this has not quietly disabled the guard."""
        run = make_run(seeded_db, status=S.IN_PROGRESS)
        outcome = apply_consequence(
            seeded_db, run, _verdict(M.OFF_TOPIC), writer=writer, message="nvidia stock"
        )
        assert outcome.consequence is Consequence.SCOPE_REPLY

    def test_a_listing_already_answered_is_not_still_waiting(self, seeded_db, writer):
        """Once something is proposed, the choice is made. An off-topic message
        after that is off-topic again."""
        run = self._listing_run(seeded_db)
        run.proposed_action = ProposedAction.CANCEL
        run.proposed_appointment_id = 1
        seeded_db.flush()
        outcome = apply_consequence(
            seeded_db, run, _verdict(M.OFF_TOPIC), writer=writer, message="nvidia stock"
        )
        assert outcome.consequence is Consequence.SCOPE_REPLY


class TestAContinuationThatNamesAnotherDepartment:
    """The other door into the subject check, and the one the patient found.

    Round 5 taught the supersede path to re-resolve the subject, which was
    correct and reachable only by messages the model called *conflicting*. Live,
    a General Medicine proposal was declined and the patient wrote "let me
    clarify — appointment for vision issues"; the model called it a
    **continuation** with no steps of its own. Nothing re-resolved, the routing
    step stayed complete, and the run offered General Medicine slots for an eye
    request — and every reworded clarification did the same thing again, because
    each one was another continuation.

    Reproduced under the mock before it was fixed, which is why the class and
    the empty ``incoming_steps`` are stated as facts here rather than guessed:
    a version of this rule conditioned on the incoming intent matching the
    run's would have been inert, because there was no incoming intent to match.
    """

    def _routed(self, session, name="General Medicine", status=S.IN_PROGRESS):
        run = make_run(session, status=status)
        department = (
            session.query(Department).filter(Department.name == name).one()
        )
        run.state = {"department_id": department.id, "department_name": department.name}
        session.flush()
        return run

    def test_naming_a_different_department_supersedes(self, seeded_db, writer):
        run = self._routed(seeded_db)
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONTINUATION),
            writer=writer,
            message="let me clarify - appointment for vision issues",
            incoming_steps=[],
        )

        assert outcome.consequence is Consequence.SUPERSEDE
        assert outcome.spawns_new_run is True
        assert run.status is S.CANCELLED

    def test_a_timing_refinement_stays_in_the_run(self, seeded_db, writer):
        """The pinned negative direction. "Some time next week" names no
        department, so the run it arrived in is still the run it belongs to."""
        run = self._routed(seeded_db)
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONTINUATION),
            writer=writer,
            message="some time next week would suit me better",
            incoming_steps=[],
        )

        assert outcome.consequence is Consequence.FEED_RUN
        assert run.status is S.IN_PROGRESS
        assert "next week" in run.request_text

    def test_naming_the_same_department_again_stays(self, seeded_db, writer):
        """Repeating the department is confidence, not a change of subject."""
        run = self._routed(seeded_db)
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONTINUATION),
            writer=writer,
            message="yes, the general medicine one",
            incoming_steps=[],
        )
        assert outcome.consequence is Consequence.FEED_RUN
        assert run.status is S.IN_PROGRESS

    def test_an_ambiguous_message_does_not_supersede(self, seeded_db, writer):
        """``ambiguous`` is the safety valve. Two desks settle nothing, and a
        supersede on a maybe destroys a request on noise."""
        run = self._routed(seeded_db)
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONTINUATION),
            writer=writer,
            message="my kid has ear pain too",
            incoming_steps=[],
        )
        assert outcome.consequence is Consequence.FEED_RUN
        assert run.status is S.IN_PROGRESS

    def test_a_run_that_has_not_routed_yet_is_never_superseded(
        self, seeded_db, writer
    ):
        """The dangerous direction. A run with no department has nothing to
        differ from, so the patient's first clarification must not destroy it."""
        run = make_run(seeded_db, status=S.IN_PROGRESS)
        assert not (run.state or {}).get("department_id")

        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONTINUATION),
            writer=writer,
            message="it's about my vision",
            incoming_steps=[],
        )
        assert outcome.consequence is Consequence.FEED_RUN
        assert run.status is S.IN_PROGRESS

    def test_a_cooperating_message_keeps_its_own_consequence(self, seeded_db, writer):
        """The most expensive mistake this module can make, pinned against.

        "Also, here's my old ECG" names Cardiology during a General Medicine
        booking and is *helping*. It classifies as complementary, and only
        continuation is upgraded — so it appends a step and the booking lives.
        """
        run = self._routed(seeded_db)
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.COMPLEMENTARY),
            writer=writer,
            message="also, here is my old ecg report",
            incoming_steps=[PlanStep.DOCUMENTS],
        )
        assert outcome.consequence is Consequence.APPEND_STEP
        assert run.status is S.IN_PROGRESS

    def test_both_directions_are_traced(self, seeded_db, writer):
        """A subject that changed and a subject that did not both leave a mark;
        an upgrade that only ever recorded itself when it fired would be
        indistinguishable from one that never ran."""
        changed = self._routed(seeded_db)
        apply_consequence(
            seeded_db,
            changed,
            _verdict(M.CONTINUATION),
            writer=writer,
            message="actually it is about my vision",
            incoming_steps=[],
        )
        same = self._routed(seeded_db)
        apply_consequence(
            seeded_db,
            same,
            _verdict(M.CONTINUATION),
            writer=writer,
            message="the general medicine appointment",
            incoming_steps=[],
        )

        seeded_db.flush()

        subjects = [
            event.payload
            for event in seeded_db.query(TraceEvent).order_by(TraceEvent.seq).all()
            if event.event_type is TraceEventType.VALIDATION
            and event.payload.get("what") == "new_subject"
        ]
        assert [s["accepted"] for s in subjects] == [True, False]
        assert subjects[0]["detail"]["named_department"] == "Ophthalmology"
        assert subjects[1]["detail"]["named_department"] == "General Medicine"


class TestCountingTheVerbsInOneMessage:
    """Round 7, item 6 — what a two-verb message asked for.

    ``validate_plan`` refuses a plan naming two appointment actions, and that
    rule stays. This only counts what the *words* asked for, so the patient can
    be told which half is being done. Same two rules as
    :func:`names_change_verb`, deliberately: withdrawal wins, and a verb needs
    an appointment noun beside it.
    """

    def test_the_live_message_names_two(self):
        assert names_appointment_verbs(
            "okay lets cancel that appointment and book a new one for skin rash"
        ) == {PlanStep.CANCEL, PlanStep.BOOK}

    def test_one_verb_is_one_verb(self):
        assert names_appointment_verbs("please cancel my appointment") == {PlanStep.CANCEL}

    def test_reschedule_does_not_read_as_a_booking(self):
        """"schedule" is inside "reschedule", and a rule that counted it would
        make every reschedule a two-verb message."""
        assert names_appointment_verbs("reschedule my appointment") == {
            PlanStep.RESCHEDULE
        }

    def test_a_verb_without_an_appointment_noun_is_not_one(self):
        """"cancel" on its own declines a proposal. The same collision
        ``names_change_verb`` already guards."""
        assert names_appointment_verbs("cancel") == set()

    def test_a_withdrawal_names_no_appointment_verb(self):
        """"cancel that request" closes a run; it acts on no appointment."""
        assert names_appointment_verbs("cancel that request") == set()

    def test_a_plain_booking_names_one(self):
        assert names_appointment_verbs("book me an appointment next week") == {
            PlanStep.BOOK
        }


class TestReadingATimingPhrase:
    """Round 7, item 4 — whether a message scopes itself to a day or a date.

    A fact about the message, used only to widen what a turn *answers*. Its
    false positives cost a slot list where a re-ask would have gone, which is
    why the word list can afford to be generous — the opposite trade from the
    safety screen's.
    """

    def test_a_weekday(self):
        assert names_timing("thursday or friday preferably in the afternoon?")

    def test_a_part_of_day(self):
        assert names_timing("anything in the morning?")

    def test_a_month(self):
        assert names_timing("something in august please")

    def test_a_relative_week(self):
        assert names_timing("do you have anything next week")

    def test_a_message_with_no_timing_in_it(self):
        assert not names_timing("hmm let me think about it")

    def test_a_bare_answer(self):
        assert not names_timing("yes")

    def test_the_words_come_from_resolve_dates_own_tables(self):
        """So the two cannot drift: anything read here as a timing phrase is a
        phrase ``resolve_date`` can turn into a window."""
        for weekday in WEEKDAYS:
            assert names_timing(f"how about {weekday}?"), weekday


class TestSayingWhenCodeDisagreed:
    """``overruled`` — one flag over the five places code changes the verdict.

    It exists because a rejected classification used to keep its prose. The
    model classified "3" — a patient choosing the third option — as a
    withdrawal and wrote "it seems you've decided to withdraw your request
    again" to match. The cue guard refused the class, so nothing was withdrawn,
    and the sentence went out anyway, above a re-ask offering the very time
    they had picked. The action was blocked; only the words got through.

    The reply cannot ask five separate questions about what happened to the
    verdict, so the mapping answers one: was the model's account of this
    message the one that was acted on?
    """

    def test_an_ordinary_class_is_not_overruled(self, seeded_db, writer):
        run = make_run(seeded_db, status=S.PENDING_CONFIRMATION)
        outcome = apply_consequence(
            seeded_db, run, _verdict(M.SIDE_QUESTION), writer=writer,
            message="what else is free that week?",
        )
        assert outcome.overruled is False

    def test_a_withdrawal_with_no_cue_is_overruled(self, seeded_db, writer):
        run = make_run(seeded_db, status=S.PENDING_CONFIRMATION)
        outcome = apply_consequence(
            seeded_db, run, _verdict(M.WITHDRAWAL), writer=writer, message="3"
        )
        assert outcome.consequence is Consequence.FEED_RUN
        assert outcome.overruled is True

    def test_a_withdrawal_the_patient_actually_asked_for_is_not(self, seeded_db, writer):
        """The narrowness that matters: a real withdrawal is applied as it
        always was, and its reply is a template either way."""
        run = make_run(seeded_db)
        outcome = apply_consequence(
            seeded_db, run, _verdict(M.WITHDRAWAL), writer=writer,
            message="never mind, forget it",
        )
        assert outcome.consequence is Consequence.WITHDRAW
        assert outcome.overruled is False

    def test_an_adjusted_class_is_overruled_too(self, seeded_db, writer):
        """``validate_class`` changes the class before this function sees it,
        and the prose was written for the class the model proposed."""
        from app.workflow.mapping import ClassVerdict

        run = make_run(seeded_db, plan=("route", "book"))
        adjusted = ClassVerdict(
            message_class=M.CONFLICTING, proposed=M.COMPLEMENTARY, adjusted=True,
            reason="same intent as the active run (book)",
        )
        outcome = apply_consequence(
            seeded_db, run, adjusted, writer=writer,
            message="actually make it an orthopedics appointment",
        )
        assert outcome.overruled is True
