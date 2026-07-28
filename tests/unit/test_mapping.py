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
from app.workflow.mapping import (
    CLASSIFICATION_ORDER,
    Consequence,
    apply_consequence,
    mentions_domain_subject,
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
            and event.payload.get("what") == "supersede_at_review"
        ]
        assert refusals and refusals[0]["accepted"] is False

    def test_the_same_message_at_in_progress_still_supersedes(
        self, seeded_db, writer
    ):
        """Distrust green: the guard is scoped to `pending_review`. A version
        that blocked every same-intent supersede everywhere would pass all of
        the above and quietly break the rephrase path."""
        run = make_run(seeded_db, status=S.IN_PROGRESS)
        outcome = apply_consequence(
            seeded_db,
            run,
            _verdict(M.CONFLICTING),
            writer=writer,
            message="looks good. lets book that time",
            incoming_steps=[PlanStep.BOOK],
        )

        assert outcome.consequence is Consequence.SUPERSEDE


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
