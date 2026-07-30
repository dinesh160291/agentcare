"""The rest of confirm-before-commit: buttons, the model's half, and the stall.

The exact-token reader shipped in Phase 4. What is pinned here is everything
around it, and the reading order it belongs to:

  (a) ✅ Confirm / ❌ Decline buttons — a typed action, zero interpretation;
  (b) exact-token match on typed text, in code;
  (c) only what the tokens cannot read reaches the model, whose **only**
      permitted verdicts are decline and non-answer.

The one rule everything here serves: **a wrongly re-asked "yes" costs one tap;
a wrongly committed "no" books an appointment against the patient's word at the
exact step built to prevent that.** So the tests below spend most of their
effort trying to get a booking committed by something other than the patient,
and expect to fail every time.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest
from google.adk.models import LlmRequest, LlmResponse

from app.config import get_settings
from app.db import SessionLocal
from app.errors import ValidationFailed
from app.models import (
    Appointment,
    AppointmentStatus,
    AuditEvent,
    ProposedAction,
    TraceAuthor,
    TraceEvent,
    TraceEventType,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.orchestrator import (
    DECLINED_REPLY,
    NOTHING_BOOKED_TO_CANCEL_REPLY,
    NOTHING_TO_CONFIRM_REPLY,
    apply_patient_action,
    run_workflow,
)
from app.providers.base import (
    AgentCareLlm,
    available_tool_names,
    called_tools,
    function_call_response,
    text_response,
)
from app.trace import assert_well_formed
from app.workflow.replies import render_reask

#: What a model writes once it has decided the patient said no. Module-level
#: rather than a class attribute because ``AgentCareLlm`` is a pydantic model
#: and an unannotated attribute there is a field.
DECLINING_PROSE = "That's fine — I won't book anything."

PATIENT_EMAIL = "asha.patient@example.invalid"
BOOKING = "I need a cardiology appointment next week"
SEEDED_APPOINTMENT_ID = 1


@pytest.fixture
def patient(seeded_db):
    seeded_db.commit()
    return seeded_db.query(User).filter(User.email == PATIENT_EMAIL).one()


def turn(user, message, session_id):
    return asyncio.run(run_workflow(user, message, session_id))


def press(user, action, session_id):
    return asyncio.run(apply_patient_action(user, action, session_id))


def fresh():
    return SessionLocal()


def booked_for(session, run_id: int) -> list[Appointment]:
    run = session.get(WorkflowRun, run_id)
    return (
        session.query(Appointment)
        .filter(Appointment.patient_id == run.patient_id)
        .filter(Appointment.id != SEEDED_APPOINTMENT_ID)
        .all()
    )


def _held_slot(run_id: int) -> int | None:
    session = fresh()
    try:
        return session.get(WorkflowRun, run_id).proposed_slot_id
    finally:
        session.close()


def _validation(session, turn_id, what):
    for event in (
        session.query(TraceEvent)
        .filter(TraceEvent.turn_id == turn_id)
        .order_by(TraceEvent.seq)
        .all()
    ):
        if event.event_type is TraceEventType.VALIDATION:
            if (event.payload or {}).get("what") == what:
                return event.payload
    raise AssertionError(f"no {what!r} validation in turn {turn_id}")


def _guard(session, turn_id, name):
    for event in (
        session.query(TraceEvent)
        .filter(TraceEvent.turn_id == turn_id)
        .order_by(TraceEvent.seq)
        .all()
    ):
        if event.event_type is TraceEventType.GUARD_VERDICT:
            if (event.payload or {}).get("guard") == name:
                return event.payload
    raise AssertionError(f"no {name!r} guard verdict in turn {turn_id}")


def _guard_or_none(session, turn_id, name):
    """The same, for the tests whose claim is that a guard did *not* run."""
    try:
        return _guard(session, turn_id, name)
    except AssertionError:
        return None


class TestTheConfirmButton:
    """A click arrives as a typed action. There is nothing to interpret, so
    nothing interprets it."""

    def test_pressing_confirm_books_the_appointment(self, patient):
        turn(patient, BOOKING, "s-btn-1")
        result = press(patient, "confirm", "s-btn-1")

        session = fresh()
        try:
            booked = booked_for(session, result.run_id)
            assert len(booked) == 1
            assert booked[0].status is AppointmentStatus.CONFIRMED
        finally:
            session.close()

    def test_the_click_is_an_inbound_event_of_its_own_kind(self, patient):
        """The system has two front doors and the grammar recognises both: an
        inbound event is a chat message **or** a typed action."""
        turn(patient, BOOKING, "s-btn-2")
        result = press(patient, "confirm", "s-btn-2")

        session = fresh()
        try:
            inbound = [
                e
                for e in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if e.event_type is TraceEventType.INBOUND
            ]
        finally:
            session.close()

        assert len(inbound) == 1
        assert inbound[0].author is TraceAuthor.PATIENT_ACTION

    def test_the_turn_is_bracketed_like_any_other(self, patient):
        turn(patient, BOOKING, "s-btn-3")
        result = press(patient, "confirm", "s-btn-3")

        session = fresh()
        try:
            kinds = [
                e.event_type
                for e in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .order_by(TraceEvent.seq)
                .all()
            ]
        finally:
            session.close()

        assert kinds[0] is TraceEventType.INBOUND
        assert kinds[-1] is TraceEventType.OUTBOUND

    def test_pressing_decline_clears_the_proposal(self, patient):
        turn(patient, BOOKING, "s-btn-4")
        result = press(patient, "decline", "s-btn-4")

        assert result.reply == DECLINED_REPLY
        assert result.status == WorkflowStatus.IN_PROGRESS.value

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.proposed_slot_id is None
            assert booked_for(session, result.run_id) == []
        finally:
            session.close()

    def test_a_second_confirm_click_is_a_calm_no_op(self, patient):
        """The motivating trace is a double-clicked button. The second request
        finds the proposal already committed and must say so, not crash and not
        book twice."""
        first = turn(patient, BOOKING, "s-btn-5")
        press(patient, "confirm", "s-btn-5")
        again = press(patient, "confirm", "s-btn-5")

        assert again.reply == NOTHING_TO_CONFIRM_REPLY

        session = fresh()
        try:
            assert len(booked_for(session, first.run_id)) == 1
        finally:
            session.close()

    def test_a_stale_click_is_audited(self, patient):
        press(patient, "confirm", "s-btn-6")

        session = fresh()
        try:
            assert (
                session.query(AuditEvent)
                .filter(AuditEvent.action == "patient_action_stale")
                .count()
                == 1
            )
        finally:
            session.close()

    def test_an_unknown_action_is_refused(self, patient):
        turn(patient, BOOKING, "s-btn-7")

        with pytest.raises(ValidationFailed):
            press(patient, "book it immediately", "s-btn-7")

    def test_the_typed_action_turn_parses_against_the_grammar(self, patient):
        """The checker's docstring has always claimed typed-action turns follow
        the same rules. Until there were typed actions, nothing exercised the
        claim — and a checker that only understood chatty turns would pass the
        system's least deterministic flows while ignoring its most."""
        turn(patient, BOOKING, "s-btn-grammar")
        result = press(patient, "confirm", "s-btn-grammar")

        session = fresh()
        try:
            assert_well_formed(session, turn_id=result.turn_id)
        finally:
            session.close()

    def test_a_typed_action_costs_no_model_call(self, patient):
        """Zero interpretation means zero interpretation."""
        turn(patient, BOOKING, "s-btn-8")
        result = press(patient, "decline", "s-btn-8")

        session = fresh()
        try:
            requests = [
                e
                for e in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if e.event_type is TraceEventType.LLM_REQUEST
            ]
        finally:
            session.close()
        assert requests == []


class TestTheModelMayNeverConfirm:
    """(c) of the reading order. The model's two permitted verdicts are
    decline and non-answer; ``confirm`` is not in the enum, so the refusal is
    structural rather than a matter of the prompt holding."""

    def test_a_model_read_decline_returns_to_slot_selection(self, patient):
        """Text the exact tokens cannot read — "not that one, thanks" carries
        no token — still has to be able to mean no."""
        turn(patient, BOOKING, "s-model-1")
        result = turn(patient, "not that one, thanks", "s-model-1")

        assert result.status == WorkflowStatus.IN_PROGRESS.value

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.proposed_slot_id is None
            assert booked_for(session, result.run_id) == []
        finally:
            session.close()

    def test_a_confirm_verdict_is_rejected_by_code(self, patient, monkeypatch):
        """The adversarial case: a provider that tries to confirm on the
        patient's behalf. It must be told no, and nothing must be booked."""
        first = turn(patient, BOOKING, "s-model-2")

        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: EagerConfirmer()
        )
        result = turn(patient, "hmm, I suppose so, maybe?", "s-model-2")

        session = fresh()
        try:
            assert booked_for(session, first.run_id) == []
            run = session.get(WorkflowRun, first.run_id)
            assert run.status is WorkflowStatus.PENDING_CONFIRMATION
            rejections = [
                e.payload
                for e in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if e.event_type is TraceEventType.VALIDATION
                and e.payload["what"] == "confirmation_verdict"
            ]
        finally:
            session.close()

        assert rejections, "the attempt must be recorded, not merely ignored"
        assert all(r["accepted"] is False for r in rejections)


class TestADeclineNeedsADeclineCue:
    """The other half of "the model may never confirm".

    A decline is not the harmless verdict it looks like beside ``confirm``: it
    clears a held proposal, which is the patient's decision thrown away. Live,
    at ``pending_confirmation`` holding a 2:00 PM slot, "yes lets confirm it"
    came back as ``decline`` — the affirmative sentence quoted in the verdict's
    own ``reason`` — and the reschedule died silently.

    Every test here drives :class:`AlwaysDeclines`, which submits that verdict
    whatever the patient said. That is the point: the guard's whole job is to be
    the thing standing between a wrong verdict and the row it would clear, so
    the provider has to be wrong on purpose.
    """

    @staticmethod
    def declining(monkeypatch):
        """Swap the provider *after* the setup booking.

        Not a fixture: this stub plans nothing, so a patch applied for the whole
        test would leave the run that is supposed to be holding a slot
        un-created — and every assertion here would then pass or fail for a
        reason that has nothing to do with the guard.
        """
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: AlwaysDeclines()
        )

    @pytest.mark.parametrize(
        "reply",
        ["yes lets confirm it", "yes please, sounds good!", "sure go ahead"],
    )
    def test_an_affirmative_is_never_applied_as_a_decline(
        self, patient, monkeypatch, reply
    ):
        session_id = f"s-cue-{abs(hash(reply))}"
        first = turn(patient, BOOKING, session_id)
        held = _held_slot(first.run_id)

        self.declining(monkeypatch)
        result = turn(patient, reply, session_id)

        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        assert _held_slot(first.run_id) == held

    def test_the_affirmative_lands_on_the_exact_token_re_ask(
        self, patient, monkeypatch
    ):
        """Round 6 built this reply for exactly this case: a yes the tokens
        cannot read is answered by naming what is held and what would settle
        it."""
        first = turn(patient, BOOKING, "s-cue-reask")
        self.declining(monkeypatch)
        result = turn(patient, "yes lets confirm it", "s-cue-reask")

        session = fresh()
        try:
            expected = render_reask(session, session.get(WorkflowRun, first.run_id))
        finally:
            session.close()

        assert result.reply == expected
        assert result.author is TraceAuthor.TEMPLATE

    def test_the_declining_models_prose_goes_with_its_verdict(
        self, patient, monkeypatch
    ):
        """An overruled verdict takes its prose with it. The sentence was
        written believing the proposal was about to be cleared, so shipping it
        above a re-ask tells the patient two contradictory things about the same
        slot."""
        turn(patient, BOOKING, "s-cue-prose")
        self.declining(monkeypatch)
        result = turn(patient, "yes lets confirm it", "s-cue-prose")

        assert DECLINING_PROSE not in result.reply

    def test_neither_cue_is_a_re_ask_too(self, patient, monkeypatch):
        """Rule (c). A cleared proposal has to be earned by the patient's own
        words, and "hmm, maybe" is not those words either."""
        first = turn(patient, BOOKING, "s-cue-neither")
        held = _held_slot(first.run_id)

        self.declining(monkeypatch)
        result = turn(patient, "hmm, maybe", "s-cue-neither")

        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        assert _held_slot(first.run_id) == held

    def test_both_cues_is_a_re_ask(self, patient, monkeypatch):
        first = turn(patient, BOOKING, "s-cue-both")
        held = _held_slot(first.run_id)

        self.declining(monkeypatch)
        turn(patient, "actually no wait yes", "s-cue-both")

        assert _held_slot(first.run_id) == held

    @pytest.mark.parametrize(
        "reply", ["no thanks", "a different day would be better", "I'd rather not"]
    )
    def test_a_real_decline_still_declines(self, patient, monkeypatch, reply):
        """The direction that must not break. The guard is a necessary
        condition on the model's verdict, not a second opinion about it."""
        session_id = f"s-cue-real-{abs(hash(reply))}"
        first = turn(patient, BOOKING, session_id)

        self.declining(monkeypatch)
        result = turn(patient, reply, session_id)

        assert result.reply == DECLINED_REPLY
        assert _held_slot(first.run_id) is None

    def test_the_refusal_is_traced(self, patient, monkeypatch):
        """A refused decline leaves no other mark: the run is where it was and
        the reply is the one a non-answer would have got."""
        turn(patient, BOOKING, "s-cue-trace")
        self.declining(monkeypatch)
        result = turn(patient, "yes lets confirm it", "s-cue-trace")

        session = fresh()
        try:
            recorded = _validation(session, result.turn_id, "decline_cue")
        finally:
            session.close()

        assert recorded["accepted"] is False
        assert recorded["detail"]["affirmative_cue"] is True
        assert recorded["detail"]["applied"] == "non_answer"

    def test_an_applied_decline_is_traced_too(self, patient, monkeypatch):
        turn(patient, BOOKING, "s-cue-trace-2")
        self.declining(monkeypatch)
        result = turn(patient, "no thanks", "s-cue-trace-2")

        session = fresh()
        try:
            recorded = _validation(session, result.turn_id, "decline_cue")
        finally:
            session.close()

        assert recorded["accepted"] is True
        assert recorded["detail"]["decline_cue"] is True


RESCHEDULING = "please reschedule my appointment to next week"


class TestCancelIsAVerbNotADecline:
    """Round 11 item 3 — the same request, two phrasings, two outcomes.

    Holding a reschedule proposal for an Ophthalmology appointment, "actually
    just cancel it instead" came back from the model as ``decline`` and was
    applied, because round 10's own work order had put "cancel" in the decline
    vocabulary. The proposal died, the patient was told nothing had been booked,
    and the appointment they had asked to cancel was still on the books. Two
    turns later "actually just cancel **this appointment** instead" superseded
    perfectly.

    The pronoun is the whole difference, and its referent was a column on the
    run this system had just written. So it is read here, and only here.
    """

    def _held_reschedule(self, patient, session_id: str):
        first = turn(patient, RESCHEDULING, session_id)
        assert first.status == WorkflowStatus.PENDING_CONFIRMATION.value
        session = fresh()
        try:
            run = session.get(WorkflowRun, first.run_id)
            assert run.proposed_action is ProposedAction.RESCHEDULE
            target = run.proposed_appointment_id
        finally:
            session.close()
        assert target is not None
        return first, target

    @pytest.mark.parametrize(
        "phrasing",
        [
            "actually just cancel it instead",
            "actually just cancel this appointment instead",
        ],
    )
    def test_either_phrasing_switches_to_a_cancellation(self, patient, phrasing):
        """Both, and the same outcome, by two different roads.

        The noun form has always gone through the model's supersede path, where
        ``resolve_target`` reads the patient's own cues against the rows — and it
        keeps going that way, because a message that names a noun may be pointing
        at a different appointment than the one being held. Only the pronoun form
        is read here. What this pins is that the *outcome* no longer depends on
        which of the two the patient typed, which is the whole item.
        """
        session_id = f"s-switch-{abs(hash(phrasing))}"
        first, target = self._held_reschedule(patient, session_id)

        result = turn(patient, phrasing, session_id)

        assert result.run_id != first.run_id
        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        session = fresh()
        try:
            old = session.get(WorkflowRun, first.run_id)
            new = session.get(WorkflowRun, result.run_id)
            assert old.status is WorkflowStatus.CANCELLED
            assert new.proposed_action is ProposedAction.CANCEL
            assert new.proposed_appointment_id == target
        finally:
            session.close()

    def test_the_exact_yes_then_cancels_that_appointment(self, patient):
        """The cost of the live failure, stated as the thing that now happens:
        the appointment the patient asked to cancel is cancelled."""
        _, target = self._held_reschedule(patient, "s-switch-yes")
        turn(patient, "actually just cancel it instead", "s-switch-yes")

        turn(patient, "yes", "s-switch-yes")

        session = fresh()
        try:
            appointment = session.get(Appointment, target)
            assert appointment.status is AppointmentStatus.CANCELLED
        finally:
            session.close()

    def test_the_new_run_never_asks_which_appointment(self, patient):
        """The referent is carried, not re-derived.

        Two live appointments, so a cancel run reading "it" from scratch would
        find no cue and ask which one — about the appointment the patient has
        just pointed at. The second one is in a different department on purpose:
        that is what lets the reschedule reach a proposal at all, and it is the
        only shape in which this claim is testable.
        """
        turn(patient, "I need a dermatology appointment next week", "s-switch-two")
        assert turn(patient, "yes", "s-switch-two").status == (
            WorkflowStatus.COMPLETED.value
        )

        first = turn(
            patient, "please reschedule my dermatology appointment", "s-switch-two"
        )
        assert first.status == WorkflowStatus.PENDING_CONFIRMATION.value
        session = fresh()
        try:
            target = session.get(WorkflowRun, first.run_id).proposed_appointment_id
        finally:
            session.close()
        assert target is not None

        result = turn(patient, "actually just cancel it instead", "s-switch-two")

        session = fresh()
        try:
            new = session.get(WorkflowRun, result.run_id)
            assert new.state["chosen_appointment_id"] == target
            assert new.proposed_appointment_id == target
        finally:
            session.close()

    def test_the_switch_is_traced(self, patient):
        self._held_reschedule(patient, "s-switch-trace")
        result = turn(patient, "actually just cancel it instead", "s-switch-trace")

        session = fresh()
        try:
            verdict = _guard(session, result.turn_id, "verb_switch")
        finally:
            session.close()

        assert verdict["detail"]["verb"] == "cancel"
        assert verdict["detail"]["held"] == "reschedule"
        assert verdict["detail"]["applied"] == "supersede"


class TestCancellingAnOfferThatDoesNotExistYet:
    """A held *booking* names no appointment, so "cancel it" can only mean the
    offer. The run closes and the reply says both halves — nothing was booked,
    and the request is closed."""

    def test_the_run_closes_and_says_why(self, patient):
        first = turn(patient, BOOKING, "s-switch-book")
        assert first.status == WorkflowStatus.PENDING_CONFIRMATION.value

        result = turn(patient, "cancel it", "s-switch-book")

        assert result.reply == NOTHING_BOOKED_TO_CANCEL_REPLY
        assert result.author is TraceAuthor.TEMPLATE
        session = fresh()
        try:
            run = session.get(WorkflowRun, first.run_id)
            assert run.status is WorkflowStatus.CANCELLED
            assert run.proposed_slot_id is None
        finally:
            session.close()

    def test_nothing_else_is_touched(self, patient):
        """"Nothing else" is the load-bearing half. The patient's existing
        appointment is not what "it" referred to, and a cancellation that
        reached it would be the worst outcome this path could have."""
        session = fresh()
        try:
            before = [
                (row.id, row.status)
                for row in session.query(Appointment).order_by(Appointment.id).all()
            ]
        finally:
            session.close()

        turn(patient, BOOKING, "s-switch-book-2")
        turn(patient, "cancel it", "s-switch-book-2")

        session = fresh()
        try:
            after = [
                (row.id, row.status)
                for row in session.query(Appointment).order_by(Appointment.id).all()
            ]
        finally:
            session.close()

        assert after == before

    def test_a_second_appointment_is_not_the_offer(self, patient):
        """A noun is its own referent, and it may not be this one.

        "Cancel my other appointment", sent over a booking offer, is a request
        about a row that exists — not a request to drop the offer. Closing the run
        there would throw away the booking the patient is in the middle of, so the
        reader stands aside for any message that names an appointment noun.
        """
        turn(patient, BOOKING, "s-switch-other")

        result = turn(
            patient, "also please cancel my other appointment", "s-switch-other"
        )

        assert result.reply != NOTHING_BOOKED_TO_CANCEL_REPLY
        session = fresh()
        try:
            assert _guard_or_none(session, result.turn_id, "verb_switch") is None
        finally:
            session.close()

    def test_a_time_change_at_a_booking_offer_is_not_a_cancellation(self, patient):
        """The expensive misreading, refused. "Move it to Friday" against an
        offer is a request for a different time — the timing machinery owns that
        sentence, and closing the run over it would throw away the request."""
        first = turn(patient, BOOKING, "s-switch-move")

        turn(patient, "can you move it to friday", "s-switch-move")

        session = fresh()
        try:
            run = session.get(WorkflowRun, first.run_id)
            assert run.status is not WorkflowStatus.CANCELLED
        finally:
            session.close()


class TestStallContainment:
    """The re-ask loop is bounded. A stall never resolves implicitly in either
    direction — no auto-commit, no auto-decline."""

    def test_each_non_answer_is_counted(self, patient):
        result = turn(patient, BOOKING, "s-stall-1")
        for _ in range(2):
            turn(patient, "hmm, maybe", "s-stall-1")

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.non_answer_count == 2
        finally:
            session.close()

    def test_at_the_cap_the_re_ask_becomes_code_templated(self, patient):
        """Still exact equality, against what code actually writes now.

        The stalled wording used to be a bare constant. It is the facts-
        carrying re-ask instead — a patient who has been asked three times is
        the one who most needs telling *what* they are being asked about — so
        the pin moved to ``render_reask`` rather than being loosened to a
        substring, which would have stopped noticing whether code or the model
        wrote it at all.
        """
        cap = get_settings().max_confirmation_non_answers
        result = turn(patient, BOOKING, "s-stall-2")

        replies = [turn(patient, "hmm, maybe", "s-stall-2") for _ in range(cap + 1)]

        session = fresh()
        try:
            expected = render_reask(session, session.get(WorkflowRun, result.run_id))
        finally:
            session.close()

        assert replies[-1].reply == expected
        assert replies[-1].author is TraceAuthor.TEMPLATE

    def test_the_templated_re_ask_costs_no_model_wording(self, patient):
        """"Costing zero further LLM turns" is about the re-ask, which is now
        code's. Classification still runs — a withdrawal after four stalls must
        still be heard."""
        cap = get_settings().max_confirmation_non_answers
        turn(patient, BOOKING, "s-stall-3")
        for _ in range(cap):
            turn(patient, "hmm, maybe", "s-stall-3")
        result = turn(patient, "hmm, maybe", "s-stall-3")

        assert result.author is TraceAuthor.TEMPLATE

    def test_a_withdrawal_still_lands_after_the_cap(self, patient):
        """The trap the class-independent counter exists to avoid: bounding the
        loop by skipping the model would also stop the system hearing "forget
        it", leaving a zombie run claiming the patient is still waiting."""
        cap = get_settings().max_confirmation_non_answers
        first = turn(patient, BOOKING, "s-stall-4")
        for _ in range(cap + 1):
            turn(patient, "hmm, maybe", "s-stall-4")
        turn(patient, "actually forget it", "s-stall-4")

        session = fresh()
        try:
            run = session.get(WorkflowRun, first.run_id)
            assert run.status is WorkflowStatus.CANCELLED
            assert run.cancellation_reason == "withdrawn"
        finally:
            session.close()

    def test_an_off_topic_stall_still_counts(self, patient):
        """Counted independently of the class. An "hmm, maybe" read as
        off-topic must not buy an extra free turn — that is how the bound
        becomes reachable only through correctly-classified paths."""
        result = turn(patient, BOOKING, "s-stall-5")
        turn(patient, "what's the weather like?", "s-stall-5")

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert run.non_answer_count == 1
        finally:
            session.close()

    def test_the_stall_never_resolves_itself(self, patient):
        cap = get_settings().max_confirmation_non_answers
        first = turn(patient, BOOKING, "s-stall-6")
        for _ in range(cap + 3):
            turn(patient, "hmm, maybe", "s-stall-6")

        session = fresh()
        try:
            run = session.get(WorkflowRun, first.run_id)
            assert run.status is WorkflowStatus.PENDING_CONFIRMATION
            assert run.proposed_slot_id is not None
            assert booked_for(session, first.run_id) == []
        finally:
            session.close()

    def test_the_counter_is_recorded_as_a_guard_verdict(self, patient):
        turn(patient, BOOKING, "s-stall-7")
        result = turn(patient, "hmm, maybe", "s-stall-7")

        session = fresh()
        try:
            stall = _guard(session, result.turn_id, "confirmation_stall")
        finally:
            session.close()
        assert stall["detail"]["non_answers"] == 1
        assert stall["passed"] is True

    def test_declining_resets_the_counter(self, patient):
        """The count belongs to the proposal. A patient who stalled once should
        not meet the terse framing on their first look at a new time."""
        first = turn(patient, BOOKING, "s-stall-8")
        turn(patient, "hmm, maybe", "s-stall-8")
        press(patient, "decline", "s-stall-8")

        session = fresh()
        try:
            run = session.get(WorkflowRun, first.run_id)
            assert run.non_answer_count == 0
        finally:
            session.close()

    def test_confirming_resets_the_counter(self, patient):
        first = turn(patient, BOOKING, "s-stall-9")
        turn(patient, "hmm, maybe", "s-stall-9")
        turn(patient, "yes", "s-stall-9")

        session = fresh()
        try:
            run = session.get(WorkflowRun, first.run_id)
            assert run.non_answer_count == 0
        finally:
            session.close()


class AlwaysDeclines(AgentCareLlm):
    """A provider that reads every answer as a refusal.

    Faithful to the live failure rather than adversarial for its own sake:
    ``gpt-4o-mini`` submitted ``decline`` for "yes lets confirm it" and quoted
    that sentence back in the ``reason`` field, so the verdict was well-formed,
    in the enum, and about the opposite of what the patient said. Nothing except
    a cue check could have caught it.

    The prose matters as much as the verdict. A model that has just decided the
    patient said no writes a sentence to match, and that sentence must not
    survive a verdict code refused.
    """

    model: str = "always-declines-stub"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        available = available_tool_names(llm_request)
        done = called_tools(llm_request)

        if "submit_safety_verdict" in available:
            if "submit_safety_verdict" in done:
                yield text_response("screened")
            else:
                yield function_call_response(
                    "submit_safety_verdict",
                    {"category": "safe", "rationale": "stub"},
                )
            return

        if "submit_confirmation_verdict" in available:
            if "submit_confirmation_verdict" in done:
                yield text_response(DECLINING_PROSE)
            else:
                yield function_call_response(
                    "submit_confirmation_verdict",
                    {"verdict": "decline", "reason": "they named another time"},
                )
            return

        yield text_response("ok")


class EagerConfirmer(AgentCareLlm):
    """A provider that tries to book on the patient's behalf.

    Not a strawman: "no wait — yes, the Tuesday one" is exactly the input that
    invites a helpful model to decide the patient meant yes. The enum is what
    stops it, and this is how we find out whether the enum is real.
    """

    model: str = "eager-confirmer-stub"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        available = available_tool_names(llm_request)
        done = called_tools(llm_request)

        if "submit_safety_verdict" in available:
            if "submit_safety_verdict" in done:
                yield text_response("screened")
            else:
                yield function_call_response(
                    "submit_safety_verdict",
                    {"category": "safe", "rationale": "stub"},
                )
            return

        if "submit_confirmation_verdict" in available:
            if "submit_confirmation_verdict" in done:
                yield text_response("I've gone ahead and booked that for you.")
            else:
                yield function_call_response(
                    "submit_confirmation_verdict",
                    {"verdict": "confirm", "reason": "they sounded willing"},
                )
            return

        yield text_response("ok")
