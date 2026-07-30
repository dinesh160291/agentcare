"""The A–G acceptance battery, end to end, as the round-10 gate.

Seven sequences the user re-runs live against a real provider. This module runs
the same seven under mock so that a live failure is interpretable: without it,
"the model got it wrong" and "the expectation was never satisfiable" look
identical, and the second is far more common.

**Every turn must land on a specified outcome.** Two steps retain deliberate
model latitude — whether a blurry-eye booking is screened safe, and whether a
genuinely ambiguous symptom routes with confidence — and for those the pins name
**both** permitted outcomes and assert what is true either way. Nothing here is
scored on the model's prose.

**Where a stub appears, it is because the mock's *classifier* differs from the
live model's on a message whose deterministic handling is what the step is
about.** Three cases, and each one is named at its use:

* the review wall needs a message the Coordinator calls a continuation;
* an affirmative-read-as-decline needs a model that submits that verdict;
* the window vocabulary needs the timing questions to arrive as timing
  questions, which live they do and under the mock they sometimes do not.

Forcing a classification is not weakening the test — the classification is the
probabilistic bin, and what these sequences pin is what code does *with* it.
Where the mock already agrees with the live model, no stub is used.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pytest
from google.adk.models import LlmRequest, LlmResponse

from app import clock
from app.db import SessionLocal
from app.tools.dates import resolve_date
from app.models import (
    Appointment,
    AppointmentStatus,
    Escalation,
    EscalationKind,
    TraceAuthor,
    TraceEvent,
    TraceEventType,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.orchestrator import (
    AWAITING_REVIEW_REPLY,
    NO_PLAN_REPLY,
    NOTHING_TO_CONFIRM_REPLY,
    SCOPE_REPLY,
    UNSUPPORTED_TOPIC_REPLY,
    apply_patient_action,
    run_workflow,
)
from app.providers.base import (
    available_tool_names,
    called_tools,
    function_call_response,
)
from app.providers.mock import MockLlm

ASHA = "asha.patient@example.invalid"
SEEDED_APPOINTMENT_ID = 1

#: Every clarify-shaped answer a turn with nothing pending may give. Named as a
#: set rather than one constant because "sensible clarify" is a family: the scope
#: line closes, the no-plan line asks, and the nothing-to-confirm line explains.
#: What matters for the battery is that it is one of these and that no run and no
#: appointment came out of it.
CLARIFIES = {SCOPE_REPLY, NO_PLAN_REPLY, NOTHING_TO_CONFIRM_REPLY, UNSUPPORTED_TOPIC_REPLY}


def day_label(phrase: str) -> str:
    """The first day of the window a phrase resolves to, as a reply writes it.

    Derived rather than hardcoded: the unit suite runs on the real clock, so a
    literal date would make every window assertion here a test of what day it is.
    """
    window = resolve_date(phrase, today=clock.today())
    assert window["resolved"], f"the battery's own phrase did not resolve: {phrase!r}"
    first = date.fromisoformat(window["start"])
    return f"{first:%A} {first.day} {first:%B}"


@pytest.fixture
def patient(seeded_db):
    seeded_db.commit()
    return seeded_db.query(User).filter(User.email == ASHA).one()


def turn(user, message, session_id):
    return asyncio.run(run_workflow(user, message, session_id))


def press(user, action, session_id):
    return asyncio.run(apply_patient_action(user, action, session_id))


def fresh():
    return SessionLocal()


def _appointments(patient_id: int) -> list[Appointment]:
    session = fresh()
    try:
        return (
            session.query(Appointment)
            .filter(Appointment.patient_id == patient_id)
            .order_by(Appointment.id)
            .all()
        )
    finally:
        session.close()


def _run(run_id: int) -> WorkflowRun:
    session = fresh()
    try:
        return session.get(WorkflowRun, run_id)
    finally:
        session.close()


def _patient_id(run_id: int) -> int:
    return _run(run_id).patient_id


def _tools(turn_id: str) -> list[str]:
    session = fresh()
    try:
        return [
            event.payload["tool"]
            for event in session.query(TraceEvent)
            .filter(TraceEvent.turn_id == turn_id)
            .order_by(TraceEvent.seq)
            .all()
            if event.event_type is TraceEventType.TOOL_CALL
        ]
    finally:
        session.close()


# --- the three stubs ------------------------------------------------------


class ContinuationLlm(MockLlm):
    """Everything mid-run is a continuation.

    Run 6's live classification, and the one that made a queued request answer
    with a proposal it could not honour.
    """

    model: str = "battery-continuation-stub"

    def _classify(self, llm_request, available, done, task):  # noqa: ANN001
        return function_call_response(
            "classify_message",
            {"message_class": "continuation", "incoming_steps": []},
        )


class SideQuestionLlm(MockLlm):
    """Everything mid-run is a side question, which is how the live model reads
    a timing question — and the mock sometimes calls the same sentence off-topic,
    which sends it to the scope reply instead of to the window reader."""

    model: str = "battery-side-question-stub"

    def _classify(self, llm_request, available, done, task):  # noqa: ANN001
        return function_call_response(
            "classify_message",
            {"message_class": "side_question", "incoming_steps": []},
        )


class DecliningLlm(MockLlm):
    """A model that reads every unread answer as a refusal.

    Live: "yes lets confirm it" came back as ``decline``, with that sentence
    quoted in the verdict's own reason field.
    """

    model: str = "battery-declining-stub"

    def _classify(self, llm_request, available, done, task):  # noqa: ANN001
        if (
            "submit_confirmation_verdict" in available
            and "submit_confirmation_verdict" not in done
        ):
            return function_call_response(
                "submit_confirmation_verdict",
                {"verdict": "decline", "reason": "they named another time"},
            )
        return function_call_response(
            "classify_message",
            {"message_class": "continuation", "incoming_steps": []},
        )


class WindowProposingLlm(MockLlm):
    """A model that answers an unreadable timing phrase with two dates.

    The mock never does this on its own — layer (a) reads the phrases it knows —
    so without a stub, "whenever the moon is full" cannot reach layer (b) at all.
    """

    model: str = "battery-window-stub"
    window_start: str = ""
    window_end: str = ""

    def _classify(self, llm_request, available, done, task):  # noqa: ANN001
        # Classify first, then propose. The class has to be forced as well as
        # the window: under the mock "whenever the moon is full" comes back
        # *off-topic*, and the scope branch returns before the answer half of
        # answer-and-stay can render anything — so the search would run and its
        # result would be thrown away. Live the model called it a side question.
        if "classify_message" in available and "classify_message" not in done:
            return function_call_response(
                "classify_message",
                {"message_class": "side_question", "incoming_steps": []},
            )
        if (
            "propose_search_window" in available
            and "propose_search_window" not in done
        ):
            return function_call_response(
                "propose_search_window",
                {"start": self.window_start, "end": self.window_end},
            )
        return super()._classify(llm_request, available, done, task)


def _provider(monkeypatch, stub) -> None:
    monkeypatch.setattr("app.agents.base.get_provider", lambda name=None: stub)


# --- A -------------------------------------------------------------------


class TestASpineOfABooking:
    """Knee pain books first try; the window is honoured; a position is read; an
    exact yes commits. Round 9's floor and round 7's selection reader, together.
    """

    def test_the_whole_sequence(self, patient):
        first = turn(patient, "I want to book an appointment for knee pain", "bat-a")
        assert first.plan == ["route", "book", "documents", "follow_up"]
        assert first.status == WorkflowStatus.PENDING_CONFIRMATION.value
        assert "Orthopedics" in first.reply

        listed = turn(patient, "any slots next week?", "bat-a")
        assert listed.status == WorkflowStatus.PENDING_CONFIRMATION.value
        assert day_label("next week") in listed.reply
        assert _run(first.run_id).non_answer_count == 0, (
            "a turn that rendered times is not a re-ask"
        )

        chosen = turn(patient, "option 2", "bat-a")
        assert chosen.status == WorkflowStatus.PENDING_CONFIRMATION.value

        committed = turn(patient, "yes", "bat-a")
        assert committed.status == WorkflowStatus.COMPLETED.value
        booked = [
            appointment
            for appointment in _appointments(_patient_id(first.run_id))
            if appointment.id != SEEDED_APPOINTMENT_ID
        ]
        assert len(booked) == 1
        assert booked[0].status is AppointmentStatus.CONFIRMED
        # The receipt is assembled from the row, so the reference is the row's.
        assert booked[0].reference_code in committed.reply


# --- B -------------------------------------------------------------------


class TestBAClashIsNeverOfferedAndNeverCommitted:
    """Round 9's priority zero and round 8's clash sentence, in one sequence.

    The booking is asked for **on the day the patient is already busy**, and that
    is what makes the sequence reach the sentence rather than the selection
    reader. "9am" on any other day is a time the run has already offered, so
    ``read_selection`` matches it against the shown list and re-holds — correct
    behaviour, costing one re-ask, and it means the live shape has to be
    reproduced faithfully: live the patient named 11am, which was *not* among the
    9, 10 and 2 they had been shown.
    """

    def _busy_hour(self):
        """When the patient is already committed — the seed ships one booking."""
        session = fresh()
        try:
            appointment = session.get(Appointment, SEEDED_APPOINTMENT_ID)
            return appointment.slot.start_time, appointment.slot.end_time
        finally:
            session.close()

    def test_the_busy_hour_is_never_offered_and_naming_it_is_explained(self, patient):
        when, _ = self._busy_hour()
        hour = when.hour % 12 or 12
        suffix = "am" if when.hour < 12 else "pm"

        first = turn(
            patient,
            f"I need a dermatology appointment for a rash on {when.day} {when:%B}",
            "bat-b",
        )
        assert first.status == WorkflowStatus.PENDING_CONFIRMATION.value
        offered = [
            line
            for line in first.reply.splitlines()
            if line.strip().startswith(("1.", "2.", "3."))
        ]
        assert offered, "no times were offered, so nothing is being checked"
        assert all(f"{hour}:00 {suffix.upper()}" not in line for line in offered), (
            "the patient's own hour was offered back to them"
        )

        asked = turn(patient, f"how about {hour}{suffix}?", "bat-b")

        assert "clashes with your Cardiology appointment" in asked.reply
        assert asked.status == WorkflowStatus.PENDING_CONFIRMATION.value

    def test_the_commit_refuses_it_even_if_something_offers_it(self, patient):
        """The floor under the other two layers. Asked of the shared helper,
        because the point is that the state is impossible rather than unoffered —
        and reschedule reaches this through the same call as book."""
        from app.tools.availability import patient_clash

        start, end = self._busy_hour()
        session = fresh()
        try:
            clash = patient_clash(
                session,
                patient_id=session.get(Appointment, SEEDED_APPOINTMENT_ID).patient_id,
                start=start,
                end=end,
            )
        finally:
            session.close()
        assert clash is not None


# --- C -------------------------------------------------------------------


class TestCTheOtherThingIAsked:
    """Round 10 item 3, on round 11 item 6's phrasing. Two verbs, two desks.

    The friendly phrasing this used to carry — "cancel that appointment and book
    a new one for cardiology" — names one department, so the resolver settled it
    and the offer named it. That hid the real shape of the live message, where
    each half names a desk of its own: the resolution is *ambiguous*, nothing was
    stored, the offer said "booking an appointment", and the "yes" then routed the
    whole two-verb sentence — ambiguous again, low confidence, queued for a human.

    The departments are swapped from the live pair so that the kept half can
    settle against the seeded appointment, which is Cardiology. The rule under
    test is unchanged: the kept run's own desk is subtracted, and what is left is
    the one the dropped half was about.
    """

    def test_the_whole_sequence(self, patient):
        split = turn(
            patient,
            "book me a dermatology appointment and cancel my cardiology one",
            "bat-c",
        )
        assert "One change at a time" in split.reply

        done = turn(patient, "yes", "bat-c")
        assert done.status == WorkflowStatus.COMPLETED.value
        assert _run(done.run_id).plan == ["cancel"]

        offered = turn(patient, "now do the other thing I asked", "bat-c")
        assert "booking a Dermatology appointment" in offered.reply
        assert offered.run_id is None, "an offer starts nothing"

        started = turn(patient, "yes", "bat-c")
        assert started.plan == ["route", "book", "documents", "follow_up"]
        assert started.status == WorkflowStatus.PENDING_CONFIRMATION.value
        assert (_run(started.run_id).state or {}).get("department_name") == (
            "Dermatology"
        )


# --- D -------------------------------------------------------------------


class TestDAVerbSwitchMidRun:
    """A supersede that keeps the patient's second intent, and a commit that
    acts on the appointment they named rather than on one of the others."""

    def test_the_whole_sequence(self, patient):
        first = turn(patient, "I need a dermatology appointment for a rash", "bat-d")
        assert first.status == WorkflowStatus.PENDING_CONFIRMATION.value

        switched = turn(
            patient, "actually please cancel my cardiology appointment instead", "bat-d"
        )
        assert switched.run_id != first.run_id, "the verb switch must supersede"
        assert _run(first.run_id).status is WorkflowStatus.CANCELLED
        assert _run(switched.run_id).plan == ["cancel"]

        committed = turn(patient, "yes", "bat-d")

        session = fresh()
        try:
            target = session.get(Appointment, SEEDED_APPOINTMENT_ID)
            assert target.status is AppointmentStatus.CANCELLED
        finally:
            session.close()
        assert committed.status == WorkflowStatus.COMPLETED.value


# --- E -------------------------------------------------------------------


class TestESafetyAndScope:
    """The four shapes, and the one deliberate latitude."""

    def test_an_emergency_escalates_before_anything_runs(self, patient):
        result = turn(patient, "I'm having chest pain", "bat-e1")

        assert result.status == WorkflowStatus.ESCALATED.value
        assert "emergency" in result.reply.lower()
        assert "propose_appointment" not in _tools(result.turn_id)

    def test_a_clinical_question_is_refused_administratively(self, patient):
        result = turn(patient, "what dose of paracetamol should I take?", "bat-e2")

        assert result.status == WorkflowStatus.ESCALATED.value
        session = fresh()
        try:
            kinds = {
                row.kind
                for row in session.query(Escalation)
                .filter(Escalation.workflow_run_id == result.run_id)
                .all()
            }
        finally:
            session.close()
        assert EscalationKind.SAFETY in kinds

    def test_a_symptom_that_is_administrative_either_books_or_escalates(
        self, patient
    ):
        """The deliberate latitude, with both outcomes specified. The screen's
        false-positive direction is the expensive one, so it is allowed to fire
        on a symptom description — what is *not* allowed is anything else."""
        result = turn(
            patient, "I need an appointment, my vision has been blurry", "bat-e3"
        )

        assert result.status in {
            WorkflowStatus.PENDING_CONFIRMATION.value,
            WorkflowStatus.PENDING_REVIEW.value,
            WorkflowStatus.ESCALATED.value,
        }
        if result.status == WorkflowStatus.PENDING_CONFIRMATION.value:
            assert "Ophthalmology" in result.reply
        else:
            assert "staff" in result.reply.lower()
        # True either way: nothing is booked without the patient's word.
        assert [
            appointment
            for appointment in _appointments(_patient_id(result.run_id))
            if appointment.id != SEEDED_APPOINTMENT_ID
        ] == []

    @pytest.mark.parametrize(
        "message", ["how is nvidia stock doing", "what is the capital city of France?"]
    )
    def test_off_topic_gets_the_administration_line_and_no_run(
        self, patient, message
    ):
        result = turn(patient, message, f"bat-e4-{abs(hash(message))}")

        assert result.reply in CLARIFIES
        assert result.run_id is None
        session = fresh()
        try:
            assert session.query(WorkflowRun).count() == 0
        finally:
            session.close()


# --- F -------------------------------------------------------------------


class TestFTheWindowVocabulary:
    """Layer (a) for the phrasings, layer (b) for the one nobody can read.

    The timing turns run under :class:`SideQuestionLlm`. That is the live
    classification for these sentences, and under the mock two of them come back
    *off-topic* — which sends them to the scope reply, so the window reader never
    runs and the step would be testing the mock's classifier instead of the
    vocabulary it is about.
    """

    @pytest.fixture
    def holding(self, patient, monkeypatch):
        first = turn(patient, "I need a cardiology appointment", "bat-f")
        assert first.status == WorkflowStatus.PENDING_CONFIRMATION.value
        _provider(monkeypatch, SideQuestionLlm())
        return first

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("any slots on Tuesdays?", "Tuesday"),
            ("what about the week after next?", "August"),
            ("anything at the weekend?", "August"),
            ("how about the afternoon of august 10th?", "10 August"),
        ],
    )
    def test_a_readable_window_is_honoured(
        self, patient, holding, message, expected
    ):
        result = turn(patient, message, "bat-f")

        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        assert result.reply not in CLARIFIES
        assert expected in result.reply
        # Either the window was honoured, or it was named as empty. What must
        # never happen is the earliest three slots with nothing said.
        assert "The time I'm holding" in result.reply

    def test_an_unreadable_phrase_costs_no_model_window_when_the_words_resolve(
        self, patient, holding
    ):
        """Layer ordering, from the trace: a phrase layer (a) can read must not
        reach the model's window tool."""
        result = turn(patient, "any slots next week?", "bat-f")

        assert "propose_search_window" not in _tools(result.turn_id)

    def test_the_moon_question_names_the_window_it_chose(self, patient, monkeypatch):
        """Item 4a. Nothing false shipped live — and nothing said why Saturday."""
        first = turn(patient, "I need a cardiology appointment", "bat-f2")
        assert first.status == WorkflowStatus.PENDING_CONFIRMATION.value

        target = clock.today() + timedelta(days=5)
        _provider(
            monkeypatch,
            WindowProposingLlm(
                window_start=target.isoformat(), window_end=target.isoformat()
            ),
        )

        result = turn(patient, "got anything whenever the moon is full?", "bat-f2")

        assert f"Times that are free on {target:%A} {target.day} {target:%B}" in (
            result.reply
        )


# --- G -------------------------------------------------------------------


class TestGReadingTheAnswer:
    """The readers, the decline cue, the withdrawal, and the stray token."""

    def _two_appointments(self, patient) -> int:
        turn(patient, "I need a dermatology appointment for a rash", "bat-g-setup")
        turn(patient, "yes", "bat-g-setup")
        live = [
            appointment
            for appointment in _appointments(1)
            if appointment.status is AppointmentStatus.CONFIRMED
        ]
        assert len(live) == 2, "the sequence needs a choice to make"
        return len(live)

    def test_a_bare_number_answers_the_appointment_list(self, patient):
        self._two_appointments(patient)

        asked = turn(patient, "please cancel my appointment", "bat-g1")
        assert "1." in asked.reply and "2." in asked.reply

        answered = turn(patient, "2", "bat-g1")

        assert answered.reply not in CLARIFIES
        assert _run(answered.run_id).proposed_appointment_id is not None
        assert "classify_message" not in _tools(answered.turn_id), (
            "a question code asked is answered by code"
        )

    def test_an_affirmative_with_extras_keeps_the_proposal(self, patient, monkeypatch):
        """Item 1. The model is forced to submit the verdict it submitted live;
        the guard is the only thing between it and a cleared proposal."""
        first = turn(patient, "I need a cardiology appointment", "bat-g2")
        held = _run(first.run_id).proposed_slot_id
        assert held is not None

        _provider(monkeypatch, DecliningLlm())
        result = turn(patient, "yes please, sounds good!", "bat-g2")

        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        assert _run(first.run_id).proposed_slot_id == held
        assert result.author is TraceAuthor.TEMPLATE
        assert "exact" in result.reply

    def test_a_genuine_decline_still_declines(self, patient, monkeypatch):
        first = turn(patient, "I need a cardiology appointment", "bat-g3")
        _provider(monkeypatch, DecliningLlm())

        turn(patient, "no thanks", "bat-g3")

        assert _run(first.run_id).proposed_slot_id is None

    def test_a_withdrawal_closes_the_run_and_offers_to_restart_it(self, patient):
        first = turn(patient, "I need a cardiology appointment", "bat-g4")

        closed = turn(patient, "actually never mind", "bat-g4")
        assert _run(first.run_id).status is WorkflowStatus.CANCELLED
        assert closed.reply not in CLARIFIES

        offered = turn(patient, "wait no, I still want it", "bat-g4")
        assert "restart" in offered.reply.lower()
        assert offered.run_id is None

        restarted = turn(patient, "yes", "bat-g4")
        assert restarted.run_id != first.run_id
        assert restarted.status == WorkflowStatus.PENDING_CONFIRMATION.value

    def test_a_stray_token_starts_nothing(self, patient):
        result = turn(patient, "ok", "bat-g5")

        assert result.reply in CLARIFIES
        assert result.run_id is None
        session = fresh()
        try:
            assert session.query(WorkflowRun).count() == 0
        finally:
            session.close()

    def test_a_queued_request_answers_from_the_wall(self, patient, monkeypatch):
        """Item 2, in the sequence that produced it: a booking message, an exact
        "yes", a position and a timing question, all arriving at a run a person
        is holding."""
        _provider(monkeypatch, ContinuationLlm())
        first = turn(patient, "book an appointment, my kid has ear pain", "bat-g6")
        assert first.status == WorkflowStatus.PENDING_REVIEW.value

        for message in (
            "book me a cardiology appointment",
            "yes",
            "option 3",
            "the earliest the better",
        ):
            result = turn(patient, message, "bat-g6")
            assert result.reply == AWAITING_REVIEW_REPLY, message
            assert "propose_appointment" not in _tools(result.turn_id), message
            assert "submit_routing" not in _tools(result.turn_id), message
