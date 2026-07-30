"""Round 10 item 3 — the short memory a refusal consults before shrugging.

The one-verb rule tells a patient "I'll start with the cancellation; ask me
about the other one right after", and until now nothing recorded what the other
one was. So "now do the other thing I asked" named no domain subject, met the
generic refusal, and the rephrase that followed handed routing the words
"booking request" — which is not a department, so it resolved to General
Medicine with low confidence and queued for a human. The patient said it once
and was asked to say it again, twice, worse each time.

Every test here runs the whole system under mock. Three things are pinned as
hard as the feature itself, because each is a direction this must not move in:

* **no memory → the old reply, byte for byte.** The generic refusal is the
  negative control, and a memory that improved it unconditionally would be a
  memory that answers "who won the fifa final" with an offer.
* **nothing starts without the patient's exact word.** The memory changes what a
  refusal offers and nothing else.
* **it expires.** A dropped verb from twelve turns ago is not a pending request.
"""

from __future__ import annotations

import asyncio

import pytest

from app.db import SessionLocal
from app.models import (
    AuditEvent,
    TraceAuthor,
    TraceEvent,
    TraceEventType,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.orchestrator import SCOPE_REPLY, run_workflow
from app.workflow.recall import MEMORY_WINDOW_TURNS

PATIENT_EMAIL = "asha.patient@example.invalid"

#: The live sentence, with a department in it so the offer has something to name.
TWO_VERBS = "okay lets cancel that appointment and book a new one for cardiology"
ASK_AGAIN = "now do the other thing I asked"
STILL_WANT = "wait no, I still want it"
OFF_TOPIC = "who won the fifa final"


@pytest.fixture
def patient(seeded_db):
    seeded_db.commit()
    return seeded_db.query(User).filter(User.email == PATIENT_EMAIL).one()


def turn(user, message, session_id):
    return asyncio.run(run_workflow(user, message, session_id))


def fresh():
    return SessionLocal()


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


def _runs(session, patient_id: int) -> list[WorkflowRun]:
    return (
        session.query(WorkflowRun)
        .filter(WorkflowRun.patient_id == patient_id)
        .order_by(WorkflowRun.id)
        .all()
    )


class TestTheDroppedVerbIsRemembered:
    """Sequence C of the acceptance battery, end to end."""

    def _split_and_finish(self, patient, session_id: str):
        """The two-verb message, then the confirm that completes the half we do."""
        first = turn(patient, TWO_VERBS, session_id)
        assert "One change at a time" in first.reply
        done = turn(patient, "yes", session_id)
        assert done.status == WorkflowStatus.COMPLETED.value
        return done

    def test_the_offer_names_the_remembered_request(self, patient):
        self._split_and_finish(patient, "s-recall-1")

        result = turn(patient, ASK_AGAIN, "s-recall-1")

        assert "booking a Cardiology appointment" in result.reply
        assert result.run_id is None
        assert result.author is TraceAuthor.TEMPLATE

    def test_an_offer_starts_nothing_by_itself(self, patient):
        """The whole memory is one sentence's worth of change. It may not create
        a run, touch the closed one, or book anything."""
        done = self._split_and_finish(patient, "s-recall-2")
        offered = turn(patient, ASK_AGAIN, "s-recall-2")
        # Asserted first, so this cannot pass by there being no offer at all:
        # "nothing was created" is true of a turn that did nothing.
        assert "booking a Cardiology appointment" in offered.reply

        session = fresh()
        try:
            runs = _runs(session, session.get(WorkflowRun, done.run_id).patient_id)
            assert [run.id for run in runs] == [done.run_id]
            assert runs[0].status is WorkflowStatus.COMPLETED
        finally:
            session.close()

    def test_an_exact_yes_starts_the_remembered_verb(self, patient):
        self._split_and_finish(patient, "s-recall-3")
        turn(patient, ASK_AGAIN, "s-recall-3")

        result = turn(patient, "yes", "s-recall-3")

        assert result.plan == ["route", "book", "documents", "follow_up"]
        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value

    def test_the_restarted_run_routes_on_the_remembered_subject(self, patient):
        """The point of storing the department rather than the sentence. Live,
        the patient's own rephrase gave routing "booking request" — not a
        department, so General Medicine at low confidence and into a queue. The
        subject words are the routing text now, so the restarted run routes
        where the original sentence said."""
        self._split_and_finish(patient, "s-recall-4")
        turn(patient, ASK_AGAIN, "s-recall-4")
        result = turn(patient, "yes", "s-recall-4")

        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert (run.state or {}).get("department_name") == "Cardiology"
            assert run.proposed_slot_id is not None
        finally:
            session.close()

    def test_one_yes_cannot_start_two_runs(self, patient):
        """The memory is spent when it is acted on. Boundedness applies to a
        writer that creates runs as much as to a retry ladder."""
        self._split_and_finish(patient, "s-recall-5")
        turn(patient, ASK_AGAIN, "s-recall-5")
        started = turn(patient, "yes", "s-recall-5")
        # Same trap as above: "no second run" is trivially true if there was
        # never a first one.
        assert started.plan == ["route", "book", "documents", "follow_up"]

        session = fresh()
        try:
            run = session.get(WorkflowRun, started.run_id)
            assert (run.state or {}).get("dropped_request") is None
            assert (run.state or {}).get("restart_offer") is None
            before = len(_runs(session, run.patient_id))
        finally:
            session.close()

        turn(patient, ASK_AGAIN, "s-recall-5")

        session = fresh()
        try:
            run = session.get(WorkflowRun, started.run_id)
            assert len(_runs(session, run.patient_id)) == before
        finally:
            session.close()

    def test_both_halves_are_recorded(self, patient):
        self._split_and_finish(patient, "s-recall-6")
        offered = turn(patient, ASK_AGAIN, "s-recall-6")
        accepted = turn(patient, "yes", "s-recall-6")

        session = fresh()
        try:
            assert _guard(session, offered.turn_id, "request_recalled")["passed"] is True
            assert _guard(session, accepted.turn_id, "request_restarted")["detail"][
                "verb"
            ] == "book"
            actions = {
                row.action
                for row in session.query(AuditEvent)
                .filter(AuditEvent.action.like("request_recall%"))
                .all()
            }
        finally:
            session.close()

        assert actions == {"request_recall_offered", "request_recall_accepted"}


class TestAClosedRunIsRememberedToo:
    """The second supplier, and it needs no writing: the run row already holds
    the request text, the plan and the department."""

    def _withdrawn(self, patient, session_id: str):
        first = turn(patient, "I need a dermatology appointment for a rash", session_id)
        turn(patient, "actually never mind", session_id)

        session = fresh()
        try:
            assert session.get(WorkflowRun, first.run_id).status is (
                WorkflowStatus.CANCELLED
            )
        finally:
            session.close()
        return first

    def test_the_offer_names_the_run_that_was_closed(self, patient):
        self._withdrawn(patient, "s-recall-w1")

        result = turn(patient, STILL_WANT, "s-recall-w1")

        assert "the Dermatology booking" in result.reply
        assert result.run_id is None

    def test_a_yes_starts_a_fresh_run_in_the_same_department(self, patient):
        first = self._withdrawn(patient, "s-recall-w2")
        turn(patient, STILL_WANT, "s-recall-w2")

        result = turn(patient, "yes", "s-recall-w2")

        assert result.run_id != first.run_id

        session = fresh()
        try:
            assert (session.get(WorkflowRun, result.run_id).state or {}).get(
                "department_name"
            ) == "Dermatology"
            assert session.get(WorkflowRun, first.run_id).status is (
                WorkflowStatus.CANCELLED
            )
        finally:
            session.close()

    def test_an_escalated_run_is_never_offered_back(self, patient):
        """The second door into the thing that makes ``escalated`` terminal.

        A booking escalated *mid-run* by a clinical question, so the run carries
        a full plan and a department — the shape that would otherwise be
        perfectly restartable. It belongs to a person now, and "I still want it"
        is exactly the moment not to be helpful: the same reasoning that exempts
        safety escalations from the tidy-up rule.

        A born-escalated safety run would make this pass for the wrong reason —
        its plan is empty, so nothing could have been recalled from it whatever
        the status rule said. Checked by sabotage, which is how the first
        version of this test was caught.

        **The state is now constructed rather than reached.** It used to arrive
        through a clinical question sent mid-run, which took the booking run
        itself to ``escalated``; round 11b stopped that — a live run is spared
        and the scare gets its own — and with it went the last door from a
        *planned* run into this state. The rule outlives the door: recall asks
        "may this run be offered back?" and the status is its input, so a run
        carrying a plan, a department and an ``escalated`` status is exactly
        what has to be refused, however it came to exist.
        """
        first = turn(patient, "I need a dermatology appointment for a rash", "s-recall-w3")

        session = fresh()
        try:
            run = session.get(WorkflowRun, first.run_id)
            assert run.plan and (run.state or {}).get("department_name")
            run.status = WorkflowStatus.ESCALATED
            session.commit()
        finally:
            session.close()

        result = turn(patient, STILL_WANT, "s-recall-w3")

        assert result.reply == SCOPE_REPLY
        assert result.run_id is None


#: The live shape of a two-verb message: each half names a desk of its own, so
#: the resolver reports ambiguity rather than a department. Departments swapped
#: from the live pair, because the kept half has to settle against an appointment
#: the patient actually has and the seeded one is Cardiology.
TWO_DEPARTMENTS = "book me a dermatology appointment and cancel my cardiology one"


class TestTheRememberedDepartment:
    """Round 11 item 6 — the offer names the desk, or it names nothing useful.

    ``TWO_VERBS`` above says "cancel that appointment", which names no department
    at all, so the message resolves to exactly one and the memory stored it. The
    live sentence names two, which resolves to *ambiguous* — and nothing was
    stored, so the offer said "booking an appointment", and the "yes" that
    accepted it routed the whole two-verb text: ambiguous again, low confidence,
    queued for a human. The friendly phrasing had hidden the whole defect.

    The subtraction is not a guess. The cancel run *acted on* Cardiology, so that
    candidate is accounted for; whatever is left is what the dropped half was
    about.
    """

    def _split(self, patient, session_id: str):
        first = turn(patient, TWO_DEPARTMENTS, session_id)
        assert "One change at a time" in first.reply
        done = turn(patient, "yes", session_id)
        assert done.status == WorkflowStatus.COMPLETED.value
        return done

    def test_the_offer_names_the_other_desk(self, patient):
        self._split(patient, "s-recall-dept-1")

        result = turn(patient, ASK_AGAIN, "s-recall-dept-1")

        assert "booking a Dermatology appointment" in result.reply

    def test_the_yes_routes_on_the_department_alone(self, patient):
        """What the generic offer cost: the restarted run's request text was the
        two-verb sentence, which resolves to nothing but ambiguity — so routing
        dropped to low confidence and the request went to a staff queue instead
        of to the patient."""
        self._split(patient, "s-recall-dept-2")
        turn(patient, ASK_AGAIN, "s-recall-dept-2")

        result = turn(patient, "yes", "s-recall-dept-2")

        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value
        session = fresh()
        try:
            run = session.get(WorkflowRun, result.run_id)
            assert (run.state or {}).get("department_name") == "Dermatology"
            assert run.request_text == "Dermatology"
        finally:
            session.close()

    def test_a_message_naming_one_desk_is_unchanged(self, patient):
        """The negative control on the other side: subtraction only ever runs on
        an ambiguous resolution, so a sentence the resolver settles keeps the
        answer it always had."""
        turn(patient, TWO_VERBS, "s-recall-dept-3")
        turn(patient, "yes", "s-recall-dept-3")

        result = turn(patient, ASK_AGAIN, "s-recall-dept-3")

        assert "booking a Cardiology appointment" in result.reply

    def test_three_desks_still_store_nothing(self, patient):
        """Two candidates left after the subtraction is still a message that does
        not say. The generic offer is the honest answer there, and keeping it is
        what stops this becoming a tiebreak."""
        first = turn(
            patient,
            "book me a dermatology appointment or maybe an eye test and cancel "
            "my cardiology one",
            "s-recall-dept-4",
        )
        assert "One change at a time" in first.reply
        turn(patient, "yes", "s-recall-dept-4")

        result = turn(patient, ASK_AGAIN, "s-recall-dept-4")

        assert "booking an appointment" in result.reply


class TestWhatTheMemoryMustNotChange:
    """The negative controls. Each one is a reply that has to stay exactly what
    it was, and together they are what keep the memory from becoming a
    conversational lottery of its own."""

    def test_a_back_reference_with_no_memory_gets_the_old_reply(self, patient):
        result = turn(patient, ASK_AGAIN, "s-recall-n1")

        assert result.reply == SCOPE_REPLY
        assert result.run_id is None

    def test_off_topic_after_a_closed_run_stays_off_topic(self, patient):
        """The direction that would have broken the acceptance battery. A
        completed booking sits in this conversation and "who won the fifa final"
        points at nothing, so the refusal is byte-identical."""
        turn(patient, "I need a dermatology appointment for a rash", "s-recall-n2")
        turn(patient, "yes", "s-recall-n2")

        assert turn(patient, OFF_TOPIC, "s-recall-n2").reply == SCOPE_REPLY
        assert (
            turn(patient, "what is the capital city of India?", "s-recall-n2").reply
            == SCOPE_REPLY
        )

    def test_a_stray_yes_starts_nothing(self, patient):
        """"ok" with nothing pending is a clarify, not a run. The memory may
        never turn a loose token into work."""
        result = turn(patient, "yes", "s-recall-n3")

        assert result.run_id is None

        session = fresh()
        try:
            profile_runs = session.query(WorkflowRun).count()
        finally:
            session.close()
        assert profile_runs == 0

    def test_an_offer_is_answerable_on_the_next_turn_only(self, patient):
        """A "yes" five messages later is an answer to something else — and by
        then the patient may have said no to this one."""
        turn(patient, TWO_VERBS, "s-recall-n4")
        turn(patient, "yes", "s-recall-n4")
        turn(patient, ASK_AGAIN, "s-recall-n4")
        turn(patient, OFF_TOPIC, "s-recall-n4")

        result = turn(patient, "yes", "s-recall-n4")

        assert result.run_id is None

    def test_the_memory_expires(self, patient):
        """Twelve turns later the dropped verb is not a pending request. The
        window is the growth rule: without one, every conversation accumulates
        offers it can still make."""
        turn(patient, TWO_VERBS, "s-recall-n5")
        turn(patient, "yes", "s-recall-n5")
        for _ in range(MEMORY_WINDOW_TURNS + 1):
            turn(patient, OFF_TOPIC, "s-recall-n5")

        result = turn(patient, ASK_AGAIN, "s-recall-n5")

        assert result.reply == SCOPE_REPLY

    def test_a_memory_from_another_conversation_is_not_offered(self, patient):
        """Keyed to the conversation, not to the patient. Two sessions are two
        conversations and "the other thing I asked" means the other thing asked
        *here*."""
        turn(patient, TWO_VERBS, "s-recall-n6a")
        turn(patient, "yes", "s-recall-n6a")

        result = turn(patient, ASK_AGAIN, "s-recall-n6b")

        assert result.reply == SCOPE_REPLY
