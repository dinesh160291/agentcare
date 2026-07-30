"""The capture-point audit: is every point actually reached by the shipped system?

``tests/unit/test_trace.py`` proves the writer *can* record all nine points, one
method at a time, against a writer nothing else is holding. That is a test of
the writer. It says nothing about whether the orchestrator, the agents, the
guards and the staff surface actually *call* those methods — and a capture point
with no caller is exactly as invisible as one that was never written.

So this module drives real traffic through the real seams (``run_workflow``,
``apply_patient_action``, ``apply_staff_decision``, a failing provider) and then
audits the rows that landed. It is deliberately an audit over a *corpus* rather
than one assertion per turn: what matters is that no point is missing from the
whole system, and pinning each point to the one turn that happens to produce it
today would break on every unrelated change to that turn.

Falsification: each assertion below names the defect it would catch, and each is
a defect this project could actually ship — a guard that stops recording its
passes, a template reply that stops declaring its author, a staff decision that
writes audit rows and no trace.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field

import pytest

from app.db import SessionLocal
from app.errors import ProviderError
from app.models import (
    TraceAuthor,
    TraceEvent,
    TraceEventType,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.orchestrator import apply_patient_action, run_workflow
from app.providers.base import function_call_response
from app.providers.mock import MockLlm
from app.trace import TraceWriter, assert_well_formed
from app.workflow.staff import apply_staff_decision

PATIENT_EMAIL = "asha.patient@example.invalid"
STAFF_EMAIL = "staff@example.invalid"

BOOKING = "I need a cardiology appointment next week"
#: The seed's deliberately ambiguous case: Pediatrics or ENT, a human decides.
AMBIGUOUS = "book an appointment, my kid has ear pain"
EMERGENCY = "I have chest pain and my left arm hurts"
OFF_TOPIC = "who won the fifa final"


class FailingLlm(MockLlm):
    """A provider that dies mid-call.

    The only way to reach ``llm_error`` on real traffic: under the mock nothing
    fails, and a rate limit is not something a test may wait for. Subclassing
    the mock rather than stubbing ``BaseLlm`` keeps the failure at the provider
    seam, which is where a real one happens.
    """

    async def generate_content_async(self, llm_request, stream=False):  # noqa: ANN001
        raise ProviderError("openai call failed: connection reset")
        yield  # pragma: no cover - unreachable, makes this an async generator


#: The stub below needs one bit of state and cannot hold it: ADK's ``BaseLlm``
#: is a pydantic model, so an undeclared instance attribute is an error rather
#: than a flag. Reset by the fixture that uses it.
_INVENTED: list[bool] = []


class InventedDepartmentLlm(MockLlm):
    """A Coordinator that proposes a department the hospital does not have.

    Under the mock every proposal is valid by construction — the routing agent
    reads ``resolve_department``'s payload and submits the name it found — so
    ordinary traffic produces no *rejected* validation at all. That is the half
    of the validation capture point that matters: an invented department which
    never reached the database still happened, and the trace is the only place
    it shows.

    Invents once, then behaves, so the turn also exercises the retry ladder
    rather than only the refusal.
    """

    def _route(self, llm_request, done, task):  # noqa: ANN001
        if not _INVENTED and "resolve_department" in done:
            _INVENTED.append(True)
            return function_call_response(
                "submit_routing",
                {"department_name": "Cardiovascular Medicine", "confidence": "high"},
            )
        return super()._route(llm_request, done, task)


@dataclass
class Coverage:
    """What the corpus's trace rows contain, summarised."""

    event_types: set[str] = field(default_factory=set)
    authors: set[str] = field(default_factory=set)
    guards: dict[bool, set[str]] = field(default_factory=lambda: {True: set(), False: set()})
    validations: dict[bool, set[str]] = field(
        default_factory=lambda: {True: set(), False: set()}
    )
    inbound_authors: set[str] = field(default_factory=set)
    outbound_authors: set[str] = field(default_factory=set)
    counts: Counter = field(default_factory=Counter)


def _turn(user, message, session_id):
    return asyncio.run(run_workflow(user, message, session_id))


def _collect(session) -> Coverage:
    coverage = Coverage()
    for event in session.query(TraceEvent).order_by(TraceEvent.id).all():
        kind = event.event_type.value
        coverage.event_types.add(kind)
        coverage.counts[kind] += 1
        if event.author is not None:
            coverage.authors.add(event.author.value)
        payload = event.payload or {}
        if event.event_type is TraceEventType.GUARD_VERDICT:
            coverage.guards[bool(payload.get("passed"))].add(payload.get("guard", "?"))
        elif event.event_type is TraceEventType.VALIDATION:
            coverage.validations[bool(payload.get("accepted"))].add(
                payload.get("what", "?")
            )
        elif event.event_type is TraceEventType.INBOUND and event.author:
            coverage.inbound_authors.add(event.author.value)
        elif event.event_type is TraceEventType.OUTBOUND and event.author:
            coverage.outbound_authors.add(event.author.value)
    return coverage


@pytest.fixture
def corpus(seeded_db, monkeypatch) -> Coverage:
    """Every door into the system, once, against one seeded database."""
    seeded_db.commit()
    patient = seeded_db.query(User).filter(User.email == PATIENT_EMAIL).one()
    staff = seeded_db.query(User).filter(User.email == STAFF_EMAIL).one()
    staff_id = staff.id

    # 1. A booking, taken to a held proposal: plan, tools, transitions, an LLM
    #    reply. The ordinary path, and the one that produces the most points.
    _turn(patient, BOOKING, "cap-book")

    # 2. The button. A typed action is an inbound event too, and it is the door
    #    a judge reaches by clicking Confirm.
    asyncio.run(apply_patient_action(patient, "confirm", "cap-book"))

    # 3. Off topic: a reply code wrote, which must not be attributed to a model.
    _turn(patient, OFF_TOPIC, "cap-scope")

    # 4. The safety screen firing, on a fresh session because the verdict is
    #    terminal for the run it lands on.
    _turn(patient, EMERGENCY, "cap-safety")

    # 5. A human decision. It writes trace rows of its own, and if it ever
    #    stopped, the timeline would show a run resuming for no visible reason.
    paused = _turn(patient, AMBIGUOUS, "cap-staff")
    assert paused.status == WorkflowStatus.PENDING_REVIEW.value, (
        "the corpus depends on the seeded ambiguity still pausing for staff"
    )
    session = SessionLocal()
    try:
        run = session.get(WorkflowRun, paused.run_id)
        writer = TraceWriter(session, session_id=run.session_id)
        apply_staff_decision(
            session,
            staff=session.get(User, staff_id),
            run_id=paused.run_id,
            action="approve",
            writer=writer,
        )
        session.commit()
    finally:
        session.close()

    # 6. A model proposing something that is not in the database. The rejection
    #    is the interesting half of the validation point and nothing on the
    #    ordinary path produces one.
    _INVENTED.clear()
    monkeypatch.setattr(
        "app.agents.base.get_provider", lambda name=None: InventedDepartmentLlm()
    )
    _turn(patient, "I need an appointment about my heart", "cap-invented")
    monkeypatch.undo()

    # 7. A provider that dies mid-call. `llm_error` has no other route: the
    #    mock never fails, and a request with no terminal partner is reserved
    #    to mean the process died.
    monkeypatch.setattr(
        "app.agents.base.get_provider", lambda name=None: FailingLlm()
    )
    with pytest.raises(ProviderError):
        _turn(patient, BOOKING, "cap-error")
    monkeypatch.undo()

    session = SessionLocal()
    try:
        return _collect(session)
    finally:
        session.close()


class TestEveryPointHasACaller:
    def test_every_event_type_is_written_by_something(self, corpus):
        """A capture point with no caller is as invisible as one never written.

        Catches: a refactor that stops recording transitions, or an agent loop
        that stops pairing its LLM requests. The writer's own tests cannot see
        either — they call the method themselves.
        """
        missing = {kind.value for kind in TraceEventType} - corpus.event_types
        assert not missing, f"no traffic reaches these capture points: {sorted(missing)}"

    def test_every_author_is_used(self, corpus):
        """``TraceAuthor`` has six members because six kinds of thing speak.

        An unused member means either a door that stopped writing (staff
        decisions) or a reply whose author is being guessed.
        """
        missing = {author.value for author in TraceAuthor} - corpus.authors
        assert not missing, f"no event is attributed to: {sorted(missing)}"

    def test_all_three_front_doors_open_a_turn(self, corpus):
        """Typed actions are inbound events too.

        The Confirm button and the staff Approve button are the system's most
        deterministic flows, and a checker that only saw chat would pass the
        least deterministic ones while ignoring these.
        """
        assert corpus.inbound_authors >= {
            TraceAuthor.PATIENT_MESSAGE.value,
            TraceAuthor.PATIENT_ACTION.value,
            TraceAuthor.STAFF_ACTION.value,
        }

    def test_code_authored_replies_say_so(self, corpus):
        """Most replies in this system are written by code, not by the model.

        If ``template`` and ``guard`` stopped appearing on outbound events, the
        timeline would imply the model said things it never wrote — which is
        the whole reason the author column exists.
        """
        assert corpus.outbound_authors >= {
            TraceAuthor.LLM.value,
            TraceAuthor.TEMPLATE.value,
            TraceAuthor.GUARD.value,
        }


class TestTheTwoHalvesNobodyRemembers:
    """Both of these have a half that ordinary traffic does not produce."""

    def test_guards_record_the_times_they_did_not_fire(self, corpus):
        """"The safety screen passed" and "the safety screen never ran" are
        different facts, and only one is acceptable."""
        assert corpus.guards[True], "no guard recorded a pass"
        assert corpus.guards[False], "no guard recorded a fire"
        assert "safety_keyword_screen" in corpus.guards[True]
        assert "safety_llm_screen" in corpus.guards[True]

    def test_a_rejected_proposal_is_recorded(self, corpus):
        """The invented department never reached the database — and still
        happened. Without this row nothing distinguishes a model that proposed
        it from one that never did."""
        assert corpus.validations[True], "no accepted validation"
        assert "routing_department" in corpus.validations[False], (
            "an invented department was rejected without leaving a record: "
            f"{sorted(corpus.validations[False])}"
        )


def test_the_whole_corpus_parses(corpus):
    """Every door, including the failed turn, leaves a well-formed trace.

    The provider failure is the interesting one: the envelope writes its
    template reply and commits *before* re-raising, so a turn that ended in an
    exception is still bracketed. A turn with no outbound would mean the
    patient was left with nothing and the trace agreed.
    """
    session = SessionLocal()
    try:
        assert_well_formed(session)
    finally:
        session.close()
