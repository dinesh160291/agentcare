"""The Coordinator's plan — validated, ordered, and superset-closed by code.

Written before the implementation. The plan is the clearest instance of the
architectural spine in the whole system: the model decides *which* specialists
a request needs, and code decides whether that decision is executable.

Three failures this module exists to prevent:

* **Freeform delegation.** "Code can't enforce ordering on a plan it can't
  parse" — a plan in prose is a slip nobody can validate, so the plan is a list
  drawn from a closed enum or it is nothing.
* **The silent narrow plan.** A two-intent message ("book me cardiology, and
  here's my ECG") whose plan comes back as documents-only *succeeds*. The
  patient gets a cheerful reply about their document and no appointment. The
  superset rule makes the narrower intent ride inside the wider one.
* **Out-of-order execution.** Booking before routing, or diffing required
  documents before a department is known, are both well-formed and both wrong.
"""

from __future__ import annotations

import pytest

from app.errors import BudgetExceeded, PlanRejected
from app.models import PlanStep, WorkflowRun, WorkflowStatus
from app.workflow.plan import (
    CANONICAL_ORDER,
    advance_plan,
    append_step,
    is_plan_complete,
    next_step,
    record_replan,
    validate_plan,
)

ROUTE, BOOK, DOCUMENTS, FOLLOW_UP = (
    PlanStep.ROUTE,
    PlanStep.BOOK,
    PlanStep.DOCUMENTS,
    PlanStep.FOLLOW_UP,
)


@pytest.fixture
def run(seeded_db):
    run = WorkflowRun(patient_id=1, status=WorkflowStatus.IN_PROGRESS)
    seeded_db.add(run)
    seeded_db.flush()
    return run


class TestSchemaValidation:
    def test_a_plan_of_known_steps_is_accepted(self):
        assert validate_plan(["route"]) == [ROUTE]

    def test_an_unknown_step_is_rejected(self):
        """The closed enum is the whole point. A model inventing "diagnose"
        must never reach execution, whatever else the plan contained."""
        with pytest.raises(PlanRejected):
            validate_plan(["route", "diagnose"])

    def test_prose_is_rejected(self):
        """The failure the typed plan exists to prevent."""
        with pytest.raises(PlanRejected):
            validate_plan("First I'll route this, then book an appointment")

    def test_an_empty_plan_is_rejected(self):
        """A plan with no steps is a model that answered nothing while
        appearing to answer."""
        with pytest.raises(PlanRejected):
            validate_plan([])

    def test_none_is_rejected(self):
        with pytest.raises(PlanRejected):
            validate_plan(None)

    def test_a_nested_structure_is_rejected(self):
        with pytest.raises(PlanRejected):
            validate_plan([["route"], ["book"]])

    def test_a_dict_of_steps_is_rejected(self):
        with pytest.raises(PlanRejected):
            validate_plan({"steps": ["route"]})

    def test_the_rejection_names_what_was_wrong(self):
        """The rejection is the debugging payload — it is what the retry ladder
        feeds back, and what makes a retry look like a retry rather than the
        model inexplicably calling twice."""
        with pytest.raises(PlanRejected) as raised:
            validate_plan(["route", "prescribe"])
        assert "prescribe" in str(raised.value)

    def test_enum_members_are_accepted_as_well_as_strings(self):
        assert validate_plan([PlanStep.ROUTE]) == [ROUTE]

    def test_duplicates_collapse(self):
        assert validate_plan(["route", "route"]) == [ROUTE]


class TestSupersetRule:
    def test_booking_pulls_in_the_whole_booking_plan(self):
        """A booking intent takes the booking plan, which already contains the
        document steps. Pinned because the failure mode is silent."""
        assert validate_plan(["book"]) == [ROUTE, BOOK, DOCUMENTS, FOLLOW_UP]

    def test_the_canonical_two_intent_message_produces_route_and_book(self):
        """The PRD's named Layer-1 scenario: "I need a cardiology appointment
        next week. I also want to attach my previous ECG"."""
        plan = validate_plan(["documents", "book"])
        assert ROUTE in plan and BOOK in plan
        assert plan == [ROUTE, BOOK, DOCUMENTS, FOLLOW_UP]

    def test_the_narrow_intent_never_swallows_the_wide_one(self):
        """The reverse of the rule, stated as its own test: documents-only must
        not acquire a booking it was never asked for."""
        assert validate_plan(["documents"]) == [DOCUMENTS]
        assert BOOK not in validate_plan(["documents"])

    def test_a_doc_only_upload_skips_routing_and_booking(self):
        plan = validate_plan(["documents", "follow_up"])
        assert plan == [DOCUMENTS, FOLLOW_UP]

    def test_a_routing_question_stays_a_routing_question(self):
        assert validate_plan(["route"]) == [ROUTE]

    def test_booking_implies_routing_even_when_the_model_forgot_it(self):
        """Dependency order cannot be enforced by rejecting the plan alone: a
        model that emits [book] has not proposed something illegal, it has
        proposed something incomplete."""
        assert validate_plan(["book"])[0] is ROUTE


class TestDependencyOrder:
    def test_steps_come_back_in_canonical_order(self):
        assert validate_plan(["follow_up", "book", "route"]) == list(CANONICAL_ORDER)

    def test_routing_always_precedes_booking(self):
        plan = validate_plan(["book", "route"])
        assert plan.index(ROUTE) < plan.index(BOOK)

    def test_the_document_diff_never_precedes_routing(self):
        """The required-documents rules are per department, so the diff is
        meaningless before a department is known."""
        plan = validate_plan(["documents", "route"])
        assert plan.index(ROUTE) < plan.index(DOCUMENTS)

    def test_follow_up_is_last(self):
        assert validate_plan(["follow_up", "documents"])[-1] is FOLLOW_UP


class TestPlanProgress:
    def test_the_next_step_is_the_first_incomplete_one(self, run):
        run.plan = ["route", "book", "documents", "follow_up"]
        run.completed_steps = ["route"]
        assert next_step(run) is BOOK

    def test_a_plan_with_nothing_done_starts_at_the_beginning(self, run):
        run.plan = ["route", "book"]
        run.completed_steps = []
        assert next_step(run) is ROUTE

    def test_a_finished_plan_has_no_next_step(self, run):
        run.plan = ["route"]
        run.completed_steps = ["route"]
        assert next_step(run) is None
        assert is_plan_complete(run) is True

    def test_an_unfinished_plan_is_not_complete(self, run):
        run.plan = ["route", "book"]
        run.completed_steps = ["route"]
        assert is_plan_complete(run) is False

    def test_advancing_records_the_step(self, run):
        run.plan = ["route", "book"]
        run.completed_steps = []
        advance_plan(run, ROUTE)
        assert run.completed_steps == ["route"]
        assert next_step(run) is BOOK

    def test_advancing_twice_does_not_duplicate(self, run):
        """A retried turn must not make the plan look further along than it is."""
        run.plan = ["route", "book"]
        run.completed_steps = []
        advance_plan(run, ROUTE)
        advance_plan(run, ROUTE)
        assert run.completed_steps == ["route"]

    def test_advancing_a_step_the_plan_never_had_is_refused(self, run):
        run.plan = ["documents"]
        run.completed_steps = []
        with pytest.raises(PlanRejected):
            advance_plan(run, BOOK)

    def test_an_empty_plan_is_not_reported_complete(self, run):
        """A run whose plan never got written is not a finished run — reporting
        it complete would transition it to ``completed`` having done nothing."""
        run.plan = []
        run.completed_steps = []
        assert is_plan_complete(run) is False


class TestComplementaryAppend:
    def test_a_step_can_be_appended_to_a_live_plan(self, run):
        """The complementary class: "also, here's my old ECG" during a booking
        appends the document step. The booking survives."""
        run.plan = ["route", "book"]
        run.completed_steps = ["route"]
        append_step(run, DOCUMENTS)
        assert DOCUMENTS.value in run.plan

    def test_the_appended_plan_keeps_canonical_order(self, run):
        run.plan = ["route", "book", "follow_up"]
        append_step(run, DOCUMENTS)
        assert run.plan == ["route", "book", "documents", "follow_up"]

    def test_appending_an_existing_step_changes_nothing(self, run):
        run.plan = ["route", "book", "documents", "follow_up"]
        run.completed_steps = ["route"]
        append_step(run, DOCUMENTS)
        assert run.plan == ["route", "book", "documents", "follow_up"]

    def test_appending_does_not_reopen_completed_steps(self, run):
        run.plan = ["route", "book"]
        run.completed_steps = ["route"]
        append_step(run, DOCUMENTS)
        assert run.completed_steps == ["route"]

    def test_appending_does_not_consume_the_replan_budget(self, run):
        """A deterministic append is not a re-plan — no model was asked."""
        run.plan = ["route", "book"]
        append_step(run, DOCUMENTS)
        assert run.replan_count == 0


class TestReplanBudget:
    def test_the_first_replan_is_allowed(self, run, settings):
        record_replan(run, max_replans=settings.max_replans_per_run)
        assert run.replan_count == 1

    def test_the_second_replan_exceeds_the_budget(self, run, settings):
        """Max one re-plan per run. A loop of *successful* calls never trips a
        retry ladder, so the cap is explicit and config-driven rather than
        inherited from whatever the framework happens to do."""
        record_replan(run, max_replans=settings.max_replans_per_run)
        with pytest.raises(BudgetExceeded):
            record_replan(run, max_replans=settings.max_replans_per_run)

    def test_the_budget_is_read_from_config_not_hardcoded(self, run):
        record_replan(run, max_replans=2)
        record_replan(run, max_replans=2)
        with pytest.raises(BudgetExceeded):
            record_replan(run, max_replans=2)
        assert run.replan_count == 2

    def test_the_counter_does_not_advance_past_the_cap(self, run):
        """A counter that keeps incrementing on the failure path turns one
        exhausted budget into an ever-growing number in the audit trail."""
        record_replan(run, max_replans=1)
        with pytest.raises(BudgetExceeded):
            record_replan(run, max_replans=1)
        assert run.replan_count == 1
