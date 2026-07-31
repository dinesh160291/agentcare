"""The Layer-2 harness, checked under the mock.

Layer 2 itself is billed and never runs in CI. That is exactly why it needs
this: **a runner nobody checks is a runner that can quietly stop checking**, and
a billed one is checked least often of all. An eval harness that has rotted is
worse than no harness, because it is discovered at the moment it was supposed to
be relied on — the evening before a submission, with a live API bill attached.

So the manifest is validated against the scenario files, and the fact grader is
exercised both ways: it passes on the appointment the reply is about, and it
**fails** on the one it is not. The second half is the point. Since round 4 the
receipt is assembled from rows, so a fact grader aimed at the right row passes
by construction — the claim worth pinning is that aiming it at the wrong row
goes red, because the live defect this exists to catch is a receipt describing
one appointment while a different one moved.
"""

from __future__ import annotations

import pytest

from tests.evals.live.facts import (
    appointment_facts,
    find_appointment,
    grade,
    grade_absent,
)
from tests.evals.live.layer2 import (
    FACT_CASES,
    SAFETY_CASES,
    SAFETY_PASS_RATE,
    CaseOutcome,
    grade_fact_case,
)
from tests.evals.runner import load_scenarios, run_scenario

SCENARIOS = {scenario["name"]: scenario for scenario in load_scenarios()}
BOOKING = next(case for case in FACT_CASES if case.scenario == "booking-happy-path")
CHOICE = next(
    case for case in FACT_CASES if case.scenario == "choosing-which-appointment"
)


class TestTheManifest:
    """Names in the manifest are strings, and a string can be wrong silently."""

    def test_every_fact_case_names_a_real_scenario(self):
        for case in FACT_CASES:
            assert case.scenario in SCENARIOS, f"no scenario named {case.scenario!r}"

    def test_every_safety_case_names_a_real_scenario(self):
        for name in SAFETY_CASES:
            assert name in SCENARIOS, f"no scenario named {name!r}"

    def test_every_fact_case_points_at_a_turn_that_exists(self):
        """An out-of-range turn would report 'the scenario stopped early' on
        every run — a permanent red that looks like a model failure."""
        for case in FACT_CASES:
            turns = len(SCENARIOS[case.scenario]["turns"])
            assert 1 <= case.turn <= turns, (
                f"{case.scenario}: turn {case.turn} of {turns}"
            )

    def test_every_fact_case_says_why(self):
        """Same rule as a scenario's `pins` note, for the same reason: the first
        time one of these fails, the question is what it was protecting."""
        for case in FACT_CASES:
            assert case.why.strip(), f"{case.scenario} has no 'why'"

    def test_the_safety_bar_is_not_unanimity(self):
        """If this ever becomes 1.0 the safety cases stop being gradable — the
        model layer is *meant* to fire early, so an occasional extra referral is
        the guard working, not a regression."""
        assert 0.0 < SAFETY_PASS_RATE < 1.0


class TestTheSubjectSelector:
    """One selector, chosen deliberately."""

    def test_two_selectors_are_refused(self, seeded_db):
        """The first version fell back from one selector to the next and graded
        the seed's Cardiology appointment as though it were the one the booking
        scenario created."""
        with pytest.raises(ValueError):
            find_appointment(
                seeded_db,
                "asha.patient@example.invalid",
                reference="AC-000001",
                newest=True,
            )

    def test_no_selector_is_refused(self, seeded_db):
        with pytest.raises(ValueError):
            find_appointment(seeded_db, "asha.patient@example.invalid")


class TestTheFactGrader:
    """Both directions, on one real run."""

    @pytest.fixture
    def booked(self, seeded_db):
        seeded_db.commit()
        return run_scenario(SCENARIOS["booking-happy-path"], session_prefix="l2-harness")

    def test_the_right_appointment_grades_clean(self, booked):
        assert grade_fact_case(BOOKING, booked) == []

    def test_the_wrong_appointment_goes_red(self, booked):
        """The falsification. Aimed at the appointment the seed shipped rather
        than the one this run created — the reply is about the latter, so every
        field that differs must be reported."""
        misaimed = type(BOOKING)(
            scenario=BOOKING.scenario,
            turn=BOOKING.turn,
            fields=BOOKING.fields,
            reference="AC-000001",
            why="sabotage",
        )
        misses = grade_fact_case(misaimed, booked)
        assert misses, "grading the wrong appointment passed — the grader reads nothing"
        assert any("reference" in miss for miss in misses)

    def test_a_forbidden_reference_is_caught(self, seeded_db):
        """`grade_absent` is the only thing standing between 'names the right
        appointment' and 'does not name the wrong one'."""
        appointment = find_appointment(
            seeded_db, "asha.patient@example.invalid", reference="AC-000001"
        )
        facts = appointment_facts(appointment)
        reply = f"Your appointment {facts.reference} is confirmed."
        assert grade_absent(reply, facts, ("reference",))
        assert grade_absent("nothing to see here", facts, ("reference",)) == []

    def test_a_field_the_row_cannot_supply_is_a_miss(self, seeded_db):
        """An empty expected value must fail loudly rather than match anything.
        `"" in reply` is True for every reply ever written."""
        appointment = find_appointment(
            seeded_db, "asha.patient@example.invalid", reference="AC-000001"
        )
        facts = appointment_facts(appointment)
        blank = type(facts)(
            reference="", department=facts.department, day=facts.day,
            time=facts.time, doctor=facts.doctor,
        )
        assert grade("any reply at all", blank, ("reference",))


class TestTheOutcomeArithmetic:
    """The bar is the whole grading; an off-by-one here passes a failing sweep."""

    def test_a_fact_case_needs_every_run(self):
        outcome = CaseOutcome(name="x", kind="fact", runs=5, passes=4)
        assert not outcome.ok
        outcome.passes = 5
        assert outcome.ok

    def test_a_safety_case_clears_the_threshold(self):
        outcome = CaseOutcome(name="x", kind="safety", runs=5, passes=4)
        assert outcome.ok
        outcome.passes = 3
        assert not outcome.ok

    def test_a_case_that_never_ran_is_not_a_pass(self):
        """Zero runs would otherwise divide to a rate of 0.0 and read as a
        failure by luck rather than by rule — and a case skipped for a typo'd
        `--only` must not be able to report anything else."""
        assert not CaseOutcome(name="x", kind="fact").ok
