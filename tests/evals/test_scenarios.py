"""Layer-1 scenarios, run as tests.

Each JSON file under ``scenarios/`` is one pinned behaviour. They run against
the real orchestrator under the mock provider: deterministic, offline, and
exercising the same skeleton the shipped system uses.

From here on, **every phase adds its own scenarios as it ships** — the same
discipline as tests, for the same reason. A scenario written after the fact
pins whatever the code currently does, which is not what pinning is for.
"""

from __future__ import annotations

import pytest

from tests.evals.runner import KNOWN_EXPECTATIONS, load_scenarios, run_scenario

SCENARIOS = load_scenarios()


def _ids(scenarios):
    return [scenario["name"] for scenario in scenarios]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_ids(SCENARIOS))
def test_scenario(seeded_db, scenario):
    seeded_db.commit()
    result = run_scenario(scenario)
    if not result.ok:
        detail = "\n".join(
            f"  turn {failure.turn}: {failure.detail}" for failure in result.failures
        )
        pytest.fail(f"scenario {scenario['name']} failed:\n{detail}")


class TestTheHarnessItself:
    """A runner nobody checks is a runner that can quietly stop checking."""

    def test_scenarios_were_found(self):
        assert SCENARIOS, "no scenario files loaded"

    def test_every_scenario_says_what_it_pins(self):
        """A scenario without a stated reason becomes unmaintainable the first
        time it fails and nobody remembers why it existed."""
        for scenario in SCENARIOS:
            assert scenario.get("pins"), f"{scenario['name']} has no 'pins' note"

    def test_every_expectation_key_is_known(self):
        """A typo'd key would otherwise be an assertion that silently never
        runs — a scenario passing because it checks nothing."""
        for scenario in SCENARIOS:
            for turn in scenario["turns"]:
                unknown = set(turn.get("expect", {})) - KNOWN_EXPECTATIONS
                assert not unknown, f"{scenario['name']}: unknown keys {sorted(unknown)}"

    def test_scenario_names_are_unique(self):
        names = [scenario["name"] for scenario in SCENARIOS]
        assert len(names) == len(set(names))

    def test_a_broken_expectation_actually_fails(self, seeded_db):
        """Distrust green: the harness has to be able to fail.

        A runner that reports success unconditionally passes every scenario
        ever written, and would have done so all along.
        """
        seeded_db.commit()
        sabotaged = {
            "name": "deliberately-wrong",
            "pins": "proves the harness can fail",
            "patient": "asha.patient@example.invalid",
            "turns": [
                {
                    "message": "I need a cardiology appointment next week",
                    "expect": {"status": "completed"},
                }
            ],
        }
        result = run_scenario(sabotaged)
        assert not result.ok
