"""The golden runner.

For each case: seed a fresh database with the clock frozen to the anchor, call
the real tool function, and diff the result against its blessed JSON file.

Re-bless intentional changes with::

    python -m pytest tests/golden --update-golden

That rewrites the JSON, and the change then shows up as a reviewable git diff —
which is the point. A golden file that can only be regenerated silently is a
golden file nobody reads.

Checksums are the one value scrubbed before comparison: they are asserted
directly in the unit tests, and pinning them here would make every golden file
churn whenever a fixture's wording changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import clock
from app.db import SessionLocal, create_all, drop_all
from tests.golden.cases import ANCHOR, ANCHOR_MOMENT, CASES

EXPECTED_DIR = Path(__file__).parent / "expected"


def _normalise(value):
    """Replace values that are stable in meaning but noisy in text."""
    if isinstance(value, dict):
        return {
            key: ("<checksum>" if key == "checksum" and isinstance(val, str) else _normalise(val))
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    return value


def _run_case(name: str):
    """Seed a fresh database at the anchor and run one case against it."""
    clock.freeze(ANCHOR_MOMENT)
    drop_all()
    create_all()

    from scripts.seed import seed

    session = SessionLocal()
    try:
        seed(session, anchor=ANCHOR)
        session.commit()
        return _normalise(CASES[name](session))
    finally:
        session.close()
        clock.unfreeze()


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden(name: str, request):
    update = request.config.getoption("--update-golden")
    actual = _run_case(name)
    path = EXPECTED_DIR / f"{name}.json"

    if update:
        EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"re-blessed {path.name}")

    assert path.exists(), (
        f"No golden file for '{name}'. Run: pytest tests/golden --update-golden"
    )
    expected = json.loads(path.read_text(encoding="utf-8"))

    assert actual == expected, (
        f"'{name}' drifted from its golden file.\n"
        f"If this change is intended, re-bless with --update-golden and review the diff."
    )


def test_every_case_has_a_golden_file():
    """A case with no blessed output is a test that silently proves nothing."""
    missing = [name for name in CASES if not (EXPECTED_DIR / f"{name}.json").exists()]
    assert missing == [], f"cases without golden files: {missing}"


def test_no_orphan_golden_files():
    """A golden file whose case was deleted is dead weight that still reads as
    coverage."""
    if not EXPECTED_DIR.exists():
        return
    orphans = [p.stem for p in EXPECTED_DIR.glob("*.json") if p.stem not in CASES]
    assert orphans == [], f"golden files with no case: {orphans}"


def test_the_runner_is_deterministic():
    """Two runs of the same case must agree, or every golden file is a
    coin flip that happened to land the same way when it was blessed."""
    name = "slots_cardiology_next_week"
    assert _run_case(name) == _run_case(name)
