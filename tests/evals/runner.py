"""The Layer-1 scenario runner.

A scenario is a JSON file: a patient, a list of turns, and what each turn must
produce. It runs against the real orchestrator with the real tools and the real
database, under the mock provider — no network, no billing, deterministic.

**Why this exists in Phase 4 rather than later.** "Before you change it, pin
it" starts binding the moment there is behaviour worth pinning, and everything
worth pinning — the state machine, the mapping classes, the confirmation
reader, the superset rule — is built here. A runner that arrived a phase later
would leave exactly the work the rule was written for unpinned.

**The expectation vocabulary is deliberately small.** Every key below is
something a scenario can be wrong about in a way that matters; a scenario
language that can express anything ends up expressing nothing anybody checks.

Every scenario also asserts the trace parses, always, without asking. A turn
that produced the right answer and an unreadable trace is half a pass, and the
half that is missing is the half you need when something else breaks.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.db import SessionLocal
from app.models import (
    Appointment,
    DocumentStatus,
    FollowUpTask,
    FollowUpTaskStatus,
    PatientDocument,
    PatientProfile,
    User,
    WorkflowRun,
)
from app.orchestrator import run_workflow
from app.trace import assert_well_formed

SCENARIO_DIR = Path(__file__).parent / "scenarios"

#: Everything a turn's ``expect`` block may say. Anything else is a typo, and a
#: silently-ignored typo is an assertion that never runs.
KNOWN_EXPECTATIONS = frozenset(
    {
        "status",
        "plan",
        "steps_run",
        "message_class",
        "reply_contains",
        "reply_excludes",
        "appointments_created",
        "cancellation_reason",
        "budget_exhausted",
        "request_text_unchanged",
        "run_state_unchanged",
        "escalations",
        # --- Phase 5: safety, escalation, confirmation, documents ---
        "escalation_kind",
        "escalation_occurrences",
        "runs_for_patient",
        "non_answer_count",
        "reply_author",
        "documents_flagged",
        "open_tasks_contain",
    }
)


@dataclass
class Failure:
    turn: int
    detail: str


@dataclass
class ScenarioResult:
    name: str
    failures: list[Failure] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def load_scenarios() -> list[dict]:
    scenarios = []
    for path in sorted(SCENARIO_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        scenarios.append(data)
    return scenarios


def _snapshot(session, run_id: int | None) -> tuple[str, str] | None:
    if run_id is None:
        return None
    run = session.get(WorkflowRun, run_id)
    if run is None:
        return None
    return run.request_text, json.dumps(run.state, sort_keys=True)


def _appointment_count(session, patient_id: int) -> int:
    return (
        session.query(Appointment)
        .filter(Appointment.patient_id == patient_id)
        .count()
    )


def run_scenario(scenario: dict) -> ScenarioResult:
    """Run one scenario's turns in order and collect every failure.

    Collects rather than stops at the first: when a scenario breaks it is
    usually more useful to see all four things that changed than the first one
    alphabetically.
    """
    result = ScenarioResult(name=scenario["name"])
    session_id = f"eval-{scenario['name']}"

    session = SessionLocal()
    try:
        user = session.query(User).filter(User.email == scenario["patient"]).one()
        profile = (
            session.query(PatientProfile)
            .filter(PatientProfile.user_id == user.id)
            .one()
        )
        patient_id = profile.id
        # One baseline, taken once, for this patient only. The seed ships a
        # booked appointment on purpose, so an absolute count would report a
        # scenario as having created something it inherited.
        baseline = _appointment_count(session, patient_id)
    finally:
        session.close()

    previous_run_id: int | None = None
    for index, turn in enumerate(scenario["turns"], start=1):
        expect = turn.get("expect", {})
        unknown = set(expect) - KNOWN_EXPECTATIONS
        if unknown:
            result.failures.append(
                Failure(index, f"unknown expectation key(s): {sorted(unknown)}")
            )
            continue

        session = SessionLocal()
        try:
            before = _snapshot(session, previous_run_id)
        finally:
            session.close()

        outcome = asyncio.run(run_workflow(user, turn["message"], session_id))
        previous_run_id = outcome.run_id or previous_run_id

        session = SessionLocal()
        try:
            created = _appointment_count(session, patient_id) - baseline
            _check(
                result, index, expect, outcome, session, before, created, patient_id
            )
            assert_well_formed(session)
        finally:
            session.close()

    return result


def _check(result, index, expect, outcome, session, before, created, patient_id) -> None:
    def fail(detail: str) -> None:
        result.failures.append(Failure(index, detail))

    if "status" in expect and outcome.status != expect["status"]:
        fail(f"status: expected {expect['status']!r}, got {outcome.status!r}")

    if "plan" in expect and outcome.plan != expect["plan"]:
        fail(f"plan: expected {expect['plan']}, got {outcome.plan}")

    if "steps_run" in expect and outcome.steps_run != expect["steps_run"]:
        fail(f"steps_run: expected {expect['steps_run']}, got {outcome.steps_run}")

    if "message_class" in expect:
        actual = outcome.message_class.value if outcome.message_class else None
        if actual != expect["message_class"]:
            fail(f"message_class: expected {expect['message_class']!r}, got {actual!r}")

    lowered = outcome.reply.lower()
    for needle in expect.get("reply_contains", []):
        if needle.lower() not in lowered:
            fail(f"reply is missing {needle!r}: {outcome.reply[:160]!r}")
    for needle in expect.get("reply_excludes", []):
        if needle.lower() in lowered:
            fail(f"reply must not contain {needle!r}: {outcome.reply[:160]!r}")

    if "budget_exhausted" in expect and outcome.budget_exhausted != expect[
        "budget_exhausted"
    ]:
        fail(f"budget_exhausted: expected {expect['budget_exhausted']}")

    if "appointments_created" in expect:
        if created != expect["appointments_created"]:
            fail(
                f"appointments_created: expected {expect['appointments_created']}, "
                f"got {created}"
            )

    if "cancellation_reason" in expect and outcome.run_id:
        run = session.get(WorkflowRun, outcome.run_id)
        if run.cancellation_reason != expect["cancellation_reason"]:
            fail(
                f"cancellation_reason: expected {expect['cancellation_reason']!r}, "
                f"got {run.cancellation_reason!r}"
            )

    if "escalations" in expect and outcome.run_id:
        run = session.get(WorkflowRun, outcome.run_id)
        if len(run.escalations) != expect["escalations"]:
            fail(f"escalations: expected {expect['escalations']}, got {len(run.escalations)}")

    if "escalation_kind" in expect and outcome.run_id:
        run = session.get(WorkflowRun, outcome.run_id)
        kinds = sorted(e.kind.value for e in run.escalations)
        if expect["escalation_kind"] not in kinds:
            fail(f"escalation_kind: expected {expect['escalation_kind']!r} in {kinds}")

    if "escalation_occurrences" in expect and outcome.run_id:
        run = session.get(WorkflowRun, outcome.run_id)
        counts = [e.occurrence_count for e in run.escalations]
        if expect["escalation_occurrences"] not in counts:
            fail(
                f"escalation_occurrences: expected "
                f"{expect['escalation_occurrences']}, got {counts}"
            )

    # The dedup check that matters: repeats must not each spawn a run. An
    # occurrence count alone would pass while five runs piled up beside it.
    if "runs_for_patient" in expect:
        total = (
            session.query(WorkflowRun)
            .filter(WorkflowRun.patient_id == patient_id)
            .count()
        )
        if total != expect["runs_for_patient"]:
            fail(f"runs_for_patient: expected {expect['runs_for_patient']}, got {total}")

    if "non_answer_count" in expect and outcome.run_id:
        run = session.get(WorkflowRun, outcome.run_id)
        if run.non_answer_count != expect["non_answer_count"]:
            fail(
                f"non_answer_count: expected {expect['non_answer_count']}, "
                f"got {run.non_answer_count}"
            )

    if "reply_author" in expect:
        actual = outcome.author.value if outcome.author else None
        if actual != expect["reply_author"]:
            fail(f"reply_author: expected {expect['reply_author']!r}, got {actual!r}")

    if "documents_flagged" in expect:
        flagged = (
            session.query(PatientDocument)
            .filter(
                PatientDocument.patient_id == patient_id,
                PatientDocument.status == DocumentStatus.FLAGGED,
            )
            .count()
        )
        if flagged != expect["documents_flagged"]:
            fail(f"documents_flagged: expected {expect['documents_flagged']}, got {flagged}")

    if "open_tasks_contain" in expect:
        outstanding = [
            item
            for task in session.query(FollowUpTask)
            .filter(
                FollowUpTask.patient_id == patient_id,
                FollowUpTask.status == FollowUpTaskStatus.OPEN,
            )
            .all()
            for item in (task.details or {}).get("missing", [])
        ]
        for needle in expect["open_tasks_contain"]:
            if needle not in outstanding:
                fail(f"open task missing {needle!r}; outstanding = {outstanding}")

    # The byte-identical checks. These are the point of the off-topic scenario:
    # not "the reply was polite" but "nothing moved".
    after = _snapshot(session, outcome.run_id)
    if expect.get("request_text_unchanged") and before and after:
        if before[0] != after[0]:
            fail(f"request_text changed: {before[0]!r} -> {after[0]!r}")
    if expect.get("run_state_unchanged") and before and after:
        if before[1] != after[1]:
            fail(f"run state changed: {before[1]} -> {after[1]}")


__all__ = [
    "KNOWN_EXPECTATIONS",
    "ScenarioResult",
    "load_scenarios",
    "run_scenario",
]
