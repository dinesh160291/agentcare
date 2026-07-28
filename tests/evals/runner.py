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
from app.models import AuditEvent
from app.orchestrator import apply_patient_action, run_workflow
from app.tools.tasks import list_open_escalations
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
        # --- Phase 6a: the other two appointment verbs ---
        # What happened to an appointment that already existed. The booking
        # scenarios could get by with `appointments_created`, because a booking
        # is the only verb that makes a row; reschedule and cancel change one,
        # and a count cannot see that.
        "appointment_status",
        # Whether a proposal is still outstanding. A declined cancellation that
        # leaves the proposal confirmable is a loop with no exit, and the
        # status alone does not show it.
        "proposed_action",
        # How many times an action was audited for this patient. A double-click
        # that commits twice leaves both rows behind; the appointment's *final*
        # state cannot see the intermediate one, and the whole point of the
        # bug was that the second commit looked like a success.
        "audit_actions",
        # The staff queue, counted across every run rather than the current
        # one. A superseded run's escalation is invisible to `escalations`,
        # which reads the run the turn ended on — and an abandoned request
        # sitting in the queue is exactly what that blind spot hid.
        "open_escalations",
        # --- Round 5: refinement must not replace the run it refines ---
        # Whether this turn ended on the *same* run as the previous one. A
        # supersede is invisible to every key above when the replacement
        # reaches the same status with the same proposal: "pending_confirmation
        # again" is exactly what a run that was destroyed and rebuilt looks
        # like. `runs_for_patient` counts them but only over a whole scenario;
        # this names the turn that did it.
        "same_run",
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


def run_scenario(scenario: dict, *, session_prefix: str = "eval") -> ScenarioResult:
    """Run one scenario's turns in order and collect every failure.

    Collects rather than stops at the first: when a scenario breaks it is
    usually more useful to see all four things that changed than the first one
    alphabetically.

    ``session_prefix`` exists for the live sweep. The Coordinator's ADK session
    is *persistent* and keyed by the conversation id, so replaying the same
    scenario name twice against the same session store would hand the second
    run the first one's transcript — turn 1 arriving with history is not turn 1.
    Under the mock that is harmless and under a live provider it is the
    difference between a sweep and a re-read of the last sweep.
    """
    result = ScenarioResult(name=scenario["name"])
    session_id = f"{session_prefix}-{scenario['name']}"

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

        # A turn is either something the patient typed or a button they pressed.
        # The two are separate front doors and are only distinguishable in the
        # trace, so a scenario about the buttons has to press them: posting the
        # word "confirm" through the message door reads the exact token and
        # commits identically, which would leave the scenario describing the
        # code rather than checking it.
        if "action" in turn:
            outcome = asyncio.run(
                apply_patient_action(user, turn["action"], session_id)
            )
        else:
            outcome = asyncio.run(run_workflow(user, turn["message"], session_id))
        prior_run_id = previous_run_id
        previous_run_id = outcome.run_id or previous_run_id

        session = SessionLocal()
        try:
            created = _appointment_count(session, patient_id) - baseline
            _check(
                result,
                index,
                expect,
                outcome,
                session,
                before,
                created,
                patient_id,
                prior_run_id,
            )
            assert_well_formed(session)
        finally:
            session.close()

    return result


def _check(
    result, index, expect, outcome, session, before, created, patient_id, prior_run_id
) -> None:
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

    if "appointment_status" in expect:
        for appointment_id, expected_status in expect["appointment_status"].items():
            appointment = session.get(Appointment, int(appointment_id))
            actual = appointment.status.value if appointment else None
            if actual != expected_status:
                fail(
                    f"appointment_status[{appointment_id}]: expected "
                    f"{expected_status!r}, got {actual!r}"
                )

    if "proposed_action" in expect and outcome.run_id:
        run = session.get(WorkflowRun, outcome.run_id)
        actual = run.proposed_action.value if run.proposed_action else None
        if actual != expect["proposed_action"]:
            fail(
                f"proposed_action: expected {expect['proposed_action']!r}, "
                f"got {actual!r}"
            )

    if "same_run" in expect and prior_run_id is not None:
        same = outcome.run_id == prior_run_id
        if same != expect["same_run"]:
            fail(
                f"same_run: expected {expect['same_run']}, got {same} "
                f"(run {prior_run_id} -> {outcome.run_id})"
            )

    if "open_escalations" in expect:
        queued = len(list_open_escalations(session))
        if queued != expect["open_escalations"]:
            fail(f"open_escalations: expected {expect['open_escalations']}, got {queued}")

    if "audit_actions" in expect:
        for action, expected_count in expect["audit_actions"].items():
            actual = (
                session.query(AuditEvent)
                .filter(AuditEvent.action == action)
                .count()
            )
            if actual != expected_count:
                fail(
                    f"audit_actions[{action}]: expected {expected_count}, got {actual}"
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
