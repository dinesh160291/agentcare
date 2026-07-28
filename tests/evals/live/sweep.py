"""Replay the scripted conversations against a live provider and grade them.

**Why this exists.** Round 5 found a *regression*: "please reschedule my
appointment to next week" worked at 7:29 PM and was routing to a staff review
by 11:14 PM the same day. Nothing failed in between — the mock suite was green
throughout, because the mock is better-behaved than ``gpt-4o-mini`` in exactly
the places the live defects live. A conversational layer whose only automated
evidence comes from an understudy has no regression detection at all; the last
five rounds of findings all arrived as screenshots of a human talking to the
system, which finds defects beautifully and finds *re*-defects never.

**What is graded.** Plumbing, mostly: run status, the plan, the message class,
the appointments delta, the proposal state, and — new for this round — whether
the turn ended on the *same run* it started on. Those are facts a live model
cannot legitimately vary. Reply text is checked only for required facts (a
reference code, a department name) and for forbidden ones (a referral to staff
that nothing referred, a claim that an earlier request was closed). Anything
tighter would be grading the model's prose, which is the one thing a live sweep
must not do — it would go red on a rewording and teach everyone to ignore it.

The grader is ``tests.evals.runner``, unchanged and shared with the mock suite.
That is deliberate: two graders would drift, and a live result that cannot be
compared with the Layer-1 result is half a signal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tests.evals.runner import KNOWN_EXPECTATIONS, run_scenario

CONVERSATIONS = Path(__file__).parent / "conversations.json"


@dataclass
class ConversationOutcome:
    """One scripted conversation's verdict, in a shape two sweeps can diff."""

    name: str
    ok: bool
    failures: list[str] = field(default_factory=list)
    #: An exception escaping a turn. The conversation stops there — the world
    #: is in an unknown state and grading the rest would be grading noise — but
    #: the sweep carries on to the next one.
    error: str | None = None


def load_conversations() -> list[dict]:
    scripts = json.loads(CONVERSATIONS.read_text(encoding="utf-8"))
    for script in scripts:
        for turn in script["turns"]:
            unknown = set(turn.get("expect", {})) - KNOWN_EXPECTATIONS
            if unknown:
                # Loudly, and before a single billed call: a typo'd key is an
                # assertion that silently never runs, and a live sweep that
                # quietly checks less than it claims is worse than none.
                raise ValueError(
                    f"{script['name']}: unknown expectation key(s) {sorted(unknown)}"
                )
    return scripts


def run_conversation(script: dict, *, session_prefix: str) -> ConversationOutcome:
    """One conversation, replayed and graded. Never raises."""
    try:
        result = run_scenario(script, session_prefix=session_prefix)
    except Exception as exc:  # noqa: BLE001 - reported, and the sweep continues
        return ConversationOutcome(
            name=script["name"], ok=False, error=f"{type(exc).__name__}: {exc}"
        )
    return ConversationOutcome(
        name=script["name"],
        ok=result.ok,
        failures=[f"turn {f.turn}: {f.detail}" for f in result.failures],
    )


def as_report(outcomes: list[ConversationOutcome]) -> dict:
    """The machine-readable half. Two of these are what a diff is made of."""
    return {
        "conversations": [asdict(outcome) for outcome in outcomes],
        "passed": sum(1 for outcome in outcomes if outcome.ok),
        "total": len(outcomes),
    }


def diff_reports(before: dict, after: dict) -> dict:
    """What changed between two sweeps, per conversation.

    The three categories that matter are named separately because they mean
    different things: a **fix** is the point of the exercise, a **regression**
    is the reason the exercise exists, and a conversation whose failure list
    merely *changed* is one that moved without being fixed.
    """
    before_by_name = {row["name"]: row for row in before["conversations"]}
    after_by_name = {row["name"]: row for row in after["conversations"]}

    fixed, regressed, changed, unchanged = [], [], [], []
    for name in sorted(set(before_by_name) | set(after_by_name)):
        was, now = before_by_name.get(name), after_by_name.get(name)
        if was is None or now is None:
            changed.append({"name": name, "detail": "present in only one sweep"})
            continue
        if not was["ok"] and now["ok"]:
            fixed.append({"name": name, "was": was["failures"] or [was["error"]]})
        elif was["ok"] and not now["ok"]:
            regressed.append({"name": name, "now": now["failures"] or [now["error"]]})
        elif was["failures"] != now["failures"] or was["error"] != now["error"]:
            changed.append(
                {
                    "name": name,
                    "was": was["failures"] or [was["error"]],
                    "now": now["failures"] or [now["error"]],
                }
            )
        else:
            unchanged.append(name)

    return {
        "fixed": fixed,
        "regressed": regressed,
        "changed": changed,
        "unchanged": unchanged,
    }


__all__ = [
    "ConversationOutcome",
    "as_report",
    "diff_reports",
    "load_conversations",
    "run_conversation",
]
