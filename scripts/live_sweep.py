"""The live eval sweep. Billed, on demand, and excluded from CI.

Every phrasing from the live transcripts, replayed through ``run_workflow``
against a real provider and graded on plumbing. Run it *before* a conversational
change and *after* it, then diff the two — which is the only way a regression
like round 5's ("please reschedule my appointment to next week" worked at
7:29 PM and was reaching a staff review by 11:14 PM) becomes visible without a
human retyping five transcripts.

    python scripts/live_sweep.py --out baseline.json
    python scripts/live_sweep.py --out after.json
    python scripts/live_sweep.py --diff baseline.json after.json

``--only NAME`` runs one conversation while iterating; ``--provider`` overrides
``LLM_PROVIDER`` for the sweep (``mock`` makes a free smoke test of the harness
itself, which is worth doing once before spending money on it).

**Each conversation gets a freshly seeded database and a unique conversation
id.** Both matter. One active run per patient means a leftover run would make
the next conversation's first message a *classification* rather than a plan; and
the Coordinator's ADK session is persistent, so a re-used id hands turn 1 the
last sweep's transcript.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Nothing from `app` or `tests` is imported at module scope, and that is
# load-bearing rather than tidy. Environment loading is `setdefault`, so
# `--provider` has to reach `os.environ` *before* the first import that builds
# a provider from settings — importing the sweep module up here made
# `--provider mock` a silent no-op that billed a full run against OpenAI. It is
# the same trap the test suite documents, arriving through a CLI flag.


def _seed() -> None:
    from scripts.seed import run as seed_run

    seed_run(reset=True)


def _print_report(report: dict) -> None:
    print("")
    print("=" * 74)
    for row in report["conversations"]:
        mark = "PASS" if row["ok"] else "FAIL"
        print(f"  [{mark}] {row['name']}")
        if row["error"]:
            print(f"         ERROR {row['error']}")
        for failure in row["failures"]:
            print(f"         {failure}")
    print("=" * 74)
    print(f"LIVE SWEEP: {report['passed']}/{report['total']} conversations passed")


def _print_diff(diff: dict) -> None:
    print("")
    print("=" * 74)
    print(f"FIXED      ({len(diff['fixed'])})")
    for row in diff["fixed"]:
        print(f"  + {row['name']}")
        for line in row["was"]:
            print(f"      was: {line}")
    print(f"REGRESSED  ({len(diff['regressed'])})")
    for row in diff["regressed"]:
        print(f"  - {row['name']}")
        for line in row["now"]:
            print(f"      now: {line}")
    print(f"CHANGED    ({len(diff['changed'])})")
    for row in diff["changed"]:
        print(f"  ~ {row['name']}")
        for line in row.get("was", []):
            print(f"      was: {line}")
        for line in row.get("now", []):
            print(f"      now: {line}")
    print(f"UNCHANGED  ({len(diff['unchanged'])})")
    print("=" * 74)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write the report here as JSON")
    parser.add_argument("--only", help="run just this conversation")
    parser.add_argument("--provider", help="override LLM_PROVIDER for the sweep")
    parser.add_argument(
        "--diff", nargs=2, metavar=("BEFORE", "AFTER"), help="diff two reports and exit"
    )
    args = parser.parse_args()

    if args.provider:
        # Before any app import builds a provider from settings.
        os.environ["LLM_PROVIDER"] = args.provider

    from tests.evals.live.sweep import (
        as_report,
        diff_reports,
        load_conversations,
        run_conversation,
    )

    if args.diff:
        before = json.loads(Path(args.diff[0]).read_text(encoding="utf-8"))
        after = json.loads(Path(args.diff[1]).read_text(encoding="utf-8"))
        _print_diff(diff_reports(before, after))
        return 0

    from app.config import get_settings

    settings = get_settings()
    scripts = load_conversations()
    if args.only:
        scripts = [s for s in scripts if s["name"] == args.only]
        if not scripts:
            print(f"No conversation named {args.only!r}.")
            return 2

    stamp = uuid.uuid4().hex[:8]
    print(f"Provider: {settings.llm_provider}")
    print(f"Conversations: {len(scripts)}   sweep id: {stamp}")

    outcomes = []
    for index, script in enumerate(scripts, start=1):
        print(f"\n[{index}/{len(scripts)}] {script['name']}", flush=True)
        _seed()
        outcome = run_conversation(script, session_prefix=f"live-{stamp}")
        outcomes.append(outcome)
        print("      " + ("PASS" if outcome.ok else "FAIL"), flush=True)

    report = as_report(outcomes)
    report["provider"] = settings.llm_provider
    report["sweep_id"] = stamp
    _print_report(report)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.out}")

    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
