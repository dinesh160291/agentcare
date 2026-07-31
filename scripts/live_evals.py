"""Layer-2 evals. Billed, on demand, and excluded from CI.

The scenario suite against a real provider, with appointment facts graded
exactly against the database row and repeated N times — see
``tests/evals/live/layer2.py`` for what each grading is for.

    python scripts/live_evals.py --provider mock      # free rehearsal, run this first
    python scripts/live_evals.py --out layer2.json    # billed

``--provider mock`` has to come back green before a billed run is worth
spending, and not only to save money: without it, "the model got it wrong" and
"the expectation was never satisfiable" look identical from the report, and the
second is far more common. ``--only NAME`` runs one case while iterating, and
``-n`` overrides the repetition count for a smoke test.

Nothing from ``app`` or ``tests`` is imported at module scope, and that is
load-bearing rather than tidy: environment loading is ``setdefault``, so
``--provider`` must reach ``os.environ`` *before* the first import that builds a
provider from settings. The live sweep learned this the expensive way —
module-scope imports made ``--provider mock`` a silent no-op that billed a full
run against OpenAI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _print_report(report: dict) -> None:
    print("")
    print("=" * 74)
    for row in report["cases"]:
        mark = "PASS" if row["ok"] else "FAIL"
        rate = f"{row['passes']}/{row['runs']}"
        print(f"  [{mark}] {row['name']}  ({row['kind']}, {rate}, bar {row['bar']:.0%})")
        for failure in row["failures"]:
            print(f"         {failure}")
    print("=" * 74)
    print(f"LAYER 2: {report['passed']}/{report['total']} cases passed")
    if not report["total"]:
        print("  NO CASES RAN — check --only against the names in FACT_CASES/SAFETY_CASES")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write the report here as JSON")
    parser.add_argument("--only", help="run just this scenario")
    parser.add_argument("--provider", help="override LLM_PROVIDER for this run")
    parser.add_argument(
        "-n", "--repetitions", type=int, default=None, help="override N (default 5)"
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=None,
        help="seconds between runs; 0 disables pacing (mock rehearsals only)",
    )
    args = parser.parse_args()

    if args.provider:
        # Before any app import builds a provider from settings.
        os.environ["LLM_PROVIDER"] = args.provider

    from app.config import get_settings
    from tests.evals.live.layer2 import PACE_SECONDS, REPETITIONS, as_report, run_layer2

    settings = get_settings()
    repetitions = args.repetitions or REPETITIONS
    pace = PACE_SECONDS if args.pace is None else args.pace

    stamp = uuid.uuid4().hex[:8]
    print(f"Provider: {settings.llm_provider}")
    print(f"N={repetitions}   pace={pace}s   run id: {stamp}")

    outcomes = run_layer2(
        repetitions=repetitions,
        pace=pace,
        session_prefix=f"l2-{stamp}",
        only=args.only,
    )

    report = as_report(outcomes)
    report["provider"] = settings.llm_provider
    report["run_id"] = stamp
    report["repetitions"] = repetitions
    _print_report(report)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.out}")

    # `passed == total` is True for 0 == 0, so a typo'd `--only` would otherwise
    # exit green having graded nothing — the exact shape of a check that passes
    # regardless. An empty run is a failure of the run, not a clean sweep.
    if not report["total"]:
        return 2
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
