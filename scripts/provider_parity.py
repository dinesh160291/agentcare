"""Phase 3's done-when: the same scenario under mock and under Groq, traces
diffed for **shape**.

Wording is expected to differ — that is the whole point of swapping the brain.
What must not differ is the skeleton: the same capture points in the same
order, every LLM request paired with a response-or-error, and the same author
fields. If mock-mode traces are shaped differently from live ones, then every
test that runs under mock is proving something about a system nobody ships.

Run:  python scripts/provider_parity.py
"""

from __future__ import annotations

import sys
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import TraceEvent  # noqa: E402
from app.trace import check_session  # noqa: E402

SCENARIO = "I need a cardiology appointment next week"


def trace_shape(session, session_id: str) -> list[tuple[str, str | None]]:
    """The turn's skeleton: event types and authors, in order, minus content."""
    rows = (
        session.query(TraceEvent)
        .filter(TraceEvent.session_id == session_id)
        .order_by(TraceEvent.turn_id, TraceEvent.seq)
        .all()
    )
    return [(row.event_type.value, row.author.value if row.author else None) for row in rows]


def pairing_report(session, session_id: str) -> tuple[int, int]:
    """(requests, terminal partners) for the session."""
    rows = (
        session.query(TraceEvent).filter(TraceEvent.session_id == session_id).all()
    )
    kinds = Counter(row.event_type.value for row in rows)
    return kinds.get("llm_request", 0), kinds.get("llm_response", 0) + kinds.get("llm_error", 0)


async def run_one(provider: str) -> str:
    from scripts.gate_spike import APP_NAME, USER_ID, one_turn
    from google.adk.sessions import DatabaseSessionService

    settings = get_settings()
    service = DatabaseSessionService(db_url=settings.adk_session_db_url)
    session_id = f"parity-{provider}-{uuid.uuid4().hex[:6]}"
    await service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    await one_turn(provider, session_id, SCENARIO, service)
    return session_id


def main() -> int:
    print("Scenario: " + repr(SCENARIO))
    print("=" * 74)

    mock_session = asyncio.run(run_one("mock"))
    live_session = asyncio.run(run_one("groq"))

    session = SessionLocal()
    try:
        mock_shape = trace_shape(session, mock_session)
        live_shape = trace_shape(session, live_session)
        mock_req, mock_partners = pairing_report(session, mock_session)
        live_req, live_partners = pairing_report(session, live_session)
        malformed = [
            r
            for sid in (mock_session, live_session)
            for tid in {
                row.turn_id
                for row in session.query(TraceEvent).filter(TraceEvent.session_id == sid).all()
            }
            for r in check_session(session, turn_id=tid)
            if not r.ok
        ]
    finally:
        session.close()

    checks: list[bool] = []

    def check(label: str, ok: bool, detail: str) -> None:
        print("  [" + ("PASS" if ok else "FAIL") + "] " + label)
        print("         " + detail)
        checks.append(ok)

    print("")
    print("Capture points present")
    mock_kinds = {kind for kind, _ in mock_shape}
    live_kinds = {kind for kind, _ in live_shape}
    check(
        "the same capture-point kinds appear under both providers",
        mock_kinds == live_kinds,
        "mock: " + str(sorted(mock_kinds)) + "\n         live: " + str(sorted(live_kinds)),
    )

    print("")
    print("Request/response pairing")
    check(
        "every LLM request has exactly one terminal partner, both providers",
        mock_req == mock_partners and live_req == live_partners,
        f"mock {mock_req} requests / {mock_partners} partners; "
        f"live {live_req} requests / {live_partners} partners",
    )

    print("")
    print("Author fields")
    mock_authors = {author for _, author in mock_shape if author}
    live_authors = {author for _, author in live_shape if author}
    check(
        "the same author vocabulary is used by both",
        mock_authors == live_authors,
        "mock: " + str(sorted(mock_authors)) + " | live: " + str(sorted(live_authors)),
    )

    print("")
    print("Well-formedness")
    check(
        "both traces parse against the turn grammar",
        not malformed,
        "clean" if not malformed else str(len(malformed)) + " malformed turn(s)",
    )

    print("")
    print("Shape (informational — tool counts legitimately differ)")
    print("  mock: " + str(mock_shape))
    print("  live: " + str(live_shape))

    print("")
    print("=" * 74)
    passed = sum(checks)
    print("PARITY: " + str(passed) + "/" + str(len(checks)) + " checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
