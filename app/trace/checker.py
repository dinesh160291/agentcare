"""The trace well-formedness checker.

Parses persisted trace events against the turn grammar and reports every way a
turn is malformed. This is what converts "everything is captured" from a claim
into an invariant — the claim is cheap, and it stays true only for as long as
nobody forgets a capture point.

The grammar for one turn::

    inbound
      -> guard verdict*
      -> [ llm_request -> (llm_response | llm_error)
           -> validation* -> tool_call -> tool_result ]*
      -> transition*
      -> outbound

Rules enforced:

* every turn opens with exactly one inbound and closes with at least one outbound
* every ``llm_request`` has exactly one terminal partner, matched by correlation id
* every ``tool_result`` matches a preceding ``tool_call``
* every outbound names its author
* nothing precedes the inbound

Typed-action turns follow the same rules. A checker that only understood chatty
turns would pass the system's least deterministic flows and ignore its most.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from sqlalchemy.orm import Session

from app.models import TraceEvent
from app.models.enums import TraceEventType

TERMINAL_LLM_EVENTS = (TraceEventType.LLM_RESPONSE, TraceEventType.LLM_ERROR)


@dataclass
class TurnReport:
    turn_id: str
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def check_turn(events: Iterable[TraceEvent]) -> TurnReport:
    """Check one turn's events, which must already be seq-ordered."""
    ordered = sorted(events, key=lambda e: e.seq)
    report = TurnReport(turn_id=ordered[0].turn_id if ordered else "<empty>")

    if not ordered:
        report.problems.append("turn has no events")
        return report

    kinds = [event.event_type for event in ordered]

    inbound_count = kinds.count(TraceEventType.INBOUND)
    if inbound_count == 0:
        report.problems.append("turn has no inbound event")
    elif inbound_count > 1:
        report.problems.append(f"turn has {inbound_count} inbound events, expected 1")
    elif kinds[0] is not TraceEventType.INBOUND:
        report.problems.append(f"turn opens with {kinds[0].value}, not inbound")

    if TraceEventType.OUTBOUND not in kinds:
        report.problems.append("turn has no outbound event")

    # Every LLM request needs exactly one terminal partner.
    partners: dict[str, list[TraceEventType]] = defaultdict(list)
    requests: set[str] = set()
    for event in ordered:
        if event.event_type is TraceEventType.LLM_REQUEST:
            if event.correlation_id is None:
                report.problems.append(f"llm_request at seq {event.seq} has no correlation id")
            else:
                requests.add(event.correlation_id)
        elif event.event_type in TERMINAL_LLM_EVENTS and event.correlation_id is not None:
            partners[event.correlation_id].append(event.event_type)

    for correlation_id in sorted(requests):
        found = partners.get(correlation_id, [])
        if not found:
            report.problems.append(
                f"llm_request {correlation_id[:8]} has no response or error "
                "(a request with no partner means the process died mid-call)"
            )
        elif len(found) > 1:
            report.problems.append(
                f"llm_request {correlation_id[:8]} has {len(found)} terminal events, expected 1"
            )

    for correlation_id in sorted(set(partners) - requests):
        report.problems.append(
            f"llm response/error {correlation_id[:8]} has no matching request"
        )

    # Tool results must follow their call.
    tool_calls = {
        event.correlation_id
        for event in ordered
        if event.event_type is TraceEventType.TOOL_CALL and event.correlation_id
    }
    for event in ordered:
        if event.event_type is TraceEventType.TOOL_RESULT:
            if event.correlation_id not in tool_calls:
                report.problems.append(
                    f"tool_result at seq {event.seq} has no matching tool_call"
                )

    for event in ordered:
        if event.event_type is TraceEventType.OUTBOUND and event.author is None:
            report.problems.append(
                f"outbound at seq {event.seq} has no author "
                "(code-authored replies must not be invisible)"
            )
        if event.event_type is TraceEventType.INBOUND and event.author is None:
            report.problems.append(f"inbound at seq {event.seq} has no author")

    expected_seqs = list(range(1, len(ordered) + 1))
    if [event.seq for event in ordered] != expected_seqs:
        report.problems.append("sequence numbers are not contiguous from 1")

    return report


def check_session(session: Session, *, turn_id: str | None = None) -> list[TurnReport]:
    """Check every turn in the database, or one named turn."""
    query = session.query(TraceEvent)
    if turn_id is not None:
        query = query.filter(TraceEvent.turn_id == turn_id)

    by_turn: dict[str, list[TraceEvent]] = defaultdict(list)
    for event in query.order_by(TraceEvent.turn_id, TraceEvent.seq).all():
        by_turn[event.turn_id].append(event)

    return [check_turn(events) for _, events in sorted(by_turn.items())]


def assert_well_formed(session: Session, *, turn_id: str | None = None) -> None:
    """Raise unless every trace in scope parses. Used by tests and evals."""
    failures = [report for report in check_session(session, turn_id=turn_id) if not report.ok]
    if failures:
        detail = "\n".join(
            f"  turn {report.turn_id[:8]}: " + "; ".join(report.problems)
            for report in failures
        )
        raise AssertionError(f"malformed traces:\n{detail}")
