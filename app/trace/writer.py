"""The TraceEvent writer — the system of record for what happened in a turn.

One :class:`TraceWriter` per turn. It stamps a ``turn_id``, keeps the sequence
counter, redacts every payload on the way in, and exposes one method per
capture point.

Two rules the writer enforces rather than trusting:

* **Every LLM request gets exactly one terminal partner** — a response or an
  error. :meth:`llm_request` returns a correlation id, and the response and
  error methods require it. A request with no partner then means precisely one
  thing: the process died mid-call. That is a diagnosis, not a mystery.
* **Every outbound reply records its author** — ``llm``, ``template``, or
  ``guard``. Without this, code-authored replies are invisible and the timeline
  implies the model said things it never said.

``workflow_run_id`` is nullable on purpose. Off-topic messages and safety
screens fire before any run exists, and those turns still have to be bracketed.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.models import TraceEvent
from app.models.enums import TraceAuthor, TraceEventType
from app.trace.redaction import redact


class TraceWriter:
    """Writes the events of a single turn, in order."""

    def __init__(
        self,
        session: Session,
        *,
        session_id: str | None = None,
        workflow_run_id: int | None = None,
        turn_id: str | None = None,
    ) -> None:
        self.session = session
        self.session_id = session_id
        self.workflow_run_id = workflow_run_id
        self.turn_id = turn_id or uuid.uuid4().hex
        self._seq = 0
        self._open_requests: set[str] = set()

    # --- internals -------------------------------------------------------

    def _write(
        self,
        event_type: TraceEventType,
        *,
        author: TraceAuthor | None = None,
        agent_name: str | None = None,
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> TraceEvent:
        self._seq += 1
        event = TraceEvent(
            workflow_run_id=self.workflow_run_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            seq=self._seq,
            event_type=event_type,
            author=author,
            agent_name=agent_name,
            payload=redact(payload or {}),
            correlation_id=correlation_id,
        )
        self.session.add(event)
        self.session.flush()
        return event

    def bind_run(self, workflow_run_id: int) -> None:
        """Attach later events to a run created mid-turn.

        A turn can begin before its run exists — the run is created once intent
        is accepted — so earlier events keep a null run id and the rest of the
        turn is attributed properly.
        """
        self.workflow_run_id = workflow_run_id

    # --- the nine capture points ----------------------------------------

    def inbound(self, content: str, *, author: TraceAuthor, **extra: Any) -> TraceEvent:
        """A message or a typed action arriving. Opens the turn.

        Typed actions (a Confirm click, a staff approval) are inbound events
        too. Without that, the system's most deterministic flows would be the
        ones the well-formedness checker could not parse.
        """
        return self._write(
            TraceEventType.INBOUND, author=author, payload={"content": content, **extra}
        )

    def guard_verdict(
        self, name: str, *, passed: bool, detail: dict[str, Any] | None = None
    ) -> TraceEvent:
        """A guard ran. Recorded whether it fired or passed.

        Passes matter: "the safety screen did not fire" and "the safety screen
        never ran" are different facts, and only one of them is acceptable.
        """
        return self._write(
            TraceEventType.GUARD_VERDICT,
            author=TraceAuthor.GUARD,
            payload={"guard": name, "passed": passed, "detail": detail or {}},
        )

    def llm_request(self, *, agent_name: str, payload: dict[str, Any]) -> str:
        """An LLM call is about to be made. Returns its correlation id."""
        correlation_id = uuid.uuid4().hex
        self._open_requests.add(correlation_id)
        self._write(
            TraceEventType.LLM_REQUEST,
            agent_name=agent_name,
            payload=payload,
            correlation_id=correlation_id,
        )
        return correlation_id

    def llm_response(
        self, correlation_id: str, *, agent_name: str, payload: dict[str, Any]
    ) -> TraceEvent:
        """The paired response for a request."""
        self._open_requests.discard(correlation_id)
        return self._write(
            TraceEventType.LLM_RESPONSE,
            author=TraceAuthor.LLM,
            agent_name=agent_name,
            payload=payload,
            correlation_id=correlation_id,
        )

    def llm_error(
        self, correlation_id: str, *, agent_name: str, error: str, kind: str = "error"
    ) -> TraceEvent:
        """The paired failure for a request — a timeout, a rate limit, a refusal."""
        self._open_requests.discard(correlation_id)
        return self._write(
            TraceEventType.LLM_ERROR,
            agent_name=agent_name,
            payload={"error": error, "kind": kind},
            correlation_id=correlation_id,
        )

    def validation(
        self, what: str, *, accepted: bool, detail: dict[str, Any] | None = None
    ) -> TraceEvent:
        """Code checked a model proposal against a closed set.

        Rejections are the interesting half: an invented department that never
        reached the database still happened, and the trace is where that shows.
        """
        return self._write(
            TraceEventType.VALIDATION,
            author=TraceAuthor.GUARD,
            payload={"what": what, "accepted": accepted, "detail": detail or {}},
        )

    def tool_call(self, name: str, *, args: dict[str, Any], agent_name: str | None = None) -> str:
        """A tool is about to run. Returns its correlation id."""
        correlation_id = uuid.uuid4().hex
        self._write(
            TraceEventType.TOOL_CALL,
            agent_name=agent_name,
            payload={"tool": name, "args": args},
            correlation_id=correlation_id,
        )
        return correlation_id

    def tool_result(
        self, correlation_id: str, *, name: str, result: Any, agent_name: str | None = None
    ) -> TraceEvent:
        return self._write(
            TraceEventType.TOOL_RESULT,
            agent_name=agent_name,
            payload={"tool": name, "result": result},
            correlation_id=correlation_id,
        )

    def transition(
        self,
        *,
        from_status: str | None,
        to_status: str,
        reason: str = "",
        **extra: Any,
    ) -> TraceEvent:
        """A state change was attempted.

        Recorded whether or not it landed: a compare-and-swap that lost its
        race is an event, and a no-op nobody can see is indistinguishable from
        a request that never arrived.
        """
        return self._write(
            TraceEventType.TRANSITION,
            author=TraceAuthor.SYSTEM,
            payload={"from": from_status, "to": to_status, "reason": reason, **extra},
        )

    def outbound(self, content: str, *, author: TraceAuthor, **extra: Any) -> TraceEvent:
        """The reply leaving the system. Closes the turn.

        ``author`` is required and has no default: guessing it would defeat the
        entire point of recording it.
        """
        return self._write(
            TraceEventType.OUTBOUND, author=author, payload={"content": content, **extra}
        )

    # --- introspection ---------------------------------------------------

    @property
    def unpaired_requests(self) -> set[str]:
        """Requests still awaiting a response or an error."""
        return set(self._open_requests)
