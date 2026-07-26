"""ADK callbacks: the trace capture points, and the iteration budget.

The spike proved these callbacks reach the capture points. It did it with a
module-level writer, which is fine for a script that runs one turn and exits
and wrong for a server: two concurrent turns would write each other's events.
Here they are closures over one turn's :class:`~app.trace.TraceWriter`.

**Parameter names must match ADK's keywords exactly** — ``callback_context``,
``llm_request``, ``llm_response``, ``tool``, ``args``, ``tool_context``,
``tool_response``. ADK passes them by keyword, so renaming one to ``ctx`` is a
mid-turn ``TypeError``, landing in the one layer whose whole job is to notice
that something went wrong.

The tool callbacks also carry the **iteration budget**. Error retries are
bounded above, but a loop of *successful* calls never trips a retry ladder: a
slot search that returns empty, prompting a wider search that returns empty,
prompting a wider search, is a sequence of perfectly good calls that never
ends. ``before_tool`` refuses past the cap and records the exhaustion, which
the orchestrator turns into a ``failed`` run and a graceful reply.
"""

from __future__ import annotations

from app.agents.memory import window_contents
from app.providers.base import request_snapshot, response_snapshot, text_response
from app.trace import TraceWriter


class TurnCallbacks:
    """One turn's capture points, shared by every agent in that turn."""

    def __init__(
        self,
        writer: TraceWriter,
        *,
        max_tool_iterations: int,
        history_window_turns: int = 0,
    ) -> None:
        self.writer = writer
        self.max_tool_iterations = max_tool_iterations
        self.history_window_turns = history_window_turns
        self.tool_iterations = 0
        self.budget_exhausted = False
        #: The request awaiting a partner. Held here rather than in ADK session
        #: state so that a provider exception can still be paired: the
        #: orchestrator reads this and writes the ``llm_error`` itself.
        self.pending_request: str | None = None
        self.pending_agent: str | None = None
        self._tool_correlations: dict[str, str] = {}

    # --- LLM ------------------------------------------------------------

    def start_agent(self) -> None:
        """Begin a new agent's turn. Resets the per-agent tool counter.

        The cap is per agent turn, not per conversation turn: a booking turn
        legitimately spends two calls in the Coordinator, two in Routing, and
        three in Appointment, and a shared counter would put an ordinary
        request within one call of its own budget. ``budget_exhausted`` is
        *not* reset — once a turn has blown a budget, it has failed.
        """
        self.tool_iterations = 0

    def before_model(self, callback_context, llm_request):  # noqa: ANN001
        # Refusing the tool is not enough to stop a loop: the model would be
        # asked again, want the same tool again, and keep going until ADK's own
        # implicit call limit fired — the framework limit the budget exists so
        # as not to depend on. Returning a response here ends the agent's loop
        # from the outside.
        if self.budget_exhausted:
            return text_response(
                "I couldn't complete this request within the steps available."
            )

        # Window *before* the snapshot, so the trace records what was actually
        # sent rather than what the framework would have sent unaided.
        if self.history_window_turns > 0 and llm_request.contents:
            llm_request.contents = window_contents(
                llm_request.contents, turns=self.history_window_turns
            )

        self.pending_agent = callback_context.agent_name
        self.pending_request = self.writer.llm_request(
            agent_name=callback_context.agent_name,
            payload=request_snapshot(llm_request),
        )
        return None

    def after_model(self, callback_context, llm_response):  # noqa: ANN001
        if self.pending_request is None:
            return None
        self.writer.llm_response(
            self.pending_request,
            agent_name=callback_context.agent_name,
            payload=response_snapshot(llm_response),
        )
        self.pending_request = None
        return None

    def fail_pending_request(self, error: str, *, kind: str = "error") -> None:
        """Close an open request with an error.

        Called by the orchestrator when the provider raised: without it the
        request has no terminal partner, and a dangling request is supposed to
        mean exactly one thing — the process died mid-call.
        """
        if self.pending_request is None:
            return
        self.writer.llm_error(
            self.pending_request,
            agent_name=self.pending_agent or "unknown",
            error=error,
            kind=kind,
        )
        self.pending_request = None

    # --- Tools ------------------------------------------------------------

    def before_tool(self, tool, args, tool_context):  # noqa: ANN001
        """Record the call — or refuse it, if the budget is spent.

        Returning a dict makes ADK skip the tool and hand the model that dict
        as the result, which is how the loop is stopped from inside rather than
        by hoping the model gets bored.
        """
        self.tool_iterations += 1
        if self.tool_iterations > self.max_tool_iterations:
            self.budget_exhausted = True
            self.writer.validation(
                "tool_iteration_budget",
                accepted=False,
                detail={
                    "tool": tool.name,
                    "iterations": self.tool_iterations,
                    "cap": self.max_tool_iterations,
                },
            )
            return {
                "error": (
                    "Tool iteration budget exhausted for this turn. Stop calling "
                    "tools and explain that this could not be completed."
                )
            }

        correlation = self.writer.tool_call(
            tool.name, args=dict(args or {}), agent_name=tool_context.agent_name
        )
        self._tool_correlations[tool.name] = correlation
        return None

    def after_tool(self, tool, args, tool_context, tool_response):  # noqa: ANN001
        correlation = self._tool_correlations.pop(tool.name, None)
        if correlation is None:
            # The call was refused by the budget, so no tool_call was written;
            # pairing a result to nothing would fail the well-formedness check.
            return None
        self.writer.tool_result(
            correlation,
            name=tool.name,
            result=tool_response,
            agent_name=tool_context.agent_name,
        )
        return None


__all__ = ["TurnCallbacks"]
