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

They also carry the tighter bound the budget is too blunt for. A decision the
model has already had accepted is *settled*, and asking for it again is not a
step toward anything. Two rules, one flag:

* an **accepted repeat** — the same tool, the same arguments, already accepted
  this agent turn — is refused and ends the loop;
* a **terminal tool** — one whose acceptance is the agent's entire output —
  ends the loop the moment it is accepted, with no further call at all.

Both matter because the budget's failure mode is expensive and misleading.
Observed live on ``openai/gpt-4o-mini``: it re-called ``submit_safety_verdict``
eight times after the verdict was accepted, spending nine calls on one screen
and then **failing a turn that had already succeeded**. Models that volunteer
text at that point — llama, the mock — merely paid for a call whose output is
thrown away, which is why the defect survived two providers.

Ending the loop is done from ``before_model``, not by refusing the tool:
refusal leaves the model being asked again, wanting the same thing again.
"""

from __future__ import annotations

import json

from app.agents.memory import window_contents
from app.providers.base import (
    current_turn_start,
    request_snapshot,
    response_snapshot,
    text_response,
)
from app.trace import TraceWriter


def _signature(name: str, args: dict | None) -> str:
    """A call's identity: its tool and its arguments, order-independent."""
    return f"{name}({json.dumps(args or {}, sort_keys=True, default=str)})"


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
        #: Set for a turn with no live run, where the Coordinator's job is to
        #: *plan* rather than to classify. A planner has nothing to learn from
        #: the transcript — whether this message needs routing and a booking is
        #: a fact about this message and the patient's record — and live it had
        #: something to lose: reasoning over a session in which "conflicting"
        #: had once been the right answer, it submitted that word as a plan step
        #: five times running and the request never started. Classification is
        #: the opposite case and is deliberately untouched: how a message
        #: relates to what came before is unanswerable without what came before.
        self.plan_context_only = False
        self.tool_iterations = 0
        self.budget_exhausted = False
        #: This agent has said everything it was asked for. Unlike
        #: ``budget_exhausted`` this is not a failure — the answer is in the
        #: toolbelt, so the turn carries on from there.
        self.settled = False
        self.terminal_tool: str | None = None
        self._accepted: set[str] = set()
        #: The request awaiting a partner. Held here rather than in ADK session
        #: state so that a provider exception can still be paired: the
        #: orchestrator reads this and writes the ``llm_error`` itself.
        self.pending_request: str | None = None
        self.pending_agent: str | None = None
        self._tool_correlations: dict[str, str] = {}

    # --- LLM ------------------------------------------------------------

    def start_agent(self, *, terminal_tool: str | None = None) -> None:
        """Begin a new agent's turn. Resets the per-agent tool counter.

        The cap is per agent turn, not per conversation turn: a booking turn
        legitimately spends two calls in the Coordinator, two in Routing, and
        three in Appointment, and a shared counter would put an ordinary
        request within one call of its own budget. ``budget_exhausted`` is
        *not* reset — once a turn has blown a budget, it has failed.

        ``settled`` and the accepted-call set *are* per agent, for the same
        reason the counter is: the Coordinator having submitted its plan says
        nothing about whether Routing has submitted a department.

        ``terminal_tool`` names the one tool, if any, whose acceptance is this
        agent's whole output. Only the caller knows: the safety screen's
        prompt forbids it a reply and :func:`app.safety.classifier.llm_screen`
        discards its text, so a call after the verdict has nothing to produce.
        Every other agent is asked to speak once its tool has run, so leaving
        this ``None`` is the right default and the accepted-repeat rule is
        what bounds them.
        """
        self.tool_iterations = 0
        self.settled = False
        self.terminal_tool = terminal_tool
        self._accepted = set()

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

        # Settled is the *successful* twin of the above, and the reply is empty
        # on purpose: this agent has nothing left to say, and the orchestrator's
        # existing fallbacks turn an empty reply into a template. A sentence
        # here would become the patient's reply — words nobody wrote for them.
        if self.settled:
            return text_response("")

        # Window *before* the snapshot, so the trace records what was actually
        # sent rather than what the framework would have sent unaided.
        if self.plan_context_only and llm_request.contents:
            # The current turn and nothing older. Cutting at
            # ``current_turn_start`` rather than counting turns reuses the one
            # definition of "this turn" the providers already read, and it
            # keeps what the turn has earned: the patient's message, and the
            # context tool's result once it has been called.
            llm_request.contents = llm_request.contents[
                current_turn_start(llm_request) :
            ]
        elif self.history_window_turns > 0 and llm_request.contents:
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

        signature = _signature(tool.name, dict(args or {}))
        if signature in self._accepted:
            # Not an error the model can correct — the answer it wants is one
            # it already has. Running the tool again would re-apply whatever
            # the first call applied, so this refuses *and* ends the loop.
            self.settled = True
            self.writer.validation(
                "repeated_tool_call",
                accepted=False,
                detail={
                    "tool": tool.name,
                    "args": dict(args or {}),
                    "problem": "already accepted this turn",
                },
            )
            return {
                "error": (
                    f"{tool.name} has already been accepted this turn with these "
                    "arguments. That decision is made; do not submit it again."
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
            # The call was refused before it ran — by the budget or as a repeat
            # — so no tool_call was written, and pairing a result to nothing
            # would fail the well-formedness check.
            return None
        self.writer.tool_result(
            correlation,
            name=tool.name,
            result=tool_response,
            agent_name=tool_context.agent_name,
        )

        # Only *accepted* results settle anything. A rejected proposal is the
        # retry ladder working — the model is meant to correct it and call
        # again — and a read-only result carries no verdict to repeat.
        accepted = (
            isinstance(tool_response, dict) and tool_response.get("accepted") is True
        )
        if not accepted:
            return None

        self._accepted.add(_signature(tool.name, dict(args or {})))
        if tool.name == self.terminal_tool:
            self.settled = True
            self.writer.validation(
                "agent_settled",
                accepted=True,
                detail={"tool": tool.name, "agent": tool_context.agent_name},
            )
        return None


__all__ = ["TurnCallbacks"]
