"""Tier-1 memory: the conversation, and how much of it the model sees.

Two stores, and the ranking between them is the load-bearing part.

**Tier 1** is ADK's ``DatabaseSessionService`` — conversation history and
session state, in SQL, surviving restarts. It also satisfies the mandatory
persistent-agent-state requirement, alongside the ``WorkflowRun`` row.

**History windowing** is the boundedness invariant applied to Tier 1. History
is an automated writer hiding behind the framework: appended every turn, never
trimmed, and sent whole to the model, so prompts grow monotonically toward the
context limit and the rate limit. The rule is *persist everything, send the
model the last N turns.* That is safe here specifically because facts are never
read from history — receipts re-read rows and status questions call tools — so
windowing costs the model old chit-chat, never truth.

**Authority ranking.** The database row wins. Session state is the model's
scratchpad: conversational hints and preferences, never a pending slot, a
resolved department, or a run status. A crash between two writes can
desynchronise the stores, and this rule decides who is right when it does.

Specialists do not appear here at all. They receive no history by design, so
they run against a throwaway in-memory session holding exactly one message —
their typed task — which is also why the durable session store grows by one row
per conversation rather than one per agent call.
"""

from __future__ import annotations

from google.adk.sessions import DatabaseSessionService, InMemorySessionService

from app.config import get_settings

APP_NAME = "agentcare"


def window_contents(contents: list, *, turns: int) -> list:
    """Keep the last ``turns`` user messages and everything after the first.

    Counts from the end so the most recent exchange is always whole: cutting
    mid-turn would hand the model a tool result whose call it cannot see.
    Returns the input unchanged when it is already short enough, so the common
    case allocates nothing.
    """
    if turns <= 0 or not contents:
        return contents

    seen = 0
    for index in range(len(contents) - 1, -1, -1):
        if getattr(contents[index], "role", None) == "user":
            seen += 1
            if seen == turns:
                return contents[index:]
    return contents


def conversation_service() -> DatabaseSessionService:
    """The durable store for a patient's conversation.

    Needs the async SQLite driver: ``DatabaseSessionService`` will not accept a
    plain ``sqlite:///`` URL, which is why this is a separate setting from
    ``DATABASE_URL`` and must stay separate.
    """
    return DatabaseSessionService(db_url=get_settings().adk_session_db_url)


def task_service() -> InMemorySessionService:
    """A throwaway store for one specialist task.

    Nothing here is consequential — the ``WorkflowRun`` row is — so persisting
    it would trade a growth rule for no benefit.
    """
    return InMemorySessionService()


__all__ = [
    "APP_NAME",
    "conversation_service",
    "task_service",
    "window_contents",
]
