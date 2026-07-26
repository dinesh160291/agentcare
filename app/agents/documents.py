"""The Document specialist.

Reports what is on file and what a department still requires. The missing-docs
diff is re-read rather than accepted from the model, and recorded as an upsert:
one open task per appointment, updated as uploads arrive, closed by itself when
the list empties.

Document types are administrative labels here. What a document *shows* is not
this system's business.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from app.agents import base
from app.agents.callbacks import TurnCallbacks
from app.agents.toolbelt import Toolbelt

NAME = "document"


def build_agent(
    toolbelt: Toolbelt, callbacks: TurnCallbacks, *, provider: str | None = None
) -> LlmAgent:
    return base.build(
        name=NAME,
        prompt_key="document",
        description="Reports documents on file and what a department still needs.",
        tools=toolbelt.document_tools(),
        callbacks=callbacks,
        provider=provider,
    )


__all__ = ["NAME", "build_agent"]
