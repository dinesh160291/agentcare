"""LLM providers — the seam selected by ``LLM_PROVIDER``.

All three are ``google.adk.models.BaseLlm`` subclasses of our own. LiteLLM is
not used and cannot be installed here; see CLAUDE.md.

``mock`` is a first-class provider, not a test fixture: under it the whole
application runs end to end, with real tool calls, real database writes, and
replies templated from persisted tool results.
"""

from app.providers.base import AgentCareLlm
from app.providers.mock import MockLlm

__all__ = ["AgentCareLlm", "MockLlm"]
