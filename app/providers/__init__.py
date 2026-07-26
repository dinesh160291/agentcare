"""LLM providers — the seam selected by ``LLM_PROVIDER``.

All three are ``google.adk.models.BaseLlm`` subclasses of our own. LiteLLM is
not used and cannot be installed here; see CLAUDE.md.

``mock`` is a first-class provider, not a test fixture: under it the whole
application runs end to end, with real tool calls, real database writes, and
replies templated from persisted tool results.
"""

from __future__ import annotations

from app.config import get_settings
from app.providers.base import AgentCareLlm
from app.providers.mock import MockLlm


def get_provider(name: str | None = None) -> AgentCareLlm:
    """Build the provider named by ``LLM_PROVIDER`` (or by ``name``).

    Imported lazily so that running under ``mock`` never touches the live SDKs,
    and a missing API key cannot break a mock-mode test run.
    """
    settings = get_settings()
    provider = (name or settings.llm_provider).lower()

    if provider == "mock":
        return MockLlm()
    if provider == "groq":
        from app.providers.groq_provider import GroqLlm

        return GroqLlm()
    if provider == "openai":
        from app.providers.openai_provider import OpenAILlm

        return OpenAILlm()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


__all__ = ["AgentCareLlm", "MockLlm", "get_provider"]
