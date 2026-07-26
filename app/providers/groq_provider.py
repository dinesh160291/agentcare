"""Groq adapter — the default live provider.

Groq is the day-to-day live provider: fast, and generous enough on the free
tier to run the eval suite without budgeting for it.
"""

from __future__ import annotations

import groq

from app.config import get_settings
from app.providers.openai_compatible import OpenAICompatibleLlm


class GroqLlm(OpenAICompatibleLlm):
    provider_name: str = "groq"
    model: str = ""

    def __init__(self, **data):
        settings = get_settings()
        data.setdefault("model", settings.groq_model)
        super().__init__(**data)
        self._cached_client = None

    def _client(self):
        if self._cached_client is None:
            settings = get_settings()
            if not settings.groq_api_key:
                from app.errors import ProviderError

                raise ProviderError("GROQ_API_KEY is not set")
            self._cached_client = groq.Groq(api_key=settings.groq_api_key)
        return self._cached_client
