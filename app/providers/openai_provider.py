"""OpenAI adapter — the alternative live provider.

Kept deliberately thin: everything except client construction is shared with
Groq, so a bug fixed in the translation layer is fixed for both.
"""

from __future__ import annotations

import openai

from app.config import get_settings
from app.providers.openai_compatible import OpenAICompatibleLlm


class OpenAILlm(OpenAICompatibleLlm):
    provider_name: str = "openai"
    model: str = ""

    def __init__(self, **data):
        settings = get_settings()
        data.setdefault("model", settings.openai_model)
        super().__init__(**data)
        self._cached_client = None

    def _client(self):
        if self._cached_client is None:
            settings = get_settings()
            if not settings.openai_api_key:
                from app.errors import ProviderError

                raise ProviderError("OPENAI_API_KEY is not set")
            self._cached_client = openai.OpenAI(api_key=settings.openai_api_key)
        return self._cached_client
