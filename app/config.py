"""Environment-based configuration.

Every setting the application reads lives here and comes from the environment
(see ``.env.example``). Nothing else in the codebase calls ``os.environ``.

Precedence note (this is a documented trap, see CLAUDE.md): pydantic-settings
gives real environment variables priority over the ``.env`` file. That is what
lets tests and eval runners force ``LLM_PROVIDER=mock`` — but it also means a
stray shell variable silently overrides ``.env``.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["mock", "openai", "groq"]


class Settings(BaseSettings):
    """Application settings, loaded once and cached."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM provider -----------------------------------------------------
    llm_provider: ProviderName = "mock"
    groq_api_key: str = ""
    openai_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    openai_model: str = "gpt-4o-mini"

    # --- Database ---------------------------------------------------------
    # A file on disk, never ":memory:" — state must survive a restart.
    database_url: str = "sqlite:///./data/agentcare.db"
    # ADK's session store. Deliberately separate: DatabaseSessionService needs
    # the async driver and will not accept a plain sqlite:/// URL.
    adk_session_db_url: str = "sqlite+aiosqlite:///./data/adk_sessions.db"

    # --- Auth -------------------------------------------------------------
    jwt_secret_key: str = "dev-only-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # --- Frontend -> backend ---------------------------------------------
    api_base_url: str = "http://localhost:8000"
    cors_origins: str = "http://localhost:8501"

    # --- Time -------------------------------------------------------------
    # Test/demo override for clock.today(). Blank in normal operation, where
    # the live app always uses the real current date.
    app_today: date | None = None

    # --- Agent limits (boundedness invariant) ----------------------------
    max_tool_iterations: int = 8
    max_replans_per_run: int = 1
    # History windowing: persist everything, send the model the last N turns.
    history_window_turns: int = 15
    max_confirmation_non_answers: int = 3
    max_reminder_attempts: int = 3

    # --- Uploads ----------------------------------------------------------
    upload_dir: str = "./data/uploads"
    max_upload_bytes: int = 10 * 1024 * 1024
    allowed_upload_mime: str = "application/pdf,image/png,image/jpeg"

    # --- Scheduler --------------------------------------------------------
    reminder_poll_seconds: int = 60

    # --- Observability ----------------------------------------------------
    log_level: str = "INFO"
    phoenix_enabled: bool = False

    @field_validator("app_today", mode="before")
    @classmethod
    def _blank_date_is_none(cls, v: object) -> object:
        """Treat ``APP_TODAY=`` (present but empty) as unset.

        ``.env.example`` ships the key with an empty value so its meaning is
        documented; without this, pydantic would reject "" as an invalid date.
        """
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalise_provider(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_upload_mime_list(self) -> list[str]:
        return [m.strip() for m in self.allowed_upload_mime.split(",") if m.strip()]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def api_key_for_active_provider(self) -> str:
        """Key for the selected provider ("" for mock, which needs none)."""
        return {
            "mock": "",
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
        }[self.llm_provider]

    def model_for_active_provider(self) -> str:
        """Model id for the selected provider.

        Per-provider variables, never one shared ``LLM_MODEL`` — that would let
        a Groq model id be sent to OpenAI on a provider switch.
        """
        return {
            "mock": "mock-understudy",
            "groq": self.groq_model,
            "openai": self.openai_model,
        }[self.llm_provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings (tests that manipulate the environment)."""
    get_settings.cache_clear()
