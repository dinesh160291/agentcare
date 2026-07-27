"""Configuration and the clock seam."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from app import clock
from app.config import Settings


class TestSettings:
    def test_tests_run_against_the_mock_provider(self, settings):
        """The suite must never make a billed call. See conftest for why."""
        assert settings.llm_provider == "mock"
        assert settings.api_key_for_active_provider() == ""

    def test_secrets_never_appear_in_the_settings_repr(self):
        """Found the hard way: a pytest assertion involving the settings
        fixture prints ``repr(settings)`` into the failure output, and a
        failure in CI puts that in a public build log. It happened in this
        repo's own test run with a live Groq key in ``.env``.

        The values are still readable through the attributes — this closes the
        accidental path, not the deliberate one.
        """
        secret = "sk-do-not-print-me-0123456789"
        settings = Settings(
            groq_api_key=secret,
            openai_api_key=secret,
            jwt_secret_key=secret,
        )

        assert secret not in repr(settings)
        assert secret not in str(settings)
        # And it is still usable where it is meant to be.
        assert settings.jwt_secret_key == secret

    def test_blank_app_today_is_treated_as_unset(self):
        """``.env.example`` ships APP_TODAY with an empty value."""
        assert Settings(app_today="").app_today is None
        assert Settings(app_today="   ").app_today is None

    def test_app_today_parses_a_real_date(self):
        assert Settings(app_today="2026-03-01").app_today == date(2026, 3, 1)

    def test_provider_name_is_normalised(self):
        assert Settings(llm_provider="GROQ").llm_provider == "groq"

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ValueError):
            Settings(llm_provider="anthropic-but-typoed")

    def test_each_provider_has_its_own_model_variable(self):
        """One shared LLM_MODEL would send a Groq model id to OpenAI."""
        groq = Settings(llm_provider="groq", groq_model="llama-x", openai_model="gpt-y")
        openai = Settings(llm_provider="openai", groq_model="llama-x", openai_model="gpt-y")
        assert groq.model_for_active_provider() == "llama-x"
        assert openai.model_for_active_provider() == "gpt-y"

    def test_list_settings_split_cleanly(self):
        s = Settings(cors_origins="http://a, http://b ,", allowed_upload_mime="application/pdf")
        assert s.cors_origin_list == ["http://a", "http://b"]
        assert s.allowed_upload_mime_list == ["application/pdf"]


class TestClock:
    def test_freeze_pins_both_now_and_today(self):
        clock.freeze(datetime(2026, 3, 1, 14, 30))
        assert clock.now() == datetime(2026, 3, 1, 14, 30)
        assert clock.today() == date(2026, 3, 1)

    def test_freezing_a_date_anchors_the_time(self):
        """A date-only freeze must still be deterministic."""
        clock.freeze(date(2026, 3, 1))
        assert clock.now() == datetime.combine(date(2026, 3, 1), clock.ANCHOR_TIME)

    def test_unfreeze_returns_to_real_time(self):
        clock.freeze(date(2000, 1, 1))
        clock.unfreeze()
        assert clock.today() == date.today()

    def test_advance_moves_a_frozen_clock(self):
        clock.freeze(datetime(2026, 3, 1, 9, 0))
        clock.advance(timedelta(hours=25))
        assert clock.now() == datetime(2026, 3, 2, 10, 0)

    def test_advance_requires_a_frozen_clock(self):
        """Advancing real time silently would make a test lie about its state."""
        with pytest.raises(RuntimeError):
            clock.advance(timedelta(days=1))

    def test_frozen_at_restores_the_previous_state(self):
        with clock.frozen_at(date(2026, 3, 1)):
            assert clock.today() == date(2026, 3, 1)
        assert clock.today() == date.today()

    def test_is_overridden_reports_a_pinned_clock(self):
        assert clock.is_overridden() is False
        clock.freeze(date(2026, 3, 1))
        assert clock.is_overridden() is True
