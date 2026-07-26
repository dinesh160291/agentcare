"""Date resolution — the defence against fabricated dates.

An LLM handed "next week" with no anchor will resolve it differently across
runs while staying schema-valid and entirely plausible. So the model never
resolves a date: it passes the patient's phrase through, and this tool decides
what it means. Every expectation below is a promise the golden set then pins.

The clock is frozen to **Monday 3 August 2026** throughout, so "next week" has
one correct answer rather than a moving one.
"""

from __future__ import annotations

from datetime import date

import pytest

from app import clock
from app.tools.dates import resolve_date

MONDAY = date(2026, 8, 3)
WEDNESDAY = date(2026, 8, 5)
SATURDAY = date(2026, 8, 8)
SUNDAY = date(2026, 8, 9)


@pytest.fixture(autouse=True)
def _monday():
    clock.freeze(MONDAY)


class TestPlainOffsets:
    def test_today(self):
        result = resolve_date("today")
        assert result["resolved"] is True
        assert result["start"] == "2026-08-03"
        assert result["end"] == "2026-08-03"
        assert result["kind"] == "exact"

    def test_tomorrow(self):
        result = resolve_date("tomorrow")
        assert (result["start"], result["end"]) == ("2026-08-04", "2026-08-04")

    def test_day_after_tomorrow(self):
        assert resolve_date("day after tomorrow")["start"] == "2026-08-05"

    def test_in_n_days(self):
        assert resolve_date("in 3 days")["start"] == "2026-08-06"

    def test_in_spelled_out_days(self):
        assert resolve_date("in three days")["start"] == "2026-08-06"

    def test_in_n_weeks(self):
        result = resolve_date("in 2 weeks")
        assert result["start"] == "2026-08-17"
        assert result["kind"] == "range"


class TestWeekRanges:
    def test_next_week_is_the_following_monday_to_sunday(self):
        """The canonical request. A range, not a point — the patient has not
        chosen a day yet, and inventing one for them is the failure mode."""
        result = resolve_date("next week")
        assert result["start"] == "2026-08-10"
        assert result["end"] == "2026-08-16"
        assert result["kind"] == "range"

    def test_this_week_runs_from_today_not_from_monday(self):
        """A past-dated half of this week is not bookable, so it is not offered."""
        clock.freeze(WEDNESDAY)
        result = resolve_date("this week")
        assert result["start"] == "2026-08-05"
        assert result["end"] == "2026-08-09"

    def test_next_week_from_a_sunday_still_means_the_following_week(self):
        """Sunday is the boundary case every week-arithmetic bug lands on."""
        clock.freeze(SUNDAY)
        result = resolve_date("next week")
        assert result["start"] == "2026-08-10"
        assert result["end"] == "2026-08-16"

    def test_next_month(self):
        result = resolve_date("next month")
        assert result["start"] == "2026-09-01"
        assert result["end"] == "2026-09-30"
        assert result["kind"] == "range"


class TestWeekdayNames:
    def test_a_bare_weekday_is_the_next_upcoming_one(self):
        assert resolve_date("friday")["start"] == "2026-08-07"

    def test_a_bare_weekday_never_resolves_to_today(self):
        """"Monday" said on a Monday means the one coming, not the one being
        lived through — its slots are already half gone."""
        assert resolve_date("monday")["start"] == "2026-08-10"

    def test_next_weekday_skips_a_week_from_the_upcoming_one(self):
        assert resolve_date("next friday")["start"] == "2026-08-14"

    def test_on_weekday_is_read_the_same_as_the_bare_form(self):
        assert resolve_date("on wednesday")["start"] == "2026-08-05"

    def test_weekday_abbreviations_are_understood(self):
        assert resolve_date("tue")["start"] == "2026-08-04"


class TestExplicitDates:
    def test_iso_date(self):
        result = resolve_date("2026-08-20")
        assert (result["start"], result["kind"]) == ("2026-08-20", "exact")

    def test_day_then_month_name(self):
        assert resolve_date("20 August")["start"] == "2026-08-20"

    def test_month_name_then_day(self):
        assert resolve_date("August 20")["start"] == "2026-08-20"

    def test_an_explicit_date_already_past_this_year_rolls_to_next_year(self):
        """"3 January" said in August means the January that is coming."""
        assert resolve_date("3 January")["start"] == "2027-01-03"


class TestPartOfDay:
    def test_morning_is_captured_alongside_the_date(self):
        result = resolve_date("tomorrow morning")
        assert result["start"] == "2026-08-04"
        assert result["part_of_day"] == "morning"

    def test_afternoon(self):
        assert resolve_date("next week afternoon")["part_of_day"] == "afternoon"

    def test_no_part_of_day_is_none_not_a_default(self):
        """Defaulting to morning would silently narrow what the patient asked
        for, and they would never see the choice being made."""
        assert resolve_date("next week")["part_of_day"] is None


class TestRefusals:
    def test_a_past_date_is_refused(self):
        result = resolve_date("2026-07-01")
        assert result["resolved"] is False
        assert result["reason"] == "in_the_past"

    def test_yesterday_is_refused(self):
        assert resolve_date("yesterday")["resolved"] is False

    def test_nonsense_is_refused_rather_than_guessed(self):
        """Returning a plausible date for an unparseable phrase is exactly the
        fabrication this tool exists to prevent."""
        result = resolve_date("sometime soonish maybe")
        assert result["resolved"] is False
        assert result["reason"] == "unparseable"

    def test_an_empty_phrase_is_refused(self):
        assert resolve_date("")["resolved"] is False

    def test_an_impossible_calendar_date_is_refused(self):
        assert resolve_date("2026-02-30")["resolved"] is False

    def test_a_refusal_carries_no_dates_at_all(self):
        """A caller reading start/end without checking `resolved` must not find
        a usable-looking value there."""
        result = resolve_date("gibberish")
        assert result["start"] is None
        assert result["end"] is None


class TestContract:
    def test_the_phrase_is_echoed_back_for_the_trace(self):
        assert resolve_date("next week")["phrase"] == "next week"

    def test_case_and_surrounding_whitespace_do_not_matter(self):
        assert resolve_date("  NEXT WEEK  ")["start"] == "2026-08-10"

    def test_a_label_is_supplied_for_replies(self):
        """Confirmation wording reads this rather than formatting dates itself."""
        assert "10 August" in resolve_date("next week")["label"]

    def test_the_result_is_json_serialisable(self):
        """Tool results are persisted in traces and diffed by the golden set."""
        import json

        json.dumps(resolve_date("next week"))
        json.dumps(resolve_date("gibberish"))

    def test_every_result_carries_the_same_keys(self):
        """A resolved and a refused result must be shape-identical, so callers
        never branch on which keys happen to exist."""
        assert set(resolve_date("next week")) == set(resolve_date("gibberish"))

    def test_resolution_follows_the_clock_seam(self):
        clock.freeze(date(2027, 1, 4))
        assert resolve_date("today")["start"] == "2027-01-04"
