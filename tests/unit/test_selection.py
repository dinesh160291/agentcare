"""Reading a choice among times already shown — the reader, on its own.

The seam these test is a pure function: words in, a slot id or ``None`` out. No
database, no run, no model. The end-to-end consequences live in
``test_orchestrator.py``; what is pinned here is the reading itself, in both
directions, because the two directions cost very different things:

* **a missed selection** costs one more message — the turn carries on to the
  Coordinator exactly as it did before this module existed;
* **a wrong selection** holds a time the patient did not choose. It still
  cannot book — only an exact "yes" or the ✅ button does that — so it costs a
  decline. That gate is the whole reason this may read more freely than
  :func:`read_confirmation` does.

The live transcript is the source of the phrasings. Seven consecutive messages
to one run — "okay lets book at 3pm then", "lets do 3pm and book it", "option
2", "3pm will work for me", "2pm will work for me", "confirm 2pm slot", "close
the previous request" — each drew the identical availability list back. Every
one of them appears below.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.workflow.selection import Offer, read_selection

#: A numbered list as the patient read it: 2pm, 3pm, 4pm on one afternoon.
SHORTLIST = [
    Offer(slot_id=101, start=datetime(2026, 7, 28, 14, 0)),
    Offer(slot_id=102, start=datetime(2026, 7, 28, 15, 0)),
    Offer(slot_id=103, start=datetime(2026, 7, 28, 16, 0)),
]

#: Everything the run ever offered — the union, including an earlier morning
#: page the patient scrolled past.
OFFERED = [
    Offer(slot_id=201, start=datetime(2026, 7, 27, 9, 0)),
    Offer(slot_id=202, start=datetime(2026, 7, 27, 10, 30)),
] + SHORTLIST


def read(text, *, shortlist=SHORTLIST, offered=OFFERED):
    return read_selection(text, shortlist=shortlist, offered=offered)


class TestATimeThePatientWasShown:
    """The sentences that drew an availability list back, seven times running."""

    @pytest.mark.parametrize(
        "message, expected",
        [
            ("okay lets book at 3pm then", 102),
            ("lets do 3pm and book it", 102),
            ("3pm will work for me", 102),
            ("2pm will work for me", 101),
            ("confirm 2pm slot", 101),
            ("lets book the 10am slot", None),  # 10:00 is not on any list here
            ("4pm", 103),
            ("the 4 pm one", 103),
            ("can we do 3:00 PM", 102),
            ("I'll take 15:00", 102),
            ("book me in at 4 p.m.", 103),
        ],
    )
    def test_a_named_time_finds_its_slot(self, message, expected):
        assert read(message) == expected

    def test_a_time_only_on_the_wider_list_still_lands(self):
        """"The 10:30 one you showed me earlier" — the shortlist has moved on,
        the union has not, and the union is why it is a union."""
        assert read("actually the 10:30 one you showed me earlier") == 202

    def test_a_time_on_two_offered_days_is_not_a_choice(self):
        """Two slots at the same clock time on different days is a question the
        patient has not answered. Guessing a day is the expensive direction, so
        this falls through to the model rather than picking the earlier one."""
        offered = OFFERED + [Offer(slot_id=301, start=datetime(2026, 7, 29, 9, 0))]
        assert read("9am please", shortlist=[], offered=offered) is None

    def test_the_shortlist_wins_over_the_wider_list(self):
        """The list in front of the patient is the referent. A 3pm on an older
        page must not outrank the 3pm they are looking at."""
        offered = [Offer(slot_id=999, start=datetime(2026, 7, 20, 15, 0))] + OFFERED
        assert read("3pm", offered=offered) == 102


class TestAListNumber:
    """"2" means something only against a list somebody recorded."""

    @pytest.mark.parametrize(
        "message, expected",
        [
            ("option 2", 102),
            ("2", 102),
            ("number 3", 103),
            ("the 2nd one", 102),
            ("the second one please", 102),
            ("first", 101),
            ("#3", 103),
            ("option 1", 101),
        ],
    )
    def test_a_position_finds_its_row(self, message, expected):
        assert read(message) == expected

    def test_a_position_past_the_end_is_refused(self):
        """Not rounded into range. A number nobody offered is a message this
        cannot read, and the model is where those belong."""
        assert read("option 7") is None

    def test_a_position_with_no_list_recorded_reads_nothing(self):
        """An empty shortlist imposes nothing — the same rule
        ``listed_appointment_ids`` uses. No list has been shown, so no number
        can refer to one."""
        assert read("option 2", shortlist=[]) is None

    def test_a_bare_number_must_be_the_whole_message(self):
        """A "2" inside a sentence is far more likely a date, a count or a
        reference than a row. Only a message that *is* the number is read."""
        assert read("I have 2 appointments already") is None

    def test_a_number_that_is_an_hour_is_read_as_an_hour(self):
        """"book the 2pm one" contains a 2 that is not a row number. The time
        is parsed first precisely so the digit cannot be stolen from it —
        without that, this returns slot 101 by accident and is right for the
        wrong reason on this list."""
        assert read("book the 2pm one") == 101
        # And on a shortlist where position and hour disagree, the hour wins.
        shortlist = [
            Offer(slot_id=501, start=datetime(2026, 7, 28, 16, 0)),
            Offer(slot_id=502, start=datetime(2026, 7, 28, 17, 0)),
            Offer(slot_id=503, start=datetime(2026, 7, 28, 14, 0)),
        ]
        assert read("book the 2pm one", shortlist=shortlist, offered=shortlist) == 503


class TestWhatIsNeverASelection:
    """The false-positive direction, one named case per way in."""

    @pytest.mark.parametrize(
        "message",
        [
            "close the previous request",
            "never mind",
            "forget it",
            "cancel that request",
            "actually I changed my mind, 3pm instead",
        ],
    )
    def test_a_withdrawal_cue_is_never_a_choice(self, message):
        """Checked before anything else. "Forget it" and "book it" must not be
        read by the same rule in the same breath — and the last case is why
        containment, not exactness, is the right test *here*: a message that
        mentions leaving is not a message choosing a time, whatever else it
        says."""
        assert read(message) is None

    def test_two_times_are_a_range_not_an_answer(self):
        """"Anything between 2pm and 4pm" is a question about availability. It
        already has a path; this one would answer it wrongly."""
        assert read("anything between 2pm and 4pm") is None

    @pytest.mark.parametrize(
        "message",
        [
            "",
            "   ",
            "what documents do I need to bring?",
            "who is the doctor?",
            "can you tell me my appointments",
            "is there parking at the hospital",
        ],
    )
    def test_a_message_that_names_no_time_and_no_position_reads_nothing(self, message):
        assert read(message) is None

    def test_nothing_offered_means_nothing_selectable(self):
        assert read("3pm", shortlist=[], offered=[]) is None

    def test_a_reference_code_is_not_a_time(self):
        """AC-000003 ends in a digit and must not become row 3."""
        assert read("what about AC-000003") is None
