"""The confirmation reader and the history window.

Two small deterministic modules, both of which fail quietly when wrong.

The reader decides whether a patient consented. Its bias is fixed by a cost
asymmetry: a wrongly re-asked "yes" costs one tap; a wrongly committed "no"
books an appointment against the patient's word at the exact step built to
prevent that. So it reads exact tokens and hands everything else on.

The window decides how much history the model sees. It is the boundedness
invariant applied to a writer that hides behind the framework: history is
appended every turn, never trimmed, and sent whole — so prompts grow toward the
context limit until something breaks a long way from here.
"""

from __future__ import annotations

import pytest
from google.genai import types

from app.agents.memory import window_contents
from app.workflow.confirmation import ConfirmationAnswer, normalise, read_confirmation

CONFIRM = ConfirmationAnswer.CONFIRM
DECLINE = ConfirmationAnswer.DECLINE
UNREAD = ConfirmationAnswer.UNREAD


class TestExactTokensCommit:
    @pytest.mark.parametrize(
        "text", ["yes", "Yes", "YES", "yes!", "  yes  ", "confirm", "ok", "book it"]
    )
    def test_a_plain_yes_confirms(self, text):
        assert read_confirmation(text) is CONFIRM

    @pytest.mark.parametrize("text", ["no", "No", "nope", "cancel", "decline", "stop"])
    def test_a_plain_no_declines(self, text):
        assert read_confirmation(text) is DECLINE


class TestAnythingElseIsHandedOn:
    @pytest.mark.parametrize(
        "text",
        [
            "no wait - yes, the Tuesday one",
            "yes, but can we do Tuesday instead?",
            "yes to the time but not that doctor",
            "yesterday would have been better",
            "I think so?",
            "maybe",
            "",
            "sure, whatever",
        ],
    )
    def test_it_is_not_read_as_a_confirmation(self, text):
        """Each of these *contains* something yes-shaped and means something
        else. A reader that matched on containment would commit all of them."""
        assert read_confirmation(text) is not CONFIRM

    def test_a_question_wearing_a_yes_is_unread(self):
        assert read_confirmation("yes, but can we do Tuesday instead?") is UNREAD

    def test_a_long_refusal_is_not_silently_a_decline_either(self):
        """Handing it on is right in both directions: the model may re-ask."""
        assert read_confirmation("no, actually, hold on, let me think") is UNREAD

    def test_normalise_strips_punctuation_only(self):
        assert normalise("Yes!!") == "yes"
        assert normalise("yes, please") == "yes please"


class TestHistoryWindow:
    @staticmethod
    def _conversation(turns: int) -> list:
        contents = []
        for index in range(turns):
            contents.append(
                types.Content(role="user", parts=[types.Part(text=f"user {index}")])
            )
            contents.append(
                types.Content(role="model", parts=[types.Part(text=f"model {index}")])
            )
        return contents

    def test_a_short_conversation_is_returned_whole(self):
        contents = self._conversation(3)
        assert window_contents(contents, turns=15) is contents

    def test_a_long_conversation_is_trimmed_to_the_last_n_turns(self):
        contents = self._conversation(40)
        windowed = window_contents(contents, turns=15)
        users = [c for c in windowed if c.role == "user"]
        assert len(users) == 15

    def test_the_window_keeps_the_most_recent_exchange(self):
        contents = self._conversation(40)
        windowed = window_contents(contents, turns=15)
        assert windowed[-1].parts[0].text == "model 39"

    def test_the_window_starts_on_a_user_message(self):
        """Cutting mid-turn would hand the model a tool result whose call it
        cannot see."""
        contents = self._conversation(40)
        assert window_contents(contents, turns=15)[0].role == "user"

    def test_a_zero_window_disables_trimming(self):
        contents = self._conversation(40)
        assert window_contents(contents, turns=0) is contents

    def test_an_empty_history_is_safe(self):
        assert window_contents([], turns=15) == []

    def test_windowing_is_bounded_regardless_of_length(self):
        """The property that matters: the prompt stops growing."""
        short = window_contents(self._conversation(20), turns=15)
        long = window_contents(self._conversation(2000), turns=15)
        assert len(short) == len(long)
