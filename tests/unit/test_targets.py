"""Which appointment the patient meant — the cue reader, on its own.

Written against a live defect with a write behind it. With a Monday Dermatology
appointment and a Thursday Orthopedics one on file, "I wanted to reschedule the
appointment on Monday to Wednesday" reached ``propose_reschedule`` as the
*Thursday* appointment's id. Every guard around it passed, because every guard
around it asks whether an id is usable and none of them asks whether it is the
one that was named. The Thursday appointment moved.

The two cases that carry the design are here and neither is an edge case:

* **"Monday to Wednesday" names two weekdays.** Matching is over *rows*, not
  over phrases, so the weekday the patient has an appointment on wins and the
  one they are moving to contributes nothing.
* **Two appointments on the same Monday** is a real ambiguity, and the answer is
  to ask rather than to pick the first.
"""

from __future__ import annotations

import pytest

from datetime import date

from app.workflow.targets import read_choice, resolve_target

#: Monday 3 August 2026, Dermatology. The one the live patient meant.
MONDAY = {
    "appointment_id": 2,
    "reference_code": "AC-000002",
    "department_id": 3,
    "department_name": "Dermatology",
    "start": "2026-08-03T09:00:00",
    "end": "2026-08-03T09:30:00",
}

#: Thursday 30 July 2026, Orthopedics. The one that actually moved.
THURSDAY = {
    "appointment_id": 3,
    "reference_code": "AC-000003",
    "department_id": 2,
    "department_name": "Orthopedics",
    "start": "2026-07-30T15:00:00",
    "end": "2026-07-30T15:30:00",
}

BOTH = [THURSDAY, MONDAY]


#: The day the live session ran. Passed explicitly rather than left to the
#: clock: the appointments above carry fixed dates, and "3 August" only means
#: 2026 to a reader standing before it. A test whose cue resolution drifts with
#: the calendar reports on the date rather than on the code.
TODAY = date(2026, 7, 28)


@pytest.fixture
def resolve(seeded_db):
    def _resolve(message, appointments=BOTH):
        return resolve_target(
            seeded_db, message=message, appointments=appointments, today=TODAY
        )

    return _resolve


class TestTheLiveFailure:
    """The exact sentence, and the exact pair of appointments."""

    MESSAGE = "I wanted to reschedule the appointment on Monday to Wednesday? any available slots?"

    def test_monday_is_the_target(self, resolve):
        assert resolve(self.MESSAGE).appointment_id == MONDAY["appointment_id"]

    def test_the_destination_weekday_is_not_a_second_candidate(self, resolve):
        """Wednesday is where they are going, not what they have. Counting
        matched *rows* rather than matched *words* is what makes that fall out
        without a rule about destinations."""
        assert resolve(self.MESSAGE).candidates == [MONDAY["appointment_id"]]

    def test_the_reason_says_a_cue_decided_it(self, resolve):
        assert resolve(self.MESSAGE).reason == "matched"

    def test_both_weekdays_are_recorded_as_cues(self, resolve):
        """The trace has to show what was read, or an override is an assertion
        nobody can check. Monday is 0 and Wednesday is 2."""
        assert resolve(self.MESSAGE).cues[:2] == ["weekday:0", "weekday:2"]


class TestEachCueOnItsOwn:
    def test_a_weekday(self, resolve):
        assert resolve("cancel my thursday appointment").appointment_id == 3

    def test_a_department_name(self, resolve):
        assert resolve("cancel my dermatology appointment").appointment_id == 2

    def test_a_department_synonym(self, resolve):
        """The Department table decides what a department word is — the same
        source routing and the new-subject rule read, so there is no keyword
        list here to drift from theirs."""
        assert resolve("reschedule the appointment about my skin").appointment_id == 2

    def test_a_reference_code(self, resolve):
        assert resolve("please cancel AC-000003").appointment_id == 3

    def test_a_reference_code_without_its_dash(self, resolve):
        assert resolve("please cancel ac000003").appointment_id == 3

    def test_a_written_date(self, resolve):
        assert resolve("move the 3 August appointment").appointment_id == 2

    def test_a_written_date_month_first(self, resolve):
        assert resolve("move the August 3rd appointment").appointment_id == 2

    def test_an_iso_date(self, resolve):
        assert resolve("move the 2026-07-30 appointment").appointment_id == 3


class TestWhenNobodyKnows:
    def test_no_cue_at_all_is_not_a_target(self, resolve):
        verdict = resolve("please reschedule my appointment")
        assert verdict.appointment_id is None
        assert verdict.reason == "no_cue"

    def test_two_appointments_on_the_same_weekday_is_ambiguous(self, seeded_db):
        """A real ambiguity, and the answer is to ask. Picking the earlier one
        would move an appointment the patient never named."""
        other_monday = dict(THURSDAY, appointment_id=4, start="2026-08-03T14:00:00")
        verdict = resolve_target(
            seeded_db, message="reschedule my monday appointment",
            appointments=[MONDAY, other_monday],
        )
        assert verdict.appointment_id is None
        assert verdict.reason == "ambiguous"
        assert sorted(verdict.candidates) == [2, 4]

    def test_a_cue_naming_neither_is_no_cue(self, resolve):
        verdict = resolve("reschedule my saturday appointment")
        assert verdict.appointment_id is None
        assert verdict.reason == "no_cue"


class TestWhenThereIsNothingToChooseBetween:
    def test_one_appointment_needs_no_cue(self, resolve):
        verdict = resolve("please reschedule my appointment", [MONDAY])
        assert verdict.appointment_id == 2
        assert verdict.reason == "only_one"

    def test_no_appointments_at_all(self, resolve):
        verdict = resolve("please reschedule my appointment", [])
        assert verdict.appointment_id is None
        assert verdict.reason == "no_cue"


class TestTheAnswerToANumberedList:
    """``read_choice`` — the other half, and the half that was missing.

    ``render_appointment_choice`` numbers the rows *so that* "2" means
    something, and then nothing in code read the answer. Live, a cancel run
    asked "which one? 1..., 2...", the patient answered "2", and the run
    **failed**: the Coordinator called the answer a new cancellation request,
    the supersede was refused for naming no new subject, and the recovery
    searched for slots on a run that had no department to search in.
    """

    LISTED = [2, 3]

    def test_a_bare_number(self):
        assert read_choice("2", listed=self.LISTED) == 3

    def test_the_first_row(self):
        assert read_choice("1", listed=self.LISTED) == 2

    def test_an_announced_position(self):
        assert read_choice("option 1", listed=self.LISTED) == 2

    def test_an_ordinal_word(self):
        assert read_choice("the second one please", listed=self.LISTED) == 3

    def test_a_leading_position_with_an_instruction_after_it(self):
        """The live sentence. A number at the front of a list answer is the
        answer; the date at the back is what to do next."""
        assert (
            read_choice(
                "1. Move it to after I finish my General Medicine appointment "
                "which is on 6th August at 10am",
                listed=self.LISTED,
            )
            == 2
        )

    def test_a_leading_position_in_brackets(self):
        assert read_choice("2) and please make it quick", listed=self.LISTED) == 3

    def test_a_number_outside_the_list_is_not_rounded_into_it(self):
        """It goes to the model, which is where a message nobody anticipated
        belongs — rounding 7 down to 2 would cancel an appointment on a
        misreading."""
        assert read_choice("7", listed=self.LISTED) is None

    def test_a_time_at_the_front_is_not_a_position(self):
        assert read_choice("1.30pm suits me", listed=self.LISTED) is None

    def test_a_sentence_naming_no_position(self):
        assert read_choice("cancel the dermatology one", listed=self.LISTED) is None

    def test_a_withdrawal_is_never_a_choice(self):
        """"Never mind" ends the request; read as a position it would propose
        cancelling an appointment instead."""
        assert read_choice("never mind, forget it", listed=self.LISTED) is None

    def test_nothing_listed_means_nothing_to_answer(self):
        """An empty list imposes nothing: "2" against no listing is a number."""
        assert read_choice("2", listed=[]) is None
