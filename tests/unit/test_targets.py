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

from app.workflow.targets import resolve_target

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
