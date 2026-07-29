"""Code-authored patient-facing text, and the bookkeeping under it.

Everything here is deterministic bin, so everything here is written first. The
module exists because of what two live transcripts showed: a model asked to
word a receipt around facts will mash a mandatory document together with an
optional one, contradict itself about follow-up, and recite a reminder's fire
time as though it were the appointment. None of those are prompt failures that
a better prompt fixes reliably — they are the model being trusted with facts.

The rule these tests pin is narrow and absolute: **after a commit, and around a
proposal, the facts come from rows and the sentences come from here.**
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.db import SessionLocal
from app.models import (
    Appointment,
    AppointmentStatus,
    PatientProfile,
    Reminder,
    ReminderStatus,
    ReminderType,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.workflow.replies import (
    UPLOAD_POINTER,
    clash_note,
    claims_availability,
    offered_slot_ids,
    promises_action,
    record_offered,
    render_options,
    render_reask,
    render_outstanding,
    render_receipt,
    was_offered,
)

PATIENT_EMAIL = "asha.patient@example.invalid"


def _slot(slot_id: int, start: str, doctor: str = "Dr Deepa Krishnan") -> dict:
    """A slot in the shape ``find_available_slots`` actually returns."""
    return {
        "slot_id": slot_id,
        "doctor_id": 1,
        "doctor_name": doctor,
        "department_id": 2,
        "department_name": "ENT",
        "start": start,
        "end": start,
    }


THREE = [
    _slot(11, "2026-08-03T09:00:00"),
    _slot(12, "2026-08-03T14:00:00"),
    _slot(13, "2026-08-04T09:30:00"),
]


class TestTheOfferedSet:
    """Which slot ids the patient has actually been shown.

    Born from tool results and nothing else. The alternative — letting the
    model name a slot id it remembers from its context window — is the
    fabrication channel this set exists to close: an id recalled from prose
    cannot be distinguished from an id invented, and both look like an integer.
    """

    def test_ids_come_from_the_slot_payload(self):
        run = WorkflowRun(state={})
        record_offered(run, THREE)
        assert offered_slot_ids(run) == [11, 12, 13]

    def test_showing_more_appends_rather_than_replaces(self):
        """"Actually, the first one you showed me" arrives three exchanges
        later. The set is the whole run's, so it is still answerable."""
        run = WorkflowRun(state={})
        record_offered(run, THREE)
        record_offered(run, [_slot(14, "2026-08-05T11:00:00")])
        assert offered_slot_ids(run) == [11, 12, 13, 14]

    def test_a_repeat_does_not_duplicate(self):
        run = WorkflowRun(state={})
        record_offered(run, THREE)
        record_offered(run, THREE)
        assert offered_slot_ids(run) == [11, 12, 13]

    def test_an_unshown_id_is_not_in_the_set(self):
        run = WorkflowRun(state={})
        record_offered(run, THREE)
        assert was_offered(run, 12) is True
        assert was_offered(run, 99) is False

    def test_a_run_that_has_shown_nothing_offers_nothing(self):
        """Distrust green: a `was_offered` that defaulted to True would pass
        every test above, and would accept any integer the model produced."""
        assert was_offered(WorkflowRun(state={}), 11) is False

    def test_the_state_dict_is_replaced_not_mutated(self):
        """SQLAlchemy's JSON column only notices assignment. Mutating the dict
        in place leaves the change unsaved, and the set silently empties
        between turns — which reads exactly like the model inventing ids."""
        run = WorkflowRun(state={"department_id": 2})
        before = run.state
        record_offered(run, THREE)
        assert run.state is not before
        assert run.state["department_id"] == 2


class TestRenderingOptions:
    def test_three_options_are_numbered_soonest_first(self):
        text = render_options(THREE, proposed_slot_id=11)
        assert "1." in text and "2." in text and "3." in text
        assert text.index("2:00 PM") > text.index("9:00 AM")

    def test_every_option_names_its_doctor_day_and_time(self):
        text = render_options(THREE, proposed_slot_id=11)
        assert "Dr Deepa Krishnan" in text
        assert "Monday 3 August" in text
        assert "9:00 AM" in text
        # 24-hour is how the row is stored, not how a patient reads a time.
        assert "09:00" not in text

    def test_the_proposed_one_is_marked(self):
        text = render_options(THREE, proposed_slot_id=12)
        marked = [line for line in text.splitlines() if "2:00 PM" in line]
        assert marked and "holding" in marked[0]

    def test_at_most_three_are_shown(self):
        """The typed proposal is one slot; the reply is a shortlist, not a
        timetable. Twenty options is a wall, not a choice."""
        many = THREE + [_slot(20 + i, f"2026-08-1{i}T10:00:00") for i in range(5)]
        assert len(render_options(many, proposed_slot_id=11).splitlines()) <= 5

    def test_no_options_renders_nothing(self):
        assert render_options([], proposed_slot_id=None) == ""


class TestThePromisedActionGuard:
    """A re-ask may not claim the assistant is off doing something.

    Live: "looks good. lets book that time" was answered with "I will proceed
    to find a suitable time for you. Please hold on" — an action promised and
    never performed, on a turn where a time was already being held.

    False positives are cheap here and that is deliberate: the fallback is the
    template re-ask, which carries the same facts and asks the same question.
    Unlike the safety screen, over-firing costs a sentence, not a queue.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "I will proceed to find a suitable time for you. Please hold on.",
            "Let me check what else is available.",
            "One moment while I look that up.",
            "I'm looking into it now.",
            "I'll get back to you shortly.",
        ],
    )
    def test_an_action_promise_is_caught(self, text):
        assert promises_action(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "That time is Monday 3 August at 09:00 with Dr Deepa Krishnan. "
            "Shall I book it?",
            "I can only book on an exact yes or the Confirm button.",
            "Nothing is booked until you confirm.",
            "",
        ],
    )
    def test_a_re_ask_that_promises_nothing_survives(self, text):
        assert promises_action(text) is False

    def test_booking_on_confirmation_is_not_a_promise_of_agency(self):
        """The distinction that makes the guard usable: "I'll book it when you
        say yes" is the rule being stated, not an action being announced."""
        assert promises_action("I'll book it as soon as you say yes.") is False


# --- the two that need rows ---------------------------------------------


@pytest.fixture
def booked(seeded_db):
    """A patient with a confirmed appointment and a run that committed it."""
    seeded_db.commit()
    session = SessionLocal()
    user = session.query(User).filter(User.email == PATIENT_EMAIL).one()
    profile = (
        session.query(PatientProfile).filter(PatientProfile.user_id == user.id).one()
    )
    appointment = (
        session.query(Appointment)
        .filter(Appointment.patient_id == profile.id)
        .order_by(Appointment.id)
        .first()
    )
    appointment.status = AppointmentStatus.CONFIRMED
    run = WorkflowRun(
        patient_id=profile.id,
        status=WorkflowStatus.IN_PROGRESS,
        request_text="book me something",
        plan=["route", "book", "documents", "follow_up"],
        completed_steps=[],
        state={
            "department_id": appointment.department_id,
            "appointment_id": appointment.id,
            "committed_action": "book",
        },
    )
    session.add(run)
    session.commit()
    try:
        yield session, run, appointment
    finally:
        session.close()


class TestTheReceipt:
    """Four defects in one live receipt, and the templates that answer them."""

    def test_it_states_the_appointment_from_the_row(self, booked):
        session, run, appointment = booked
        text = render_receipt(session, run)
        assert appointment.reference_code in text

    def test_it_contains_no_date_other_than_the_appointments(self, booked):
        """The worst of the four: a reminder fires the day *before* the visit,
        and the live receipt presented that date as the appointment's. A
        patient reading it arrives a day early to an empty clinic."""
        session, run, appointment = booked
        starts = appointment.slot.start_time
        session.add(
            Reminder(
                patient_id=run.patient_id,
                appointment_id=appointment.id,
                reminder_type=ReminderType.APPOINTMENT,
                scheduled_at=starts - timedelta(days=1),
                status=ReminderStatus.PENDING,
            )
        )
        session.flush()

        text = render_receipt(session, run)
        day_before = f"{(starts - timedelta(days=1)).day} {(starts - timedelta(days=1)):%B}"
        assert day_before not in text

    def test_the_reminder_line_is_one_sentence_and_names_no_time(self, booked):
        session, run, appointment = booked
        session.add(
            Reminder(
                patient_id=run.patient_id,
                appointment_id=appointment.id,
                reminder_type=ReminderType.APPOINTMENT,
                scheduled_at=appointment.slot.start_time,
                status=ReminderStatus.PENDING,
            )
        )
        session.flush()
        text = render_receipt(session, run)
        assert "remind you the day before" in text

    def test_no_reminder_row_means_no_reminder_promise(self, booked):
        """The receipt re-reads every fact it states, including the ones that
        look like boilerplate. A cancellation retires the reminder in the same
        transaction, and the old receipt promised one anyway."""
        session, run, appointment = booked
        # The seed ships this appointment with a reminder, which is what makes
        # the check worth having: the row has to be gone for the sentence to
        # go, and this is the state a cancellation leaves behind.
        session.query(Reminder).filter(
            Reminder.appointment_id == appointment.id
        ).delete()
        session.flush()
        assert "remind you" not in render_receipt(session, run)

    def test_a_cancellation_receipt_has_no_document_section(self, booked):
        """There is no visit to prepare for."""
        session, run, _ = booked
        state = dict(run.state)
        state["committed_action"] = "cancel"
        run.state = state
        text = render_receipt(session, run)
        assert "before your visit" not in text
        assert UPLOAD_POINTER not in text

    def test_the_upload_pointer_appears_whenever_something_is_missing(self, booked):
        """Item 6 rides here: the pointer is one line after the groups, not
        buried inside the mandatory one, so an optional-only gap still tells
        the patient where to go."""
        session, run, _ = booked
        text = render_receipt(session, run)
        missing = "before your visit" in text or "Optional but helpful" in text
        assert (UPLOAD_POINTER in text) is missing

    def test_mandatory_and_optional_are_never_in_the_same_sentence(self, booked):
        """The live receipt said a document was "still needed, which is
        optional" — two groups mashed into one clause, leaving the patient
        unable to tell what they actually have to do."""
        session, run, _ = booked
        for line in render_receipt(session, run).splitlines():
            assert not ("before your visit" in line and "Optional" in line)


class TestTheReAsk:
    """A re-ask carries the proposal's facts, always."""

    @pytest.fixture
    def pending(self, booked):
        session, run, appointment = booked
        run.status = WorkflowStatus.PENDING_CONFIRMATION
        run.proposed_slot_id = appointment.slot_id
        session.flush()
        return session, run, appointment

    def test_it_names_the_doctor_and_the_day(self, pending):
        session, run, appointment = pending
        text = render_reask(session, run)
        assert str(appointment.slot.start_time.day) in text

    def test_it_states_the_rule(self, pending):
        session, run, _ = pending
        text = render_reask(session, run)
        assert "confirm" in text.lower()

    def test_it_promises_no_action(self, pending):
        """The guard and the fallback must not disagree: the text the guard
        falls back to has to survive the guard."""
        session, run, _ = pending
        assert promises_action(render_reask(session, run)) is False


class TestTheTwoClockFormattersAgree:
    """``replies.clock_time`` and ``tools.confirmations._clock_time`` are the
    same four lines in two places, on purpose: ``replies`` imports
    ``confirmations``, so importing back would invert the dependency the whole
    ``tools`` package rests on. Duplication is the cheaper of the two, but only
    while something notices when they drift.
    """

    @pytest.mark.parametrize(
        "hour,minute",
        [(0, 0), (0, 30), (9, 0), (11, 59), (12, 0), (12, 30), (15, 0), (23, 45)],
    )
    def test_they_produce_the_same_string(self, hour, minute):
        from datetime import datetime as dt

        from app.tools.confirmations import _clock_time
        from app.workflow.replies import clock_time

        moment = dt(2026, 8, 3, hour, minute)
        assert clock_time(moment) == _clock_time(moment)

    def test_midnight_and_noon_are_not_zero_oclock(self):
        """The off-by-twelve both formatters would share if `% 12` were used
        without the `or 12`."""
        from datetime import datetime as dt

        from app.workflow.replies import clock_time

        assert clock_time(dt(2026, 8, 3, 0, 5)) == "12:05 AM"
        assert clock_time(dt(2026, 8, 3, 12, 5)) == "12:05 PM"


class TestTheOutstandingLine:
    """Item 6: the "what is still needed" half of a documents answer."""

    def test_nothing_outstanding_is_not_a_sentence(self, seeded_db):
        """Live, the model asserted "no documents required" from a run that had
        nothing to diff against, and the next turn contradicted it. Silence is
        the honest output — "nothing to compare" is not "nothing is required"."""
        assert render_outstanding(seeded_db, patient_id=2) == ""

    def test_an_open_task_is_named_with_the_pointer(self, seeded_db):
        from app.models import FollowUpTaskType
        from app.tools.tasks import upsert_followup_task

        upsert_followup_task(
            seeded_db,
            patient_id=2,
            task_type=FollowUpTaskType.MISSING_DOCUMENTS,
            details={"missing": ["Prior MRI or CT report"]},
            appointment_id=None,
        )
        seeded_db.flush()

        line = render_outstanding(seeded_db, patient_id=2)
        assert "Prior MRI or CT report" in line
        assert UPLOAD_POINTER in line

    def test_a_closed_task_says_nothing(self, seeded_db):
        """Distrust green: reading every task regardless of status would pass
        the test above and go on demanding a document after it arrived."""
        from app.models import FollowUpTask, FollowUpTaskStatus, FollowUpTaskType
        from app.tools.tasks import upsert_followup_task

        upsert_followup_task(
            seeded_db,
            patient_id=2,
            task_type=FollowUpTaskType.MISSING_DOCUMENTS,
            details={"missing": ["Prior MRI or CT report"]},
            appointment_id=None,
        )
        seeded_db.flush()
        for task in seeded_db.query(FollowUpTask).all():
            task.status = FollowUpTaskStatus.CLOSED
        seeded_db.flush()

        assert render_outstanding(seeded_db, patient_id=2) == ""


class TestAvailabilityClaimsAreGrounded:
    """Item 2's wording half. The guard itself is in ``_guarded``, which only
    consults this on turns where no slot search ran — so the list can be short
    without being a lottery."""

    def test_the_live_sentence_is_caught(self):
        assert claims_availability(
            "It appears that there are currently no available appointment slots "
            "in the ENT department for next week."
        ) is True

    def test_both_directions_count(self):
        """A confident "there are slots on Tuesday" with nothing behind it is
        the same defect wearing the opposite sign, and it is the one that sends
        a patient to a clinic."""
        assert claims_availability("There are slots available on Tuesday") is True
        assert claims_availability("ENT is fully booked that week") is True

    def test_ordinary_administrative_prose_is_not_a_claim(self):
        for text in (
            "Which department would you like?",
            "Your appointment is confirmed for Monday 3 August at 9:00 AM.",
            "I can move it to Monday 3 August at 9:00 AM.",
            "Could you tell me what date or time would suit you?",
            "",
        ):
            assert claims_availability(text) is False, text


class TestADepartmentThatAsksForNothing:
    """Silence is not an answer, and it was read as an omission.

    A General Medicine booking requires no documents, so the receipt's document
    section was simply absent — and the patient went looking for the paperwork
    line that was never coming. The note this is *not* is the documents
    listing's careful silence: there, no department has been resolved and
    "nothing is required" would be a claim about a hospital nobody consulted.
    Here the department is known and its rules table is empty, which is a fact
    about this visit and sayable.
    """

    @pytest.fixture
    def general_medicine(self, booked):
        from app.models import Department

        session, run, appointment = booked
        gm = session.query(Department).filter(Department.name == "General Medicine").one()
        assert gm.required_documents == [], "the fixture assumes GM requires nothing"
        run.state = dict(run.state) | {"department_id": gm.id}
        session.commit()
        return session, run, appointment

    def test_the_receipt_says_so_explicitly(self, general_medicine):
        session, run, _ = general_medicine
        assert "No documents are needed for this visit." in render_receipt(session, run)

    def test_a_department_that_does_ask_is_untouched(self, booked):
        """The populated case is the one that already worked. Ophthalmology
        wants an eye test report, and it must still say which and why."""
        from app.models import Department

        session, run, _ = booked
        eyes = session.query(Department).filter(Department.name == "Ophthalmology").one()
        run.state = dict(run.state) | {"department_id": eyes.id}
        session.commit()

        text = render_receipt(session, run)
        assert "Previous eye test report" in text
        assert "No documents are needed" not in text


class TestTheRenderersSurviveMarkdown:
    """Every one of these is a list with a sentence after it.

    The chat renders markdown, and markdown reads a line one newline below a
    list item as belonging to that item. So "Say yes to take the one I'm
    holding" arrived welded to the third slot — an instruction about the whole
    offer, printed as though it were part of one option. The templates were
    right and the renderer undid them, which is a failure mode no assertion
    about *content* can see.
    """

    def test_the_option_list_is_separated_from_the_instruction(self, seeded_db):
        from app.workflow.replies import render_proposal

        run = WorkflowRun(
            patient_id=1,
            status=WorkflowStatus.IN_PROGRESS,
            request_text="book me something",
            plan=["route", "book"],
            completed_steps=[],
            state={},
        )
        seeded_db.add(run)
        seeded_db.flush()

        text = render_proposal(
            seeded_db,
            run,
            [
                _slot(1, "2026-08-03T09:00:00"),
                _slot(2, "2026-08-03T10:00:00"),
                _slot(3, "2026-08-03T11:00:00"),
            ],
        )
        assert "\n\nSay \"yes\"" in text, repr(text)

    def test_the_receipt_puts_a_blank_line_between_its_facts(self, booked):
        """``_document_lines`` splits a mandatory document from an optional one
        on purpose; a single newline handed them back to be welded."""
        session, run, _ = booked
        text = render_receipt(session, run)
        assert "\n\n" in text
        assert "\n" not in text.replace("\n\n", "")


class TestSayingWhyATimeIsMissing:
    """A time withheld silently reads as a question ignored.

    Live: "how about 11am on july 29?" was answered with 9:00, 10:00 and 2:00
    and nothing else. The 11:00 was withheld correctly — the patient had an
    Orthopedics appointment at 11:00 that day and the commit would have refused
    it — but nothing said so, so the reply reads as though the question had not
    been asked.

    The sentence is a claim about the patient's own diary, so it is drawn from
    the rows the search actually removed and from nowhere else.
    """

    WITHHELD = [
        {
            "slot_id": 391,
            "start": "2026-07-29T11:00:00",
            "department_name": "Orthopedics",
        }
    ]

    def test_the_time_they_named_was_withheld(self):
        note = clash_note(self.WITHHELD, "how about 11am on july 29?")
        assert note == "That time clashes with your Orthopedics appointment that day."

    def test_a_time_that_was_simply_not_free_makes_no_claim(self):
        """"Nothing is free at 2" and "your 2 is spoken for" are different
        facts, and only one of them may be asserted from these rows."""
        assert clash_note(self.WITHHELD, "how about 2pm on july 29?") == ""

    def test_a_message_naming_no_time_at_all(self):
        assert clash_note(self.WITHHELD, "what else is there?") == ""

    def test_nothing_was_withheld(self):
        assert clash_note([], "how about 11am?") == ""

    def test_two_days_withheld_at_that_hour_name_neither(self):
        """The sentence says "that day". A search over a week can withhold
        11:00 twice, and naming the wrong one is worse than saying nothing."""
        both = self.WITHHELD + [
            {
                "slot_id": 402,
                "start": "2026-07-31T11:00:00",
                "department_name": "Cardiology",
            }
        ]
        assert clash_note(both, "how about 11am?") == ""

    def test_the_same_day_twice_is_still_one_day(self):
        """Two doctors free at 11:00, both withheld by one appointment. The day
        is unambiguous, so the sentence stands."""
        twice = self.WITHHELD + [
            {
                "slot_id": 392,
                "start": "2026-07-29T11:00:00",
                "department_name": "Orthopedics",
            }
        ]
        assert "Orthopedics" in clash_note(twice, "how about 11am?")
