"""What a reschedule or cancellation proposal refuses to record.

These are the deterministic half of "name exactly which appointment". Code
cannot know which appointment the patient *meant* — that is language, and the
model's job — but it can refuse to write a proposal against one that is not
this patient's, is not changeable, or points at a slot that is gone.

Tested through the toolbelt, which is the seam the agent actually holds: the
tools are bound to the acting patient for the duration of one turn, and the
binding is a guard as much as a convenience. A model that got creative with an
integer must find nothing.

Each of these was written after a coverage pass showed the guards existed and
nothing exercised them. A guard nobody has ever tripped is a decoration.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app import clock
from app.agents.toolbelt import Toolbelt
from app.models import (
    Appointment,
    AppointmentSlot,
    AppointmentStatus,
    Department,
    Doctor,
    ProposedAction,
    SlotStatus,
    TraceEvent,
    TraceEventType,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.tools.dates import resolve_date
from app.trace import TraceWriter
from app.workflow.replies import window_heading, window_note

SEEDED_APPOINTMENT_ID = 1
ASHA_PROFILE_ID = 1
ROHAN_PROFILE_ID = 2


@pytest.fixture
def asha(seeded_db):
    return (
        seeded_db.query(User)
        .filter(User.email == "asha.patient@example.invalid")
        .one()
    )


@pytest.fixture
def run(seeded_db):
    row = WorkflowRun(
        patient_id=ASHA_PROFILE_ID,
        status=WorkflowStatus.IN_PROGRESS,
        plan=["cancel"],
        completed_steps=[],
        state={},
        session_id="s-guards",
    )
    seeded_db.add(row)
    seeded_db.flush()
    return row


def belt_for(seeded_db, user, run, *, patient_id: int = ASHA_PROFILE_ID) -> Toolbelt:
    return Toolbelt(
        seeded_db,
        user=user,
        patient_id=patient_id,
        writer=TraceWriter(seeded_db, session_id="s-guards"),
        run=run,
    )


def slot_in(seeded_db, department_name: str) -> AppointmentSlot:
    """A free slot in that department, **in the future**.

    ``get_slot``'s ``available`` verdict rules out a slot that has already
    started, and the seed's earliest slot is today at 09:00 — so a helper that
    took the earliest one would make these tests pass in the morning and fail
    in the afternoon.
    """
    slot = (
        seeded_db.query(AppointmentSlot)
        .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
        .join(Department, Department.id == Doctor.department_id)
        .filter(
            Department.name == department_name,
            AppointmentSlot.status == SlotStatus.AVAILABLE,
            AppointmentSlot.start_time > clock.now(),
        )
        .order_by(AppointmentSlot.start_time)
        .first()
    )
    assert slot is not None, f"no future free slot in {department_name}"
    return slot


class TestOwnership:
    def test_a_model_cannot_cancel_another_patients_appointment(
        self, seeded_db, asha, run
    ):
        """The one-digit edit, from inside the agent loop rather than over HTTP.

        Asha's own appointment, but a belt bound to Rohan: the appointment id
        is an integer and the binding is what makes guessing it useless.
        """
        belt = belt_for(seeded_db, asha, run, patient_id=ROHAN_PROFILE_ID)
        result = belt._propose_change(
            ProposedAction.CANCEL, appointment_id=SEEDED_APPOINTMENT_ID, slot_id=None
        )

        assert result["accepted"] is False
        assert "not one of this patient's" in result["problem"]

    def test_the_refused_proposal_is_not_written_to_the_run(self, seeded_db, asha, run):
        belt = belt_for(seeded_db, asha, run, patient_id=ROHAN_PROFILE_ID)
        belt._propose_change(
            ProposedAction.CANCEL, appointment_id=SEEDED_APPOINTMENT_ID, slot_id=None
        )

        assert run.proposed_action is None
        assert run.proposed_appointment_id is None
        assert run.status is WorkflowStatus.IN_PROGRESS

    def test_an_appointment_that_does_not_exist_is_refused(self, seeded_db, asha, run):
        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_change(
            ProposedAction.CANCEL, appointment_id=99999, slot_id=None
        )
        assert result["accepted"] is False

    def test_the_refusal_is_recorded_as_a_validation_event(self, seeded_db, asha, run):
        """A rejected slip that leaves no trace is a rejection nobody can audit."""
        from app.models import TraceEvent, TraceEventType

        belt = belt_for(seeded_db, asha, run, patient_id=ROHAN_PROFILE_ID)
        belt._propose_change(
            ProposedAction.CANCEL, appointment_id=SEEDED_APPOINTMENT_ID, slot_id=None
        )
        seeded_db.flush()

        events = [
            event
            for event in seeded_db.query(TraceEvent).all()
            if event.event_type is TraceEventType.VALIDATION
            and (event.payload or {}).get("accepted") is False
        ]
        assert events, "the refused proposal was never recorded"


class TestTheAppointmentMustStillBeChangeable:
    def test_a_cancelled_appointment_cannot_be_cancelled_again(
        self, seeded_db, asha, run
    ):
        seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).status = (
            AppointmentStatus.CANCELLED
        )
        seeded_db.flush()

        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_change(
            ProposedAction.CANCEL, appointment_id=SEEDED_APPOINTMENT_ID, slot_id=None
        )

        assert result["accepted"] is False
        assert "cancelled" in result["problem"]

    def test_a_completed_appointment_cannot_be_rescheduled(self, seeded_db, asha, run):
        seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).status = (
            AppointmentStatus.COMPLETED
        )
        seeded_db.flush()

        belt = belt_for(seeded_db, asha, run)
        target = slot_in(seeded_db, "Cardiology")
        result = belt._propose_change(
            ProposedAction.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=target.id,
        )

        assert result["accepted"] is False
        assert "no longer be changed" in result["problem"]


class TestTheNewSlotMustBeUsable:
    def test_a_slot_that_does_not_exist_is_refused(self, seeded_db, asha, run):
        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_change(
            ProposedAction.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=99999,
        )
        assert result["accepted"] is False
        assert "does not exist" in result["problem"]

    def test_a_reschedule_with_no_slot_at_all_is_refused(self, seeded_db, asha, run):
        """A reschedule needs somewhere to go. Without one it would record a
        proposal the commit could not act on."""
        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_change(
            ProposedAction.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=None,
        )
        assert result["accepted"] is False

    def test_an_already_taken_slot_is_refused(self, seeded_db, asha, run):
        target = slot_in(seeded_db, "Cardiology")
        target.status = SlotStatus.BOOKED
        seeded_db.flush()

        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_change(
            ProposedAction.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=target.id,
        )
        assert result["accepted"] is False
        assert "no longer available" in result["problem"]

    def test_moving_to_another_department_is_refused(self, seeded_db, asha, run):
        """A reschedule moves the time, not the department.

        Its plan closes over neither routing nor the required-documents diff,
        so a cross-department move would land the patient in a department that
        was never routed to and whose document rules were never checked — a
        booking wearing a reschedule's plan.
        """
        target = slot_in(seeded_db, "Dermatology")
        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_change(
            ProposedAction.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=target.id,
        )

        assert result["accepted"] is False
        assert "different department" in result["problem"]
        assert run.proposed_action is None


class TestTheSlotMustBeFreeInThePatientsOwnDiary:
    """Round 9, item 1 — the proposal half of the clash guard.

    The commit refuses a double booking outright; this stops the patient ever
    being *asked* to agree to one. Live, the run held a Dermatology appointment
    and the model proposed the 9:00 AM that the patient's brand-new
    Ophthalmology appointment already occupied — a Confirm guaranteed to bounce,
    offered as though it were a choice.

    ``message`` names the department on purpose. With two live appointments and
    no cue in the words, ``_settle_target`` refuses one step earlier — so a
    test written without it passes green while saying nothing whatever about
    the guard below it.
    """

    @pytest.fixture
    def clash(self, seeded_db, asha):
        """Asha's seeded Cardiology appointment, plus a Dermatology one
        occupying the exact time of a free Cardiology slot."""
        target = slot_in(seeded_db, "Cardiology")
        occupied = (
            seeded_db.query(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .join(Department, Department.id == Doctor.department_id)
            .filter(
                Department.name == "Dermatology",
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                AppointmentSlot.start_time == target.start_time,
            )
            .first()
        )
        assert occupied is not None, "no Dermatology slot at that Cardiology time"

        occupied.status = SlotStatus.BOOKED
        second = Appointment(
            patient_id=ASHA_PROFILE_ID,
            doctor_id=occupied.doctor_id,
            slot_id=occupied.id,
            department_id=occupied.doctor.department_id,
            status=AppointmentStatus.CONFIRMED,
            reference_code="AC-000099",
            reason="skin rash",
        )
        seeded_db.add(second)
        seeded_db.flush()
        return target

    def test_a_slot_the_patient_is_already_busy_at_is_refused(
        self, seeded_db, asha, run, clash
    ):
        belt = belt_for(seeded_db, asha, run)
        belt.message = "move my cardiology appointment"
        result = belt._propose_change(
            ProposedAction.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=clash.id,
        )

        assert result["accepted"] is False
        assert "already have a" in result["problem"]

    def test_the_refused_proposal_is_not_written_to_the_run(
        self, seeded_db, asha, run, clash
    ):
        belt = belt_for(seeded_db, asha, run)
        belt.message = "move my cardiology appointment"
        belt._propose_change(
            ProposedAction.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=clash.id,
        )

        assert run.proposed_action is None
        assert run.proposed_slot_id is None
        assert run.status is WorkflowStatus.IN_PROGRESS

    def test_both_directions_are_traced(self, seeded_db, asha, run, clash):
        """A refused clash leaves no other mark — the run is untouched, so
        without the row there is nothing to say the guard ran at all."""
        belt = belt_for(seeded_db, asha, run)
        belt.message = "move my cardiology appointment"
        belt._propose_change(
            ProposedAction.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=clash.id,
        )
        seeded_db.flush()

        rejections = [
            event
            for event in seeded_db.query(TraceEvent)
            .filter(TraceEvent.event_type == TraceEventType.VALIDATION)
            .all()
            if event.payload.get("what") == "appointment_change_proposal"
            and event.payload.get("accepted") is False
        ]
        assert any(
            "clash" in (event.payload.get("detail") or {}).get("problem", "")
            for event in rejections
        )

    def test_a_free_time_is_still_accepted(self, seeded_db, asha, run, clash):
        """The negative control. The same run, the same second appointment,
        a Cardiology slot at an hour the patient is not already spoken for."""
        free = (
            seeded_db.query(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .join(Department, Department.id == Doctor.department_id)
            .filter(
                Department.name == "Cardiology",
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                AppointmentSlot.start_time > clash.start_time,
            )
            .order_by(AppointmentSlot.start_time)
            .first()
        )
        belt = belt_for(seeded_db, asha, run)
        belt.message = "move my cardiology appointment"
        result = belt._propose_change(
            ProposedAction.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=free.id,
        )

        assert result["accepted"] is True

    def test_the_appointment_being_moved_is_not_its_own_clash(
        self, seeded_db, asha, run
    ):
        """Self-exclusion at the proposal layer.

        Asha's own Cardiology appointment sits at some hour; the other
        cardiologist is free at that same hour. Moving to it is a change of
        doctor, and without excluding the row being moved the guard reads it as
        a clash with itself and refuses every such move.
        """
        current = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID)
        twin = (
            seeded_db.query(AppointmentSlot)
            .filter(
                AppointmentSlot.start_time == current.slot.start_time,
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                AppointmentSlot.doctor_id != current.doctor_id,
            )
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .filter(Doctor.department_id == current.department_id)
            .first()
        )
        assert twin is not None, "no second cardiologist free at that time"

        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_change(
            ProposedAction.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=twin.id,
        )

        assert result["accepted"] is True


class TestAProposedSearchWindowIsDisposedByCode:
    """Round 9, item 5b — the model may propose a window; code decides.

    The deterministic vocabulary (5a) covers the forms patients actually write,
    and it will never cover all of them. This is the fallback for the phrase
    nobody anticipated, and it is shaped so that being wrong is cheap: the
    model hands over two dates and no prose, cannot name a slot, cannot commit,
    and the search that runs is the same one every other path runs, bound to
    the same patient.

    A rejection falls through to layer (c), which says plainly that the
    constraint could not be read — the honest answer, and the one the live turn
    should have given instead of silently showing the earliest three.
    """

    @pytest.fixture
    def holding(self, seeded_db, asha, run):
        """A run holding a proposal, which is where timing questions land."""
        run.status = WorkflowStatus.PENDING_CONFIRMATION
        run.proposed_action = ProposedAction.BOOK
        run.state = {"department_id": 1}
        target = slot_in(seeded_db, "Cardiology")
        run.proposed_slot_id = target.id
        seeded_db.flush()
        return run

    def test_a_usable_window_is_searched(self, seeded_db, asha, holding):
        belt = belt_for(seeded_db, asha, holding)
        today = clock.today()

        result = belt._propose_search_window(
            today.isoformat(), (today + timedelta(days=6)).isoformat()
        )

        assert result["accepted"] is True
        assert belt.proposals.offered_slots
        assert belt.proposals.searched_slots is True

    @pytest.mark.parametrize(
        "start,end,fragment",
        [
            ("2026-08-10", "2026-08-03", "not be after"),
            ("not-a-date", "2026-08-10", "calendar dates"),
            ("2027-06-01", "2027-06-07", "beyond what can be booked"),
            ("2020-01-01", "2020-01-07", "in the past"),
        ],
    )
    def test_an_unusable_window_is_refused(
        self, seeded_db, asha, holding, start, end, fragment
    ):
        belt = belt_for(seeded_db, asha, holding)

        result = belt._propose_search_window(start, end)

        assert result["accepted"] is False
        assert fragment in result["problem"]

    def test_a_refusal_leaves_the_turn_to_admit_it(self, seeded_db, asha, holding):
        """The handover to layer (c): nothing was searched, so nothing may be
        rendered as though it had been."""
        belt = belt_for(seeded_db, asha, holding)

        belt._propose_search_window("2027-06-01", "2027-06-07")

        assert belt.proposals.offered_slots == []
        assert belt.proposals.searched_slots is False

    def test_both_directions_are_traced(self, seeded_db, asha, holding):
        """A refused window changes nothing, so the row is the only evidence
        the guard ran — and an accepted one has to say a window was code's
        decision rather than the model's."""
        belt = belt_for(seeded_db, asha, holding)
        today = clock.today()
        belt._propose_search_window("2027-06-01", "2027-06-07")
        belt._propose_search_window(
            today.isoformat(), (today + timedelta(days=6)).isoformat()
        )
        seeded_db.flush()

        verdicts = [
            event.payload
            for event in seeded_db.query(TraceEvent)
            .filter(TraceEvent.event_type == TraceEventType.VALIDATION)
            .all()
            if event.payload.get("what") == "search_window"
        ]
        assert [v["accepted"] for v in verdicts] == [False, True]

    def test_the_window_cannot_reach_another_patient(self, seeded_db, asha, holding):
        """The binding is the guard: the tool takes dates, never a patient."""
        belt = belt_for(seeded_db, asha, holding)
        today = clock.today()

        belt._propose_search_window(
            today.isoformat(), (today + timedelta(days=6)).isoformat()
        )

        assert belt.patient_id == ASHA_PROFILE_ID

    def test_a_withheld_time_is_still_explained(self, seeded_db, asha, holding):
        """Found by the live sweep, not by this suite — the round's one
        regression, and it was mine.

        `gpt-4o-mini` answered "how about 9am next monday?" by calling this
        tool rather than `list_other_slots`. The search withheld the 9:00 that
        the patient's own Cardiology appointment occupied, exactly as it
        should; the reply then listed other times and said nothing about it,
        because this path did not compute the clash note. Round 8's guard was
        intact and simply not on the road the turn took.

        A second route to one answer owes everything the first one produced.
        """
        seeded = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID)
        when = seeded.slot.start_time
        belt = belt_for(seeded_db, asha, holding)
        hour = when.hour % 12 or 12
        belt.message = f"how about {hour}{'am' if when.hour < 12 else 'pm'}?"

        belt._propose_search_window(
            when.date().isoformat(), when.date().isoformat()
        )

        assert "clashes with your Cardiology appointment" in belt.proposals.clash_note

    def test_the_patients_own_words_outrank_the_models_window(
        self, seeded_db, asha, holding
    ):
        """Found by the live sweep, and the sharper half of the same lesson.

        "How about 9am next monday?" is a phrase layer (a) resolves exactly.
        The live model called this tool anyway and proposed a window covering
        *today* — accepted, because it was well-formed and in horizon — so the
        search ran over the wrong week entirely.

        A fallback that can be chosen *instead of* the thing it falls back from
        is not a fallback. Where the patient's words resolve, they decide, and
        the model's dates are discarded.
        """
        belt = belt_for(seeded_db, asha, holding)
        belt.message = "how about 9am next monday?"
        expected = resolve_date("next monday", today=clock.today())

        belt._propose_search_window("2026-08-03", "2026-08-03")

        offered = {slot["start"][:10] for slot in belt.proposals.offered_slots}
        assert offered, "the deterministic window returned nothing to check"
        assert offered == {expected["start"]}

    def test_the_override_is_traced(self, seeded_db, asha, holding):
        belt = belt_for(seeded_db, asha, holding)
        belt.message = "how about 9am next monday?"
        belt._propose_search_window("2026-08-03", "2026-08-03")
        seeded_db.flush()

        overrides = [
            event.payload
            for event in seeded_db.query(TraceEvent)
            .filter(TraceEvent.event_type == TraceEventType.VALIDATION)
            .all()
            if event.payload.get("what") == "search_window"
            and event.payload.get("accepted") is False
        ]
        assert any(
            "own words" in (o.get("detail") or {}).get("problem", "") for o in overrides
        )

    def test_the_model_still_decides_when_the_words_say_nothing(
        self, seeded_db, asha, holding
    ):
        """The negative control that keeps layer (b) alive: with no readable
        phrase, the model's window is what there is."""
        belt = belt_for(seeded_db, asha, holding)
        belt.message = "what else have you got?"
        today = clock.today()

        result = belt._propose_search_window(
            today.isoformat(), (today + timedelta(days=6)).isoformat()
        )

        assert result["accepted"] is True

    def test_it_is_offered_only_while_a_proposal_stands(self, seeded_db, asha, run):
        """Handed out where the timing questions land, and nowhere else."""
        run.status = WorkflowStatus.IN_PROGRESS
        seeded_db.flush()
        names = {tool.__name__ for tool in belt_for(seeded_db, asha, run).coordinator_tools()}
        assert "propose_search_window" not in names

        run.status = WorkflowStatus.PENDING_CONFIRMATION
        seeded_db.flush()
        names = {tool.__name__ for tool in belt_for(seeded_db, asha, run).coordinator_tools()}
        assert "propose_search_window" in names


class TestThePhraseTheSearchActuallyUses:
    """Round 10 item 4 — the last door a stated constraint could vanish through.

    ``phrase`` is the model's summary of the patient's timing words, and an
    *empty* one is not evidence that nothing was said. The mock's own extractor
    returns "" for "more slots in the afternoon?", so that question searched
    unfiltered and came back with 10 AM, 11 AM and 2 PM — and layer (c) stayed
    silent, because ``unreadable`` is computed from the phrase and the phrase was
    empty. Round 9 closed the front door and this was the side one.
    """

    @pytest.fixture
    def holding(self, seeded_db, asha, run):
        run.status = WorkflowStatus.PENDING_CONFIRMATION
        run.proposed_action = ProposedAction.BOOK
        run.state = {"department_id": 1}
        run.proposed_slot_id = slot_in(seeded_db, "Cardiology").id
        seeded_db.flush()
        return run

    def test_an_absent_phrase_falls_back_to_the_patients_words(
        self, seeded_db, asha, holding
    ):
        belt = belt_for(seeded_db, asha, holding)
        belt.message = "more slots in the afternoon?"

        listed = belt._list_other_slots()

        assert listed["slots"], "nothing came back, so nothing is being checked"
        hours = {
            datetime.fromisoformat(slot["start"]).hour for slot in listed["slots"]
        }
        assert hours and all(hour >= 12 for hour in hours)

    def test_the_note_names_what_it_could_not_read(self, seeded_db, asha, holding):
        """Both halves of that phrase are now true at once. The day was
        unreadable and the time of day was honoured, so apologising for reading
        nothing — directly above a list of afternoon times — would be false as
        well as confusing."""
        belt = belt_for(seeded_db, asha, holding)
        belt.message = "more slots in the afternoon?"

        belt._list_other_slots()

        assert belt.proposals.window_unreadable is True
        assert belt.proposals.window_part_of_day == "afternoon"
        assert window_note(
            unreadable=True, empty_label=None, part_of_day="afternoon"
        ) == (
            "I couldn't read that as a day — here are the earliest afternoon "
            "times free:"
        )

    def test_a_model_supplied_phrase_still_wins(self, seeded_db, asha, holding):
        """It may legitimately be narrower than the whole sentence, and
        second-guessing that would put code in the reading bin for a failure
        nobody has seen."""
        belt = belt_for(seeded_db, asha, holding)
        belt.message = "anything in the afternoon or maybe next week?"
        expected = resolve_date("next week", today=clock.today())

        belt._list_other_slots("next week")

        days = {slot["start"][:10] for slot in belt.proposals.offered_slots}
        assert days
        assert min(days) >= expected["start"]

    def test_a_message_with_no_timing_words_changes_nothing(
        self, seeded_db, asha, holding
    ):
        """The negative control. The fallback must not turn every recovery
        listing into a filtered search: "option 3" names no time, so the list is
        the one it always was and layer (c) has nothing to admit."""
        belt = belt_for(seeded_db, asha, holding)
        belt.message = "option 3"

        belt._list_other_slots()

        assert belt.proposals.window_unreadable is False
        assert belt.proposals.window_part_of_day is None
        assert belt.proposals.offered_slots


class TestWhoseWindowTheListAnswers:
    """Round 10 item 4a — a working answer that reads as an ignored question.

    Live: "got anything whenever the moon is full?" Layer (b) worked
    mechanically — the model proposed an Aug-1 window, code validated it, the
    search ran, the astronomy prose was suppressed and Saturday times were
    rendered. Nothing false shipped, and nothing said why Saturday.
    """

    @pytest.fixture
    def holding(self, seeded_db, asha, run):
        run.status = WorkflowStatus.PENDING_CONFIRMATION
        run.proposed_action = ProposedAction.BOOK
        run.state = {"department_id": 1}
        run.proposed_slot_id = slot_in(seeded_db, "Cardiology").id
        seeded_db.flush()
        return run

    def test_a_model_proposed_window_is_named(self, seeded_db, asha, holding):
        belt = belt_for(seeded_db, asha, holding)
        belt.message = "got anything whenever the moon is full?"
        target = clock.today() + timedelta(days=5)

        belt._propose_search_window(target.isoformat(), target.isoformat())

        assert belt.proposals.window_provenance == f"on {target:%A} {target.day} {target:%B}"
        assert window_heading(belt.proposals.window_provenance).startswith(
            "Times that are free on"
        )

    def test_a_window_the_vocabulary_read_is_not_captioned(
        self, seeded_db, asha, holding
    ):
        """A rule that worked is not something to caption, and a caption on
        every list is a caption nobody reads."""
        belt = belt_for(seeded_db, asha, holding)
        belt.message = "what else is free next week?"

        belt._list_other_slots("next week")

        assert belt.proposals.window_provenance is None

    def test_the_patients_own_words_take_their_heading_with_them(
        self, seeded_db, asha, holding
    ):
        """The round-9 override and the round-10 caption are one path: when the
        patient's words win, the model's window never happened, so there is
        nothing of the model's to name."""
        belt = belt_for(seeded_db, asha, holding)
        belt.message = "how about next monday?"

        belt._propose_search_window("2026-08-03", "2026-08-03")

        assert belt.proposals.window_provenance is None


class TestWhatTheSearchHandsTheModel:
    """Round 9, item 1 — the search filtered correctly and the payload undid it.

    ``find_available_slots`` has withheld a patient's own busy times since
    round 7, and returned them in ``withheld_for_patient`` since round 8 so the
    reply could say *why* a named time is missing. That field is read here and
    nowhere downstream — but the tool handed it to the model, which is a list
    of slot ids labelled "not available to this patient", given to the one
    component whose job is to choose a slot id. Live, the model chose one.
    """

    @pytest.fixture
    def busy(self, seeded_db, asha):
        """Asha booked into the exact time of a free Cardiology slot."""
        target = slot_in(seeded_db, "Cardiology")
        occupied = (
            seeded_db.query(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .join(Department, Department.id == Doctor.department_id)
            .filter(
                Department.name == "Dermatology",
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                AppointmentSlot.start_time == target.start_time,
            )
            .first()
        )
        occupied.status = SlotStatus.BOOKED
        seeded_db.add(
            Appointment(
                patient_id=ASHA_PROFILE_ID,
                doctor_id=occupied.doctor_id,
                slot_id=occupied.id,
                department_id=occupied.doctor.department_id,
                status=AppointmentStatus.CONFIRMED,
                reference_code="AC-000099",
                reason="skin rash",
            )
        )
        seeded_db.flush()
        return target

    def _tool(self, belt, name: str):
        return next(t for t in belt.appointment_tools() if t.__name__ == name)

    def test_the_reschedule_search_withholds_nothing_by_name(
        self, seeded_db, asha, run, busy
    ):
        belt = belt_for(seeded_db, asha, run)
        found = self._tool(belt, "find_slots_for_reschedule")(SEEDED_APPOINTMENT_ID)

        assert "withheld_for_patient" not in found
        assert busy.id not in {slot["slot_id"] for slot in found["slots"]}

    def test_the_booking_search_withholds_nothing_by_name(
        self, seeded_db, asha, run, busy
    ):
        run.state = {"department_id": 1}
        belt = belt_for(seeded_db, asha, run)
        found = self._tool(belt, "find_available_slots")("", "", "")

        assert "withheld_for_patient" not in found
        assert busy.id not in {slot["slot_id"] for slot in found["slots"]}

    def test_the_appointment_being_moved_does_not_hide_its_own_hour(
        self, seeded_db, asha, run
    ):
        """Self-exclusion in the search.

        Asha's Cardiology appointment occupies an hour; the other cardiologist
        is free in it. Counting the appointment against its own search removes
        that slot, so the answer to "can I see the other doctor at the same
        time" is a list that does not contain the only slot that answers it.
        """
        current = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID)
        twin = (
            seeded_db.query(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .filter(
                Doctor.department_id == current.department_id,
                AppointmentSlot.start_time == current.slot.start_time,
                AppointmentSlot.status == SlotStatus.AVAILABLE,
            )
            .first()
        )
        assert twin is not None, "no second cardiologist free at that time"

        belt = belt_for(seeded_db, asha, run)
        # Scoped to the day: the default limit is 20 and the seed lays down
        # fourteen days of slots, so an unscoped search would drop this slot
        # for being far down the list and the assertion would pass or fail on
        # the limit rather than on the rule.
        day = current.slot.start_time.date().isoformat()
        found = self._tool(belt, "find_slots_for_reschedule")(
            SEEDED_APPOINTMENT_ID, day, day
        )

        assert twin.id in {slot["slot_id"] for slot in found["slots"]}


class TestTheAcceptedPath:
    def test_a_valid_cancellation_proposal_pauses_the_run(self, seeded_db, asha, run):
        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_change(
            ProposedAction.CANCEL, appointment_id=SEEDED_APPOINTMENT_ID, slot_id=None
        )

        assert result["accepted"] is True
        assert run.proposed_action is ProposedAction.CANCEL
        assert run.proposed_appointment_id == SEEDED_APPOINTMENT_ID
        assert run.status is WorkflowStatus.PENDING_CONFIRMATION

    def test_the_proposal_carries_facts_read_back_from_the_row(
        self, seeded_db, asha, run
    ):
        """Not from anything the model remembered — that is what makes
        "exactly which appointment" worth saying."""
        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_change(
            ProposedAction.CANCEL, appointment_id=SEEDED_APPOINTMENT_ID, slot_id=None
        )

        facts = result["proposed"]
        assert facts["reference_code"] == "AC-000001"
        assert facts["department_name"] == "Cardiology"

    def test_a_reschedule_proposal_carries_the_slot_it_would_move_to(
        self, seeded_db, asha, run
    ):
        target = slot_in(seeded_db, "Cardiology")
        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_change(
            ProposedAction.RESCHEDULE,
            appointment_id=SEEDED_APPOINTMENT_ID,
            slot_id=target.id,
        )

        assert result["accepted"] is True
        assert result["new_slot"]["slot_id"] == target.id
        assert run.proposed_slot_id == target.id


class TestMovingTheOfferToAnotherSlot:
    """The re-proposal guard: only a time this patient has actually been shown.

    The tool exists so "the 2pm one" can move the offer without going anywhere
    near a commit. What makes it safe is that the id cannot come from the
    model's memory of the conversation — a slot id recalled from prose and a
    slot id invented both arrive as an integer, and only one of them was ever
    on a list the patient saw.

    Two refusals, and the second is the one that would be easy to forget: a
    slot on the shown list can be taken between being shown and being chosen.
    That is the commit-time slot-taken discipline, one step earlier.
    """

    @pytest.fixture
    def holding(self, seeded_db, run):
        """A run holding a proposal, having shown three slots."""
        offered = (
            seeded_db.query(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .join(Department, Department.id == Doctor.department_id)
            .filter(
                Department.name == "Cardiology",
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                AppointmentSlot.start_time > clock.now(),
            )
            .order_by(AppointmentSlot.start_time)
            .limit(3)
            .all()
        )
        assert len(offered) == 3
        run.plan = ["route", "book"]
        run.status = WorkflowStatus.PENDING_CONFIRMATION
        run.proposed_action = ProposedAction.BOOK
        run.proposed_slot_id = offered[0].id
        run.state = {
            "department_id": offered[0].doctor.department_id,
            "offered_slot_ids": [slot.id for slot in offered],
        }
        seeded_db.flush()
        return offered

    def test_a_slot_that_was_shown_can_be_taken(self, seeded_db, asha, run, holding):
        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_another_slot(holding[1].id)

        assert result["accepted"] is True
        assert run.proposed_slot_id == holding[1].id
        assert run.status is WorkflowStatus.PENDING_CONFIRMATION

    def test_a_slot_that_was_never_shown_is_refused(
        self, seeded_db, asha, run, holding
    ):
        """The invented-id case. It must not become a proposal, and the old
        proposal must survive: a patient left holding nothing because the model
        named a number is worse off than one who was told no."""
        unshown = slot_in(seeded_db, "Dermatology")
        assert unshown.id not in run.state["offered_slot_ids"]
        held = run.proposed_slot_id

        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_another_slot(unshown.id)

        assert result["accepted"] is False
        assert run.proposed_slot_id == held
        assert run.status is WorkflowStatus.PENDING_CONFIRMATION

    def test_the_refusal_is_written_to_the_trace(self, seeded_db, asha, run, holding):
        """A refused invention leaves no other mark. If it is not an event,
        nobody reviewing the run can tell it happened."""
        unshown = slot_in(seeded_db, "Dermatology")
        belt = belt_for(seeded_db, asha, run)
        belt._propose_another_slot(unshown.id)
        seeded_db.flush()

        rejections = [
            event.payload
            for event in seeded_db.query(TraceEvent).all()
            if event.event_type is TraceEventType.VALIDATION
            and event.payload.get("what") == "reproposal_slot_offered"
            and event.payload.get("accepted") is False
        ]
        assert rejections, "the refusal must be an event, not merely a return value"
        assert rejections[0]["detail"]["slot_id"] == unshown.id

    def test_an_acceptance_is_written_too(self, seeded_db, asha, run, holding):
        """Both directions. A guard that only records its refusals cannot be
        told apart from one that never ran."""
        belt = belt_for(seeded_db, asha, run)
        belt._propose_another_slot(holding[2].id)
        seeded_db.flush()

        accepted = [
            event.payload
            for event in seeded_db.query(TraceEvent).all()
            if event.event_type is TraceEventType.VALIDATION
            and event.payload.get("what") == "reproposal_slot_offered"
            and event.payload.get("accepted") is True
        ]
        assert accepted

    def test_a_shown_slot_that_has_since_been_taken_is_refused(
        self, seeded_db, asha, run, holding
    ):
        """Being on the list is not the same as being free. The liveness check
        is ``_propose_appointment``'s own — the same one the original proposal
        ran — rather than a second copy that could drift from it."""
        holding[1].status = SlotStatus.BOOKED
        seeded_db.flush()
        held = run.proposed_slot_id

        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_another_slot(holding[1].id)

        assert result["accepted"] is False
        assert run.proposed_slot_id == held

    def test_a_dead_slot_comes_back_with_something_to_choose_from(
        self, seeded_db, asha, run, holding
    ):
        """A refusal with no alternatives is where a conversation stops."""
        holding[1].status = SlotStatus.BOOKED
        seeded_db.flush()

        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_another_slot(holding[1].id)

        assert result["slots"], "the patient is told no and shown nothing else"

    def test_listing_other_times_never_touches_the_proposal(
        self, seeded_db, asha, run, holding
    ):
        """The whole of answer-and-stay, at the tool: read-only means the run
        is byte-identical afterwards except for what it has now shown."""
        before = (run.proposed_slot_id, run.proposed_action, run.status)

        belt = belt_for(seeded_db, asha, run)
        listed = belt._list_other_slots()

        assert (run.proposed_slot_id, run.proposed_action, run.status) == before
        assert all(
            slot["slot_id"] != run.proposed_slot_id for slot in listed["slots"]
        ), "the time being held is not an alternative to itself"


class TestAChoiceMustComeFromTheListThatWasShown:
    """The appointment-side twin of the unshown-slot rejection.

    An appointment id the model recalls from its context is indistinguishable
    from one it invented — both arrive as an integer — so once a run has *asked*
    which appointment, the answer has to be one of the ones it asked about.
    ``listed_appointment_ids`` is written by ``render_appointment_choice`` at
    the moment the numbered list is rendered, and by nothing else.
    """

    def _second_appointment(self, seeded_db) -> Appointment:
        """A live Dermatology appointment beside Asha's seeded Cardiology one."""
        slot = slot_in(seeded_db, "Dermatology")
        doctor = seeded_db.get(Doctor, slot.doctor_id)
        appointment = Appointment(
            patient_id=ASHA_PROFILE_ID,
            doctor_id=doctor.id,
            slot_id=slot.id,
            department_id=doctor.department_id,
            status=AppointmentStatus.CONFIRMED,
            reason="second appointment for the choice tests",
            reference_code="AC-009001",
        )
        slot.status = SlotStatus.BOOKED
        seeded_db.add(appointment)
        seeded_db.flush()
        return appointment

    def test_an_unlisted_appointment_is_refused(self, seeded_db, asha, run):
        other = self._second_appointment(seeded_db)
        run.state = {"listed_appointment_ids": [SEEDED_APPOINTMENT_ID]}
        seeded_db.flush()

        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_change(
            ProposedAction.CANCEL, appointment_id=other.id, slot_id=None
        )

        assert result["accepted"] is False
        assert "not one of the choices" in result["problem"]
        assert run.proposed_action is None

    def test_a_listed_appointment_is_accepted(self, seeded_db, asha, run):
        other = self._second_appointment(seeded_db)
        run.state = {"listed_appointment_ids": [SEEDED_APPOINTMENT_ID, other.id]}
        seeded_db.flush()

        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_change(
            ProposedAction.CANCEL, appointment_id=other.id, slot_id=None
        )

        assert result["accepted"] is True
        assert run.proposed_appointment_id == other.id

    def test_no_listing_imposes_nothing(self, seeded_db, asha, run):
        """The single-appointment auto-target path. An empty set is "no list has
        been shown", not "nothing is allowed" — reading it the other way would
        break every run that never had to ask."""
        assert run.state == {}
        belt = belt_for(seeded_db, asha, run)
        result = belt._propose_change(
            ProposedAction.CANCEL, appointment_id=SEEDED_APPOINTMENT_ID, slot_id=None
        )
        assert result["accepted"] is True

    def test_both_directions_are_traced(self, seeded_db, asha, run):
        """A refused invention leaves no other mark, and an accepted choice has
        to be distinguishable from one that was never checked."""
        other = self._second_appointment(seeded_db)
        run.state = {"listed_appointment_ids": [other.id]}
        seeded_db.flush()
        belt = belt_for(seeded_db, asha, run)

        belt._propose_change(
            ProposedAction.CANCEL, appointment_id=SEEDED_APPOINTMENT_ID, slot_id=None
        )
        belt._propose_change(
            ProposedAction.CANCEL, appointment_id=other.id, slot_id=None
        )

        verdicts = [
            event.payload["accepted"]
            for event in seeded_db.query(TraceEvent).all()
            if event.event_type is TraceEventType.VALIDATION
            and event.payload["what"] == "appointment_choice"
        ]
        assert verdicts == [False, True]


class TestTheTargetComesFromThePatientsWords:
    """The one unguarded model argument, and the write behind it.

    Live, with a Monday Dermatology appointment and a Thursday Orthopedics one,
    "reschedule the appointment on Monday" arrived here as the *Thursday* id.
    Ownership passed, liveness passed, the slot was free and in the right
    department, the proposal was recorded, the patient confirmed — and the
    wrong appointment moved. Every check in this module asks whether an id is
    *usable*; none of them asked whether it was the one that was named.
    """

    @pytest.fixture
    def two(self, seeded_db, asha):
        """Asha's seeded Cardiology appointment, plus a Dermatology one."""
        seeded = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID)
        slot = slot_in(seeded_db, "Dermatology")
        second = Appointment(
            patient_id=ASHA_PROFILE_ID,
            department_id=slot.doctor.department_id,
            doctor_id=slot.doctor_id,
            slot_id=slot.id,
            status=AppointmentStatus.CONFIRMED,
            reference_code="AC-009002",
            reason="skin",
        )
        slot.status = SlotStatus.BOOKED
        seeded_db.add(second)
        seeded_db.flush()
        return seeded, second

    def belt(self, seeded_db, asha, run, message):
        return Toolbelt(
            seeded_db,
            user=asha,
            patient_id=ASHA_PROFILE_ID,
            writer=TraceWriter(seeded_db, session_id="s-guards"),
            run=run,
            message=message,
        )

    def test_a_named_department_overrides_the_models_pick(self, seeded_db, asha, run, two):
        """The whole defect in one assertion: the model asks for the wrong one
        and the patient's own words win."""
        seeded, second = two
        belt = self.belt(seeded_db, asha, run, "cancel my dermatology appointment")

        result = belt._propose_change(
            ProposedAction.CANCEL, appointment_id=seeded.id, slot_id=None
        )

        assert result["accepted"] is True
        assert run.proposed_appointment_id == second.id

    def test_the_override_is_written_to_the_trace(self, seeded_db, asha, run, two):
        """A silent override is a second unchecked decision. What was proposed,
        what was resolved, and which cues did it all have to be readable."""
        seeded, second = two
        belt = self.belt(seeded_db, asha, run, "cancel my dermatology appointment")
        belt._propose_change(ProposedAction.CANCEL, appointment_id=seeded.id, slot_id=None)
        seeded_db.flush()

        events = [
            event.payload
            for event in seeded_db.query(TraceEvent).all()
            if event.event_type is TraceEventType.VALIDATION
            and event.payload.get("what") == "appointment_target"
        ]
        assert events, "an override that leaves no event cannot be reviewed"
        assert events[0]["accepted"] is False
        assert events[0]["detail"]["proposed"] == seeded.id
        assert events[0]["detail"]["resolved"] == second.id

    def test_agreement_is_written_too(self, seeded_db, asha, run, two):
        """Both directions. A check that only records its overrides cannot be
        told apart from one that never ran."""
        seeded, second = two
        belt = self.belt(seeded_db, asha, run, "cancel my dermatology appointment")
        belt._propose_change(ProposedAction.CANCEL, appointment_id=second.id, slot_id=None)
        seeded_db.flush()

        events = [
            event.payload
            for event in seeded_db.query(TraceEvent).all()
            if event.event_type is TraceEventType.VALIDATION
            and event.payload.get("what") == "appointment_target"
        ]
        assert [event["accepted"] for event in events] == [True]

    def test_no_cue_with_two_appointments_refuses(self, seeded_db, asha, run, two):
        """Nobody knows which, so nothing is recorded — and the orchestrator's
        own numbered list is what asks. Guessing here is what the live failure
        was made of."""
        seeded, _ = two
        belt = self.belt(seeded_db, asha, run, "please cancel my appointment")

        result = belt._propose_change(
            ProposedAction.CANCEL, appointment_id=seeded.id, slot_id=None
        )

        assert result["accepted"] is False
        assert run.proposed_appointment_id is None

    def test_one_appointment_needs_no_cue(self, seeded_db, asha, run):
        """The auto-target case, unchanged. There was never a choice to get
        wrong, so nothing is imposed."""
        belt = self.belt(seeded_db, asha, run, "please cancel my appointment")

        result = belt._propose_change(
            ProposedAction.CANCEL, appointment_id=SEEDED_APPOINTMENT_ID, slot_id=None
        )

        assert result["accepted"] is True

    def test_a_missing_id_is_still_refused_with_one_appointment(self, seeded_db, asha, run):
        """And it is not quietly rewritten into the one row that happens to be
        there. A slip must stay a slip."""
        belt = self.belt(seeded_db, asha, run, "please cancel my appointment")

        result = belt._propose_change(
            ProposedAction.CANCEL, appointment_id=99999, slot_id=None
        )

        assert result["accepted"] is False

    def test_an_answer_to_a_numbered_list_is_left_to_the_listed_guard(
        self, seeded_db, asha, run, two
    ):
        """"2" names no weekday and no department, so this would read it as no
        cue and refuse the very list it was answering. Where a choice has been
        shown, `listed_appointment_ids` is the authority."""
        seeded, second = two
        run.state = {"listed_appointment_ids": [seeded.id, second.id]}
        seeded_db.flush()
        belt = self.belt(seeded_db, asha, run, "2")

        result = belt._propose_change(
            ProposedAction.CANCEL, appointment_id=second.id, slot_id=None
        )

        assert result["accepted"] is True
        assert run.proposed_appointment_id == second.id


class TestAHeldSlotIsAlwaysAnswerable:
    """Whatever the reply around it looks like, the held time is in the set.

    ``offered_slot_ids`` means "times this patient has been shown", and both
    readers of it — the re-proposal guard and ``read_selection`` — take it at
    that word. The proposal card names the held slot on *every* turn that holds
    one, including the turns where ``render_proposal`` draws nothing because no
    search ran and there is no shortlist to draw. Recording it only where a
    shortlist is rendered would leave the patient holding a time they could
    then not name, which is the hole item 2 exists to close rather than move.

    Reached through the toolbelt because that is the only way to hold a slot
    without a search in front of it — which is exactly what a live model does
    when it recalls an id from its context window.
    """

    def test_a_proposal_with_no_search_behind_it_is_still_recorded(
        self, seeded_db, asha, run
    ):
        slot = slot_in(seeded_db, "Cardiology")
        belt = belt_for(seeded_db, asha, run)

        result = belt._propose_appointment(slot.id)

        assert result["accepted"] is True
        assert run.state["offered_slot_ids"] == [slot.id]

    def test_moving_the_offer_leaves_both_in_the_set(self, seeded_db, asha, run):
        """The union is a union. "Actually the first one you showed me" arrives
        three exchanges later and a set that shrank would refuse it."""
        first = slot_in(seeded_db, "Cardiology")
        belt = belt_for(seeded_db, asha, run)
        belt._propose_appointment(first.id)
        second = (
            seeded_db.query(AppointmentSlot)
            .filter(
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                AppointmentSlot.doctor_id == first.doctor_id,
                AppointmentSlot.id != first.id,
                AppointmentSlot.start_time > clock.now(),
            )
            .order_by(AppointmentSlot.start_time)
            .first()
        )

        belt._propose_appointment(second.id)

        assert run.state["offered_slot_ids"] == [first.id, second.id]
