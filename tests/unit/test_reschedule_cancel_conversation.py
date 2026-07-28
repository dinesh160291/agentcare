"""Reschedule and cancel, driven the way a patient drives them.

The companion module pins the commit dispatch — the actual root of the dead
path. This one pins that there is a route to that commit from something someone
would plausibly type, which is the half that was missing: the tools worked all
along and nothing could reach them.

Two behaviours here are not merely wiring:

* **Never guess which appointment.** With two live appointments and a request
  naming neither, the agent asks and records no proposal. A wrongly guessed
  booking costs a tap; a wrongly guessed cancellation costs the visit.
* **The word "cancel" means three different things** depending on where it
  lands — a verb for this feature, the exact token that *declines* a proposal,
  and part of the phrase that withdraws the whole run. They must not collide.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from app import clock
from app.db import SessionLocal
from app.models import (
    Appointment,
    AppointmentSlot,
    AppointmentStatus,
    Department,
    Doctor,
    SlotStatus,
    TraceAuthor,
    TraceEvent,
    TraceEventType,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.providers.base import text_response
from app.providers.mock import MockLlm
from app.workflow.replies import clock_time
from app.orchestrator import run_workflow

PATIENT_EMAIL = "asha.patient@example.invalid"
SEEDED_APPOINTMENT_ID = 1


@pytest.fixture
def patient(seeded_db):
    seeded_db.commit()
    return seeded_db.query(User).filter(User.email == PATIENT_EMAIL).one()


def turn(user, message, session_id):
    return asyncio.run(run_workflow(user, message, session_id))


class TestTheWholeWayFromASentence:
    def test_a_cancellation_request_reaches_pending_confirmation(self, patient):
        result = turn(patient, "I want to cancel my appointment", "s-e2e-1")
        assert result.plan == ["cancel"]
        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value

    def test_the_cancellation_plan_does_not_drag_documents_in(self, patient):
        """A missing-documents task for a visit being called off is the defect
        that decided against modelling this as a booking."""
        result = turn(patient, "I want to cancel my appointment", "s-e2e-2")
        assert "documents" not in result.plan
        assert "route" not in result.plan

    def test_nothing_is_cancelled_until_the_patient_confirms(self, patient, seeded_db):
        turn(patient, "I want to cancel my appointment", "s-e2e-3")
        seeded_db.expire_all()
        assert (
            seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).status
            is AppointmentStatus.CONFIRMED
        )

    def test_the_proposal_names_exactly_which_appointment(self, patient):
        """Story 20: destructive actions are never ambiguous. The reference
        code reaches the sentence through render_confirmation's re-read."""
        result = turn(patient, "I want to cancel my appointment", "s-e2e-4")
        assert "AC-000001" in result.reply
        assert "Cardiology" in result.reply

    def test_confirming_in_words_cancels(self, patient, seeded_db):
        turn(patient, "I want to cancel my appointment", "s-e2e-5")
        turn(patient, "yes", "s-e2e-5")

        seeded_db.expire_all()
        assert (
            seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).status
            is AppointmentStatus.CANCELLED
        )

    def test_the_receipt_never_promises_a_retired_reminder(self, patient, seeded_db):
        """Cancelling retires the reminder in the same transaction, so a
        receipt promising one states a fact that is already false — the
        stale-fact bug the receipt discipline exists to kill, arriving inside
        the receipt itself."""
        turn(patient, "I want to cancel my appointment", "s-e2e-6")
        result = turn(patient, "yes", "s-e2e-6")
        assert "reminder the day before" not in result.reply

    def test_a_reschedule_request_reaches_pending_confirmation(self, patient):
        result = turn(patient, "please reschedule my appointment to next week", "s-e2e-7")
        assert result.plan == ["reschedule", "follow_up"]
        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value

    def test_the_reschedule_proposal_names_the_time_it_is_leaving_and_arriving(
        self, patient
    ):
        result = turn(patient, "please reschedule my appointment to next week", "s-e2e-8")
        assert "AC-000001" in result.reply

        # Both times, not the connective words that used to join them. The
        # wording moved from "from X to Y" to patient-speak ("I found your
        # appointment on X. I can move it to Y"), and a test pinned to " from "
        # was checking the sentence's grammar rather than its content.
        session = SessionLocal()
        try:
            appointment = session.get(Appointment, SEEDED_APPOINTMENT_ID)
            run = session.get(WorkflowRun, result.run_id)
            leaving = clock_time(appointment.slot.start_time)
            arriving = clock_time(
                session.get(AppointmentSlot, run.proposed_slot_id).start_time
            )
        finally:
            session.close()
        assert leaving in result.reply
        assert arriving in result.reply

    def test_confirming_a_reschedule_moves_it(self, patient, seeded_db):
        original_slot_id = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).slot_id
        turn(patient, "please reschedule my appointment to next week", "s-e2e-9")
        turn(patient, "yes", "s-e2e-9")

        seeded_db.expire_all()
        appointment = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID)
        assert appointment.slot_id != original_slot_id
        assert appointment.status is AppointmentStatus.CONFIRMED

    def test_a_patient_with_nothing_booked_is_told_so(self, seeded_db):
        """Rohan has no appointments. The turn must say so, not fail."""
        rohan = (
            seeded_db.query(User)
            .filter(User.email == "rohan.patient@example.invalid")
            .one()
        )
        seeded_db.commit()
        result = turn(rohan, "I want to cancel my appointment", "s-e2e-10")
        assert "appointment" in result.reply.lower()

        seeded_db.expire_all()
        run = seeded_db.get(WorkflowRun, result.run_id)
        assert run is None or run.proposed_action is None


class TestNeverGuessWhichAppointment:
    """Two live appointments, and a request that names neither."""

    @pytest.fixture
    def two_appointments(self, patient, seeded_db):
        """Asha's seeded Cardiology visit, plus a Neurology one."""
        slot = (
            seeded_db.query(AppointmentSlot)
            .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
            .join(Department, Department.id == Doctor.department_id)
            .filter(
                Department.name == "Neurology",
                AppointmentSlot.status == SlotStatus.AVAILABLE,
                # Upcoming, not merely free: the seed's earliest slot is today
                # at 09:00, so without this the fixture builds a past
                # "upcoming appointment" from lunchtime onwards.
                AppointmentSlot.start_time > clock.now(),
            )
            .order_by(AppointmentSlot.start_time)
            .first()
        )
        slot.status = SlotStatus.BOOKED
        seeded_db.add(
            Appointment(
                id=2,
                patient_id=1,
                doctor_id=slot.doctor_id,
                slot_id=slot.id,
                department_id=seeded_db.get(Doctor, slot.doctor_id).department_id,
                status=AppointmentStatus.CONFIRMED,
                reason="synthetic second appointment",
                reference_code="AC-000002",
            )
        )
        seeded_db.commit()
        return patient

    def test_an_ambiguous_cancellation_proposes_nothing(
        self, two_appointments, seeded_db
    ):
        result = turn(two_appointments, "I want to cancel my appointment", "s-amb-1")
        assert result.status == WorkflowStatus.IN_PROGRESS.value

        seeded_db.expire_all()
        run = seeded_db.get(WorkflowRun, result.run_id)
        assert run.proposed_action is None
        assert run.proposed_appointment_id is None

    def test_it_asks_which_one_and_lists_them(self, two_appointments):
        result = turn(two_appointments, "I want to cancel my appointment", "s-amb-2")
        assert "Cardiology" in result.reply and "Neurology" in result.reply
        assert "which" in result.reply.lower()

    def test_neither_appointment_is_touched(self, two_appointments, seeded_db):
        turn(two_appointments, "I want to cancel my appointment", "s-amb-3")
        seeded_db.expire_all()
        for appointment_id in (1, 2):
            assert (
                seeded_db.get(Appointment, appointment_id).status
                is AppointmentStatus.CONFIRMED
            )

    def test_naming_the_department_resolves_it(self, two_appointments, seeded_db):
        result = turn(
            two_appointments, "please cancel my Neurology appointment", "s-amb-4"
        )
        assert result.status == WorkflowStatus.PENDING_CONFIRMATION.value

        seeded_db.expire_all()
        assert seeded_db.get(WorkflowRun, result.run_id).proposed_appointment_id == 2

    def test_the_named_one_is_the_one_that_gets_cancelled(
        self, two_appointments, seeded_db
    ):
        turn(two_appointments, "please cancel my Neurology appointment", "s-amb-5")
        turn(two_appointments, "yes", "s-amb-5")

        seeded_db.expire_all()
        assert seeded_db.get(Appointment, 2).status is AppointmentStatus.CANCELLED
        assert seeded_db.get(Appointment, 1).status is AppointmentStatus.CONFIRMED


class TestTheCancelTokenCollision:
    """The word lands in three places and must not be confused between them."""

    def test_a_bare_cancel_at_a_proposal_declines_it(self, patient, seeded_db):
        """Answering a proposal with the bare token means "don't" — not
        "cancel the appointment you just offered to cancel"."""
        offered = turn(patient, "I want to cancel my appointment", "s-tok-1")
        assert offered.status == WorkflowStatus.PENDING_CONFIRMATION.value

        after = turn(patient, "cancel", "s-tok-1")

        seeded_db.expire_all()
        assert after.status == WorkflowStatus.IN_PROGRESS.value
        assert (
            seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).status
            is AppointmentStatus.CONFIRMED
        )

    def test_withdrawing_is_not_cancelling_an_appointment(self, patient, seeded_db):
        """"cancel that request" closes the run. Reading it as an appointment
        cancellation would leave the visit standing while the reply said it had
        been dealt with."""
        turn(patient, "I want to cancel my appointment", "s-tok-2")
        result = turn(patient, "actually, cancel that request", "s-tok-2")

        seeded_db.expire_all()
        assert result.status == WorkflowStatus.CANCELLED.value
        assert (
            seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID).status
            is AppointmentStatus.CONFIRMED
        )


class DriftingLlm(MockLlm):
    """The mock, plus the sentence ``gpt-4o-mini`` actually produced.

    Live: "I found your appointment with Dr. Deepa Krishnan in the ENT
    department on Monday, 3 August 2026, at 9:00 AM. Would you like me to
    reschedule it?" — where 9:00 was the **new** slot and the appointment was
    at 10:00. Two facts of the same shape in one payload, welded into one, and
    the sentence names no new time at all: a patient reading it is being asked
    to approve moving an appointment to the hour it already occupies.

    The mock keeps the two apart, which is why it cannot show this on its own.
    """

    model: str = "drifting-stub"

    def _from_change_proposal(self, payload, step):  # noqa: ANN001
        new_slot = payload.get("new_slot") or {}
        when = str(new_slot.get("start") or "")
        spoken = clock_time(datetime.fromisoformat(when)) if when else "9:00 AM"
        return text_response(
            f"I found your appointment with Dr. Nobody in the Wrong department "
            f"at {spoken}. Would you like me to {step} it?"
        )


class TestAChangeProposalStatesBothTimesFromRows:
    """Item 7. The model may wrap; it may not supply numbers."""

    @pytest.fixture(autouse=True)
    def _drift(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: DriftingLlm()
        )

    def test_the_models_sentence_never_reaches_the_patient(self, patient):
        result = turn(patient, "please reschedule my appointment to next week", "s-drift-1")

        assert "Dr. Nobody" not in result.reply
        assert "Wrong department" not in result.reply
        assert result.author is TraceAuthor.TEMPLATE

    def test_both_times_are_present_and_the_right_way_round(self, patient):
        """The assertion the live sentence fails: the time the patient *has*
        and the time they are being *offered* are different facts, and the
        reply has to carry both, each attached to its own clause."""
        result = turn(patient, "please reschedule my appointment to next week", "s-drift-2")

        session = SessionLocal()
        try:
            appointment = session.get(Appointment, SEEDED_APPOINTMENT_ID)
            run = session.get(WorkflowRun, result.run_id)
            leaving = clock_time(appointment.slot.start_time)
            arriving = clock_time(
                session.get(AppointmentSlot, run.proposed_slot_id).start_time
            )
        finally:
            session.close()

        assert f"currently" in result.reply
        assert leaving in result.reply.split("I can move it to")[0]
        assert arriving in result.reply.split("I can move it to")[1]

    def test_the_reference_code_survives(self, patient):
        """Story 20 rides along: a destructive change names exactly which
        appointment, and a template that dropped the code would be a quieter
        version of the same ambiguity."""
        result = turn(patient, "I want to cancel my appointment", "s-drift-3")
        assert "AC-000001" in result.reply
        assert "Dr. Nobody" not in result.reply


class InventingLlm(MockLlm):
    """A Coordinator that reports an empty schedule it never looked at.

    The live turn, exactly: two ``list_other_slots`` results reading *"No
    department has been decided yet"* — a refusal to search — reported to the
    patient as "there are currently no available appointment slots in the ENT
    department for next week". Two turns later a search of that same window
    returned **72** slots.

    The stub answers the coordinator's classify step with prose instead, which
    is the shape that reaches the patient ungrounded.
    """

    model: str = "inventing-stub"

    def _change_appointment(self, llm_request, done, task, step):  # noqa: ANN001
        return text_response(
            "It appears that there are currently no available appointment "
            "slots in the Cardiology department for next week."
        )


class TestAnAvailabilityClaimNeedsASearchBehindIt:
    """Item 2. A reply about the schedule is grounded or it is not said."""

    @pytest.fixture
    def inventing(self, monkeypatch):
        monkeypatch.setattr(
            "app.agents.base.get_provider", lambda name=None: InventingLlm()
        )

    def test_an_ungrounded_claim_never_reaches_the_patient(self, patient, inventing):
        result = turn(patient, "I want to reschedule my appointment", "s-invent-1")

        assert "no available appointment slots" not in result.reply
        assert "fully booked" not in result.reply
        # The turn still answers. The guard drops the ungrounded sentence, not
        # the turn — a silent reply would be its own defect.
        assert result.reply.strip()

    def test_the_rejection_is_traced(self, patient, inventing):
        result = turn(patient, "I want to reschedule my appointment", "s-invent-2")

        session = SessionLocal()
        try:
            verdicts = [
                event.payload
                for event in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if event.event_type is TraceEventType.GUARD_VERDICT
                and event.payload["guard"] == "reply_claims_availability"
            ]
        finally:
            session.close()

        assert len(verdicts) == 1
        assert verdicts[0]["passed"] is False


class TestARescheduleRunKnowsItsOwnDepartment:
    """Item 2's other half. A reschedule never went through routing, so the
    run's state holds no department — but the department is not missing, it is
    on the appointment being moved.

    Live, "some time next week?" inside a reschedule run reached
    ``list_other_slots``, got "No department has been decided yet", and the
    patient was asked to name a department the system already knew — then
    offered a **cancelled** Dermatology appointment as the likely answer.
    """

    def test_a_timing_question_returns_real_times(self, patient):
        turn(patient, "I want to reschedule my appointment", "s-dept-1")
        result = turn(patient, "some time next week?", "s-dept-1")

        assert "no department" not in result.reply.lower()
        assert "which department" not in result.reply.lower()

        session = SessionLocal()
        try:
            appointment = session.get(Appointment, SEEDED_APPOINTMENT_ID)
            department = appointment.department.name
        finally:
            session.close()

        # Real rows, in the appointment's own department, inside the window.
        assert "August" in result.reply
        assert department == "Cardiology"

    def test_the_times_offered_are_that_departments(self, patient):
        """Not merely non-empty: a search in the wrong department would also
        return times, and would move the patient's Cardiology visit to a
        clinic that has nothing to do with it."""
        result = turn(patient, "I want to reschedule my appointment", "s-dept-2")
        turn(patient, "some time next week?", "s-dept-2")

        session = SessionLocal()
        try:
            run = session.get(WorkflowRun, result.run_id)
            offered = (run.state or {}).get("offered_slot_ids") or []
            departments = {
                session.get(AppointmentSlot, slot_id).doctor.department_id
                for slot_id in offered
            }
            appointment = session.get(Appointment, SEEDED_APPOINTMENT_ID)
        finally:
            session.close()

        assert offered, "nothing was offered; the search never ran"
        assert departments == {appointment.department_id}
