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

import pytest

from app import clock
from app.models import (
    Appointment,
    AppointmentSlot,
    AppointmentStatus,
    Department,
    Doctor,
    SlotStatus,
    User,
    WorkflowRun,
    WorkflowStatus,
)
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
        lowered = result.reply.lower()
        assert " from " in lowered and " to " in lowered

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
