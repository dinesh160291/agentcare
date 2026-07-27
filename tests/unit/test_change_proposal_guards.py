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

import pytest

from app.agents.toolbelt import Toolbelt
from app.models import (
    Appointment,
    AppointmentSlot,
    AppointmentStatus,
    Department,
    Doctor,
    ProposedAction,
    SlotStatus,
    User,
    WorkflowRun,
    WorkflowStatus,
)
from app.trace import TraceWriter

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
    return (
        seeded_db.query(AppointmentSlot)
        .join(Doctor, Doctor.id == AppointmentSlot.doctor_id)
        .join(Department, Department.id == Doctor.department_id)
        .filter(
            Department.name == department_name,
            AppointmentSlot.status == SlotStatus.AVAILABLE,
        )
        .order_by(AppointmentSlot.start_time)
        .first()
    )


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
