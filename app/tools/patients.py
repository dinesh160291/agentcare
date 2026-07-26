"""Patient context — the Coordinator's view of who it is talking to.

One call, assembled from persisted rows, so an agent never has to ask the
patient something the database already knows. Ownership-scoped: a patient sees
their own record, staff may look up any patient, and there is no third path.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.auth.ownership import patient_profile_for, require_own_patient_id
from app.models import (
    Appointment,
    AppointmentSlot,
    AppointmentStatus,
    Department,
    Doctor,
    PatientProfile,
    User,
    UserRole,
    WorkflowRun,
)
from app.models.enums import TERMINAL_WORKFLOW_STATUSES
from app.tools.documents import list_patient_documents
from app.tools.reminders import list_patient_reminders
from app.tools.tasks import list_open_tasks


def _upcoming_appointments(session: Session, patient_id: int) -> list[dict[str, Any]]:
    rows = (
        session.query(Appointment, AppointmentSlot, Doctor, Department)
        .join(AppointmentSlot, AppointmentSlot.id == Appointment.slot_id)
        .join(Doctor, Doctor.id == Appointment.doctor_id)
        .join(Department, Department.id == Appointment.department_id)
        .filter(
            Appointment.patient_id == patient_id,
            Appointment.status.in_(
                (AppointmentStatus.PENDING, AppointmentStatus.CONFIRMED)
            ),
        )
        .order_by(AppointmentSlot.start_time, Appointment.id)
        .all()
    )
    return [
        {
            "appointment_id": appointment.id,
            "reference_code": appointment.reference_code,
            "department_id": department.id,
            "department_name": department.name,
            "doctor_name": doctor.name,
            "start": slot.start_time.isoformat(),
            "end": slot.end_time.isoformat(),
            "status": appointment.status.value,
            "reason": appointment.reason,
        }
        for appointment, slot, doctor, department in rows
    ]


def get_patient_context(
    session: Session, user: User, *, patient_id: int | None = None
) -> dict[str, Any]:
    """Everything already known about a patient.

    A patient may only read their own context; staff must name a
    ``patient_id``. Both routes go through the ownership helpers, so a patient
    passing someone else's id gets the same 404 as for an id that never
    existed.
    """
    if user.role == UserRole.STAFF:
        if patient_id is None:
            raise ValueError("Staff must supply a patient_id")
        require_own_patient_id(session, user, patient_id)
        profile = session.get(PatientProfile, patient_id)
        if profile is None:
            from app.errors import RecordNotFound

            raise RecordNotFound("PatientProfile", patient_id)
    else:
        profile = patient_profile_for(session, user)
        if patient_id is not None:
            require_own_patient_id(session, user, patient_id)

    active_run = (
        session.query(WorkflowRun)
        .filter(
            WorkflowRun.patient_id == profile.id,
            WorkflowRun.status.notin_([s.value for s in TERMINAL_WORKFLOW_STATUSES]),
        )
        .order_by(WorkflowRun.id.desc())
        .first()
    )

    return {
        "patient_id": profile.id,
        "name": profile.user.name,
        "preferred_language": profile.preferred_language,
        "date_of_birth": profile.date_of_birth.isoformat() if profile.date_of_birth else None,
        "phone": profile.phone,
        "upcoming_appointments": _upcoming_appointments(session, profile.id),
        "documents": list_patient_documents(session, patient_id=profile.id),
        "open_tasks": list_open_tasks(session, patient_id=profile.id),
        "pending_reminders": list_patient_reminders(session, patient_id=profile.id),
        "active_run": (
            {
                "run_id": active_run.id,
                "status": active_run.status.value,
                "current_step": active_run.current_step,
                "plan": active_run.plan,
                "has_pending_proposal": active_run.proposed_action is not None,
            }
            if active_run
            else None
        ),
    }
