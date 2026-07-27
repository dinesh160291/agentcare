"""Appointments — read only, and deliberately so.

There is no ``POST /appointments``, no ``PATCH``, no ``DELETE``. Booking,
rescheduling, and cancelling all happen through the workflow, which proposes an
exact doctor, date and time and waits for the patient to confirm before
anything is committed. A direct mutation endpoint would be a second road to the
same state change with none of that on it — the confirmation step, the slot
compare-and-swap, the reminder derivation — and the second road is the one a
client eventually takes because it is shorter.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.dependencies import CurrentUser, PatientUser
from app.auth.ownership import get_owned_or_404, patient_profile_for
from app.db import get_session
from app.models import Appointment
from app.tools import describe_appointment, list_patient_appointments

router = APIRouter(tags=["appointments"])

DbSession = Annotated[Session, Depends(get_session)]


@router.get("/appointments")
def list_appointments(
    user: PatientUser, session: DbSession, live_only: bool = False
) -> list[dict[str, Any]]:
    """Every appointment the caller has, whatever its status."""
    profile = patient_profile_for(session, user)
    return list_patient_appointments(session, patient_id=profile.id, live_only=live_only)


@router.get("/appointments/{appointment_id}")
def read_appointment(
    appointment_id: int, user: CurrentUser, session: DbSession
) -> dict[str, Any]:
    appointment = get_owned_or_404(session, Appointment, appointment_id, user)
    return describe_appointment(session, appointment)
