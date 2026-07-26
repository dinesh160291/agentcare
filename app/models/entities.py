"""SQLAlchemy ORM models — the persistent shape of the system.

Covers the hackathon's suggested data model plus the four entities the design
requires beyond it: ``DepartmentRequiredDocument`` (required-docs rules as a
table, so the seed and the diff cannot disagree), ``DepartmentSynonym``
(deterministic routing vocabulary), ``TraceEvent`` (the observability system of
record), ``Notification`` (the delivery channel for staff-caused terminals),
and ``FollowUpTask``.

``patient_id`` throughout refers to ``PatientProfile.id``, never ``User.id``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app import clock
from app.db import Base
from app.models.enums import (
    AppointmentStatus,
    DocumentStatus,
    EscalationKind,
    EscalationStatus,
    FollowUpTaskStatus,
    FollowUpTaskType,
    NotificationKind,
    ProposedAction,
    ReminderStatus,
    ReminderType,
    SlotStatus,
    TraceAuthor,
    TraceEventType,
    UserRole,
    WorkflowStatus,
)


def _enum(enum_cls: type, name: str) -> SAEnum:
    """Portable enum column: VARCHAR + CHECK on both SQLite and PostgreSQL."""
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=False,
        values_callable=lambda e: [m.value for m in e],
        length=40,
    )


def _now() -> datetime:
    """Timestamp default routed through the clock seam, never datetime.now()."""
    return clock.now()


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_now, onupdate=_now, nullable=False
    )


# ---------------------------------------------------------------------------
# Identity and access
# ---------------------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(_enum(UserRole, "user_role"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)

    profile: Mapped["PatientProfile | None"] = relationship(
        back_populates="user", uselist=False
    )

    @property
    def is_staff(self) -> bool:
        return self.role == UserRole.STAFF


class PatientProfile(Base, TimestampMixin):
    __tablename__ = "patient_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    phone: Mapped[str | None] = mapped_column(String(40))
    preferred_language: Mapped[str] = mapped_column(String(40), default="English")
    emergency_contact: Mapped[str | None] = mapped_column(String(160))

    user: Mapped[User] = relationship(back_populates="profile")


# ---------------------------------------------------------------------------
# Hospital reference data
# ---------------------------------------------------------------------------


class Department(Base):
    __tablename__ = "departments"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    doctors: Mapped[list["Doctor"]] = relationship(back_populates="department")
    synonyms: Mapped[list["DepartmentSynonym"]] = relationship(
        back_populates="department", cascade="all, delete-orphan"
    )
    required_documents: Mapped[list["DepartmentRequiredDocument"]] = relationship(
        back_populates="department", cascade="all, delete-orphan"
    )


class DepartmentSynonym(Base):
    """Routing vocabulary as rows, so department resolution stays deterministic."""

    __tablename__ = "department_synonyms"
    __table_args__ = (UniqueConstraint("term", name="uq_department_synonym_term"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    department_id: Mapped[int] = mapped_column(
        ForeignKey("departments.id", ondelete="CASCADE"), nullable=False
    )
    term: Mapped[str] = mapped_column(String(80), nullable=False, index=True)

    department: Mapped[Department] = relationship(back_populates="synonyms")


class DepartmentRequiredDocument(Base):
    """Required-docs rules as a table, not a delimited column.

    The diff needs to tell mandatory from optional to report "missing" rather
    than "nice to have", and a column would force every consumer to re-parse.
    """

    __tablename__ = "department_required_documents"
    __table_args__ = (
        UniqueConstraint(
            "department_id", "document_type", name="uq_department_required_document"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    department_id: Mapped[int] = mapped_column(
        ForeignKey("departments.id", ondelete="CASCADE"), nullable=False
    )
    document_type: Mapped[str] = mapped_column(String(80), nullable=False)
    mandatory: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    department: Mapped[Department] = relationship(back_populates="required_documents")


class Doctor(Base):
    __tablename__ = "doctors"

    id: Mapped[int] = mapped_column(primary_key=True)
    department_id: Mapped[int] = mapped_column(
        ForeignKey("departments.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    department: Mapped[Department] = relationship(back_populates="doctors")
    slots: Mapped[list["AppointmentSlot"]] = relationship(back_populates="doctor")


class AppointmentSlot(Base):
    __tablename__ = "appointment_slots"
    __table_args__ = (
        UniqueConstraint("doctor_id", "start_time", name="uq_slot_doctor_start"),
        Index("ix_slot_lookup", "doctor_id", "status", "start_time"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    doctor_id: Mapped[int] = mapped_column(
        ForeignKey("doctors.id", ondelete="CASCADE"), nullable=False
    )
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[SlotStatus] = mapped_column(
        _enum(SlotStatus, "slot_status"), default=SlotStatus.AVAILABLE, nullable=False
    )

    doctor: Mapped[Doctor] = relationship(back_populates="slots")


# ---------------------------------------------------------------------------
# Patient-facing records
# ---------------------------------------------------------------------------


class Appointment(Base, TimestampMixin):
    __tablename__ = "appointments"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patient_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    doctor_id: Mapped[int] = mapped_column(ForeignKey("doctors.id"), nullable=False)
    slot_id: Mapped[int | None] = mapped_column(ForeignKey("appointment_slots.id"))
    department_id: Mapped[int] = mapped_column(ForeignKey("departments.id"), nullable=False)
    status: Mapped[AppointmentStatus] = mapped_column(
        _enum(AppointmentStatus, "appointment_status"),
        default=AppointmentStatus.PENDING,
        nullable=False,
    )
    reason: Mapped[str] = mapped_column(Text, default="")
    reference_code: Mapped[str | None] = mapped_column(String(20), unique=True)

    doctor: Mapped[Doctor] = relationship()
    slot: Mapped[AppointmentSlot | None] = relationship()
    department: Mapped[Department] = relationship()
    reminders: Mapped[list["Reminder"]] = relationship(back_populates="appointment")


class PatientDocument(Base):
    __tablename__ = "patient_documents"
    __table_args__ = (
        Index("ix_document_patient_checksum", "patient_id", "checksum"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patient_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: What the patient said this file is.
    declared_type: Mapped[str] = mapped_column(String(80), nullable=False)
    #: What verification concluded it is (null until verification runs).
    detected_type: Mapped[str | None] = mapped_column(String(80))
    document_type: Mapped[str] = mapped_column(String(80), nullable=False)
    #: Server-generated. The client filename is a path-traversal vector and is
    #: kept only as a display label.
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(255))
    mime_type: Mapped[str | None] = mapped_column(String(120))
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    document_date: Mapped[date | None] = mapped_column(Date)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[DocumentStatus] = mapped_column(
        _enum(DocumentStatus, "document_status"),
        default=DocumentStatus.PENDING_VERIFICATION,
        nullable=False,
    )
    verification_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)


# ---------------------------------------------------------------------------
# Workflow state
# ---------------------------------------------------------------------------


class WorkflowRun(Base, TimestampMixin):
    """The deterministic machine everything hangs off.

    The pending proposal is typed state on this row, not prose in a transcript:
    confirm-before-commit has to survive history windowing, session expiry, and
    restarts.
    """

    __tablename__ = "workflow_runs"
    __table_args__ = (
        CheckConstraint("non_answer_count >= 0", name="ck_run_non_answer_count"),
        Index("ix_run_patient_status", "patient_id", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patient_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[WorkflowStatus] = mapped_column(
        _enum(WorkflowStatus, "workflow_status"), nullable=False
    )
    current_step: Mapped[str | None] = mapped_column(String(40))
    #: Validated plan: a list of PlanStep values, never freeform prose.
    plan: Mapped[list[str]] = mapped_column(JSON, default=list)
    completed_steps: Mapped[list[str]] = mapped_column(JSON, default=list)
    #: Arbitrary workflow state (resolved department, dates, ids).
    state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    #: The patient's request text. Off-topic messages must never append here.
    request_text: Mapped[str] = mapped_column(Text, default="")

    # --- pending proposal (typed, never prose) ---------------------------
    proposed_action: Mapped[ProposedAction | None] = mapped_column(
        _enum(ProposedAction, "proposed_action")
    )
    proposed_slot_id: Mapped[int | None] = mapped_column(ForeignKey("appointment_slots.id"))
    proposed_appointment_id: Mapped[int | None] = mapped_column(ForeignKey("appointments.id"))

    # --- boundedness counters --------------------------------------------
    non_answer_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    replan_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    cancellation_reason: Mapped[str | None] = mapped_column(String(40))
    session_id: Mapped[str | None] = mapped_column(String(80), index=True)

    escalations: Mapped[list["Escalation"]] = relationship(back_populates="workflow_run")

    @property
    def is_terminal(self) -> bool:
        from app.models.enums import TERMINAL_WORKFLOW_STATUSES

        return self.status in TERMINAL_WORKFLOW_STATUSES

    def clear_proposal(self) -> None:
        """Drop the pending proposal, and the stall counter that belongs to it.

        Called on commit failure as well as on success: a proposal pointing at
        a dead slot, still confirmable, is a loop with no exit.

        The non-answer count goes with it because it counts non-answers *to
        this proposal*. Carrying it forward would make the next proposal open
        already part-way through its patience, and a patient who stalled once
        would meet the terse framing on their first look at a new time.
        """
        self.proposed_action = None
        self.proposed_slot_id = None
        self.proposed_appointment_id = None
        self.non_answer_count = 0


class Escalation(Base):
    """One open escalation per run — repeats attach rather than multiply."""

    __tablename__ = "escalations"

    id: Mapped[int] = mapped_column(primary_key=True)
    workflow_run_id: Mapped[int] = mapped_column(
        ForeignKey("workflow_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[EscalationKind] = mapped_column(
        _enum(EscalationKind, "escalation_kind"), nullable=False
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[EscalationStatus] = mapped_column(
        _enum(EscalationStatus, "escalation_status"),
        default=EscalationStatus.OPEN,
        nullable=False,
    )
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    latest_message: Mapped[str | None] = mapped_column(Text)
    reviewed_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    resolution_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_now, onupdate=_now, nullable=False
    )

    workflow_run: Mapped[WorkflowRun] = relationship(back_populates="escalations")


# ---------------------------------------------------------------------------
# Derived records (every one has an update rule in its source's transaction)
# ---------------------------------------------------------------------------


class Reminder(Base):
    __tablename__ = "reminders"
    __table_args__ = (Index("ix_reminder_due", "status", "scheduled_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patient_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    appointment_id: Mapped[int | None] = mapped_column(
        ForeignKey("appointments.id", ondelete="CASCADE"), index=True
    )
    reminder_type: Mapped[ReminderType] = mapped_column(
        _enum(ReminderType, "reminder_type"), nullable=False
    )
    scheduled_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[ReminderStatus] = mapped_column(
        _enum(ReminderStatus, "reminder_status"),
        default=ReminderStatus.PENDING,
        nullable=False,
    )
    #: Incremented before each delivery attempt, so a row that poisons the
    #: worker still reaches a terminal state instead of being retried forever.
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    message: Mapped[str] = mapped_column(Text, default="")
    sent_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)

    appointment: Mapped[Appointment | None] = relationship(back_populates="reminders")


class FollowUpTask(Base, TimestampMixin):
    """Upserted, never blindly inserted — one open task per (patient, type, appointment)."""

    __tablename__ = "follow_up_tasks"
    __table_args__ = (
        Index("ix_task_open", "patient_id", "task_type", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patient_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    appointment_id: Mapped[int | None] = mapped_column(
        ForeignKey("appointments.id", ondelete="CASCADE")
    )
    task_type: Mapped[FollowUpTaskType] = mapped_column(
        _enum(FollowUpTaskType, "follow_up_task_type"), nullable=False
    )
    status: Mapped[FollowUpTaskStatus] = mapped_column(
        _enum(FollowUpTaskStatus, "follow_up_task_status"),
        default=FollowUpTaskStatus.OPEN,
        nullable=False,
    )
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    due_date: Mapped[date | None] = mapped_column(Date)


class Notification(Base):
    """In-app delivery channel.

    ``reminder_id`` is unique so the poll job's insert is idempotent: a crash
    between "delivered" and "marked sent" costs a bounded invisible re-attempt,
    not a duplicate notification.
    """

    __tablename__ = "notifications"
    __table_args__ = (
        UniqueConstraint("reminder_id", name="uq_notification_reminder"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patient_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[NotificationKind] = mapped_column(
        _enum(NotificationKind, "notification_kind"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    body: Mapped[str] = mapped_column(Text, default="")
    read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reminder_id: Mapped[int | None] = mapped_column(ForeignKey("reminders.id"))
    workflow_run_id: Mapped[int | None] = mapped_column(ForeignKey("workflow_runs.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class AuditEvent(Base):
    """Who did what to which row. Written for every state transition, and by
    the scheduler, which has no run and no session to trace against."""

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_entity", "entity_type", "entity_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    #: "system" when the actor is the scheduler rather than a user.
    actor_kind: Mapped[str] = mapped_column(String(20), default="user", nullable=False)
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(60), nullable=False)
    entity_id: Mapped[int | None] = mapped_column(Integer)
    event_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)


class TraceEvent(Base):
    """The observability system of record.

    ``workflow_run_id`` is nullable on purpose: off-topic turns and safety
    screens can fire before any run exists, and those turns still have to be
    bracketed. ``turn_id`` groups the events of one turn; ``seq`` orders them.
    """

    __tablename__ = "trace_events"
    __table_args__ = (
        Index("ix_trace_turn", "turn_id", "seq"),
        Index("ix_trace_run", "workflow_run_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    workflow_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("workflow_runs.id", ondelete="CASCADE"), index=True
    )
    session_id: Mapped[str | None] = mapped_column(String(80), index=True)
    turn_id: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[TraceEventType] = mapped_column(
        _enum(TraceEventType, "trace_event_type"), nullable=False
    )
    author: Mapped[TraceAuthor | None] = mapped_column(_enum(TraceAuthor, "trace_author"))
    agent_name: Mapped[str | None] = mapped_column(String(60))
    #: Redacted before it gets here — redaction happens at the choke points.
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    #: Pairs an llm_response/llm_error back to its llm_request.
    correlation_id: Mapped[str | None] = mapped_column(String(80), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)
