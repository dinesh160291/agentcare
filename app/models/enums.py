"""Closed vocabularies.

Every status in the system is an enum, not a free string. This is the storage
half of "the model proposes; the code disposes": a classification only becomes
a stored consequence after it has been validated against one of these sets.

Stored as VARCHAR with a CHECK constraint (``native_enum=False``) so the schema
behaves identically on SQLite and PostgreSQL.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String-valued enum that compares and serialises as its value."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class UserRole(StrEnum):
    PATIENT = "patient"
    STAFF = "staff"


class WorkflowStatus(StrEnum):
    """States of ``WorkflowRun`` — the machine everything else hangs off.

    Terminal for automation: COMPLETED, REJECTED, FAILED, CANCELLED, ESCALATED.
    """

    IN_PROGRESS = "in_progress"
    PENDING_CONFIRMATION = "pending_confirmation"
    PENDING_REVIEW = "pending_review"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ESCALATED = "escalated"


TERMINAL_WORKFLOW_STATUSES = frozenset(
    {
        WorkflowStatus.COMPLETED,
        WorkflowStatus.REJECTED,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
        WorkflowStatus.ESCALATED,
    }
)


class CancellationReason(StrEnum):
    WITHDRAWN = "withdrawn"
    WITHDRAWN_DURING_REVIEW = "withdrawn_during_review"
    SUPERSEDED = "superseded"


class PlanStep(StrEnum):
    """The closed enum a Coordinator plan is drawn from.

    A plan is a schema-validated list of these — never freeform prose. Code
    cannot enforce dependency order on a plan it cannot parse.
    """

    ROUTE = "route"
    BOOK = "book"
    #: The other two appointment verbs are steps of their own rather than
    #: ``BOOK`` carrying a mode. ``BOOK``'s closure pulls in the
    #: required-documents diff, so a cancel modelled as a book would open a
    #: missing-documents task for the appointment it is cancelling.
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"
    DOCUMENTS = "documents"
    FOLLOW_UP = "follow_up"


#: The three verbs that mutate an appointment. Exactly one may appear in a
#: plan: the pending proposal is a single ``ProposedAction``, so a plan naming
#: two of them has no answer to "what is the patient confirming?".
APPOINTMENT_VERBS: frozenset[PlanStep] = frozenset(
    {PlanStep.BOOK, PlanStep.RESCHEDULE, PlanStep.CANCEL}
)


class ProposedAction(StrEnum):
    """What a pending confirmation is actually proposing."""

    BOOK = "book"
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"


#: The plan step each proposal verb belongs to. One mapping, so the dispatch in
#: the orchestrator and the step bookkeeping cannot disagree about which verb
#: is in flight.
STEP_FOR_PROPOSED_ACTION: dict[ProposedAction, PlanStep] = {
    ProposedAction.BOOK: PlanStep.BOOK,
    ProposedAction.RESCHEDULE: PlanStep.RESCHEDULE,
    ProposedAction.CANCEL: PlanStep.CANCEL,
}


class AppointmentStatus(StrEnum):
    """Every transition has a named actor.

    pending -> confirmed (patient), confirmed -> cancelled (patient/agent),
    confirmed -> completed (the poll job's sweep — time-triggered, no human
    actor exists), completed <-> missed (staff typed action).
    """

    PENDING = "pending"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    MISSED = "missed"
    CANCELLED = "cancelled"


class SlotStatus(StrEnum):
    AVAILABLE = "available"
    BOOKED = "booked"


class DocumentStatus(StrEnum):
    """``FLAGGED`` is the deterministic consequence of a probabilistic verdict:
    the model proposes that content does not match the declared type; code sets
    the status and routes it to a human."""

    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    FLAGGED = "flagged"
    REJECTED = "rejected"


class ReminderStatus(StrEnum):
    PENDING = "pending"
    SENT = "sent"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ReminderType(StrEnum):
    APPOINTMENT = "appointment"
    DOCUMENT = "document"
    FOLLOW_UP = "follow_up"


class EscalationStatus(StrEnum):
    """Two lifecycles, deliberately named apart.

    Review-type escalations are approved or rejected. Safety and emergency
    escalations are acknowledged and resolved — staff UI language must never
    imply that someone "approved" an emergency.
    """

    OPEN = "open"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class EscalationKind(StrEnum):
    SAFETY = "safety"
    LOW_CONFIDENCE_ROUTING = "low_confidence_routing"
    UNSUPPORTED_REQUEST = "unsupported_request"
    #: A run that failed on a budget. The failure template has always said "I've
    #: flagged it for them"; until this existed it was a sentence about nothing,
    #: and a patient whose request died twice was told twice that help was
    #: coming while the staff queue stayed empty.
    SYSTEM_FAILURE = "system_failure"


class FollowUpTaskType(StrEnum):
    MISSING_DOCUMENTS = "missing_documents"
    POST_VISIT = "post_visit"
    MISSED_VISIT = "missed_visit"


class FollowUpTaskStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class NotificationKind(StrEnum):
    REMINDER = "reminder"
    WORKFLOW_UPDATE = "workflow_update"
    STAFF_DECISION = "staff_decision"


class TraceEventType(StrEnum):
    """The turn grammar's alphabet.

    A well-formed turn is:
    inbound -> guard verdict -> [llm request -> (llm response | llm error)
    -> validation -> tool call -> tool result]* -> (transition) -> outbound.
    """

    INBOUND = "inbound"
    GUARD_VERDICT = "guard_verdict"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_ERROR = "llm_error"
    VALIDATION = "validation"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TRANSITION = "transition"
    OUTBOUND = "outbound"


class TraceAuthor(StrEnum):
    """Who wrote this event's content.

    Without ``TEMPLATE`` and ``GUARD``, code-authored replies are invisible in
    the trace and the timeline implies an LLM said things it never said.
    """

    PATIENT_MESSAGE = "patient-message"
    PATIENT_ACTION = "patient-action"
    STAFF_ACTION = "staff-action"
    LLM = "llm"
    TEMPLATE = "template"
    GUARD = "guard"
    SYSTEM = "system"


class MessageClass(StrEnum):
    """How an incoming message relates to an already-active run.

    Evaluated in this order: WITHDRAWAL, OFF_TOPIC, SIDE_QUESTION,
    COMPLEMENTARY, CONFLICTING, CONTINUATION. Only CONFLICTING supersedes.
    """

    WITHDRAWAL = "withdrawal"
    OFF_TOPIC = "off_topic"
    SIDE_QUESTION = "side_question"
    COMPLEMENTARY = "complementary"
    CONFLICTING = "conflicting"
    CONTINUATION = "continuation"
