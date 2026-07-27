"""Tool functions — plain Python, framework-agnostic.

Nothing here imports FastAPI or the agent framework. Tools take a session, the
acting user where a mutation is involved, and primitives; they return
JSON-serialisable dicts. That keeps two promises at once: the golden set can
diff their output directly, and swapping ADK for LangGraph touches the
orchestrator rather than every tool.

Every result dict is shape-stable — a refusal carries the same keys as a
success, with ``ok``/``resolved``/``status`` saying which it is — so callers
never branch on which keys happen to be present.

Mutating tools take the acting user and enforce **role and ownership** through
``app.auth.ownership``, which audits a denial before raising. Ownership guards
run before any mutation, because that audit commits the acting session.
"""

from app.tools.appointments import (
    book_appointment,
    cancel_appointment,
    describe_appointment,
    list_patient_appointments,
    reschedule_appointment,
)
from app.tools.availability import find_available_slots, get_slot
from app.tools.confirmations import render_confirmation
from app.tools.dates import resolve_date
from app.tools.departments import (
    list_departments,
    resolve_department,
    validate_department,
)
from app.tools.documents import (
    apply_verification,
    checksum_bytes,
    describe_document,
    diff_required_documents,
    extract_document_text,
    find_duplicate,
    ingest_document,
    list_flagged_documents,
    list_patient_documents,
    list_unverified_documents,
)
from app.tools.patients import get_patient_context
from app.tools.reminders import list_due_reminders, list_patient_reminders
from app.tools.tasks import (
    close_followup_tasks,
    create_escalation,
    list_open_escalations,
    list_open_tasks,
    upsert_followup_task,
)

__all__ = [
    "apply_verification",
    "book_appointment",
    "cancel_appointment",
    "checksum_bytes",
    "close_followup_tasks",
    "create_escalation",
    "describe_appointment",
    "describe_document",
    "diff_required_documents",
    "extract_document_text",
    "find_available_slots",
    "find_duplicate",
    "get_patient_context",
    "get_slot",
    "ingest_document",
    "list_departments",
    "list_flagged_documents",
    "list_due_reminders",
    "list_open_escalations",
    "list_open_tasks",
    "list_patient_appointments",
    "list_patient_documents",
    "list_unverified_documents",
    "list_patient_reminders",
    "render_confirmation",
    "reschedule_appointment",
    "resolve_date",
    "resolve_department",
    "upsert_followup_task",
    "validate_department",
]
