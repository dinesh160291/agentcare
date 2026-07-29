"""The deterministic half of the workflow — the "code disposes" side.

None of these modules call an LLM:

* :mod:`app.workflow.state_machine` — the pinned transition table, applied as a
  compare-and-swap and written to both ledgers.
* :mod:`app.workflow.plan` — validation of the Coordinator's proposed plan
  against a closed enum, plus the dependency order and the superset rule.
* :mod:`app.workflow.mapping` — how a message arriving during an active run is
  classified, and what each class is allowed to do.
* :mod:`app.workflow.confirmation` — the exact tokens that may commit.
* :mod:`app.workflow.selection` — which of the times already shown a message
  picks. It holds a slot; it cannot commit.
* :mod:`app.workflow.targets` — which of the patient's appointments a message
  points at, checked against the id the model proposed to act on.
* :mod:`app.workflow.queries` — read-only listing questions, answered from rows.
* :mod:`app.workflow.replies` — what the patient is told, assembled from rows.

Each of them takes a *proposal* — a status the caller wants, a plan the model
emitted, a class the model chose, an appointment id it picked — and decides
whether it becomes a consequence. Nothing here trusts its input; everything
here is testable without a network call.
"""

from app.workflow.state_machine import (
    INITIAL_STATUSES,
    LEGAL_TRANSITIONS,
    TransitionResult,
    create_run,
    is_legal,
    transition,
)

__all__ = [
    "INITIAL_STATUSES",
    "LEGAL_TRANSITIONS",
    "TransitionResult",
    "create_run",
    "is_legal",
    "transition",
]
