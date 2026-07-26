"""The deterministic half of the workflow — the "code disposes" side.

Three modules, none of which call an LLM:

* :mod:`app.workflow.state_machine` — the pinned transition table, applied as a
  compare-and-swap and written to both ledgers.
* :mod:`app.workflow.plan` — validation of the Coordinator's proposed plan
  against a closed enum, plus the dependency order and the superset rule.
* :mod:`app.workflow.mapping` — how a message arriving during an active run is
  classified, and what each class is allowed to do.

Each of them takes a *proposal* — a status the caller wants, a plan the model
emitted, a class the model chose — and decides whether it becomes a
consequence. Nothing here trusts its input; everything here is testable without
a network call.
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
