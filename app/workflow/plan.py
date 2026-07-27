"""The Coordinator's plan: a typed artifact, not a paragraph.

The model decides *which* specialists a request needs — a document-only upload
should not drag routing and booking along behind it. But it emits that decision
as a list drawn from :class:`~app.models.enums.PlanStep`, and everything after
that point is code:

* **Validation.** Anything that is not a list of known steps is rejected and
  handed to the retry ladder. Code cannot enforce ordering on a plan it cannot
  parse, which is the whole argument against freeform delegation.
* **Superset closure.** Each step implies the steps it cannot work without, and
  the plan is the union of those closures. A booking intent therefore always
  arrives as the full booking plan, and the document steps ride inside it. The
  reverse never happens: documents-only stays documents-only.
* **Canonical order.** Routing precedes booking; the required-documents diff
  precedes nothing it depends on. Both orderings are properties of the domain,
  not of the sentence the patient happened to type.

The silent failure this prevents is worth naming: a two-intent message whose
plan comes back as documents-only *succeeds*. The patient is told about their
uploaded ECG and never learns the appointment they asked for was not booked.
"""

from __future__ import annotations

from app.errors import BudgetExceeded, PlanRejected
from app.models import APPOINTMENT_VERBS, PlanStep, WorkflowRun

#: Dependency order. Routing first (the department is an input to everything
#: else), the appointment verbs next, the required-documents diff after the
#: department is known, follow-up last because it summarises the rest.
CANONICAL_ORDER: tuple[PlanStep, ...] = (
    PlanStep.ROUTE,
    PlanStep.BOOK,
    PlanStep.RESCHEDULE,
    PlanStep.CANCEL,
    PlanStep.DOCUMENTS,
    PlanStep.FOLLOW_UP,
)

#: What each step cannot work without. ``BOOK`` closes over the whole booking
#: plan: a new appointment needs a department, brings required-document rules
#: with it, and produces something to follow up on.
#:
#: The other two verbs close over far less, and the difference is the argument
#: for their existing at all. ``RESCHEDULE`` keeps the department and the
#: document requirements it was booked with — only the time moves — but the
#: reminder moves with it, so there is something to summarise. ``CANCEL``
#: closes over nothing: routing a request that is being withdrawn is pointless,
#: and diffing required documents for it would open a task telling the patient
#: to prepare for an appointment they have just called off.
STEP_CLOSURE: dict[PlanStep, frozenset[PlanStep]] = {
    PlanStep.ROUTE: frozenset({PlanStep.ROUTE}),
    PlanStep.BOOK: frozenset(
        {PlanStep.ROUTE, PlanStep.BOOK, PlanStep.DOCUMENTS, PlanStep.FOLLOW_UP}
    ),
    PlanStep.RESCHEDULE: frozenset({PlanStep.RESCHEDULE, PlanStep.FOLLOW_UP}),
    PlanStep.CANCEL: frozenset({PlanStep.CANCEL}),
    PlanStep.DOCUMENTS: frozenset({PlanStep.DOCUMENTS}),
    PlanStep.FOLLOW_UP: frozenset({PlanStep.FOLLOW_UP}),
}


def _as_step(value: object) -> PlanStep:
    if isinstance(value, PlanStep):
        return value
    if not isinstance(value, str):
        raise PlanRejected(f"Plan step {value!r} is not a step name.")
    try:
        return PlanStep(value)
    except ValueError:
        known = ", ".join(step.value for step in CANONICAL_ORDER)
        raise PlanRejected(
            f"Unknown plan step {value!r}. The plan enum is closed: {known}."
        ) from None


def _order(steps: set[PlanStep]) -> list[PlanStep]:
    return [step for step in CANONICAL_ORDER if step in steps]


def validate_plan(proposed: object) -> list[PlanStep]:
    """Turn a model's proposal into an executable plan, or refuse it.

    :raises PlanRejected: the proposal is not a non-empty list of known steps.
    """
    if isinstance(proposed, str) or not isinstance(proposed, (list, tuple)):
        raise PlanRejected(
            f"A plan must be a list of steps, got {type(proposed).__name__}. "
            "Freeform prose is a slip nobody can validate."
        )
    if not proposed:
        raise PlanRejected("A plan must contain at least one step.")

    steps = {_as_step(item) for item in proposed}

    # One appointment verb, at most. The pending proposal is a single
    # ``ProposedAction`` and the commit dispatches on it, so a plan naming two
    # verbs would commit one and drop the other — a half-executed request that
    # reports success. Caught here, it is a re-plan instead.
    verbs = steps & APPOINTMENT_VERBS
    if len(verbs) > 1:
        named = ", ".join(sorted(verb.value for verb in verbs))
        raise PlanRejected(
            f"A plan may name at most one appointment action, got: {named}. "
            "One request confirms one thing."
        )

    closed: set[PlanStep] = set()
    for step in steps:
        closed |= STEP_CLOSURE[step]
    return _order(closed)


def next_step(run: WorkflowRun) -> PlanStep | None:
    """The first planned step that has not been completed."""
    done = set(run.completed_steps or [])
    for value in run.plan or []:
        if value not in done:
            return _as_step(value)
    return None


def is_plan_complete(run: WorkflowRun) -> bool:
    """Whether every planned step is done.

    An empty plan is *not* complete: a run whose plan never got written has
    done nothing, and reporting it finished would transition it to
    ``completed`` on the strength of an absence.
    """
    if not run.plan:
        return False
    return next_step(run) is None


def advance_plan(run: WorkflowRun, step: PlanStep) -> None:
    """Record a step as done.

    Idempotent: a retried turn must not make the plan look further along than
    it is.

    :raises PlanRejected: the step is not in this run's plan.
    """
    if step.value not in (run.plan or []):
        raise PlanRejected(
            f"Step {step.value!r} is not in this run's plan ({run.plan})."
        )
    completed = list(run.completed_steps or [])
    if step.value not in completed:
        completed.append(step.value)
        run.completed_steps = completed


def append_step(run: WorkflowRun, step: PlanStep) -> None:
    """Add a step to a live plan, in canonical position.

    This is the complementary class's consequence — "also, here's my old ECG"
    during a booking. A deterministic append, not an LLM re-plan: no model is
    asked, so it does not consume the re-plan budget, and the booking survives
    rather than being superseded.
    """
    existing = {_as_step(value) for value in run.plan or []}
    if step in existing:
        return
    run.plan = [item.value for item in _order(existing | {step})]


def record_replan(run: WorkflowRun, *, max_replans: int) -> None:
    """Consume one re-plan from the run's budget.

    A loop of *successful* calls never trips a retry ladder — a Coordinator
    that re-plans, gets an empty result, and re-plans again is making perfectly
    valid calls forever. Hence an explicit cap.

    :raises BudgetExceeded: the run has already used its re-plans. The counter
        is left at the cap rather than incremented past it, so the audit trail
        records an exhausted budget instead of an ever-growing number.
    """
    if run.replan_count >= max_replans:
        raise BudgetExceeded(
            f"Re-plan budget exhausted for run {run.id}: {run.replan_count} of "
            f"{max_replans} used."
        )
    run.replan_count += 1
