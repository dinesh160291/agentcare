"""Domain exceptions.

Raised by tools and services, which stay framework-agnostic; the API layer maps
them to HTTP status codes. Tools importing FastAPI would defeat the point of
keeping them plain functions.

``RecordNotFound`` carries the weight here. A failed **ownership** check raises
it too — not a permission error — because a 403 confirms the record exists and
turns the id field into an enumeration oracle. Patient A probing patient B's
appointment id must be indistinguishable from probing an id that was never
issued.
"""

from __future__ import annotations


class AgentCareError(Exception):
    """Base class for every domain error."""


class RecordNotFound(AgentCareError):
    """No such row — or one the caller does not own. Maps to HTTP 404."""

    def __init__(self, entity: str, entity_id: object | None = None) -> None:
        self.entity = entity
        self.entity_id = entity_id
        super().__init__(f"{entity} not found")


class PermissionDenied(AgentCareError):
    """Wrong role for this operation. Maps to HTTP 403.

    Used for *role* failures only, where the endpoint's existence is not a
    secret (a patient calling a staff-only endpoint). Ownership failures use
    :class:`RecordNotFound`.
    """

    def __init__(self, message: str = "You do not have access to this operation") -> None:
        super().__init__(message)


class AuthenticationError(AgentCareError):
    """Bad or missing credentials. Maps to HTTP 401."""


class ValidationFailed(AgentCareError):
    """Input a human supplied is not usable. Maps to HTTP 422."""


class InvalidTransition(AgentCareError):
    """An attempt to move a workflow along an edge the machine does not have."""

    def __init__(self, from_status: object, to_status: object) -> None:
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(f"Illegal workflow transition: {from_status} -> {to_status}")


class PlanRejected(AgentCareError):
    """The model's proposed plan did not survive validation.

    Deliberately distinct from :class:`ValidationFailed`, which is for input a
    *human* supplied and maps to HTTP 422. A malformed plan is model output:
    it belongs to the retry ladder, and it is never executed.
    """


class ClassRejected(AgentCareError):
    """The model's proposed message→run class did not survive validation.

    Kept apart from :class:`PlanRejected` because the two lead somewhere
    different: a rejected plan is re-planned and consumes the run's re-plan
    budget, while a rejected class is re-classified and does not.
    """


class BudgetExceeded(AgentCareError):
    """A bounded loop hit its cap (tool iterations, re-plans, retries)."""


class ProviderError(AgentCareError):
    """The LLM provider failed after the retry ladder was exhausted."""


class RateLimited(ProviderError):
    """Provider throttling — a distinct failure class.

    Honours ``retry_after`` and does **not** consume a bounded-retry slot:
    retry budgets exist for broken calls, not throttled ones.
    """

    def __init__(self, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__("Provider rate limit reached")
