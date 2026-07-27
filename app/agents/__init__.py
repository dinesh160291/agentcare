"""The five agents, and the machinery that binds them to a turn.

Hub and spoke: the Coordinator decides what a request needs and says so as a
validated plan; the orchestrator dispatches each planned step to the specialist
that owns it. Specialists never talk to each other. What crosses between them
crosses through persisted rows — the resolved department feeds the slot search
and the required-documents rules; the appointment id and the missing-documents
list jointly feed follow-up.

**Why the orchestrator dispatches instead of ADK's ``transfer_to_agent``.** A
transfer hands the sub-agent the entire session history, and the context
contract says specialists get no history at all — only a typed task and the
state they need. Where language *is* the job (Routing), the request text rides
inside that task. So delegation is a plan step the orchestrator executes, which
also means one patient message costs full context once rather than once per
agent, and there is no path by which a specialist can bounce work onward.
"""

from app.agents import appointment, coordinator, documents, followup, routing
from app.agents.callbacks import TurnCallbacks
from app.agents.toolbelt import Toolbelt, TurnProposals

#: Which specialist owns which plan step. The orchestrator reads this rather
#: than branching, so adding a step is one entry rather than one more `elif`.
SPECIALIST_FOR_STEP = {
    "route": routing,
    "book": appointment,
    # All three appointment verbs are the Appointment specialist's work — what
    # differs is the tool it reaches for, not who owns the step.
    "reschedule": appointment,
    "cancel": appointment,
    "documents": documents,
    "follow_up": followup,
}

__all__ = [
    "SPECIALIST_FOR_STEP",
    "Toolbelt",
    "TurnCallbacks",
    "TurnProposals",
    "appointment",
    "coordinator",
    "documents",
    "followup",
    "routing",
]
