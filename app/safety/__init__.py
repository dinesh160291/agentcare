"""The safety guardrail layer.

Safety is **not** a sixth agent. The five agents are a hub-and-spoke topology
with a plan and a delegation order; this is a guard that runs across all of
them, on every message, before any of them get a turn. It has no plan, nothing
delegates to it, and nothing delegates from it.

Two layers, in this order and never the other one:

1. :func:`keyword_screen` — deterministic, instant, and immune to anything the
   model does. It runs first and always.
2. :func:`llm_screen` — a second opinion on the subtler phrasings a phrase list
   cannot hold, taken through the provider seam like every other model call.

Both write a guard verdict whether they fire or pass, and both end in the same
place when they fire: :func:`escalate`, which puts the run in front of a human
and keeps it there.
"""

from app.safety.classifier import llm_screen
from app.safety.escalate import (
    CLINICAL_REPLY,
    EMERGENCY_REPLY,
    escalate,
    reply_for,
)
from app.safety.screen import (
    PASSED,
    SafetyCategory,
    SafetyVerdict,
    keyword_screen,
)

__all__ = [
    "CLINICAL_REPLY",
    "EMERGENCY_REPLY",
    "PASSED",
    "SafetyCategory",
    "SafetyVerdict",
    "escalate",
    "keyword_screen",
    "llm_screen",
    "reply_for",
]
