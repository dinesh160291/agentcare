"""The deterministic safety screen — first, always, and before any model call.

This is the first-layer guard. It is deliberately dumb: a list of compiled
patterns, no model, no network, no state. That is the point. It has to work
when the LLM is down, when the LLM is confused, and when the LLM has been
talked out of its instructions — "ignore previous instructions" cannot argue
with a regex.

**Two failure directions, and they cost differently.** A missed emergency is
the worst outcome the system has. But a screen that fires on ordinary
administrative language is not the safe side of that trade — it routes the
document flow and the routing demo into a human review queue, and a queue that
is mostly noise is a queue nobody reads. So the patterns are specific phrases
with word boundaries rather than bare symptom words, and the subtler phrasings
are left to the LLM pass that runs after this one.

Three traps are encoded here rather than discovered later:

* ``pain`` alone is not an emergency. The seed's own ambiguous-routing case is
  "my kid has ear pain"; a screen keyed on the word would escalate the demo.
* ``prescription`` is a **document type** — Ophthalmology requires a prior eye
  prescription. Only a *request* for one is clinical.
* ``emergency contact`` is a field on the patient's profile, so ``emergency``
  is matched only when the next word is not ``contact``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.enums import StrEnum


class SafetyCategory(StrEnum):
    """What kind of unsafe, which decides the reply and the escalation reason.

    Both categories reach the same state — ``escalated`` — but a patient told
    to seek urgent care and a patient told the service is administrative are
    being told different things, and staff triaging the queue need the
    difference too.
    """

    EMERGENCY = "emergency"
    CLINICAL_ADVICE = "clinical_advice"


@dataclass(frozen=True)
class SafetyVerdict:
    """One screen's answer. Recorded whether it fired or passed.

    ``rule`` names what matched. A guard that logs only its firings is half an
    instrument: the expensive question is always "why did it *not* fire?", and
    that one is answerable only from the passes.
    """

    fired: bool
    category: SafetyCategory | None = None
    rule: str = ""
    source: str = "keyword"
    detail: str = ""

    def as_trace_detail(self) -> dict[str, object]:
        return {
            "category": self.category.value if self.category else None,
            "rule": self.rule,
            "source": self.source,
            "detail": self.detail,
        }


#: A verdict that fired on nothing. Reused so "passed" is one object, not many.
PASSED = SafetyVerdict(fired=False)


def _rules(pairs: tuple[tuple[str, str], ...]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    return tuple((name, re.compile(pattern, re.IGNORECASE)) for name, pattern in pairs)


#: Emergencies. Phrases, not symptom words — see the module docstring.
EMERGENCY_RULES = _rules(
    (
        ("chest", r"\bchest\b(?:\s+\w+){0,2}\s*\b(pain|pains|tight|tightness|pressure|hurts)\b"),
        ("chest_reverse", r"\b(pain|pressure|tightness)\b(?:\s+\w+){0,3}\s*\bchest\b"),
        ("breathing", r"\b(can'?t|cannot|can not|unable to|struggling to|trouble|difficulty|hard to)\b(?:\s+\w+){0,2}\s*\bbreath(e|ing)\b"),
        ("breathless", r"\b(short(ness)? of breath|gasping for (air|breath))\b"),
        ("heart_attack", r"\bheart attack\b"),
        ("cardiac_arrest", r"\bcardiac arrest\b"),
        ("stroke", r"\b(having a stroke|had a stroke|signs of a stroke)\b"),
        ("unconscious", r"\b(unconscious|unresponsive|passed out|blacked out|collapsed)\b"),
        ("bleeding", r"\b(bleeding\s+(heavily|badly|a lot)|severe bleeding|heavy bleeding|won'?t stop bleeding|can'?t stop the bleeding|bleeding\s+(wo|does)n'?t stop|bleeding will not stop)\b"),
        ("self_harm", r"\b(suicidal|kill myself|end my life|take my own life|harm myself|hurt myself)\b"),
        ("overdose", r"\bover ?dos(e|ed|ing)\b"),
        ("seizure", r"\b(seizure|seizures|convulsion|convulsions|fitting)\b"),
        ("choking", r"\bchok(ing|ed)\b"),
        ("anaphylaxis", r"\b(anaphylaxis|anaphylactic|severe allergic reaction)\b"),
        # "emergency contact" is a profile field, so the word alone is not a
        # trigger. Everything else about an emergency is.
        ("emergency", r"\bemergency\b(?!\s+contact)"),
        ("ambulance", r"\b(ambulance|999|911)\b"),
        ("urgent_care", r"\b(a and e|a&e|casualty|emergency room|\ber\b)\b"),
    )
)

#: Requests for clinical judgement: diagnosis, prescription, dosage, treatment.
CLINICAL_RULES = _rules(
    (
        ("diagnose", r"\bdiagnos(e|es|ing)\b"),
        ("diagnosis_request", r"\b(what'?s|what is|whats)\s+(my\s+)?diagnosis\b"),
        ("whats_wrong", r"\bwhat'?s wrong with me\b|\bwhat is wrong with me\b"),
        ("do_i_have_condition", r"\bdo you think i (have|might have)\b"),
        ("is_it_serious", r"\bis (it|this|that) serious\b"),
        # Only a *request* for a prescription. The noun on its own is a
        # document type this system files.
        ("prescribe", r"\bprescrib(e|es|ing)\b"),
        ("want_prescription", r"\b(get|need|want|give me)\b(?:\s+\w+){0,2}\s*\ba prescription\b"),
        ("write_prescription", r"\bwrite me a prescription\b"),
        ("dosage", r"\bdosages?\b|\bwhat dose\b|\bwhich dose\b"),
        ("how_much_to_take", r"\bhow (much|many)\b(?:\s+\w+){0,4}\s*\bshould i take\b"),
        ("should_i_take", r"\bshould i (take|stop taking|keep taking|be taking)\b"),
        ("which_medication", r"\b(what|which)\b(?:\s+\w+){0,2}\s*\b(medicine|medication|medications|drug|drugs|tablet|tablets|pill|pills)\b(?:\s+\w+){0,2}\s*\b(should|do|can)\b"),
        ("treatment_advice", r"\b(what treatment|how (do|should) i treat|how is (it|this) treated)\b"),
    )
)


def keyword_screen(message: str | None) -> SafetyVerdict:
    """Screen one message. Never raises, never calls out, never changes state.

    Emergencies are checked before clinical requests: a message can be both
    ("I have chest pain, what should I take?") and the emergency reply is the
    one that has to reach the patient.
    """
    text = (message or "").strip()
    if not text:
        return PASSED

    for category, rules in (
        (SafetyCategory.EMERGENCY, EMERGENCY_RULES),
        (SafetyCategory.CLINICAL_ADVICE, CLINICAL_RULES),
    ):
        for name, pattern in rules:
            match = pattern.search(text)
            if match:
                return SafetyVerdict(
                    fired=True,
                    category=category,
                    rule=name,
                    source="keyword",
                    detail=match.group(0),
                )
    return PASSED


__all__ = [
    "CLINICAL_RULES",
    "EMERGENCY_RULES",
    "PASSED",
    "SafetyCategory",
    "SafetyVerdict",
    "keyword_screen",
]
