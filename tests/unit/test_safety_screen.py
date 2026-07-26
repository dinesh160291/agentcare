"""The deterministic safety screen — written before the screen existed.

This is first-layer, deterministic-bin code, so the test transcribes the pinned
behaviour rather than describing whatever the implementation turned out to do.

Two halves, and the second is the one that decides whether the screen is usable
at all. Catching "chest pain" is easy; the hard part is *not* catching the
administrative sentences that happen to sit near clinical vocabulary. A screen
that fires on "my kid has ear pain" sends the seed's own ambiguous-routing demo
to the escalation queue, and a screen that fires on "I want to upload my
prescription" makes the document flow unreachable.

The false-positive cases below are therefore not padding. Each one is a
sentence this system actually produces or receives.
"""

from __future__ import annotations

import pytest

from app.safety import SafetyCategory, keyword_screen


class TestEmergenciesFire:
    @pytest.mark.parametrize(
        "message",
        [
            "I have chest pain and it won't go away",
            "my chest feels tight",
            "I can't breathe properly",
            "cannot breathe since this morning",
            "having difficulty breathing",
            "I think my father is having a heart attack",
            "my mum is having a stroke",
            "he collapsed and is unconscious",
            "she passed out in the kitchen",
            "the bleeding won't stop",
            "I am bleeding heavily",
            "I feel suicidal",
            "I want to kill myself",
            "I think I took an overdose",
            "my son had a seizure",
            "he is choking",
            "severe allergic reaction to something she ate",
            "should I call an ambulance?",
        ],
    )
    def test_emergency_language_is_caught(self, message):
        verdict = keyword_screen(message)

        assert verdict.fired
        assert verdict.category is SafetyCategory.EMERGENCY
        assert verdict.rule, "a fired verdict must name the rule that matched"


class TestClinicalRequestsFire:
    @pytest.mark.parametrize(
        "message",
        [
            "can you diagnose me?",
            "what's wrong with me?",
            "what is wrong with me",
            "do you think I have something serious",
            "can you prescribe me something",
            "I need a prescription",
            "write me a prescription please",
            "what dose should I be on",
            "what is the dosage",
            "how much paracetamol should I take",
            "should I take my tablets before the appointment",
            "which medication should I be on",
            "how do I treat this",
        ],
    )
    def test_clinical_advice_seeking_is_caught(self, message):
        verdict = keyword_screen(message)

        assert verdict.fired
        assert verdict.category is SafetyCategory.CLINICAL_ADVICE


class TestAdministrativeMessagesPass:
    """The half that decides whether the screen can ship.

    Every sentence here is one the system genuinely handles. A screen that
    fires on any of them has not become safer — it has become unusable, and
    the queue it floods is a human's.
    """

    @pytest.mark.parametrize(
        "message",
        [
            "I need a cardiology appointment next week",
            # The seed's deliberate ambiguous-routing case. "pain" alone must
            # never be enough, or the low-confidence demo escalates instead.
            "book an appointment, my kid has ear pain",
            "my child has ear pain, which department is that?",
            # "prescription" is a document type — Ophthalmology requires one.
            "I want to upload my prescription",
            "here is my prior eye prescription",
            "attaching my blood report and ECG",
            # "emergency contact" is a field on the patient's own profile.
            "please update my emergency contact",
            "my emergency contact is my sister",
            # "do I have" is a side-question opener, not a diagnosis request.
            "do I have any documents on file?",
            "what documents do I still need?",
            "cancel my cardiology appointment",
            "can I move my appointment to Tuesday?",
            "yes",
            "no",
            "what's the weather like today?",
            "when is my next appointment?",
            "I'd like to see a heart doctor about a follow-up",
        ],
    )
    def test_administrative_language_passes(self, message):
        verdict = keyword_screen(message)

        assert not verdict.fired, f"{message!r} fired on rule {verdict.rule!r}"
        assert verdict.category is None


class TestTheScreenIsTotal:
    """It runs on every message, so every message must have an answer."""

    @pytest.mark.parametrize("message", ["", "   ", None])
    def test_empty_input_passes_rather_than_raising(self, message):
        verdict = keyword_screen(message)

        assert not verdict.fired

    def test_the_verdict_names_its_source(self):
        assert keyword_screen("I have chest pain").source == "keyword"
        assert keyword_screen("book an appointment").source == "keyword"

    def test_matching_ignores_case_and_punctuation(self):
        assert keyword_screen("CHEST PAIN!!!").fired
        assert keyword_screen("Chest  pain").fired
