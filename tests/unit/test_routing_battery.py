"""The phrasings a patient actually types, pinned one row at a time.

``test_tools_departments.py`` tests the *matcher* — word boundaries, ambiguity,
the invented-name refusal. This file tests the **vocabulary**, which is data,
and it exists because the two fail in completely different ways. A broken
matcher is loud. A missing synonym is silent: "back pain" simply resolves to
nothing, the request takes the clarify-or-review path, and everything downstream
behaves impeccably while the patient is asked a question the table could have
answered. A 25-phrase probe of ordinary phrasings scored 15 that way — zero
misroutes, and two thirds of the traffic on the slow path.

So every common phrasing gets a row, and the row is the point: a future synonym
edit that orphans "headache" fails *here*, by name, instead of quietly moving
one more conversation into the staff queue.

Three of the rows are pinned to something other than a department, and those
are the ones worth reading:

* **"ear pain for my kid" is ambiguous, and must stay so.** Two desks are
  defensible, and picking one silently is how a patient lands in the wrong
  queue with nobody aware a choice was made.
* **"blood pressure" is ambiguous by construction.** ``uq_department_synonym_term``
  is global, so a term cannot be filed under two departments; the pair that
  makes this ask rather than guess is Cardiology's "blood pressure" and General
  Medicine's bare "pressure", which is the same mechanism as kid-ear-pain.
* **"tooth pain" is unresolved, and that is correct.** This hospital has no
  dental department. The negative control matters more than any positive one:
  a vocabulary that resolves everything has stopped distinguishing between what
  it handles and what it does not, and the patient would be booked into a desk
  that cannot see them.
"""

from __future__ import annotations

import pytest

from app.tools.departments import resolve_department

#: ``(phrase, expected)``. A string expects ``resolved`` to that department, a
#: set expects ``ambiguous`` with exactly those candidates, and ``None``
#: expects ``unsupported``.
BATTERY: list[tuple[str, object]] = [
    # --- the ten the probe found unresolved --------------------------------
    ("I have back pain", "Orthopedics"),
    ("I keep getting a headache", "Neurology"),
    ("I have had a fever for two days", "General Medicine"),
    ("appointment about diarrhea", "Gastroenterology"),
    ("I want to see someone about hair loss", "Dermatology"),
    ("I think I fractured my arm", "Orthopedics"),
    ("my eyes are red", "Ophthalmology"),
    ("I am pregnant and need a check", "Gynecology & Obstetrics"),
    ("book an appointment for my son", "Pediatrics"),
    ("I need something for my sprained ankle", "Orthopedics"),
    # --- the ones that already worked, kept so a regression is visible -----
    ("I need a Cardiology appointment", "Cardiology"),
    ("something to do with my heart", "Cardiology"),
    ("my knee has been hurting", "Orthopedics"),
    ("I have a skin rash", "Dermatology"),
    ("I would like a general check-up", "General Medicine"),
    ("my child needs to be seen", "Pediatrics"),
    ("I get migraines", "Neurology"),
    ("my throat is sore", "ENT"),
    ("my vision has gone blurry", "Ophthalmology"),
    ("I need a prenatal appointment", "Gynecology & Obstetrics"),
    # --- the rest of the new vocabulary ------------------------------------
    ("stomach acidity and indigestion", "Gastroenterology"),
    ("I need new glasses", "Ophthalmology"),
    ("vaccination for my toddler", "Pediatrics"),
    # --- pinned to something other than a department -----------------------
    ("ear pain for my kid", {"ENT", "Pediatrics"}),
    ("I have tooth pain", None),
]

#: The plural forms are rows in the seed rather than a rule in the matcher —
#: matching is on word boundaries, so "ears" is a different string from "ear"
#: and no amount of care in the matcher would have found it. These probe the
#: seam that decision creates.
PLURALS: list[tuple[str, str]] = [
    ("my eyes are watery", "Ophthalmology"),
    ("both my ears hurt", "ENT"),
    ("pain in my knees", "Orthopedics"),
    ("my shoulders ache", "Orthopedics"),
    ("I get bad headaches", "Neurology"),
    ("appointment about my periods", "Gynecology & Obstetrics"),
    ("my kids need to be seen", "Pediatrics"),
]


def _describe(result: dict) -> str:
    """What the resolver actually said, for a failure message worth reading."""
    if result["status"] == "resolved":
        return f"resolved -> {result['department']['name']}"
    if result["status"] == "ambiguous":
        return f"ambiguous -> {sorted(c['name'] for c in result['candidates'])}"
    return "unsupported"


@pytest.mark.parametrize(("phrase", "expected"), BATTERY, ids=[p for p, _ in BATTERY])
def test_the_phrase_battery(phrase, expected, seeded_db):
    result = resolve_department(seeded_db, phrase)
    actual = _describe(result)

    if expected is None:
        assert result["status"] == "unsupported", f"{phrase!r}: {actual}"
    elif isinstance(expected, set):
        assert result["status"] == "ambiguous", f"{phrase!r}: {actual}"
        assert {c["name"] for c in result["candidates"]} == expected, (
            f"{phrase!r}: {actual}"
        )
    else:
        assert result["status"] == "resolved", f"{phrase!r}: {actual}"
        assert result["department"]["name"] == expected, f"{phrase!r}: {actual}"


@pytest.mark.parametrize(("phrase", "expected"), PLURALS, ids=[p for p, _ in PLURALS])
def test_plural_phrasings_resolve(phrase, expected, seeded_db):
    result = resolve_department(seeded_db, phrase)
    assert result["status"] == "resolved", f"{phrase!r}: {_describe(result)}"
    assert result["department"]["name"] == expected, f"{phrase!r}: {_describe(result)}"


class TestTheDeliberateAmbiguities:
    """Two phrases where *asking* is the right answer, and one where it is not.

    Both directions are pinned. An ambiguity that quietly became a resolution
    is a silent misroute, and a resolution that quietly became an ambiguity is
    a question the patient did not need to be asked — and neither shows up
    anywhere else, because the conversation stays polite either way.
    """

    def test_blood_pressure_asks_rather_than_guessing(self, seeded_db):
        result = resolve_department(seeded_db, "I need my blood pressure checked")
        assert result["status"] == "ambiguous", _describe(result)
        assert {c["name"] for c in result["candidates"]} == {
            "Cardiology",
            "General Medicine",
        }

    def test_heartburn_is_not_a_cardiology_request(self, seeded_db):
        """The word contains "heart" and the boundary rule is what saves it.

        ``\\bheart\\b`` does not match inside "heartburn", so the only term that
        fires is Gastroenterology's own. The two-word spelling "heart burn"
        still reaches Cardiology — defusing *that* would need a bare "burn"
        under Gastroenterology, which would misroute an actual burn, and a new
        misroute is a worse trade than a rarer spelling on the slow path.
        """
        result = resolve_department(seeded_db, "I get heartburn after meals")
        assert result["status"] == "resolved", _describe(result)
        assert result["department"]["name"] == "Gastroenterology"

    def test_a_word_the_hospital_has_no_desk_for_stays_unresolved(self, seeded_db):
        assert resolve_department(seeded_db, "I have tooth pain")["status"] == (
            "unsupported"
        )


class TestTheVocabularyDoesNotSwallowOrdinaryEnglish:
    """The rows that were *not* added, and the sentences that explain why.

    A synonym list is only safe while it stays a list of things a department
    handles. "Back" is also half of "push back my appointment", and a term that
    turned a reschedule into an Orthopedics request would hand the refinement
    rule a subject the patient never named — so the row is "back pain".
    """

    def test_pushing_an_appointment_back_names_no_department(self, seeded_db):
        result = resolve_department(seeded_db, "can you push back my appointment")
        assert result["status"] == "unsupported", _describe(result)

    def test_next_year_is_still_not_ENT(self, seeded_db):
        """The original boundary case, re-checked against the longer list."""
        result = resolve_department(seeded_db, "an appointment some time next year")
        assert result["status"] == "unsupported", _describe(result)

    def test_the_weather_is_still_off_topic(self, seeded_db):
        result = resolve_department(seeded_db, "what's the weather like today")
        assert result["status"] == "unsupported", _describe(result)
