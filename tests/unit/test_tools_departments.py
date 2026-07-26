"""Department resolution and validation.

Two distinct jobs live here, and the split is the architecture in miniature:

* :func:`resolve_department` reads the patient's words and *proposes* — it may
  return nothing, or several candidates.
* :func:`validate_department` takes a name the model produced and checks it
  against the ``Department`` table. A department the model invented does not
  become real by being spelled confidently.

Ambiguity is a first-class outcome, not a failure: it is what feeds the
low-confidence staff-review path.
"""

from __future__ import annotations

import pytest

from app.tools.departments import (
    list_departments,
    resolve_department,
    validate_department,
)


class TestExactAndSynonymMatches:
    def test_department_name_resolves(self, seeded_db):
        result = resolve_department(seeded_db, "I need a Cardiology appointment")
        assert result["status"] == "resolved"
        assert result["department"]["name"] == "Cardiology"

    def test_a_synonym_resolves(self, seeded_db):
        result = resolve_department(seeded_db, "something about my heart")
        assert result["status"] == "resolved"
        assert result["department"]["name"] == "Cardiology"

    def test_matching_is_case_insensitive(self, seeded_db):
        assert resolve_department(seeded_db, "CARDIOLOGY")["status"] == "resolved"

    def test_a_multi_word_synonym_resolves(self, seeded_db):
        result = resolve_department(seeded_db, "I want a general medicine appointment")
        assert result["department"]["name"] == "General Medicine"

    def test_the_matched_terms_are_reported(self, seeded_db):
        """The staff reviewer needs to see *why* the router landed where it did."""
        result = resolve_department(seeded_db, "my heart has been bothering me")
        assert "heart" in result["matched_terms"]


class TestWordBoundaries:
    def test_a_synonym_inside_a_longer_word_does_not_match(self, seeded_db):
        """"ear" sits inside "year", "research", "clearly". Substring matching
        would route a request about next year to ENT."""
        result = resolve_department(seeded_db, "I would like an appointment next year")
        assert result["status"] == "unsupported"

    def test_a_hyphenated_synonym_still_matches(self, seeded_db):
        result = resolve_department(seeded_db, "I need a check-up")
        assert result["department"]["name"] == "General Medicine"


class TestAmbiguity:
    def test_two_departments_matched_is_ambiguous_not_a_coin_flip(self, seeded_db):
        """The seeded ambiguous case. Picking one silently is how a patient
        ends up in the wrong queue with nobody aware a choice was made."""
        result = resolve_department(seeded_db, "my kid has ear pain")
        assert result["status"] == "ambiguous"
        names = {c["name"] for c in result["candidates"]}
        assert names == {"Pediatrics", "ENT"}

    def test_an_ambiguous_result_proposes_no_department(self, seeded_db):
        result = resolve_department(seeded_db, "my kid has ear pain")
        assert result["department"] is None

    def test_candidates_are_ordered_deterministically(self, seeded_db):
        """Golden files diff this output; set ordering would make them flap."""
        first = resolve_department(seeded_db, "my kid has ear pain")["candidates"]
        second = resolve_department(seeded_db, "my kid has ear pain")["candidates"]
        assert first == second
        assert [c["name"] for c in first] == sorted(c["name"] for c in first)

    def test_repeating_one_departments_synonyms_is_not_ambiguous(self, seeded_db):
        """Several hits on the *same* department is confidence, not conflict."""
        result = resolve_department(seeded_db, "my heart — a cardiac ecg follow-up")
        assert result["status"] == "resolved"
        assert result["department"]["name"] == "Cardiology"


class TestUnsupported:
    def test_an_unrelated_request_is_unsupported(self, seeded_db):
        result = resolve_department(seeded_db, "can you tell me the weather")
        assert result["status"] == "unsupported"
        assert result["department"] is None
        assert result["candidates"] == []

    def test_empty_text_is_unsupported(self, seeded_db):
        assert resolve_department(seeded_db, "")["status"] == "unsupported"

    def test_an_inactive_department_is_never_resolved(self, seeded_db):
        """Deactivating a department must take it out of routing immediately."""
        from app.models import Department

        cardiology = seeded_db.query(Department).filter_by(name="Cardiology").one()
        cardiology.active = False
        seeded_db.flush()

        assert resolve_department(seeded_db, "heart")["status"] == "unsupported"


class TestValidation:
    def test_a_real_department_name_validates(self, seeded_db):
        result = validate_department(seeded_db, "Cardiology")
        assert result["valid"] is True
        assert result["department"]["id"] == 1

    def test_validation_is_case_insensitive(self, seeded_db):
        assert validate_department(seeded_db, "cardiology")["valid"] is True

    def test_an_invented_department_is_rejected(self, seeded_db):
        """The model proposes; the table disposes. "Cardiovascular Medicine"
        is plausible, well-formed, and not a department this hospital has."""
        result = validate_department(seeded_db, "Cardiovascular Medicine")
        assert result["valid"] is False
        assert result["department"] is None

    def test_an_inactive_department_does_not_validate(self, seeded_db):
        from app.models import Department

        seeded_db.query(Department).filter_by(name="ENT").one().active = False
        seeded_db.flush()
        assert validate_department(seeded_db, "ENT")["valid"] is False

    def test_validation_rejects_empty_input(self, seeded_db):
        assert validate_department(seeded_db, "")["valid"] is False


class TestListing:
    def test_all_ten_active_departments_are_listed(self, seeded_db):
        departments = list_departments(seeded_db)
        assert len(departments) == 10
        assert [d["name"] for d in departments] == sorted(d["name"] for d in departments)

    def test_listing_excludes_inactive_departments(self, seeded_db):
        from app.models import Department

        seeded_db.query(Department).filter_by(name="ENT").one().active = False
        seeded_db.flush()
        assert len(list_departments(seeded_db)) == 9

    def test_results_are_json_serialisable(self, seeded_db):
        import json

        json.dumps(resolve_department(seeded_db, "heart"))
        json.dumps(validate_department(seeded_db, "nope"))
        json.dumps(list_departments(seeded_db))

    def test_every_resolve_result_carries_the_same_keys(self, seeded_db):
        resolved = resolve_department(seeded_db, "heart")
        ambiguous = resolve_department(seeded_db, "my kid has ear pain")
        unsupported = resolve_department(seeded_db, "the weather")
        assert set(resolved) == set(ambiguous) == set(unsupported)
