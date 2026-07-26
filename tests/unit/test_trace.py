"""The trace writer, redaction, and the well-formedness checker.

The checker's own tests matter as much as the writer's: an alarm nobody has
watched fail is a decoration, so each rule is proven by feeding it a trace that
breaks exactly that rule.
"""

from __future__ import annotations

import pytest

from app.models import TraceEvent, WorkflowRun, WorkflowStatus
from app.models.enums import TraceAuthor, TraceEventType
from app.trace import TraceWriter, assert_well_formed, check_session, redact, redact_text


class TestRedaction:
    def test_sensitive_keys_are_masked_whatever_the_value(self):
        out = redact({"api_key": "gsk_live_abc123", "password": "hunter2"})
        assert out == {"api_key": "[redacted]", "password": "[redacted]"}

    def test_key_matching_is_on_substrings(self):
        out = redact({"GROQ_API_KEY": "x", "user_password_hash": "y", "jwt": "z"})
        assert set(out.values()) == {"[redacted]"}

    def test_emails_and_phones_are_masked_inside_free_text(self):
        """The same detail arrives as a field sometimes and as prose others."""
        masked = redact_text("write to asha@example.invalid or call +1-555-0100")
        assert "asha@example.invalid" not in masked
        assert "555-0100" not in masked

    def test_redaction_reaches_into_nested_structures(self):
        out = redact({"outer": [{"token": "abc"}, {"note": "call +1-555-0100"}]})
        assert out["outer"][0]["token"] == "[redacted]"
        assert "555-0100" not in out["outer"][1]["note"]

    def test_ordinary_values_survive(self):
        payload = {"department": "Cardiology", "slot_id": 7, "ok": True}
        assert redact(payload) == payload

    def test_the_callers_object_is_not_mutated(self):
        """A redactor that edits in place corrupts the row the application is
        still working with."""
        original = {"api_key": "secret", "keep": "value"}
        redact(original)
        assert original["api_key"] == "secret"


class TestWriter:
    def test_events_are_numbered_from_one_in_order(self, db):
        writer = TraceWriter(db, session_id="s1")
        writer.inbound("hello", author=TraceAuthor.PATIENT_MESSAGE)
        writer.outbound("hi", author=TraceAuthor.TEMPLATE)

        events = db.query(TraceEvent).order_by(TraceEvent.seq).all()
        assert [e.seq for e in events] == [1, 2]

    def test_payloads_are_redacted_on_the_way_in(self, db):
        """Redaction at the choke point, not at the reader."""
        writer = TraceWriter(db, session_id="s1")
        writer.inbound("hello", author=TraceAuthor.PATIENT_MESSAGE, api_key="gsk_secret")
        event = db.query(TraceEvent).one()
        assert event.payload["api_key"] == "[redacted]"

    def test_a_request_correlates_with_its_response(self, db):
        writer = TraceWriter(db, session_id="s1")
        cid = writer.llm_request(agent_name="coordinator", payload={"prompt": "x"})
        writer.llm_response(cid, agent_name="coordinator", payload={"text": "y"})

        events = db.query(TraceEvent).order_by(TraceEvent.seq).all()
        assert events[0].correlation_id == events[1].correlation_id

    def test_an_unanswered_request_is_visible_to_the_writer(self, db):
        writer = TraceWriter(db, session_id="s1")
        cid = writer.llm_request(agent_name="coordinator", payload={})
        assert writer.unpaired_requests == {cid}
        writer.llm_error(cid, agent_name="coordinator", error="timeout")
        assert writer.unpaired_requests == set()

    def test_a_turn_can_be_bound_to_a_run_created_mid_turn(self, seeded_db):
        """A run is created once intent is accepted, which is after the turn
        has already begun."""
        run = WorkflowRun(patient_id=1, status=WorkflowStatus.IN_PROGRESS)
        seeded_db.add(run)
        seeded_db.flush()

        writer = TraceWriter(seeded_db, session_id="s1")
        writer.inbound("book me in", author=TraceAuthor.PATIENT_MESSAGE)
        writer.bind_run(run.id)
        writer.outbound("done", author=TraceAuthor.TEMPLATE)

        events = seeded_db.query(TraceEvent).order_by(TraceEvent.seq).all()
        assert events[0].workflow_run_id is None
        assert events[1].workflow_run_id == run.id

    def test_guard_passes_are_recorded_not_only_failures(self, db):
        """"The screen did not fire" and "the screen never ran" are different
        facts, and only one of them is acceptable."""
        writer = TraceWriter(db, session_id="s1")
        writer.guard_verdict("safety_keywords", passed=True)
        assert db.query(TraceEvent).one().payload["passed"] is True

    def test_rejected_proposals_are_recorded(self, db):
        writer = TraceWriter(db, session_id="s1")
        writer.validation("department", accepted=False,
                          detail={"proposed": "Cardiovascular Medicine"})
        event = db.query(TraceEvent).one()
        assert event.payload["accepted"] is False
        assert event.event_type is TraceEventType.VALIDATION


def _well_formed_turn(db, writer: TraceWriter) -> None:
    writer.inbound("book cardiology", author=TraceAuthor.PATIENT_MESSAGE)
    writer.guard_verdict("safety_keywords", passed=True)
    cid = writer.llm_request(agent_name="coordinator", payload={"prompt": "x"})
    writer.llm_response(cid, agent_name="coordinator", payload={"text": "y"})
    writer.validation("plan", accepted=True)
    tool_cid = writer.tool_call("find_available_slots", args={"department_id": 1})
    writer.tool_result(tool_cid, name="find_available_slots", result={"slots": []})
    writer.transition(from_status=None, to_status="in_progress")
    writer.outbound("Here are some times.", author=TraceAuthor.LLM)


class TestChecker:
    def test_a_complete_turn_passes(self, db):
        _well_formed_turn(db, TraceWriter(db, session_id="s1"))
        assert_well_formed(db)

    def test_a_typed_action_turn_passes(self, db):
        """The most deterministic flows must parse too, not just the chattiest."""
        writer = TraceWriter(db, session_id="s1")
        writer.inbound("confirm", author=TraceAuthor.PATIENT_ACTION)
        writer.tool_result(
            writer.tool_call("book_appointment", args={"slot_id": 1}),
            name="book_appointment", result={"ok": True},
        )
        writer.transition(from_status="pending_confirmation", to_status="in_progress")
        writer.outbound("Booked.", author=TraceAuthor.TEMPLATE)
        assert_well_formed(db)

    def test_a_dangling_request_is_caught(self, db):
        writer = TraceWriter(db, session_id="s1")
        writer.inbound("hello", author=TraceAuthor.PATIENT_MESSAGE)
        writer.llm_request(agent_name="coordinator", payload={})
        writer.outbound("hi", author=TraceAuthor.TEMPLATE)

        problems = check_session(db)[0].problems
        assert any("no response or error" in p for p in problems)

    def test_an_error_counts_as_a_valid_partner(self, db):
        """A failed call is a completed record, not a gap."""
        writer = TraceWriter(db, session_id="s1")
        writer.inbound("hello", author=TraceAuthor.PATIENT_MESSAGE)
        cid = writer.llm_request(agent_name="coordinator", payload={})
        writer.llm_error(cid, agent_name="coordinator", error="rate limited", kind="rate_limit")
        writer.outbound("Please try again shortly.", author=TraceAuthor.TEMPLATE)
        assert_well_formed(db)

    def test_a_turn_with_no_outbound_is_caught(self, db):
        writer = TraceWriter(db, session_id="s1")
        writer.inbound("hello", author=TraceAuthor.PATIENT_MESSAGE)
        assert any("no outbound" in p for p in check_session(db)[0].problems)

    def test_a_turn_with_no_inbound_is_caught(self, db):
        writer = TraceWriter(db, session_id="s1")
        writer.outbound("out of nowhere", author=TraceAuthor.TEMPLATE)
        assert any("no inbound" in p for p in check_session(db)[0].problems)

    def test_an_outbound_without_an_author_is_caught(self, db):
        writer = TraceWriter(db, session_id="s1")
        writer.inbound("hello", author=TraceAuthor.PATIENT_MESSAGE)
        writer._write(TraceEventType.OUTBOUND, author=None, payload={"content": "x"})
        assert any("no author" in p for p in check_session(db)[0].problems)

    def test_an_orphan_tool_result_is_caught(self, db):
        writer = TraceWriter(db, session_id="s1")
        writer.inbound("hello", author=TraceAuthor.PATIENT_MESSAGE)
        writer._write(TraceEventType.TOOL_RESULT, payload={"tool": "x"}, correlation_id="nope")
        writer.outbound("hi", author=TraceAuthor.TEMPLATE)
        assert any("no matching tool_call" in p for p in check_session(db)[0].problems)

    def test_a_second_inbound_in_one_turn_is_caught(self, db):
        writer = TraceWriter(db, session_id="s1")
        writer.inbound("hello", author=TraceAuthor.PATIENT_MESSAGE)
        writer.inbound("again", author=TraceAuthor.PATIENT_MESSAGE)
        writer.outbound("hi", author=TraceAuthor.TEMPLATE)
        assert any("2 inbound events" in p for p in check_session(db)[0].problems)

    def test_turns_are_checked_independently(self, db):
        _well_formed_turn(db, TraceWriter(db, session_id="s1"))
        broken = TraceWriter(db, session_id="s1")
        broken.outbound("orphan", author=TraceAuthor.TEMPLATE)

        reports = check_session(db)
        assert sum(1 for r in reports if r.ok) == 1
        assert sum(1 for r in reports if not r.ok) == 1

    def test_assert_well_formed_raises_with_detail(self, db):
        writer = TraceWriter(db, session_id="s1")
        writer.inbound("hello", author=TraceAuthor.PATIENT_MESSAGE)
        with pytest.raises(AssertionError, match="malformed traces"):
            assert_well_formed(db)
