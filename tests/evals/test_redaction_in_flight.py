"""Redaction, proven on traffic rather than on the redactor.

``tests/unit/test_trace.py`` proves the function masks what it is handed. This
module proves the *system* hands it everything: a patient who types their email
and phone number into the chat must not leave those strings in a trace row, and
the profile fields the model is given as context must not either.

The posture, stated here because a test is where an intention stops being one.
Raw chat history is stored raw **by design** — it is the model's working memory
and cannot be redacted without lobotomising it — and is protected by
ownership-checked endpoints instead. The redaction guarantee covers what a
*wider* audience reads: trace payloads, audit metadata, and log lines. Those are
the three choke points, and each has its own test below.

The audit that produced this module also found a defect worth keeping named: a
loose phone pattern matches a bare ISO date, so every search window in the trace
read ``start: "[redacted]"`` — the first thing a reviewer looks at when asking
which week the system searched. Dates now survive and a date of birth is caught
by its key instead.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from app.db import SessionLocal
from app.models import AuditEvent, Reminder, TraceEvent, User, WorkflowRun
from app.orchestrator import run_workflow
from app.scheduler.delivery import deliver
from app.trace import TraceWriter, redact
from app.models.enums import TraceAuthor

PATIENT_EMAIL = "asha.patient@example.invalid"
CONTACT_EMAIL = "asha.reachme@example.invalid"
CONTACT_PHONE = "+1-555-0100"
WITH_CONTACT = (
    f"please book me a cardiology appointment next week, my email is "
    f"{CONTACT_EMAIL} and my mobile is {CONTACT_PHONE}"
)


@pytest.fixture
def told_contact_details(seeded_db):
    """One ordinary booking turn, with contact details typed into the message."""
    seeded_db.commit()
    patient = seeded_db.query(User).filter(User.email == PATIENT_EMAIL).one()
    return asyncio.run(run_workflow(patient, WITH_CONTACT, "redaction-1"))


def _payloads(session) -> str:
    return "\n".join(
        json.dumps(event.payload or {}) for event in session.query(TraceEvent).all()
    )


class TestTheTracePayloads:
    def test_contact_details_do_not_survive_the_turn(self, told_contact_details):
        """The whole guarantee, in one assertion, over every row of a real turn.

        A turn writes the patient's words into the inbound event, into every
        LLM request that carries the transcript, into ``resolve_department``'s
        arguments and back out in its result — five separate paths, and it only
        takes one of them forgetting.
        """
        session = SessionLocal()
        try:
            blob = _payloads(session)
            assert CONTACT_EMAIL not in blob
            assert "555-0100" not in blob
            assert "[redacted]" in blob, "nothing was redacted at all"
        finally:
            session.close()

    def test_profile_fields_are_masked_by_their_key(self, told_contact_details):
        """``load_patient_context`` hands the model the patient's record.

        Its date of birth and phone are redacted because of what the *keys* are,
        not because of what the values happen to look like — which is what makes
        the guarantee hold for a DOB the value pattern would not recognise.
        """
        session = SessionLocal()
        try:
            context = [
                event
                for event in session.query(TraceEvent).all()
                if (event.payload or {}).get("tool") == "load_patient_context"
                and "result" in (event.payload or {})
            ]
            assert context, "the corpus no longer loads patient context"
            result = context[0].payload["result"]
            assert result["date_of_birth"] == "[redacted]"
            assert result["phone"] == "[redacted]"
            # The bookkeeping beside it stays readable, or the trace explains
            # nothing: an id is not an identity.
            assert isinstance(result["patient_id"], int)
        finally:
            session.close()

    def test_the_searched_window_is_still_legible(self, told_contact_details):
        """The defect this module was written to catch.

        ``start`` and ``end`` are ISO dates, and a phone pattern loose enough to
        catch ``555.010.0100`` matches ten characters of digits and hyphens too.
        The window came out ``[redacted]`` on both ends — a trace that cannot
        answer "which week did it search?", which is the first question asked of
        every availability defect this project has had.
        """
        session = SessionLocal()
        try:
            searches = [
                (event.payload or {}).get("args", {})
                for event in session.query(TraceEvent).all()
                if (event.payload or {}).get("tool") == "find_available_slots"
                and "args" in (event.payload or {})
            ]
            assert searches, "no slot search in the turn"
            windows = [
                args for args in searches if args.get("start") or args.get("end")
            ]
            assert windows, "the search recorded no window at all"
            for args in windows:
                for edge in ("start", "end"):
                    if args.get(edge):
                        assert args[edge] != "[redacted]", (
                            f"the {edge} of the searched window is unreadable"
                        )
        finally:
            session.close()

    def test_history_is_raw_by_design(self, told_contact_details):
        """Stated as a test so it reads as a decision rather than an oversight.

        The run's request text is the patient's own words, kept intact because
        the model works from them. It is protected by ownership, not by masking
        — and pinning it here means a future change to that posture has to be
        made on purpose.
        """
        session = SessionLocal()
        try:
            run = session.get(WorkflowRun, told_contact_details.run_id)
            assert CONTACT_EMAIL in (run.request_text or "")
        finally:
            session.close()


class TestTheOtherTwoChokePoints:
    def test_audit_metadata_is_redacted_at_the_writer(self, db):
        """Most audit metadata is enums and ids — and then there is the
        escalation reason, the staff note, and a refusal's problem string."""
        from app.audit import write_audit

        event = write_audit(
            db,
            action="escalation_opened",
            entity_type="Escalation",
            metadata={"reason": f"patient asked us to call {CONTACT_PHONE}"},
        )
        db.flush()
        assert "555-0100" not in json.dumps(event.event_metadata)

    def test_the_simulated_channel_does_not_log_contact_details(
        self, seeded_db, caplog
    ):
        """The log stands in for SMS. It is read by operators and shipped
        wherever logs go, so it is redacted like a trace payload — while the
        notification row the patient reads keeps the message intact."""
        reminder = seeded_db.query(Reminder).first()
        assert reminder is not None
        reminder.message = f"Reminder: call the clinic on {CONTACT_PHONE}"
        seeded_db.flush()

        with caplog.at_level(logging.INFO, logger="agentcare.reminders"):
            notification = deliver(seeded_db, reminder)

        assert "555-0100" not in caplog.text
        assert "SIMULATED NOTIFICATION" in caplog.text
        assert notification is not None and "555-0100" in notification.body


class TestTheAlarmCanFire:
    """"An alarm you've never tripped is a decoration." Each rule below is fed
    a value that breaks exactly that rule."""

    def test_an_unmasked_key_would_be_caught(self):
        assert redact({"date_of_birth": "1985-03-12"}) == {"date_of_birth": "[redacted]"}

    def test_a_date_is_not_a_phone_number(self):
        assert redact({"start": "2026-08-03"}) == {"start": "2026-08-03"}

    def test_a_phone_number_is_still_a_phone_number(self):
        assert redact({"note": "ring 555.010.0100"}) == {"note": "ring [redacted]"}

    def test_the_writer_redacts_before_the_row_exists(self, db):
        """Not at the reader, and not on the way out: at the pen."""
        writer = TraceWriter(db, session_id="s-redact")
        writer.inbound(
            f"my number is {CONTACT_PHONE}", author=TraceAuthor.PATIENT_MESSAGE
        )
        stored = db.query(TraceEvent).one()
        assert "555-0100" not in json.dumps(stored.payload)
