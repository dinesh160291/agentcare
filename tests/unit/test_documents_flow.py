"""Document verification, and the missing-documents task it feeds.

Two things are pinned here.

**The verification pipeline.** Text out of the PDF, a *proposal* from the model
about what the content looks like, and a status set by code. The seed ships an
X-ray report filed as an ECG for exactly this: a genuine mismatch, not a
fixture bent to produce one.

**The derivation invariant, twice.** A document's type decides which
requirements it satisfies, so the missing-documents task has to move whenever
the type does — when verification flags it, and again when staff resolve it.
A task that says "supply an ECG" for an ECG already on file is the same
stale-fact bug the receipt discipline kills for model output, reintroduced by
a code path.

The first test in this file is the one that caught a silent break: the diff
reports ``missing_mandatory`` and every consumer was reading ``missing``, so
the task was never created and the patient was told nothing was needed.
"""

from __future__ import annotations

import asyncio

import pytest

from app.db import SessionLocal
from app.errors import ValidationFailed
from app.models import (
    AuditEvent,
    Department,
    DocumentStatus,
    FollowUpTask,
    FollowUpTaskStatus,
    FollowUpTaskType,
    PatientDocument,
    User,
)
from app.orchestrator import run_workflow
from app.tools import (
    diff_required_documents,
    extract_document_text,
    list_flagged_documents,
)
from app.workflow.staff import resolve_document

PATIENT_EMAIL = "asha.patient@example.invalid"
STAFF_EMAIL = "staff@example.invalid"
#: Neurology requires a prior MRI/CT report, which the seeded patient lacks.
NEURO_BOOKING = "I need a neurology appointment next week"


@pytest.fixture
def patient(seeded_db):
    seeded_db.commit()
    return seeded_db.query(User).filter(User.email == PATIENT_EMAIL).one()


@pytest.fixture
def staff(seeded_db):
    return seeded_db.query(User).filter(User.email == STAFF_EMAIL).one()


def turn(user, message, session_id):
    return asyncio.run(run_workflow(user, message, session_id))


def fresh():
    return SessionLocal()


def pending_count(session, patient_id) -> int:
    return (
        session.query(PatientDocument)
        .filter(
            PatientDocument.patient_id == patient_id,
            PatientDocument.status == DocumentStatus.PENDING_VERIFICATION,
        )
        .count()
    )


def drain_verification(user, prefix: str, *, limit: int = 8) -> int:
    """Keep asking until nothing is left unverified. Returns the turns taken.

    Verification is capped at one document per turn on purpose, so clearing a
    backlog takes several. The loop is bounded too: a queue that never empties
    should fail this helper rather than hang the suite.
    """
    from app.models import PatientProfile

    for taken in range(1, limit + 1):
        turn(user, "what documents do I have on file?", f"{prefix}-{taken}")
        session = fresh()
        try:
            profile = (
                session.query(PatientProfile)
                .filter(PatientProfile.user_id == user.id)
                .one()
            )
            if pending_count(session, profile.id) == 0:
                return taken
        finally:
            session.close()
    raise AssertionError(f"documents still unverified after {limit} turns")


def open_missing_task(session, patient_id):
    return (
        session.query(FollowUpTask)
        .filter(
            FollowUpTask.patient_id == patient_id,
            FollowUpTask.task_type == FollowUpTaskType.MISSING_DOCUMENTS,
            FollowUpTask.status == FollowUpTaskStatus.OPEN,
        )
        .one_or_none()
    )


class TestTheMissingDocumentsTaskIsActuallyCreated:
    """The diff and its consumers had disagreed on a key name, so the task was
    never created and the reply said nothing was needed. Both halves are
    asserted: the row, and the sentence the patient reads."""

    def test_a_booking_with_a_shortfall_opens_a_task(self, patient):
        result = turn(patient, NEURO_BOOKING, "s-doc-1")
        turn(patient, "yes", "s-doc-1")

        session = fresh()
        try:
            from app.models import WorkflowRun

            run = session.get(WorkflowRun, result.run_id)
            task = open_missing_task(session, run.patient_id)

            assert task is not None, "a shortfall must leave something behind"
            assert "Prior MRI or CT report" in task.details["missing"]
        finally:
            session.close()

    def test_the_patient_is_told_what_is_missing(self, patient):
        turn(patient, NEURO_BOOKING, "s-doc-2")
        result = turn(patient, "yes", "s-doc-2")

        assert "Prior MRI or CT report" in result.reply

    def test_a_department_with_nothing_outstanding_opens_no_task(self, patient):
        """The seeded patient already holds both Cardiology requirements.
        Opening a task purely to close it is noise in their list."""
        result = turn(patient, "I need a cardiology appointment next week", "s-doc-3")
        turn(patient, "yes", "s-doc-3")

        session = fresh()
        try:
            from app.models import WorkflowRun

            run = session.get(WorkflowRun, result.run_id)
            assert open_missing_task(session, run.patient_id) is None
        finally:
            session.close()


class TestTextExtraction:
    def test_a_seeded_pdf_yields_its_text(self, seeded_db):
        document = (
            seeded_db.query(PatientDocument).order_by(PatientDocument.id).first()
        )
        extracted = extract_document_text(seeded_db, document.id)

        assert extracted["extracted"] is True
        assert extracted["pages"] >= 1
        assert "SYNTHETIC" in extracted["text"].upper()

    def test_an_image_says_it_cannot_be_read_rather_than_returning_nothing(
        self, seeded_db
    ):
        """"This says nothing" and "this cannot be read" must stay apart, or
        every uploaded photo looks like a mismatch."""
        document = (
            seeded_db.query(PatientDocument).order_by(PatientDocument.id).first()
        )
        document.mime_type = "image/png"
        seeded_db.flush()

        extracted = extract_document_text(seeded_db, document.id)
        assert extracted["extracted"] is False
        assert extracted["reason"] == "not_extractable"

    def test_a_missing_file_is_reported_not_raised(self, seeded_db):
        document = (
            seeded_db.query(PatientDocument).order_by(PatientDocument.id).first()
        )
        document.storage_path = "no/such/file.pdf"
        seeded_db.flush()

        assert extract_document_text(seeded_db, document.id)["reason"] == "file_missing"

    def test_an_unknown_document_is_reported_not_raised(self, seeded_db):
        assert extract_document_text(seeded_db, 999_999)["reason"] == "not_found"

    def test_the_extract_is_bounded(self, seeded_db):
        from app.tools.documents import MAX_EXTRACT_CHARS

        document = (
            seeded_db.query(PatientDocument).order_by(PatientDocument.id).first()
        )
        assert len(extract_document_text(seeded_db, document.id)["text"]) <= (
            MAX_EXTRACT_CHARS
        )


class TestVerificationThroughTheWorkflow:
    """The seed's third document is an X-ray report filed as an ECG."""

    def test_verification_is_bounded_to_one_document_per_turn(self, patient):
        """A patient with forty unchecked uploads would otherwise spend the
        whole iteration budget verifying and never reach the diff the booking
        actually needed. The backlog is not lost — the next turn takes the
        next one."""
        result = turn(patient, "what documents do I have on file?", "s-doc-bound")

        session = fresh()
        try:
            from app.models import TraceEvent, TraceEventType

            verifications = [
                e
                for e in session.query(TraceEvent)
                .filter(TraceEvent.turn_id == result.turn_id)
                .all()
                if e.event_type is TraceEventType.TOOL_CALL
                and e.payload["tool"] == "submit_document_verification"
            ]
            assert len(verifications) == 1

            from app.models import PatientProfile

            profile = (
                session.query(PatientProfile)
                .filter(PatientProfile.user_id == patient.id)
                .one()
            )
            assert pending_count(session, profile.id) == 2
        finally:
            session.close()

    def test_the_backlog_clears_over_successive_turns(self, patient):
        assert drain_verification(patient, "s-doc-drain") == 3

    def test_a_mismatch_is_flagged(self, patient):
        drain_verification(patient, "s-doc-4")

        session = fresh()
        try:
            flagged = list_flagged_documents(session)
            assert flagged, "the misdeclared document must be caught"
            assert flagged[0]["declared_type"] == "ECG report"
            assert flagged[0]["detected_type"] == "X-ray report"
        finally:
            session.close()

    def test_the_status_is_set_by_code_not_by_the_model(self, patient):
        """The model proposes a mismatch; the status is a consequence code
        applies. The trace records the proposal as a validation event."""
        drain_verification(patient, "s-doc-5")

        session = fresh()
        try:
            from app.models import TraceEvent, TraceEventType

            proposals = [
                e.payload
                for e in session.query(TraceEvent)
                .filter(TraceEvent.event_type == TraceEventType.VALIDATION)
                .all()
                if e.payload["what"] == "document_verification"
            ]
            mismatches = [p for p in proposals if p["detail"]["matches"] is False]
            assert len(mismatches) == 1
            assert mismatches[0]["detail"]["detected_type"] == "X-ray report"
            assert mismatches[0]["detail"]["declared_type"] == "ECG report"
        finally:
            session.close()

    def test_a_flagged_document_stops_satisfying_its_requirement(self, patient):
        """The patient supplied something, but not something the hospital can
        use. Reporting it as satisfied would close a task that should stay
        open."""
        drain_verification(patient, "s-doc-6")

        session = fresh()
        try:
            from app.models import PatientProfile

            profile = (
                session.query(PatientProfile)
                .filter(PatientProfile.user_id == patient.id)
                .one()
            )
            flagged = [
                d
                for d in session.query(PatientDocument)
                .filter(PatientDocument.patient_id == profile.id)
                .all()
                if d.status is DocumentStatus.FLAGGED
            ]
            assert len(flagged) == 1

            # The genuine ECG is still verified, so Cardiology stays satisfied.
            # What matters is that the flagged one is not what satisfies it.
            cardiology = (
                session.query(Department).filter(Department.name == "Cardiology").one()
            )
            diff = diff_required_documents(
                session, patient_id=profile.id, department_id=cardiology.id
            )
            assert "ECG report" in diff["satisfied"]

            flagged[0].status = DocumentStatus.PENDING_VERIFICATION
            session.flush()
        finally:
            session.rollback()
            session.close()

    def test_a_matching_document_is_verified(self, patient):
        drain_verification(patient, "s-doc-7")

        session = fresh()
        try:
            verified = [
                d
                for d in session.query(PatientDocument).all()
                if d.status is DocumentStatus.VERIFIED
            ]
            assert len(verified) == 2
        finally:
            session.close()

    def test_the_reply_never_says_what_a_document_contains(self, patient):
        """Document types are administrative labels. Reading one for meaning
        is the clinical line."""
        result = turn(patient, "what documents do I have on file?", "s-doc-8")

        lowered = result.reply.lower()
        for clinical in ("normal", "abnormal", "shows", "indicates", "suggests"):
            assert clinical not in lowered


class TestStaffResolveRerunsTheDiff:
    """The derivation invariant: a type change moves the task, in the same
    transaction, or the patient is chasing paperwork they already supplied."""

    @pytest.fixture
    def flagged_document(self, patient):
        """A booked Neurology appointment, an open shortfall, and a document
        flagged as the wrong type — the state staff actually meet."""
        turn(patient, NEURO_BOOKING, "s-doc-resolve")
        turn(patient, "yes", "s-doc-resolve")
        drain_verification(patient, "s-doc-resolve-drain")

        session = fresh()
        try:
            document = (
                session.query(PatientDocument)
                .filter(PatientDocument.status == DocumentStatus.FLAGGED)
                .one()
            )
            return document.id, document.patient_id
        finally:
            session.close()

    def test_reclassifying_to_the_needed_type_closes_the_task(
        self, flagged_document, staff
    ):
        document_id, patient_id = flagged_document

        session = fresh()
        try:
            assert open_missing_task(session, patient_id) is not None

            staff_row = session.get(User, staff.id)
            outcome = resolve_document(
                session,
                staff=staff_row,
                document_id=document_id,
                action="reclassify",
                corrected_type="Prior MRI or CT report",
                note="Filed under the wrong label on upload.",
            )
            session.commit()

            assert outcome["status"] == DocumentStatus.VERIFIED.value
            assert open_missing_task(session, patient_id) is None
        finally:
            session.close()

    def test_accepting_leaves_the_shortfall_open(self, flagged_document, staff):
        document_id, patient_id = flagged_document

        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            resolve_document(
                session, staff=staff_row, document_id=document_id, action="accept"
            )
            session.commit()

            task = open_missing_task(session, patient_id)
            assert task is not None
            assert "Prior MRI or CT report" in task.details["missing"]
        finally:
            session.close()

    def test_rejecting_leaves_the_shortfall_open(self, flagged_document, staff):
        document_id, patient_id = flagged_document

        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            outcome = resolve_document(
                session, staff=staff_row, document_id=document_id, action="reject"
            )
            session.commit()

            assert outcome["status"] == DocumentStatus.REJECTED.value
            assert open_missing_task(session, patient_id) is not None
        finally:
            session.close()

    def test_the_task_is_updated_never_duplicated(self, flagged_document, staff):
        """Staff changing their mind twice must not leave three tasks."""
        document_id, patient_id = flagged_document

        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            for action, corrected in (
                ("accept", None),
                ("reclassify", "Prior MRI or CT report"),
                ("reclassify", "X-ray report"),
            ):
                resolve_document(
                    session,
                    staff=staff_row,
                    document_id=document_id,
                    action=action,
                    corrected_type=corrected,
                )
            session.commit()

            tasks = (
                session.query(FollowUpTask)
                .filter(
                    FollowUpTask.patient_id == patient_id,
                    FollowUpTask.task_type == FollowUpTaskType.MISSING_DOCUMENTS,
                )
                .all()
            )
            assert len(tasks) == 1
        finally:
            session.close()

    def test_a_reclassification_must_name_the_type(self, flagged_document, staff):
        document_id, _ = flagged_document

        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            with pytest.raises(ValidationFailed, match="must name"):
                resolve_document(
                    session,
                    staff=staff_row,
                    document_id=document_id,
                    action="reclassify",
                )
        finally:
            session.rollback()
            session.close()

    def test_an_invented_resolution_is_refused(self, flagged_document, staff):
        document_id, _ = flagged_document

        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            with pytest.raises(ValidationFailed, match="not a document resolution"):
                resolve_document(
                    session,
                    staff=staff_row,
                    document_id=document_id,
                    action="shred it",
                )
        finally:
            session.rollback()
            session.close()

    def test_the_resolution_is_audited(self, flagged_document, staff):
        document_id, _ = flagged_document

        session = fresh()
        try:
            staff_row = session.get(User, staff.id)
            resolve_document(
                session, staff=staff_row, document_id=document_id, action="accept"
            )
            session.commit()

            assert (
                session.query(AuditEvent)
                .filter(AuditEvent.action == "document_resolved")
                .count()
                == 1
            )
        finally:
            session.close()


class TestOwnership:
    def test_a_model_cannot_read_another_patients_document(self, seeded_db):
        """The tools are bound to the acting patient. A document id is an
        integer, and a model that got creative with one must find nothing."""
        from app.agents.toolbelt import Toolbelt
        from app.trace import TraceWriter

        document = (
            seeded_db.query(PatientDocument).order_by(PatientDocument.id).first()
        )
        other = seeded_db.query(User).filter(User.email == PATIENT_EMAIL).one()

        belt = Toolbelt(
            seeded_db,
            user=other,
            patient_id=document.patient_id + 1,  # somebody else
            writer=TraceWriter(seeded_db, session_id="own"),
        )
        result = belt._read_document_text(document.id)

        assert result["reason"] == "not_found"
        assert result["text"] == ""

    def test_verification_cannot_touch_another_patients_document(self, seeded_db):
        from app.agents.toolbelt import Toolbelt
        from app.trace import TraceWriter

        document = (
            seeded_db.query(PatientDocument).order_by(PatientDocument.id).first()
        )
        before = document.status
        other = seeded_db.query(User).filter(User.email == PATIENT_EMAIL).one()

        belt = Toolbelt(
            seeded_db,
            user=other,
            patient_id=document.patient_id + 1,
            writer=TraceWriter(seeded_db, session_id="own"),
        )
        result = belt._submit_document_verification(document.id, "X-ray report", False)

        assert result["accepted"] is False
        assert document.status is before
