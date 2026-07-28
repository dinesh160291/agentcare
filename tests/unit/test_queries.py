"""The deterministic query path: "what do I have?", answered from the rows.

Three phrasings dead-ended in one live session — "show my appointments", "show
my upcoming appointments?", "what documents I have on file?" — while a fourth,
"what appointments I have coming up?", worked. Nothing separated them except
which sentence the Coordinator felt like emitting a plan for.

Two halves are tested here and they fail differently. **Detection** is a word
problem, and its dangerous direction is a false positive on an *action* — a
booking read as a listing is a booking that never happens. **Rendering** is a
facts problem, and its dangerous direction is asserting something the rows do
not say, which is what "no additional documents are required" did while
meaning "I had nothing to compare against".
"""

from __future__ import annotations

import pytest

from app.models import Appointment, AppointmentStatus, PatientProfile, User
from app.workflow.queries import QueryKind, answer_query, detect_query

ASHA_PROFILE_ID = 1
ROHAN_PROFILE_ID = 2
SEEDED_APPOINTMENT_ID = 1


class TestDetection:
    """What counts as a listing question, and — more important — what does not."""

    @pytest.mark.parametrize(
        "message, kind",
        [
            ("show my appointments", QueryKind.APPOINTMENTS),
            ("show my upcoming appointments?", QueryKind.APPOINTMENTS),
            ("what appointments I have coming up?", QueryKind.APPOINTMENTS),
            ("can you tell me my appointments", QueryKind.APPOINTMENTS),
            ("list my bookings", QueryKind.APPOINTMENTS),
            ("what documents I have on file?", QueryKind.DOCUMENTS),
            ("can you tell me what documents I have on file?", QueryKind.DOCUMENTS),
            ("what documents do I need to bring for my ENT visit?", QueryKind.DOCUMENTS),
            ("any reminders for me?", QueryKind.REMINDERS),
            ("what reminders do I have", QueryKind.REMINDERS),
        ],
    )
    def test_the_live_phrasings_are_all_recognised(self, message, kind):
        assert detect_query(message) is kind

    @pytest.mark.parametrize(
        "message",
        [
            "I need a cardiology appointment next week",
            "book me an appointment",
            "please reschedule my appointment to next week",
            "cancel my appointment",
            "move my appointment to Friday",
            "I want to upload a document",
            "can I change my booking",
        ],
    )
    def test_an_action_is_never_a_listing_question(self, message):
        """The expensive direction. A booking request read as a listing is a
        booking that silently never happens, and every one of these contains a
        subject noun — which is exactly why the subject alone cannot decide."""
        assert detect_query(message) is None

    @pytest.mark.parametrize(
        "message",
        [
            "who won the fifa world cup?",
            "what's the weather like today?",
            "how is nvidia stock doing",
            "tell me a joke",
            "",
            "   ",
        ],
    )
    def test_off_topic_is_not_a_listing_question(self, message):
        assert detect_query(message) is None

    def test_a_subject_without_an_ask_is_not_a_question(self):
        """"My appointment situation is confusing" names the subject and asks
        for no list. It falls through to planning, as it should."""
        assert detect_query("my appointment situation is confusing") is None

    def test_documents_outrank_appointments_when_both_are_named(self):
        """"What documents do I need for my appointment" names both nouns and
        is about one of them. Answering the other half would be a fluent reply
        to a question nobody asked."""
        assert (
            detect_query("what documents do I need for my appointment")
            is QueryKind.DOCUMENTS
        )


class TestRenderingAppointmentsAndReminders:
    def test_an_appointment_is_stated_from_its_row(self, seeded_db):
        answer = answer_query(
            seeded_db,
            patient_id=ASHA_PROFILE_ID,
            kind=QueryKind.APPOINTMENTS,
            message="show my appointments",
        )
        appointment = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID)
        assert appointment.reference_code in answer
        assert appointment.department.name in answer
        assert appointment.doctor.name in answer

    def test_times_are_twelve_hour(self, seeded_db):
        """One formatter for everything a patient reads. A listing that said
        09:00 beside a proposal saying 9:00 AM reads as two appointments."""
        answer = answer_query(
            seeded_db,
            patient_id=ASHA_PROFILE_ID,
            kind=QueryKind.APPOINTMENTS,
            message="show my appointments",
        )
        assert "AM" in answer or "PM" in answer
        assert "09:00" not in answer

    def test_a_cancelled_appointment_is_not_listed_as_upcoming(self, seeded_db):
        appointment = seeded_db.get(Appointment, SEEDED_APPOINTMENT_ID)
        appointment.status = AppointmentStatus.CANCELLED
        seeded_db.flush()

        answer = answer_query(
            seeded_db,
            patient_id=ASHA_PROFILE_ID,
            kind=QueryKind.APPOINTMENTS,
            message="show my appointments",
        )
        assert appointment.reference_code not in answer
        assert "no upcoming appointments" in answer

    def test_nothing_scheduled_says_so_plainly(self, seeded_db):
        answer = answer_query(
            seeded_db,
            patient_id=ROHAN_PROFILE_ID,
            kind=QueryKind.REMINDERS,
            message="any reminders?",
        )
        assert "no reminders" in answer.lower()

    def test_another_patients_rows_are_never_shown(self, seeded_db):
        """The listing is bound to the patient it was asked for."""
        answer = answer_query(
            seeded_db,
            patient_id=ROHAN_PROFILE_ID,
            kind=QueryKind.APPOINTMENTS,
            message="show my appointments",
        )
        assert "AC-000001" not in answer


class TestRenderingDocuments:
    """Item 4. The requirements half needs a department, and there are two
    places to find one — neither of which was being read."""

    def test_the_department_comes_from_the_message(self, seeded_db):
        """The live question, verbatim. ENT is named in it and the answer said
        "no department has been decided for this request"."""
        answer = answer_query(
            seeded_db,
            patient_id=ROHAN_PROFILE_ID,
            kind=QueryKind.DOCUMENTS,
            message="what documents do I need to bring for my ent visit?",
        )
        assert "Optional but helpful for ENT: Previous audiometry report." in answer

    def test_an_empty_mandatory_list_renders_cleanly(self, seeded_db):
        """ENT asks for nothing mandatory. The optional line has to stand on
        its own — a template that only knew how to print a "required" section
        would answer this by saying nothing is needed, which is false."""
        answer = answer_query(
            seeded_db,
            patient_id=ROHAN_PROFILE_ID,
            kind=QueryKind.DOCUMENTS,
            message="what documents do I need for my ENT visit?",
        )
        assert "Required before" not in answer
        assert "Optional but helpful for ENT" in answer
        assert "nothing is needed" not in answer.lower()
        assert "no additional documents" not in answer.lower()

    def test_the_department_falls_back_to_the_appointment(self, seeded_db):
        """Nothing named in the message, one appointment on the books. Asha's
        is Cardiology, which does have mandatory requirements."""
        answer = answer_query(
            seeded_db,
            patient_id=ASHA_PROFILE_ID,
            kind=QueryKind.DOCUMENTS,
            message="what documents do I have on file?",
        )
        assert "Cardiology" in answer

    def test_no_department_anywhere_claims_nothing(self, seeded_db):
        """The defect in one assertion. With no department named and no
        appointment to borrow one from there is nothing to diff against — and
        "nothing to compare" must never be said as "nothing is required"."""
        answer = answer_query(
            seeded_db,
            patient_id=ROHAN_PROFILE_ID,
            kind=QueryKind.DOCUMENTS,
            message="what documents do I have on file?",
        )
        assert "no additional documents" not in answer.lower()
        assert "nothing else is needed" not in answer.lower()
        assert "not required" not in answer.lower()

    def test_the_upload_pointer_rides_with_a_shortfall(self, seeded_db):
        answer = answer_query(
            seeded_db,
            patient_id=ROHAN_PROFILE_ID,
            kind=QueryKind.DOCUMENTS,
            message="what documents do I need to bring for my ent visit?",
        )
        assert "Documents page" in answer
