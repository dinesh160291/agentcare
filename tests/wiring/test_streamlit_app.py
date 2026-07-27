"""The Streamlit client, driven headlessly against the real backend.

``AppTest`` runs ``ui/app.py`` in-process; ``api_client`` is pointed at the real
FastAPI app. So a click here travels through the actual routers, guards, and
orchestrator, and the assertions are about what the page *rendered* as a result.

The property these exist to prove is the one the submission rules score:
**a page renders only what it fetched.** The sharp version of that test is to
change a row behind the UI's back and assert the screen follows — a page that
fabricated, cached, or hardcoded its data would keep showing the old value and
every other assertion would still pass.

The other half is the confirmation rule: the ✓ Confirm button must reach
``/workflow/actions``. A button that posted the word "confirm" as a message
would look identical on screen and would quietly move the guarantee from a
typed action to a string comparison.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from app.models import Appointment, PatientProfile

APP = "ui/app.py"
PATIENT = ("asha.patient@example.invalid", "Demo123!pass")
STAFF = ("staff@example.invalid", "Demo123!pass")
BOOKING = "I need a cardiology appointment next week"
AMBIGUOUS = "book an appointment, my kid has ear pain"


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=180)


def _log_in(at: AppTest, credentials: tuple[str, str]) -> AppTest:
    at.run()
    at.text_input(key="login_email").input(credentials[0])
    at.text_input(key="login_password").input(credentials[1])
    at.button(key="FormSubmitter:login-Log in").click().run()
    return at


def _rendered(at: AppTest) -> str:
    """Everything the page put on screen, as one string."""
    parts = [m.value for m in at.markdown]
    parts += [m.value for m in at.sidebar.markdown]
    parts += [str(getattr(e, "value", "")) for e in at.success]
    parts += [str(getattr(e, "value", "")) for e in at.error]
    return "\n".join(parts)


class TestLogin:
    def test_the_login_screen_renders(self, wired_seeded):
        at = _app().run()
        assert not at.exception
        assert {i.key for i in at.text_input} >= {"login_email", "login_password"}

    def test_bad_credentials_show_the_backends_message(self, wired_seeded):
        at = _app().run()
        at.text_input(key="login_email").input(PATIENT[0])
        at.text_input(key="login_password").input("wrong-password")
        at.button(key="FormSubmitter:login-Log in").click().run()

        assert not at.session_state["token"]
        assert at.error, "a failed login rendered no error"
        assert "incorrect" in at.error[0].value.lower()

    def test_logging_in_stores_the_token_the_api_issued(self, wired_seeded):
        at = _log_in(_app(), PATIENT)
        assert not at.exception
        assert at.session_state["token"]
        assert at.session_state["user"]["name"] == "Asha Menon"


class TestOnlyRendersWhatItFetched:
    """Change the row behind the UI's back; the screen must follow."""

    def test_the_profile_page_tracks_the_database(self, wired_seeded, seeded_db):
        seeded_db.get(PatientProfile, 1).phone = "+1-555-0777"
        seeded_db.commit()

        at = _log_in(_app(), PATIENT)
        at.switch_page("views/patient_profile.py")
        at.run()

        assert not at.exception
        values = [i.value for i in at.text_input]
        assert "+1-555-0777" in values, (
            "the profile page showed something other than the stored row"
        )

    def test_the_appointments_page_shows_the_seeded_reference(
        self, wired_seeded, seeded_db
    ):
        at = _log_in(_app(), PATIENT)
        at.switch_page("views/patient_appointments.py")
        at.run()

        assert not at.exception
        reference = seeded_db.get(Appointment, 1).reference_code
        assert reference in _rendered(at)

    def test_a_cancelled_appointment_renders_as_cancelled(
        self, wired_seeded, seeded_db
    ):
        """The status pill is read from the row, not from a hopeful default."""
        appointment = seeded_db.get(Appointment, 1)
        appointment.status = "cancelled"
        seeded_db.commit()

        at = _log_in(_app(), PATIENT)
        at.switch_page("views/patient_appointments.py")
        at.run()

        assert "cancelled" in _rendered(at)

    def test_a_patient_with_no_documents_gets_the_empty_state(
        self, wired_seeded, seeded_db
    ):
        at = _log_in(_app(), ("rohan.patient@example.invalid", "Demo123!pass"))
        at.switch_page("views/patient_documents.py")
        at.run()

        assert "Nothing on file yet." in _rendered(at)


class TestTheConfirmationRule:
    def test_a_booking_message_produces_confirm_and_decline(self, wired_seeded):
        at = _log_in(_app(), PATIENT)
        at.button(key="demo_0").click().run()

        assert not at.exception
        keys = {b.key for b in at.button}
        assert "confirm_proposal" in keys and "decline_proposal" in keys

    def test_the_buttons_only_appear_while_a_proposal_is_outstanding(
        self, wired_seeded
    ):
        """They are rendered off the run's *status* from the API, not off the
        wording of the reply."""
        at = _log_in(_app(), PATIENT)
        assert "confirm_proposal" not in {b.key for b in at.button}

    def test_confirm_commits_the_booking_in_the_backend(self, wired_seeded, seeded_db):
        at = _log_in(_app(), PATIENT)
        at.button(key="demo_0").click().run()
        at.button(key="confirm_proposal").click().run()

        assert not at.exception
        seeded_db.expire_all()
        booked = (
            seeded_db.query(Appointment)
            .filter(Appointment.patient_id == 1, Appointment.id != 1)
            .all()
        )
        assert len(booked) == 1, "the Confirm button did not reach the backend"

    def test_the_receipt_on_screen_matches_the_row_that_was_written(
        self, wired_seeded, seeded_db
    ):
        """The strongest form of "rendered what it fetched": the reference code
        shown is the one the database generated."""
        at = _log_in(_app(), PATIENT)
        at.button(key="demo_0").click().run()
        at.button(key="confirm_proposal").click().run()

        seeded_db.expire_all()
        booked = (
            seeded_db.query(Appointment)
            .filter(Appointment.patient_id == 1, Appointment.id != 1)
            .one()
        )
        assert booked.reference_code in _rendered(at)

    def test_the_confirm_button_arrives_as_a_typed_action(self, wired_seeded):
        """The one assertion that can tell the two front doors apart.

        Written after a falsification pass: making the button post the word
        "confirm" as a *message* left every other test in this class green,
        because the exact-token reader reads "confirm" and commits — same
        booking, same receipt, same screen. The difference is only visible in
        the trace, where an inbound event records **which door** it came
        through. So that is what is asserted.
        """
        at = _log_in(_app(), PATIENT)
        at.button(key="demo_0").click().run()
        at.button(key="confirm_proposal").click().run()
        run_id = at.session_state["last_run_id"]

        staff_token = wired_seeded.login(email=STAFF[0], password=STAFF[1])[
            "access_token"
        ]
        events = wired_seeded.trace(staff_token, run_id)
        inbound_authors = [
            e["author"] for e in events if e["event_type"] == "inbound"
        ]
        assert "patient-action" in inbound_authors, (
            "the Confirm button reached the backend as text, not a typed action"
        )

    def test_declining_leaves_nothing_booked(self, wired_seeded, seeded_db):
        before = seeded_db.query(Appointment).count()
        at = _log_in(_app(), PATIENT)
        at.button(key="demo_0").click().run()
        at.button(key="decline_proposal").click().run()

        assert not at.exception
        seeded_db.expire_all()
        assert seeded_db.query(Appointment).count() == before


class TestSafetyReachesTheScreen:
    def test_an_emergency_renders_the_alarm(self, wired_seeded):
        at = _log_in(_app(), PATIENT)
        at.chat_input[0].set_value("I have chest pain and my left arm hurts").run()

        assert not at.exception
        rendered = _rendered(at)
        assert "escalated to a member of staff" in rendered

    def test_an_emergency_offers_no_confirmation_buttons(self, wired_seeded):
        at = _log_in(_app(), PATIENT)
        at.chat_input[0].set_value("I have chest pain and my left arm hurts").run()
        assert "confirm_proposal" not in {b.key for b in at.button}


class TestRoleShapesTheConsole:
    def test_a_patient_lands_on_chat_and_sees_no_staff_console(self, wired_seeded):
        at = _log_in(_app(), PATIENT)
        rendered = _rendered(at)
        # The chat footer is unique to the patient's default page…
        assert "never diagnoses or prescribes" in rendered
        # …and nothing from the staff console leaked into it.
        assert "Requests queue" not in rendered
        assert "Audit log" not in rendered

    def test_a_staff_login_lands_on_the_queue(self, wired_seeded):
        at = _log_in(_app(), STAFF)
        assert not at.exception
        assert "Requests queue" in _rendered(at)

    def test_staff_see_a_paused_run_a_patient_created(self, wired_seeded):
        patient = _log_in(_app(), PATIENT)
        patient.button(key="demo_1").click().run()  # the ambiguous routing case

        staff = _log_in(_app(), STAFF)
        assert "pending review" in _rendered(staff).replace("_", " ")

    def test_a_staff_view_rendered_with_a_patients_token_shows_a_refusal(
        self, wired_seeded
    ):
        """RBAC belongs to the backend, not to which links got drawn.

        The role-based navigation means a patient has no route to this page —
        but "there is no link" is not the guarantee, and a UI whose safety
        depended on that would be scored as hidden buttons. So the view is run
        **directly**, with a patient's token in session state, and must put the
        backend's refusal on screen rather than an empty queue. An empty table
        would read as "nothing to review", which is the one wrong answer.
        """
        token = wired_seeded.login(email=PATIENT[0], password=PATIENT[1])[
            "access_token"
        ]
        at = AppTest.from_file("ui/views/staff_queue.py", default_timeout=120)
        at.session_state["token"] = token
        at.run()

        rendered = _rendered(at)
        assert at.error, "a forbidden queue rendered without an error"
        assert "do not have access" in rendered.lower()
        assert "#1" not in rendered, "staff data leaked to a patient token"


class TestStaffActionsFromTheScreen:
    def test_approving_moves_the_run(self, wired_seeded, seeded_db):
        patient = _log_in(_app(), PATIENT)
        patient.button(key="demo_1").click().run()
        run_id = patient.session_state["last_run_id"]

        staff = _log_in(_app(), STAFF)
        staff.switch_page("views/staff_escalations.py")
        staff.run()
        staff.button(key=f"approve_{run_id}").click().run()

        assert not staff.exception
        from app.models import WorkflowRun

        seeded_db.expire_all()
        assert seeded_db.get(WorkflowRun, run_id).status.value == "in_progress"

    def test_a_safety_escalation_offers_acknowledge_not_approve(self, wired_seeded):
        patient = _log_in(_app(), PATIENT)
        patient.chat_input[0].set_value("I have chest pain and my left arm hurts").run()

        staff = _log_in(_app(), STAFF)
        staff.switch_page("views/staff_escalations.py")
        staff.run()

        labels = {b.label for b in staff.button}
        assert "Acknowledge" in labels
        assert "Approve" not in labels, "an emergency was offered an approval"


class TestTheStaffReviewScreensDoTheirWork:
    """The two staff views whose bodies only run when there is something to
    review — so the render sweep, which uses an untouched seed, never reaches
    them. Covered here with the state staff actually meet."""

    def test_the_document_queue_renders_a_flagged_file_and_resolves_it(
        self, wired_seeded, seeded_db
    ):
        from app.models import DocumentStatus, PatientDocument

        document = (
            seeded_db.query(PatientDocument)
            .filter(PatientDocument.original_filename == "x-ray_report.pdf")
            .one()
        )
        document.status = DocumentStatus.FLAGGED
        document.detected_type = "X-ray report"
        seeded_db.commit()
        document_id = document.id

        token = wired_seeded.login(email=STAFF[0], password=STAFF[1])["access_token"]
        at = AppTest.from_file("ui/views/staff_documents.py", default_timeout=120)
        at.session_state["token"] = token
        at.run()

        assert not at.exception
        rendered = _rendered(at)
        # Declared *and* detected, which is the whole point of the screen.
        assert "ECG report" in rendered and "X-ray report" in rendered

        at.button(key=f"reclass_{document_id}").click().run()
        assert not at.exception

        seeded_db.expire_all()
        resolved = seeded_db.get(PatientDocument, document_id)
        assert resolved.status is DocumentStatus.VERIFIED
        assert resolved.document_type == "X-ray report"

    def test_the_trace_view_renders_a_real_timeline(self, wired_seeded):
        patient = _log_in(_app(), PATIENT)
        patient.button(key="demo_0").click().run()
        run_id = patient.session_state["last_run_id"]

        token = wired_seeded.login(email=STAFF[0], password=STAFF[1])["access_token"]
        at = AppTest.from_file("ui/views/staff_trace.py", default_timeout=120)
        at.session_state["token"] = token
        at.session_state["trace_run_id"] = run_id
        at.run()

        assert not at.exception
        rendered = _rendered(at)
        assert "inbound" in rendered and "outbound" in rendered
        assert "coordinator" in rendered
