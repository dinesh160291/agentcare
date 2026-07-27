"""Every view, rendered once, against the real backend.

The behavioural tests cover the pages that *do* something. This sweep covers
the rest — and, more usefully, it fails when a new view is added and never
looked at. A page that raises on first render is invisible to a suite that only
drives the two or three screens someone remembered to write tests for.

The list is checked against the directory, so adding a view without adding it
here is itself a failure. Same discipline as the API's cross-patient probe: a
check that was never written must not be indistinguishable from one that passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

VIEWS = Path("ui/views")

#: view file → which role's token it needs.
ROLE_FOR_VIEW = {
    "patient_chat.py": "patient",
    "patient_appointments.py": "patient",
    "patient_documents.py": "patient",
    "patient_reminders.py": "patient",
    "patient_profile.py": "patient",
    "staff_queue.py": "staff",
    "staff_escalations.py": "staff",
    "staff_documents.py": "staff",
    "staff_trace.py": "staff",
    "staff_audit.py": "staff",
    "staff_capacity.py": "staff",
}


def test_every_view_file_is_in_the_sweep():
    """A view nobody rendered is a view nobody knows about."""
    on_disk = {
        path.name
        for path in VIEWS.glob("*.py")
        if path.name != "__init__.py"
    }
    assert on_disk == set(ROLE_FOR_VIEW), (
        f"views not covered: {on_disk - set(ROLE_FOR_VIEW)}; "
        f"listed but missing: {set(ROLE_FOR_VIEW) - on_disk}"
    )


@pytest.mark.parametrize("view", sorted(ROLE_FOR_VIEW))
def test_the_view_renders(view, wired_seeded, patient_token, staff_token):
    """No exception, with a real token and real data behind it."""
    token = patient_token if ROLE_FOR_VIEW[view] == "patient" else staff_token

    at = AppTest.from_file(str(VIEWS / view), default_timeout=120)
    at.session_state["token"] = token
    # The chat view keeps a display buffer; every view is written to cope with
    # an empty one, and this is where that is checked.
    at.session_state["transcript"] = []
    at.session_state["session_id"] = None
    at.session_state["last_run_id"] = None
    at.run()

    assert not at.exception, f"{view} raised: {at.exception}"
    assert not at.error, f"{view} rendered an error: {[e.value for e in at.error]}"
