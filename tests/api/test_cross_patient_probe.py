"""The one-digit edit, against every endpoint that accepts an id.

A judge's first probe is to take their own valid token, find someone else's
appointment id, and change a digit. The tools already enforce ownership; what
this file proves is that no *router* routed around them.

**The route list is read out of the running app, not typed here.** Every path
with a parameter in it must appear in ``PROBES`` or this module fails — the
same discipline the eval runner applies to expectation keys, and for the same
reason: a probe that was never written is indistinguishable from a probe that
passes. Adding an id-taking endpoint in a later phase therefore breaks this
test until someone says what the probe should be, which is the point.

Two expectations, and the difference between them matters:

* **patient-scoped** rows answer **404** — never 403, which would confirm the
  record exists and turn the id field into an enumeration oracle;
* **staff-only** routes answer **403**, because the existence of the staff area
  is not a secret and pretending otherwise would only confuse the client.
"""

from __future__ import annotations

import pytest
from tests.api.conftest import auth_header

from app.main import app
from app.models import AuditEvent, Notification

ASHA, ROHAN, STAFF = 1, 2, 5
AMBIGUOUS = "book an appointment, my kid has ear pain"

#: 404 — Asha's row, probed by Rohan. 403 — staff-only, probed by a patient.
OWNED, STAFF_ONLY = "owned", "staff_only"

#: One entry per parameterised route. ``id`` is a real row belonging to Asha
#: wherever the probe is an ownership probe, because ownership can only be
#: tested against a row that exists — a probe at id 9999 would 404 for the
#: boring reason and prove nothing.
PROBES: dict[tuple[str, str], dict] = {
    ("GET", "/workflow/runs/{run_id}"): {"kind": OWNED, "entity": "WorkflowRun"},
    ("GET", "/appointments/{appointment_id}"): {"kind": OWNED, "entity": "Appointment"},
    ("GET", "/documents/{document_id}"): {"kind": OWNED, "entity": "PatientDocument"},
    ("POST", "/notifications/{notification_id}/read"): {
        "kind": OWNED,
        "entity": "Notification",
    },
    ("GET", "/staff/patients/{patient_id}"): {"kind": STAFF_ONLY},
    ("GET", "/staff/runs/{run_id}/trace"): {"kind": STAFF_ONLY},
    ("POST", "/staff/runs/{run_id}/decision"): {
        "kind": STAFF_ONLY,
        "json": {"action": "approve"},
    },
    ("POST", "/staff/appointments/{appointment_id}/visit"): {
        "kind": STAFF_ONLY,
        "json": {"action": "missed"},
    },
    ("POST", "/staff/escalations/{escalation_id}/resolve"): {
        "kind": STAFF_ONLY,
        "json": {"status": "acknowledged"},
    },
    ("POST", "/staff/documents/{document_id}/resolve"): {
        "kind": STAFF_ONLY,
        "json": {"action": "accept"},
    },
    ("PATCH", "/staff/departments/{department_id}"): {
        "kind": STAFF_ONLY,
        "json": {"active": False},
    },
    ("PATCH", "/staff/doctors/{doctor_id}"): {
        "kind": STAFF_ONLY,
        "json": {"active": False},
    },
    ("POST", "/staff/doctors/{doctor_id}/slots"): {
        "kind": STAFF_ONLY,
        "json": {"start_times": ["2099-02-01T09:00:00"]},
    },
}


def parameterised_routes() -> set[tuple[str, str]]:
    """Every (method, path) the app serves that takes an id in its path.

    Read from the OpenAPI document rather than by walking ``app.routes``:
    included routers are nested objects whose shape has changed between FastAPI
    versions, and the published schema is both stable and the actual contract.
    """
    return {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        if "{" in path
        for method in operations
        if method.upper() not in ("HEAD", "OPTIONS")
    }


@pytest.fixture
def asha_owns(seeded_client, seeded_db):
    """Real ids belonging to Asha, one per patient-scoped entity."""
    run_id = seeded_client.post(
        "/workflow/messages",
        headers=auth_header(ASHA),
        json={"message": AMBIGUOUS, "session_id": "probe-setup"},
    ).json()["run_id"]

    seeded_db.add(
        Notification(id=901, patient_id=1, kind="workflow_update", title="Hi", body="")
    )
    seeded_db.commit()

    return {
        "{run_id}": run_id,
        "{appointment_id}": 1,
        "{document_id}": 1,
        "{notification_id}": 901,
        "{patient_id}": 1,
        "{escalation_id}": 1,
        "{department_id}": 1,
        "{doctor_id}": 1,
    }


def fill(path: str, ids: dict[str, int]) -> str:
    for placeholder, value in ids.items():
        path = path.replace(placeholder, str(value))
    return path


class TestTheProbeListIsComplete:
    def test_every_id_taking_route_has_a_probe(self):
        """A new endpoint with an id in its path fails here until it is probed."""
        unprobed = parameterised_routes() - set(PROBES)
        assert not unprobed, f"these routes take an id and are never probed: {unprobed}"

    def test_no_probe_names_a_route_that_no_longer_exists(self):
        """A probe against a deleted route is an assertion that never runs."""
        stale = set(PROBES) - parameterised_routes()
        assert not stale, f"these probes name routes the app does not serve: {stale}"


@pytest.mark.parametrize(("method", "path"), sorted(PROBES))
def test_the_one_digit_edit_is_refused(method, path, seeded_client, asha_owns):
    """Rohan's token, Asha's id — or a patient's token on a staff route."""
    probe = PROBES[(method, path)]
    expected = 404 if probe["kind"] == OWNED else 403

    response = seeded_client.request(
        method,
        fill(path, asha_owns),
        headers=auth_header(ROHAN),
        json=probe.get("json"),
    )
    assert response.status_code == expected, (
        f"{method} {path} answered {response.status_code}, expected {expected}: "
        f"{response.text}"
    )


@pytest.mark.parametrize(
    ("method", "path"),
    sorted(key for key, probe in PROBES.items() if probe["kind"] == OWNED),
)
def test_a_denied_probe_is_audited(method, path, seeded_client, seeded_db, asha_owns):
    """A denial that leaves no record is indistinguishable from a request that
    never happened — which is precisely the record you want when someone is
    walking the id space."""
    entity = PROBES[(method, path)]["entity"]
    seeded_client.request(
        method, fill(path, asha_owns), headers=auth_header(ROHAN), json=None
    )

    seeded_db.expire_all()
    denials = (
        seeded_db.query(AuditEvent)
        .filter(
            AuditEvent.action == "access_denied",
            AuditEvent.actor_id == ROHAN,
            AuditEvent.entity_type == entity,
        )
        .all()
    )
    assert denials, f"{method} {path} denied the probe without auditing it"


@pytest.mark.parametrize(
    ("method", "path"),
    sorted(key for key, probe in PROBES.items() if probe["kind"] == OWNED),
)
def test_a_real_row_probes_identically_to_a_missing_one(
    method, path, seeded_client, asha_owns
):
    """Status *and* body. A difference in either enumerates which ids exist."""
    forbidden = seeded_client.request(
        method, fill(path, asha_owns), headers=auth_header(ROHAN), json=None
    )
    missing = seeded_client.request(
        method,
        fill(path, dict.fromkeys(asha_owns, 999_999)),
        headers=auth_header(ROHAN),
        json=None,
    )
    assert forbidden.status_code == missing.status_code == 404
    assert forbidden.json() == missing.json()
