"""Streamlit's **only** path to the backend.

Every rule this project enforces lives behind the HTTP boundary — RBAC,
ownership, the confirmation reader, the state machine. A page that reached past
this module for a "quick" database read would be rendering data no guard ever
saw, and the submission rules score that as faked. So there is exactly one door,
and this is it: no page imports ``sqlalchemy``, ``app.models``, or anything else
from ``app/``.

The client is a thin transport. It adds the bearer token, turns a non-2xx into
a typed :class:`ApiError`, and hands back whatever JSON the API sent — it does
not reshape payloads, compute anything, or decide what a status means. Those
are the backend's answers, and a second copy here would be a second copy to
drift.

**The injection seam.** ``get_client`` returns a process-wide client built from
``API_BASE_URL``; :func:`set_client` replaces it. Tests use that to point the
whole UI at the real FastAPI app over an ASGI transport, so the wiring tests
exercise Streamlit against the actual backend rather than a mock of it — which
is the only version of those tests worth having.

The one import from ``app/`` in the whole of ``ui/`` is ``app.config``, and it
is deliberate rather than a leak: ``API_BASE_URL`` is configuration, not logic,
and the project's standing rule is that ``app/config.py`` is the only module
that reads the environment. A second ``os.environ`` lookup here would be the
violation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

#: Long enough for a real agent turn: a chat message drives the Coordinator,
#: a specialist, and several tool calls before it answers.
DEFAULT_TIMEOUT = 120.0


class ApiError(Exception):
    """A non-2xx response, carrying enough for a page to say what happened."""

    def __init__(self, status_code: int, detail: str, payload: Any = None) -> None:
        self.status_code = status_code
        self.detail = detail
        self.payload = payload
        super().__init__(f"{status_code}: {detail}")

    @property
    def is_auth(self) -> bool:
        return self.status_code == 401

    @property
    def is_forbidden(self) -> bool:
        return self.status_code == 403

    @property
    def is_missing(self) -> bool:
        return self.status_code == 404


@dataclass
class ApiClient:
    """Bound to one base URL (or one ASGI transport, under test)."""

    http: httpx.Client

    # --- plumbing --------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        token: str | None = None,
        json: Any = None,
        params: dict[str, Any] | None = None,
        files: Any = None,
        data: Any = None,
    ) -> Any:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = self.http.request(
            method,
            path,
            headers=headers,
            json=json,
            params=params,
            files=files,
            data=data,
        )
        if response.status_code >= 400:
            raise ApiError(
                response.status_code, _detail_of(response), _payload_of(response)
            )
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    # --- auth ------------------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health")

    def register(self, *, name: str, email: str, password: str) -> dict:
        return self._request(
            "POST",
            "/auth/register",
            json={"name": name, "email": email, "password": password},
        )

    def login(self, *, email: str, password: str) -> dict:
        return self._request(
            "POST", "/auth/login", json={"email": email, "password": password}
        )

    def me(self, token: str) -> dict:
        return self._request("GET", "/auth/me", token=token)

    # --- patient ---------------------------------------------------------

    def profile(self, token: str) -> dict:
        return self._request("GET", "/patients/me", token=token)

    def update_profile(self, token: str, changes: dict[str, Any]) -> dict:
        return self._request("PATCH", "/patients/me", token=token, json=changes)

    def context(self, token: str) -> dict:
        return self._request("GET", "/patients/me/context", token=token)

    # --- the two front doors ---------------------------------------------

    def send_message(self, token: str, *, message: str, session_id: str | None) -> dict:
        """A free-text turn."""
        body: dict[str, Any] = {"message": message}
        if session_id:
            body["session_id"] = session_id
        return self._request("POST", "/workflow/messages", token=token, json=body)

    def send_action(self, token: str, *, action: str, session_id: str) -> dict:
        """A ✅ Confirm / ❌ Decline click.

        A **different endpoint** from :meth:`send_message`, not a message that
        happens to say "confirm". The button carries no free text, so nothing
        downstream reads language — which is the whole guarantee.
        """
        return self._request(
            "POST",
            "/workflow/actions",
            token=token,
            json={"action": action, "session_id": session_id},
        )

    def runs(self, token: str) -> list[dict]:
        return self._request("GET", "/workflow/runs", token=token)

    def run(self, token: str, run_id: int) -> dict:
        return self._request("GET", f"/workflow/runs/{run_id}", token=token)

    # --- patient records -------------------------------------------------

    def appointments(self, token: str, *, live_only: bool = False) -> list[dict]:
        return self._request(
            "GET", "/appointments", token=token, params={"live_only": live_only}
        )

    def documents(self, token: str) -> list[dict]:
        return self._request("GET", "/documents", token=token)

    def upload_document(
        self,
        token: str,
        *,
        filename: str,
        content: bytes,
        declared_type: str,
        content_type: str = "application/octet-stream",
    ) -> dict:
        """Upload a file. Refusals arrive as :class:`ApiError` with the payload.

        The backend sniffs magic bytes and generates its own filename, so
        whatever is passed here is a label — this client neither validates nor
        renames it, because doing so would be a second, weaker copy of a check
        that already exists on the other side.
        """
        return self._request(
            "POST",
            "/documents",
            token=token,
            files={"file": (filename, content, content_type)},
            data={"declared_type": declared_type},
        )

    def reminders(self, token: str, *, include_inactive: bool = False) -> list[dict]:
        return self._request(
            "GET",
            "/reminders",
            token=token,
            params={"include_inactive": include_inactive},
        )

    def tasks(self, token: str) -> list[dict]:
        return self._request("GET", "/tasks", token=token)

    def notifications(self, token: str) -> list[dict]:
        return self._request("GET", "/notifications", token=token)

    def mark_notification_read(self, token: str, notification_id: int) -> dict:
        return self._request(
            "POST", f"/notifications/{notification_id}/read", token=token
        )

    # --- staff -----------------------------------------------------------

    def queue(self, token: str, *, status: str | None = None) -> list[dict]:
        params = {"status": status} if status else None
        return self._request("GET", "/staff/queue", token=token, params=params)

    def patient_view(self, token: str, patient_id: int) -> dict:
        return self._request("GET", f"/staff/patients/{patient_id}", token=token)

    def escalations(self, token: str) -> list[dict]:
        return self._request("GET", "/staff/escalations", token=token)

    def flagged_documents(self, token: str) -> list[dict]:
        return self._request("GET", "/staff/documents/flagged", token=token)

    def swept_visits(self, token: str) -> list[dict]:
        return self._request("GET", "/staff/visits", token=token)

    def correct_visit(self, token: str, appointment_id: int, *, action: str) -> dict:
        return self._request(
            "POST",
            f"/staff/appointments/{appointment_id}/visit",
            token=token,
            json={"action": action},
        )

    def decide(
        self,
        token: str,
        run_id: int,
        *,
        action: str,
        department_name: str | None = None,
        note: str = "",
    ) -> dict:
        return self._request(
            "POST",
            f"/staff/runs/{run_id}/decision",
            token=token,
            json={
                "action": action,
                "department_name": department_name,
                "note": note,
            },
        )

    def resolve_escalation(
        self, token: str, escalation_id: int, *, status: str, note: str = ""
    ) -> dict:
        return self._request(
            "POST",
            f"/staff/escalations/{escalation_id}/resolve",
            token=token,
            json={"status": status, "note": note},
        )

    def resolve_document(
        self,
        token: str,
        document_id: int,
        *,
        action: str,
        corrected_type: str | None = None,
        note: str = "",
    ) -> dict:
        return self._request(
            "POST",
            f"/staff/documents/{document_id}/resolve",
            token=token,
            json={
                "action": action,
                "corrected_type": corrected_type,
                "note": note,
            },
        )

    def trace(self, token: str, run_id: int) -> list[dict]:
        return self._request("GET", f"/staff/runs/{run_id}/trace", token=token)

    def audit(
        self, token: str, *, action: str | None = None, limit: int = 100
    ) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if action:
            params["action"] = action
        return self._request("GET", "/staff/audit", token=token, params=params)

    def departments(self, token: str) -> list[dict]:
        return self._request("GET", "/staff/departments", token=token)

    def set_department_active(
        self, token: str, department_id: int, *, active: bool
    ) -> dict:
        return self._request(
            "PATCH",
            f"/staff/departments/{department_id}",
            token=token,
            json={"active": active},
        )

    def set_doctor_active(self, token: str, doctor_id: int, *, active: bool) -> dict:
        return self._request(
            "PATCH", f"/staff/doctors/{doctor_id}", token=token, json={"active": active}
        )

    def add_slots(
        self,
        token: str,
        doctor_id: int,
        *,
        start_times: list[str],
        duration_minutes: int = 30,
    ) -> dict:
        return self._request(
            "POST",
            f"/staff/doctors/{doctor_id}/slots",
            token=token,
            json={
                "start_times": start_times,
                "duration_minutes": duration_minutes,
            },
        )


def _detail_of(response: httpx.Response) -> str:
    """The API's own message, or a readable stand-in.

    FastAPI's validation errors arrive as a list under ``detail``; the routers'
    own refusals arrive as a string. Both are flattened to something a page can
    put on screen, and neither is reinterpreted.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text or f"Request failed ({response.status_code})"

    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list) and detail:
        parts = [
            str(item.get("msg", item)) for item in detail if isinstance(item, dict)
        ]
        if parts:
            return "; ".join(parts)
    # The upload path answers with the tool's own shape-stable refusal dict.
    if isinstance(body, dict) and body.get("message"):
        return str(body["message"])
    return f"Request failed ({response.status_code})"


def _payload_of(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


# --- the process-wide client, and the seam tests replace ------------------

_client: ApiClient | None = None


def build_client(base_url: str, *, timeout: float = DEFAULT_TIMEOUT) -> ApiClient:
    return ApiClient(httpx.Client(base_url=base_url, timeout=timeout))


def set_client(client: ApiClient | None) -> None:
    """Point the whole UI at a given client. Used by the wiring tests."""
    global _client
    _client = client


def get_client() -> ApiClient:
    """The client every page uses.

    Built lazily from ``API_BASE_URL`` so that importing a page does not open a
    socket, and so tests can inject before the first call.
    """
    global _client
    if _client is None:
        from app.config import get_settings

        _client = build_client(get_settings().api_base_url)
    return _client


__all__ = [
    "ApiClient",
    "ApiError",
    "build_client",
    "get_client",
    "set_client",
]
