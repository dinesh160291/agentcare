"""The bits every view repeats: the top bar, the token, the error wrapper.

Small on purpose. A view should read as "fetch, then render", and anything that
makes that harder to see belongs here instead.

:func:`header` draws the design reference's 56px top bar — page title left,
then the notification bell, the signed-in name, and Log out on the right. It is
drawn per page rather than once for the app because **Streamlit has no
app-level top region**: everything above the page belongs to the sidebar or to
the toolbar. So it scrolls with the content instead of staying pinned, which is
the one part of that bar the framework will not do.
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from ui import theme
from ui.api_client import ApiClient, ApiError, get_client


def token() -> str:
    return st.session_state.get("token") or ""


def client() -> ApiClient:
    return get_client()


def role() -> str:
    return (st.session_state.get("user") or {}).get("role") or ""


def sign_out() -> None:
    """Drop everything this browser tab knew. Lives here because the top bar
    carries the button and every page draws the top bar."""
    for key in ("token", "user", "session_id", "transcript", "last_run_id", "pending"):
        st.session_state.pop(key, None)


@st.cache_data(ttl=8, show_spinner=False)
def _unread(auth: str) -> int | None:
    """How many notifications are unread, for the bell's badge.

    Cached briefly because this is drawn on *every* page render, including the
    one that exists purely to paint a sent message before its turn runs — an
    uncached call there would put a round trip in front of the thing that fix
    was for. Eight seconds of staleness on a badge costs nothing; the page a
    reader opens to act on it fetches for itself.

    ``None`` means "could not ask", and draws no badge rather than a zero: the
    page's own fetch reports the outage, and two banners for one cause is one
    too many.
    """
    try:
        return sum(1 for note in get_client().notifications(auth) if not note["read"])
    except ApiError:
        return None


def _identity() -> None:
    """Bell, name, and Log out — the right-hand end of the top bar."""
    meta, action = st.columns([3, 1.3], vertical_alignment="center")

    user = st.session_state.get("user") or {}
    bell = ""
    if role() == "patient":
        unread = _unread(token())
        badge = f'<span class="ac-badge">{unread}</span>' if unread else ""
        bell = f'<span class="ac-bell">☉{badge}</span>'

    meta.markdown(
        f'<div class="ac-identity">{bell}'
        f'<span>{theme.esc(user.get("name"))}</span></div>',
        unsafe_allow_html=True,
    )
    if action.button("Log out", key="logout"):
        sign_out()
        st.rerun()


def header(title: str, subtitle: str | None = None) -> None:
    theme.inject()

    left, right = st.columns([5, 3], vertical_alignment="center")
    with left:
        st.markdown(
            f'<h2 style="margin:0 0 2px;">{theme.esc(title)}</h2>'
            + (f'<p class="ac-dim" style="margin:0;">{theme.esc(subtitle)}</p>'
               if subtitle else ""),
            unsafe_allow_html=True,
        )
    with right:
        _identity()

    st.markdown('<div class="ac-rule"></div>', unsafe_allow_html=True)


def fetch(call: Callable[[], Any], *, default: Any = None) -> Any:
    """Run an API read, and put a failure on screen rather than a traceback.

    Deliberately does **not** swallow the error into an empty list silently:
    an empty table and a broken backend must not look the same, because one of
    them means "nothing to review" and the other means "you are not reviewing".
    """
    try:
        return call()
    except ApiError as exc:
        if exc.is_auth:
            st.warning("Your session has expired. Log out and back in.")
        else:
            st.error(f"Could not load this: {exc.detail}")
        return default


def act(call: Callable[[], Any]) -> tuple[bool, Any]:
    """Run an API write. Returns ``(ok, result_or_error)``."""
    try:
        return True, call()
    except ApiError as exc:
        return False, exc


__all__ = ["act", "client", "fetch", "header", "role", "sign_out", "token"]
