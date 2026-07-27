"""The bits every view repeats: the header, the token, the error wrapper.

Small on purpose. A view should read as "fetch, then render", and anything that
makes that harder to see belongs here instead.
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


def header(title: str, subtitle: str | None = None) -> None:
    theme.inject()
    st.markdown(
        f'<h2 style="margin:0 0 2px;">{theme.esc(title)}</h2>'
        + (f'<p class="ac-dim" style="margin:0 0 14px;">{theme.esc(subtitle)}</p>'
           if subtitle else '<div style="height:10px;"></div>'),
        unsafe_allow_html=True,
    )


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


__all__ = ["act", "client", "fetch", "header", "token"]
