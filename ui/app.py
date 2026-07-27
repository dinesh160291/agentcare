"""AgentCare — the Streamlit client.

A thin client, and thin is a rule rather than an aspiration. This script owns
three things: who is logged in, which pages that role may see, and the shell
those pages render into. Everything else — what a message means, whether a
booking may commit, which rows a user may read — is answered on the other side
of ``api_client``.

**Role-based navigation is a convenience, not the control.** A patient who
guesses a staff URL gets a 403 from the backend, because RBAC is enforced there
and not by which links this file drew. The navigation is built per role so the
console does not offer doors that will slam; it is not what keeps them shut.

Run it with::

    python -m streamlit run ui/app.py
"""

from __future__ import annotations

# --- sys.path, and why this is here -------------------------------------
#
# Streamlit prepends the entry script's *directory* to ``sys.path``. This file
# is ``ui/app.py``, so ``ui/`` goes on the front — and from there ``import app``
# finds this very file, as a module, before it finds the backend's ``app/``
# package. The failure is ``No module named 'app.config'; 'app' is not a
# package``, and it happens only when Streamlit launches the script: under
# pytest's ``AppTest`` the repository root is already first, so the whole suite
# passes while ``streamlit run`` dies on the login screen.
#
# Putting the root ahead of ``ui/`` restores the ordering. Kept at the very top
# because it has to run before anything imports ``app``.
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_ROOT = str(Path(__file__).resolve().parent.parent)
if sys.path[:1] != [_ROOT]:
    sys.path.insert(0, _ROOT)

import streamlit as st  # noqa: E402

from ui import theme  # noqa: E402
from ui.api_client import ApiError, get_client  # noqa: E402

#: Printed by ``scripts/seed.py``. Synthetic, and documented in the README —
#: offering them saves a judge from typing a password they have to go and find.
DEMO_ACCOUNTS = [
    ("Asha Menon", "asha.patient@example.invalid", "patient"),
    ("Rohan Gupta", "rohan.patient@example.invalid", "patient"),
    ("Priya Desk", "staff@example.invalid", "staff"),
]
DEMO_PASSWORD = "Demo123!pass"


def _init_state() -> None:
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("session_id", None)
    # A display buffer, not a source of truth. The conversation itself lives in
    # the backend's session store and survives a restart; this is only what
    # this browser tab has seen, and it is rebuilt from replies the API sent.
    st.session_state.setdefault("transcript", [])
    st.session_state.setdefault("last_run_id", None)


def sign_out() -> None:
    for key in ("token", "user", "session_id", "transcript", "last_run_id"):
        st.session_state.pop(key, None)
    _init_state()


def _login_form() -> None:
    client = get_client()

    st.markdown(
        '<h1 style="letter-spacing:0.04em;margin-bottom:0;">AGENTCARE</h1>'
        '<p class="ac-dim" style="margin-top:2px;">Agentic patient administration '
        "and care coordination.</p>",
        unsafe_allow_html=True,
    )

    login_tab, register_tab = st.tabs(["Log in", "Register"])

    with login_tab:
        with st.form("login"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            try:
                result = client.login(email=email, password=password)
            except ApiError as exc:
                st.error(exc.detail)
            else:
                st.session_state.token = result["access_token"]
                st.session_state.user = {
                    "user_id": result["user_id"],
                    "name": result["name"],
                    "role": result["role"],
                }
                st.rerun()

    with register_tab:
        with st.form("register"):
            name = st.text_input("Full name", key="reg_name")
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_password")
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            try:
                client.register(name=name, email=email, password=password)
            except ApiError as exc:
                st.error(exc.detail)
            else:
                st.success("Account created. Log in with those details.")

    st.markdown(
        '<p class="ac-dim"><b>Your session ends if you refresh the page.</b> The '
        "access token is held in server-side session state and is deliberately "
        "never written to the URL or to a browser cookie — a token in either one "
        "outlives the tab it was issued to. Log in again to carry on: an open "
        "request is held against your record, not against this tab, so the next "
        "message picks it up where it was. Only the on-screen transcript restarts."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        theme.card(
            "".join(
                f'<div style="display:flex;gap:10px;padding:3px 0;font-size:13px;">'
                f'<span style="width:120px;font-weight:600;">{theme.esc(n)}</span>'
                f"{theme.tag(role)}"
                f'<span class="ac-dim ac-num">{theme.esc(mail)}</span></div>'
                for n, mail, role in DEMO_ACCOUNTS
            )
            + f'<p class="ac-dim" style="margin:8px 0 0;">Password for every seeded '
            f"account: <b>{theme.esc(DEMO_PASSWORD)}</b></p>",
            kicker="Synthetic demo accounts",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="ac-dim">Auth events are audit-logged. Roles and row ownership are '
        "enforced in the backend — another patient's id returns 404, never 403.</p>",
        unsafe_allow_html=True,
    )


def _sidebar_footer() -> None:
    user = st.session_state.user
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            f'<p class="ac-dim">{theme.esc(user["name"])} · '
            f'{theme.esc(user["role"].title())}</p>',
            unsafe_allow_html=True,
        )
        try:
            health = get_client().health()
            st.markdown(
                f'<span class="ac-tag" style="color:{theme.TEXT_SECONDARY};'
                f'border-color:{theme.DIVIDER};">LLM_PROVIDER='
                f'{theme.esc(health.get("provider"))}</span>',
                unsafe_allow_html=True,
            )
        except ApiError:
            st.markdown(
                '<span class="ac-dim">backend unreachable</span>',
                unsafe_allow_html=True,
            )
        if st.button("Log out", key="logout"):
            sign_out()
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="AgentCare", page_icon="✚", layout="wide")
    _init_state()
    theme.inject()

    if not st.session_state.token:
        _login_form()
        return

    role = (st.session_state.user or {}).get("role")

    if role == "staff":
        pages = [
            st.Page("views/staff_queue.py", title="Requests queue", default=True),
            st.Page("views/staff_escalations.py", title="Escalations & reviews"),
            st.Page("views/staff_documents.py", title="Document reviews"),
            st.Page("views/staff_trace.py", title="Trace timeline"),
            st.Page("views/staff_audit.py", title="Audit log"),
            st.Page("views/staff_capacity.py", title="Capacity"),
        ]
    else:
        pages = [
            st.Page("views/patient_chat.py", title="Chat", default=True),
            st.Page("views/patient_appointments.py", title="My appointments"),
            st.Page("views/patient_documents.py", title="Documents"),
            st.Page("views/patient_reminders.py", title="Reminders & notifications"),
            st.Page("views/patient_profile.py", title="Profile"),
        ]

    _sidebar_footer()
    st.navigation(pages).run()


main()
