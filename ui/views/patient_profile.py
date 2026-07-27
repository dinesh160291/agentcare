"""The patient's administrative details.

A patch, not a replace: only the fields this form submits are sent, and the
backend leaves anything absent alone. Administrative only — there is no
clinical field on this page and there is not meant to be one.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from ui import theme
from ui.shell import act, client, fetch, header, token

LANGUAGES = ["English", "Hindi", "Tamil", "Bengali", "Marathi"]

header("Profile", "Administrative details the hospital holds for you.")

profile = fetch(lambda: client().profile(token()))
if profile is None:
    st.stop()

current_dob = profile.get("date_of_birth")
languages = LANGUAGES if profile.get("preferred_language") in LANGUAGES else (
    [profile.get("preferred_language") or "English", *LANGUAGES]
)

with st.form("profile"):
    dob = st.date_input(
        "Date of birth",
        value=date.fromisoformat(current_dob) if current_dob else None,
        min_value=date(1900, 1, 1),
        format="YYYY-MM-DD",
    )
    phone = st.text_input("Phone", value=profile.get("phone") or "")
    language = st.selectbox(
        "Preferred language",
        languages,
        index=languages.index(profile.get("preferred_language") or "English"),
    )
    emergency = st.text_input(
        "Emergency contact", value=profile.get("emergency_contact") or ""
    )
    submitted = st.form_submit_button("Save profile", type="primary")

if submitted:
    changes = {
        "date_of_birth": dob.isoformat() if dob else None,
        "phone": phone or None,
        "preferred_language": language,
        "emergency_contact": emergency or None,
    }
    ok, result = act(lambda: client().update_profile(token(), changes))
    if ok:
        st.success("Saved — the change is audit-logged.")
        st.rerun()
    else:
        st.error(result.detail)

st.markdown(
    theme.card(
        theme.facts(
            [
                ("Patient id", profile.get("patient_id")),
                ("Name", profile.get("name")),
                ("Email", profile.get("email")),
            ]
        ),
        kicker="On record",
    ),
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="ac-dim">Contact details are redacted in traces and logs at the '
    "moment they are written, with no override.</p>",
    unsafe_allow_html=True,
)
