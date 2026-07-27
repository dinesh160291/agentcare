"""The audit ledger: who did what, to which entity, when.

Distinct from the trace. A trace explains one conversation turn; audit explains
one row's history — and it is where anything with no run and no session lands,
including denied access attempts and, once it exists, the scheduler.
"""

from __future__ import annotations

import streamlit as st

from ui import theme
from ui.shell import client, fetch, header, token

#: Common actions, offered as a filter. The API accepts any action string, so
#: this is a shortcut rather than the vocabulary.
COMMON_ACTIONS = [
    "all",
    "access_denied",
    "login_succeeded",
    "login_failed",
    "workflow_transition",
    "staff_decision_refused",
    "escalation_created",
    "escalation_decided",
    "document_uploaded",
    "document_resolved",
    "appointment_booked",
    "appointment_cancelled",
    "appointment_rescheduled",
]

header("Audit log", "Every consequential action, newest first.")

chosen = st.selectbox("Action", COMMON_ACTIONS, index=0)
limit = st.slider("Rows", min_value=25, max_value=500, value=100, step=25)

rows = fetch(
    lambda: client().audit(
        token(), action=None if chosen == "all" else chosen, limit=limit
    ),
    default=[],
) or []

if not rows:
    theme.empty("Nothing recorded under that filter.")
    st.stop()

st.dataframe(
    [
        {
            "id": row["id"],
            "when": row["created_at"][:19].replace("T", " "),
            "actor": row["actor_id"] if row["actor_id"] is not None else row["actor_kind"],
            "action": row["action"],
            "entity": f"{row['entity_type']}#{row['entity_id']}"
            if row["entity_id"] is not None
            else row["entity_type"],
            "details": ", ".join(f"{k}={v}" for k, v in (row["metadata"] or {}).items()),
        }
        for row in rows
    ],
    width="stretch",
    hide_index=True,
)

st.markdown(
    '<p class="ac-dim">Denied access attempts are here too — a probe at another '
    "patient's id returns 404 to the caller and leaves a row that says exactly "
    "what was tried.</p>",
    unsafe_allow_html=True,
)
