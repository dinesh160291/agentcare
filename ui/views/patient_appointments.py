"""Every appointment on file, and the two ways to change one.

**Reschedule and Cancel are chat-first.** They compose the request and send it
through ``/workflow/messages`` — they are not direct mutations, because there
is no direct mutation to make: booking, moving and cancelling all go through
the workflow, which names exactly what it is about to do and waits to be told
to do it. A button here that cancelled outright would be a second road to the
same state change with the confirmation step missing from it.
"""

from __future__ import annotations

import streamlit as st

from ui import theme
from ui.shell import act, client, fetch, header, token

LIVE = ("pending", "confirmed")


def _compose(text: str) -> None:
    """Hand the request to the agent, exactly as if it had been typed."""
    st.session_state.transcript.append({"who": "You", "text": text, "result": None})
    ok, result = act(
        lambda: client().send_message(
            token(), message=text, session_id=st.session_state.session_id
        )
    )
    if ok:
        st.session_state.session_id = result["session_id"]
        st.session_state.last_run_id = result.get("run_id")
        st.session_state.transcript.append(
            {"who": "AgentCare", "text": result["reply"], "result": result}
        )
    else:
        st.session_state.transcript.append(
            {"who": "AgentCare", "text": f"Something went wrong: {result.detail}",
             "result": None}
        )


header("My appointments", "Changes go through the agent, which confirms first.")

appointments = fetch(lambda: client().appointments(token()), default=[]) or []
tasks = fetch(lambda: client().tasks(token()), default=[]) or []

missing_by_appointment: dict[int, list[str]] = {}
for task in tasks:
    if task.get("task_type") == "missing_documents":
        items = (task.get("details") or {}).get("missing") or []
        if task.get("appointment_id") and items:
            missing_by_appointment[task["appointment_id"]] = items

if not appointments:
    theme.empty("No appointments yet — book one in chat.")
else:
    for appointment in appointments:
        when = appointment.get("start")
        rows = [
            ("Department", appointment.get("department_name")),
            ("Doctor", appointment.get("doctor_name")),
            ("When", f"{when[:10]} at {when[11:16]}" if when else "time released"),
            ("Reference", appointment.get("reference_code")),
        ]
        body = (
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<span class="ac-num" style="font-weight:600;">'
            f'#{theme.esc(appointment["appointment_id"])}</span>'
            f'{theme.tag(appointment["status"])}</div>'
            + theme.facts(rows)
        )
        missing = missing_by_appointment.get(appointment["appointment_id"])
        if missing:
            body += (
                f'<p class="ac-dim" style="margin:8px 0 0;color:#8a6a25;">'
                f'⚠ Still needed: {theme.esc(", ".join(missing))}</p>'
            )
        st.markdown(theme.card(body), unsafe_allow_html=True)

        if appointment["status"] in LIVE:
            left, right, _ = st.columns([1, 1, 5])
            key = appointment["appointment_id"]
            label = appointment.get("department_name") or "appointment"
            if left.button("Reschedule", key=f"resched_{key}"):
                _compose(f"please reschedule my {label} appointment to next week")
                st.switch_page("views/patient_chat.py")
            if right.button("Cancel", key=f"cancel_{key}"):
                _compose(f"please cancel my {label} appointment")
                st.switch_page("views/patient_chat.py")

st.markdown(
    '<p class="ac-dim">Nothing on this page changes anything by itself. Both '
    "buttons start a request in chat, and the agent names the exact appointment "
    "before it acts.</p>",
    unsafe_allow_html=True,
)
