"""Departments, doctors, and adding slots.

Creation comes from the seed script — this is capacity management, not an
admin CRUD. Closing a department makes routing to it unsupported, so requests
for it pause for review instead of booking into a service that is shut.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import streamlit as st

from ui import theme
from ui.shell import act, client, fetch, header, token

#: The clinic day the "add a day of slots" shortcut lays down.
SLOT_TIMES = [time(9, 0), time(9, 30), time(10, 0), time(10, 30), time(11, 0),
              time(14, 0), time(14, 30), time(15, 0), time(15, 30), time(16, 0)]

header("Capacity", "Open or close a service, and add times to a doctor's diary.")

departments = fetch(lambda: client().departments(token()), default=[]) or []

st.markdown("#### Departments")
st.markdown(
    '<p class="ac-dim">Closing one makes routing to it unsupported — requests '
    "pause for staff review rather than booking.</p>",
    unsafe_allow_html=True,
)

for department in departments:
    body = (
        f'<div style="display:flex;gap:10px;align-items:baseline;">'
        f'<span style="font-weight:600;">{theme.esc(department["name"])}</span>'
        f'{theme.tag("open" if department["active"] else "closed", state="resolved" if department["active"] else "cancelled")}'
        f'<span style="flex:1;"></span>'
        f'<span class="ac-dim">{theme.esc(department.get("description") or "")}</span>'
        f"</div>"
    )
    st.markdown(theme.card(body, blueprint=False), unsafe_allow_html=True)
    label = "Close" if department["active"] else "Re-open"
    if st.button(label, key=f"dept_{department['id']}"):
        ok, result = act(
            lambda did=department["id"], a=not department["active"]:
            client().set_department_active(token(), did, active=a)
        )
        if ok:
            st.rerun()
        else:
            st.error(result.detail)

st.markdown("#### Add slots for a doctor")
st.markdown(
    '<p class="ac-dim">Times are parsed before anything is created, so one bad '
    "entry adds nothing. A time the doctor already has is skipped, not "
    "duplicated.</p>",
    unsafe_allow_html=True,
)

with st.form("slots"):
    doctor_id = st.number_input("Doctor id", min_value=1, step=1, value=1)
    day = st.date_input("Day", value=date.today() + timedelta(days=14))
    duration = st.number_input(
        "Slot length (minutes)", min_value=5, max_value=240, step=5, value=30
    )
    submitted = st.form_submit_button("Add a day of slots", type="primary")

if submitted:
    starts = [datetime.combine(day, slot).isoformat() for slot in SLOT_TIMES]
    ok, result = act(
        lambda: client().add_slots(
            token(),
            int(doctor_id),
            start_times=starts,
            duration_minutes=int(duration),
        )
    )
    if ok:
        st.success(
            f"Created {result['created']} slot(s); skipped {result['skipped']} "
            "already on the diary."
        )
    else:
        st.error(result.detail)

st.markdown("#### Take a doctor off the roster")
with st.form("doctor"):
    target = st.number_input("Doctor id", min_value=1, step=1, value=1, key="doc_id")
    active = st.checkbox("On the roster", value=True)
    toggled = st.form_submit_button("Apply")

if toggled:
    ok, result = act(
        lambda: client().set_doctor_active(token(), int(target), active=active)
    )
    if ok:
        st.success(
            f"{result['name']} is {'on' if result['active'] else 'off'} the roster."
        )
    else:
        st.error(result.detail)
