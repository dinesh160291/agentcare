"""The requests queue: every run and the state it is in.

Views read terminal states; the machine never keeps a run alive to power a UI.
So withdrawn and superseded runs are here too, collapsed below the live ones —
visible through a filter rather than by leaving a dead request looking active.
"""

from __future__ import annotations

import streamlit as st

from ui import theme
from ui.shell import act, client, fetch, header, token

LIVE_STATES = ("in_progress", "pending_confirmation", "pending_review")
CLOSED_STATES = ("completed", "cancelled", "rejected", "failed", "escalated")
FILTERS = ["all", *LIVE_STATES, *CLOSED_STATES]

header("Requests queue", "Every patient request and where it has got to.")

chosen = st.selectbox("Filter state", FILTERS, index=0)
runs = fetch(
    lambda: client().queue(token(), status=None if chosen == "all" else chosen),
    default=[],
) or []

live = [r for r in runs if r["status"] in LIVE_STATES]
closed = [r for r in runs if r["status"] not in LIVE_STATES]

counts = [
    ("Open", len(live)),
    ("Awaiting review", len([r for r in runs if r["status"] == "pending_review"])),
    ("Awaiting patient", len([r for r in runs if r["status"] == "pending_confirmation"])),
    ("Escalated", len([r for r in runs if r["status"] == "escalated"])),
]
columns = st.columns(len(counts))
for column, (label, value) in zip(columns, counts):
    column.markdown(
        theme.card(
            f'<p class="k">{theme.esc(label)}</p>'
            f'<p style="font-size:28px;margin:0;" class="ac-num">{value}</p>',
            blueprint=True,
        ),
        unsafe_allow_html=True,
    )


def _row(run: dict) -> None:
    body = (
        f'<div style="display:flex;gap:10px;align-items:baseline;">'
        f'<span class="ac-num" style="font-weight:600;">#{run["run_id"]}</span>'
        f'{theme.tag(run["status"])}'
        f'<span class="ac-dim">patient {run["patient_id"]}</span>'
        f'<span style="flex:1;"></span>'
        f'<span class="ac-dim ac-num">'
        f'{theme.esc(run["updated_at"][:16].replace("T", " "))}</span></div>'
        + theme.facts(
            [
                ("Department", run.get("department_name")),
                ("Plan", " → ".join(run.get("plan") or []) or None),
                ("Done", " → ".join(run.get("completed_steps") or []) or None),
                ("Request", (run.get("request_text") or "")[:140] or None),
            ]
        )
    )
    st.markdown(theme.card(body), unsafe_allow_html=True)
    if st.button("Open trace", key=f"trace_{run['run_id']}"):
        st.session_state["trace_run_id"] = run["run_id"]
        st.switch_page("views/staff_trace.py")


if chosen == "all":
    st.markdown("#### Live")
    if not live:
        theme.empty("Nothing in flight.")
    for run in live:
        _row(run)

    with st.expander(f"Closed, withdrawn or superseded ({len(closed)})"):
        if not closed:
            theme.empty("Nothing closed yet.")
        for run in closed:
            _row(run)
else:
    if not runs:
        theme.empty(f"No runs are {chosen.replace('_', ' ')}.")
    for run in runs:
        _row(run)


# --- visits the poll job has closed ---------------------------------------
# The sweep can only see that an end time has passed. Whether the patient
# turned up is not a fact the clock has, so `completed` is a default and this
# is where a person corrects it — the only thing that opens a missed-visit
# follow-up. Both statuses are listed: marking a no-show must not remove the
# row you would use to undo it.

st.markdown("#### Visits closed by the system")

visits = fetch(lambda: client().swept_visits(token()), default=[]) or []

if not visits:
    theme.empty("No visits have been swept yet. They appear once their end time passes.")

for visit in visits:
    missed = visit.get("status") == "missed"
    st.markdown(
        theme.card(
            f'<div style="display:flex;gap:10px;align-items:baseline;">'
            f'<span class="ac-num" style="font-weight:600;">'
            f'{theme.esc(visit.get("reference_code"))}</span>'
            f'{theme.tag(visit.get("status"))}'
            f'<span class="ac-dim">patient {visit.get("patient_id")}</span></div>'
            + theme.facts(
                [
                    ("Department", visit.get("department_name")),
                    ("Doctor", visit.get("doctor_name")),
                    ("When", (visit.get("start") or "")[:16].replace("T", " ") or None),
                ]
            )
        ),
        unsafe_allow_html=True,
    )
    label = "Mark attended" if missed else "Mark missed"
    action = "completed" if missed else "missed"
    if st.button(label, key=f"visit_{visit['appointment_id']}"):
        ok, result = act(
            lambda: client().correct_visit(
                token(), visit["appointment_id"], action=action
            )
        )
        if ok:
            st.rerun()
        else:
            st.error(str(result))
