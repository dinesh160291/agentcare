"""The per-run trace timeline.

Every LLM request, response, tool call, tool result, guard verdict, validation,
transition and reply, in the order they were written, colour-coded by the agent
that wrote them.

The endpoint returns **whole turns**, not only the events bound to the run — a
turn opens before its run exists, so the inbound event and the safety screen
carry a null run id by design, and a timeline that started after them would
begin in the middle of the story.
"""

from __future__ import annotations

import json

import streamlit as st

from ui import theme
from ui.shell import client, fetch, header, token

KINDS = [
    "all",
    "inbound",
    "guard_verdict",
    "llm_request",
    "llm_response",
    "llm_error",
    "validation",
    "tool_call",
    "tool_result",
    "transition",
    "outbound",
]

header("Trace timeline", "What happened inside a run, event by event.")

runs = fetch(lambda: client().queue(token()), default=[]) or []
if not runs:
    theme.empty("No runs to inspect yet.")
    st.stop()

labels = {
    f"#{r['run_id']} · {r['status'].replace('_', ' ')} · "
    f"{(r.get('request_text') or 'no request text')[:48]}": r["run_id"]
    for r in runs
}
preselected = st.session_state.get("trace_run_id")
options = list(labels)
default_index = next(
    (i for i, label in enumerate(options) if labels[label] == preselected), 0
)

chosen_label = st.selectbox("Run", options, index=default_index)
run_id = labels[chosen_label]
st.session_state["trace_run_id"] = run_id
kind = st.selectbox("Event type", KINDS, index=0)

events = fetch(lambda: client().trace(token(), run_id), default=[]) or []
if kind != "all":
    events = [e for e in events if e["event_type"] == kind]

if not events:
    theme.empty("No events of that kind in this run.")
    st.stop()

turns = sorted({e["turn_id"] for e in events})
st.markdown(
    f'<p class="ac-dim">{len(events)} event(s) across {len(turns)} turn(s). '
    "Grammar: inbound → guard verdict → [LLM request → response → validation → "
    "tool call → tool result]* → transition → outbound. Scheduler activity lives "
    "in the audit ledger, not here.</p>",
    unsafe_allow_html=True,
)

current_turn = None
for event in events:
    if event["turn_id"] != current_turn:
        current_turn = event["turn_id"]
        st.markdown(
            f'<p class="ac-who" style="margin-top:14px;">turn '
            f'{theme.esc(current_turn[:12])}</p>',
            unsafe_allow_html=True,
        )

    author = event.get("agent_name") or event.get("author") or "code"

    st.markdown(
        f'<div style="display:flex;gap:10px;align-items:baseline;padding:2px 0;">'
        f'<span class="ac-dim ac-num" style="width:34px;">#{event["seq"]}</span>'
        f'<span class="ac-tag" style="width:120px;text-align:center;'
        f'border-color:{theme.DIVIDER};color:{theme.TEXT_SECONDARY};">'
        f'{theme.esc(event["event_type"])}</span>'
        f"{theme.agent_tag(author)}"
        f'<span class="ac-dim ac-num">'
        f'{theme.esc(event["created_at"][11:19])}</span></div>',
        unsafe_allow_html=True,
    )
    with st.expander(f"#{event['seq']} payload", expanded=False):
        st.code(json.dumps(event.get("payload") or {}, indent=2), language="json")

st.markdown(
    '<p class="ac-dim">Sensitive fields are redacted at write time, with no '
    "override — what is here is what was persisted.</p>",
    unsafe_allow_html=True,
)
