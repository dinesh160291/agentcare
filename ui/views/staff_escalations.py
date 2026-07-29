"""Escalations and paused workflows — two queues that must never merge.

**A safety escalation is acknowledged and resolved. It is never approved.** The
backend enforces that, and this page keeps the vocabulary apart on screen too:
the two lists are separate, and neither offers the other's verbs. Nothing here
should ever be able to put the words "approved" and "chest pain" on one row.

Approvals are lazy-continue: the decision flips the state and notifies the
patient, and the run advances on their next message. So a successful approval
shows a state change, not a conversation.
"""

from __future__ import annotations

import streamlit as st

from ui import theme
from ui.shell import act, client, fetch, header, token

def _messages(escalation: dict) -> str:
    """Every message that triggered this case, oldest first.

    Not just the latest one. Dedup is right — five triggers are one case — but
    it used to *overwrite*, so a queue item could say a patient had been
    escalated three times and show only the third thing they said. The message
    that opened the case is usually the one a human needs.
    """
    messages = escalation.get("messages") or []
    if not messages:
        single = escalation.get("latest_message")
        messages = [single] if single else []
    if not messages:
        return ""
    # ``esc`` is not optional here: these are the patient's own words going
    # into markup rendered with ``unsafe_allow_html``.
    rows = "".join(
        f'<li style="margin:2px 0;">{theme.esc(message)}</li>' for message in messages
    )
    return (
        '<div class="ac-dim" style="margin-top:8px;">Messages</div>'
        f'<ol style="margin:4px 0 0 18px;padding:0;">{rows}</ol>'
    )


header("Escalations & reviews", "Paused workflows, and everything awaiting a human.")

escalations = fetch(lambda: client().escalations(token()), default=[]) or []
departments = fetch(lambda: client().departments(token()), default=[]) or []
department_names = [d["name"] for d in departments if d.get("active")]

safety = [e for e in escalations if e.get("kind") == "safety"]
reviews = [e for e in escalations if e.get("kind") != "safety"]

approvals_tab, safety_tab = st.tabs(
    [f"Approvals ({len(reviews)})", f"Safety escalations ({len(safety)})"]
)

with approvals_tab:
    st.markdown(
        '<p class="ac-dim">Staff decisions are typed actions applied by code — no '
        "model reads or rephrases them. Approving flips the state and notifies "
        "the patient; the run advances when they next write.</p>",
        unsafe_allow_html=True,
    )
    if not reviews:
        theme.empty("No paused workflows awaiting review.")

    for escalation in reviews:
        run_id = escalation["workflow_run_id"]
        body = (
            f'<div style="display:flex;gap:10px;align-items:baseline;">'
            f'<span class="ac-num" style="font-weight:600;">run #{run_id}</span>'
            f'{theme.tag(escalation["kind"])}{theme.tag(escalation["status"])}</div>'
            + theme.facts(
                [
                    ("Reason", escalation.get("reason")),
                    ("Triggers", escalation.get("occurrence_count")),
                ]
            )
            + _messages(escalation)
        )
        st.markdown(theme.card(body), unsafe_allow_html=True)

        approve, redirect_col, reject = st.columns([1, 2, 2])

        if approve.button("Approve", key=f"approve_{run_id}", type="primary"):
            ok, result = act(
                lambda rid=run_id: client().decide(token(), rid, action="approve")
            )
            if ok:
                st.success(f"Routed to {result.get('department_name')}.")
                st.rerun()
            else:
                st.error(result.detail)

        with redirect_col:
            target = st.selectbox(
                "Redirect to",
                department_names,
                key=f"dept_{run_id}",
                label_visibility="collapsed",
                index=None,
                placeholder="Redirect to…",
            )
            if st.button("Redirect", key=f"redirect_{run_id}", disabled=not target):
                ok, result = act(
                    lambda rid=run_id, name=target: client().decide(
                        token(), rid, action="redirect", department_name=name
                    )
                )
                if ok:
                    st.success(f"Redirected to {result.get('department_name')}.")
                    st.rerun()
                else:
                    st.error(result.detail)

        with reject:
            note = st.text_input(
                "Reason",
                key=f"note_{run_id}",
                label_visibility="collapsed",
                placeholder="Reason for closing…",
            )
            if st.button("Reject", key=f"reject_{run_id}"):
                ok, result = act(
                    lambda rid=run_id, n=note: client().decide(
                        token(), rid, action="reject", note=n
                    )
                )
                if ok:
                    st.success("Closed, and the patient has been notified.")
                    st.rerun()
                else:
                    st.error(result.detail)

with safety_tab:
    st.markdown(
        '<p class="ac-dim">Acknowledged and resolved — never approved. Repeat '
        "triggers attach to the one open record, so five frightened messages are "
        "one queue item with five recorded triggers.</p>",
        unsafe_allow_html=True,
    )
    if not safety:
        theme.empty("No open safety escalations.")

    for escalation in safety:
        escalation_id = escalation["escalation_id"]
        body = (
            f'<div style="display:flex;gap:10px;align-items:baseline;">'
            f'<span class="ac-num" style="font-weight:600;">'
            f'#{escalation_id}</span>{theme.tag(escalation["status"])}'
            f'<span class="ac-dim">run {escalation["workflow_run_id"]} · '
            f'{escalation.get("occurrence_count")} trigger(s)</span></div>'
            + theme.facts(
                [
                    ("Reason", escalation.get("reason")),
                ]
            )
            + _messages(escalation)
        )
        st.markdown(
            f'<div class="ac-card" style="border-left:3px solid #a33a2a;">{body}</div>',
            unsafe_allow_html=True,
        )

        acknowledge, resolve, _ = st.columns([1, 1, 4])
        if acknowledge.button("Acknowledge", key=f"ack_{escalation_id}", type="primary"):
            ok, result = act(
                lambda eid=escalation_id: client().resolve_escalation(
                    token(), eid, status="acknowledged"
                )
            )
            if ok:
                st.rerun()
            else:
                st.error(result.detail)
        if resolve.button("Resolve", key=f"resolve_{escalation_id}"):
            ok, result = act(
                lambda eid=escalation_id: client().resolve_escalation(
                    token(), eid, status="resolved"
                )
            )
            if ok:
                st.rerun()
            else:
                st.error(result.detail)
