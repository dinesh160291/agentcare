"""Notifications, reminders, and open follow-up tasks.

All three are rows the system derived from something else — a booking, a staff
decision, a document shortfall — so nothing on this page creates one. The only
write is marking a notification read, which is the reader's own fact about
their own row.
"""

from __future__ import annotations

import streamlit as st

from ui import theme
from ui.shell import act, client, fetch, header, token

header("Reminders & notifications", "What the system has told you, and what is due.")

notifications = fetch(lambda: client().notifications(token()), default=[]) or []
reminders = fetch(
    lambda: client().reminders(token(), include_inactive=True), default=[]
) or []
tasks = fetch(lambda: client().tasks(token()), default=[]) or []

st.markdown("#### In-app notifications")
if not notifications:
    theme.empty("Nothing yet. Staff decisions and fired reminders land here.")
for note in notifications:
    body = (
        f'<div style="display:flex;gap:10px;align-items:baseline;">'
        f'<span style="font-weight:600;">{theme.esc(note["title"])}</span>'
        f'{theme.tag("unread" if not note["read"] else "read", state="pending" if not note["read"] else "resolved")}'
        f'<span style="flex:1;"></span>'
        f'<span class="ac-dim ac-num">{theme.esc(note["created_at"][:16].replace("T", " "))}</span>'
        f"</div>"
        + (f'<p class="ac-msg" style="font-size:13.5px;margin-top:4px;">'
           f'{theme.esc(note["body"])}</p>' if note.get("body") else "")
    )
    st.markdown(theme.card(body, blueprint=False), unsafe_allow_html=True)
    if not note["read"]:
        if st.button("Mark read", key=f"read_{note['notification_id']}"):
            ok, result = act(
                lambda nid=note["notification_id"]: client().mark_notification_read(
                    token(), nid
                )
            )
            if ok:
                st.rerun()
            else:
                st.error(result.detail)

st.markdown("#### Reminders")
st.markdown(
    '<p class="ac-dim">Scheduled automatically when you book, and delivered by a '
    "background job — not by this page.</p>",
    unsafe_allow_html=True,
)
if not reminders:
    theme.empty("No reminders — they are scheduled automatically when you book.")
for reminder in reminders:
    body = (
        f'<div style="display:flex;gap:10px;align-items:baseline;">'
        f'<span style="flex:1;">{theme.esc(reminder["message"])}</span>'
        f'{theme.tag(reminder["status"])}</div>'
        f'<p class="ac-dim ac-num" style="margin:4px 0 0;">due '
        f'{theme.esc(reminder["scheduled_at"][:16].replace("T", " "))}'
        f' · attempts {theme.esc(reminder["attempts"])}</p>'
    )
    st.markdown(theme.card(body, blueprint=False), unsafe_allow_html=True)

st.markdown("#### Follow-up tasks")
if not tasks:
    theme.empty("No open tasks.")
for task in tasks:
    items = (task.get("details") or {}).get("missing") or []
    body = (
        f'<div style="display:flex;gap:10px;align-items:baseline;">'
        f'<span style="font-weight:600;">'
        f'{theme.esc(task["task_type"].replace("_", " ").title())}</span>'
        f'{theme.tag(task["status"])}</div>'
        + "".join(
            f'<p class="ac-dim" style="margin:3px 0 0;">▸ {theme.esc(item)}</p>'
            for item in items
        )
    )
    st.markdown(theme.card(body), unsafe_allow_html=True)
