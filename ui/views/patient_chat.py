"""The patient's chat — and the one screen where the confirmation rule shows.

Two things here are load-bearing and easy to get subtly wrong:

**Confirm and Decline post to a different endpoint from the message box.** They
are typed actions (``/workflow/actions``), not a message that happens to say
"confirm". If they posted text, the whole guarantee — that a commitment needs a
click or an exact token and nothing else — would depend on the wording this
file chose.

**The buttons render with the first proposal**, not after the patient has
already been misunderstood once. That is the PRD's reading order: buttons, then
exact tokens, then the model's decline-or-re-ask.

The transcript in ``session_state`` is a display buffer for this browser tab.
The conversation itself lives in the backend's session store and survives a
restart — which is why the run card beside it is fetched, not remembered.
"""

from __future__ import annotations

import streamlit as st

from ui import theme
from ui.shell import act, client, fetch, header, token

DEMO_PROMPTS = [
    "I need a cardiology appointment next week",
    "book an appointment, my kid has ear pain",
    "please reschedule my appointment to next week",
    "I want to cancel my appointment",
    "what documents do I have on file?",
    "actually never mind, forget it",
]

#: Statuses where the patient is being asked to confirm something.
AWAITING = "pending_confirmation"


def _remember(who: str, text: str, result: dict | None = None) -> None:
    st.session_state.transcript.append({"who": who, "text": text, "result": result})


def _send_message(text: str) -> None:
    _remember("You", text)
    ok, result = act(
        lambda: client().send_message(
            token(), message=text, session_id=st.session_state.session_id
        )
    )
    if not ok:
        _remember("AgentCare", f"Something went wrong: {result.detail}")
        return
    st.session_state.session_id = result["session_id"]
    st.session_state.last_run_id = result.get("run_id")
    _remember("AgentCare", result["reply"], result)


def _send_action(action: str) -> None:
    """A button press. No free text, so nothing here reads language."""
    ok, result = act(
        lambda: client().send_action(
            token(), action=action, session_id=st.session_state.session_id
        )
    )
    if not ok:
        _remember("AgentCare", f"Something went wrong: {result.detail}")
        return
    _remember("You", "✓ Confirm" if action == "confirm" else "✕ Decline")
    st.session_state.last_run_id = result.get("run_id")
    _remember("AgentCare", result["reply"], result)


def _run_card() -> None:
    """What the backend says the live request is. Fetched every rerun."""
    runs = fetch(lambda: client().runs(token()), default=[]) or []
    live = [r for r in runs if r["status"] not in
            ("completed", "cancelled", "rejected", "failed", "escalated")]

    with st.sidebar:
        st.markdown("#### Active workflow")
        if not live:
            theme.empty("None — start a request in chat.")
        else:
            run = live[0]
            body = theme.facts(
                [
                    ("Run", f"#{run['run_id']}"),
                    ("State", run["status"].replace("_", " ")),
                    ("Department", run.get("department_name")),
                    ("Plan", " → ".join(run.get("plan") or []) or None),
                    ("Done", " → ".join(run.get("completed_steps") or []) or None),
                ]
            )
            st.markdown(theme.card(body), unsafe_allow_html=True)

        if len(runs) > len(live):
            st.markdown("###### Recent requests")
            for run in runs[: 5]:
                st.markdown(
                    f'<div style="display:flex;gap:6px;align-items:center;'
                    f'font-size:12px;padding:1px 0;">'
                    f'<span class="ac-num">#{run["run_id"]}</span>'
                    f'{theme.tag(run["status"])}</div>',
                    unsafe_allow_html=True,
                )


def _render_turn(entry: dict) -> None:
    who = entry["who"]
    result = entry.get("result") or {}
    is_agent = who == "AgentCare"
    css = "ac-msg-agent" if is_agent else "ac-msg-user"

    st.markdown(f'<p class="ac-who">{theme.esc(who)}</p>', unsafe_allow_html=True)

    if result.get("status") == "escalated":
        st.markdown(
            '<div class="ac-alarm">⚠ This request has been escalated to a member '
            "of staff. If this is an emergency, seek urgent care now.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<p class="ac-msg {css}">{theme.esc(entry["text"])}</p>',
        unsafe_allow_html=True,
    )

    if is_agent and result:
        bits = []
        if result.get("run_id"):
            bits.append(f"run #{result['run_id']}")
        if result.get("status"):
            bits.append(result["status"].replace("_", " "))
        if result.get("author"):
            bits.append(f"author: {result['author']}")
        if bits:
            st.markdown(
                f'<p class="ac-dim ac-num" style="margin:3px 0 0;">'
                f'{theme.esc(" · ".join(bits))}</p>',
                unsafe_allow_html=True,
            )


def _confirmation_controls() -> None:
    """Rendered whenever the backend says a proposal is outstanding.

    The condition is the run's **status**, read from the API — not something
    this file inferred from the wording of the reply.
    """
    transcript = st.session_state.transcript
    last = next(
        (e for e in reversed(transcript) if e["who"] == "AgentCare" and e.get("result")),
        None,
    )
    if not last or (last["result"] or {}).get("status") != AWAITING:
        return

    run_id = last["result"].get("run_id")
    run = fetch(lambda: client().run(token(), run_id), default={}) if run_id else {}

    body = theme.facts(
        [
            ("Action", (run.get("proposed_action") or "").replace("_", " ")),
            ("Department", run.get("department_name")),
            ("Run", f"#{run_id}" if run_id else None),
        ]
    )
    st.markdown(
        theme.card(body, kicker="Waiting for your confirmation"),
        unsafe_allow_html=True,
    )

    left, right, _ = st.columns([1, 1, 4])
    if left.button("✓ Confirm", type="primary", key="confirm_proposal"):
        _send_action("confirm")
        st.rerun()
    if right.button("✕ Decline", key="decline_proposal"):
        _send_action("decline")
        st.rerun()

    st.markdown(
        '<p class="ac-dim">These buttons are typed actions — nothing interprets '
        'them. Typing works too: an exact "yes" or "no" is read in code, and '
        "anything ambiguous is asked again rather than committed.</p>",
        unsafe_allow_html=True,
    )


# --- the page ------------------------------------------------------------

header("Chat", "Ask for an appointment, a change, an upload, or a status.")
_run_card()

if not st.session_state.transcript:
    theme.empty(
        "No messages yet. Ask for an appointment, a reschedule, an upload, or a "
        "status — the agent plans the steps and asks before committing anything."
    )

for entry in st.session_state.transcript:
    _render_turn(entry)

_confirmation_controls()

with st.expander("Demo prompts"):
    for index, prompt in enumerate(DEMO_PROMPTS):
        if st.button(prompt, key=f"demo_{index}"):
            _send_message(prompt)
            st.rerun()

typed = st.chat_input("Describe what you need…")
if typed:
    _send_message(typed)
    st.rerun()

st.markdown(
    '<p class="ac-foot">AgentCare handles administration only — it never '
    "diagnoses or prescribes. Anything urgent is escalated to staff "
    "immediately.</p>",
    unsafe_allow_html=True,
)
