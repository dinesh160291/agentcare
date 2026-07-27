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

**A turn is sent across two script runs, not one.** A message used to be posted
inside the same run that read it, so the whole screen sat still for as long as
five agents took, and then everything repainted at once — the sent message
included, which read as a flicker rather than as a send. Now the first run only
records what the patient asked for and reruns; the second paints that message
and a status line *before* the call, so the wait happens under something rather
than under nothing. The sidebar card is rendered last for the same reason: by
then the turn has committed, so it needs no second rerun to catch up.

The typing effect is cosmetic and is honest about it — the reply is complete
before the first word appears. Real token streaming would need SSE through the
API and would split one LLM response across many trace rows, which is the
pairing invariant the trace's well-formedness check exists to hold.
"""

from __future__ import annotations

import re
import time

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


#: Seconds between words of the typing effect. Small enough that a long
#: receipt does not outlast a reader's patience.
TYPING_DELAY = 0.016


def _remember(who: str, text: str, result: dict | None = None) -> None:
    st.session_state.transcript.append({"who": who, "text": text, "result": result})


# --- phase one: record what was asked for, and repaint --------------------


def _queue_message(text: str) -> None:
    """Show the patient's words now; send them on the next script run."""
    _remember("You", text)
    st.session_state.pending = {"kind": "message", "text": text}


def _queue_action(action: str) -> None:
    """A button press. No free text, so nothing here reads language."""
    _remember("You", "✓ Confirm" if action == "confirm" else "✕ Decline")
    st.session_state.pending = {"kind": "action", "action": action}


# --- phase two: send it, under a status the patient can see ---------------


def _send(pending: dict) -> dict | None:
    """Post the queued turn. Returns the entry to render, or ``None``."""
    if pending["kind"] == "action":
        call = lambda: client().send_action(  # noqa: E731
            token(), action=pending["action"], session_id=st.session_state.session_id
        )
    else:
        call = lambda: client().send_message(  # noqa: E731
            token(), message=pending["text"], session_id=st.session_state.session_id
        )

    ok, result = act(call)
    if not ok:
        _remember("AgentCare", f"Something went wrong: {result.detail}")
        return None

    if pending["kind"] == "message":
        st.session_state.session_id = result["session_id"]
    st.session_state.last_run_id = result.get("run_id")
    _remember("AgentCare", result["reply"], result)
    return st.session_state.transcript[-1]


def _typewriter(text: str):
    """Yield a finished reply one word at a time.

    Cosmetic, and deliberately so: the whole reply is already in hand. What
    this buys is that a long receipt arrives at a readable pace instead of
    landing as a wall — not any claim about how it was produced.
    """
    for word in re.findall(r"\S+\s*", text):
        yield word
        time.sleep(TYPING_DELAY)


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


def _render_turn(entry: dict, *, stream: bool = False) -> None:
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

    if stream:
        # Keyed so the stylesheet can give Streamlit's own markdown the same
        # left rule and measure as ``.ac-msg-agent``; without it the reply
        # would visibly change shape the moment the typing finished.
        with st.container(key="ac_stream"):
            st.write_stream(_typewriter(entry["text"]))
    else:
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
        _queue_action("confirm")
        st.rerun()
    if right.button("✕ Decline", key="decline_proposal"):
        _queue_action("decline")
        st.rerun()

    st.markdown(
        '<p class="ac-dim">These buttons are typed actions — nothing interprets '
        'them. Typing works too: an exact "yes" or "no" is read in code, and '
        "anything ambiguous is asked again rather than committed.</p>",
        unsafe_allow_html=True,
    )


# --- the page ------------------------------------------------------------

header("Chat", "Ask for an appointment, a change, an upload, or a status.")

if not st.session_state.transcript:
    theme.empty(
        "No messages yet. Ask for an appointment, a reschedule, an upload, or a "
        "status — the agent plans the steps and asks before committing anything."
    )

for entry in st.session_state.transcript:
    _render_turn(entry)

# Everything above is already on screen — including the message queued by the
# run that led here — so the wait now happens under a status rather than under
# a still page.
# ``.get``, not attribute access: the view-render sweep loads this file on its
# own, without ``ui/app.py``'s initialiser, and a page that only works when
# something else ran first is a page with a hidden prerequisite.
pending = st.session_state.get("pending")
if pending:
    st.session_state.pending = None
    with st.status("AgentCare is working…", expanded=False) as status:
        fresh = _send(pending)
        status.update(label="Done", state="complete")
    if fresh is not None:
        _render_turn(fresh, stream=True)
    else:
        _render_turn(st.session_state.transcript[-1])

_confirmation_controls()

with st.expander("Demo prompts"):
    for index, prompt in enumerate(DEMO_PROMPTS):
        if st.button(prompt, key=f"demo_{index}"):
            _queue_message(prompt)
            st.rerun()

typed = st.chat_input("Describe what you need…")
if typed:
    _queue_message(typed)
    st.rerun()

# Last, not first: by this point the turn above has committed, so the card
# reads the run's new state without needing a second rerun to catch up.
_run_card()

st.markdown(
    '<p class="ac-foot">AgentCare handles administration only — it never '
    "diagnoses or prescribes. Anything urgent is escalated to staff "
    "immediately.</p>",
    unsafe_allow_html=True,
)
