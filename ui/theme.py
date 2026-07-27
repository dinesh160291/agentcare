"""The look, and the few render helpers that carry it.

Derived from the "Industry" design system the Phase 7 reference was built in:
a light technical ground, one steel accent, square corners, hairline borders,
and blueprint registration marks at the corners of framed objects. Condensed
headings over a plain body face, tabular numerals wherever a number sits in a
column.

**Nothing here talks to the backend or decides anything.** These are string
builders: given values a page already fetched, they return markup. Keeping
them here rather than inline in each page is what stops six pages inventing six
slightly different status pills.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

# --- tokens, lifted from the design system's :root ------------------------

BG = "#f2f2f3"
SURFACE = "#ffffff"
TEXT = "#1d1f20"
TEXT_SECONDARY = "#5b6066"
ACCENT = "#5980a6"
ACCENT_700 = "#3f5f80"
DIVIDER = "#d7d8da"

#: One colour per workflow/appointment/document state, so a status reads the
#: same wherever it appears. Taken from the reference's own scale.
STATE_COLORS: dict[str, str] = {
    "in_progress": "#3a5f8a",
    "pending_confirmation": "#5b4b8a",
    "pending_review": "#8a6a25",
    "escalated": "#a33a2a",
    "completed": "#2f6b4f",
    "cancelled": "#5b6066",
    "rejected": "#a33a2a",
    "failed": "#a33a2a",
    "confirmed": "#3a5f8a",
    "pending": "#5b4b8a",
    "missed": "#a33a2a",
    "verified": "#2f6b4f",
    "flagged": "#8a6a25",
    "pending_verification": "#5b4b8a",
    "open": "#a33a2a",
    "acknowledged": "#5b4b8a",
    "resolved": "#2f6b4f",
    "approved": "#2f6b4f",
    "sent": "#2f6b4f",
}

#: One colour per trace author/agent, for the timeline.
AGENT_COLORS: dict[str, str] = {
    "coordinator": "#3a5f8a",
    "department_routing": "#2f6b6b",
    "appointment": "#5b4b8a",
    "documents": "#2f6b4f",
    "followup": "#8a3a6b",
    "safety_screen": "#a33a2a",
    "orchestrator": "#5b6066",
    "guard": "#a33a2a",
    "template": "#5b6066",
    "llm": "#3a5f8a",
}

CSS = f"""
<style>
  html, body, [class*="css"] {{ font-feature-settings: "tnum" 0; }}
  .stApp {{ background: {BG}; }}

  /* Square everything: this system is a line drawing, not a set of pills. */
  .stButton > button, .stTextInput input, .stTextArea textarea,
  .stSelectbox div[data-baseweb="select"] > div, .stFileUploader section {{
      border-radius: 0 !important;
  }}
  .stButton > button {{
      border: 1px solid {DIVIDER};
      font-weight: 500;
  }}
  .stButton > button:focus-visible {{
      outline: 2px solid {ACCENT}; outline-offset: 2px;
  }}

  h1, h2, h3 {{ letter-spacing: 0.02em; }}

  /* A framed object: hairline border, square, registration marks. */
  .ac-card {{
      position: relative;
      border: 1px solid {DIVIDER};
      background: {SURFACE};
      padding: 14px 16px;
      margin: 6px 0 12px;
  }}
  .ac-card .k {{
      font-size: 10.5px; letter-spacing: 0.08em; text-transform: uppercase;
      color: {TEXT_SECONDARY}; margin: 0 0 6px;
  }}
  .ac-card.blueprint::before, .ac-card.blueprint::after,
  .ac-card.blueprint > i::before, .ac-card.blueprint > i::after {{
      content: ""; position: absolute; width: 7px; height: 7px;
      border-color: {ACCENT}; border-style: solid; border-width: 0;
  }}
  .ac-card.blueprint::before {{ top: -1px; left: -1px; border-top-width: 1px; border-left-width: 1px; }}
  .ac-card.blueprint::after  {{ top: -1px; right: -1px; border-top-width: 1px; border-right-width: 1px; }}
  .ac-card.blueprint > i::before {{ bottom: -1px; left: -1px; border-bottom-width: 1px; border-left-width: 1px; }}
  .ac-card.blueprint > i::after  {{ bottom: -1px; right: -1px; border-bottom-width: 1px; border-right-width: 1px; }}

  .ac-tag {{
      display: inline-block; padding: 1px 7px; font-size: 11.5px;
      border: 1px solid; white-space: nowrap; font-weight: 600;
  }}
  .ac-num {{ font-variant-numeric: tabular-nums; }}
  .ac-dim {{ color: {TEXT_SECONDARY}; font-size: 12.5px; }}

  /* A chat turn. The author label is small caps above the words. */
  .ac-who {{
      font-size: 10.5px; letter-spacing: 0.08em; text-transform: uppercase;
      color: {TEXT_SECONDARY}; margin: 0 0 2px;
  }}
  .ac-msg {{ font-size: 15px; line-height: 1.6; margin: 0; white-space: pre-line; }}
  .ac-msg-agent {{
      border-left: 2px solid {ACCENT}; padding: 2px 0 2px 12px;
  }}
  .ac-msg-user {{
      border-left: 2px solid {DIVIDER}; padding: 2px 0 2px 12px;
  }}
  /* The reply while it is typing. ``st.write_stream`` renders Streamlit's own
     markdown, which this cannot class directly — but ``st.container(key=...)``
     stamps ``st-key-<key>`` on the wrapper, so the finished reply and the
     typing one can be given the same rule and measure. Without this the words
     visibly re-flow the instant the effect ends. */
  .st-key-ac_stream p {{
      font-size: 15px; line-height: 1.6; margin: 0;
      border-left: 2px solid {ACCENT}; padding: 2px 0 2px 12px;
  }}
  .ac-alarm {{
      border: 1px solid #a33a2a; border-left: 3px solid #a33a2a;
      background: #fdf3f2; color: #7a2b20;
      font-size: 13px; font-weight: 600; padding: 7px 10px; margin: 6px 0;
  }}
  .ac-foot {{
      font-size: 11px; color: {TEXT_SECONDARY}; text-align: center; margin-top: 8px;
  }}
  .ac-facts {{
      display: grid; grid-template-columns: auto 1fr; gap: 4px 18px;
      font-size: 14px; margin-top: 8px;
  }}
  .ac-facts .fk {{ color: {TEXT_SECONDARY}; }}
  .ac-facts .fv {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
</style>
"""


def inject() -> None:
    """Put the stylesheet on the page. Idempotent per rerun."""
    st.markdown(CSS, unsafe_allow_html=True)


def esc(value: Any) -> str:
    """Escape anything bound for ``unsafe_allow_html`` markup.

    Every helper below renders values that came from the API — which means
    values a patient typed. Escaping is not politeness: a reply echoing a
    message containing markup would otherwise render it.
    """
    return html.escape("" if value is None else str(value))


def tag(label: Any, *, state: str | None = None) -> str:
    """A status pill, coloured from the shared state scale."""
    key = str(state if state is not None else label or "").lower()
    color = STATE_COLORS.get(key, TEXT_SECONDARY)
    text = esc(str(label or "").replace("_", " "))
    return (
        f'<span class="ac-tag" style="color:{color};border-color:{color}55;'
        f'background:{color}12;">{text}</span>'
    )


def agent_tag(name: Any) -> str:
    key = str(name or "").lower()
    color = AGENT_COLORS.get(key, TEXT_SECONDARY)
    return (
        f'<span class="ac-tag" style="color:{color};border-color:{color}55;'
        f'background:{color}12;">{esc(name)}</span>'
    )


def card(body_html: str, *, kicker: str | None = None, blueprint: bool = True) -> str:
    """A framed object. ``body_html`` is already-escaped markup."""
    marks = "<i></i>" if blueprint else ""
    head = f'<p class="k">{esc(kicker)}</p>' if kicker else ""
    cls = "ac-card blueprint" if blueprint else "ac-card"
    return f'<div class="{cls}">{marks}{head}{body_html}</div>'


def facts(rows: list[tuple[str, Any]]) -> str:
    """The key/value grid a proposal and a receipt are read from."""
    cells = "".join(
        f'<span class="fk">{esc(k)}</span><span class="fv">{esc(v)}</span>'
        for k, v in rows
        if v not in (None, "")
    )
    return f'<div class="ac-facts">{cells}</div>'


def empty(message: str) -> None:
    """The house empty state — a sentence, not a shrug."""
    st.markdown(
        f'<p class="ac-dim" style="margin:6px 0;">{esc(message)}</p>',
        unsafe_allow_html=True,
    )


__all__ = [
    "AGENT_COLORS",
    "STATE_COLORS",
    "agent_tag",
    "card",
    "empty",
    "esc",
    "facts",
    "inject",
    "tag",
]
