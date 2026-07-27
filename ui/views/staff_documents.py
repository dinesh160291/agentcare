"""The document review queue — declared type against detected type.

Three resolutions, because the PRD has three and each means something
different: **accept** (the label stands, the reader was wrong), **reclassify**
(the content is right, the label was wrong), **reject** (neither is usable).
The design reference offered only the first two; collapsing reject into accept
would leave a patient's unusable file counting towards their requirements.

Resolving re-runs the required-documents diff in the same transaction, so a
reclassified X-ray closes the task that was chasing the ECG it now satisfies.
"""

from __future__ import annotations

import streamlit as st

from ui import theme
from ui.shell import act, client, fetch, header, token

TYPES = [
    "ECG report",
    "Blood test report",
    "X-ray report",
    "MRI or CT report",
    "Eye prescription",
    "Referral letter",
    "Identification",
    "Other",
]

header("Document reviews", "Where the content did not match the declared type.")

flagged = fetch(lambda: client().flagged_documents(token()), default=[]) or []

if not flagged:
    theme.empty("No flagged documents.")

for document in flagged:
    document_id = document["document_id"]
    body = (
        f'<div style="display:flex;gap:10px;align-items:baseline;">'
        f'<span class="ac-num" style="font-weight:600;">#{document_id}</span>'
        f'<span>{theme.esc(document.get("original_filename") or "document")}</span>'
        f'<span class="ac-dim">patient {document.get("patient_id")}</span>'
        f'<span style="flex:1;"></span>'
        f'{theme.tag(document["status"])}</div>'
        + theme.facts(
            [
                ("Declared", document.get("declared_type")),
                ("Detected", document.get("detected_type")),
                ("Note", document.get("verification_note")),
            ]
        )
    )
    st.markdown(theme.card(body), unsafe_allow_html=True)

    accept, reclassify, reject = st.columns([1, 2, 1])

    if accept.button(
        f"Accept as {document.get('declared_type')}",
        key=f"accept_{document_id}",
        type="primary",
    ):
        ok, result = act(
            lambda did=document_id: client().resolve_document(
                token(), did, action="accept"
            )
        )
        if ok:
            st.success("Accepted; the declared type stands.")
            st.rerun()
        else:
            st.error(result.detail)

    with reclassify:
        detected = document.get("detected_type")
        options = TYPES if detected in TYPES else [detected, *TYPES] if detected else TYPES
        corrected = st.selectbox(
            "File as",
            options,
            key=f"type_{document_id}",
            label_visibility="collapsed",
            index=options.index(detected) if detected in options else 0,
        )
        if st.button("Reclassify", key=f"reclass_{document_id}"):
            ok, result = act(
                lambda did=document_id, t=corrected: client().resolve_document(
                    token(), did, action="reclassify", corrected_type=t
                )
            )
            if ok:
                st.success(
                    f"Filed as {result.get('document_type')}; requirements re-checked."
                )
                st.rerun()
            else:
                st.error(result.detail)

    if reject.button("Reject", key=f"reject_doc_{document_id}"):
        ok, result = act(
            lambda did=document_id: client().resolve_document(
                token(), did, action="reject"
            )
        )
        if ok:
            st.success("Rejected. It no longer counts towards anything.")
            st.rerun()
        else:
            st.error(result.detail)

st.markdown(
    '<p class="ac-dim">Resolving re-runs the required-documents check for every '
    "appointment this patient holds, so the follow-up task moves with the "
    "decision rather than after it.</p>",
    unsafe_allow_html=True,
)
