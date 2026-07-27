"""Uploads, and what is on file.

The client sends the bytes and the declared type and nothing else. It does not
check the size, sniff the type, or sanitise the filename — all three are done on
the other side, against magic bytes rather than the name, and a weaker copy here
would only be a second thing to keep in step. A refusal comes back as a status
code with the backend's own message, and that message is what is shown.
"""

from __future__ import annotations

import streamlit as st

from ui import theme
from ui.shell import act, client, fetch, header, token

#: The types a patient may declare. Mirrors the seed's required-document rules;
#: the backend accepts any non-empty string, so this is a convenience, not a
#: validation.
DECLARED_TYPES = [
    "ECG report",
    "Blood test report",
    "X-ray report",
    "MRI or CT report",
    "Eye prescription",
    "Referral letter",
    "Identification",
    "Other",
]

header("Documents", "Upload a report and say what it is; the agent verifies it.")

with st.form("upload", clear_on_submit=True):
    declared = st.selectbox("Document type", DECLARED_TYPES)
    upload = st.file_uploader("File", type=["pdf", "png", "jpg", "jpeg"])
    submitted = st.form_submit_button("Upload", type="primary")

if submitted:
    if upload is None:
        st.warning("Choose a file first.")
    else:
        ok, result = act(
            lambda: client().upload_document(
                token(),
                filename=upload.name,
                content=upload.getvalue(),
                declared_type=declared,
                content_type=upload.type or "application/octet-stream",
            )
        )
        if ok:
            st.success(result.get("message") or "Document received.")
        else:
            st.error(result.detail)

documents = fetch(lambda: client().documents(token()), default=[]) or []

if not documents:
    theme.empty("Nothing on file yet.")
else:
    for document in documents:
        rows = [
            ("Declared", document.get("declared_type")),
            ("Detected", document.get("detected_type")),
            ("Filed as", document.get("document_type")),
            ("Dated", document.get("document_date")),
            ("Checksum", (document.get("checksum") or "")[:16] + "…"),
        ]
        body = (
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<span class="ac-num" style="font-weight:600;">'
            f'#{theme.esc(document["document_id"])}</span>'
            f'<span>{theme.esc(document.get("original_filename") or "document")}</span>'
            f'{theme.tag(document["status"])}</div>' + theme.facts(rows)
        )
        if document["status"] == "flagged":
            body += (
                '<p class="ac-dim" style="margin:8px 0 0;color:#8a6a25;">'
                "⚠ A member of staff is reviewing this one. It does not count "
                "towards your required documents until they do.</p>"
            )
        st.markdown(theme.card(body), unsafe_allow_html=True)

st.markdown(
    '<p class="ac-dim">Content is checked against the type you declared (PDF text '
    "only — images are taken at their declared type, there is no OCR). An exact "
    "duplicate of a file you already sent is rejected by checksum.</p>",
    unsafe_allow_html=True,
)
