"""Minimal, deterministic PDF writer for synthetic seed documents.

Hand-rolled rather than pulled from a library for two reasons: the PDF
libraries that *write* files are heavy additions for a handful of fixtures,
and — more importantly — the output here is byte-for-byte deterministic. No
creation timestamp, no producer string, no object ordering that varies between
runs. That matters because seed documents are stored with a checksum, and the
duplicate-detection tests assert on exact checksum values.

The files are real PDFs with a real text layer, so Phase 5's pypdf extraction
has something genuine to read.
"""

from __future__ import annotations

# A PDF cross-reference entry is exactly 20 bytes: 10-digit offset, space,
# 5-digit generation, space, type character, space, newline.
_XREF_ENTRY = "{offset:010d} {generation:05d} {kind} \n"


def _escape(text: str) -> str:
    """Escape the three characters that are special inside a PDF string."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(title: str, lines: list[str]) -> bytes:
    """Build a one-page PDF containing ``title`` and ``lines`` as text.

    Returns identical bytes for identical input, every time.
    """
    content_parts = ["BT", "/F1 16 Tf", "72 720 Td", f"({_escape(title)}) Tj", "/F1 11 Tf"]
    for line in lines:
        content_parts.append("0 -20 Td")
        content_parts.append(f"({_escape(line)}) Tj")
    content_parts.append("ET")
    stream = "\n".join(content_parts).encode("latin-1")

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode("ascii")
        out += body
        out += b"\nendobj\n"

    xref_offset = len(out)
    count = len(objects) + 1
    out += f"xref\n0 {count}\n".encode("ascii")
    out += _XREF_ENTRY.format(offset=0, generation=65535, kind="f").encode("ascii")
    for offset in offsets:
        out += _XREF_ENTRY.format(offset=offset, generation=0, kind="n").encode("ascii")

    out += f"trailer\n<< /Size {count} /Root 1 0 R >>\n".encode("ascii")
    out += f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    return bytes(out)
