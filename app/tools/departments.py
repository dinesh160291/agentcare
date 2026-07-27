"""Department resolution and validation.

Two jobs, deliberately separate:

:func:`resolve_department`
    Reads the patient's words against the ``DepartmentSynonym`` table and
    *proposes*. It may find nothing, one department, or several — and several
    is reported as ambiguity rather than resolved by a tiebreak, because a
    silent tiebreak puts a patient in the wrong queue with nobody aware a
    choice was made. Ambiguity is what feeds the staff-review path.

:func:`validate_department`
    Checks a name the model produced against the ``Department`` table. This is
    the "code disposes" half: a department the model invented does not become
    real by being spelled confidently, and "Cardiovascular Medicine" is exactly
    the kind of plausible, well-formed, non-existent answer a model produces.

Matching is on word boundaries. Substring matching would route "an appointment
next year" to ENT, because "ear" sits inside "year".
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy.orm import Session

from app.models import Department, DepartmentSynonym


def _serialise(department: Department) -> dict[str, Any]:
    return {
        "id": department.id,
        "name": department.name,
        "description": department.description,
        # Whether the department is open. Carried on every department dict
        # rather than only the staff-facing one: a second shape is a second
        # thing to keep in step, and a consumer that has to know which flavour
        # it is holding will eventually guess wrong.
        "active": department.active,
    }


def _result(
    *,
    text: str,
    status: str,
    department: Department | None = None,
    candidates: list[Department] | None = None,
    matched_terms: list[str] | None = None,
) -> dict[str, Any]:
    """Shape-stable result: the same keys whatever the outcome."""
    return {
        "text": text,
        "status": status,
        "department": _serialise(department) if department else None,
        "candidates": [_serialise(d) for d in (candidates or [])],
        "matched_terms": sorted(matched_terms or []),
    }


def list_departments(
    session: Session, *, active_only: bool = True
) -> list[dict[str, Any]]:
    """Departments, name-ordered so output is stable.

    Active-only by default, because the agents' question is "where can this be
    booked" and a closed department is not an answer to it.

    ``active_only=False`` is for the staff console, and it is not a nicety: a
    closed department that disappears from the only page that can re-open it is
    a one-way door. The listing an operator manages capacity with has to
    include the capacity they switched off.
    """
    query = session.query(Department)
    if active_only:
        query = query.filter(Department.active.is_(True))
    return [_serialise(d) for d in query.order_by(Department.name).all()]


def resolve_department(session: Session, text: str) -> dict[str, Any]:
    """Propose a department from free text.

    ``status`` is one of ``resolved`` (exactly one department matched),
    ``ambiguous`` (several did — candidates listed, none chosen), or
    ``unsupported`` (none did).
    """
    haystack = (text or "").strip().lower()
    if not haystack:
        return _result(text=text or "", status="unsupported")

    matches: dict[int, set[str]] = {}

    # Department names first — an explicit name is the strongest signal there is.
    for department in session.query(Department).filter(Department.active.is_(True)).all():
        if re.search(rf"\b{re.escape(department.name.lower())}\b", haystack):
            matches.setdefault(department.id, set()).add(department.name.lower())

    synonyms = (
        session.query(DepartmentSynonym)
        .join(Department, Department.id == DepartmentSynonym.department_id)
        .filter(Department.active.is_(True))
        .all()
    )
    for synonym in synonyms:
        if re.search(rf"\b{re.escape(synonym.term)}\b", haystack):
            matches.setdefault(synonym.department_id, set()).add(synonym.term)

    if not matches:
        return _result(text=text, status="unsupported")

    all_terms = sorted({term for terms in matches.values() for term in terms})

    if len(matches) == 1:
        department_id = next(iter(matches))
        return _result(
            text=text,
            status="resolved",
            department=session.get(Department, department_id),
            matched_terms=all_terms,
        )

    # Several departments matched. Ordered by name so golden diffs are stable.
    candidates = sorted(
        (session.get(Department, department_id) for department_id in matches),
        key=lambda d: d.name,
    )
    return _result(
        text=text,
        status="ambiguous",
        candidates=candidates,
        matched_terms=all_terms,
    )


def validate_department(session: Session, name: str) -> dict[str, Any]:
    """Check a model-supplied department name against the table.

    Exact (case-insensitive) name match only — no fuzzy matching. Accepting a
    near-miss here would reintroduce, one layer down, the guessing this
    function exists to stop.
    """
    candidate = (name or "").strip()
    if not candidate:
        return {"name": name or "", "valid": False, "department": None}

    department = (
        session.query(Department)
        .filter(Department.name.ilike(candidate), Department.active.is_(True))
        .one_or_none()
    )
    return {
        "name": candidate,
        "valid": department is not None,
        "department": _serialise(department) if department else None,
    }
