"""Domain exception → HTTP status, in one place.

Tools and services raise :mod:`app.errors` exceptions and never import FastAPI;
this module is the only translation between the two vocabularies. Keeping it
here rather than in each router is what stops one endpoint answering 403 where
its neighbour answers 404 — and on the ownership path that difference is not
cosmetic, it is an id oracle.

**One thing these handlers cannot do.** FastAPI unwinds the dependency stack
before an exception handler runs, so the request's session is already closed by
the time we get here: a handler cannot commit. That matters for exactly one
callee — ``apply_staff_decision`` writes its own audit and trace rows *before*
raising ``ValidationFailed``, and those rows are the interesting part of a
refusal. That router therefore catches the exception itself, commits, and
re-raises as an ``HTTPException``. The handler below is the safety net for
every other raiser, not the mechanism for that one.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.schemas import TurnOut
from app.errors import (
    AgentCareError,
    AuthenticationError,
    PermissionDenied,
    ProviderError,
    RateLimited,
    RecordNotFound,
    ValidationFailed,
)

#: The whole mapping. A domain error not listed here is a programming error and
#: is allowed to surface as a 500 — see the module docstring in ``app.errors``
#: for why ``RecordNotFound`` covers ownership failures rather than 403.
STATUS_FOR: dict[type[AgentCareError], int] = {
    AuthenticationError: 401,
    PermissionDenied: 403,
    RecordNotFound: 404,
    ValidationFailed: 422,
    ProviderError: 503,
}


def _json(status_code: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail}, headers=headers)


def install_exception_handlers(app: FastAPI) -> None:
    """Attach one handler per domain exception class."""

    @app.exception_handler(AuthenticationError)
    async def _authentication(_request: Request, exc: AuthenticationError) -> JSONResponse:
        return _json(401, str(exc), headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(PermissionDenied)
    async def _permission(_request: Request, exc: PermissionDenied) -> JSONResponse:
        return _json(403, str(exc))

    @app.exception_handler(RecordNotFound)
    async def _not_found(_request: Request, exc: RecordNotFound) -> JSONResponse:
        # Deliberately says only what was not found, never why. "You do not own
        # this" and "no such row" must be indistinguishable from out here.
        return _json(404, str(exc))

    @app.exception_handler(ValidationFailed)
    async def _validation(_request: Request, exc: ValidationFailed) -> JSONResponse:
        return _json(422, str(exc))

    @app.exception_handler(ProviderError)
    async def _provider(_request: Request, exc: ProviderError) -> JSONResponse:
        """The model provider failed. Serve the reply the trace already recorded.

        A turn that dies inside the provider is not silent: the envelope writes
        a template outbound event and **commits** it before re-raising, so the
        trace says the patient was let down gently. Without this handler the
        patient got a bare 500 instead — one turn with two different accounts
        of what was said, and the trace vouching for words nobody read.

        Still an error status, because the request genuinely was not carried
        out. The body is a ``TurnOut`` so a client can render the reply and
        keep the turn id, plus a ``detail`` key because that is where every
        other refusal on this API puts its message and the client reads it
        there. Attaching ``turn`` is the envelope's job; a provider raising
        outside a turn has none, and gets the plain refusal.
        """
        retry_after = getattr(exc, "retry_after", None) if isinstance(exc, RateLimited) else None
        headers = {"Retry-After": str(int(retry_after))} if retry_after else None

        if exc.turn is None:
            return _json(503, "The assistant is unavailable right now.", headers=headers)

        body = TurnOut.of(exc.turn).model_dump()
        body["detail"] = exc.turn.reply
        return JSONResponse(status_code=503, content=body, headers=headers)


__all__ = ["STATUS_FOR", "install_exception_handlers"]
