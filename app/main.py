"""The FastAPI application.

Thin by construction. This module wires four things and decides nothing:
CORS from config, the domain-exception handlers, the routers, and a health
check. Every rule the API enforces lives below it — in the auth dependencies,
the ownership helpers, the tools, and the orchestrator.

``create_all`` runs at import. On SQLite that is the project's whole migration
story, stated plainly: the schema is the SQLAlchemy models, and the seed script
is the initialization step.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import install_exception_handlers
from app.api.routers import (
    appointments,
    auth,
    documents,
    patients,
    reminders,
    staff,
    workflow,
)
from app.config import get_settings
from app.db import create_all
from app import scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201, ARG001
    """Start the poll job with the process, and stop it with the process.

    In ``lifespan`` rather than at import: the scheduler is a *running
    application's* concern, and a module that starts a background thread merely
    by being imported would start one in every test collection and every
    ``python -c "import app.main"``. ``scheduler.start()`` is itself a no-op
    when configuration disables it, which is how the suite keeps the thread
    off while still exercising this path.
    """
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AgentCare",
        lifespan=lifespan,
        version="0.8.0",
        description=(
            "Agentic patient administration — registration, routing, booking, "
            "documents, reminders, and follow-up. Administrative only: the "
            "system never diagnoses, prescribes, or advises on dosage."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    install_exception_handlers(app)
    app.include_router(auth.router)
    app.include_router(patients.router)
    app.include_router(workflow.router)
    app.include_router(appointments.router)
    app.include_router(documents.router)
    app.include_router(reminders.router)
    app.include_router(staff.router)

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        """Liveness, and which provider the process is actually running."""
        return {"status": "ok", "provider": settings.llm_provider}

    return app


create_all()
app = create_app()
