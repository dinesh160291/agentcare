# AgentCare

An agentic AI system for hospital **patient administration** — registration, administrative intent detection, department routing, appointment booking, document coordination, confirmations and reminders, and follow-up scheduling.

> **Not a clinical system.** AgentCare performs no diagnosis, prescribes nothing, recommends no dosages, and does not replace a healthcare professional. It routes administrative requests and escalates anything clinical, urgent, or uncertain to a human.

> ⚠️ **Status: day-1 scaffold.** The repository currently contains project configuration, the CI workflow, and the specification. Application code lands over days 1–3. Sections marked _(pending)_ are placeholders and will be filled as each component ships — nothing below claims a feature that isn't built.

---

## Specification

The full design lives in **[`docs/prd-agentcare.md`](docs/prd-agentcare.md)** — user stories, the pinned workflow state machine, agent architecture, safety and confirmation flows, the data model, and the testing strategy. **[`CLAUDE.md`](CLAUDE.md)** holds the working rules and build conventions.

## Stack

| Layer | Choice |
|---|---|
| Backend | Python 3.11 + FastAPI — auth, RBAC, orchestration, tools, persistence |
| Frontend | Streamlit thin client (all calls via a shared `api_client` module) |
| Agents | Google ADK |
| LLM | Pluggable provider: `mock` \| `openai` \| `groq` |
| Database | SQLAlchemy + SQLite file (Postgres for the optional deploy) |
| Scheduler | APScheduler (reminder polling, visit-completion sweep) |

## Setup

Requires Python 3.11 (developed and tested on 3.11.6).

```bash
git clone https://github.com/dinesh160291/agentcare.git
cd agentcare

python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env        # then edit .env
```

**Running with no API key at all:** set `LLM_PROVIDER=mock` in `.env`. The mock provider is a first-class provider, not a test stub — the entire application runs end to end on it, invoking the same tools and writing the same database rows, with replies templated from persisted results. Use it to evaluate the system offline.

_(Run commands, seeding, and demo credentials: pending.)_

## Architecture

_(Pending — agents, tools, and orchestration described here as they ship. See [`docs/prd-agentcare.md`](docs/prd-agentcare.md) for the full design.)_

## Database models and initialization

- **Models** — SQLAlchemy ORM models under `app/models/` _(pending)_
- **Initialization** — schema creation via `create_all` plus `scripts/seed.py`, which loads synthetic departments, doctors, slots, patients, and documents _(pending)_

Alembic migrations are deliberately not used: the schema is created fresh from the models, which is sufficient for a project of this scope.

## Sample data

All seed data is synthetic and obviously fake. No real patient information, and no real credentials, appear anywhere in this repository.

_(Seed script and demo accounts: pending.)_

## Tests

_(Pending — golden-set retrieval tests, unit and RBAC tests, mock-forced scenario evals, and frontend↔backend wiring tests.)_

## Configuration

Every setting is environment-based and documented in [`.env.example`](.env.example). `.env` is gitignored and contains no committed values.

## License

Built for the AgentCare Build Challenge 2026.
