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

## Known limitations

Honest, current, and drawn from a scripted sweep of every phrasing in five live
transcripts replayed against a real model (`scripts/live_sweep.py`). The
conversational layer is **frozen**: the edges below are documented rather than
fixed, because each remaining one is a judgement about language, and language
judgements are the part of this system that is allowed to be wrong.

Every limitation is stated with the floor underneath it. That floor is the same
in each case and it is structural, not a matter of the wording holding: **the
failure mode is a re-ask, never a wrong commit.** Nothing is booked, moved or
cancelled without an exact `yes` or the ✅ Confirm button, and that reader is
code — no model output can reach it.

- **Some phrasings get a clarifying question instead of an offer.** "Please
  reschedule my appointment to next week" goes straight to a held time. "Lets
  reschedule my appointment", with no date in it, is answered with "when would
  you like it?" — and answering *that* can come back as a list of free times
  rather than a held offer, so the patient has to name a time in a sentence
  that also names the verb. If a run has to ask twice it ends on the failure
  notice and opens a staff escalation rather than looping. *Floor: no
  appointment moves. The cost is another message.*
- **An unusual reply to a pending confirmation may be re-asked rather than
  read.** "Not that one, thanks" was read as a decline on one live replay and
  as a non-answer on another. *Floor: the confirmation reader accepts an exact
  token or the button and nothing else, so a paraphrase cannot commit in either
  direction; the re-ask restates the exact time being held.*
- **The safety screen errs towards escalation, deliberately.** "I have severe
  knee pain" is passed to a human rather than routed to Orthopedics. A missed
  emergency is the worst outcome this system has, so the second (model) layer
  is tuned to over-refer. *Floor: the cost is an administrative request
  reaching a person who redirects it — never a clinical claim, which no layer
  is permitted to make.*
- **Uploaded documents are verified one per conversational turn.** A patient
  who files three at once has them checked over the next few exchanges rather
  than all at once. This is a deliberate bound, not a backlog: verifying an
  unbounded number in one turn is what used to exhaust the agent's tool budget
  and take the rest of the booking down with it. *Floor: nothing waits on
  verification — the appointment, the reminder and the required-documents list
  are all unaffected by it, and a mismatch is flagged for staff whenever it is
  found.*
- **"What documents do I have on file?" is not always recognised as a
  question.** Appointment and reminder listings are answered deterministically
  from the database; the documents listing still goes through planning, because
  pre-empting it would silently stop the verification the step performs — so
  some phrasings come back as "could you tell me a little more". *Floor:
  read-only either way; no document state changes.*

## Configuration

Every setting is environment-based and documented in [`.env.example`](.env.example). `.env` is gitignored and contains no committed values.

## About

Built for the AgentCare Build Challenge 2026.
