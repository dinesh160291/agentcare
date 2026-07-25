# AgentCare — working instructions

Agentic AI for patient **administration** (registration → intent → routing → booking → documents → confirmation/reminders → follow-up). Hackathon build, 4 days, solo.

**The spec is `docs/prd-agentcare.md`.** It is the source of truth and it is detailed — read the relevant section before implementing that area rather than inferring from this file. This file holds the rules, the layout, and the traps.

Where this file and the PRD disagree, the PRD wins — this file's tables (decision bins, invariants, context contract) are mirrors for speed, not authorities.

`problem_statement_hackathon_details.md` holds the organizer's rules verbatim. When this file and that file disagree, that file wins.

---

## Environment

- Windows, PowerShell. Python **3.11.6**, venv at `.venv/` (already created, deps installed).
- Activate: `.\.venv\Scripts\Activate.ps1` — or call `.\.venv\Scripts\python.exe` directly.
- PowerShell 5.1: **no `&&`**. Use `A; if ($?) { B }`.

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload      # API  :8000
.\.venv\Scripts\python.exe -m streamlit run ui/app.py            # UI   :8501
.\.venv\Scripts\python.exe -m pytest -q                          # tests
.\.venv\Scripts\python.exe -m pytest --cov=app --cov-report=term-missing
.\.venv\Scripts\python.exe scripts/seed.py                       # reset + seed DB
```

Add a dependency → pin it in `requirements.txt` with `==` (judges must get the tree we tested).

---

## Non-negotiable rules (any one breaks → score zero)

1. **Python backend.** Real logic in Python.
2. **Agentic.** LLM + multi-step tool-using workflow. Not CRUD, not a chat box that does nothing.
3. **Persistent SQL.** SQLite file on disk — **never `:memory:`**, never lost on restart.
4. **No diagnosis, prescription, or dosage.** Anywhere. Including routing replies, seed data, prompts, docstrings, and UI copy. Administrative routing only ("Cardiology handles this") — never a clinical claim ("this sounds like angina").
5. **No real data or secrets in the repo.** `.env` is gitignored from commit zero — never commit-then-gitignore. Before the repo goes public, scan tracked files **and full git history**. All seed data obviously synthetic.
6. **No hardcoded final responses presented as agent results.** A tool that returns a fixed value regardless of input scores zero. This binds the mock provider too — see below.

Also mandatory: RBAC enforced in the backend (not hidden buttons), audit logging, human escalation/approval, error handling with retries, environment-based config, working UI genuinely wired to the backend.

---

## Architecture spine

**"The model proposes; the code disposes."** Every feature splits into two bins. Getting a decision into the wrong bin is the main way this project fails.

| Probabilistic (LLM may decide) | Deterministic (code decides, always) |
|---|---|
| Reply wording | Department validity (against the `Department` table) |
| Intent / entity extraction | Workflow state transitions |
| Classification *proposals* | RBAC (role **and** ownership) |
| Which specialists a request needs | First-layer safety screen |
| | Booking/slot transactions |
| | Date resolution (`resolve_date`) |
| | Reading the confirmation answer |

The LLM never *applies* a consequence. It proposes a class or a plan; code validates it against a closed enum and applies it.

**Five agents, hub-and-spoke**: Coordinator, Department Routing, Appointment, Document, Follow-up. Safety is a **guardrail layer** (callbacks + escalation tool), not an agent. Specialists never talk peer-to-peer — all flow is Coordinator delegation + persisted state.

**Context contract**: Coordinator gets the windowed transcript (last N=15). Specialists get **no history** — only a typed task and the state they need. Where language *is* the job (Routing), the current request text rides inside the typed task.

**Six seams** — the only public boundaries; test through these, never internals: HTTP API · tool functions · **provider seam** (`BaseLlm` adapters in `providers/`) · orchestrator entry point (`run_workflow`) · clock (`clock.today()`) · `api_client`.

---

## Three invariants (state them in the README; enforce them in code)

1. **Boundedness** — every automated writer has a growth rule. Retry ladders, re-ask loops, reminder attempts, escalation/task spawning, agent tool iterations (max 8/turn, one re-plan/run). Failure paths too: a commit failure must clear the proposal, never leave it re-confirmable.
2. **Derivation** — every derived row has an update rule applied *in the source's transaction*. Cancel/reschedule an appointment → its reminders update atomically. Change the doc set → the missing-docs task updates.
3. **Trace completeness** — every turn bracketed, every LLM request paired with a response-or-error, every rejection recorded. Enforced by the well-formedness checker, not by good intentions.

---

## Layout

```
app/
  main.py            FastAPI app, WAL pragma at engine creation
  config.py          pydantic-settings; LLM_PROVIDER, DATABASE_URL, APP_TODAY...
  clock.py           clock.today() — the seam; nothing else calls date.today()
  db.py              engine, session, create_all
  models/            SQLAlchemy models (= the "database models" submission item)
  auth/              JWT, password hashing, role+ownership dependencies
  api/               routers, thin — validate, delegate, serialize
  agents/            one module per agent; prompts/ separate from logic
  providers/         BaseLlm adapters: mock.py, openai.py, groq.py
  tools/             plain Python functions, framework-agnostic
  orchestrator.py    run_workflow(user, message, session_id) — ADK confined here
  safety/            keyword screen, LLM pass, escalation
  trace/             TraceEvent writer, redaction, well-formedness checker
  scheduler/         APScheduler poll job (reminders + visit-completion sweep)
ui/
  app.py, pages/, api_client.py    api_client is Streamlit's ONLY path to the API
scripts/seed.py      seed + self-check (= "initialization files")
tests/               unit/ golden/ evals/ wiring/
docs/prd-agentcare.md
```

Tools stay plain functions and prompts stay in their own module **so the LangGraph fallback is cheap** — ADK lives only behind `orchestrator.py`.

---

## Traps already discovered — don't rediscover them

- **LiteLLM does not install here** (its build chain needs a Rust toolchain; `google-adk[extensions]` fails). All three providers are our own `google.adk.models.BaseLlm` subclasses calling the `openai`/`groq` SDKs directly. Do **not** retry the extensions install as a "fix". This is also the better design: identical plumbing for all three, and exact LLM request/response trace capture because we own the adapter.
- **ADK sessions need the async SQLite driver.** `DatabaseSessionService` takes `sqlite+aiosqlite:///...` (pip: `aiosqlite`) — a plain `sqlite:///` URL fails. This is a separate URL from the app's `DATABASE_URL`; don't unify them.
- **ADK callback parameter names must match exactly** (`callback_context`, `llm_request`, `tool`, `args`, `tool_context`, `tool_response`) — ADK passes them by keyword; renaming one to `ctx` is a mid-turn TypeError. The callbacks are the nine trace capture points, so this failure lands in the layer we care most about. Set `PYTHONUTF8=1` if a `UnicodeDecodeError` ever appears (documented ADK-on-Windows caveat).
- **Tests and eval runners force `LLM_PROVIDER=mock` in code, before imports** — env loading is setdefault, so the shell beats `.env`, and an import that constructs a provider before the override sees the wrong one. Setting it in the shell or after imports = live calls from CI.
- **WAL mode at engine creation.** FastAPI handlers and the scheduler share one SQLite file; without WAL they meet "database is locked". SQLite-only pragma, no-op under Postgres.
- **`clock.today()` everywhere.** Never `date.today()` in app code — golden tests freeze the clock. Live app uses the real date; `APP_TODAY` is a test/demo override only.
- **The mock provider is an understudy, not a fixture.** `LLM_PROVIDER=mock` must run the *whole* app end to end: real tool calls, real DB writes, replies **templated from persisted tool results** — never canned strings (that would break rule 6). Mock and live emit identically-shaped traces. If a feature works in live mode but not mock, the feature is not done.
- **Streamlit is a thin client.** Zero business logic, zero direct DB access. It calls `api_client`; the API enforces everything. A page that renders data it didn't fetch from the backend scores as faked.
- **Ownership, not just role.** Every id-taking endpoint verifies the row belongs to the acting user. Failed ownership → **404, not 403** (403 confirms the record exists), and audit the denied attempt. Judges try the one-digit edit first.
- **Upload hardening**: max file size, MIME allowlist via magic bytes (`filetype`), server-generated filenames. The client filename is a path-traversal vector.

---

## Conventions

- **Test as you go**, not on a final test day. Each component ships with its tests. ~80% backend coverage; Streamlit rendering excluded, but frontend↔backend **wiring is in scope** (httpx `TestClient` + Streamlit `AppTest`).
- **"Before you change it, pin it."** Prompt or guard changes get a scenario first.
- Every state transition writes an `AuditEvent` **and** a `TraceEvent`, and executes as a **compare-and-swap** (set new status only where the row still holds the expected old status; zero rows affected = someone else won, no-op).
- Trace events record author: `llm` | `template` | `guard`. Code-authored replies are not invisible.
- Redact PII at the three choke points before anything is persisted or logged.
- Keep replies free of clinical language even when the model is only summarizing.

---

## Never

- Never write to `.env` or echo secrets into the transcript, logs, or trace rows.
- Never commit a `.db` file, uploaded documents, or anything under `data/uploads/`.
- Never let the model apply a state change, confirm a booking, or invent a date.
- Never reference the author's prior personal voice-agent project in any committed file — the design ideas are ours to keep, the attributions stay out.
- Never add `.github/workflows/agentcare-checks.yml` from memory — it must be the organizer's file, downloaded from their URL.
