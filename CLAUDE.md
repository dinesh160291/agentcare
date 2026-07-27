# AgentCare — working instructions

Agentic AI for patient **administration** (registration → intent → routing → booking → documents → confirmation/reminders → follow-up). Hackathon build, 4 days, solo.

**Starting a fresh session? Read [`HANDOFF.md`](HANDOFF.md) first** — where the last session stopped, what to do next, and open items. Then [`PLAN.md`](PLAN.md) for the phase checklist. Update both before ending a session.

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
.\.venv\Scripts\python.exe -m pytest --cov=app --cov-branch --cov-report=term-missing
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
  orchestrator.py    run_workflow(...) + apply_patient_action(...) — ADK confined here
  workflow/          state machine, plan, message mapping, confirmation, staff actions
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
- **ADK sessions are keyed by the numeric user id**, which is what `run_workflow` passes — not the email. Asking `get_session` with the wrong key returns `None`, which reads as "zero events" and looks exactly like state failing to survive a restart. Cost a debugging cycle in the gate script; the system was fine.
- **ADK sessions need the async SQLite driver.** `DatabaseSessionService` takes `sqlite+aiosqlite:///...` (pip: `aiosqlite`) — a plain `sqlite:///` URL fails. This is a separate URL from the app's `DATABASE_URL`; don't unify them.
- **ADK callback parameter names must match exactly** (`callback_context`, `llm_request`, `tool`, `args`, `tool_context`, `tool_response`) — ADK passes them by keyword; renaming one to `ctx` is a mid-turn TypeError. The callbacks are the nine trace capture points, so this failure lands in the layer we care most about. Set `PYTHONUTF8=1` if a `UnicodeDecodeError` ever appears (documented ADK-on-Windows caveat).
- **The Coordinator's session is persistent, so "has this tool been called?" must mean *this turn*.** `tool_results` / `called_tools` / `latest_tool_result` are scoped to the last thing the patient actually said (`current_turn_start`). Unscoped, turn 2 sees turn 1's `classify_message` result, answers with the *previous* turn's verdict, and never classifies the current message — a withdrawal then leaves the run in `pending_confirmation` claiming the patient is still waiting to confirm. Note that ADK sends tool results with `role="user"` too, so a turn boundary is a user content with **text and no `function_response`**.
- **Refusing a tool does not stop an agent loop.** Returning an error dict from `before_tool_callback` leaves the model being asked again, wanting the same tool again — until ADK's own 500-LLM-call limit fires, which is the implicit framework limit the iteration budget exists so as not to depend on. The budget must *also* return an `LlmResponse` from `before_model_callback`. It can blow in two places: inside a specialist, and inside the Coordinator before a run exists (where there is no run to transition to `failed`).
- **ADK's `transfer_to_agent` hands the sub-agent the whole session history**, which the context contract forbids. Delegation is therefore dispatched by `orchestrator.py` from the validated plan, and specialists hold no transfer tool. Don't "restore" sub-agent wiring — it breaks the context contract and re-opens the unbounded-delegation loop the gate found.
- **Tests and eval runners force `LLM_PROVIDER=mock` in code, before imports** — env loading is setdefault, so the shell beats `.env`, and an import that constructs a provider before the override sees the wrong one. Setting it in the shell or after imports = live calls from CI.
- **WAL mode at engine creation.** FastAPI handlers and the scheduler share one SQLite file; without WAL they meet "database is locked". SQLite-only pragma, no-op under Postgres.
- **`clock.today()` everywhere.** Never `date.today()` in app code — golden tests freeze the clock. Live app uses the real date; `APP_TODAY` is a test/demo override only.
- **The mock provider is an understudy, not a fixture.** `LLM_PROVIDER=mock` must run the *whole* app end to end: real tool calls, real DB writes, replies **templated from persisted tool results** — never canned strings (that would break rule 6). Mock and live emit identically-shaped traces. If a feature works in live mode but not mock, the feature is not done.
- **Streamlit is a thin client.** Zero business logic, zero direct DB access. It calls `api_client`; the API enforces everything. A page that renders data it didn't fetch from the backend scores as faked.
- **A second opinion that agrees with the first is not a check.** Sabotaging the deterministic safety screen left *every* scenario green — the mock's LLM layer happened to subsume it on all of them, so the whole first layer could have been deleted unnoticed. A hybrid guard needs at least one pinned case per layer that **only that layer** can catch (self-harm for the phrase list, which has no body-part-plus-severity co-occurrence; a worsening symptom for the LLM pass, which carries no listed phrase). Falsify each guard *individually*.
- **The safety screen's false-positive direction is the one that makes it unusable.** A missed emergency is the worst outcome; a screen that fires on ordinary administration is not the safe side of that trade, because a review queue that is mostly noise is a queue nobody reads. Three live traps: `pain` alone is not an emergency (the seed's own ambiguous-routing case is "my kid has ear pain"), `prescription` is a **document type** Ophthalmology requires (only a *request* for one is clinical), and `emergency contact` is a field on the patient's profile.
- **After a safety trigger the run is terminal, so it is no longer the *active* run.** Escalation dedup therefore cannot rely on `active_run`: the naive path creates a fresh run for every repeat, turning one frightened patient typing "chest pain" five times into five queue items. Look explicitly for this session's escalated run with an open escalation, and attach.
- **A bound tied to a classification is bypassed by a misclassification.** The confirmation non-answer counter is incremented for *any* turn that leaves the run waiting, not for the classes that "look like" non-answers — an "hmm, maybe" read as off-topic would otherwise buy an extra free turn forever. A bound reachable only through correctly-classified paths is not a bound.
- **Result dicts and their consumers drift silently.** `diff_required_documents` returns `missing_mandatory`; the toolbelt and the mock both read `missing`, so the missing-documents task was never created and patients were told nothing was needed. Nothing failed — the key was simply absent and `or []` did the rest. When a tool's refusal path returns a differently-shaped dict from its success path, that is the same bug waiting.
- **`repr(Settings())` printed the live API key.** pytest puts `repr(settings)` in its failure output whenever an assertion mentions the settings object, so one unrelated red test publishes the developer's keys — to the terminal, and in CI to a public build log. This repo did it. Every secret field carries `Field(..., repr=False)`, pinned by a test. Any new secret setting must too.
- **"cancel" is three different things and they must not meet.** It is the verb for cancelling an *appointment*, it is an exact token in `DECLINE_TOKENS` that answers a *proposal*, and it opens the withdrawal phrase that closes a *run* ("cancel that request"). The reader is safe because it matches the whole normalised string, so only a bare "cancel" declines — but intent detection is not, and reading "cancel my appointment" as a withdrawal leaves the appointment standing while the reply says it was dealt with. Rule: an appointment verb needs verb **plus** an appointment noun, and withdrawal is checked first.
- **A completion marker that says "a row exists" is meaningless for verbs acting on rows that already existed.** The `book` step was complete when `state["appointment_id"]` appeared, which works only because booking is the verb that *creates* the row. Reschedule and cancel act on an appointment that was already there, so the same check reports them done before they run. The marker is `state["committed_action"]` — which verb actually committed — not the presence of an id.
- **A reschedule run never goes through routing, so it carries no department.** `find_available_slots` reads the department from the run's state, which is correct for booking and empty here; the department belongs to the *appointment* being moved. Hence a separate `find_slots_for_reschedule(appointment_id, ...)` that derives it from the ownership-checked appointment, rather than loosening the bound tool to take a department id from the model.
- **A receipt can carry a stale fact of its own.** The booking receipt appended "You will get a reminder the day before" unconditionally; cancelling retires the reminder in the same transaction, so the cancellation receipt promised a reminder that had just been deleted. Receipts re-read the row for *every* fact they state, including the ones that look like boilerplate.
- **A trace timeline filtered by `workflow_run_id` starts in the middle of the turn.** A turn opens *before* its run exists — inbound event, safety screen, classification — so those rows carry a null run id by design, and `bind_run` only attaches what comes after. Selecting `WHERE workflow_run_id = ?` therefore drops exactly the part a reviewer needs, and it looks like a complete timeline because it is ordered and non-empty (the staff viewer's first version began at seq 23 with no inbound event). Find the run's `turn_id`s first, then return every event of those turns.
- **An exception handler cannot commit — the session is already closed.** FastAPI unwinds the dependency stack before handlers run, so anything a callee wrote *before* raising is gone by then. That is fine for refusals that wrote nothing, and wrong for `apply_staff_decision`, which writes its audit and trace rows and then raises: that call site catches `ValidationFailed` itself, commits, and re-raises as a 422. A refused decision is a thing a human did and belongs in the timeline.
- **Ownership, not just role.** Every id-taking endpoint verifies the row belongs to the acting user. Failed ownership → **404, not 403** (403 confirms the record exists), and audit the denied attempt. Judges try the one-digit edit first. The probe sweep enumerates routes from `app.openapi()["paths"]`, not by walking `app.routes` — included routers are nested and their shape has changed between FastAPI versions, so the walk silently returns nothing and every completeness check passes vacuously.
- **Upload hardening**: max file size, MIME allowlist via magic bytes (`filetype`), server-generated filenames. The client filename is a path-traversal vector.

---

## Conventions

- **Test as you go**, not on a final test day. Each component ships with its tests. ~80% backend coverage; Streamlit rendering excluded, but frontend↔backend **wiring is in scope** (httpx `TestClient` + Streamlit `AppTest`).
- **"Before you change it, pin it."** Prompt or guard changes get a scenario first.
- Every state transition writes an `AuditEvent` **and** a `TraceEvent`, and executes as a **compare-and-swap** (set new status only where the row still holds the expected old status; zero rows affected = someone else won, no-op).
- Trace events record author: `llm` | `template` | `guard`. Code-authored replies are not invisible.
- Redact PII at the three choke points before anything is persisted or logged.
- Keep replies free of clinical language even when the model is only summarizing.
- Deterministic-bin code is written test-first — the test transcribes the PRD's pinned behavior before the implementation exists. Everything else ships with tests in the same commit, tested through the seams, never through internals.

---

## Change discipline

- **Smallest diff that fixes the problem.** No drive-by refactors, renames, or cleanups
  of code you weren't asked to touch. If you see something worth improving, note it in
  HANDOFF.md — don't fold it into an unrelated change.
- **Diagnose before changing.** When a test fails, first verify the test delivered the
  input you think it did; the suspected layer is often innocent. Name the root cause
  before writing the fix.
- **Never make a test pass by weakening it.** No deleted tests, no loosened assertions,
  no broadened matchers to get green. A red test is information; killing the messenger
  is the one forbidden move.
- **Distrust green.** A passing check is evidence only if it could have failed:
  before trusting a PASS, ask what the check would have shown had the claim been
  false. A query that counts stale rows, or a key every process writes, passes
  regardless — that's a check vouching for nothing. 
- **Fail loud.** No broad try/except that swallows errors to keep things running.
  Catch specifically, or let it raise.
- **No speculative structure.** Don't add abstractions, options, or generality for
  futures nobody asked for. Three concrete uses before an abstraction.
- **Match the neighborhood.** Follow the file's existing style, naming, and idioms —
  and prefer editing in place over rewriting whole files.
- **Stop-and-ask thresholds**: a fix that wants to touch >3 files, add a dependency,
  or change a public seam gets explained first, not done first.

---

## Never

- Never write to `.env` or echo secrets into the transcript, logs, or trace rows.
- Never commit a `.db` file, uploaded documents, or anything under `data/uploads/`.
- Never let the model apply a state change, confirm a booking, or invent a date.
- Never add `.github/workflows/agentcare-checks.yml` from memory — it must be the organizer's file, downloaded from their URL.
