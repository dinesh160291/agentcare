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
  app.py, views/, api_client.py    api_client is Streamlit's ONLY path to the API
  theme.py, shell.py               the look, and the fetch/render helpers
                                   (views/, not pages/: st.navigation builds the
                                   role-based nav, and Streamlit's automatic
                                   pages/ discovery would compete with it)
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
- **An accepted submit tool does not stop the model asking to submit again.** Found live on `openai/gpt-4o-mini`: `submit_safety_verdict(safe)` was accepted at seq 5–7, then re-called *eight* more times until the iteration budget fired and the turn failed — nine calls for one screen, on a decision that was already made. The function_response round-trip was intact, so it is the model, not the adapter: gpt-4o-mini volunteers no text while it holds a single mandatory tool. llama and the mock happen to answer with prose there, which is the only reason a defect living in **every** submit-style loop survived two providers. And it is not merely expensive — `orchestrator.py` checks `budget_exhausted` *before* `verdict.fired`, so a screen that blew its budget after accepting an **emergency** verdict answered with the generic failure template and opened no escalation. Two bounds now, in the deterministic bin, both ending the loop from `before_model` (refusing the tool does not, per the trap above): a **terminal tool** (`run_agent(terminal_tool=...)`) whose acceptance is the agent's whole output ends the loop immediately — the safety screen is the only one, because its prompt forbids it a reply and `llm_screen` discards its text; and an **accepted repeat** — same tool, same args, already accepted this agent turn — is refused and ends the loop, which is what bounds the other five submit/classify/propose tools, whose prose *is* the patient's reply and must not be cut off before it exists. Both reset per agent, like the iteration counter. Falsify them separately: the terminal declaration is pinned only by the LLM-*request* count, because the repeat rule catches the second tool call either way.
- **ADK's `transfer_to_agent` hands the sub-agent the whole session history**, which the context contract forbids. Delegation is therefore dispatched by `orchestrator.py` from the validated plan, and specialists hold no transfer tool. Don't "restore" sub-agent wiring — it breaks the context contract and re-opens the unbounded-delegation loop the gate found.
- **Tests and eval runners force `LLM_PROVIDER=mock` in code, before imports** — env loading is setdefault, so the shell beats `.env`, and an import that constructs a provider before the override sees the wrong one. Setting it in the shell or after imports = live calls from CI.
- **WAL mode at engine creation.** FastAPI handlers and the scheduler share one SQLite file; without WAL they meet "database is locked". SQLite-only pragma, no-op under Postgres.
- **"The earliest available slot" is in the past by lunchtime.** The seed lays slots down from today at 09:00, and the mutating tools refuse a slot whose start time has passed — so a test helper that picks the earliest free slot passes in the morning and fails in the afternoon. Two Phase 6a tests did exactly that, and the suite reported on the clock rather than on the code. Any helper choosing a slot filters `start_time > clock.now()`.
- **ADK declares a tool's parameters in one of two places, and this version uses the one we weren't reading.** `FunctionTool._get_declaration()` fills either `parameters` (a genai `Schema`) or `parameters_json_schema` (plain JSON Schema), depending on the `JSON_SCHEMA_FOR_FUNC_DECL` feature flag — which is on here, so `parameters` is **`None`** on every tool. The adapter read only `parameters`, so `_json_schema(None)` returned `{"type": "object", "properties": {}}` and **every tool went to OpenAI and Groq declaring that it takes no arguments at all.** The model was inferring argument shapes from the docstring; that is how `submit_plan` came back as `[{"route": "..."}]`, rejected, resubmitted identically until the iteration budget fired. Nothing local could see it: the mock provider does not read schemas, and the seam's tests were built from hand-written declarations, which have the shape the code expected by construction. Build the declaration from a **real `FunctionTool`** or the test is checking a fixture. Where a value is drawn from a closed enum, say so with `Literal[...]` — that is the difference between a shape enforced at generation and a shape corrected afterwards through the retry ladder.
- **A `role: "tool"` message with no assistant `tool_calls` before it is a chat-completions 400**, and the error names `messages.[1].role` rather than the cause. It happens whenever a history begins mid-turn: ADK carries tool results as `role="user"` contents, so `window_contents` counts them as turns and can cut between a call and its result. `_messages_payload` therefore **re-pairs** an orphaned `function_response` with a synthesised assistant call rather than dropping it — the result is real and the model needs it; only the original arguments are unrecoverable. The invariant to hold, and to test as one: every tool message has an assistant `tool_call` bearing its id somewhere before it.
- **One turn must have one reply.** The turn envelope writes a template outbound event and **commits** it before re-raising, so a provider failure leaves a trace saying the patient was let down gently — while the HTTP layer, with nothing mapping `ProviderError`, answered a bare 500 and the UI showed "Internal Server Error". Two accounts of one turn, and the trace vouching for words nobody read. `ProviderError` carries `turn` (attached by the envelope, since a provider has no turn to hand over) and the API serves it as **503 with the `TurnOut` body plus a `detail` key**. Still an error status — the request genuinely did not happen — but the words are the ones the trace recorded.
- **The mapping table's *answer* half needs somewhere to happen.** `side_question` has always meant answer-and-stay, but at `pending_confirmation` `_continue_run` short-circuited to decline-or-re-ask before the `ANSWER_AND_STAY` branch could run — so "show me more available slots" got the re-ask nag, twice, live. A class whose consequence is unreachable from a state is not implemented in that state, and it fails silently because the reply is still polite and the run is still correct. Answering now happens through bound tools (`list_other_slots`, `propose_another_slot`) that cannot commit; the pinned rule underneath is untouched.
- **A slot id the model remembers is indistinguishable from one it invented** — both arrive as an integer. So a re-proposal is validated against `state["offered_slot_ids"]`, a set built *only* from `find_available_slots` payloads at the moment options are rendered (rendering and recording are one step, or the patient gets offered a time the guard then refuses). It is the union over the whole run, because "actually, the first one you showed me" arrives three exchanges later; staleness is handled by re-running the *same* liveness check the original proposal ran, never by shrinking the set. Both directions of the check are traced — a refused invention leaves no other mark.
- **Facts do not survive a paraphrase.** One live receipt carried four defects at once: "still needed, which is optional" (mandatory and optional welded into one clause), "recorded for follow-up" beside "no outstanding follow-up tasks", the *reminder's* fire date printed as the appointment's (a patient arrives a day early), and a context dump nobody asked for. None is a prompt that could have been worded better. After a commit the reply is assembled in `app/workflow/replies.py` from rows; specialists still run, but what the patient is told is code's.
- **A re-ask that announces an errand describes a turn that is already over.** "I will proceed to find a suitable time for you. Please hold on" — said while a time was already being held, and nothing runs after the reply, so the patient waits, repeats themselves, and the non-answer counter reads it as a stall. Guarded by `promises_action`, whose false-positive direction is cheap on purpose (the fallback carries the same facts) — the opposite of the safety screen's trade, and worth stating because the two guards look alike.
- **Substring cue matching turns latent false positives into bugs the moment something depends on them.** `"erm"` is inside `d**erm**atology`, so "book me a dermatology appointment instead" read as a *hesitation* while a proposal stood. It reached the right class anyway for months — until hesitation started deciding whether a message was a new request. Short cues need `_has_word`, not `_has`; and the general shape is that a wrong-but-inconsequential signal is a bug already, waiting for a consumer.
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
- **`streamlit run ui/app.py` puts `ui/` first on `sys.path`, so `ui/app.py` shadows the `app` package.** `import app.config` then resolves to the entry script and dies with `No module named 'app.config'; 'app' is not a package`. Under pytest the repo root is already first, so **the entire wiring suite passes while the real app crashes on its login screen** — this was found by opening a browser, not by a test. `ui/app.py` re-inserts the repo root at `sys.path[0]` before any import; `tests/wiring/test_entrypoint_import_order.py` reproduces the launcher's ordering in a subprocess and is the only test that can fail on it.
- **A UI test can pass while the guarantee it names is gone.** Making the Confirm button POST the word "confirm" to `/workflow/messages` instead of `/workflow/actions` left every assertion green: the exact-token reader reads "confirm" and commits, so the booking, the receipt and the screen are identical. The two front doors are only distinguishable in the **trace**, where the inbound event records `patient-action` vs `patient-message`. Assert on that, or the test is describing the code rather than checking it.
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
