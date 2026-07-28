# AgentCare

An agentic AI system for hospital **patient administration** — registration, administrative intent detection, department routing, appointment booking, document coordination, confirmations and reminders, and follow-up scheduling.

> **Not a clinical system.** AgentCare performs no diagnosis, prescribes nothing, recommends no dosages, and does not replace a healthcare professional. It routes administrative requests and escalates anything clinical, urgent, or uncertain to a human.

A patient types *"I need a cardiology appointment next week, and I'd like to attach my previous ECG"*. Five agents classify it, route it to a department, find a slot, hold it, and stop. Nothing is booked until the patient says an exact `yes` or presses **✅ Confirm** — and the code that reads that answer is code, not a model. Then the booking, the document diff, the reminder and the follow-up task are written in one transaction, and the receipt the patient reads is assembled from those rows.

---

## The one idea

**The model proposes; the code disposes.**

Every decision in the system is sorted into one of two bins, and getting a decision into the wrong bin is the main way a project like this fails.

| The model may decide | Code decides, always |
|---|---|
| Reply wording | Whether a department exists (checked against the table) |
| Intent and entity extraction | Every workflow state transition |
| Classification *proposals* | Role **and** ownership checks |
| Which specialists a request needs | The first-layer safety screen |
| | Booking and slot transactions |
| | Date resolution |
| | Reading the confirmation answer |

The LLM never *applies* a consequence. It proposes a class or a plan; code validates that against a closed enum and applies it. A department the model invents does not become real by being spelled confidently, and a paraphrase of "yes" never books anything.

### Three invariants

1. **Boundedness** — every automated writer has a growth rule: retry ladders, re-ask loops, reminder attempts, escalation spawning, and a per-turn cap on agent tool iterations. Failure paths too — a failed commit clears its proposal rather than leaving it re-confirmable.
2. **Derivation** — every derived row is updated *in the source's transaction*. Cancel an appointment and its reminder is retired atomically; change the document set and the missing-documents task moves with it.
3. **Trace completeness** — every turn is bracketed, every LLM request is paired with a response or an error, and every rejection is recorded. Enforced by a well-formedness checker, not by good intentions.

---

## Architecture

### Agents

Five agents, hub-and-spoke. Each has its own prompt module and its own bound toolset; specialists never talk to each other, so all coordination is Coordinator delegation plus persisted state.

| Agent | Responsibility | Tools it is given |
|---|---|---|
| **Coordinator** | Reads the request, produces a validated plan, classifies follow-up messages against the active run, delegates, and reports completion or failure. The only agent with conversation history. | `load_patient_context`, plus **either** `submit_plan` or `classify_message` (see below) |
| **Department Routing** | Maps free text to a real department, reports its own confidence, and hands uncertainty to a human instead of guessing. | `resolve_department`, `list_departments`, `submit_routing` |
| **Appointment** | Finds slots, checks conflicts, and *proposes* a booking, reschedule or cancellation. It cannot commit one. | `resolve_date`, `find_available_slots`, `find_slots_for_reschedule`, `propose_appointment`, `propose_reschedule`, `propose_cancellation`, `render_confirmation`, `list_my_appointments` |
| **Document** | Verifies uploaded documents against their declared type, diffs what a department requires against what is on file, and records the shortfall. | `list_patient_documents`, `list_unverified_documents`, `read_document_text`, `submit_document_verification`, `diff_required_documents`, `record_missing_documents` |
| **Follow-up** | Opens reminders and follow-up tasks, and reports what is still outstanding. | `list_patient_reminders`, `list_open_tasks` |

**The Coordinator's toolset depends on the state of the run**, and that is a guard rather than a convenience: the wrong tool is *absent* rather than merely discouraged. With no active run it is given `submit_plan` and not `classify_message` — there is nothing to classify against. With one, the reverse — so it cannot quietly start a second run. `submit_confirmation_verdict` appears only while a proposal is actually pending, and `propose_another_slot` only when the thing being held is a booking, so there is exactly one place a held offer can be disturbed.

**Safety is a guardrail layer, not an agent.** It runs on every inbound message before any planning, in two layers that fail differently on purpose: a code-owned phrase list (deterministic, catches self-harm phrasing that no severity heuristic would) and an LLM pass (catches a worsening symptom that carries no listed phrase). Each layer has at least one pinned case that *only that layer* can catch — a second opinion that agrees with the first is not a check.

The trade is deliberately one-directional: a missed emergency is the worst outcome this system has, so the screen over-refers. It is also tuned against the *other* failure — a review queue that is mostly noise is a queue nobody reads — which is why `pain` alone is not an emergency, "prescription" is a document type rather than a clinical request, and "emergency contact" is a field on a profile.

### Context contract

The Coordinator receives a windowed transcript. **Specialists receive no history at all** — only a typed task and the state they need. Where language *is* the job (routing), the request text rides inside the typed task. Delegation is dispatched by the orchestrator from the validated plan rather than by handing a sub-agent the whole session.

### Orchestration

```
HTTP request
  └─ turn envelope ──── opens the trace, guarantees exactly one reply per turn
       ├─ safety screen ─────────── keyword layer, then LLM layer
       ├─ deterministic query path ─ "what do I have?" answered from rows, no planning
       ├─ Coordinator ───────────── plan (validated against a closed enum)
       │                            or message→run classification
       ├─ per-step dispatch ─────── one specialist per plan step
       ├─ confirmation reader ───── exact token or button; code only
       └─ commit + receipt ──────── one transaction; receipt assembled from rows
```

Every state transition is a **compare-and-swap** against a pinned legal-transition table, and writes both an `AuditEvent` and a `TraceEvent`. Zero rows affected means someone else won the race, and the turn no-ops rather than double-booking.

### Six seams

The only public boundaries, and the only places the tests touch: the **HTTP API**, the **tool functions**, the **provider seam** (`BaseLlm` adapters), the **orchestrator entry point**, the **clock**, and the UI's **`api_client`**.

The provider seam is why the whole application runs with no API key. `mock`, `openai` and `groq` are three implementations of one adapter, all writing identically-shaped traces.

---

## Stack

| Layer | Choice |
|---|---|
| Backend | Python 3.11 + FastAPI — auth, RBAC, orchestration, tools, persistence |
| Frontend | Streamlit thin client (every call goes through one `api_client` module) |
| Agents | Google ADK |
| LLM | Pluggable provider: `mock` \| `openai` \| `groq` |
| Database | SQLAlchemy + SQLite file, WAL mode (Postgres-compatible) |
| Scheduler | APScheduler — reminder delivery and visit-completion sweep |

---

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
python scripts/seed.py      # create and populate the database
```

**Running with no API key at all:** set `LLM_PROVIDER=mock` in `.env`. The mock provider is a first-class provider, not a test stub — the entire application runs end to end on it, calling the same tools, writing the same database rows, and templating its replies from persisted tool results rather than from canned strings. Use it to evaluate the system offline.

### Run it

Two processes. Start the API first.

```bash
python -m uvicorn app.main:app --reload      # API  → http://localhost:8000
python -m streamlit run ui/app.py            # UI   → http://localhost:8501
```

Interactive API docs are at `http://localhost:8000/docs`.

### Demo accounts

Created by `scripts/seed.py`. All synthetic; the `.invalid` TLD is reserved by RFC 2606 and can never resolve.

| Role | Email | Password |
|---|---|---|
| Patient | `asha.patient@example.invalid` | `Demo123!pass` |
| Patient | `rohan.patient@example.invalid` | `Demo123!pass` |
| Staff | `staff@example.invalid` | `Demo123!pass` |

Try, as the patient: *"I need a cardiology appointment next week"* → **✅ Confirm**. Then as staff, open the queue and read the run's trace.

---

## Database models and initialization

**Models** — SQLAlchemy ORM under [`app/models/`](app/models/). Sixteen entities:

`User` · `PatientProfile` · `Department` · `DepartmentSynonym` · `DepartmentRequiredDocument` · `Doctor` · `AppointmentSlot` · `Appointment` · `PatientDocument` · `WorkflowRun` · `Escalation` · `Reminder` · `FollowUpTask` · `Notification` · `AuditEvent` · `TraceEvent`

Two are worth calling out. `DepartmentSynonym` makes routing vocabulary a **table** rather than a constant, so the deterministic half of routing is data the staff console can inspect. `TraceEvent` is separate from `AuditEvent` on purpose: audit records what a *user* did, trace records what the *system* did — including every LLM request and response, every tool call, every guard verdict, and every rejection.

**Initialization** — the schema is created from the models via `create_all`; [`scripts/seed.py`](scripts/seed.py) populates it and then runs a **self-check** that fails loudly if the seeded world is not demo-ready (a department with no bookable capacity, a slot dated in the past, a booked slot with no appointment, or a non-terminal workflow run left lying around).

Alembic migrations are deliberately not used: the schema is created fresh from the models, which is sufficient at this scope.

---

## Sample data

All seed data is synthetic and obviously so. No real patient information and no real credentials appear anywhere in this repository.

Ten departments with their routing vocabulary and required-document rules, 22 doctors, ~1,600 bookable slots across a two-week window, four patients, one staff user, one pre-existing appointment (so reschedule and cancel have something to act on), three documents and one reminder.

The seeded world contains a deliberate ambiguity — *"my kid has ear pain"* matches both Pediatrics and ENT — because ambiguity is a first-class outcome here, not a failure. It resolves to a question and a staff review rather than to a silent coin-flip.

---

## Tests

```bash
python -m pytest -q                                              # everything
python -m pytest --cov=app --cov-branch --cov-report=term-missing # with coverage
```

**1,423 tests**, organised by what they can falsify:

| Suite | What it holds |
|---|---|
| `tests/unit/` | Tools, state machine, mapping, safety, replies, scheduler, and the orchestrator driven end to end under the mock provider |
| `tests/api/` | HTTP contracts, RBAC, and a cross-patient probe that enumerates every id-taking route from the live OpenAPI schema and one-digit-edits it |
| `tests/wiring/` | Streamlit `AppTest` against the real FastAPI app over an ASGI transport — the UI exercised against the actual backend, never a mock of it |
| `tests/golden/` | Frozen outputs for tool results and rendered replies |
| `tests/evals/` | Scripted conversations replayed through the orchestrator and graded on plumbing keys, not on wording |

Three conventions do most of the work:

- **Deterministic-bin code is written test-first.** The test transcribes the pinned behaviour before the implementation exists.
- **Every guard is falsified individually.** Before a passing check is trusted, it is sabotaged to confirm it *could* have failed. This has repeatedly found guards that vouched for nothing — including lines written minutes earlier.
- **Tests go through the six seams**, never through internals.

`scripts/live_sweep.py` replays 25 scripted conversations against a real provider and diffs the result against a previous run. It is excluded from CI because it is billed.

---

## Security and access control

- **Ownership, not just role.** Every id-taking endpoint verifies the row belongs to the acting user. A failed ownership check answers **404, not 403** — a 403 confirms the record exists and turns the id field into an enumeration oracle — and the denied attempt is audited.
- **Uploads** are size-capped, MIME-checked by magic bytes rather than by the client's filename, and stored under a server-generated name. The client filename is a path-traversal vector.
- **Secrets never print.** Every secret setting is declared `repr=False`, pinned by a test, because pytest puts a settings `repr` into its failure output and one unrelated red test would otherwise publish a key to a build log.
- **PII is redacted** at three choke points before anything is persisted or logged.

---

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
- **The safety screen errs towards escalation, and which phrasing trips it is
  not stable.** The first layer is a code-owned phrase list and behaves the
  same way every time. The second is a model, and it is both conservative and
  sensitive to wording: "I have severe knee pain" has been passed to a human
  rather than routed to Orthopedics, while "severe ear pain" — the same
  sentence shape — has gone through to routing on another run. A missed
  emergency is the worst outcome this system has, so the layer is tuned to
  over-refer and no attempt is made to talk it down. *Floor: the trade is
  one-directional. An over-refer sends an administrative request to a person
  who redirects it; nothing routes **past** the screen, and no layer of it is
  permitted to make a clinical claim in either direction.*
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

---

## Specification

The full design lives in **[`docs/prd-agentcare.md`](docs/prd-agentcare.md)** — user stories, the pinned workflow state machine, agent architecture, safety and confirmation flows, the data model, and the testing strategy. **[`CLAUDE.md`](CLAUDE.md)** holds the working rules, the layout, and a long list of traps found the hard way.

## Configuration

Every setting is environment-based and documented in [`.env.example`](.env.example). `.env` is gitignored and contains no committed values.

## About

Built for the AgentCare Build Challenge 2026.
