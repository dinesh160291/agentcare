# AgentCare — architecture

How the system is put together, and why each piece sits where it does. The
[PRD](prd-agentcare.md) is the specification; this document is the map. For the
running list of failures that shaped these decisions, see
[`CLAUDE.md`](../CLAUDE.md).

---

## 1. The organising idea

**The model proposes; the code disposes.**

Every decision the system makes is sorted into one of two bins before it is
implemented. Getting a decision into the wrong bin is the main way a project
like this fails, and most of the defects this codebase has recorded were
exactly that — a consequence that a model was allowed to apply.

| Probabilistic — the model may decide | Deterministic — code decides, always |
|---|---|
| How a reply is worded | Whether a department exists (checked against the `Department` table) |
| Intent and entity extraction | Every workflow state transition |
| Classification *proposals* (`continuation`, `conflicting`, `withdrawal`, …) | Role **and** ownership checks |
| Which specialists a request needs | The first-layer safety screen |
| Which slot to offer, from a validated set | Booking, reschedule and cancellation transactions |
| Which department to route to, when the table is silent | Date resolution (`resolve_date`) |
| Which appointment a sentence refers to, when the cues are ambiguous | Reading the confirmation answer |
| | Reading a numbered choice, or a time from a list already shown |
| | Which facts a receipt states, and in what units |

The model never *applies* a consequence. It proposes a class, a plan, a
department or a slot id; code validates that against a closed enum or a
database row and then applies it — or refuses, and records the refusal.

Three refinements of that rule are worth stating, because each one cost a live
failure to learn:

- **Validating a class name is not validating a consequence.** `withdrawal` is
  a real member of the enum, so nothing objects when a model returns it for the
  patient's reply of "2". Where a consequence is destructive, code re-checks the
  patient's *words* before applying it.
- **Whatever a layer refuses, the layer above must not offer.** A slot withheld
  from the patient because it clashes with their own diary must not appear in
  the payload the model picks from.
- **When code overrules the model's classification, check that the tools the
  correct classification would have called have actually run.** A corrected
  verdict sitting on the wrong turn's tool results is a verdict about nothing.

---

## 2. The three invariants

Stated here, enforced in code, and checked by tests that were written to fail.

1. **Boundedness.** Every automated writer has a growth rule — retry ladders,
   re-ask loops, reminder delivery attempts, escalation spawning, agent tool
   iterations (max 8 per turn, one re-plan per run). Failure paths too: a failed
   commit clears its proposal rather than leaving it re-confirmable. Two
   corollaries learned the hard way: a bound tied to a *classification* is
   bypassed by a misclassification, so the confirmation stall counter counts any
   turn that leaves the run waiting; and asking the patient a question is not
   failing to make progress, so a step that ends by asking does not spend the
   re-plan budget.
2. **Derivation.** Every derived row is updated *inside the source's
   transaction*. Cancel an appointment and its reminder retires atomically;
   change the document set and the missing-documents task moves with it; close a
   run by supersede or withdrawal and its open escalations close with it (safety
   escalations excepted — those outlive the conversation on purpose).
3. **Trace completeness.** Every turn is bracketed by an inbound and an outbound
   event, every LLM request is paired with a response or an error, and every
   rejection is recorded. Enforced by a well-formedness checker rather than by
   good intentions.

---

## 3. Agents

Five agents, hub-and-spoke. Specialists never talk to each other, so all
coordination is Coordinator delegation plus persisted state.

```
                        ┌───────────────┐
                        │  Coordinator  │   plan · classify · delegate
                        └───────┬───────┘
            ┌───────────┬───────┴───────┬───────────┐
            ▼           ▼               ▼           ▼
     ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌───────────┐
     │ Department │ │Appointment │ │ Document │ │ Follow-up │
     │  Routing   │ │            │ │          │ │           │
     └────────────┘ └────────────┘ └──────────┘ └───────────┘
```

### The tool map

| Agent | Tools |
|---|---|
| **Coordinator** | `load_patient_context`; **either** `submit_plan` (no active run) **or** `classify_message` (active run); and while a time is held: `submit_confirmation_verdict`, `list_other_slots`, `propose_another_slot`, `propose_search_window` |
| **Department Routing** | `resolve_department`, `list_departments`, `submit_routing` |
| **Appointment** | `resolve_date`, `find_available_slots`, `find_slots_for_reschedule`, `propose_appointment`, `propose_reschedule`, `propose_cancellation`, `render_confirmation`, `list_my_appointments` |
| **Document** | `list_patient_documents`, `list_unverified_documents`, `read_document_text`, `submit_document_verification`, `diff_required_documents`, `record_missing_documents` |
| **Follow-up** | `list_patient_reminders`, `list_open_tasks` |

Three properties of this table are load-bearing.

**The toolset is a guard, not a convenience.** The wrong tool is *absent*
rather than discouraged. A Coordinator with no active run cannot classify
against one; a Coordinator with a live run cannot quietly start a second. At
`pending_review` — where the run is a queue item and a *person* holds the work —
the slot tools are removed altogether, because a tool that is merely unused is
one a model can still reach for.

**No specialist can commit.** Every appointment tool above either reads or
*proposes*. The proposal is typed state on the `WorkflowRun` row, not a sentence
in the transcript, so the patient's answer is resolved against a column that
survives history windowing, session expiry and restarts.

**No agent holds a transfer tool.** ADK's `transfer_to_agent` hands the
sub-agent the whole session history, which the context contract forbids.
Delegation is dispatched by the orchestrator from the validated plan.

### The context contract

| Who | What they see |
|---|---|
| Coordinator, mid-run | The windowed transcript (last 15 turns) — "how does this message relate to the request already running" is unanswerable without the thing it relates to |
| Coordinator, fresh turn | The current turn plus the patient's own record, and nothing older |
| Every specialist | A typed task and the state it needs. **No history at all.** Where language *is* the job (routing), the request text rides inside the typed task |

The split at the Coordinator exists because the planner and the classifier are
one agent with two jobs and only one of them wants history. Live, a plan step
came back as the word `conflicting` — a classifier word — because the context
began at a previous run's first message.

### Safety is a layer, not an agent

It runs on every inbound message before any planning, in two layers that fail
differently on purpose:

- a **code-owned phrase list** — deterministic, catches self-harm phrasing that
  no severity heuristic would;
- an **LLM pass** — catches a worsening symptom that carries no listed phrase.

Each layer has at least one pinned case that *only that layer* can catch. This
is not decoration: sabotaging the deterministic screen once left every scenario
green, because the model layer happened to subsume it everywhere, and the whole
first layer could have been deleted unnoticed. A second opinion that agrees with
the first is not a check.

The trade is deliberately one-directional — a missed emergency is the worst
outcome this system has — but it is also tuned against the *other* failure,
because a review queue that is mostly noise is a queue nobody reads. Hence
`pain` alone is not an emergency, "prescription" is a document type rather than
a clinical request, and "emergency contact" is a field on a profile.

---

## 4. A turn, in order

`run_workflow` is the orchestrator's entry point and one of the six seams. The
ordering below is the architecture; almost every stage before the Coordinator
exists because a model answered that question wrong at least once.

```
POST /workflow/messages
  │
  └─ turn envelope ──────── opens the trace, guarantees exactly one reply per turn
       │
       ├─ 0.  safety screen ───────── phrase list, then the model pass. First, always,
       │                              whatever state the run is in
       ├─ 0b. listing questions ───── "what do I have?" answered from rows, before
       │                              planning — planning it made the answer a lottery
       ├─ 0c. exact confirmation ──── an exact "yes" to the offer the last turn made
       ├─ 1.  choice reader ────────── "2", or a department name, answering "which
       │                              appointment?" — and it drives the next step itself
       ├─ 1a. selection reader ─────── "the 2 PM one", "option 2" — matched against
       │                              times this patient has actually been shown
       ├─ 1a2. verb switch ─────────── "actually just cancel it instead", read only where
       │                              the held proposal makes "it" a column
       ├─ 1b. confirmation reader ──── exact token or the button; code only
       ├─ 2.  Coordinator ──────────── plan (validated against a closed enum), or
       │                              classify this message against the live run
       ├─ 3.  scope gate ───────────── a plan must be earned by the message
       ├─ 4.  per-step dispatch ────── one specialist per plan step, in code-enforced order
       └─ 5.  commit + receipt ─────── one transaction; the receipt assembled from rows
```

Three things to notice.

**The two numbered-list readers cannot both apply.** "2" against a list of
appointments and "2" against a list of times are the same word answering
different questions, and the run's own state says which was asked: a list of
appointments is outstanding only while no proposal exists, and a list of times
only ever exists once one does. Each list is also **spent when it is answered** —
a stale one captured a "1" meant for the list beside it and read it as an
appointment.

**Everything above stage 2 either reads or holds.** None of it commits. That
asymmetry is what lets the selection reader be generous where the confirmation
reader must be strict: a misread selection costs one decline, a misread
confirmation books against the patient's word.

**Stage 5 does not paraphrase.** After a commit the reply is assembled in
`app/workflow/replies.py` from the rows themselves. Specialists still run, but
what the patient is *told* is code's. One live receipt once welded a mandatory
document to an optional one, printed a reminder's fire date as the
appointment's, and claimed a follow-up task existed beside a line saying none
did — none of which is a prompt that could have been worded better.

---

## 5. The workflow state machine

Runs are created in `in_progress` — or, for an emergency on a session's opening
message, directly in `escalated`, so the `Escalation` row has a run to key to.

| From | May go to |
|---|---|
| `in_progress` | `pending_confirmation` · `pending_review` · `completed` · `cancelled` · `escalated` · `failed` |
| `pending_confirmation` | `in_progress` · `cancelled` · `escalated` |
| `pending_review` | `in_progress` · `rejected` · `cancelled` · `escalated` |
| `completed` `rejected` `failed` `cancelled` `escalated` | *terminal* |

Three rules are enforced at the transition rather than trusted:

1. **Every transition is a compare-and-swap** — the new status is set only where
   the row still holds the expected old one. Zero rows affected means another
   request won the race, and the loser no-ops. The motivating trace is a
   double-clicked Confirm.
2. **Every transition writes both ledgers** — an `AuditEvent` for the row's
   history and a `TraceEvent` for the turn's story. Neither is optional.
3. **Cancellation names its reason** — "withdrawn while pending" and
   "superseded" are different queue facts, recorded at the one moment they are
   known.

Callers never assign `run.status` directly. A corollary that cost two HTTP 500s:
code that *wants* to fail a run must ask the table whether that edge exists
first, because a run waiting on the patient is not this turn's to end.

---

## 6. The trace

`TraceEvent` is separate from `AuditEvent` on purpose: **audit records what a
user did, trace records what the system did.** Nine capture points:

| Point | Records |
|---|---|
| `inbound` | the patient's message, or a button press |
| `guard_verdict` | a deterministic check and which way it went |
| `llm_request` / `llm_response` | the exact payload sent to the provider, and what came back |
| `tool_call` / `tool_result` | what an agent asked for and what it got |
| `validation` | a model proposal accepted **or rejected** |
| `transition` | a state change, beside its audit twin |
| `outbound` | the reply, and its author — `llm`, `template`, or `guard` |

The author field matters: code-authored replies are not invisible, and when code
replaces a model's sentence the author changes with it, or the trace vouches for
words nobody wrote.

Two structural notes. A turn opens *before* its run exists — inbound event,
safety screen, classification — so those rows carry a null run id by design;
selecting a timeline `WHERE workflow_run_id = ?` therefore starts in the middle
of the turn and looks complete while missing exactly the part a reviewer needs.
Find the turn ids first. And PII is redacted at the writer rather than at each
call site, because a rule applied at call sites is a habit.

---

## 7. The six seams

The only public boundaries, and the only places the tests touch.

| Seam | What it isolates |
|---|---|
| **HTTP API** | Everything the UI can do. RBAC and ownership are enforced here, never in the client |
| **Tool functions** | Plain Python, framework-agnostic — no ADK imports |
| **Provider seam** | `BaseLlm` adapters in `app/providers/`: `mock`, `openai`, `groq` |
| **Orchestrator entry** | `run_workflow(...)` and `apply_patient_action(...)` — the only ADK-aware module |
| **Clock** | `clock.today()`. Nothing else in app code calls `date.today()`, so golden tests can freeze time |
| **`api_client`** | Streamlit's only path to the backend |

The provider seam is why the whole application runs with no API key. `mock` is a
first-class provider, not a fixture: it calls the same tools, writes the same
rows, and templates its replies from persisted tool results. If a feature works
live but not under mock, the feature is not done.

It is also a seam with a warning attached. **The understudy cannot falsify
everything.** A bound that lives only in the mock is documentation; a defect
that only a live model produces is invisible to 1,891 passing tests. Several
guards here exist because `scripts/live_sweep.py` found something the suite
could not.

### Why LiteLLM is not used

`google-adk[extensions]` does not install on this platform — its build chain
needs a Rust toolchain. All three providers are our own `BaseLlm` subclasses
calling the `openai` and `groq` SDKs directly. This turned out to be the better
design anyway: identical plumbing for all three, and exact LLM request/response
capture because we own the adapter.

---

## 8. Code map

```
app/
  main.py            FastAPI app; WAL pragma at engine creation; scheduler lifespan
  config.py          pydantic-settings — every secret field is repr=False
  clock.py           the clock seam
  models/            SQLAlchemy models — 16 entities
  auth/              JWT, password hashing, role + ownership dependencies
  api/routers/       thin: validate, delegate, serialize
  agents/            one module per agent; prompts separate from logic; toolbelt binds them
  providers/         mock.py · openai.py · groq.py
  tools/             plain functions — appointments, availability, dates, departments,
                     documents, patients, reminders, tasks, confirmations
  orchestrator.py    run_workflow + apply_patient_action — ADK confined here
  workflow/          state_machine · plan · mapping · confirmation · selection · targets ·
                     queries · replies · recall · staff
  safety/            keyword screen · LLM pass · escalation
  trace/             writer · redaction · well-formedness checker
  scheduler/         poll (the tick) · delivery (the channel) · service (the timer)
ui/                  app.py · views/ · api_client.py · theme.py · shell.py
scripts/             seed.py · live_sweep.py · gate_spike.py · provider_parity.py
```

Tools stay plain functions and prompts stay in their own module **so a LangGraph
fallback stays cheap** — ADK lives only behind `orchestrator.py`.

`ui/views/` rather than `ui/pages/`: `st.navigation` builds the role-based nav,
and Streamlit's automatic `pages/` discovery would compete with it.

---

## 9. Persistence

SQLite file on disk, WAL mode enabled at engine creation because the FastAPI
handlers and the scheduler share one file. The pragma is SQLite-only and a
no-op under Postgres, which the models are otherwise compatible with.

The schema is created from the models via `create_all`; `scripts/seed.py`
populates it and then runs a **self-check** that fails loudly if the seeded
world is not demo-ready — a department with no bookable capacity, a slot in the
past, a booked slot with no appointment, or a non-terminal run left lying
around. Alembic is deliberately not used at this scope.

Two modelling decisions are worth calling out. `DepartmentSynonym` makes routing
vocabulary a **table** rather than a constant, so the deterministic half of
routing is data the staff console can inspect — and its global unique constraint
is a design input rather than an obstacle: a term belongs to exactly one desk, so
a phrase that must *ask* rather than guess is built from two overlapping terms
owned by two departments. And the ADK session store uses a **separate**
`sqlite+aiosqlite:///` URL from the application's `DATABASE_URL`; they are not
unified because `DatabaseSessionService` requires the async driver.

---

## 10. Where A2A and MCP would apply

Agents here are **in-process**. There is no agent-to-agent protocol, and that is
a scope decision rather than an oversight — at five agents behind one
orchestrator, a wire protocol would add serialization, versioning and a second
failure mode without changing a single decision the system makes.

If the deployment split, the boundaries are already drawn:

- **A2A** would sit on the Coordinator→specialist edge, which is the only edge
  that exists. The typed task is already the message: specialists receive no
  history, take a structured task plus the state they need, and return a
  structured result. Making that an RPC is a transport change, not a redesign.
  What would *not* survive the split unchanged is the trace — the nine capture
  points assume one writer and one transaction per turn, so a distributed
  version needs the turn id propagated as a correlation id and the
  well-formedness checker taught to join across services.
- **MCP** would sit at the tool seam. `app/tools/` is already plain Python with
  no framework imports, and the toolbelt is the only thing that binds those
  functions to an agent — so exposing them over MCP is a second binding
  alongside the existing one, not a rewrite. The natural first server is the
  read-only set (`list_departments`, `find_available_slots`,
  `list_patient_documents`), because those carry no consequence and no
  ownership decision beyond the patient id.

The thing that must **not** move across either boundary is the deterministic
bin. Ownership checks, state transitions, the confirmation reader and the commit
transaction stay on the server that owns the database. An MCP tool server that
could book an appointment would put the consequence back on the far side of a
model, which is the one arrangement this architecture exists to prevent.
