# Framework gate — ADK vs LangGraph

**Decision: Google ADK stays. LangGraph remains a fallback, not a plan.**
Taken 2026-07-26, end of Phase 3. Re-evidenced against the real system at the
end of Phase 4. Everything below is reproducible.

```powershell
.\.venv\Scripts\python.exe scripts\gate_spike.py --live      # the five boxes
.\.venv\Scripts\python.exe scripts\provider_parity.py        # mock vs live trace shape
```

Both scripts reset and re-seed the database first. Without that, a run left
active by a previous invocation would make the first turn classify against it
instead of planning fresh, and the gate would be measuring its own leftovers.

---

## What changed after Phase 4, and why the scripts were rewritten

The gate was originally answered by a **spike**: a throwaway two-agent
hub-and-spoke wired with ADK's `transfer_to_agent`, built to test the framework
before any of the system existed. It answered yes.

Phase 4 then built the real thing, and deliberately did **not** use
`transfer_to_agent`. A transfer hands the sub-agent the entire session history,
and the context contract says specialists receive no history at all — only a
typed task, with the patient's words riding inside it where language is the job.
Delegation is therefore dispatched by `run_workflow` from the validated plan.
Hub-and-spoke is intact: the Coordinator decides *which* specialists a request
needs, and code enforces the order.

That left the spike script driving a topology the system no longer had, and it
began reporting `GATE: 2/5` for a decision that had not changed. A script
reporting a failed gate for a passing system is worse than no script, so both
scripts now drive `run_workflow`. **The boxes are unchanged; the evidence is
now the code that ships.**

---

## The five boxes

Each was evidenced by running, not argued. `--live` includes box 1, which needs
a Groq key in `.env`.

| # | Box | Result | Evidence |
|---|---|---|---|
| 1 | An agent answers via Groq through **our own** `BaseLlm` adapter | ✅ | Adapter chain `GroqLlm → OpenAICompatibleLlm → AgentCareLlm → BaseLlm`, model `llama-3.3-70b-versatile`. The live model drove the whole workflow: **10 tool calls** across `coordinator`, `department_routing`, `appointment`, ending in a real proposal. **No LiteLLM** — it does not install here. |
| 2 | Trace rows from **multiple agents** in one hub-and-spoke turn | ✅ | 41 rows scoped to the run's own session; `agents: ['appointment', 'coordinator', 'department_routing']`. Callbacks reached the capture points with ADK's exact keyword parameter names. |
| 3 | Restart the process; a follow-up message still works | ✅ | pid 20436 → 27152. The second interpreter's "yes" landed on a run the first one created: `pending_confirmation → completed`, steps `[book, documents, follow_up]`. The restart is a **real second interpreter**, spawned via `subprocess`. |
| 4 | State written before the restart is readable after | ✅ | Before acting, pid 27152 read `department='Cardiology'`, `proposed_slot=1`, and replayed **6 ADK conversation events** from `sqlite+aiosqlite`. Both values are written only during routing and proposal — both of which happened in pid 20436 — and both are read *before* the second process writes anything. |
| 5 | The mock `BaseLlm` short-circuits the real LLM and drives the loop | ✅ | The mock ran the full workflow — plan, routing, date resolution, slot search, proposal — with **7 tool calls** and no network call. |

### Why an earlier run reporting 5/5 was wrong

Worth keeping, because the failure mode is the interesting part and the same
trap was available again when these scripts were rewritten.

* **Box 2 was vouched for by a previous crashed run.** The trace query was not
  scoped to the session under test, so 36 stale rows — some from runs that died
  mid-turn — were counted as this run's evidence. Now scoped to `session_id`,
  and the parent resets the database before it starts.
* **Box 4 was satisfied by `last_request_id`**, a key *every* turn writes. Its
  presence after a restart proved nothing at all. The replacement is
  deliberately chosen the other way: the department and the proposed slot are
  written only while routing and proposing, and are read by the second process
  *before* it acts.

Tightening both dropped the honest score to 3/5, and two real bugs stood between
that and a genuine 5/5.

### And one more, found while rewriting

Box 4 failed on the first run of the new script, reporting **0 conversation
events replayed**. The cause was in the check, not the system: it asked the ADK
session store for a session keyed by the patient's *email*, while `run_workflow`
keys sessions by the numeric user id. A lookup that finds nothing reports zero,
and zero looks exactly like "state did not survive". The row-state half of the
box was passing at the time, which is what made the contradiction visible.

---

## Two bugs found by the original spike, both in the mock, neither in ADK

**1. The specialist was reading ADK's narration as the patient's message.**
On a transfer, ADK does not hand the sub-agent a `function_response`. It injects
`role="user"` prose:

```
For context:
[coordinator] called tool `transfer_to_agent` with parameters: {'agent_name': ...}
[coordinator] `transfer_to_agent` tool returned result: {'result': None}
```

`latest_user_text` returned that framing, so the policy saw no booking cue and
fell through to a scope reply. Fixed at the seam (`ADK_CONTEXT_MARKERS`).

This is the finding that shaped Phase 4's topology. The context contract says
that where language *is* the job, the specialist receives the patient's words —
and ADK's default delegation actively obscures them. Rather than filter the
narration forever, the orchestrator now passes the request text explicitly in a
typed task and no transfer happens at all. The filter remains, and remains
pinned by `tests/unit/test_provider_translation.py`, which records the narration
verbatim so an ADK upgrade that rewords it fails loudly.

**2. Delegation was unbounded.** ADK gives the *specialist* a
`transfer_to_agent` tool too, so when the mock wanted a tool the specialist
lacked, it bounced the work onward and kept bouncing until ADK's session store
failed with a stale-session error.

That crash was **the iteration-budget problem arriving early**, exactly as the
PRD anticipated. Phase 4's caps are not theoretical — and building them turned
up the sharper version of the same lesson: refusing a *tool* does not stop a
loop, because the model is simply asked again. The budget also has to stop the
turn asking the model, or ADK's own implicit 500-call limit is what actually
ends the run — the framework limit the budget exists so as not to depend on.

Under the current topology, neither bug is reachable: specialists hold no
transfer tool and receive no history.

---

## Provider parity

Same scenario — *"I need a cardiology appointment next week"* — under `mock` and
under `groq`, through the real `run_workflow`, traces diffed for shape. The two
providers run against different patients, because one active run per patient is
a rule the system enforces and sharing one would mean the second provider
classified a message against the first's run instead of planning its own.

| Check | Result |
|---|---|
| Same capture-point kinds under both | ✅ `inbound, llm_request, llm_response, outbound, tool_call, tool_result, transition, validation` |
| Every LLM request has exactly one terminal partner | ✅ mock 10/10, live 13/13 |
| Same author vocabulary | ✅ `guard`, `llm`, `patient-message`, `system` |
| Both parse against the turn grammar | ✅ clean |

**A correction to the Phase 3 record.** That run reported the two shapes as
*byte-identical*, and this one does not: 41 events under mock against 53 under
Groq. Nothing regressed — the earlier spike gave both providers one small
toolset and one obvious move, so they made the same number of calls. Against
five agents and real toolsets a live model legitimately chooses differently.
Byte-identity was never the criterion; the four checks above are, and the shape
that matters is the skeleton, not its length.

---

## What this buys, and what would reverse it

ADK is carrying: the agent loop, `FunctionTool` dispatch, the nine callback
capture points, and Tier-1 memory via `DatabaseSessionService`. Hub-and-spoke
delegation is ours, by choice, for the context-contract reason above.

The fallback stays cheap by construction — tools are plain functions returning
JSON-able dicts, prompts live apart from logic, the deterministic core in
`app/workflow/` imports no framework at all, and ADK is confined to
`orchestrator.py` and `app/agents/`. Nothing found here argues for spending that
option.

Reversing the decision would take something ADK's design prevents rather than
something it merely made awkward: an inability to see or bound the agent loop,
or session state that cannot be trusted across a restart. Both were tested
directly, twice now, and both hold.
