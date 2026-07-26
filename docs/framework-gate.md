# Framework gate — ADK vs LangGraph

**Decision: Google ADK stays. LangGraph remains a fallback, not a plan.**
Taken 2026-07-26, end of Phase 3. Evidence below is reproducible.

```powershell
.\.venv\Scripts\python.exe scripts\gate_spike.py --live      # the five boxes
.\.venv\Scripts\python.exe scripts\provider_parity.py        # mock vs live trace shape
```

---

## The five boxes

Each was evidenced by running, not argued. `--live` includes box 1, which needs
a Groq key in `.env`.

| # | Box | Result | Evidence |
|---|---|---|---|
| 1 | An agent answers via Groq/OpenAI through **our own** `BaseLlm` adapter | ✅ | Adapter chain `GroqLlm → OpenAICompatibleLlm → AgentCareLlm → BaseLlm`. Groq called `list_departments`, produced real `tool_call`/`tool_result` rows, replied *"You can book an appointment with the following departments: Cardiology…"*. **No LiteLLM** — it does not install here. |
| 2 | Trace rows from **both** agents in a two-agent hub-and-spoke | ✅ | 10 rows scoped to the run's own session; `agents: ['coordinator', 'department_specialist']`. Callbacks reached the capture points with ADK's exact keyword parameter names. |
| 3 | Restart the process; a follow-up message still works | ✅ | pid 28856 → 13168, 8 events replayed from `sqlite+aiosqlite:///`. The restart is a **real second interpreter**, spawned via `subprocess`. |
| 4 | A `state[...]` value written before the restart is readable after | ✅ | `tool_resolve_department`, written by `before_tool_callback` in pid 28856, read in pid 13168. Excluded per-turn keys, which both processes write. |
| 5 | The mock `BaseLlm` short-circuits the real LLM and drives the same loop | ✅ | Mock ran the full ADK runner — delegation, tool call, reply — with no network call. |

### Why an earlier run reporting 5/5 was wrong

Worth recording, because the failure mode is the interesting part.

* **Box 2 was vouched for by a previous crashed run.** The trace query was not
  scoped to the session under test, so 36 stale rows — some from runs that died
  mid-turn — were counted as this run's evidence. Now scoped to `session_id`.
* **Box 4 was satisfied by `last_request_id`**, a key *every* turn writes. Its
  presence after a restart proved nothing at all. Now excluded explicitly.

Tightening both dropped the honest score to 3/5, and the two real bugs below
were what stood between that and a genuine 5/5.

---

## Two bugs found, both in the mock, neither in ADK

**1. The specialist was reading ADK's narration as the patient's message.**
On a transfer, ADK does not hand the sub-agent a `function_response`. It injects
`role="user"` prose:

```
For context:
[coordinator] called tool `transfer_to_agent` with parameters: {'agent_name': ...}
[coordinator] `transfer_to_agent` tool returned result: {'result': None}
```

`latest_user_text` returned that framing, so the policy saw no booking cue and
fell through to a scope reply. Fixed at the seam (`ADK_CONTEXT_MARKERS`), which
fixes it for all three providers.

This is not a mock-only concern. The context contract says that where language
*is* the job, the specialist receives the patient's words — and ADK's default
delegation actively obscures them. Live routing would have classified the
delegation machinery instead of the request. Pinned by
`tests/unit/test_provider_translation.py`, which records the narration verbatim
so an ADK upgrade that rewords it fails loudly.

**2. Delegation was unbounded.** ADK gives the *specialist* a
`transfer_to_agent` tool too, so when the mock wanted a tool the specialist
lacked, it bounced the work onward and kept bouncing until ADK's session store
failed with a stale-session error. The mock now delegates at most once per turn.

That crash is **the iteration-budget problem arriving early**, exactly as the
PRD anticipated — Phase 4's caps (8 tool iterations/turn, one re-plan/run) are
not theoretical. It also showed ADK's session store fails *loudly* on
concurrent appends rather than silently interleaving, which is the better
failure mode.

---

## Provider parity (Phase 3's done-when)

Same scenario — *"I need a cardiology appointment next week"* — under `mock` and
under `groq`, traces diffed for shape. Wording differs by design; the skeleton
must not.

| Check | Result |
|---|---|
| Same capture-point kinds under both | ✅ `inbound, llm_request, llm_response, tool_call, tool_result, outbound` |
| Every LLM request has exactly one terminal partner | ✅ mock 3/3, live 3/3 |
| Same author vocabulary | ✅ `llm`, `patient-message` |
| Both parse against the turn grammar | ✅ clean |

The two shapes came out **byte-identical**, which is stronger than the check
requires. Mock-mode tests are therefore exercising the same skeleton the
shipped system uses.

---

## What this buys, and what would reverse it

ADK is carrying: the agent loop, hub-and-spoke delegation, `FunctionTool`
dispatch, the nine callback capture points, and Tier-1 memory via
`DatabaseSessionService`.

The fallback stays cheap by construction — tools are plain functions returning
JSON-able dicts, prompts live apart from logic, and ADK is confined behind the
orchestrator. Nothing found here argues for spending that option.

Reversing the decision would take something ADK's design prevents rather than
something it merely made awkward: an inability to see or bound the agent loop,
or session state that cannot be trusted across a restart. Both were tested
directly, and both hold.

**One consequence to carry into Phase 4:** ADK's delegation narration means the
Coordinator must pass the patient's request text into a specialist's typed task
explicitly, rather than relying on the transferred history to carry it. That is
what the context contract already requires — the gate just showed what happens
when it is left implicit.
