# Repeatable use cases: record once with an LLM, replay thousands of times without one

**Status:** proposed — design only, nothing implemented yet
**Author:** drafted 2026-08-23
**Companion:** [`TODO.md`](../../TODO.md) at the repo root is the implementation checklist derived from this document.

---

## 1. The problem

Every browser action today goes through the agent loop in `backend/agent.py`: send history +
page observation to the LLM, get a tool call back, execute it, append the result, repeat. That
is the right design for *figuring out* how to do something. It is the wrong design for doing
the same thing a thousand times.

The cost is not hypothetical. Measured from `data/runs.db`, run `0436a2a8` ("sign in to IXL and
answer a practice question"):

| Measure | Value |
| --- | --- |
| Steps | 35 |
| Tool calls | 34 |
| Tool results that **failed** | 13 of 34 (38%) |
| Total tool-result text | 73,497 chars |
| …of which accessibility snapshots | 64,099 chars (87%) |
| Cumulative text re-sent across turns | ~989,676 chars ≈ **~247,000 input tokens** |

The history is re-sent on every turn, so cost grows roughly triangularly with step count. One
run of one task costs a quarter of a million input tokens. Multiply by 1,000 records and the
approach is economically dead — which is exactly the problem this design solves.

**Goal:** a successful run can be promoted into a stored *use case* — a durable, parameterised
list of steps — that executes against new inputs with **zero LLM calls**.

---

## 2. Why the obvious approach does not work

The obvious approach is "save the `tool_call` events and replay them." Every one of the five
reasons it fails is visible in the runs already in the database.

### 2.1 Element refs are ephemeral

Steps 4 and 5 of run `0436a2a8`:

```
s4  browser_click  {"target": "ref=e17", "element": "Username textbox"}
s5  browser_type   {"target": "input[placeholder=\"Username\"]", "text": "Nitinasati"}
```

`ref=e17` is an index into *one* accessibility snapshot. Reload the page and `e17` is a
different element or nothing at all. Roughly a third of recorded targets are refs. Replaying
them verbatim clicks the wrong thing — silently.

### 2.2 A "successful" run is mostly failed attempts

The model flails and recovers. From the same run:

```
s12 browser_click           input[type="radio"][value="Nayra Asati"]   -> failed
s13 browser_click           input[type="radio"]                        -> failed
s14 browser_run_code_unsafe radios[0].click()                          -> failed
s15 browser_click           div.signin-avatar                          -> failed
s16 browser_run_code_unsafe page.locator('div.signin-avatar').click()  -> succeeded
```

13 of 34 calls failed. A naive recorder replays all 13 failures, burning wall-clock time and
producing an unusable script. **The distiller must keep only what actually worked.**

### 2.3 A third of the steps exist only to feed the model

`browser_snapshot` appears 10 times and produces 87% of the token cost. Its only purpose is to
let the LLM see the page. In a deterministic replay there is no LLM to inform — these steps
carry no action and must be dropped from the recorded script.

### 2.4 Values are baked in

```
"value": "Nitinasati"          <- an input
"value": "<the real account password>"   <- a credential, in plaintext
"url": ".../multiply-decimals-using-area-models"   <- an input
```

Nothing is parameterised. Replaying gives you the same record a thousand times.

### 2.5 Nothing verifies success

The agent decides it is done by *reasoning*. A deterministic replay has no judgement. Without
explicit assertions, a run against 1,000 records fails silently on record 12 and reports success
on all 1,000.

---

## 3. The insight that makes this tractable

Two facts, both verified against real data in this repo.

**Fact 1 — the snapshot format is machine-parseable.** Playwright MCP returns a strict,
regular YAML-ish tree:

```yaml
- navigation "Shortcuts menu" [ref=f2e3]:
  - heading "Skip to" [level=2] [ref=f2e4]
  - link "main content" [ref=f2e7] [cursor=pointer]:
    - /url: "#skippedLink"
```

Every interactive node carries `role`, an accessible `name`, and its `ref`. So
`ref=e17 → (role=textbox, name="Username")` is a **deterministic lookup**, not a guess. We
resolve refs at distillation time against the snapshot captured immediately before that step.

**Fact 2 — snapshots are free during replay.** Snapshots are expensive only because they enter
the *LLM context*. With no LLM in the loop, `browser_snapshot` is just a local tool call costing
milliseconds and zero tokens.

Together these give the core execution strategy:

> Store each step's target as **role + accessible name**. At replay time, take a fresh snapshot,
> parse it, find the node whose role and name match, and use its **live ref**.

This is dramatically more durable than storing CSS. It survives class-name churn, DOM
restructuring, and A/B markup changes — it breaks only when the page's actual semantics change,
which is when a human *should* be involved. And it uses `browser_click`/`browser_type` exactly
as they were designed to be used.

---

## 4. Architecture

Three phases, with a hard rule: **only phase 1 may import an LLM client.**

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │ PHASE 1 · DISTILL          runs once per use case      ~3-5k tokens  │
 │                                                                      │
 │  succeeded run ──► deterministic pre-filter ──► one LLM call ──► UseCase
 │   (34 events)        (drops noise, resolves      (structures &    (8 steps,
 │                       refs → role+name)           parameterises)   versioned)
 └──────────────────────────────────────────────────────────────────────┘
                                    │
 ┌──────────────────────────────────▼───────────────────────────────────┐
 │ PHASE 2 · EXECUTE          runs N times              0 tokens        │
 │                                                                      │
 │  UseCase + input row ──► replay.py ──► MCP ──► events ──► Execution   │
 │                          (locator ladder, assertions, secrets)       │
 └──────────────────────────────────────────────────────────────────────┘
                                    │ on failure, opt-in
 ┌──────────────────────────────────▼───────────────────────────────────┐
 │ PHASE 3 · HEAL             rare, budgeted, off by default            │
 │  failed step ──► agent repairs that one step ──► new UseCase version │
 └──────────────────────────────────────────────────────────────────────┘
```

### Why a separate executor rather than a flag on `BrowserAgent`

`BrowserAgent` is built around an LLM turn: `_loop()` calls `_llm_turn()` before every action.
Threading a "no LLM" mode through it would leave the expensive path one bug away from being
re-entered. A separate `backend/replay.py` that is *never constructed with an LLM client* makes
the zero-token guarantee structural rather than a promise. Both paths share `MCPBrowserSession`,
`policy.py`, `events.py` and `store.py`, so the dashboard, the allowlist and the approval gate
keep working unchanged.

---

## 5. Phase 1 — Distillation

### 5.1 Deterministic pre-filter (no LLM)

Runs first and does the heavy lifting, so the LLM call is small and cheap.

1. **Index snapshots.** Parse every `browser_snapshot` tool result into `{ref → (role, name)}`,
   keyed by the `seq` at which it was captured.
2. **Drop observation-only calls.** `browser_snapshot`, `browser_take_screenshot`,
   `browser_console_messages`, `browser_network_requests` carry no action.
3. **Drop failures.** Join each `tool_call` to its `tool_result` on `call_id`; discard any where
   `ok == false`. This alone removed 13 of 34 calls in the sample run.
4. **Collapse retry clusters.** Consecutive surviving calls that target the same element and
   differ only in locator syntax are one logical action — keep the last one, which is the one
   that worked. Record the discarded ones as *fallback locator candidates*; they are free
   robustness.
5. **Resolve refs.** For any `target` of the form `ref=X`, look up `X` in the snapshot index at
   the nearest preceding `seq` and rewrite it to `role + name`. A ref that cannot be resolved
   is flagged for human review rather than silently kept.
6. **Mine literals.** Collect every typed value, URL and form field value as a parameter
   candidate, cross-referenced against the original task text so the LLM can name them well.

Expected reduction on the sample run: **34 calls → ~8 steps.**

### 5.2 The single LLM call

The pre-filtered step list plus the original task text goes to the model **once**, with the
`UseCase` JSON schema supplied as a tool definition so the output is schema-validated rather
than parsed out of prose (`llm.py` already does tool-use; reuse that path). The model's job is
narrow and well-suited to an LLM:

- name the use case and write a description
- decide which literals are **inputs**, which are **secrets**, and which are constants
- name and type the parameters
- insert `wait_for` conditions where the recorded timing implies one
- propose **assertions** — the part a human must review most carefully

Everything else was already decided deterministically. Budget: one call, a few thousand tokens,
amortised over every future execution.

### 5.3 Human review is part of the flow, not optional

Distillation is a best effort over a noisy recording. The UI **must** present the generated use
case for review and editing before it is marked runnable — reorder or delete steps, rename
parameters, promote a value to a secret, edit assertions, mark a step `optional`. A use case
goes to `status: draft` on creation and only a human moves it to `ready`.

---

## 6. The `UseCase` schema

Versioned (`schema_version`), stored as JSON in SQLite, and immutable once published — edits
create a new version so in-flight batches are not changed under their feet.

```jsonc
{
  "schema_version": 1,
  "id": "uc_9c2f1a…",
  "name": "IXL — sign in and answer a practice question",
  "status": "draft",              // draft | ready | archived
  "version": 3,
  "source_run_id": "0436a2a8…",
  "allowed_domains": ["*.ixl.com"],
  "allow_scripts": false,          // gate for browser_run_code_unsafe steps (§9)

  "inputs": [
    { "name": "practice_url",  "type": "url",    "required": true },
    { "name": "answer",        "type": "string", "required": true }
  ],
  "secrets": [
    { "name": "username",    "required": true },
    { "name": "password",    "required": true,  "description": "IXL account password" },
    { "name": "secret_word", "required": false }
  ],

  // Runs ONCE per batch, on the shared session. Sign-in lives here.
  "setup_steps": [
    {
      "id": "u1",
      "action": "navigate",
      "url": "https://www.ixl.com/signin",
      "wait_for": { "kind": "load_state", "state": "domcontentloaded", "timeout_ms": 15000 }
    },
    {
      "id": "u2",
      "action": "fill",
      "description": "Username textbox",
      "locators": [
        { "strategy": "role", "role": "textbox", "name": "Username" },
        { "strategy": "css",  "selector": "input[placeholder='Username']" },
        { "strategy": "css",  "selector": "input[type='text']" }
      ],
      "value": "{{secret.username}}",
      "optional": false,
      "on_failure": "abort"        // abort | continue | heal
    },
    {
      "id": "u3",
      "action": "fill",
      "locators": [{ "strategy": "role", "role": "textbox", "name": "Password" }],
      "value": "{{secret.password}}"
    },
    {
      "id": "u4",
      "action": "click",
      "locators": [
        { "strategy": "role", "role": "button", "name": "Sign in" },
        { "strategy": "css",  "selector": "button[type='submit']" }
      ]
    }
  ],

  // Cheap proof the shared session is still authenticated. Checked between rows.
  "session_check": { "kind": "url_contains", "value": "/signin", "negate": true },

  // Returns the browser to a known state before each row.
  "row_reset": { "action": "navigate", "url": "{{input.practice_url}}" },

  // Runs ONCE PER INPUT ROW.
  "row_steps": [
    {
      "id": "s1",
      "action": "fill",
      "locators": [{ "strategy": "role", "role": "textbox", "name": "Answer" }],
      "value": "{{input.answer}}"
    },
    {
      "id": "s2",
      "action": "click",
      "locators": [{ "strategy": "role", "role": "button", "name": "Submit" }]
    },
    {
      "id": "s3",
      "action": "assert",
      "assert": { "kind": "text_present", "value": "Correct" },
      "timeout_ms": 20000
    },
    {
      "id": "s4",
      "action": "extract",
      "locators": [{ "strategy": "role", "role": "status", "name": "Score" }],
      "output": "score"            // becomes a column in the results CSV
    }
  ],

  "teardown_steps": [],
  "outputs": ["score"]
}
```

### 6.0 Why the steps are split in three

The batch decision (§8) is *one browser session for the whole file*, so a use case cannot be a
single flat list. Sign-in must not run 1,000 times.

- **`setup_steps`** — once per batch. Almost always the login. Consumes `{{secret.*}}`.
- **`row_steps`** — once per input row. Consumes `{{input.*}}`.
- **`row_reset`** — returns the browser to a known state before each row, so row *n+1* does not
  inherit row *n*'s scroll position, modal, or half-filled form.
- **`session_check`** — a cheap assertion proving the shared session is still authenticated.
- **`teardown_steps`** — optional; once at the end.

**The distiller proposes the split; a human confirms it.** The proposal rule is deterministic and
defensible: every step *before the first step that consumes a per-row input* is a setup
candidate. In the sample run that puts the whole sign-in sequence in `setup_steps` and the
answer-and-submit sequence in `row_steps`, which is correct. It will sometimes be wrong, which is
precisely why §5.3 makes review mandatory.

### 6.1 Action vocabulary

Each action maps to exactly one MCP tool from the discovered set. No action is invented; every
one below exists on the server today.

| Action | MCP tool | Notes |
| --- | --- | --- |
| `navigate` | `browser_navigate` | URL supports templating |
| `click` | `browser_click` | resolves locator → live ref |
| `fill` | `browser_type` | value supports templating |
| `fill_form` | `browser_fill_form` | batches several fields in one call — faster |
| `select` | `browser_select_option` | |
| `press` | `browser_press_key` | |
| `hover` | `browser_hover` | |
| `upload` | `browser_file_upload` | path comes from an input |
| `wait` | `browser_wait_for` | time, or text appears/disappears |
| `assert` | `browser_snapshot` / `browser_evaluate` | evaluated locally, no LLM |
| `extract` | `browser_evaluate` | writes into the execution's output payload |
| `script` | `browser_run_code_unsafe` | **gated** — see §9 |

### 6.2 The locator ladder

`locators` is an ordered list; the executor tries each until one resolves.

1. **`role`** — resolved against a live snapshot to a current ref. Most durable; the default.
2. **`css`** — the selector from the recorded successful call.
3. **`text`** — visible-text match.
4. **`nth`** — positional. Recorded when nothing better exists, and flagged brittle in the UI.

The executor records **which rung matched**. If a use case starts falling through to rung 3 or
4, the site has drifted and the dashboard should say so *before* the whole thing breaks.

**A rung is taken only when it matches exactly one element.** Playwright reads an accessible
name as a case-insensitive *substring* unless told otherwise, so a button recorded as `Invite`
also finds `+ Invite User` — and taking the first of those is how a batch acts on the wrong
control. Codegen writes `exact=True` when it needs to tell two such names apart; the recorded
locator carries it as `exact`, and every named rung is additionally tried in its strict reading
first, so a recording made before that field existed still resolves. When no reading of the
ladder picks a single element the step fails as **ambiguous**, naming what matched, rather than
acting on whichever came first.

### 6.3 Templating rules

`{{input.x}}`, `{{secret.x}}` and `{{env.x}}` are substituted in **value-bearing fields only**:
`url`, `value`, `assert.value`, and `fill_form` field values. Substitution into `locators` is
rejected by the schema validator — a templated selector is a selector-injection hole and makes
locator drift undebuggable. If a genuine use case for dynamic selectors appears, it gets its own
explicit, reviewed field rather than a general escape hatch.

---

## 7. Phase 2 — The executor (`backend/replay.py`)

```python
class UseCaseExecutor:
    def __init__(self, usecase, mcp, sink, secrets):   # note: no `llm` parameter, ever
        async def run_setup(self) -> None: ...            # once per session
        async def run_row(self, inputs: dict) -> Execution: ...   # once per input row
```

The split mirrors §6.0: `run_setup` executes `setup_steps` against a freshly opened session,
`run_row` executes `row_reset` + `row_steps`. A single-row execution from the UI is just
`run_setup` followed by one `run_row`, so there is no second code path to keep in sync with the
batch runner.

Per step:

1. Check the deadline and the step budget.
2. Render templates (secrets pulled from the vault, held only in local scope).
3. Resolve the locator ladder. `role` rung → `browser_snapshot`, parse, match, take live ref.
4. Run the policy gate. `check_navigation` still applies — a use case cannot escape the
   allowlist just because a human approved it once during recording.
5. Call the MCP tool with a bounded retry, reusing the existing backoff schedule.
6. Evaluate `wait_for` / `assert`.
7. On failure, apply `on_failure`: `abort` (default), `continue` (for `optional` steps), or
   `heal` (phase 3, only when explicitly enabled).
8. Emit the **same events** as the agent loop, so `RunView`, the timeline and the WebSocket
   replay work with no frontend changes.

Two new event types are needed, added to `events.py` and mirrored in `frontend/src/lib/events.ts`
(`test_events.py` enforces that mirror):

- `step_started` — `{ step_id, action, description, locator_rung }`
- `step_finished` — `{ step_id, ok, duration_ms, matched_locator, assertion_result }`

### Cost accounting

An execution records `llm_calls` and `llm_tokens`. For a pure replay both are **0**, and the UI
should show that number prominently — it is the entire point of the feature.

---

## 8. Phase 4 — Batch execution

The reason the feature exists: thousands of records. Two decisions shape it, both settled:
**one shared browser session for the whole file**, and **one batch at a time, sequential.**

### 8.1 The execution model

```
open ONE MCP session
  └─ run setup_steps once            ← the login happens here, and only here
     └─ for each row, in order:
          row_reset          → known state
          row_steps          → the actual work
          record execution   → succeeded | failed + outputs
          session_check      → is the shared session still alive?
  └─ run teardown_steps
close session
```

- `POST /api/usecases/{id}/batch` takes a CSV upload (or JSON rows) plus a credential binding.
- Rows are validated against the `inputs` schema **before the browser opens**; a bad column
  fails in a millisecond rather than on record 700.
- Concurrency is **1**. No worker pool, no `BATCH_MAX_CONCURRENCY`. A single-slot lock in
  `RunManager` means a second batch request while one is running returns `409` naming the batch
  in flight, rather than queueing invisibly or fighting over the browser.
- Configurable inter-row delay, default a few hundred ms — politeness, and it keeps a fast use
  case from looking like a denial-of-service to the target site.

### 8.2 Because the session is shared, failure isolation needs a contract

A shared session is fast and avoids 1,000 logins, but it couples the rows: one bad row can leave
a modal open, a form half-filled, or the session logged out. Sequential execution makes the
recovery rules simple and predictable:

1. **A failed row never aborts the batch by itself.** Mark it `failed`, keep its failing step id
   and screenshot, move on.
2. **Reset before every row**, failed or not — `row_reset` runs unconditionally.
3. **Verify the session between rows.** If `session_check` fails, the session is presumed
   logged out: re-run `setup_steps` **once**, then re-check. This is the honest answer to "what
   if it gets logged out halfway through 1,000 rows."
4. **If re-login fails, stop.** Remaining rows stay `pending`, not `failed` — they were never
   attempted, and saying otherwise would corrupt the results file.
5. **Circuit breaker.** After `N` consecutive row failures (default 5) abort the batch. Ten
   minutes of a broken selector failing 400 rows is worse than stopping and telling someone.

### 8.3 Resume

Re-submitting a batch re-runs only rows that are not `succeeded` — which covers all three ways a
batch ends early: re-login failure, circuit breaker, and process restart. Resume opens a new
session and re-runs `setup_steps`, so it is a clean start over the unfinished tail.

### 8.4 Results

CSV export only — no webhooks. One row out per row in, joining the input columns to `status`,
the declared `outputs`, `failed_step_id`, `error`, `duration_ms` and `llm_tokens` (0 unless a
heal fired). Column order is stable so the file diffs cleanly between runs.

---

## 9. Security

### 9.1 Credentials

- Stored in a `credentials` table, encrypted at rest with Fernet; key from a new
  `CREDENTIALS_KEY` env var. **If the key is unset, credential storage is disabled entirely**
  rather than silently falling back to plaintext.
- Referenced only as `{{secret.name}}`. Plaintext exists in one place: a local variable inside
  the executor, at the moment of the tool call.
- **Redacted before persistence.** A redaction pass sits in front of `Store.append_event` and
  replaces any active secret value in tool arguments or result text with `«redacted»`.
- Never logged, never in an artifact, never returned by the API.

### 9.2 Existing plaintext leak — fix before shipping this

This is already a live problem, independent of the new feature:

```
s6 browser_fill_form {"fields":[…{"name":"Password","value":"<real password>"}…]}
```

Real credentials are sitting in `data/runs.db` in plaintext, and `.env` holds a live
`ANTHROPIC_API_KEY` that has already been printed to a terminal in this repo. The redaction pass
and a one-off purge/rotation are **phase 0**, not phase 5 — building a credentials vault on top
of a logger that writes passwords to disk would be worse than not building it.

### 9.3 `browser_run_code_unsafe`

The model reached for it constantly — 9 of 34 calls in the sample run — and it is arbitrary
JavaScript against a live authenticated session. A recorded `script` step is therefore:

- **off by default**: `allow_scripts: false`, and such steps refuse to execute
- **explicitly opted into** by a human during review, per use case
- **shown as raw source** in the review UI, never collapsed

Distillation should actively prefer structured actions and only emit `script` when the
pre-filter shows no structured alternative succeeded.

### 9.4 The allowlist still applies

`check_navigation` runs on every executor tool call, exactly as in the agent loop. A use case
carries its own `allowed_domains`, defaulting to the domains observed during recording — which
is a *tighter* default than the agent's.

---

## 10. Data model

Four new tables, following the existing `store.py` conventions (JSON in `TEXT`, ISO-8601
timestamps, indices on the query paths):

```sql
CREATE TABLE usecases (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    current_version INTEGER NOT NULL DEFAULT 1,
    source_run_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);

CREATE TABLE usecase_versions (            -- immutable; edits append
    usecase_id TEXT NOT NULL, version INTEGER NOT NULL,
    definition TEXT NOT NULL,              -- the UseCase JSON of §6
    created_at TEXT NOT NULL, created_by TEXT,
    PRIMARY KEY (usecase_id, version)
);

CREATE TABLE credentials (
    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
    ciphertext BLOB NOT NULL,              -- Fernet(JSON of {slot: value})
    created_at TEXT NOT NULL, last_used_at TEXT
);

CREATE TABLE batches (
    id TEXT PRIMARY KEY, usecase_id TEXT NOT NULL, version INTEGER NOT NULL,
    status TEXT NOT NULL, total INTEGER NOT NULL,
    succeeded INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0,
    credential_id TEXT, created_at TEXT NOT NULL, finished_at TEXT
);

CREATE TABLE executions (
    id TEXT PRIMARY KEY, batch_id TEXT, usecase_id TEXT NOT NULL,
    version INTEGER NOT NULL, run_id TEXT,   -- FK to runs, for the event timeline
    row_index INTEGER, inputs TEXT NOT NULL, outputs TEXT,
    status TEXT NOT NULL, failed_step_id TEXT, error TEXT,
    llm_calls INTEGER NOT NULL DEFAULT 0, llm_tokens INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER, created_at TEXT NOT NULL
);
```

`store.py` currently creates its schema with `CREATE TABLE IF NOT EXISTS` on every connect and
has no migration mechanism. Adding five tables is safe under that scheme, but altering an
existing one is not — a minimal `schema_version` pragma and a migration step should land with
this work.

---

## 11. API surface

```
POST   /api/runs/{run_id}/distill      -> { usecase_id, version, warnings[] }   (the button)
GET    /api/usecases                   -> list
GET    /api/usecases/{id}              -> definition + versions
PUT    /api/usecases/{id}              -> new version (review edits)
POST   /api/usecases/{id}/publish      -> draft → ready
DELETE /api/usecases/{id}              -> archive

POST   /api/usecases/{id}/execute      -> { execution_id, run_id }   single row, streams live
POST   /api/usecases/{id}/batch        -> { batch_id }   CSV or JSON rows; 409 if one is running
GET    /api/batches/{id}               -> progress + per-row status
GET    /api/batches/{id}/results.csv   -> export
POST   /api/batches/{id}/resume        -> re-run rows that are not `succeeded`
POST   /api/batches/{id}/cancel        -> stop after the current row finishes
GET    /api/executions/active          -> what holds the single execution slot, if anything

POST   /api/credentials                -> { id }   (write-only; values never read back)
GET    /api/credentials                -> names and slots only
DELETE /api/credentials/{id}
```

---

## 12. UI

- **`RunView`** — a **"Save as use case"** button, enabled only when `status === 'succeeded'`.
  This is the entry point the whole feature hangs off.
- **`UseCaseEditor`** (new) — the review screen: ordered step list, each showing action,
  description, the locator ladder with the brittle rungs flagged, and its value. Inline editing,
  reorder, delete, mark optional, promote a literal to input or secret. Publishes draft → ready.
- **`UseCaseList`** (new) — nav item beside "History": name, last run, success rate, drift
  warning.
- **`RunUseCase`** (new) — a form generated from the `inputs` schema for a single run, plus a
  CSV drop zone for a batch, plus a credential picker.
- **`BatchView`** (new) — progress bar, live per-row table, failed-row drill-down into the
  existing timeline, results export.

The dashboard should show **tokens used: 0** on every replay execution. That number is the
feature's whole justification and it should be impossible to miss.

---

## 13. Risks and open questions

| Risk | Mitigation |
| --- | --- |
| Distillation produces a wrong or incomplete script | Mandatory human review; `draft` status; a use case cannot batch-run until published |
| Site markup drifts and locators break | Role+name ladder is drift-tolerant; executor reports which rung matched; drift warning in the list view; optional healing |
| Anti-bot defences trip at volume | Sequential execution, configurable inter-row delay, one steady session. **Out of scope to defeat** — this tool automates sites the user is authorised to use |
| Healing silently re-introduces token cost | Off by default per use case, per-batch token budget, healing events clearly labelled, `llm_tokens` recorded per execution and in the CSV |
| Credentials leak through screenshots | Screenshots are artifacts, not text — redaction cannot reach them. Recommend `screenshot_every_step: false` for use cases that carry secrets |
| Shared session dies mid-batch | `session_check` between rows, one automatic re-login, then stop with remaining rows `pending` (§8.2) |
| Long batch blocks all other work | Single-slot lock returns `409` naming the batch in flight, so the block is explicit rather than mysterious |

### 13.1 Decisions (settled 2026-08-23)

1. **Session reuse — one session for the whole file.** Sign in once, not once per row. This is
   what forced the `setup_steps` / `row_steps` split in §6.0 and the recovery contract in §8.2.
   Playwright MCP's `--storage-state` (`MCP_STORAGE_STATE`) is the mechanism for persisting the
   authenticated state across a resume.
2. **Healing — build it, phase 5.** Kept last so the zero-token path is proven and measurable
   before anything is allowed to spend tokens again.
3. **Outputs — CSV export only.** No webhooks.
4. **Scale — one use case execution at a time.** Concurrency 1, a single-slot lock, no worker
   pool. SQLite stays comfortably adequate; the Postgres note in `store.py` stays a note.

The remaining judgement calls I made rather than asking, all reversible and all flagged in the
text: circuit breaker at 5 consecutive failures, exactly one automatic re-login attempt,
unattempted rows left `pending` rather than `failed`, and the deterministic setup/row split rule
in §6.0.

---

## 14. Phasing

| Phase | Delivers | Independently useful? |
| --- | --- | --- |
| **0 · Secret hygiene** | Redaction pass, purge of existing plaintext, key rotation, snapshot parser + tests | Yes — fixes a live leak |
| **1 · Distillation** | Pre-filter, ref resolution, one-shot LLM distiller, storage, `POST /distill` | Yes — inspect generated use cases as JSON |
| **2 · Executor** | `replay.py`, locator ladder, assertions, single-row execute | **Yes — this is where token cost hits zero** |
| **3 · UI** | Save button, editor, list, run form, credential vault | Yes — usable without batch |
| **4 · Batch** | CSV in, shared session, sequential rows, recovery contract, resume, export | Yes — the thousands-of-records goal |
| **5 · Healing** | Budgeted repair, version bump on repair | Confirmed in scope |

Phases 0–2 are the spine: at the end of phase 2 a use case runs against new inputs with zero
tokens, exercised from `curl`. Everything after is ergonomics and scale.

The detailed task breakdown lives in [`TODO.md`](../../TODO.md).
