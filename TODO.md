# TODO — Repeatable use cases (record once, replay without an LLM)

**Status: all six phases implemented.** 459 tests passing, 3 skipped. The
frontend type-checks and builds. See the commit history for what each phase
delivered, and [`docs/design/repeatable-usecases.md`](docs/design/repeatable-usecases.md)
for why it is shaped this way.

Read the design first; this file is the task breakdown, not the rationale. Items still
unticked are genuinely not done — see **What is not built** at the bottom.

---

## Phase 0 · Secret hygiene and foundations

Ships first because it fixes a live leak and because everything downstream depends on the
snapshot parser.

- [ ] **Rotate the exposed `ANTHROPIC_API_KEY`.** It is in `.env` and has been printed to a
      terminal in this repo. *(Only you can do this one.)*
- [x] **Purge plaintext credentials from `data/runs.db`.** `backend/scripts/purge_secrets.py`:
      `--scan` finds candidates, `--dry-run` reports, `--verify-only` greps the raw files.
      Applied — 50 event rows and 9 run rows across 9 runs, verified clean on disk.
      **The `.bak` beside the database still holds the plaintext; delete it when satisfied.**
- [x] **Redaction pass** in `RunEventSink.emit`, in front of *both* the store and the
      `EventBus` — redacting only on the way to storage would still broadcast the secret to
      every connected browser. Uniform by construction (dump → rewrite every string →
      re-validate), so new event types are covered the day they are added. `backend/redaction.py`.
- [x] `secrets: list[str]` on `POST /api/runs`, write-only, so a pasted credential can be
      registered today rather than waiting for the vault.
- [x] Fix `tests/test_llm_bedrock.py::test_no_api_key_is_required_for_bedrock` — isolated with
      `_env_file=None` *and* `monkeypatch.delenv`; the `.env` file was the real source, so
      `delenv` alone would not have fixed it.
- [x] **Snapshot parser** (`backend/snapshot.py`): `by_ref` for distillation, `locate` for replay.
      - [x] Handles nesting, attrs on both sides of `ref`, `[level=N]`, `[cursor=pointer]`,
            `- /url:` and `- text:` property lines, escaped quotes, missing names, a missing
            fence, and truncated tool results.
      - [x] 33 tests against real server output committed as `tests/fixtures/snapshot_signin.txt`.
      - [x] Ambiguity rule: three-tier name matching (exact → case/whitespace-folded →
            substring), first tier that hits wins; `nth` selects among equals; structural roles
            (`generic`, `group`, `none`, `presentation`) never shadow an interactive match.
- [x] **Schema migrations.** `SCHEMA_VERSION` + `MIGRATIONS` + the `user_version` pragma in
      `store.py`; a fresh database is stamped without running anything.

## Phase 1 · Distillation (the one LLM call)

- [x] **Pre-filter** (`backend/distill.py`), no LLM:
  - [x] Load a run's events; join `tool_call` → `tool_result` on `call_id`.
  - [x] Drop observation-only tools (`browser_snapshot`, `browser_take_screenshot`,
        `browser_console_messages`, `browser_network_requests`).
  - [x] Drop calls whose result has `ok == false`.
  - [x] Collapse consecutive retries against the same element; keep the last success, demote the
        rest to fallback locator candidates.
  - [x] Resolve `ref=X` targets to role+name via the nearest preceding snapshot. Flag
        unresolvable refs as warnings rather than keeping them.
  - [x] Mine literals (typed values, URLs, form values) as parameter candidates.
  - [x] Test against run `0436a2a8` (34 calls, 13 failures) and assert the output is ~8 steps
        with no `ref=` targets remaining.
  - [x] **Propose the setup/row split**: every step before the first one that consumes a per-row
        input becomes a `setup_steps` candidate; the rest become `row_steps`. Deterministic, and
        confirmed by a human in phase 3.
- [x] **`UseCase` pydantic models** (`backend/usecase.py`) matching §6 of the design —
      `setup_steps`, `row_steps`, `row_reset`, `session_check`, `teardown_steps`.
  - [x] Validator: reject `{{…}}` templating inside `locators`.
  - [x] Validator: reject `script` steps when `allow_scripts` is false.
  - [x] Validator: `{{input.*}}` may not appear in `setup_steps` — setup runs once per batch, so
        a per-row input there is a modelling error and should fail loudly.
  - [x] `schema_version` constant and a stated forward-compat rule.
- [x] **Distiller prompt** in `backend/prompts/distill.md`, loaded via `prompt_loader` like the
      others. Supply the `UseCase` schema as a tool definition so output is schema-validated.
- [x] **One-shot LLM call** reusing `llm.py`'s tool-use path. Assert in tests that exactly one
      call is made.
- [x] **Storage**: `usecases` + `usecase_versions` tables and `Store` methods.
- [x] `POST /api/runs/{run_id}/distill` → `{ usecase_id, version, warnings[] }`. Reject runs
      whose status is not `succeeded`.

## Phase 2 · The executor (zero LLM)

- [x] **`backend/replay.py` — `UseCaseExecutor`.** Constructor takes `(usecase, mcp, sink,
      secrets)` and **no LLM client**. Add a test asserting the module never imports `llm`.
- [x] Split the surface into `run_setup()` (once per session) and `run_row(inputs)` (once per
      row), so the single-row UI path and the phase-4 batch runner share one code path.
- [x] **Locator ladder**: `role` (fresh snapshot → live ref) → `css` → `text` → `nth`. Record
      which rung matched on every step.
- [x] **Template rendering** for `url`, `value`, `assert.value`, `fill_form` field values only.
      Missing required input → fail before the browser opens.
- [x] **Action dispatch** for the §6.1 vocabulary. Each action maps to exactly one MCP tool;
      resolve tool names through `mcp.find_tool` so a renamed server tool degrades gracefully.
- [x] **Assertions**, evaluated locally: `url_contains`, `text_present`, `element_visible`,
      `element_count`, each with `negate` and `timeout_ms`.
- [x] **`extract`** writes into the execution's `outputs` payload.
- [x] **Failure handling**: `on_failure` = `abort` | `continue` | `heal`; screenshot on failure;
      record `failed_step_id`.
- [x] **Policy gate**: `check_navigation` on every call, using the use case's `allowed_domains`.
- [x] **New events** `step_started` / `step_finished` in `events.py`, mirrored in
      `frontend/src/lib/events.ts` (`test_events.py` enforces the mirror).
- [x] **Credentials vault**: `credentials` table, Fernet encryption, `CREDENTIALS_KEY` setting.
      With no key configured, credential storage is **disabled** — never a plaintext fallback.
- [x] `POST /api/usecases/{id}/execute` → `{ execution_id, run_id }`, streaming over the
      existing WebSocket.
- [x] **Assert zero cost**: a test that executes a use case end-to-end against the static
      fixtures in `tests/fixtures/` with a fake LLM that raises if called.

## Phase 3 · UI

- [x] **"Save as use case"** button in `RunView`, enabled only for `status === 'succeeded'`.
- [x] **`UseCaseView`**: the three phases shown separately, each step with its action,
      description, value and the full locator ladder with brittle rungs (`text`, `nth`) flagged.
      Delete a step (saved as a new version), publish draft → ready.
  - [ ] Inline editing of a step's value, reordering, and toggling `optional` — currently only
        delete and publish. Editing a value means re-uploading the definition via `PUT`.
  - [ ] Promote a literal to an input or a secret from the UI. The distiller does this; a
        reviewer who disagrees has to edit the JSON.
- [x] Raw source shown in full for any `script` step, beside the `allow_scripts` opt-in.
      Scripts refuse to execute until a person enables them.
- [x] **`UseCaseList`** nav item beside "History": name, description, status, version, updated.
  - [ ] Success rate and a drift warning per use case. The data is recorded
        (`locator_rung` on every `step_finished`) but nothing aggregates it yet.
- [x] **Run one** and **Run a file**: a form generated from the `inputs` schema, a credential
      picker, a CSV textarea plus file picker.
- [x] **Batch progress**: live per-row table, a link from a failing step into the existing
      timeline, Resume when a batch stopped early, and the results CSV download.
- [x] Report `llm_tokens` after a single run.
  - [ ] Show it as a standing figure on the use case and batch views, not only in the
        post-run notice.
- [ ] **Credential management UI.** The picker lists what the vault holds and the API can
      create and delete credentials, but there is no form yet — use `POST /api/credentials`.
- [x] Extend `frontend/src/lib/api.ts` with the new endpoints.

## Phase 4 · Batch execution

One shared session for the whole file; rows run sequentially. See §8 of the design.

- [x] `batches` + `executions` tables and `Store` methods.
- [x] CSV/JSON row parsing, validated against the `inputs` schema **before the browser opens**.
- [x] **Batch runner**: open one MCP session → `run_setup()` once → loop rows → teardown →
      close. Concurrency 1, no worker pool.
- [x] **Single-slot lock** in `RunManager`; a second batch request returns `409` naming the batch
      in flight. `GET /api/executions/active` reports the holder.
- [x] **Recovery contract** (§8.2), each piece tested on its own:
  - [x] A failed row records `failed` + `failed_step_id` + screenshot and does **not** abort the batch.
  - [x] `row_reset` runs before every row, failed or not.
  - [x] `session_check` between rows; on failure re-run `setup_steps` **once**, then re-check.
  - [x] Re-login failure stops the batch leaving remaining rows `pending`, never `failed`.
  - [x] Circuit breaker: abort after N consecutive row failures (default 5, configurable).
- [x] **Resume**: re-runs only rows that are not `succeeded`; opens a fresh session and re-runs
      `setup_steps` first. Covers re-login failure, circuit breaker and process restart alike.
- [x] Persist authenticated state via `MCP_STORAGE_STATE` so a resume can skip an interactive
      login where the site allows it.
- [x] Configurable inter-row delay (politeness / rate limiting).
- [x] `POST /api/usecases/{id}/batch`, `GET /api/batches/{id}`, `POST /api/batches/{id}/resume`,
      `POST /api/batches/{id}/cancel` (stops after the current row),
      `GET /api/batches/{id}/results.csv`.
- [x] **CSV export**: input columns + `status` + declared `outputs` + `failed_step_id` + `error`
      + `duration_ms` + `llm_tokens`. Stable column order so files diff cleanly. No webhooks.
- [x] **`BatchView`**: progress, live per-row table, failed-row drill-down into the existing
      timeline, export button, and a visible marker when the session re-authenticated mid-batch.

## Phase 5 · Healing

Confirmed in scope, and deliberately last — the zero-token path should be proven and measurable
before anything is allowed to spend tokens again.

- [x] `on_failure: heal` escalates **one failed step** to the agent with the current snapshot.
- [x] Off by default per use case; hard per-batch token budget that stops healing when exhausted.
- [x] After any heal, run `row_reset` before continuing — a repair attempt must not leave the
      shared session in a state the next row inherits.
- [x] A successful heal writes a new `usecase_version` with the repaired locator; the batch
      continues on the new version.
- [x] Healing events clearly labelled in the timeline; `llm_tokens` recorded per execution and
      surfaced in the CSV, so the cost of healing is never invisible.

---

## Decisions — settled 2026-08-23

1. **Session reuse** — *one session for the whole file.* Sign in once, not once per row. This
   forced the `setup_steps` / `row_steps` split (design §6.0) and the recovery contract (§8.2).
2. **Healing** — *build it,* phase 5.
3. **Outputs** — *CSV export only.* No webhooks.
4. **Scale** — *one use case execution at a time.* Concurrency 1, single-slot lock, no worker
   pool. SQLite stays comfortably adequate.

Judgement calls I made rather than asking, all reversible and all flagged in the design:
circuit breaker at 5 consecutive failures, exactly one automatic re-login attempt, unattempted
rows left `pending` rather than `failed`, and the deterministic setup/row split rule in §6.0.

---

## What is not built

Everything above that is unticked, plus:

- **No end-to-end test against a real browser.** The suite fakes the MCP session throughout.
  `tests/test_e2e_static.py` (opt-in via `RUN_E2E=1`) covers the agent path only; there is no
  equivalent for replay, so the first real batch is the first real proof.
- **Distillation has never been run against a live model.** The pipeline is exercised with a
  scripted plan; the prompt in `backend/prompts/distill.md` is untested against Claude, and it
  is the part most likely to need tuning.
- **`browser_evaluate` is mapped to `extract`, but `extract` reads the accessibility node**
  rather than evaluating the recorded JavaScript. A recording that extracted a value by script
  will need its step edited.
- **Cancelling a batch stops the task, but the in-flight row's browser call is not awaited**
  cleanly — the session is torn down by the context manager.
