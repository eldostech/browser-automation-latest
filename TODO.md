# TODO — Repeatable use cases (record once, replay without an LLM)

Implementation checklist for [`docs/design/repeatable-usecases.md`](docs/design/repeatable-usecases.md).
Read the design first; this file is the task breakdown, not the rationale.

Phases are ordered by dependency. **Phases 0–2 are the spine** — at the end of phase 2 a use
case executes against new inputs with zero LLM tokens. Everything after is ergonomics and scale.

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

- [ ] **Pre-filter** (`backend/distill.py`), no LLM:
  - [ ] Load a run's events; join `tool_call` → `tool_result` on `call_id`.
  - [ ] Drop observation-only tools (`browser_snapshot`, `browser_take_screenshot`,
        `browser_console_messages`, `browser_network_requests`).
  - [ ] Drop calls whose result has `ok == false`.
  - [ ] Collapse consecutive retries against the same element; keep the last success, demote the
        rest to fallback locator candidates.
  - [ ] Resolve `ref=X` targets to role+name via the nearest preceding snapshot. Flag
        unresolvable refs as warnings rather than keeping them.
  - [ ] Mine literals (typed values, URLs, form values) as parameter candidates.
  - [ ] Test against run `0436a2a8` (34 calls, 13 failures) and assert the output is ~8 steps
        with no `ref=` targets remaining.
  - [ ] **Propose the setup/row split**: every step before the first one that consumes a per-row
        input becomes a `setup_steps` candidate; the rest become `row_steps`. Deterministic, and
        confirmed by a human in phase 3.
- [ ] **`UseCase` pydantic models** (`backend/usecase.py`) matching §6 of the design —
      `setup_steps`, `row_steps`, `row_reset`, `session_check`, `teardown_steps`.
  - [ ] Validator: reject `{{…}}` templating inside `locators`.
  - [ ] Validator: reject `script` steps when `allow_scripts` is false.
  - [ ] Validator: `{{input.*}}` may not appear in `setup_steps` — setup runs once per batch, so
        a per-row input there is a modelling error and should fail loudly.
  - [ ] `schema_version` constant and a stated forward-compat rule.
- [ ] **Distiller prompt** in `backend/prompts/distill.md`, loaded via `prompt_loader` like the
      others. Supply the `UseCase` schema as a tool definition so output is schema-validated.
- [ ] **One-shot LLM call** reusing `llm.py`'s tool-use path. Assert in tests that exactly one
      call is made.
- [ ] **Storage**: `usecases` + `usecase_versions` tables and `Store` methods.
- [ ] `POST /api/runs/{run_id}/distill` → `{ usecase_id, version, warnings[] }`. Reject runs
      whose status is not `succeeded`.

## Phase 2 · The executor (zero LLM)

- [ ] **`backend/replay.py` — `UseCaseExecutor`.** Constructor takes `(usecase, mcp, sink,
      secrets)` and **no LLM client**. Add a test asserting the module never imports `llm`.
- [ ] Split the surface into `run_setup()` (once per session) and `run_row(inputs)` (once per
      row), so the single-row UI path and the phase-4 batch runner share one code path.
- [ ] **Locator ladder**: `role` (fresh snapshot → live ref) → `css` → `text` → `nth`. Record
      which rung matched on every step.
- [ ] **Template rendering** for `url`, `value`, `assert.value`, `fill_form` field values only.
      Missing required input → fail before the browser opens.
- [ ] **Action dispatch** for the §6.1 vocabulary. Each action maps to exactly one MCP tool;
      resolve tool names through `mcp.find_tool` so a renamed server tool degrades gracefully.
- [ ] **Assertions**, evaluated locally: `url_contains`, `text_present`, `element_visible`,
      `element_count`, each with `negate` and `timeout_ms`.
- [ ] **`extract`** writes into the execution's `outputs` payload.
- [ ] **Failure handling**: `on_failure` = `abort` | `continue` | `heal`; screenshot on failure;
      record `failed_step_id`.
- [ ] **Policy gate**: `check_navigation` on every call, using the use case's `allowed_domains`.
- [ ] **New events** `step_started` / `step_finished` in `events.py`, mirrored in
      `frontend/src/lib/events.ts` (`test_events.py` enforces the mirror).
- [ ] **Credentials vault**: `credentials` table, Fernet encryption, `CREDENTIALS_KEY` setting.
      With no key configured, credential storage is **disabled** — never a plaintext fallback.
- [ ] `POST /api/usecases/{id}/execute` → `{ execution_id, run_id }`, streaming over the
      existing WebSocket.
- [ ] **Assert zero cost**: a test that executes a use case end-to-end against the static
      fixtures in `tests/fixtures/` with a fake LLM that raises if called.

## Phase 3 · UI

- [ ] **"Save as use case"** button in `RunView`, enabled only for `status === 'succeeded'`.
- [ ] **`UseCaseEditor`**: step list with action, description, locator ladder (brittle rungs
      flagged), value. Inline edit, reorder, delete, mark optional, promote literal → input or
      secret. Publish draft → ready.
- [ ] Raw source shown for any `script` step, with the `allow_scripts` opt-in beside it.
- [ ] **`UseCaseList`** nav item beside "History": name, last run, success rate, drift warning.
- [ ] **`RunUseCase`**: form generated from the `inputs` schema, credential picker, CSV drop zone.
- [ ] Show **tokens used: 0** on replay executions.
- [ ] Extend `frontend/src/lib/api.ts` with the new endpoints.

## Phase 4 · Batch execution

One shared session for the whole file; rows run sequentially. See §8 of the design.

- [ ] `batches` + `executions` tables and `Store` methods.
- [ ] CSV/JSON row parsing, validated against the `inputs` schema **before the browser opens**.
- [ ] **Batch runner**: open one MCP session → `run_setup()` once → loop rows → teardown →
      close. Concurrency 1, no worker pool.
- [ ] **Single-slot lock** in `RunManager`; a second batch request returns `409` naming the batch
      in flight. `GET /api/executions/active` reports the holder.
- [ ] **Recovery contract** (§8.2), each piece tested on its own:
  - [ ] A failed row records `failed` + `failed_step_id` + screenshot and does **not** abort the batch.
  - [ ] `row_reset` runs before every row, failed or not.
  - [ ] `session_check` between rows; on failure re-run `setup_steps` **once**, then re-check.
  - [ ] Re-login failure stops the batch leaving remaining rows `pending`, never `failed`.
  - [ ] Circuit breaker: abort after N consecutive row failures (default 5, configurable).
- [ ] **Resume**: re-runs only rows that are not `succeeded`; opens a fresh session and re-runs
      `setup_steps` first. Covers re-login failure, circuit breaker and process restart alike.
- [ ] Persist authenticated state via `MCP_STORAGE_STATE` so a resume can skip an interactive
      login where the site allows it.
- [ ] Configurable inter-row delay (politeness / rate limiting).
- [ ] `POST /api/usecases/{id}/batch`, `GET /api/batches/{id}`, `POST /api/batches/{id}/resume`,
      `POST /api/batches/{id}/cancel` (stops after the current row),
      `GET /api/batches/{id}/results.csv`.
- [ ] **CSV export**: input columns + `status` + declared `outputs` + `failed_step_id` + `error`
      + `duration_ms` + `llm_tokens`. Stable column order so files diff cleanly. No webhooks.
- [ ] **`BatchView`**: progress, live per-row table, failed-row drill-down into the existing
      timeline, export button, and a visible marker when the session re-authenticated mid-batch.

## Phase 5 · Healing

Confirmed in scope, and deliberately last — the zero-token path should be proven and measurable
before anything is allowed to spend tokens again.

- [ ] `on_failure: heal` escalates **one failed step** to the agent with the current snapshot.
- [ ] Off by default per use case; hard per-batch token budget that stops healing when exhausted.
- [ ] After any heal, run `row_reset` before continuing — a repair attempt must not leave the
      shared session in a state the next row inherits.
- [ ] A successful heal writes a new `usecase_version` with the repaired locator; the batch
      continues on the new version.
- [ ] Healing events clearly labelled in the timeline; `llm_tokens` recorded per execution and
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
