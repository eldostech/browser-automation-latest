# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Record a browser workflow once with `playwright codegen`, map spreadsheet columns
onto it, then replay it over many rows **with no LLM in the loop**. A model is
asked exactly two questions in the whole lifecycle: which column fills which
field (mapping), and where a control moved when a site redesign breaks a step
(healing/repair). `README.md` is unusually complete — read it before making
architectural changes; `docs/design/data-model.md` covers every table.

FastAPI + SQLAlchemy 2.0 async + PostgreSQL/pgvector backend; React 18 + Vite +
TypeScript frontend (no test runner — typecheck and build are the gate).

## Commands

Paths below are Windows (`.venv/Scripts/...`); on macOS/Linux use `.venv/bin/...`.
There is a `Makefile` with the same targets (`make help`), but nothing depends on it.

```bash
# setup
python -m venv .venv && .venv/Scripts/python -m pip install -r backend/requirements.txt
.venv/Scripts/python -m playwright install chromium   # one browser serves record AND replay
cd frontend && npm install

# migrations — always from backend/, that is where alembic.ini lives
cd backend && ../.venv/Scripts/python -m alembic upgrade head
cd backend && ../.venv/Scripts/python -m alembic revision --autogenerate -m "what changed"
cd backend && ../.venv/Scripts/python -m alembic check     # CI fails if models and migrations disagree

# run (two terminals)
cd backend && ../.venv/Scripts/python serve.py     # :8000 — do NOT use `uvicorn --reload` (see below)
cd frontend && npm run dev                         # :5173, proxies /api and /healthz (incl. WS) to :8000

# separate batch worker (set WORKER_ENABLED=false on the API first)
cd backend && ../.venv/Scripts/python -m worker

# tests — need a running PostgreSQL; they use their own schema (browser_test)
cd backend && ../.venv/Scripts/python -m pytest -q
cd backend && ../.venv/Scripts/python -m pytest tests/test_codegen.py -q
cd backend && ../.venv/Scripts/python -m pytest tests/test_batch.py::test_name -q
cd backend && RUN_E2E=1 ../.venv/Scripts/python -m pytest -q -m e2e   # opt-in, launches real Chromium

# lint / typecheck
cd frontend && npm run typecheck && npm run build
.venv/Scripts/python -m compileall -q backend
```

Test database connection comes from `TEST_DB_HOST` / `TEST_DB_PORT` / `TEST_DB_NAME` /
`TEST_DB_USER` / `TEST_DB_PASSWORD` / `TEST_DB_SCHEMA`, each falling back to the
matching `DB_*`. Nothing in the default run touches the network, spawns a browser,
or calls a model.

## Architecture

Three phases, and only one of them can spend tokens:

| Phase | Modules | Model cost |
|---|---|---|
| Record | `recorder.py` (codegen subprocess) → `codegen.py` (`ast` parse) | none |
| Map | `ingest.py` (pandas) → `mapping.py` (heuristics first) | fallback only |
| Replay | `runner.py` → `batch.py` → `engine.py` → `browser.py` | **none, ever** |
| Heal | `healing.py` + `memory.py` (pgvector recall) | one budgeted call, off by default |

`usecase.py` is the contract between phases. A `UseCase` splits steps three ways —
`setup_steps` (once per session), `row_steps` (once per input row), `row_reset`
(between rows) — because a batch shares one browser session and sign-in must not
run per row. Locators are a *ranked ladder* (role + accessible name first, recorded
selectors as fallbacks), not a single selector. `{{input.x}}` / `{{secret.x}}`
templating is confined to value-bearing fields; the validator refuses it in locators.

HTTP layering: `main.py` wires app state in `lifespan` and does nothing else →
`routers/` (one module per resource) → `services.py` (logic shared by the single-row
and batch paths) → `store.py`. `deps.py` supplies everything a handler needs, so a
test swaps a database, a fake LLM or a fixed principal by overriding one dependency.

Batches run on a durable Postgres queue (`jobs.py`, `SELECT ... FOR UPDATE SKIP LOCKED`,
leases not flags) so a restart does not lose them; `bus.py` fans events out across
processes via `LISTEN/NOTIFY` carrying `run_id:seq:origin`, never the payload.
Every event has a sequence number and that number is the client's resume token.

## Invariants — do not break these

- **`engine.py` must never import `llm`.** `UseCaseExecutor` has no parameter that
  could accept a model client; a healer is *injected*. `tests/test_e2e_engine.py`
  and `tests/test_healing.py` assert the absence of `import llm` in the source.
- **Tenancy is enforced by construction.** Scoped operations live on
  `WorkspaceStore` (`store.workspace(id)`), never on `Store`. Adding a scoped query
  to `Store` removes the guarantee that a forgotten filter cannot compile. Only
  genuinely cross-tenant work (startup reaping, health, workspace creation) stays unscoped.
- **Authorization is a dependency, not an `if`.** Declare `require(Permission.X)`
  on the route so the check is visible in the route definition and in the OpenAPI.
- **Codegen output is data, never code.** Parse with `ast`; never `exec`, `eval`
  or import it. Unrecognised lines become `Unsupported` entries shown to the user —
  never a guessed-at step.
- **The model can never invent a locator.** Healing and repair present a numbered
  list of controls actually on the page and take back an index.
- **Secrets are registered with the redactor before any event is emitted**, and
  batches take a stored credential id, never inline values.
- **`.env.example` is checked against the `Settings` model by `tests/test_config.py`.**
  Adding, renaming or removing a setting means editing both.
- **Every prompt lives in `backend/prompts/*.md`**, loaded by `prompt_loader.py`
  with `string.Template` (`$name`, literal `$` written `$$`). No inline prompt strings.

## Gotchas specific to this codebase

- **`DB_SCHEMA` is bound into SQLAlchemy's `MetaData` at import time**, before any
  settings object exists — environment variable first, `.env` second, and the app
  refuses to start if the two disagree. `tests/conftest.py` sets it before the
  `sys.path` insert for the same reason.
- **`serve.py` exists because of two colliding Windows constraints.** Playwright
  needs a `ProactorEventLoop`; `uvicorn --reload` forces a `SelectorEventLoop`, and
  `--loop none` alone makes the reloader hand the child an inherited socket that
  cannot join its IOCP (`[WinError 87]`, *after* startup completes). `serve.py`
  reloads a level up with `watchfiles`. If you must start uvicorn by hand, add `--loop none`.
- **The backend is a flat module tree, not an installed package.** `pyproject.toml`
  sets `pythonpath = ["."]`; run pytest and alembic from `backend/`.
  (That file's `[project]` metadata still describes the retired MCP/agent design.)
- **Migrations are a linear chain in `backend/migrations/versions/`.** A schema
  migrated by another branch produces `Can't locate revision identified by ...` —
  point `DB_SCHEMA` at a fresh schema rather than fighting it.
- **Recording is local-only** (`RECORDER_ENABLED`); a codegen window needs a display,
  so Docker and the EKS manifests in `deploy/` run replay only.

## Style

Prose comments and module docstrings carry *why*, including the alternative that
was rejected and the failure it would have caused. Match that: modules open with a
docstring stating the decisions the file embodies, and tests are named as sentences
(`test_a_failed_run_can_be_repaired_without_re_recording`). The code favours being
obvious over being clever.
