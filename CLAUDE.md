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
selectors as fallbacks), not a single selector. A rung also says *where* to look:
`within` scopes it inside another rung, `has_text` filters, `frames` descends
through iframes. `{{input.x}}` / `{{secret.x}}` templating is confined to
value-bearing fields; the validator refuses it in locators, `within` included.

A ladder is editable after recording — `LocatorEditor.tsx` writes it back through
`PUT /usecases/{id}`, and `POST /usecases/{id}/locator-check` opens a page and
reports what each rung matches. That check resolves through `engine.build_locator`,
the executor's own composer; a second implementation would answer a different
question than the run does, on the one screen whose job is to say whether an edit
will work.

HTTP layering: `main.py` wires app state in `lifespan` and does nothing else →
`routers/` (one module per resource) → `services.py` (logic shared by the single-row
and batch paths) → `store.py`. `deps.py` supplies everything a handler needs, so a
test swaps a database, a fake LLM or a fixed principal by overriding one dependency.

Batches run on a durable Postgres queue (`jobs.py`, `SELECT ... FOR UPDATE SKIP LOCKED`,
leases not flags) so a restart does not lose them; `bus.py` fans events out across
processes via `LISTEN/NOTIFY` carrying a `run_id` and a seq *range*, never the payload.
Every event has a sequence number and that number is the client's resume token.

**Writes are batched, because latency is the deployment's and round trips are ours.**
`eventbuffer.py` collects a run's events and writes them in one statement; step rows
go the same way, flushed per row. A ten-step row costs 5 remote round trips instead
of 94. Two rules keep it honest: an event is published to the bus only *after* the
batch is durable (the resume token must never name a seq a catch-up read cannot
return), and a flush happens at every row boundary and in `RunLifecycle._persist`
before the run row is marked finished. The live view never waited on any of this
anyway — the WebSocket serves from an in-memory queue and touches the database only
to catch up after a reconnect.

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
- **Two gates, and which one a defect belongs to is the whole design.**
  `unreplayable_reasons` blocks *publishing* and holds only what can never work on
  any record — today, a step whose every rung is an unnamed structural role
  (`role=generic [24]`; `Snapshot.locate` refuses those by design).
  `unbatchable_reasons` blocks *starting a batch* and holds what breaks only across
  records — a row that signs out while signing in lives only in setup with no
  `session_check`. Putting the second at publish was a mistake that cost a user
  their recording: it ran one record perfectly, so the refusal was wrong, and the
  only way out was to delete it. A gate must refuse where the failure is and always
  name the way through.
- **An unnamed wrapper is recorded by the named control inside it.**
  `agent/marks.describe_element` used to emit `role=generic [24]` for a click on an
  anonymous card; it now borrows the card's own uniquely-named descendant (a radio
  with a person's name on it) and counts `matches` against *that* rung, so the
  warning cannot contradict the ladder. Refused when the descendant's name is not
  unique — swapping one ambiguity for another is not a fix.
- **Keyboard navigation is not workflow.** `drop_focus_keystrokes` removes
  recorded `Tab`/`Shift+Tab` presses, which codegen aims at whatever had focus —
  in one real recording, `Shift+Tab` on a "Forgot password?" link in the middle
  of a login. Safe because `fill` focuses what it fills.
- **A URL must never be made of one sign-in's data.** `usecase.clean_recorded_urls`
  strips the OAuth/OIDC/SAML single-use parameters (`VOLATILE_QUERY_PARAMS`) from
  recorded URLs and drops a navigate step that is only an authorization callback.
  It edits the *raw query text* rather than parsing and rebuilding it — rebuilding
  re-encodes a percent-encoded `redirect_uri` and escapes the braces of a templated
  parameter, turning `{{input.x}}` into a literal that matches nothing.
  `application_origin` picks the app rather than the identity provider for
  `{{env.base_url}}`, reading the authorization request's own `redirect_uri` when
  the recording never leaves the IdP.
- **A locator must never be made of the row's own data.** `usecase.data_derived_clicks`
  catches a suggestion clicked after a per-row search, and any locator whose text
  repeats a declared input's recorded value; `strip_data_locators` removes the name
  and keeps it in `rejected_locators` for the review screen. Both recorders call it
  (`routers/recordings.py`, `agent/distil.py`) right after parameterisation, which
  is the last point at which "this came from the spreadsheet" is known. The name is
  *deleted, not demoted*: a ladder takes the first rung matching one element, so a
  demoted data rung sits unused on every row that works and fires on exactly the
  ambiguous ones — acting only when it is certainly wrong.
- **Templating is allowed in a locator's name, text and has_text, and refused in
  its selector and frames.** The split is "is this field a query language"; see
  `TEMPLATABLE_LOCATOR_FIELDS`. The executor renders a rung before it describes it,
  so a failure names what that row actually looked for.
- **The model can never invent a locator.** Healing and repair present a numbered
  list of controls actually on the page and take back an index. The list is not
  deduplicated — six "Edit" links are six lines, numbered and labelled with the
  row they sit in — and `snapshot.locator_for` turns the chosen node into the
  narrowest locator that finds it alone, returning `None` when nothing can.
  Callers must refuse on `None` rather than approximate: `nth=0` means "no
  position given", so the first of several identical controls with nothing named
  around it has no spelling, and writing one anyway is a repair that looks
  applied and fails identically.
- **A fix is remembered only after the retry says it worked.** `StepHealer.confirm`
  is called by `engine._heal` after the repaired step re-runs. Writing at proposal
  time recorded what the model believed; recall then argued for repeating a
  confident wrong answer every time that site broke again.
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
- **`locator.count()` counts hidden elements, so ambiguity is judged over
  `filter(visible=True)`.** Which rungs this affects is not the obvious answer:
  `get_by_role` reads the accessibility tree and never saw a `display:none`
  duplicate, while `text` and `css` rungs match the DOM and did — and those are
  the *fallback* rungs `codegen._ladder` writes under every role rung. A hidden
  duplicate therefore costs nothing until the day the role rung stops matching.
- **Nothing on the replay path reads back what it just wrote.** The visual diff
  uses the screenshot bytes still in memory, and a step's baseline image is fetched
  once per run rather than once per row — it is the same object on every row, so a
  thousand-row batch was fetching one identical object a thousand times.
- **An action can move the page out from under the next observation.**
  `_after_action` settles, adopts any tab the site opened, then snapshots, in
  that order. The snapshot taken when a step fails is what a repair is proposed
  from and what goes into healing memory, so observing a page mid-navigation
  writes something false into a memory that is then recalled forever.

## Style

Prose comments and module docstrings carry *why*, including the alternative that
was rejected and the failure it would have caused. Match that: modules open with a
docstring stating the decisions the file embodies, and tests are named as sentences
(`test_a_failed_run_can_be_repaired_without_re_recording`). The code favours being
obvious over being clever.
