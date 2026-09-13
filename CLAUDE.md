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
| Export | `export.py` (a `UseCase` → a Playwright Python script) | none |
| Map | `ingest.py` (pandas) → `mapping.py` (heuristics first) | fallback only |
| Replay | `runner.py` → `batch.py` → `engine.py` → `browser.py` | **none, ever** |
| Heal | `healing.py` + `memory.py` (pgvector recall) | one budgeted call, off by default |

**Two passes bracket an agent recording** (`agent/brief.py`), each one model
call, both *injected* so a caller that passes no `scribe` makes neither. Before:
the request is restated as a goal, the values expected to vary per row, what
proves a row worked, and what the request did not say -- and **no clicks**,
because a model that has never seen the site inventing an "Advanced search"
link sends the recorder hunting for something that does not exist. After: the
distilled steps are described in plain language onto `UseCase.instructions`,
with a purpose per step onto `Step.intent`. Neither pass can add, remove or
alter a step; both are failure-tolerant, because the recording is the expensive
part and prose is not allowed to cost one.

**OpenRouter can be switched off completely, and that is a kill switch rather
than a preference.** `OPENROUTER_ENABLED=false` means the provider is not
resolvable (`ModelChoice.resolve`), not buildable (`build_llm`, `chat_model`),
and its catalogue is never fetched -- that last one being an HTTP request to
openrouter.ai and the path most easily forgotten, since `prices_for` reads it
from the costing layer. `Settings.openrouter_available` is the single predicate
so "disabled" and "no key" cannot drift apart, `LLM_PROVIDER=openrouter` with
it false is refused at startup, and `tests/test_models.py` asserts every path
refuses. Asked for by an operator taking this into a company: off has to mean
no packet, not "no feature I can think of".

**Bedrock models are discovered, with the configured list kept.** `catalog.py`
reads `ListFoundationModels` *and* `ListInferenceProfiles`, because a model
whose only inference type is `INFERENCE_PROFILE` cannot be invoked by its own
id and the `us.`-prefixed profile is what runs. Discovery is additive and
degrades to `BEDROCK_MODELS` with the missing IAM permission named; the check
button is the only thing that can tell "listed by the region" from "invokable
by this account". `conftest.test_settings` forces `bedrock_discover` off, since
the suite's guarantee is that a default run makes no network call.

**Two model providers, chosen per request.** `llm.ModelChoice` is a
`(provider, model)` pair; `llm.ModelPool` caches a client per choice. Bedrock is
a configured list, OpenRouter is a live catalogue (`catalog.py`, fetched and
cached, prices included). The choice is *not* server state — it lives in the
browser (`frontend/src/lib/model.ts`) and travels on every request that can
spend a token, because comparing two models means running two at once and a
shared setting would serialise exactly that. `GET /api/models` lists what can be
reached; `POST /api/models/check` makes one tiny call to prove it. The Bedrock
prompt-cache flag is sent only to Bedrock — `ChatOpenAI` rejects an unknown
keyword, so sending it everywhere would fail every OpenRouter call.

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
- **A promotion is a document, and the import says what is still missing.**
  `GET /usecases/{id}/export` wraps the definition with where it came from;
  `POST /usecases/import` takes that envelope or a bare definition, forces
  `status=draft` and `allow_scripts=false`, drops `source_run_id`, and keeps
  the id so a revision appends a version rather than duplicating. Nothing
  sensitive travels: secrets are slot *names* and each environment supplies its
  own values. The hazard worth knowing is quieter than a failure -- a
  definition carries `base_url` as the fallback when no target answers, so a
  use case promoted from dev into UAT *runs*, against dev. `_promotion_gaps`
  therefore reports a named target this deployment does not have, a missing
  target where only a recorded address remains, and credential slots nothing
  here can fill. Warnings, not refusals: an import is how a document arrives,
  and refusing it leaves nowhere to fix the gap from.
- **Tenancy is enforced by construction.** Scoped operations live on
  `WorkspaceStore` (`store.workspace(id)`), never on `Store`. Adding a scoped query
  to `Store` removes the guarantee that a forgotten filter cannot compile. Only
  genuinely cross-tenant work (startup reaping, health, workspace creation) stays unscoped.
- **Authorization is a dependency, not an `if`.** Declare `require(Permission.X)`
  on the route so the check is visible in the route definition and in the OpenAPI.
- **The token ceiling counts fresh tokens, not cache reads.** `Spend.tokens`
  stays the true total, and `Spend.fresh_tokens` is what `exceeded()` compares.
  A real session recorded 405,305 tokens, cost 44 cents, and was stopped
  mid-record by a 400,000 ceiling with over half its dollar budget unspent --
  nearly all of it one prompt prefix re-read every turn, which is the thing
  caching exists to make cheap. `usd` is the limit that governs; the token cap
  is there for a genuine runaway, and a recognised prefix is not one.
- **A draft is not replayed when replaying it can say nothing.** `run._why_not_verify`:
  a session that stopped before finishing a record has already reported that,
  and `AGENT_VERIFY_DRAFT=false` turns the pass off for a deployment that would
  rather have the wall clock. The replay itself spends **no tokens** -- it is the
  engine, which has no path to a model -- so what is being saved is a browser
  launch, and what is given up is learning that a draft does not replay before
  somebody publishes it. The reason is always carried on
  `Verification.skipped`: "not verified" with no reason reads as a failure.
- **Verification mends, then proves.** `agent/verify.py` replays a draft twice:
  once with a healer attached, and again with none at all once anything has been
  mended. The second pass is the evidence — the first only proves a *model* can
  get through. What was mended lands in `Verification.repairs`, the proved
  definition in `.patched`, and `run.py` keeps that as the draft. Without a
  healer it is one pass and one verdict, exactly as before.
- **Keep what the MCP server says it ran.** Every acting reply carries the
  Playwright statement the server executed, and `agent/ran.py` parses it into a
  rung that `distil.py` appends behind the semantic ones. It is the only
  evidence of which **DOM element** received the action, and the tree and the
  DOM disagree: a styled radio is a tree node with an accessible name and an
  input nobody can click, and the server clicks its label. A real recording of
  one produced `role=radio name="Nayra Asati"` from the tree while the server
  had run `locator('label').filter({hasText: 'Nayra Asati'})`; replay found the
  radio and spent thirty seconds failing to click it, twice.
- **Consent banners are rejected, and that rule lives in the prompts.**
  `author.md`, `recover.md` and `explore.md` all carry it, and `heal.md` treats
  a banner as the problem rather than a candidate. Deliberately not code: a list
  of button labels in Python would be a judgement about a page made in the place
  with the least context, and `tests/test_prompts.py` asserts no such list
  exists in the engine or the recorder. Two reasons in one rule -- accepting
  consents on behalf of somebody who is not in the conversation and cannot be
  undone from inside a run, and an undismissed banner is an overlay that
  intercepts every click underneath it, which arrives as a timeout on an
  element that was found.
- **Playwright's call log is the only place the cause of a timeout is written
  down, so it is kept.** `engine._reason` used to keep the first line on the
  stated grounds that it "carries the actual cause" -- true of every error
  except the one that matters. `Locator.click: Timeout 30000ms exceeded` names
  a locator and nothing else; the log says the element was found and something
  was covering it, or it would not hold still, or it was disabled.
  `_cause_in` reads that and puts it on the end of the message, naming the
  offending element; `call_log_of` keeps the log on the `step_failed` event, and
  `repair.gather_context` carries it into the repair prompt. Without it a
  repair answers a timeout by re-spelling the locator, which is what happened:
  a covered column header was "fixed" by turning `exact` off, and the next run
  happened to pass, so the diagnosis was never made and the failure came back.
- **A rung that resolves but cannot be acted on falls through to the next one.**
  `engine._do_element_action` walks the ladder over *actions*, not only over
  resolution, giving each attempt a share of the step's timeout. Safe because
  Playwright's actionability timeout means the action never dispatched. A step
  that fails every rung says they were all found, because "no element matched"
  said of three rungs that all matched sends somebody after a problem they do
  not have.
- **The asymmetry to keep in mind whenever touching the recorder.** An agent acts
  on `ref=e12`, an index into a snapshot seconds old that always names exactly
  one element. A replay acts on a *description*. So recording cannot fail the way
  replay fails, and anything that makes a step describable-but-wrong is invisible
  until the first replay. `session._cannot_be_described` refuses such a click
  once, with what to do instead, then allows a repeat — the click is fine, the
  *recording* of it is not, and a hard refusal would block a task the agent can
  do. `session._only_a_position` is the same gesture one rung down: the element
  has a name, several others share it, and the only thing left distinguishing it
  is `nth`. A position is a claim about ordering that the next release can
  falsify, and when it does the step acts on a different record and *succeeds*
  — so the agent is asked, while the page is still on screen, for something
  that says which row it means.
- **`Step.recorded_page` is evidence, never executed.** The page a step was
  recorded against, capped, carried in the definition so a repair in UAT can
  compare then against now rather than guess from now alone. `healing.py` and
  `repair.py` both render it; the listing itself is `snapshot.named_controls`,
  shared because the two copies drifted once already -- the field was wired
  into healing and *reported* as wired into repair when it was not.
- **`Step.intent` and `UseCase.instructions` are evidence too.** A step's
  `description` renders its locator, which says what the step does and nothing
  about why, so healing and repair were being asked "which of these forty
  controls resembles a link named Billing" when the answerable question is
  "which of these opens the customer's billing tab". The sentence already
  existed: the authoring agent must write one before every tool call
  (`session.py`'s `observation`) and `distil.py` dropped it. Both fields narrow
  a candidate list and can never add to it, and both prompts say so.
- **Accuracy is measured, not asserted.** `tests/test_eval_replay.py` (`RUN_EVAL=1`)
  runs every case several times and reports step success, false refusals, wrong
  element, and run-to-run spread. Four numbers because they pull against each
  other: a bolder resolver cuts refusals and raises wrong clicks, and one rate
  hides that entirely.
- **A recorder reports what it cannot represent; it never raises.** The codegen
  path has always turned an unparseable line into an `Unsupported` entry and kept
  the rest. `agent/distil.py` used to let a `ValidationError` out of `_steps`, so
  one call the schema refused destroyed a whole authoring session — and the only
  thing a person could do with it was delete it. Both paths now warn per call and
  carry on. Adding a `Step` field with a required companion (`wait`/`wait_for`)
  means teaching `_step` to build it, or every session using that tool dies.
- **Codegen output is data, never code.** Parse with `ast`; never `exec`, `eval`
  or import it. Unrecognised lines become `Unsupported` entries shown to the user —
  never a guessed-at step. **And in the other direction too:** `export.py` renders a
  use case *back* into a Playwright script, returns a string, and nothing runs it.
  The document stays the source of truth, because a file the platform read back
  would be a second one that drifts. An export carries only the leading rung of
  each ladder, with the rest as comments -- a script that fell through a ladder
  would be the engine, reimplemented in generated code.
- **A rung that says only *where* to look is checked against what the element
  says.** `Step.expect_text` holds the words the recorded element carried, and
  `engine._still_says_what_it_said` compares before acting -- but only when the
  winning rung does not itself match on text (`Locator.matches_on_text`). A
  role-and-name rung has already proved the wording; a CSS path, a bare role or a
  test id has proved only that something sits in that position, and those are the
  rungs that land on a different control and *succeed*. A step that succeeds
  against the wrong control is the worst outcome here: it is recorded as success
  and repeats on every row. Containment either way, not equality -- a wrapper's
  text includes its children's, so equality would refuse correct steps.
- **`Step.when` decides whether a step runs; `optional` decides whether its
  failure matters.** They are different statements and conflating them is why a
  cookie banner that appears on one row in four still cost a timeout and left a
  failure to read. A `when` is an `Assertion` -- the same locally-evaluated check
  as everywhere else, nothing to execute -- and it is evaluated **once**, with no
  retry: a condition asks what is on the page now, and waiting would make "the
  banner is absent" cost the full timeout on every row.
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
- **A credential pasted into a task is caught at the boundary, not redacted
  downstream.** The redactor only knows values it was told about, and a task
  reading "Password : x" is the run row, the `run_started` event, the audit
  detail, the use case's description and the first message to a model -- none of
  which know that string is a password. `redaction.scrub_credentials` recognises
  it by shape (a label, a `:` or `=`, and a value; prose *about* a password has
  no separator and is untouched) and replaces it with `{{secret.slot}}` when a
  bound credential has a slot for it, which is the literal the recorder is
  already told to type. With nothing bound, the session is **refused** rather
  than started: it could not sign in anyway. `scripts/purge_secrets.py` cleans
  rows written before this existed -- it speaks Postgres now, having spoken
  sqlite long after the backend stopped, which is how real credentials survived
  in a live database.
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
