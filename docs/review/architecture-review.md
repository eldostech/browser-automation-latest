# Architecture review: from single-operator tool to multi-user platform

**Date:** 2026-08-24 · **Scope:** full codebase (backend 10,258 lines / frontend 3,140 lines) · **Changes made:** none — findings and a prioritized TODO only.

---

## 1. Executive summary — and two premises corrected first

This review was requested on two premises: that the project lacks an orchestrator
pattern, and that removing boilerplate is the main path to enterprise readiness.
Both deserve correction before the findings, because a review that flatters its
premises is worthless.

**An orchestrator is already in use.** As of commit `f0037a0`, the agent loop is a
LangGraph `StateGraph` (`graph.py`), tools are bound with `bind_tools`
(`llm.py:129`), and provider wiring is `langchain-aws`. The measured result of that
adoption was **+338 lines and 23 additional packages**, not a reduction. That
number should calibrate every "adopt a framework to shrink the code" instinct in
this document: frameworks buy *capabilities* (checkpointing, maintained provider
wiring), not smaller codebases.

**The gap to enterprise is not boilerplate — it is identity, tenancy, and shared
infrastructure.** Measured today:

| Fact | Evidence |
|---|---|
| Zero authentication on ~30 HTTP endpoints and the WebSocket | no `Security(...)`/bearer scheme anywhere in `main.py` |
| Zero ownership: no `user_id`/`tenant` column on any table | `grep -cE "user_id\|owner\|tenant" store.py` → 0 |
| All coordination is in-process | `EventBus` (in-memory fan-out, `runner.py`), single execution slot (`ReplayManager._slot`), LangGraph `InMemorySaver` (`graph.py:87`) |
| Storage is single-node | SQLite + artifacts on local disk |
| No CI | no `.github/workflows` |

A second user cannot exist safely in this system today — not because the code is
verbose, but because nothing in it knows who anyone is. That is where the roadmap
below starts.

**What is genuinely worth keeping** (do not refactor these away):

- The **ports-and-adapters seams** already exist informally: `LLMClient`
  (Protocol), `EventSink` (Protocol), `Store` (designed swappable), `ApprovalGate`
  (Protocol). This is why the LangChain swap touched neither distillation nor
  repair. The recommendation is to *formalize* this, not invent it.
- The **zero-token guarantee is structural** — `replay.py` cannot import an LLM,
  and tests assert it. This is the project's most valuable property. Several
  recommendations below explicitly exclude touching it.
- The **event-sourced run timeline** (append-only `events` table, `seq` as resume
  token) is the right foundation for the multi-process world — it just needs a
  distributed transport behind it.
- **Test discipline**: 608 tests, offline by default, with regression tests that
  name the incident they pin. Keep this bar through every change below.

---

## 2. Findings

Each finding: **severity** (Blocker / High / Medium / Low for the enterprise goal),
**effort** (S <1 day, M 1–3 days, L 1–2 weeks), the applicable pattern, and evidence.

### A. Identity & tenancy — the actual blockers

**A1. No authentication or authorization.** *(Blocker, L)*
Every endpoint — including `POST /api/credentials`, `DELETE /api/usecases/{id}?purge=true`,
and the approval endpoint — is open to anyone who can reach the port. The WebSocket
stream is equally open. → OIDC/SSO at the edge (FastAPI `Security` dependencies;
`fastapi-users` or an API gateway), with an RBAC layer distinguishing at minimum
*operator* (run, approve) from *author* (create, publish, delete) from *viewer*.
**Pattern:** policy enforcement point as a router-level dependency, not per-endpoint
`if` checks.

**A2. No resource ownership.** *(Blocker, M — schema; L — enforcement)*
`usecases`, `credentials`, `runs`, `batches` have no owner or tenant column. Credential
*names* are globally `UNIQUE` (`store.py:93`), so two users cannot both have an
"IXL account". → Add `owner_id`/`workspace_id` to every aggregate; scope every
query by it; make uniqueness `(workspace_id, name)`. This is the single largest
schema change on the list and everything multi-user stacks on it — do it before,
not after, the storage migration (A/B below), so Alembic carries it forward.

**A3. Approvals and audit have no identity.** *(High, S once A1 exists)*
`approved_by_human` is a boolean (`events.py`); nothing records *who* approved a
sensitive action, published a use case, or purged one. Enterprise review will ask
for this on day one. → Stamp actor identity into `ApprovalResolved`, publish/purge
events, and a dedicated audit log (append-only, the events table pattern already
in use is fine).

**A4. Secrets management is a single static key.** *(High, M)*
One Fernet key in an env var encrypts every credential, with no rotation story and
no per-tenant separation. Fine for one operator; not for a platform holding many
users' site logins. → Envelope encryption with a KMS-held master key (AWS KMS,
given the Bedrock commitment), per-credential data keys, and a re-encryption job
for rotation. The write-only API surface is already right — keep it.

**A5. `allow_scripts` changes meaning in a multi-user world.** *(High, S to gate, M to sandbox)*
A recorded `script` step is arbitrary JavaScript against a signed-in session.
Today one person approves their own scripts; with tenants, user A's script must
never run under user B's credentials, and script approval becomes a privileged
action. → Make `allow_scripts` a role-gated permission, record who enabled it
(A3), and treat cross-user sharing of script-bearing use cases as a distinct,
reviewable act.

### B. Shared state & scale-out — the second wall

**B1. The resume-after-restart promise is not yet real.** *(High, S)*
The LangGraph checkpointer is `InMemorySaver` (`graph.py:87`) — state dies with the
process, so the checkpointing rationale from commit `f0037a0` is currently only
latent. This is the cheapest high-value fix in the document: swap in
`AsyncSqliteSaver` now (one dependency, same interface), `PostgresSaver` when B2
lands, and wire `reap_orphaned_runs` to *resume* instead of fail-and-discard.

**B2. SQLite + local artifacts → Postgres + object storage.** *(High, L)*
`store.py`'s own docstring names Postgres as the swap-in and the surface was kept
small deliberately — honor that design. Artifacts (screenshots) move to S3-compatible
storage with signed URLs. Do this *with* A2 so ownership columns are born in the
migration, and via SQLAlchemy/Alembic (D2) rather than extending the hand-rolled
`MIGRATIONS` dict — which today is a version stamp with zero actual migrations in it.

**B3. In-memory EventBus → Redis pub/sub, with an outbox.** *(High, M)*
`EventBus` says it plainly: "One process only; move to Redis to scale out." With
two workers, a WebSocket connected to worker 1 misses events from worker 2. The
append-only events table makes this clean: **outbox pattern** — persist first
(already done), publish via Redis, and replay-from-`seq` (already implemented for
reconnects) covers any pub/sub gap.

**B4. The single execution slot → a real job queue.** *(High, L)*
`ReplayManager._slot` is an in-process dict; `RunManager._tasks` is an in-process
map. Neither survives a restart nor spans workers, and "one execution at a time"
was a *product* decision for one user — as a platform it becomes *per-workspace
concurrency limits*. → A queue with persistent jobs and per-key concurrency
(arq or Celery for the modest version; Temporal if long-running workflows with
retries/resume become central — it would also absorb the batch recovery contract
and heartbeat/timeout handling, at the cost of new infrastructure).
**Pattern:** work queue + worker pool; batch recovery is already written as an
explicit contract (`batch.py` rules 1–5), which transplants cleanly.

**B5. Browser capacity is unmanaged.** *(Medium, M)*
Each run spawns a Playwright MCP child process. Multi-user means a browser *pool*
with per-tenant quotas, TTLs, and warm instances — otherwise ten users equal ten
Chromium forks on one box. The `MCPBrowserSession` abstraction is the right seam;
put a pool behind it rather than changing callers.

### C. Orchestration consolidation — real duplication, named

**C1. Three run lifecycles share one unwritten template.** *(High, M — the best
boilerplate win in the codebase)*
`RunManager._execute`, `ReplayManager.execute_once`/`_drive`, and `_run_batch`/`_finalise_batch`
(all in `runner.py`, 983 lines) each hand-roll the same arc: create run → build
sink/redactor → open MCP session → drive → shielded finalise → emit `RunFinished` →
persist terminal status. The shielded-finally subtlety (a cancelled run must still
close its socket) is duplicated three times — and subtle code duplicated is where
the next bug lives. → One `RunLifecycle` async context manager (**template
method**) with three small strategies inside it. Estimated ~200 lines removed and,
more importantly, one place for the subtlety.

**C2. `healing.py` and `repair.py` are the same idea twice.** *(Medium, M)*
Two modules (274 + 450 lines), two prompts, two "model picks an element by index"
tool schemas, two budget mechanisms — one mid-run, one post-mortem. The safety
invariant (model cannot invent a locator) is implemented twice and must be kept
true twice. → One `proposals` module owning the candidate-listing, the
choose-by-index schema, and the budget; healing and repair become two thin entry
points. **Pattern:** strategy over a shared kernel.

**C3. Approvals should become LangGraph `interrupt()` — when B1/B4 land.** *(Medium, M)*
Today two pause mechanisms coexist: the graph, and a custom future-based
rendezvous (`RunApprovalGate`). `interrupt()`/`Command(resume=)` with a persistent
checkpointer would make a pending approval itself survive restart — a genuinely
better property — but it restructures the run-task lifecycle and the approve
endpoint. Sequenced deliberately *after* B1 (persistent checkpointer) makes it
real rather than cosmetic.

**C4. Finish the message migration.** *(Medium, S)*
`chat.py`'s bridge and `agent._history_from` exist only because distillation,
healing, and repair still speak Anthropic-shaped dicts — the `f0037a0` commit
message already flags ~150 removable lines. Migrating those three call sites to
LangChain messages deletes the bridge and one of the two message dialects.

### D. Boilerplate & structure — where the "less code" instinct is right

**D1. `main.py` is a 1,280-line monolith.** *(High, M)*
Routes, validation, orchestration and business logic in one file: 21 hand-written
404 raises, 24 `Depends(get_store)` repetitions, zero `APIRouter`s. →
**Service layer + thin routers**: `routers/{runs,usecases,credentials,batches,health}.py`,
a `UseCaseService`/`RunService` owning the logic now inlined in endpoints, and
shared dependencies like `get_usecase_or_404` replacing the repeated lookup-then-raise
blocks. This is also the precondition for testing business logic without `TestClient`.

**D2. Hand-written SQL + row mapping → SQLAlchemy 2.0 + Alembic.** *(High, L)*
`store.py` is 953 lines, a large fraction of which is `INSERT`/`UPDATE` strings and
`_row_to_*` mapping — exactly the boilerplate an ORM erases — and the migration
scaffold has no real migrations. Alembic gives the schema-change story A2 and B2
require. **Pattern:** repository per aggregate (runs, use cases, credentials,
batches) + unit-of-work, replacing one god-Store. Do it as part of the B2
migration, not as a separate rewrite of the SQLite layer.

**D3. The module-global `settings` singleton keeps causing real incidents.** *(Medium, S)*
`config.settings` is imported at module level everywhere; tests monkeypatch it. A
`.env`-leaks-into-tests bug bit **three separate times this project** (API key,
default model, repair model). → Construct `Settings` once in the lifespan, inject
via `Depends`; no import-time instantiation. Small change, closes a recurring
bug class.

**D4. `events.py` ↔ `events.ts` hand-mirroring → codegen.** *(Medium, S)*
380 TS lines maintained by hand with a test that only checks type-name presence,
not field shapes. → Generate from the pydantic models (`pydantic-to-typescript`,
or an OpenAPI-driven client for the REST surface too). Deletes the mirror *and*
strengthens the guarantee.

**D5. Frontend: 771-line `UseCaseView`, hand-rolled polling.** *(Medium, M)*
One component owns review, publish, rename, credentials, run-one, batch, repair
and delete, with bespoke `setInterval` polling beside an existing WebSocket. →
Split by mode into components + hooks; adopt TanStack Query for
fetch/cache/poll lifecycle (removes most `useState`/`useEffect` plumbing); stream
batch progress over the already-built event channel instead of polling.

**D6. Duplicate request-side logic.** *(Low, S)*
Credential resolution and missing-slot validation are repeated across the execute
and batch endpoints; archive/purge/rename each re-implement the fetch-or-404 dance.
Falls out of D1 mostly for free.

### E. Operability — table stakes it doesn't have yet

**E1. No CI.** *(High, S)* 608 tests and no pipeline running them. A GitHub Actions
workflow — backend pytest, frontend `tsc` + build, secret scan (the manual
`git grep` habit from this project, automated) — is an afternoon.

**E2. No metrics or tracing.** *(Medium, M)* Structured JSON logs exist with
`run_id` binding — good — but there are no counters (runs, failures, token spend
per role, locator-drift rate — the data is already recorded per step), no
latencies, and no trace linking HTTP request → run → LLM call. → OpenTelemetry
(FastAPI + LangChain instrumentors exist), Prometheus counters, correlation IDs
end-to-end. Token-spend metrics matter doubly here because "zero tokens" is the
product's claim — make the dashboard prove it continuously.

**E3. No end-to-end proof against a real browser.** *(Medium, M)* Already
self-flagged in `TODO.md`: every test fakes the MCP session, so the first real
batch is the first real proof. One opt-in E2E (like the existing `RUN_E2E=1`
pattern) recording and replaying against a local static site would close the
biggest confidence gap in the test suite.

**E4. Container/runtime hardening.** *(Low, S)* Non-root images, pinned lockfile
(`make lock` exists — enforce it in CI), health/readiness probes split,
resource limits for browser processes.

---

## 3. What *not* to do

Anti-recommendations are half the value of a review like this.

- **Do not split into microservices.** One modular monolith with the seams above
  scales to many users on a queue + Postgres + Redis long before service
  boundaries pay for themselves.
- **Do not rebuild `replay.py` on the graph or "unify" it with the agent.** Its
  inability to reach a model is the product's core guarantee, held structurally.
- **Do not adopt more framework to shrink code.** Measured on this codebase:
  LangGraph/LangChain adoption was **+338 lines**; framework CSV/file tools were
  evaluated and lost to two lines of stdlib. Frameworks here buy capabilities,
  not brevity — adopt them only when the capability (checkpointing, queue,
  codegen) is the point.
- **Do not add full CQRS/event-sourcing.** The events table is a timeline, not
  the system of record, and that division is serving the project well.

---

## 4. Prioritized TODO

Phases are ordered by dependency, not preference: tenancy columns must precede the
storage migration; the persistent checkpointer must precede interrupt-based
approvals.

### P0 — Make multi-user *possible* (identity, safety, CI)
- [ ] **A1** Authentication (OIDC/SSO) + RBAC dependency on every router; authenticate the WebSocket
- [ ] **A2** `workspace_id`/`owner_id` on usecases, credentials, runs, batches; scope all queries; per-workspace credential-name uniqueness
- [ ] **A3** Actor identity on approvals, publish, purge; append-only audit log
- [ ] **A5** Role-gate `allow_scripts`; record who enabled scripts on a use case
- [ ] **E1** CI: pytest + tsc/build + secret scan + lockfile check
- [ ] **B1** Swap `InMemorySaver` → `AsyncSqliteSaver`; make orphaned-run reaping resume instead of discard

### P1 — Make multi-user *work* (shared infrastructure)
- [ ] **B2 + D2** Postgres via SQLAlchemy 2.0 + Alembic (repositories + unit-of-work, ownership columns from day one); artifacts → object storage with signed URLs
- [ ] **B3** Redis pub/sub behind `EventBus` (outbox: persist → publish → replay-from-seq)
- [ ] **B4** Job queue (arq/Celery; evaluate Temporal if workflow durability becomes central) replacing the in-process slot and task map; per-workspace concurrency limits
- [ ] **B5** Browser session pool with per-tenant quotas behind `MCPBrowserSession`
- [ ] **A4** KMS envelope encryption for credentials; key-rotation job

### P2 — Consolidate the orchestration (the real boilerplate wins)
- [ ] **C1** One `RunLifecycle` template for the three run arcs in `runner.py` (~200 lines, one home for the shielded-finalise subtlety)
- [ ] **D1** Routers + service layer; `get_or_404` dependencies; split `runner.py` (bus / sinks / managers / batch driver)
- [ ] **C2** Merge healing + repair on a shared proposal kernel (one prompt family, one budget, one choose-by-index schema)
- [ ] **C4** Migrate distill/heal/repair to LangChain messages; delete `chat.py`'s bridge and `_history_from` (~150 lines)
- [ ] **C3** Approvals via LangGraph `interrupt()` + `Command(resume=)` (after B1; pending approvals survive restart)
- [ ] **D3** `Settings` via DI, no module-global singleton

### P3 — Polish, prove, observe
- [ ] **D4** Codegen `events.ts` (and API client) from pydantic/OpenAPI; delete the hand mirror
- [ ] **D5** Split `UseCaseView`; TanStack Query; batch progress over the WebSocket instead of polling
- [ ] **E2** OpenTelemetry traces (HTTP → run → LLM), Prometheus metrics incl. token spend per role and locator-drift rate
- [ ] **E3** Opt-in E2E: record → distill → replay against a local static site in CI (nightly)
- [ ] **D6/E4** Endpoint dedup; container hardening; enforce the lockfile

---

*Everything above is findings only — no code was changed. The measured claims
(line counts, grep counts, the +338 framework delta) are reproducible from the
repo at the commit this document lands in.*
