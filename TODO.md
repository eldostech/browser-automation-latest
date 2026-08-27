# TODO — from single-operator tool to multi-user platform

Derived from [`docs/review/architecture-review.md`](docs/review/architecture-review.md)
(24 Aug 2026). That document carries the evidence and the reasoning; this file is
the actionable list.

Phases are ordered by **dependency, not preference**. Two orderings matter and are
easy to get wrong:

- ownership columns (**A2**) must land *before* the Postgres migration (**B2**), so
  Alembic carries them forward rather than bolting them on afterwards;
- the persistent checkpointer (**B1**) must land *before* interrupt-based approvals
  (**C3**), or the rewrite is cosmetic rather than durable.

---

## Two things to hold on to

**Frameworks here buy capabilities, not brevity.** Adopting LangGraph + LangChain
measured at **+338 lines and 23 packages** (commit `f0037a0`). Evaluate every
"adopt X to reduce code" item against that number.

**The zero-token guarantee is structural.** `replay.py` cannot import an LLM and
tests assert it. Nothing below may weaken that — it is the product's core claim.

---

## P0 · Make multi-user *possible* — identity, safety, CI

Nothing else on this list is safe to ship until these land. Today ~30 endpoints and
the WebSocket are unauthenticated, and no table has an owner column.

- [ ] **A1 — Authentication and authorization** *(Blocker, L)*
      OIDC/SSO at the edge; RBAC as a router-level dependency (policy enforcement
      point), not per-endpoint `if` checks. Roles: viewer / operator (run, approve)
      / author (create, publish, delete). **Authenticate the WebSocket too** — it is
      currently as open as the REST surface.
- [ ] **A2 — Resource ownership** *(Blocker, M schema / L enforcement)*
      Add `workspace_id`/`owner_id` to `usecases`, `credentials`, `runs`, `batches`,
      `executions`. Scope every query by it. Change credential uniqueness from
      global `name` (`store.py:93`) to `(workspace_id, name)` — today two users
      cannot both have an "IXL account".
- [ ] **A3 — Identity on approvals and an audit log** *(High, S after A1)*
      `approved_by_human` is a boolean; nothing records *who* approved, published or
      purged. Stamp actor identity into those events and add an append-only audit
      log — the events-table pattern already in use is the right shape.
- [ ] **A5 — Role-gate `allow_scripts`** *(High, S to gate / M to sandbox)*
      A `script` step is arbitrary JavaScript against a signed-in session. With
      tenants, user A's script must never run under user B's credentials. Make
      enabling scripts a privileged action, record who did it, and treat sharing a
      script-bearing use case as a distinct reviewable act.
- [ ] **E1 — CI** *(High, S)*
      608 tests and nothing runs them. GitHub Actions: backend pytest, frontend
      `tsc` + build, secret scan (automate the manual `git grep` habit), and enforce
      the lockfile (`make lock` already exists).
- [ ] **B1 — Make resume-after-restart real** *(High, S)*
      The checkpointer is `InMemorySaver` (`graph.py:87`), so state dies with the
      process and the LangGraph rationale is currently latent. Swap to
      `AsyncSqliteSaver` now, `PostgresSaver` with B2, and change
      `reap_orphaned_runs` to **resume** rather than fail-and-discard.
      *Cheapest high-value item on this list.*

## P1 · Make multi-user *work* — shared infrastructure

Everything here exists because coordination is currently in-process: a second
worker would not see the first one's events, jobs, or browser sessions.

- [ ] **B2 + D2 — Postgres via SQLAlchemy 2.0 + Alembic** *(High, L)*
      `store.py` (953 lines) is largely INSERT/UPDATE strings and `_row_to_*`
      mapping, and `MIGRATIONS` is a version stamp containing **no actual
      migrations**. Move to repository-per-aggregate + unit-of-work, with A2's
      ownership columns born in the migration. Artifacts (screenshots) to
      S3-compatible storage with signed URLs. Do these together, not as two
      rewrites of the same layer.
- [ ] **B3 — Redis pub/sub behind `EventBus`, via the outbox pattern** *(High, M)*
      The bus says it itself: "one process only". Persist first (already done),
      publish via Redis, and rely on replay-from-`seq` (already implemented for
      reconnects) to cover any gap.
- [ ] **B4 — A real job queue** *(High, L)*
      `ReplayManager._slot` and `RunManager._tasks` are in-process dicts: they
      survive neither a restart nor a second worker. "One execution at a time" was a
      product decision for one user; as a platform it becomes **per-workspace
      concurrency limits**. arq or Celery for the modest version; evaluate Temporal
      if durable long-running workflows become central — it would absorb the batch
      recovery contract, at the cost of new infrastructure. The contract in
      `batch.py` (rules 1–5) transplants cleanly either way.
- [ ] **B5 — Browser session pool** *(Medium, M)*
      Every run forks a Chromium via Playwright MCP; ten users means ten forks on
      one box. Put a pool with per-tenant quotas and TTLs behind the existing
      `MCPBrowserSession` seam rather than changing callers.
- [ ] **A4 — KMS envelope encryption for credentials** *(High, M)*
      One static Fernet key encrypts every user's site logins, with no rotation
      story. Move to a KMS-held master key (AWS KMS, given the Bedrock commitment)
      with per-credential data keys and a re-encryption job. Keep the write-only API
      surface — that part is already right.

## P2 · Consolidate the orchestration — the real boilerplate wins

This is where "remove boilerplate" is genuinely correct, and it is worth roughly
350 lines plus a reduction in duplicated subtlety.

- [ ] **C1 — One `RunLifecycle` for the three run arcs** *(High, M)*
      `RunManager._execute`, `ReplayManager._drive` and `_run_batch`/`_finalise_batch`
      each hand-roll: create run → build sink/redactor → open MCP session → drive →
      **shielded finalise** → emit `RunFinished` → persist terminal status. That
      shielded-finally subtlety is duplicated three times, and duplicated subtlety
      is where the next bug lives. One async context manager (template method) with
      three small strategies. **≈200 lines, best single win.**
- [ ] **D1 — Routers and a service layer** *(High, M)*
      `main.py` is 1,280 lines with 21 hand-written 404 raises, 24 repetitions of
      `Depends(get_store)`, and zero `APIRouter`s. Split into
      `routers/{runs,usecases,credentials,batches,health}.py` with
      `UseCaseService`/`RunService` owning logic now inlined in endpoints, and a
      shared `get_usecase_or_404` dependency. Also the precondition for testing
      business logic without `TestClient`. Split `runner.py` (983 lines) the same
      way: bus / sinks / managers / batch driver.
- [ ] **C2 — Merge `healing.py` and `repair.py` onto one kernel** *(Medium, M)*
      274 + 450 lines implementing the same idea at two moments (mid-run vs
      post-mortem): two prompts, two choose-element-by-index schemas, two budgets.
      The safety invariant — *the model cannot invent a locator* — is implemented
      twice and must be kept true twice. One `proposals` module, two thin entry
      points (strategy over a shared kernel).
- [ ] **C4 — Finish the message migration** *(Medium, S)*
      `chat.py`'s bridge and `agent._history_from` exist only because distillation,
      healing and repair still speak Anthropic-shaped dicts. Migrating those three
      call sites to LangChain messages deletes **≈150 lines** and one of the two
      message dialects. Already flagged in the `f0037a0` commit message.
- [ ] **C3 — Approvals via LangGraph `interrupt()`** *(Medium, M — after B1)*
      Two pause mechanisms coexist today: the graph, and a custom future-based
      rendezvous (`RunApprovalGate`). With a persistent checkpointer,
      `interrupt()` + `Command(resume=)` makes a **pending approval survive a
      restart** — a genuinely new property. Restructures the run-task lifecycle and
      the approve endpoint, so it needs its own change.
- [ ] **D3 — `Settings` by injection, not a module global** *(Medium, S)*
      `config.settings` is instantiated at import and monkeypatched in tests. The
      ".env leaks into tests" bug has bitten **three separate times** in this
      project (API key, default model, repair model). Construct it once in the
      lifespan and inject via `Depends`. Small change; closes a recurring class.
- [ ] **D6 — Deduplicate request-side logic** *(Low, S)*
      Credential resolution and missing-slot validation repeat across the execute
      and batch endpoints; archive/purge/rename each re-implement fetch-or-404.
      Mostly falls out of D1 for free.

## P3 · Polish, prove, observe

- [ ] **D4 — Generate `events.ts` instead of hand-mirroring it** *(Medium, S)*
      380 TypeScript lines maintained by hand, with a sync test that checks type
      *names* but not field shapes. Generate from the pydantic models
      (`pydantic-to-typescript`), or drive the whole client from OpenAPI. Deletes
      the mirror **and** strengthens the guarantee.
- [ ] **D5 — Break up `UseCaseView` (771 lines)** *(Medium, M)*
      One component owns review, publish, rename, credentials, run-one, batch,
      repair and delete — with bespoke `setInterval` polling sitting beside an
      already-built WebSocket. Split by mode into components + hooks, adopt
      TanStack Query for the fetch/cache/poll lifecycle, and stream batch progress
      over the event channel instead of polling.
- [ ] **E2 — OpenTelemetry and metrics** *(Medium, M)*
      Structured logs with `run_id` binding exist and are good, but there are no
      counters, no latencies, and no trace linking HTTP request → run → LLM call.
      **Token spend per role and locator-drift rate are already recorded per step
      and nothing aggregates them** — and "zero tokens" is the product's central
      claim, so the dashboard should prove it continuously.
- [ ] **E3 — One end-to-end test against a real browser** *(Medium, M)*
      Every test fakes the MCP session, so the first real batch is the first real
      proof. Add an opt-in nightly run (the existing `RUN_E2E=1` pattern):
      record → distil → replay against a local static site.
- [ ] **E4 — Container and runtime hardening** *(Low, S)*
      Non-root images, split health/readiness probes, resource limits on browser
      processes, and CI enforcement of the lockfile.

---

## Do NOT do these

Anti-recommendations, kept in the list because they are the tempting wrong turns.

- **No microservices.** A modular monolith on queue + Postgres + Redis scales to
  many users long before service boundaries pay for themselves.
- **Do not rebuild `replay.py` on the graph, or "unify" it with the agent.** Its
  inability to reach a model is the product's core guarantee, held structurally.
- **Do not adopt more framework to shrink code.** Measured here: the framework move
  was +338 lines, and framework file-reading tools lost to two lines of stdlib.
  Adopt for capability, never for brevity.
- **No CQRS or full event-sourcing.** The events table is a timeline, not the
  system of record, and that division is serving the project well.

---

## Housekeeping

- [ ] **Delete `ANTHROPIC_API_KEY` from `.env`.** Nothing reads it since
      `29ded6e` made the project Bedrock-only. It was exposed earlier in this
      project's history, so treat it as compromised rather than merely unused.
- [ ] **Delete `data/runs.db.*.bak`** once you are satisfied with the secret purge
      — the backup still contains the plaintext credential.

*The previous `TODO.md` tracked the record-and-replay feature through phases 0–5,
all shipped. Its content is in git history and the rationale lives in
[`docs/design/repeatable-usecases.md`](docs/design/repeatable-usecases.md).*
