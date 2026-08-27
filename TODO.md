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

- [x] **A1 — Local accounts + RBAC** *(done — `auth/`, `deps.require`)*
      No SSO/OIDC by decision. `auth/rbac.py` is free of HTTP and storage, so
      an OIDC provider replaces how identity is *established* without touching
      what it *permits*. The WebSocket authenticates too.
- [x] **A2 — Resource ownership** *(done — `db/models.py`, `WorkspaceStore`)*
      Scoped operations live on `WorkspaceStore`, not `Store`: forgetting the
      tenant filter is not possible because the scoped object has no method
      that can reach another tenant's row. Credential names are unique per
      workspace now.
- [x] **A3 — Identity on approvals and an audit log** *(done)*
      Approvals, publishes, script grants, credential writes and purges all
      record the actor in an append-only `audit_log`.
- [x] **A5 — Role-gate `allow_scripts`** *(done)*
      `script:enable` is admin-only and sets a flag on the *resource*.
      Execution checks that flag as well as the definition's `allow_scripts`,
      so an author cannot grant themselves code execution by editing JSON.
- [x] **E1 — CI** *(done — `.github/workflows/ci.yml`)*
      Backend on 3.11/3.13 against a real Postgres, `alembic check`, frontend
      type-check and build, and a secret scan. The scan caught a real password
      during this work, which is the argument for automating the habit.
- [x] **B1 — Persistent checkpointer** *(done — `checkpoints.py`)*
      Postgres where psycopg's async mode can run, a SQLite *file* on Windows
      where it cannot (psycopg needs a Selector loop; the browser subprocess
      needs Proactor). Durable on both.
- [x] **B2 + D2 — Postgres via SQLAlchemy 2.0 + Alembic** *(done)*
      Ownership columns were born in the initial migration, as planned.
      Artifacts still go to local disk — S3 remains outstanding.
- [x] **B3 — Cross-process events** *(done — `bus.py`, LISTEN/NOTIFY)*
      Postgres rather than Redis: no new infrastructure. The notification
      carries a pointer, not the payload, because a snapshot exceeds NOTIFY's
      8000-byte cap.
- [x] **B4 — Durable job queue** *(done — `jobs.py`)*
      `SELECT ... FOR UPDATE SKIP LOCKED` with leases rather than lock flags,
      so a worker that dies releases its work. Per-workspace concurrency
      limits. **Not yet wired**: `ReplayManager` still runs batches in-process;
      moving them onto the queue is the remaining step.
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

- [x] **C1 — One `RunLifecycle`** *(done — `lifecycle.py`)*
      The shielded finaliser exists once instead of three times. Note the line
      count went **up** by 133, not down by 200: extract for single-source-of-
      truth, not for brevity.
- [x] **D1 — Routers and a service layer** *(done)*
      `main.py` 1,280 → 197 lines. Seven routers, `services.py`, and shared
      404 lookups in `deps.py` replacing 21 hand-written raises.
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
- [x] **D3 — `Settings` by injection** *(done)*
      The module-level `settings` object is gone; `create_app(settings)` is a
      factory. This is what closed the "`.env` leaks into tests" class.
- [x] **D6 — Deduplicate request-side logic** *(done — `services.py`)*
      Credential resolution, use-case loading and missing-slot validation are
      written once and shared by the execute and batch paths.
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
- [x] **E4 — Container hardening** *(done)*
      Non-root (`pwuser`), an explicit liveness `HEALTHCHECK` kept separate
      from readiness, and compose runs migrations before the server starts.
- [x] **Delete `ANTHROPIC_API_KEY` from `.env`.** Already gone — the key is no
      longer present in the file. Rotating it at the provider remains worth
      doing if it was ever real, since it was exposed earlier in this
      project's history.
- [ ] **Delete `data/runs.db.*.bak`** once you are satisfied with the secret purge
      — the backup still contains the plaintext credential.

*The previous `TODO.md` tracked the record-and-replay feature through phases 0–5,
all shipped. Its content is in git history and the rationale lives in
[`docs/design/repeatable-usecases.md`](docs/design/repeatable-usecases.md).*
