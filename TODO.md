# TODO — from single-operator tool to multi-user platform

Derived from [`docs/review/architecture-review.md`](docs/review/architecture-review.md)
(24 Aug 2026). That document carries the evidence and the reasoning; this file is
the actionable list.

> **The redesign supersedes parts of this file.**
> [`docs/design/deterministic-automation-platform.md`](docs/design/deterministic-automation-platform.md)
> replaces the LLM-agent recorder with `playwright codegen` and the MCP replay
> with async Playwright. Items below that concern `agent.py`, `mcp_client.py`,
> `chat.py`, `graph.py` or `checkpoints.py` — **B5**, **C3**, **C4** — are
> resolved by deleting that code rather than by doing the work. Its P2 has also
> landed: uploads are a resource (`ingest.py`, `mapping.py`, the `datasets`
> table), and `batch.py` no longer reads files. P3 has landed too: `codegen.py`
> parses what `playwright codegen` writes, `recorder.py` owns the subprocess,
> and a recording costs nothing. The agent recorder still exists and is deleted
> in P4 -- which has now landed: the agent, its MCP client, the distiller and
> the old replay engine are gone, and `engine.py` drives Playwright directly.
> **B5** and **C3** are resolved by that deletion rather than by doing the work.
> **C4** is the one that survived and got sharper: `chat.py` is now the *only*
> reason two message dialects exist, because healing and repair still speak
> the Anthropic-shaped one. Phases 5 to 7 have landed too: the step trail and
> its visual diff, healing memory on pgvector with the human-in-the-loop, and
> the EKS manifests under `deploy/`. **E2** (metrics) is now the largest gap:
> token spend per repair and locator-drift rate are recorded per step and
> nothing aggregates them, and "zero tokens per row" is the product's central
> claim. **C2** (merge
> `healing.py` and `repair.py`) is folded into the redesign's P6, and **E3**
> (one real end-to-end test) becomes a precondition of its P4 rather than a
> nice-to-have. The redesign's phase list is in §10 of that document.

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
- [x] **B1 — Persistent checkpointer** *(done, then simplified — `agent/graph.py:memory_checkpointer`)*
      The Postgres/SQLite dual-backend checkpointer (`checkpoints.py`) that
      this line used to describe is gone: the `create_agent` rewrite moved to
      LangGraph's in-memory saver, one per session. `checkpoints.py` no
      longer exists — do not point new readers at it.
- [x] **B2 + D2 — Postgres via SQLAlchemy 2.0 + Alembic** *(done)*
      Ownership columns were born in the initial migration, as planned.
      Artifacts still go to local disk — S3 remains outstanding.
- [x] **B3 — Cross-process events** *(done — `bus.py`, LISTEN/NOTIFY)*
      Postgres rather than Redis: no new infrastructure. The notification
      carries a pointer, not the payload, because a snapshot exceeds NOTIFY's
      8000-byte cap.
- [x] **B4 — Durable job queue** *(done and wired — `jobs.py`, `worker.py`)*
      `SELECT ... FOR UPDATE SKIP LOCKED` with leases rather than lock flags,
      so a worker that dies releases its work. Per-workspace concurrency
      limits. Batches are enqueued by `ReplayManager.start_batch` and run by
      `run_batch_job`, in this process when `WORKER_ENABLED` (the default) or
      in `make worker` beside it. The `ExecutionBusy` single slot is gone.
      Two things fell out of it: a batch now stores its input rows (resume used
      to replay blanks for rows it never attempted), and a batch takes a
      credential id rather than inline secrets. `tests/test_jobs.py` covers the
      queue, which had no tests at all — a large part of how it stayed unwired.
- [ ] ~~**B5 — Browser session pool**~~ *(superseded — the redesign's P4 removes
      the per-run MCP fork; pooling belongs to `engine.py` if it is still needed
      once one browser serves a whole batch)*
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
- [ ] **C2 — Merge `healing.py` and `repair.py` onto one kernel** *(Medium, M —
      the last piece of the redesign's P6)*
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
- [x] **C3 — Approvals via LangGraph `interrupt()`** *(done —
      `HumanInTheLoopMiddleware` in `agent/graph.py`)*
      An irreversible action suspends the graph via `interrupt()`; `resume()`
      answers with `Command(resume=...)`. The custom future-based rendezvous
      is gone. The one caveat B1 used to promise — surviving a process
      restart — no longer holds now that the checkpointer is in-memory rather
      than Postgres-backed; a pending approval survives within the process,
      not across a restart.
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
- [x] **E3 — One end-to-end test against a real browser** *(done —
      `tests/test_e2e_engine.py`)*
      Eleven tests, a real Chromium, a two-page site served from a temp
      directory, nothing stubbed: record a codegen script, parse it, replay it,
      and check parameterisation per row, the setup/row split, locator drift,
      assertions, extraction, screenshots, traces and the allowlist. Writing it
      found three things the design document had asserted and should not have.
      Run with `make test-e2e`.

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
