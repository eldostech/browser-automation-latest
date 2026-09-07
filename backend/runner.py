"""Run orchestration: owns the lifecycle of every execution and the fan-out of
its events to WebSocket subscribers.

What is left here after the agent went:

* sequence-number allocation and persistence of every event
* one browser session per execution, torn down in a ``finally`` that also runs
  on cancellation
* the single-row path and the batch path, sharing one executor so they cannot
  drift apart
* an in-memory pub/sub so N dashboard tabs can watch one run

The approval rendezvous went with the agent, and so did ``RunManager``. Both
existed because a model was deciding what to do next and sometimes had to be
stopped before it did it. A replay performs steps a person recorded and
reviewed, so there is nothing to approve mid-run -- the review happened before
it was published.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Literal

from batch import BatchProgress, BatchRunner, new_batch_id
from batch import summarise as batch_summarise
from bus import EventBus
from config import Settings
from events import (
    AgentEvent,
    ErrorEvent,
    RunFinished,
    RunStarted,
    RunStatus,
    dump_event,
)
from lifecycle import RunLifecycle, Terminal
from llm import LLMClient
from credentials import Vault, VaultError
from jobs import ClaimedJob, JobQueue
from logging_setup import bind_run_id
from redaction import NULL_REDACTOR, Redactor
from browser import BrowserConfig, BrowserError, PlaywrightSession
from engine import RowResult, UseCaseExecutor, emit_replay_error
from store import Store, WorkspaceStore
from usecase import effective_mode, resolve_base_url, Mode, UseCase

log = logging.getLogger(__name__)

ApprovalDecision = Literal["approved", "rejected", "timeout"]


class ModeUnavailable(RuntimeError):
    """A use case asks for a mode this deployment cannot provide.

    Raised rather than degraded. Every other mode difference here is a matter
    of how much a run may spend; this one is whether the run does the work at
    all, and quietly doing nothing is not a cheaper version of doing it.
    """


# ---------------------------------------------------------------------------
# Pub/sub
# ---------------------------------------------------------------------------


# EventBus lives in bus.py so that the in-memory and cross-process versions
# are interchangeable; it is re-exported here because this module has always
# been where callers import it from.


# ---------------------------------------------------------------------------
# Event sink
# ---------------------------------------------------------------------------


class RunEventSink:
    """Implements ``agent.EventSink``: allocate seq, redact, persist, broadcast.

    Redaction happens here rather than in ``Store`` because this is the single
    point every event passes through on its way to *both* destinations. Doing
    it in the store would leave the WebSocket broadcasting the unredacted copy.
    """

    def __init__(
        self,
        run_id: str,
        store: "WorkspaceStore",
        bus: EventBus,
        api_base: str = "",
        redactor: Redactor | None = None,
    ) -> None:
        self.run_id = run_id
        self.store = store
        self.bus = bus
        self.api_base = api_base.rstrip("/")
        self.redactor = redactor or NULL_REDACTOR
        self._seq = 0

    def reserve_seq(self) -> int:
        self._seq += 1
        return self._seq

    @property
    def last_seq(self) -> int:
        return self._seq

    async def emit(self, event: AgentEvent) -> None:
        event = self.redactor.event(event)
        try:
            await self.store.append_event(event)
        except Exception as exc:  # noqa: BLE001 - never let logging kill a run
            log.error("failed to persist event", extra={"run_id": self.run_id, "error": str(exc)})
        self.bus.publish(self.run_id, event)

    async def save_screenshot(
        self, data: bytes, *, seq: int, mime: str = "image/png"
    ) -> tuple[str, str] | None:
        try:
            record = await self.store.save_artifact(
                self.run_id, data, kind="screenshot", mime=mime, seq=seq
            )
        except Exception as exc:  # noqa: BLE001
            log.error("failed to save screenshot", extra={"run_id": self.run_id, "error": str(exc)})
            return None
        return record.id, f"{self.api_base}/api/artifacts/{record.id}"

    async def save_download(
        self, data: bytes, *, seq: int, filename: str, mime: str
    ) -> tuple[str, str] | None:
        """Keep a downloaded document beside the screenshots and traces.

        Same storage as everything else -- a directory locally, S3 in a
        cluster -- so a migration's documents land wherever the deployment
        already puts its artifacts, and are fetched back by the same endpoint.
        """
        suffix = Path(filename).suffix or ".bin"
        try:
            record = await self.store.save_artifact(
                self.run_id,
                data,
                kind="download",
                mime=mime,
                seq=seq,
                suffix=suffix,
                filename=filename,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "failed to save a download",
                extra={"run_id": self.run_id, "filename": filename, "error": str(exc)},
            )
            return None
        return record.id, f"{self.api_base}/api/artifacts/{record.id}"

    async def record_step(self, **fields) -> None:
        """One row per step, for the finished timeline and the visual diff."""
        await self.store.record_step(**fields)

    async def read_artifact(self, artifact_id: str) -> bytes | None:
        """An artifact's bytes, for comparing this run against the baseline."""
        record = await self.store.get_artifact(artifact_id)
        if record is None:
            return None
        return await self.store.read_artifact(record)


# ---------------------------------------------------------------------------
# Use-case execution (no LLM)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExecutionRequest:
    usecase: UseCase
    version: int
    inputs: dict[str, Any] = field(default_factory=dict)
    secrets: dict[str, str] = field(default_factory=dict)
    #: A base URL given when this run was started, winning over the use case's
    #: target. For a one-off against a branch deployment or a single customer's
    #: tenant, where a standing target would be ceremony for one run.
    base_url: str = ""

    headless: bool | None = None
    browser: str | None = None
    #: The tenant this execution belongs to, and who asked for it.
    workspace_id: str = ""
    owner_id: str | None = None
    #: Denormalized beside owner_id so the record of who ran this survives the
    #: account being deleted. See db/models.py.
    owner_email: str = ""


class ReplayManager:
    """Runs stored use cases. Deliberately has no LLM client of any kind.

    Batches go through the job queue rather than an asyncio task owned by
    whichever process accepted the upload. That is what makes them survive a
    restart, and it is what lets a second worker exist at all.

    A single in-process slot used to guard execution, and refusing the second
    caller with a 409 was the whole concurrency story. It is gone: the queue
    enforces one batch per workspace at a time, which is the same guarantee for
    one tenant and a better one for several. What remains is ``active``, which
    reports what *this* process is running -- an observation, not a lock.
    """

    def __init__(
        self,
        store: Store,
        settings: Settings,
        bus: EventBus,
        llm_factory: Any = None,
        queue: JobQueue | None = None,
        vault: Vault | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.bus = bus
        #: Supplies a model *only* for healing, and only when healing is
        #: enabled. Left None, there is no route from a replay to an LLM.
        self.llm_factory = llm_factory
        #: Where batches are queued. Left None, ``start_batch`` refuses rather
        #: than silently falling back to an in-process task that a restart
        #: would lose.
        self.queue = queue
        #: Opens the credential a queued batch names. The worker resolves
        #: secrets itself, because they are deliberately not in the payload.
        self.vault = vault
        self._slot: dict[str, Any] | None = None
        self._task: asyncio.Task | None = None

    def data(self, workspace_id: str) -> WorkspaceStore:
        """The store, confined to one workspace.

        One manager serves every tenant, so the scope comes from the request
        rather than from the manager.
        """
        return self.store.workspace(workspace_id)

    def make_healer(
        self,
        workspace_id: str = "",
        usecase_id: str | None = None,
        *,
        mode: "Mode | None" = None,
    ) -> Any:
        """A healer, or None when this use case runs strictly.

        Returning None is the common case and is what keeps the executor's
        zero-token guarantee true by construction.

        Whether a run may spend a token is a property of the *use case*, not of
        the installation: one workflow runs against a site that is rebuilt every
        sprint and another against a form that has not changed in four years,
        and a single environment variable cannot be right for both. The setting
        remains as a ceiling -- see :func:`effective_mode`.

        The memory is attached here rather than inside the healer so that the
        one object which can reach a model is also the one that decides whether
        it may read what other runs learned -- and so that a workspace's fixes
        are looked up through its own scoped store.
        """
        if self.llm_factory is None:
            return None
        if effective_mode(
            mode, healing_enabled=self.settings.replay_healing_enabled
        ) != "guided":
            return None
        from healing import HealingBudget, StepHealer

        return StepHealer(
            self.llm_factory(),
            HealingBudget(
                max_attempts=self.settings.replay_heal_max_attempts,
                max_tokens=self.settings.replay_heal_max_tokens,
            ),
            memory=self.make_memory(workspace_id),
            usecase_id=usecase_id,
        )


    async def resolve_env(
        self, usecase: UseCase, workspace_id: str, override: str = ""
    ) -> dict[str, str]:
        """What ``{{env.*}}`` answers with for this run.

        Targets are read per run rather than cached: an operator who has just
        corrected a target's address expects the next run to use it, and the
        read is one indexed row per workspace.

        Raises ``TargetMissing`` when the use case names a target this
        deployment has no address for. That propagates to the caller as a
        refusal, which is the point -- falling back to the recorded URL would
        send a use case promoted to production at whatever host it was recorded
        against, silently.

        Skips the read entirely when ``override`` is already given:
        ``resolve_base_url`` returns it outright without ever looking at
        ``targets`` in that case, so querying them first was a round trip
        spent on an answer that was always going to be thrown away -- a batch
        that already pinned its base URL in ``start_batch`` pays for this
        query a second time, for nothing, on every row-group it drives.
        """
        targets = (
            await self.data(workspace_id).target_urls()
            if workspace_id and not override
            else {}
        )
        base = resolve_base_url(
            target=usecase.target,
            targets=targets,
            recorded=usecase.base_url,
            override=override,
        )
        # Whatever else the deployment answers still applies; the base URL is
        # simply the one every recording needs.
        return {**self.settings.usecase_env, **({"base_url": base} if base else {})}

    def make_row_runner(
        self, executor: Any, usecase: UseCase, run_id: str, sink: Any, redactor: Any
    ) -> Any:
        """How one row runs: the engine alone, or the engine with an agent on call.

        None means "the executor's own method", which is every Strict row and
        every Guided row in a deployment without the agent installed. That is
        the default and it stays the default: this returns something only when
        a use case has asked for Guided *and* this deployment can actually
        provide a recovery, so nothing changes underneath an installation that
        did not opt in.

        The agent here is a second line, not a replacement. The healer inside
        the engine still gets first refusal on a step whose locator moved,
        because it is cheaper and it is right far more often.
        """
        if self.llm_factory is None:
            return None
        mode = effective_mode(
            usecase.mode, healing_enabled=self.settings.replay_healing_enabled
        )
        if mode not in {"guided", "explore"}:
            return None
        reachable = getattr(self.settings, "agent_enabled", False)
        if reachable:
            try:
                from agent.graph import available
                from agent.operate import run_row_with_agent
            except ImportError:
                reachable = False
            else:
                reachable = available()

        if not reachable:
            if mode == "explore":
                # Refused, not downgraded. A use case in Explore has no plan to
                # follow, so falling back to the engine would replay an empty
                # step list and report every row as a success -- the worst
                # possible outcome, because nobody goes looking for it.
                raise ModeUnavailable(
                    f"{usecase.name!r} runs in Explore mode, which works each row "
                    "out with a model, and the agent is not available in this "
                    "deployment. Enable it (AGENT_ENABLED, plus "
                    "requirements-agent.txt and Node), or record the workflow and "
                    "run it in Strict or Guided."
                )
            return None

        llm = self.llm_factory()
        allowed = tuple(usecase.allowed_domains)

        async def run(row: dict[str, Any]):
            return await run_row_with_agent(
                executor,
                row,
                llm=llm,
                emit=_stamp(sink),
                run_id=run_id,
                allowed_domains=allowed,
                redactor=redactor,
                explore=mode == "explore",
                usecase=usecase,
            )

        return run

    def make_memory(self, workspace_id: str) -> Any:
        """What this workspace has learned about broken locators, or None."""
        if not workspace_id or not getattr(
            self.settings, "healing_memory_enabled", False
        ):
            return None
        from embeddings import build_embedder
        from memory import HealingMemory

        return HealingMemory(self.data(workspace_id), build_embedder(self.settings))

    # -- the single slot ----------------------------------------------------
    @property
    def active(self) -> dict[str, Any] | None:
        """What holds the execution slot, if anything."""
        if self._task is not None and self._task.done():
            self._slot = None
            self._task = None
        return self._slot

    def _claim(self, label: str, **fields: Any) -> None:
        """Record what this process is running. Does not refuse a second caller.

        Concurrency belongs to the queue now. This is what ``GET
        /api/executions/active`` reports and what cancellation matches against.
        """
        self._slot = {"label": label, "started_at": _iso_now(), **fields}

    def _release(self) -> None:
        self._slot = None
        self._task = None

    def release_slot(self) -> None:
        """Free the execution slot. Idempotent, and safe to call twice.

        Public because the batch driver releases it at a specific point -- once
        the browser session is closed but before the batch row is marked
        finished -- rather than leaving it to the task wrapper, which runs
        later.
        """
        self._release()

    def is_active(self, run_id: str) -> bool:
        """Is this process running that run right now?

        A run is an execution or a batch now -- there is no agent -- so the
        answer comes from the slot rather than from a table of tasks.
        """
        active = self.active
        return bool(active and active.get("run_id") == run_id)

    async def cancel_run(self, run_id: str) -> bool:
        """Stop the run if this worker is the one driving it."""
        if not self.is_active(run_id):
            return False
        return await self.cancel_active()

    async def cancel_active(self) -> bool:
        if self._task is None or self._task.done():
            return False
        self._task.cancel()
        return True

    # -- batches ------------------------------------------------------------
    async def start_batch(self, request: BatchRequest) -> str:
        """Queue a batch and return its id immediately.

        A thousand rows is not an HTTP request, so progress is polled from
        ``GET /api/batches/{id}``.

        The batch row, its rows and its job are written in **one transaction**.
        That atomicity is the reason this queue is a table rather than a
        broker: there is no window in which the UI shows a queued batch that
        nothing will ever claim, and none in which a job points at a batch that
        does not exist.
        """
        if self.queue is None:
            raise RuntimeError("no job queue is configured, so batches cannot be queued")

        usecase = request.usecase
        indices = (
            list(request.only_rows)
            if request.only_rows is not None
            else list(range(len(request.rows)))
        )
        batch_id = new_batch_id()
        # Resolved here rather than in the worker, for two reasons: a missing
        # target is reported to whoever pressed the button instead of failing
        # minutes later in a queue, and the address is pinned for the whole
        # batch so a target edited halfway through cannot move the remaining
        # rows to a different deployment.
        base_url = (
            await self.resolve_env(usecase, request.workspace_id, request.base_url)
        ).get("base_url", "")

        async with self.store.sessions() as session:
            job_id = await self.queue.enqueue(
                "batch",
                {
                    "batch_id": batch_id,
                    "usecase_id": usecase.id,
                    "version": request.version,
                    "only_rows": indices,
                    "credential_id": request.credential_id,
                    "base_url": base_url,
                    "headless": request.headless,
                    "browser": request.browser,
                    "owner_email": request.owner_email,
                },
                workspace_id=request.workspace_id,
                owner_id=request.owner_id,
                # One job per batch, so a retried POST cannot queue the same
                # work twice.
                dedupe_key=f"batch:{batch_id}",
                session=session,
            )
            await self.data(request.workspace_id).create_batch(
                batch_id,
                usecase.id,
                request.version,
                total=len(indices),
                rows=request.rows,
                job_id=job_id,
                dataset_id=request.dataset_id,
                credential_id=request.credential_id,
                base_url=base_url,
                owner_id=request.owner_id,
                owner_email=request.owner_email,
                session=session,
            )
            await session.commit()

        log.info(
            "queued batch",
            extra={"batch_id": batch_id, "job_id": job_id, "rows": len(indices)},
        )
        return batch_id

    async def run_batch_job(self, job: ClaimedJob) -> None:
        """Worker entry point for a queued batch.

        Everything the batch needs is re-read here rather than carried in the
        payload, because the worker may not be the process that accepted the
        upload. Secrets in particular are resolved from the vault, so the queue
        never holds a password.

        Raising is how a job is marked failed; the batch row is closed by
        ``_run_batch``'s own finaliser either way.
        """
        payload = job.payload
        batch_id = str(payload["batch_id"])
        data = self.data(job.workspace_id)

        rows = await data.get_batch_rows(batch_id)
        if rows is None:
            raise RuntimeError(f"batch {batch_id} no longer exists")

        usecase_id = str(payload["usecase_id"])
        version = int(payload["version"])
        definition = await data.get_usecase(usecase_id, version)
        if definition is None:
            raise RuntimeError(f"use case {usecase_id} v{version} no longer exists")
        usecase = UseCase.model_validate(definition)

        only_rows = payload.get("only_rows")
        request = BatchRequest(
            usecase=usecase,
            version=version,
            rows=rows,
            secrets=await self._secrets_for(data, payload.get("credential_id")),
            credential_id=payload.get("credential_id"),
            # Pinned when the batch was queued; see start_batch.
            base_url=str(payload.get("base_url") or ""),
            only_rows=only_rows,
            headless=payload.get("headless"),
            browser=payload.get("browser"),
            workspace_id=job.workspace_id,
            owner_id=job.owner_id,
            owner_email=str(payload.get("owner_email") or ""),
        )

        count = len(only_rows) if only_rows is not None else len(rows)
        self._claim(
            f"{usecase.name} ({count} rows)",
            batch_id=batch_id,
            usecase_id=usecase.id,
            rows=count,
        )
        # The worker owns the task; recording it here is what lets a cancel
        # request reach the batch actually in flight.
        self._task = asyncio.current_task()
        try:
            await _run_batch(self, batch_id, request)
        finally:
            self._release()

    async def _secrets_for(
        self, data: WorkspaceStore, credential_id: str | None
    ) -> dict[str, str]:
        """Open the credential a queued batch named, or return nothing.

        A use case that needs a login and has no stored credential was already
        refused at the API boundary, so arriving here without one means the use
        case does not sign in.
        """
        if not credential_id:
            return {}
        if self.vault is None:
            raise RuntimeError("credential storage is not configured on this worker")
        ciphertext = await data.get_credential_ciphertext(credential_id)
        if ciphertext is None:
            raise RuntimeError("the credential this batch was queued with no longer exists")
        try:
            values = self.vault.open(ciphertext)
        except VaultError as exc:
            raise RuntimeError(f"the credential could not be opened: {exc}") from exc
        await data.touch_credential(credential_id)
        return values

    async def cancel_batch(self, batch_id: str, workspace_id: str) -> bool:
        """Stop a batch, whether it is queued elsewhere or running here.

        Both cases have to be covered now that the work is not necessarily in
        this process: a batch nothing has claimed is cancelled in the queue,
        and one this worker is driving is cancelled by stopping its task.
        """
        data = self.data(workspace_id)
        if self.queue is not None:
            batch = await data.get_batch(batch_id)
            job_id = (batch or {}).get("job_id")
            if job_id and await self.queue.cancel(job_id, workspace_id):
                await data.update_batch(
                    batch_id,
                    status="cancelled",
                    error="Cancelled before it started.",
                    finished=True,
                )
                return True

        active = self.active
        if active and active.get("batch_id") == batch_id:
            return await self.cancel_active()
        return False

    async def pending_row_indices(
        self, batch_id: str, workspace_id: str, total: int | None = None
    ) -> list[int]:
        """Row indices that have not succeeded.

        This is what makes resume cover all three early exits identically:
        re-login failure, the circuit breaker, and a process restart.

        ``total`` is the number of rows the batch was queued with, and passing
        it is what makes the fourth case work: a batch that stopped before it
        created an execution row for every row it was given. Listing only the
        executions that exist answers "which attempts failed", not "what is
        left to do", and those differ by exactly the rows never attempted.
        """
        executions = await self.data(workspace_id).list_executions(batch_id=batch_id)
        succeeded = {
            int(row["row_index"])
            for row in executions
            if row["status"] == "succeeded" and row["row_index"] is not None
        }
        if total is not None:
            return [index for index in range(total) if index not in succeeded]
        return [
            int(row["row_index"])
            for row in executions
            if row["status"] != "succeeded" and row["row_index"] is not None
        ]

    # -- single-row execution ----------------------------------------------
    async def execute_once(self, request: ExecutionRequest) -> dict[str, Any]:
        """Run setup plus one row, and return the outcome.

        This is the same code path a batch uses, one row wide, so the two
        cannot drift apart.
        """
        usecase = request.usecase
        run_id = uuid.uuid4().hex
        execution_id = uuid.uuid4().hex

        data = self.data(request.workspace_id)
        self._claim(
            f"{usecase.name} (single row)", run_id=run_id, usecase_id=usecase.id, rows=1
        )
        try:
            await data.create_run(
                run_id,
                f"Replay: {usecase.name}",
                None,
                {"usecase_id": usecase.id, "version": request.version, "replay": True},
                owner_id=request.owner_id,
                owner_email=request.owner_email,
            )
            await data.create_execution(
                execution_id,
                usecase.id,
                request.version,
                run_id=run_id,
                inputs=request.inputs,
                owner_id=request.owner_id,
                owner_email=request.owner_email,
            )
            redactor = Redactor(request.secrets.values())
            async with RunLifecycle(
                run_id, data, self.bus, redactor=redactor
            ) as run:
                result = await self._drive(request, run_id, run.sink, redactor)
                if result.repair_proposal is not None:
                    # Before `run.finish`, not after: the live event and the
                    # stored execution must agree on the final error text.
                    await _persist_repair_proposal(
                        data, usecase.id, result,
                        owner_id=request.owner_id, owner_email=request.owner_email,
                    )
                run.finish(replay_terminal(result))

            await data.finish_execution(
                execution_id,
                "succeeded" if result.ok else "failed",
                outputs=result.outputs,
                failed_step_id=result.failed_step_id,
                error=result.error,
                duration_ms=result.duration_ms,
                llm_calls=result.llm_calls,
                llm_tokens=result.llm_tokens,
            )
            return {
                "execution_id": execution_id,
                "run_id": run_id,
                "status": "succeeded" if result.ok else "failed",
                **result.to_dict(),
            }
        finally:
            self._release()

    async def _drive(
        self, request: ExecutionRequest, run_id: str, sink: Any, redactor: Redactor
    ) -> RowResult:
        """Open one session, run setup, run one row, tear down.

        Closing the run is the lifecycle's job, not this method's -- so the
        shielded terminal write lives in exactly one place for every run type.
        """
        browser_config = BrowserConfig.from_settings(
            self.settings, headless=request.headless, browser=request.browser
        )

        await sink.emit(
            RunStarted(
                run_id=run_id,
                seq=sink.reserve_seq(),
                task=f"Replay: {request.usecase.name}",
                options={
                    "usecase_id": request.usecase.id,
                    "version": request.version,
                    "replay": True,
                    "llm_calls": 0,
                },
            )
        )

        result = RowResult(ok=False, error="replay did not start")
        try:
            # Neither read depends on the other's result, so they cost one
            # round trip's worth of wait rather than two -- on a remote
            # database this is the difference between waiting out one RTT
            # and waiting out two before the browser even opens.
            baselines, env = await asyncio.gather(
                self.data(request.workspace_id).baseline_steps(
                    request.usecase.id, request.version, exclude_run=run_id
                ),
                self.resolve_env(request.usecase, request.workspace_id, request.base_url),
            )
            async with PlaywrightSession(browser_config) as browser:
                executor = UseCaseExecutor(
                    request.usecase,
                    browser,
                    sink,
                    run_id=run_id,
                    secrets=request.secrets,
                    redactor=redactor,
                    step_timeout=self.settings.replay_step_timeout,
                    env=env,
                    screenshots=self.settings.replay_screenshots,
                    healer=self.make_healer(
                        request.workspace_id,
                        request.usecase.id,
                        mode=request.usecase.mode,
                    ),
                    baselines=baselines,
                    read_artifact=sink.read_artifact,
                )
                setup = await executor.run_setup()
                if not setup.ok:
                    await emit_replay_error(
                        sink, run_id, "setup_failed", setup.error or "setup failed"
                    )
                    result = setup
                else:
                    row_runner = self.make_row_runner(
                        executor, request.usecase, run_id, sink, redactor
                    )
                    result = await (
                        row_runner(request.inputs)
                        if row_runner
                        else executor.run_row(request.inputs)
                    )
                    await executor.run_teardown()
                    if not result.ok:
                        await emit_replay_error(
                            sink, run_id, "row_failed", result.error or "row failed"
                        )
        except ModeUnavailable as exc:
            await emit_replay_error(sink, run_id, "mode_unavailable", str(exc))
            result = RowResult(ok=False, error=str(exc))
        except BrowserError as exc:
            await emit_replay_error(sink, run_id, "browser_unavailable", str(exc))
            result = RowResult(ok=False, error=str(exc))
        return result


def _stamp(sink: Any):
    """Give each event from the agent its sequence number.

    The same adapter the authoring sessions use, for the same reason: the agent
    package emits with ``seq=0`` because it owns no counter, and the run's sink
    owns the one that is the client's resume token.
    """

    async def emit(event: Any) -> None:
        await sink.emit(event.model_copy(update={"seq": sink.reserve_seq()}))

    return emit


def replay_terminal(result: RowResult) -> Terminal:
    """How a replayed row ended, in the shape the finaliser wants.

    The token counts are reported rather than omitted: they are the number this
    whole feature exists to produce, and a dashboard showing a blank cannot
    tell "free" from "not measured".

    They used to be *hardcoded* to zero, which was true while healing was the
    only thing that could spend and nothing carried what it spent. It stopped
    being true the moment a use case could choose Guided mode, and a run that
    repaired itself twice still claimed to have cost nothing. They come off the
    result now, where a Strict row still reports a measured zero.
    """
    return Terminal(
        status="succeeded" if result.ok else "failed",
        steps=len(result.steps),
        duration_ms=result.duration_ms,
        summary="replay finished" if result.ok else None,
        result={
            "outputs": result.outputs,
            "llm_calls": result.llm_calls,
            "llm_tokens": result.llm_tokens,
        },
        tokens=result.llm_tokens,
        cost_usd=result.llm_usd,
        error=result.error,
    )


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class BatchRequest:
    usecase: UseCase
    version: int
    rows: list[dict[str, Any]]
    secrets: dict[str, str] = field(default_factory=dict)
    #: A base URL given when this run was started, winning over the use case's
    #: target. For a one-off against a branch deployment or a single customer's
    #: tenant, where a standing target would be ceremony for one run.
    base_url: str = ""

    credential_id: str | None = None
    #: The uploaded file these rows came from, when they came from one. Kept so
    #: a run can be traced back to its source; the rows are copied onto the
    #: batch, so deleting the dataset does not rewrite history.
    dataset_id: str | None = None
    headless: bool | None = None
    browser: str | None = None
    #: Row indices to run. ``None`` means all of them; a resume passes the
    #: indices that are not yet ``succeeded``.
    only_rows: list[int] | None = None
    #: The tenant this batch belongs to, and who started it.
    workspace_id: str = ""
    owner_id: str | None = None
    #: Denormalized beside owner_id so the record of who ran this survives the
    #: account being deleted. See db/models.py.
    owner_email: str = ""


async def _run_batch(manager: "ReplayManager", batch_id: str, request: BatchRequest) -> None:
    """Drive one batch to completion on a single shared browser session."""
    store = manager.data(request.workspace_id)
    usecase = request.usecase
    indices = (
        list(request.only_rows)
        if request.only_rows is not None
        else list(range(len(request.rows)))
    )
    rows = [request.rows[i] for i in indices]

    run_id = uuid.uuid4().hex
    redactor = Redactor(request.secrets.values())
    browser_config = BrowserConfig.from_settings(
        manager.settings, headless=request.headless, browser=request.browser
    )

    await store.create_run(
        run_id,
        f"Batch: {usecase.name} ({len(rows)} rows)",
        None,
        {"usecase_id": usecase.id, "version": request.version, "batch_id": batch_id,
         "replay": True, "rows": len(rows)},
        owner_id=request.owner_id,
        owner_email=request.owner_email,
    )
    await store.update_batch(batch_id, status="running")

    async with RunLifecycle(run_id, store, manager.bus, redactor=redactor) as run:
        await _drive_batch(manager, store, run, batch_id, run_id, request, rows, indices,
                           browser_config, redactor)


async def _drive_batch(
    manager: "ReplayManager",
    store: WorkspaceStore,
    run: RunLifecycle,
    batch_id: str,
    run_id: str,
    request: BatchRequest,
    rows: list[dict[str, Any]],
    indices: list[int],
    browser_config: BrowserConfig,
    redactor: Redactor,
) -> None:
    """The batch itself, inside an open lifecycle."""
    usecase = request.usecase
    sink = run.sink
    await sink.emit(
        RunStarted(
            run_id=run_id,
            seq=sink.reserve_seq(),
            task=f"Batch: {usecase.name}",
            options={"usecase_id": usecase.id, "batch_id": batch_id, "rows": len(rows),
                     "replay": True, "llm_calls": 0},
        )
    )

    # One execution row per input row, created up-front so a batch that stops
    # early leaves the unattempted rows visibly `pending` rather than absent.
    execution_ids: dict[int, str] = {}
    for position in range(len(indices)):
        execution_ids[position] = uuid.uuid4().hex
    await store.create_executions(
        [
            {
                "id": execution_ids[position],
                "usecase_id": usecase.id,
                "version": request.version,
                "run_id": run_id,
                "batch_id": batch_id,
                "row_index": row_index,
                "inputs": rows[position],
                "owner_id": request.owner_id,
                "owner_email": request.owner_email,
            }
            for position, row_index in enumerate(indices)
        ]
    )

    progress = BatchProgress(total=len(rows), pending=len(rows))
    try:
        # See the identical comment in `_drive` -- one round trip's wait
        # instead of two, when a remote database makes that wait real.
        baselines, env = await asyncio.gather(
            store.baseline_steps(usecase.id, request.version, exclude_run=run_id),
            manager.resolve_env(usecase, request.workspace_id, request.base_url),
        )
        async with PlaywrightSession(browser_config) as browser:
            executor = UseCaseExecutor(
                usecase,
                browser,
                sink,
                run_id=run_id,
                secrets=request.secrets,
                redactor=redactor,
                step_timeout=manager.settings.replay_step_timeout,
                env=env,
                screenshots=manager.settings.replay_screenshots,
                healer=manager.make_healer(
                    request.workspace_id, usecase.id, mode=usecase.mode
                ),
                baselines=baselines,
                read_artifact=sink.read_artifact,
            )

            runner = BatchRunner(
                executor,
                rows,
                failure_streak_limit=manager.settings.replay_failure_streak_limit,
                # The use case wins when it names one. Politeness belongs to
                # the site, not the installation: one vendor tolerates a
                # request a second and another starts refusing after three,
                # and the person who recorded the workflow knows which is
                # which. See UseCase.row_delay_seconds.
                row_delay=(
                    usecase.row_delay_seconds
                    if usecase.row_delay_seconds is not None
                    else manager.settings.replay_row_delay_seconds
                ),
                sleep=asyncio.sleep,
                indices=indices,
                # Guided rows get the operate graph where the deployment can
                # provide it: replay first, and an agent only on a row that
                # failed. A batch of four thousand rows that all work makes no
                # model call at all, which is the same as it made before.
                run_row=manager.make_row_runner(
                    executor, usecase, run_id, sink, redactor
                ),
            )

            #: Steps a repair has already been proposed for in this batch.
            #: Without this, four thousand rows hitting the same broken
            #: locator would each independently produce their own diagnosis
            #: and their own near-identical draft version -- one is enough
            #: for a person to review; the rest just add noise.
            proposed_for: set[str] = set()

            async def record(position: int, row: dict[str, Any], result: RowResult) -> None:
                """Persist each row as it finishes, so progress survives a crash."""
                if result.repair_proposal is not None:
                    step_id = result.failed_step_id or ""
                    if step_id and step_id in proposed_for:
                        result.repair_proposal = None
                    else:
                        proposed_for.add(step_id)
                        await _persist_repair_proposal(
                            store, usecase.id, result,
                            owner_id=request.owner_id, owner_email=request.owner_email,
                        )
                await store.finish_execution(
                    execution_ids[position],
                    "succeeded" if result.ok else "failed",
                    outputs=result.outputs,
                    failed_step_id=result.failed_step_id,
                    error=result.error,
                    duration_ms=result.duration_ms,
                    llm_calls=result.llm_calls,
                    llm_tokens=result.llm_tokens,
                )
                await store.update_batch(
                    batch_id,
                    succeeded=runner.progress.succeeded,
                    failed=runner.progress.failed,
                )

            runner.on_row = record
            progress = await runner.run()

            if executor.healed:
                await _persist_repairs(store, usecase.id, executor.healed)

    except asyncio.CancelledError:
        progress.stopped_reason = "cancelled"
        raise
    except BrowserError as exc:
        progress.stopped_reason = str(exc)
        await emit_replay_error(sink, run_id, "browser_unavailable", str(exc))
    except Exception as exc:  # noqa: BLE001 - a crash must still close the batch
        log.exception("batch crashed", extra={"batch_id": batch_id})
        progress.stopped_reason = f"{type(exc).__name__}: {exc}"
        await emit_replay_error(sink, run_id, "internal_error", progress.stopped_reason)
    finally:
        run.finish(batch_terminal(progress))
        # The browser session closed when the `async with` above exited, so
        # nothing this batch held is still held from here. Released before the
        # batch row is marked finished, so that "this batch is done" implies
        # "this worker is free" rather than the other way round.
        manager.release_slot()
        # Shielded for the same reason the lifecycle shields its own write: a
        # cancelled batch must still leave a closed batch row behind.
        await asyncio.shield(_close_batch_row(store, batch_id, progress))


def batch_terminal(progress: BatchProgress) -> Terminal:
    """How a batch ended.

    A batch succeeds only if every row did *and* nothing stopped it early --
    the circuit breaker and a failed re-login both leave rows unattempted, and
    reporting that as success would hide exactly the case a resume exists for.
    """
    ok = progress.failed == 0 and not progress.stopped_reason
    return Terminal(
        status="succeeded" if ok else "failed",
        steps=progress.attempted,
        duration_ms=0,
        summary=batch_summary(progress),
        result={**progress.to_dict(), "llm_calls": 0, "llm_tokens": 0},
        error=progress.stopped_reason,
    )


async def _close_batch_row(
    store: WorkspaceStore, batch_id: str, progress: BatchProgress
) -> None:
    """The part of finishing a batch that is not finishing its run.

    The run's terminal event and status are the lifecycle's job; this is the
    batch-shaped record beside it.
    """
    terminal = batch_terminal(progress)
    await store.update_batch(
        batch_id,
        status=terminal.status,
        succeeded=progress.succeeded,
        failed=progress.failed,
        error=progress.stopped_reason,
        finished=True,
    )
    log.info(
        "batch finished",
        extra={"batch_id": batch_id, "status": terminal.status, **progress.to_dict()},
    )


def batch_summary(progress: BatchProgress) -> str:
    return batch_summarise(progress)


async def _persist_repairs(store: WorkspaceStore, usecase_id: str, repairs: list) -> None:
    """Write healed locators back as a new use case version.

    The whole point of healing is that the repair is paid for once. Leaving it
    only in memory would mean paying again on the next batch.
    """
    from healing import apply_repairs

    definition = await store.get_usecase(usecase_id)
    if definition is None:
        return
    patched = apply_repairs(definition, repairs)
    _, version = await store.save_usecase(patched, created_by="healing")
    log.info(
        "wrote healed locators back",
        extra={"usecase_id": usecase_id, "version": version, "repairs": len(repairs)},
    )


async def _persist_repair_proposal(
    store: WorkspaceStore,
    usecase_id: str,
    result: RowResult,
    *,
    owner_id: str | None,
    owner_email: str = "",
) -> None:
    """Turn a row's automatic repair proposal into a draft version, or drop it.

    `agent/operate.py` produced the diagnosis but has no store -- the same
    reason `engine.py` cannot import `llm` -- so this is where the "propose,
    never apply unreviewed" rule that `repair.py`'s own router enforces gets
    enforced here too: a draft version, published only when a person presses
    Publish, exactly the gate every other repair goes through.

    Consumes `result.repair_proposal` either way (sets it back to `None`), so
    nothing downstream of this call sees a pending proposal that either was
    already turned into a draft or was never going to be one.
    """
    pending = result.repair_proposal
    result.repair_proposal = None
    if pending is None:
        return

    from pydantic import ValidationError
    from repair import apply_fixes, candidates, is_unchanged, validate_patched

    try:
        definition = await store.get_usecase(usecase_id)
        if definition is None:
            return
        patched, applied = apply_fixes(
            definition, pending.proposal, candidates(pending.snapshot), pending.snapshot
        )
        if is_unchanged(definition, patched):
            return
        patched["status"] = "draft"
        validate_patched(patched)
        _, version = await store.save_usecase(
            patched, created_by="auto-repair", created_by_id=owner_id
        )
        await store.set_usecase_status(usecase_id, "draft")
        await store.audit(
            "usecase.repair",
            actor_id=owner_id,
            actor_email=owner_email,
            resource_type="usecase",
            resource_id=usecase_id,
            detail={
                "version": version,
                "fixes": len(applied),
                "tokens": pending.proposal.tokens,
                "automatic": True,
            },
        )
    except ValidationError:
        log.warning(
            "an automatic repair proposal produced an invalid draft, dropped",
            extra={"usecase_id": usecase_id},
        )
        return
    except Exception:  # noqa: BLE001 - a bonus attempt must not break the row
        log.exception(
            "failed to save an automatic repair proposal", extra={"usecase_id": usecase_id}
        )
        return

    log.info(
        "an agent recovery could not clear the way; proposed a repair automatically",
        extra={"usecase_id": usecase_id, "version": version},
    )
    result.error = (
        f"{result.error or ''} A repair was proposed automatically and saved as "
        f"draft version {version} for review: {pending.proposal.diagnosis}"
    ).strip()
