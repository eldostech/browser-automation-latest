"""Run orchestration: owns the lifecycle of every run and the fan-out of its
events to WebSocket subscribers.

Responsibilities kept here rather than in ``agent.py`` so the agent loop stays
readable and testable in isolation:

* one ``asyncio.Task`` per run, cancellable from the API
* one MCP/browser session per run, torn down in a ``finally`` that also runs on
  cancellation
* sequence-number allocation and persistence of every event
* the approval rendezvous between the paused loop and the HTTP endpoint
* an in-memory pub/sub so N dashboard tabs can watch one run
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from agent import AgentOutcome, AgentSpec, BrowserAgent, RunOptions
from batch import BatchProgress, BatchRunner, new_batch_id
from batch import summarise as batch_summarise
from config import Settings
from events import (
    AgentEvent,
    ApprovalRequired,
    ErrorEvent,
    RunFinished,
    RunStarted,
    RunStatus,
    dump_event,
)
from llm import LLMClient, build_llm
from logging_setup import bind_run_id
from mcp_client import MCPBrowserSession, MCPConfig, MCPConnectionError
from redaction import NULL_REDACTOR, Redactor
from replay import RowResult, UseCaseExecutor, emit_replay_error
from store import Store
from usecase import UseCase

log = logging.getLogger(__name__)

ApprovalDecision = Literal["approved", "rejected", "timeout"]


# ---------------------------------------------------------------------------
# Pub/sub
# ---------------------------------------------------------------------------


class EventBus:
    """In-memory fan-out. One process only; move to Redis to scale out."""

    def __init__(self, queue_size: int = 1000) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._queue_size = queue_size

    def subscribe(self, run_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.setdefault(run_id, set()).add(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        subscribers = self._subscribers.get(run_id)
        if not subscribers:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(run_id, None)

    def publish(self, run_id: str, event: AgentEvent) -> None:
        for queue in list(self._subscribers.get(run_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A stalled client must not slow the agent down. It will
                # reconnect and replay from its last seq.
                log.warning("dropping event for slow subscriber", extra={"run_id": run_id})

    def subscriber_count(self, run_id: str) -> int:
        return len(self._subscribers.get(run_id, ()))


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
        store: Store,
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


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PendingApproval:
    approval_id: str
    future: asyncio.Future
    event: ApprovalRequired | None = None


class RunApprovalGate:
    """The rendezvous between a paused agent loop and ``POST /approve``."""

    def __init__(self, run_id: str, manager: "RunManager") -> None:
        self.run_id = run_id
        self.manager = manager

    async def request(
        self, approval_id: str, timeout: float
    ) -> tuple[ApprovalDecision, str | None]:
        loop = asyncio.get_running_loop()
        pending = PendingApproval(approval_id=approval_id, future=loop.create_future())
        self.manager.register_approval(self.run_id, pending)
        try:
            decision, note = await asyncio.wait_for(pending.future, timeout=timeout)
            return decision, note
        except asyncio.TimeoutError:
            log.warning(
                "approval timed out", extra={"run_id": self.run_id, "approval_id": approval_id}
            )
            return "timeout", None
        finally:
            self.manager.clear_approval(self.run_id, approval_id)

    async def on_pause(self) -> None:
        await self.manager.store.set_status(self.run_id, "awaiting_approval")

    async def on_resume(self) -> None:
        await self.manager.store.set_status(self.run_id, "running")


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RunRequest:
    task: str
    start_url: str | None = None
    options: RunOptions = None  # type: ignore[assignment]
    headless: bool | None = None
    browser: str | None = None
    #: Values to keep out of the event log, the database and the logs. Anything
    #: here is replaced with a placeholder on its way to any of the three.
    secrets: list[str] = field(default_factory=list)


class RunManager:
    def __init__(
        self,
        store: Store,
        settings: Settings,
        bus: EventBus | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.bus = bus or EventBus()
        self._llm = llm
        self._tasks: dict[str, asyncio.Task] = {}
        self._approvals: dict[str, dict[str, PendingApproval]] = {}
        self._pending_events: dict[str, ApprovalRequired] = {}

    # -- llm ----------------------------------------------------------------
    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    # -- defaults -----------------------------------------------------------
    def default_options(self) -> RunOptions:
        s = self.settings
        return RunOptions(
            max_steps=s.agent_max_steps,
            timeout_seconds=s.agent_timeout_seconds,
            allowed_domains=list(s.agent_allowed_domains),
            require_approval=s.agent_require_approval,
            approval_timeout_seconds=s.agent_approval_timeout_seconds,
            screenshot_every_step=s.agent_screenshot_every_step,
            max_tool_result_chars=s.agent_max_tool_result_chars,
            max_history_messages=s.agent_max_history_messages,
        )

    # -- lifecycle ----------------------------------------------------------
    async def start_run(self, request: RunRequest) -> str:
        run_id = uuid.uuid4().hex
        options = request.options or self.default_options()
        spec = AgentSpec(
            run_id=run_id, task=request.task, start_url=request.start_url, options=options
        )

        persisted = {
            **options.to_dict(),
            "headless": self.settings.mcp_headless if request.headless is None else request.headless,
            "browser": request.browser or self.settings.mcp_browser,
        }
        await self.store.create_run(run_id, request.task, request.start_url, persisted)

        task = asyncio.create_task(self._execute(spec, request), name=f"run-{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda _t, rid=run_id: self._tasks.pop(rid, None))
        return run_id

    async def cancel_run(self, run_id: str) -> bool:
        # Resolve any pending approval first so the loop is not blocked when
        # the cancellation lands.
        for pending in list(self._approvals.get(run_id, {}).values()):
            if not pending.future.done():
                pending.future.set_result(("rejected", "run cancelled"))

        task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    def is_active(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        return task is not None and not task.done()

    async def shutdown(self) -> None:
        for run_id in list(self._tasks):
            await self.cancel_run(run_id)
        tasks = list(self._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # -- approvals ----------------------------------------------------------
    def register_approval(self, run_id: str, pending: PendingApproval) -> None:
        self._approvals.setdefault(run_id, {})[pending.approval_id] = pending

    def clear_approval(self, run_id: str, approval_id: str) -> None:
        approvals = self._approvals.get(run_id)
        if approvals:
            approvals.pop(approval_id, None)
            if not approvals:
                self._approvals.pop(run_id, None)
        self._pending_events.pop(run_id, None)

    def note_pending_event(self, run_id: str, event: ApprovalRequired) -> None:
        self._pending_events[run_id] = event

    def pending_approval(self, run_id: str) -> dict[str, Any] | None:
        event = self._pending_events.get(run_id)
        return dump_event(event) if event else None

    def resolve_approval(
        self, run_id: str, approval_id: str | None, decision: ApprovalDecision, note: str | None
    ) -> bool:
        approvals = self._approvals.get(run_id) or {}
        if approval_id:
            pending = approvals.get(approval_id)
        else:
            # Convenience: approve "whatever is waiting" when there is exactly one.
            pending = next(iter(approvals.values()), None) if len(approvals) == 1 else None
        if pending is None or pending.future.done():
            return False
        pending.future.set_result((decision, note))
        return True

    # -- execution ----------------------------------------------------------
    async def _execute(self, spec: AgentSpec, request: RunRequest) -> None:
        run_id = spec.run_id
        bind_run_id(run_id)
        started = time.monotonic()

        redactor = Redactor(request.secrets)
        sink = _TrackingSink(
            RunEventSink(run_id, self.store, self.bus, redactor=redactor), self, run_id
        )
        gate = RunApprovalGate(run_id, self)
        outcome: AgentOutcome | None = None
        cancelled = False
        agent: BrowserAgent | None = None

        mcp_config = MCPConfig.from_settings(
            self.settings,
            headless=request.headless,
            browser=request.browser,
        )

        try:
            await self.store.mark_started(run_id)
            log.info("run starting", extra={"task": spec.task, "start_url": spec.start_url})

            async with MCPBrowserSession(mcp_config) as mcp:
                agent = BrowserAgent(spec, mcp, self.llm, sink, gate)
                outcome = await agent.run()

        except asyncio.CancelledError:
            cancelled = True
            log.info("run cancelled")
        except MCPConnectionError as exc:
            outcome = AgentOutcome(
                status="failed",
                steps=agent.step if agent else 0,
                duration_ms=int((time.monotonic() - started) * 1000),
                error=str(exc),
            )
            await sink.emit(
                ErrorEvent(
                    run_id=run_id,
                    seq=sink.reserve_seq(),
                    kind="mcp_unavailable",
                    message=str(exc),
                    recoverable=False,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a crash must still close the run
            log.exception("run crashed")
            outcome = AgentOutcome(
                status="failed",
                steps=agent.step if agent else 0,
                duration_ms=int((time.monotonic() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
            await sink.emit(
                ErrorEvent(
                    run_id=run_id,
                    seq=sink.reserve_seq(),
                    kind="internal_error",
                    message=str(exc),
                    recoverable=False,
                )
            )
        finally:
            # Shielded: even a cancelled run must persist its terminal event,
            # otherwise the dashboard's WebSocket would wait forever.
            await asyncio.shield(
                self._finalise(
                    run_id=run_id,
                    sink=sink,
                    outcome=outcome,
                    cancelled=cancelled,
                    steps=agent.step if agent else 0,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            )
            bind_run_id(None)

        if cancelled:
            raise asyncio.CancelledError()

    async def _finalise(
        self,
        *,
        run_id: str,
        sink: "_TrackingSink",
        outcome: AgentOutcome | None,
        cancelled: bool,
        steps: int,
        duration_ms: int,
    ) -> None:
        if cancelled or outcome is None:
            status: RunStatus = "cancelled" if cancelled else "failed"
            error = None if cancelled else "The run ended without producing an outcome."
            summary = None
            result = None
        else:
            status = outcome.status
            error = outcome.error
            summary = outcome.summary
            result = outcome.result
            steps = outcome.steps
            duration_ms = outcome.duration_ms

        event = RunFinished(
            run_id=run_id,
            seq=sink.reserve_seq(),
            status=status,
            steps=steps,
            duration_ms=duration_ms,
            summary=summary,
            result=result,
            error=error,
        )
        await sink.emit(event)
        await self.store.finish_run(
            run_id,
            status,
            steps=steps,
            duration_ms=duration_ms,
            summary=summary,
            result=result,
            error=error,
        )
        self._approvals.pop(run_id, None)
        self._pending_events.pop(run_id, None)
        log.info(
            "run finished",
            extra={"status": status, "steps": steps, "duration_ms": duration_ms},
        )


class _TrackingSink:
    """Wraps :class:`RunEventSink` so the manager can remember the approval
    currently blocking a run (used by ``GET /api/runs/{id}``)."""

    def __init__(self, inner: RunEventSink, manager: RunManager, run_id: str) -> None:
        self._inner = inner
        self._manager = manager
        self._run_id = run_id

    def reserve_seq(self) -> int:
        return self._inner.reserve_seq()

    async def emit(self, event: AgentEvent) -> None:
        if isinstance(event, ApprovalRequired):
            self._manager.note_pending_event(self._run_id, event)
        await self._inner.emit(event)

    async def save_screenshot(
        self, data: bytes, *, seq: int, mime: str = "image/png"
    ) -> tuple[str, str] | None:
        return await self._inner.save_screenshot(data, seq=seq, mime=mime)


# ---------------------------------------------------------------------------
# Use-case execution (no LLM)
# ---------------------------------------------------------------------------


class ExecutionBusy(RuntimeError):
    """Something already holds the single execution slot."""

    def __init__(self, holder: dict[str, Any]) -> None:
        super().__init__(
            f"another use-case execution is already running ({holder.get('label')}). "
            "One at a time: they share the browser."
        )
        self.holder = holder


@dataclass(slots=True)
class ExecutionRequest:
    usecase: UseCase
    version: int
    inputs: dict[str, Any] = field(default_factory=dict)
    secrets: dict[str, str] = field(default_factory=dict)
    headless: bool | None = None
    browser: str | None = None


class ReplayManager:
    """Runs stored use cases. Deliberately has no LLM client of any kind.

    A single slot guards execution because every run drives a real browser and
    the design settled on one at a time. Holding it explicitly -- and reporting
    who holds it -- beats queueing invisibly or letting two runs fight over the
    same session.
    """

    def __init__(self, store: Store, settings: Settings, bus: EventBus) -> None:
        self.store = store
        self.settings = settings
        self.bus = bus
        self._slot: dict[str, Any] | None = None
        self._task: asyncio.Task | None = None

    # -- the single slot ----------------------------------------------------
    @property
    def active(self) -> dict[str, Any] | None:
        """What holds the execution slot, if anything."""
        if self._task is not None and self._task.done():
            self._slot = None
            self._task = None
        return self._slot

    def _claim(self, label: str, **fields: Any) -> None:
        holder = self.active
        if holder is not None:
            raise ExecutionBusy(holder)
        self._slot = {"label": label, "started_at": _iso_now(), **fields}

    def _release(self) -> None:
        self._slot = None
        self._task = None

    async def cancel_active(self) -> bool:
        if self._task is None or self._task.done():
            return False
        self._task.cancel()
        return True

    # -- batches ------------------------------------------------------------
    async def start_batch(self, request: BatchRequest) -> str:
        """Claim the slot and drive a batch in the background.

        Returns immediately with the batch id: a thousand rows is not an HTTP
        request, so progress is polled from ``GET /api/batches/{id}``.
        """
        usecase = request.usecase
        indices = (
            list(request.only_rows)
            if request.only_rows is not None
            else list(range(len(request.rows)))
        )
        batch_id = new_batch_id()

        self._claim(
            f"{usecase.name} ({len(indices)} rows)",
            batch_id=batch_id,
            usecase_id=usecase.id,
            rows=len(indices),
        )
        await self.store.create_batch(
            batch_id,
            usecase.id,
            request.version,
            total=len(indices),
            credential_id=request.credential_id,
        )

        async def drive() -> None:
            try:
                await _run_batch(self, batch_id, request)
            finally:
                self._release()

        self._task = asyncio.create_task(drive(), name=f"batch-{batch_id}")
        return batch_id

    async def pending_row_indices(self, batch_id: str) -> list[int]:
        """Row indices that are not ``succeeded``.

        This is what makes resume cover all three early exits identically:
        re-login failure, the circuit breaker, and a process restart.
        """
        executions = await self.store.list_executions(batch_id=batch_id)
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

        self._claim(
            f"{usecase.name} (single row)", run_id=run_id, usecase_id=usecase.id, rows=1
        )
        try:
            await self.store.create_run(
                run_id,
                f"Replay: {usecase.name}",
                None,
                {"usecase_id": usecase.id, "version": request.version, "replay": True},
            )
            await self.store.create_execution(
                execution_id,
                usecase.id,
                request.version,
                run_id=run_id,
                inputs=request.inputs,
            )
            await self.store.mark_started(run_id)

            result = await self._drive(request, run_id)

            await self.store.finish_execution(
                execution_id,
                "succeeded" if result.ok else "failed",
                outputs=result.outputs,
                failed_step_id=result.failed_step_id,
                error=result.error,
                duration_ms=result.duration_ms,
            )
            await self.store.finish_run(
                run_id,
                "succeeded" if result.ok else "failed",
                steps=len(result.steps),
                duration_ms=result.duration_ms,
                summary=None if result.ok else result.error,
                result={"outputs": result.outputs, "llm_tokens": 0},
                error=result.error,
            )
            return {
                "execution_id": execution_id,
                "run_id": run_id,
                "status": "succeeded" if result.ok else "failed",
                **result.to_dict(),
            }
        finally:
            self._release()

    async def _drive(self, request: ExecutionRequest, run_id: str) -> RowResult:
        """Open one session, run setup, run one row, tear down."""
        redactor = Redactor(request.secrets.values())
        sink = RunEventSink(run_id, self.store, self.bus, redactor=redactor)
        mcp_config = MCPConfig.from_settings(
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
            async with MCPBrowserSession(mcp_config) as mcp:
                executor = UseCaseExecutor(
                    request.usecase,
                    mcp,
                    sink,
                    run_id=run_id,
                    secrets=request.secrets,
                    redactor=redactor,
                    step_timeout=self.settings.replay_step_timeout,
                )
                setup = await executor.run_setup()
                if not setup.ok:
                    await emit_replay_error(
                        sink, run_id, "setup_failed", setup.error or "setup failed"
                    )
                    result = setup
                else:
                    result = await executor.run_row(request.inputs)
                    await executor.run_teardown()
                    if not result.ok:
                        await emit_replay_error(
                            sink, run_id, "row_failed", result.error or "row failed"
                        )
        except MCPConnectionError as exc:
            await emit_replay_error(sink, run_id, "mcp_unavailable", str(exc))
            result = RowResult(ok=False, error=str(exc))
        finally:
            # Shielded like the agent's own finaliser: a cancelled replay must
            # still close the run, or the dashboard's socket waits forever.
            await asyncio.shield(
                sink.emit(
                    RunFinished(
                        run_id=run_id,
                        seq=sink.reserve_seq(),
                        status="succeeded" if result.ok else "failed",
                        steps=len(result.steps),
                        duration_ms=result.duration_ms,
                        summary="replay finished" if result.ok else None,
                        # The number this whole feature exists to produce.
                        result={"outputs": result.outputs, "llm_calls": 0, "llm_tokens": 0},
                        error=result.error,
                    )
                )
            )
        return result


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class BatchRequest:
    usecase: UseCase
    version: int
    rows: list[dict[str, Any]]
    secrets: dict[str, str] = field(default_factory=dict)
    credential_id: str | None = None
    headless: bool | None = None
    browser: str | None = None
    #: Row indices to run. ``None`` means all of them; a resume passes the
    #: indices that are not yet ``succeeded``.
    only_rows: list[int] | None = None


async def _run_batch(manager: "ReplayManager", batch_id: str, request: BatchRequest) -> None:
    """Drive one batch to completion on a single shared browser session."""
    store = manager.store
    usecase = request.usecase
    indices = (
        list(request.only_rows)
        if request.only_rows is not None
        else list(range(len(request.rows)))
    )
    rows = [request.rows[i] for i in indices]

    run_id = uuid.uuid4().hex
    redactor = Redactor(request.secrets.values())
    sink = RunEventSink(run_id, store, manager.bus, redactor=redactor)
    mcp_config = MCPConfig.from_settings(
        manager.settings, headless=request.headless, browser=request.browser
    )

    await store.create_run(
        run_id,
        f"Batch: {usecase.name} ({len(rows)} rows)",
        None,
        {"usecase_id": usecase.id, "version": request.version, "batch_id": batch_id,
         "replay": True, "rows": len(rows)},
    )
    await store.mark_started(run_id)
    await store.update_batch(batch_id, status="running")
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
    for position, row_index in enumerate(indices):
        execution_id = uuid.uuid4().hex
        execution_ids[position] = execution_id
        await store.create_execution(
            execution_id,
            usecase.id,
            request.version,
            run_id=run_id,
            batch_id=batch_id,
            row_index=row_index,
            inputs=rows[position],
        )

    progress = BatchProgress(total=len(rows), pending=len(rows))
    try:
        async with MCPBrowserSession(mcp_config) as mcp:
            executor = UseCaseExecutor(
                usecase,
                mcp,
                sink,
                run_id=run_id,
                secrets=request.secrets,
                redactor=redactor,
                step_timeout=manager.settings.replay_step_timeout,
            )

            runner = BatchRunner(
                executor,
                rows,
                failure_streak_limit=manager.settings.replay_failure_streak_limit,
                row_delay=manager.settings.replay_row_delay_seconds,
                sleep=asyncio.sleep,
            )

            async def record(position: int, row: dict[str, Any], result: RowResult) -> None:
                """Persist each row as it finishes, so progress survives a crash."""
                await store.finish_execution(
                    execution_ids[position],
                    "succeeded" if result.ok else "failed",
                    outputs=result.outputs,
                    failed_step_id=result.failed_step_id,
                    error=result.error,
                    duration_ms=result.duration_ms,
                )
                await store.update_batch(
                    batch_id,
                    succeeded=runner.progress.succeeded,
                    failed=runner.progress.failed,
                )

            runner.on_row = record
            progress = await runner.run()

    except asyncio.CancelledError:
        progress.stopped_reason = "cancelled"
        raise
    except MCPConnectionError as exc:
        progress.stopped_reason = str(exc)
        await emit_replay_error(sink, run_id, "mcp_unavailable", str(exc))
    except Exception as exc:  # noqa: BLE001 - a crash must still close the batch
        log.exception("batch crashed", extra={"batch_id": batch_id})
        progress.stopped_reason = f"{type(exc).__name__}: {exc}"
        await emit_replay_error(sink, run_id, "internal_error", progress.stopped_reason)
    finally:
        await asyncio.shield(
            _finalise_batch(manager, batch_id, run_id, sink, progress)
        )


async def _finalise_batch(
    manager: "ReplayManager",
    batch_id: str,
    run_id: str,
    sink: RunEventSink,
    progress: BatchProgress,
) -> None:
    status = "succeeded" if progress.failed == 0 and not progress.stopped_reason else "failed"
    await manager.store.update_batch(
        batch_id,
        status=status,
        succeeded=progress.succeeded,
        failed=progress.failed,
        error=progress.stopped_reason,
        finished=True,
    )
    await manager.store.finish_run(
        run_id,
        "succeeded" if status == "succeeded" else "failed",
        steps=progress.attempted,
        duration_ms=0,
        summary=batch_summary(progress),
        result={**progress.to_dict(), "llm_calls": 0, "llm_tokens": 0},
        error=progress.stopped_reason,
    )
    await sink.emit(
        RunFinished(
            run_id=run_id,
            seq=sink.reserve_seq(),
            status="succeeded" if status == "succeeded" else "failed",
            steps=progress.attempted,
            duration_ms=0,
            summary=batch_summary(progress),
            result={**progress.to_dict(), "llm_calls": 0, "llm_tokens": 0},
            error=progress.stopped_reason,
        )
    )
    log.info(
        "batch finished",
        extra={"batch_id": batch_id, "status": status, **progress.to_dict()},
    )


def batch_summary(progress: BatchProgress) -> str:
    return batch_summarise(progress)
