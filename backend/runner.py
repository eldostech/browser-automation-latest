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
from bus import EventBus
from config import Settings
from fields import FieldSet, SECRET_PLACEHOLDER_RE
from events import (
    AgentEvent,
    ApprovalRequired,
    ErrorEvent,
    RunFinished,
    RunStarted,
    RunStatus,
    dump_event,
)
from lifecycle import RunLifecycle, Terminal
from llm import LLMAccessError, LLMClient, build_llm
from logging_setup import bind_run_id
from mcp_client import MCPBrowserSession, MCPConfig, MCPConnectionError
from redaction import NULL_REDACTOR, Redactor
from replay import RowResult, UseCaseExecutor, emit_replay_error
from store import Store, WorkspaceStore
from usecase import UseCase

log = logging.getLogger(__name__)

ApprovalDecision = Literal["approved", "rejected", "timeout"]


# ---------------------------------------------------------------------------
# Pub/sub
# ---------------------------------------------------------------------------


# EventBus lives in bus.py so that the in-memory and cross-process versions
# are interchangeable; it is re-exported here because this module has always
# been where callers import it from.


def _revealer(secret_values: dict[str, str]):
    """Build the function that turns placeholders back into credentials.

    Returns None when the run declared no credentials, so the agent's dispatch
    path keeps its arguments untouched in the common case rather than walking
    every tool call looking for something that cannot be there.

    The substitution is the last thing that happens before a value leaves this
    process for the browser. Everything upstream of it -- the prompt, the
    model's reply, the message history, the recorded event -- carries only
    «secret:slot».
    """
    if not secret_values:
        return None

    def reveal(node):
        if isinstance(node, str):
            return SECRET_PLACEHOLDER_RE.sub(
                lambda m: secret_values.get(m.group(1), m.group(0)), node
            )
        if isinstance(node, dict):
            return {k: reveal(v) for k, v in node.items()}
        if isinstance(node, list):
            return [reveal(v) for v in node]
        return node

    return reveal


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

    def __init__(
        self, run_id: str, manager: "RunManager", data: "WorkspaceStore | None" = None
    ) -> None:
        self.run_id = run_id
        self.manager = manager
        #: Scoped store for the status flips below. The manager holds an
        #: unscoped Store serving every tenant, so the gate is handed the view
        #: belonging to this run rather than reaching through the manager.
        self.data = data

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
        if self.data is not None:
            await self.data.set_status(self.run_id, "awaiting_approval")

    async def on_resume(self) -> None:
        if self.data is not None:
            await self.data.set_status(self.run_id, "running")


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
    #: Named values the user declared before recording: the inputs that will
    #: become use case parameters, and the credentials that must never be
    #: written down. See fields.py for why declaring beats inferring.
    fields: FieldSet = field(default_factory=FieldSet)
    #: Which tenant this run belongs to, and who asked for it. Carried on the
    #: request rather than held on the manager because the manager is a
    #: process-wide singleton serving every workspace at once.
    workspace_id: str = ""
    owner_id: str | None = None
    #: Denormalized beside owner_id so the record of who ran this survives the
    #: account being deleted. See db/models.py.
    owner_email: str = ""


class RunManager:
    def __init__(
        self,
        store: Store,
        settings: Settings,
        bus: EventBus | None = None,
        llm: LLMClient | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.bus = bus or EventBus()
        #: Passed to every agent graph, so an interrupted run has state to
        #: resume from. See checkpoints.py.
        self.checkpointer = checkpointer
        #: An explicit override, when one is supplied. It serves EVERY role,
        #: so a test that scripts one client still covers all three.
        self._llm = llm
        #: Lazily built, one per role. Kept separate from `_llm` -- building
        #: the driver into that slot would have silently made every other role
        #: return the driver.
        self._driver_llm: LLMClient | None = None
        self._distill_llm: LLMClient | None = None
        self._repair_llm: LLMClient | None = None
        self._tasks: dict[str, asyncio.Task] = {}
        self._approvals: dict[str, dict[str, PendingApproval]] = {}
        self._pending_events: dict[str, ApprovalRequired] = {}

    # -- llm ----------------------------------------------------------------
    #
    # Three roles, built lazily and cached. A test that injects `_llm` gets
    # that client for every role, so a scripted model still covers all of them.
    @property
    def llm(self) -> LLMClient:
        """The driver: the agent loop that records a use case."""
        if self._llm is not None:
            return self._llm
        if self._driver_llm is None:
            self._driver_llm = build_llm(self.settings)
        return self._driver_llm

    @property
    def distill_llm(self) -> LLMClient:
        """The single call that turns a recording into a use case."""
        if self._llm is not None:
            return self._llm
        if self._distill_llm is None:
            self._distill_llm = build_llm(self.settings, self.settings.distill_model)
        return self._distill_llm

    @property
    def repair_llm(self) -> LLMClient:
        """Self-healing mid-run, and repairing a failed use case afterwards."""
        if self._llm is not None:
            return self._llm
        if self._repair_llm is None:
            self._repair_llm = build_llm(self.settings, self.settings.llm_repair_model)
        return self._repair_llm

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

    def _data(self, request: RunRequest) -> WorkspaceStore:
        """The store, confined to the workspace this run belongs to.

        Built per request rather than held on the manager: one manager serves
        every tenant, so a scoped store on ``self`` would be the wrong one for
        all but the first caller.
        """
        return self.store.workspace(request.workspace_id)

    # -- lifecycle ----------------------------------------------------------
    async def start_run(self, request: RunRequest) -> str:
        run_id = uuid.uuid4().hex
        options = request.options or self.default_options()
        spec = AgentSpec(
            run_id=run_id,
            task=request.task,
            start_url=request.start_url,
            options=options,
            fields_block=request.fields.prompt_block(),
        )

        persisted = {
            **options.to_dict(),
            "headless": self.settings.mcp_headless if request.headless is None else request.headless,
            "browser": request.browser or self.settings.mcp_browser,
            # Input names and values, and secret slot *names* only. This is what
            # lets distillation parameterise the recording deterministically
            # later, and it is safe to store because the secret values are not
            # in it.
            "declared": request.fields.persistable(),
        }
        await self._data(request).create_run(
            run_id,
            request.task,
            request.start_url,
            persisted,
            owner_id=request.owner_id,
            owner_email=request.owner_email,
        )

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
        data = self._data(request)
        outcome: AgentOutcome | None = None
        agent: BrowserAgent | None = None

        mcp_config = MCPConfig.from_settings(
            self.settings,
            headless=request.headless,
            browser=request.browser,
        )

        try:
            async with RunLifecycle(
                run_id,
                data,
                self.bus,
                secrets=request.secrets,
                sink_wrapper=lambda inner: _TrackingSink(inner, self, run_id),
            ) as run:
                gate = RunApprovalGate(run_id, self, data)
                try:
                    log.info(
                        "run starting",
                        extra={"task": spec.task, "start_url": spec.start_url},
                    )
                    async with MCPBrowserSession(mcp_config) as mcp:
                        agent = BrowserAgent(
                            spec, mcp, self.llm, run.sink, gate,
                            checkpointer=self.checkpointer,
                            reveal_secrets=_revealer(request.fields.secret_values),
                        )
                        outcome = await agent.run()

                except asyncio.CancelledError:
                    log.info("run cancelled")
                    raise
                except (MCPConnectionError, LLMAccessError) as exc:
                    outcome = AgentOutcome(
                        status="failed",
                        steps=agent.step if agent else 0,
                        duration_ms=run.elapsed_ms,
                        error=str(exc),
                    )
                    await run.sink.emit(
                        ErrorEvent(
                            run_id=run_id,
                            seq=run.sink.reserve_seq(),
                            kind=(
                                "llm_unavailable"
                                if isinstance(exc, LLMAccessError)
                                else "mcp_unavailable"
                            ),
                            message=str(exc),
                            recoverable=False,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - the run fails, the process does not
                    log.exception("run crashed", extra={"run_id": run_id})
                    outcome = AgentOutcome(
                        status="failed",
                        steps=agent.step if agent else 0,
                        duration_ms=run.elapsed_ms,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    await run.sink.emit(
                        ErrorEvent(
                            run_id=run_id,
                            seq=run.sink.reserve_seq(),
                            kind="internal_error",
                            message=str(exc),
                            recoverable=False,
                        )
                    )
                finally:
                    if outcome is not None:
                        run.finish(
                            Terminal(
                                status=outcome.status,
                                steps=outcome.steps,
                                duration_ms=outcome.duration_ms,
                                summary=outcome.summary,
                                result=outcome.result,
                                error=outcome.error,
                            )
                        )
                    # Whatever happened, this run is no longer waiting on
                    # anyone. Kept here rather than in the lifecycle because
                    # approvals are the manager's bookkeeping, not the run's.
                    self._approvals.pop(run_id, None)
                    self._pending_events.pop(run_id, None)
        except asyncio.CancelledError:
            # The lifecycle has already persisted a "cancelled" terminal event.
            raise


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
    #: The tenant this execution belongs to, and who asked for it.
    workspace_id: str = ""
    owner_id: str | None = None
    #: Denormalized beside owner_id so the record of who ran this survives the
    #: account being deleted. See db/models.py.
    owner_email: str = ""


class ReplayManager:
    """Runs stored use cases. Deliberately has no LLM client of any kind.

    A single slot guards execution because every run drives a real browser and
    the design settled on one at a time. Holding it explicitly -- and reporting
    who holds it -- beats queueing invisibly or letting two runs fight over the
    same session.
    """

    def __init__(
        self,
        store: Store,
        settings: Settings,
        bus: EventBus,
        llm_factory: Any = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.bus = bus
        #: Supplies a model *only* for healing, and only when healing is
        #: enabled. Left None, there is no route from a replay to an LLM.
        self.llm_factory = llm_factory
        self._slot: dict[str, Any] | None = None
        self._task: asyncio.Task | None = None

    def data(self, workspace_id: str) -> WorkspaceStore:
        """The store, confined to one workspace.

        One manager serves every tenant, so the scope comes from the request
        rather than from the manager.
        """
        return self.store.workspace(workspace_id)

    def make_healer(self) -> Any:
        """A healer, or None when healing is off.

        Returning None is the common case and is what keeps the executor's
        zero-token guarantee true by construction.
        """
        if not self.settings.replay_healing_enabled or self.llm_factory is None:
            return None
        from healing import HealingBudget, StepHealer

        return StepHealer(
            self.llm_factory(),
            HealingBudget(
                max_attempts=self.settings.replay_heal_max_attempts,
                max_tokens=self.settings.replay_heal_max_tokens,
            ),
        )

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

    def release_slot(self) -> None:
        """Free the execution slot. Idempotent, and safe to call twice.

        Public because the batch driver releases it at a specific point -- once
        the browser session is closed but before the batch row is marked
        finished -- rather than leaving it to the task wrapper, which runs
        later.
        """
        self._release()

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
        await self.data(request.workspace_id).create_batch(
            batch_id,
            usecase.id,
            request.version,
            total=len(indices),
            credential_id=request.credential_id,
            owner_id=request.owner_id,
            owner_email=request.owner_email,
        )

        async def drive() -> None:
            try:
                await _run_batch(self, batch_id, request)
            finally:
                self._release()

        self._task = asyncio.create_task(drive(), name=f"batch-{batch_id}")
        return batch_id

    async def pending_row_indices(self, batch_id: str, workspace_id: str) -> list[int]:
        """Row indices that are not ``succeeded``.

        This is what makes resume cover all three early exits identically:
        re-login failure, the circuit breaker, and a process restart.
        """
        executions = await self.data(workspace_id).list_executions(batch_id=batch_id)
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
                run.finish(replay_terminal(result))

            await data.finish_execution(
                execution_id,
                "succeeded" if result.ok else "failed",
                outputs=result.outputs,
                failed_step_id=result.failed_step_id,
                error=result.error,
                duration_ms=result.duration_ms,
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
                    screenshots=self.settings.replay_screenshots,
                    healer=self.make_healer(),
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
        return result


def replay_terminal(result: RowResult) -> Terminal:
    """How a replayed row ended, in the shape the finaliser wants.

    ``llm_calls`` and ``llm_tokens`` are reported as zero rather than omitted:
    they are the number this whole feature exists to produce, and a dashboard
    that shows a blank cannot tell "free" from "not measured".
    """
    return Terminal(
        status="succeeded" if result.ok else "failed",
        steps=len(result.steps),
        duration_ms=result.duration_ms,
        summary="replay finished" if result.ok else None,
        result={"outputs": result.outputs, "llm_calls": 0, "llm_tokens": 0},
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
    credential_id: str | None = None
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
    mcp_config = MCPConfig.from_settings(
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
                           mcp_config, redactor)


async def _drive_batch(
    manager: "ReplayManager",
    store: WorkspaceStore,
    run: RunLifecycle,
    batch_id: str,
    run_id: str,
    request: BatchRequest,
    rows: list[dict[str, Any]],
    indices: list[int],
    mcp_config: MCPConfig,
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
            owner_id=request.owner_id,
            owner_email=request.owner_email,
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
                screenshots=manager.settings.replay_screenshots,
                healer=manager.make_healer(),
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

            if executor.healed:
                await _persist_repairs(store, usecase.id, executor.healed)

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
        run.finish(batch_terminal(progress))
        # The browser session closed when the `async with` above exited, so the
        # execution slot is genuinely free from here. Release it *before*
        # marking the batch finished, so that "this batch is done" implies
        # "another one can start" -- the reverse order left a window in which
        # the UI showed a completed batch and the next request got a 409.
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
