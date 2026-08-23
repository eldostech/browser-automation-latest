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
from dataclasses import dataclass
from typing import Any, Literal

from agent import AgentOutcome, AgentSpec, BrowserAgent, RunOptions
from config import Settings
from events import (
    AgentEvent,
    ApprovalRequired,
    ErrorEvent,
    RunFinished,
    RunStatus,
    dump_event,
)
from llm import LLMClient, build_llm
from logging_setup import bind_run_id
from mcp_client import MCPBrowserSession, MCPConfig, MCPConnectionError
from store import Store

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
    """Implements ``agent.EventSink``: allocate seq, persist, broadcast."""

    def __init__(self, run_id: str, store: Store, bus: EventBus, api_base: str = "") -> None:
        self.run_id = run_id
        self.store = store
        self.bus = bus
        self.api_base = api_base.rstrip("/")
        self._seq = 0

    def reserve_seq(self) -> int:
        self._seq += 1
        return self._seq

    @property
    def last_seq(self) -> int:
        return self._seq

    async def emit(self, event: AgentEvent) -> None:
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

        sink = _TrackingSink(RunEventSink(run_id, self.store, self.bus), self, run_id)
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
