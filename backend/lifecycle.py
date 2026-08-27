"""The arc every run shares, written once.

Three places used to hand-roll the same sequence: the recording agent, a
single-row replay, and a batch. Each one bound the run id for logging, built a
redactor and a sink, opened a browser session, drove something, and then --
inside a ``finally``, wrapped in ``asyncio.shield`` -- emitted a terminal event
and persisted a terminal status.

That shield is the subtlety worth centralising. Without it a cancelled run
never writes ``run_finished``, and the dashboard's WebSocket waits forever for
an event that will not come. It was correct in all three copies, which is
luckier than it sounds: duplicated subtlety is where the next bug lives, and
the next person to add a fourth run type would have had to know to copy it.

What differs between the three is only *what happened*, expressed as a
:class:`Terminal`. The lifecycle owns everything around that.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from events import RunFinished, RunStatus
from logging_setup import bind_run_id
from redaction import NULL_REDACTOR, Redactor
from store import WorkspaceStore

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Terminal:
    """How a run ended, in the one shape the finaliser needs.

    Every run type produces one of these from whatever it actually computed --
    an ``AgentOutcome``, a ``RowResult``, a ``BatchProgress`` -- so the
    finaliser does not need to know which kind of run it just closed.
    """

    status: RunStatus = "failed"
    steps: int = 0
    duration_ms: int = 0
    summary: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class RunLifecycle:
    """Async context manager owning the shared parts of a run.

    Used as::

        async with RunLifecycle(run_id, data, bus, secrets=...) as run:
            ...drive the work, then...
            run.finish(Terminal(status="succeeded", steps=n, ...))

    Leaving the block -- normally, by exception, or by cancellation -- persists
    whatever ``finish`` was last given. A block that never calls it closes the
    run as failed rather than leaving it running forever, because "the process
    stopped without saying why" is still an ending the UI has to be told about.
    """

    def __init__(
        self,
        run_id: str,
        data: WorkspaceStore,
        bus: Any,
        *,
        secrets: Any = (),
        api_base: str = "",
        redactor: Redactor | None = None,
        sink_wrapper: Any = None,
        mark_started: bool = True,
    ) -> None:
        from runner import RunEventSink  # circular at module scope; fine here

        self.run_id = run_id
        self.data = data
        self.redactor = redactor or (Redactor(secrets) if secrets else NULL_REDACTOR)
        inner = RunEventSink(run_id, data, bus, api_base=api_base, redactor=self.redactor)
        #: The agent wraps its sink to intercept approval events; replays do
        #: not. Applying the wrapper here keeps the difference to one argument.
        self.sink = sink_wrapper(inner) if sink_wrapper else inner
        self._mark_started = mark_started
        self._terminal = Terminal(error="The run ended without producing an outcome.")
        self._started = 0.0
        self._finished = False

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._started) * 1000)

    def finish(self, terminal: Terminal) -> None:
        """Record how this run ended. The last call before exit wins."""
        self._terminal = terminal
        self._finished = True

    async def __aenter__(self) -> "RunLifecycle":
        bind_run_id(self.run_id)
        self._started = time.monotonic()
        if self._mark_started:
            await self.data.mark_started(self.run_id)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        cancelled = exc_type is asyncio.CancelledError
        if cancelled and not self._finished:
            self._terminal = Terminal(status="cancelled", duration_ms=self.elapsed_ms)

        try:
            # Shielded: a cancelled run must still persist its terminal event.
            # Everything downstream -- the dashboard socket, the batch runner,
            # the history list -- waits on this and has no other way to learn
            # the run is over.
            await asyncio.shield(self._persist())
        except Exception:  # noqa: BLE001 - finalising must not mask the real error
            log.exception("failed to finalise a run", extra={"run_id": self.run_id})
        finally:
            bind_run_id(None)

        # Never swallow: a cancellation has to keep propagating, and a genuine
        # error is the caller's to see.
        return False

    async def _persist(self) -> None:
        terminal = self._terminal
        await self.sink.emit(
            RunFinished(
                run_id=self.run_id,
                seq=self.sink.reserve_seq(),
                status=terminal.status,
                steps=terminal.steps,
                duration_ms=terminal.duration_ms,
                summary=terminal.summary,
                result=terminal.result,
                error=terminal.error,
            )
        )
        await self.data.finish_run(
            self.run_id,
            terminal.status,
            steps=terminal.steps,
            duration_ms=terminal.duration_ms,
            summary=terminal.summary,
            result=terminal.result,
            error=terminal.error,
        )
        log.info(
            "run finished",
            extra={
                "run_id": self.run_id,
                "status": terminal.status,
                "steps": terminal.steps,
                "duration_ms": terminal.duration_ms,
            },
        )


__all__ = ["RunLifecycle", "Terminal"]
