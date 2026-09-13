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
    #: What this run spent. On the terminal rather than only in ``result`` so
    #: it lands in columns that can be summed: "what has this workspace spent
    #: this month" is a question with a JSONB answer otherwise, which is a
    #: question nobody asks twice.
    tokens: int = 0
    cost_usd: float = 0.0
    #: True when the run already emitted its own `run_finished` through this
    #: same sink before this block exited. The agent graph does exactly that
    #: -- `author.py`'s `finish` node announces the session itself, since
    #: `run_agent_session` is also used standalone with no lifecycle wrapping
    #: it at all. Persisting here must not announce it a second time with a
    #: status computed by a different rule; every other run type leaves this
    #: False and gets the one announcement this class exists to guarantee.
    announced: bool = False


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
        flush_interval: float | None = None,
        max_batch: int | None = None,
    ) -> None:
        from runner import RunEventSink  # circular at module scope; fine here

        self.run_id = run_id
        self.data = data
        self.redactor = redactor or (Redactor(secrets) if secrets else NULL_REDACTOR)
        from eventbuffer import DEFAULT_INTERVAL, DEFAULT_MAX_BATCH

        inner = RunEventSink(
            run_id,
            data,
            bus,
            api_base=api_base,
            redactor=self.redactor,
            flush_interval=DEFAULT_INTERVAL if flush_interval is None else flush_interval,
            max_batch=DEFAULT_MAX_BATCH if max_batch is None else max_batch,
        )
        #: Kept beside the wrapped one so the buffered writer can be flushed
        #: and closed on the way out. A wrapper intercepts `emit`; it has no
        #: reason to know that writes are batched, and no reason to forward a
        #: method it does not use.
        self._inner = inner
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
        if not terminal.announced:
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
        # Before the run row says the run is over, and that order is the point.
        # Events are written in batches now, so the terminal event and whatever
        # else is still buffered have to reach the database before anything
        # tells a client to stop watching -- otherwise a dashboard sees
        # "succeeded" and a catch-up read that is missing the last few events.
        await self._close_sink()

        await self.data.finish_run(
            self.run_id,
            terminal.status,
            steps=terminal.steps,
            duration_ms=terminal.duration_ms,
            summary=terminal.summary,
            result=terminal.result,
            error=terminal.error,
            tokens=terminal.tokens,
            cost_usd=terminal.cost_usd,
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


    async def _close_sink(self) -> None:
        """Flush and stop the buffered writer. Never raises into finalisation."""
        close = getattr(self._inner, "aclose", None)
        if close is None:
            return
        try:
            await close()
        except Exception:  # noqa: BLE001 - finalising must not mask the real error
            log.exception("failed to flush a run's events", extra={"run_id": self.run_id})


__all__ = ["RunLifecycle", "Terminal"]
