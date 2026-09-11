"""Writing a run's events in batches instead of one at a time.

Why this exists
---------------
A replayed step used to cost eleven remote round trips, measured rather than
guessed: four events at three round trips each (the pool's pre-ping, the INSERT
and the COMMIT), a step row, and a fresh Postgres connection per event for the
``NOTIFY``. With the database in the same rack as the browser that is a few
milliseconds and nobody notices. With it a datacenter away it is most of a
second per step, and a use case with twenty steps spends most of its life
waiting on a socket rather than on a page.

None of that latency is on the path of anything a person is waiting for, and
that is the whole argument. **An event is a record about work that has already
happened.** The browser does not need it written before the next click, the
row's result does not depend on it, and the live view does not read it -- the
WebSocket serves a subscriber from an in-memory queue (see ``bus.py``) and
touches the database only to catch up after a reconnect.

So the work is moved off the step: :meth:`EventBuffer.add` appends to a list
and returns, and a background task writes whatever has accumulated in one
statement.

What is *not* traded away
-------------------------
**Events are published only after they are durable.** The buffer writes the
batch first and hands it to the bus second, which is the ordering the
unbuffered sink already had, and it is the ordering the resume contract rests
on: ``seq`` is the token a reconnecting client sends, and a client must never
see a ``seq`` that a later catch-up read cannot return. Publishing eagerly and
persisting later would be faster still and would open exactly that gap -- a
reconnect landing in the window would skip the events in flight, permanently.

The cost of that choice is up to one flush interval of delay before an event
reaches the live view. At the default interval a person cannot see it, and
what they *can* see is every step arriving sooner because the step is no
longer waiting on the database.

**The flush interval is a bound on staleness, not on loss.** A run that ends,
crashes or is cancelled flushes on the way out; :class:`EventBuffer` is closed
from the same ``finally`` that closes the browser. What a hard kill can lose is
at most one interval of *narration* -- the run's own status and its results are
written by the lifecycle, not from here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Sequence

log = logging.getLogger(__name__)

#: How long a batch may wait before being written, in seconds.
#:
#: The number is a trade between round trips saved and how stale the live view
#: may be, and it is deliberately well under the threshold where a person reads
#: a delay as a stall. Lower it and the saving shrinks; raise it and a watcher
#: starts to notice the timeline arriving in lumps.
DEFAULT_INTERVAL = 0.2

#: How many events may accumulate before a flush happens regardless of the
#: interval. Bounds memory, and keeps one very chatty run from building a batch
#: so large that writing it becomes its own latency problem.
DEFAULT_MAX_BATCH = 200

Writer = Callable[[list[Any]], Awaitable[None]]
Publisher = Callable[[list[Any]], None]


class EventBuffer:
    """Collects events, writes them in batches, then publishes them.

    Not a queue with a consumer, deliberately. A queue would let the producer
    outrun the writer without bound; this holds a list, flushes it whole, and
    flushes early when it grows past ``max_batch``.
    """

    def __init__(
        self,
        write: Writer,
        publish: Publisher,
        *,
        interval: float = DEFAULT_INTERVAL,
        max_batch: int = DEFAULT_MAX_BATCH,
    ) -> None:
        self._write = write
        self._publish = publish
        self._interval = max(interval, 0.0)
        self._max_batch = max(max_batch, 1)
        self._pending: list[Any] = []
        self._task: asyncio.Task | None = None
        self._closed = False
        #: Held across a flush so two flushes cannot interleave and publish
        #: out of order -- the interval timer and an explicit `flush()` at a
        #: row boundary can otherwise arrive together.
        self._lock = asyncio.Lock()

    def add(self, event: Any) -> None:
        """Take an event. Does no I/O and never blocks the step that made it."""
        if self._closed:
            # A late event after close still has to reach somebody: this is
            # the teardown path, where the last few events of a run are
            # exactly the ones worth having.
            self._pending.append(event)
            return
        self._pending.append(event)
        self._ensure_running()
        if len(self._pending) >= self._max_batch:
            # Scheduled rather than awaited: `add` is called from the step, and
            # the whole point is that the step does not wait for the database.
            self._schedule_flush()

    @property
    def pending(self) -> int:
        return len(self._pending)

    async def flush(self) -> None:
        """Write and publish everything collected so far.

        Called on the interval, when a batch fills, at the end of each row and
        on the way out. Safe to call when there is nothing to do.
        """
        async with self._lock:
            batch, self._pending = self._pending, []
            if not batch:
                return
            try:
                await self._write(batch)
            except Exception as exc:  # noqa: BLE001 - never let logging kill a run
                log.error(
                    "failed to persist %d event(s)",
                    len(batch),
                    extra={"error": str(exc)},
                )
                # Published anyway. A watcher seeing the run is worth more than
                # consistency with a write that has already failed, and the
                # alternative is a live view that silently stops.
            self._publish(batch)
        self._stop_idle_timer()

    async def aclose(self) -> None:
        """Stop the timer and write what is left. Idempotent."""
        self._closed = True
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await self.flush()

    # -- internals ----------------------------------------------------------
    def _ensure_running(self) -> None:
        if self._task is None or self._task.done():
            try:
                self._task = asyncio.create_task(self._tick())
            except RuntimeError:
                # No running loop. Nothing is lost: `flush` is still awaited
                # at row boundaries and on close, which is where durability
                # actually matters.
                self._task = None

    def _schedule_flush(self) -> None:
        task = asyncio.create_task(self.flush())
        # Held only so the task is not garbage collected mid-flight, which is
        # how a fire-and-forget write disappears without a trace.
        self._inflight = getattr(self, "_inflight", set())
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    def _stop_idle_timer(self) -> None:
        """Cancel the interval timer when there is nothing left to flush.

        The timer exists to write what is waiting; with nothing waiting it is a
        sleeping task holding the buffer alive. `add` starts a new one, so this
        costs a task creation per burst of activity rather than one sleeping
        for the life of the process -- which is what a run finishing without
        anybody closing its sink used to leave behind.

        Never cancels the task it is *called from*: `_tick` flushes, and a
        flush that cancelled its own caller would raise into the timer rather
        than ending it.
        """
        task = self._task
        if task is None or task.done() or self._pending:
            return
        try:
            if asyncio.current_task() is task:
                return
        except RuntimeError:  # pragma: no cover - no running loop
            return
        task.cancel()
        self._task = None

    async def _tick(self) -> None:
        """Flush on the interval, and stop once there is nothing to flush.

        Self-limiting rather than running for the life of the sink: `add`
        restarts it. A timer that outlives its work is a task leak, and a run
        that finishes without anybody closing its sink -- a test, a code path
        that raised on the way to `aclose` -- would otherwise leave one
        sleeping forever.
        """
        while not self._closed:
            await asyncio.sleep(self._interval)
            try:
                await self.flush()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a flush failure is logged inside
                log.debug("event flush failed", exc_info=True)
            if not self._pending:
                return


class RowBuffer:
    """The same idea for step rows, flushed at row boundaries rather than on a timer.

    Step rows are not narration: they are what the finished timeline and the
    visual diff are built from, and nothing reads them while the row is still
    running. A row is therefore the natural unit -- collect a row's steps,
    write them in one statement when the row ends -- and there is no interval
    to tune because there is nobody waiting.
    """

    def __init__(self, write: Callable[[list[dict]], Awaitable[None]]) -> None:
        self._write = write
        self._pending: list[dict] = []

    def add(self, row: dict) -> None:
        self._pending.append(row)

    @property
    def pending(self) -> int:
        return len(self._pending)

    async def flush(self) -> None:
        batch, self._pending = self._pending, []
        if batch:
            await self._write(batch)


def coalesce(events: Sequence[Any]) -> tuple[int, int]:
    """The ``(first, last)`` seq of a batch, for one notification instead of N."""
    seqs = [event.seq for event in events]
    return (min(seqs), max(seqs)) if seqs else (0, 0)


__all__ = [
    "DEFAULT_INTERVAL",
    "DEFAULT_MAX_BATCH",
    "EventBuffer",
    "RowBuffer",
    "coalesce",
]
