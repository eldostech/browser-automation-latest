"""Event fan-out to connected UIs, within a process and between processes.

:class:`EventBus` is the in-memory fan-out this project has always had: a set
of asyncio queues per run, one per open WebSocket.

:class:`PostgresEventBus` adds the part that a second worker needs. A run
executing on worker A must reach a browser tab whose WebSocket is held by
worker B, and an in-memory dict cannot do that.

**Why Postgres rather than Redis.** Redis would work, and is the conventional
answer. It is also a second piece of infrastructure to run, secure, monitor and
back up, for a fan-out whose entire volume is a few events per second to a
handful of tabs. ``LISTEN/NOTIFY`` is already present in the database this
application cannot run without, and it delivers exactly the semantics wanted
here: at-most-once, in-order per channel, and dropped entirely if nobody is
listening -- which is correct, because a UI with no tab open needs no events.

**The notification carries a pointer, not the payload.** Postgres caps a NOTIFY
payload at 8000 bytes and an event carrying an accessibility snapshot is far
larger, so the message is ``run_id:seq:origin`` and the receiving process reads
the event from the events table. That also makes the database the single source
of truth for what a client is shown, rather than a second copy racing the
first.

**Durability is not this layer's job.** Events are persisted before they are
published, and a reconnecting client replays from its last ``seq``. So a
notification lost in transit costs a few hundred milliseconds of staleness, not
data -- which is why at-most-once delivery is acceptable here and would not be
for the job queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from typing import Any, Callable, Awaitable

import asyncpg

from config import Settings
from events import AgentEvent

log = logging.getLogger(__name__)

#: One channel for every run. Postgres channel names are identifiers, so a
#: per-run channel would mean an unbounded set of them and a LISTEN storm as
#: tabs open and close; one channel plus a cheap filter is simpler and, at this
#: volume, indistinguishable in cost.
CHANNEL = "browser_run_events"


class EventBus:
    """In-memory fan-out, one process."""

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
        self._deliver_locally(run_id, event)

    def _deliver_locally(self, run_id: str, event: AgentEvent) -> None:
        for queue in list(self._subscribers.get(run_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A stalled client must not slow the agent down. It will
                # reconnect and replay from its last seq.
                log.warning("dropping event for slow subscriber", extra={"run_id": run_id})

    def subscriber_count(self, run_id: str) -> int:
        return len(self._subscribers.get(run_id, ()))

    def watched_runs(self) -> set[str]:
        return set(self._subscribers)

    # -- lifecycle, so callers can treat both buses the same -----------------
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class PostgresEventBus(EventBus):
    """Fan-out across processes, over LISTEN/NOTIFY.

    Falls back to behaving exactly like the in-memory bus if the listener
    connection cannot be established. That is deliberate: losing cross-process
    delivery degrades a multi-worker deployment to single-worker behaviour,
    which is worse but still works. Refusing to start the application because a
    UI-convenience channel is unavailable would be a poor trade.
    """

    def __init__(
        self,
        settings: Settings,
        fetch_event: Callable[[str, int], Awaitable[AgentEvent | None]],
        *,
        queue_size: int = 1000,
    ) -> None:
        super().__init__(queue_size=queue_size)
        self._settings = settings
        self._fetch_event = fetch_event
        #: Distinguishes our own notifications from other workers', so a local
        #: publish is not delivered twice.
        self._origin = uuid.uuid4().hex[:12]
        self._listener: asyncpg.Connection | None = None
        self._notify_tasks: set[asyncio.Task] = set()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def start(self) -> None:
        try:
            self._listener = await asyncpg.connect(
                host=self._settings.db_host,
                port=self._settings.db_port,
                user=self._settings.db_user,
                password=self._settings.db_password,
                database=self._settings.db_name,
            )
            await self._listener.add_listener(CHANNEL, self._on_notify)
            self._connected = True
            log.info("event bus listening", extra={"channel": CHANNEL, "origin": self._origin})
        except Exception:  # noqa: BLE001 - see the class docstring
            self._connected = False
            log.exception(
                "could not open the cross-process event channel; "
                "falling back to in-process fan-out"
            )

    async def stop(self) -> None:
        for task in list(self._notify_tasks):
            task.cancel()
        self._notify_tasks.clear()
        if self._listener is not None:
            with contextlib.suppress(Exception):
                await self._listener.remove_listener(CHANNEL, self._on_notify)
                await self._listener.close()
            self._listener = None
        self._connected = False

    def publish(self, run_id: str, event: AgentEvent) -> None:
        # Local subscribers get the object we already hold -- no round trip,
        # and no dependence on the notification arriving.
        self._deliver_locally(run_id, event)
        if self._connected:
            task = asyncio.create_task(self._notify(run_id, event.seq))
            self._notify_tasks.add(task)
            task.add_done_callback(self._notify_tasks.discard)

    async def _notify(self, run_id: str, seq: int) -> None:
        payload = json.dumps({"run_id": run_id, "seq": seq, "origin": self._origin})
        try:
            # A separate short-lived connection rather than the listener: a
            # connection in LISTEN mode may not be used for other statements
            # while a notification is being dispatched on it.
            conn = await asyncpg.connect(
                host=self._settings.db_host,
                port=self._settings.db_port,
                user=self._settings.db_user,
                password=self._settings.db_password,
                database=self._settings.db_name,
            )
            try:
                await conn.execute(f"NOTIFY {CHANNEL}, $1", payload)
            finally:
                await conn.close()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a missed notification costs staleness, not data
            log.debug("event notification failed", extra={"run_id": run_id, "seq": seq})

    def _on_notify(self, _conn, _pid, _channel, payload: str) -> None:  # noqa: ANN001 - asyncpg hook
        try:
            message = json.loads(payload)
        except (TypeError, ValueError):
            return
        if message.get("origin") == self._origin:
            return  # our own publish, already delivered locally
        run_id = message.get("run_id")
        seq = message.get("seq")
        if not run_id or seq is None or not self._subscribers.get(run_id):
            return  # nobody here is watching this run

        task = asyncio.create_task(self._deliver_remote(run_id, int(seq)))
        self._notify_tasks.add(task)
        task.add_done_callback(self._notify_tasks.discard)

    async def _deliver_remote(self, run_id: str, seq: int) -> None:
        try:
            event = await self._fetch_event(run_id, seq)
        except Exception:  # noqa: BLE001 - never break the listener
            log.debug("could not read a notified event", extra={"run_id": run_id, "seq": seq})
            return
        if event is not None:
            self._deliver_locally(run_id, event)


def build_bus(
    settings: Settings,
    fetch_event: Callable[[str, int], Awaitable[AgentEvent | None]] | None = None,
    *,
    cross_process: bool = True,
) -> EventBus:
    """The bus this deployment should use.

    Tests and single-process runs take the plain in-memory bus; there is no
    reason to open a listener connection for a fan-out that never leaves the
    process.
    """
    if cross_process and fetch_event is not None:
        return PostgresEventBus(settings, fetch_event)
    return EventBus()


__all__ = ["CHANNEL", "EventBus", "PostgresEventBus", "build_bus"]
