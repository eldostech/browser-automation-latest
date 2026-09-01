"""A batch worker with no HTTP server attached.

``main.py`` runs one of these inside the API process by default, which is what
makes a single-machine install work with nothing else started. This module is
the same worker on its own, for the deployment where the two are separated:

* on EKS, the API is a Deployment behind an ALB and this is a second Deployment
  that KEDA scales on queue depth;
* locally, ``make worker`` beside ``make backend`` is how you watch a batch run
  in a browser while still using the dashboard.

Set ``WORKER_ENABLED=false`` on the API when you run this, or both will claim
work -- which is safe (the queue hands each job to exactly one claimant) but
means the API process opens browsers you did not expect it to.

**No LLM client is constructed here**, and none is passed to
``ReplayManager``, so a worker started this way cannot heal: it has no
``llm_factory``, and ``make_healer`` returns None without one. That is
deliberate. Healing spends money, and a background process that quietly starts
doing so because it inherited a model client is the failure this project has
been structured to make impossible.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from bus import build_bus
from config import Settings, get_settings
from credentials import Vault
from jobs import JobQueue, Worker
from logging_setup import configure_logging
from runner import ReplayManager
from store import Store

log = logging.getLogger(__name__)


async def _fetch_event(store: Store, run_id: str, seq: int):
    """Read one event for the cross-process bus.

    The same shape as the API's own fetcher: the bus is delivering to a
    subscriber that was already authorised for this run, so it must not care
    which workspace the run belongs to.
    """
    workspace_id = await store.default_workspace_id()
    if workspace_id is None:
        return None
    events = await store.workspace(workspace_id).get_events(run_id, after_seq=seq - 1, limit=1)
    return events[0] if events else None


async def run(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    configure_logging(
        settings.log_level,
        to_file=settings.log_to_file,
        log_path=settings.log_path / settings.log_file_name,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
    )

    store = Store(settings)
    await store.connect()
    queue = JobQueue(store.sessions)

    # Cross-process: a batch running here has to reach a dashboard tab whose
    # WebSocket is held by the API process, and an in-memory fan-out cannot do
    # that.
    bus = build_bus(settings, lambda run_id, seq: _fetch_event(store, run_id, seq))
    await bus.start()

    replays = ReplayManager(
        store,
        settings,
        bus=bus,
        queue=queue,
        vault=Vault(settings.credentials_key or None),
    )

    reclaimed = await queue.reclaim_expired()
    if reclaimed:
        log.warning("requeued jobs from a stopped worker", extra={"count": reclaimed})

    worker = Worker(
        queue,
        {"batch": replays.run_batch_job},
        poll_interval=settings.worker_poll_seconds,
        workspace_concurrency=settings.worker_workspace_concurrency,
    )
    await worker.start()
    log.info(
        "worker started",
        extra={
            "worker_id": queue.worker_id,
            "workspace_concurrency": settings.worker_workspace_concurrency,
        },
    )

    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, stopping.set)
        except NotImplementedError:
            # Windows has no signal handlers on the proactor loop. Ctrl+C still
            # raises KeyboardInterrupt out of asyncio.run, which the caller
            # below turns into the same orderly shutdown.
            pass

    try:
        await stopping.wait()
    finally:
        # A job still in flight releases its lease on the way out and is picked
        # straight back up, rather than waiting for the lease to expire.
        await worker.stop()
        await bus.stop()
        await store.close()
        log.info("worker stopped")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
