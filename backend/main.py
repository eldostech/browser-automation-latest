"""FastAPI application assembly.

This module wires things together and does nothing else. The endpoints live in
``routers/``, the logic between HTTP and storage in ``services.py``, and the
dependency graph in ``deps.py``.

The LLM credentials never leave this process and never appear in a URL. The
frontend sends a task; the backend decides what the browser does.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.service import AuthService
from bus import build_bus
from config import Settings, get_settings
from credentials import Vault
from llm import RepairModel
from jobs import JobQueue, Worker
from logging_setup import configure_logging
from agent.manager import AgentSessions
from recorder import Recorder
from routers import ALL_ROUTERS
from storage import build_storage
from runner import ReplayManager
from store import Store

log = logging.getLogger(__name__)


async def _fetch_event_for_bus(app: FastAPI, run_id: str, seq: int):
    """Read one event by ``(run_id, seq)`` for the cross-process bus.

    The bus is handed this rather than a Store because it must not care which
    workspace a run belongs to: it is delivering to a subscriber who has
    already been authorised for that run.
    """
    store: Store = app.state.store
    workspace_id = await store.default_workspace_id()
    if workspace_id is None:
        return None
    events = await store.workspace(workspace_id).get_events(run_id, after_seq=seq - 1, limit=1)
    return events[0] if events else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    configure_logging(
        settings.log_level,
        log_dir=settings.log_path if settings.log_to_file else None,
        file_name=settings.log_file_name,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
    )
    log.info(
        "starting backend",
        extra={
            "browser": settings.browser_engine,
            "headless": settings.browser_headless,
            "db_schema": settings.db_schema,
        },
    )

    store = Store(settings)
    await store.connect()

    app.state.settings = settings
    app.state.store = store
    app.state.auth = AuthService(store.sessions, settings)
    app.state.queue = JobQueue(store.sessions)
    app.state.storage = build_storage(settings)
    log.info(
        "artifact storage ready",
        extra={
            "backend": app.state.storage.name,
            "location": settings.s3_bucket if settings.storage_backend == "s3"
            else str(settings.artifacts_path),
        },
    )

    # A deployment must have an administrator to be reachable at all.
    generated = await app.state.auth.bootstrap()
    if generated:
        # Printed once, and only on the boot that created the account. There is
        # no way to retrieve it afterwards -- only bcrypt output is stored.
        log.warning(
            "=" * 72
            + f"\nCreated the first administrator: {settings.bootstrap_admin_email}"
            + f"\nOne-time password: {generated}"
            + "\nSign in and change it. This will not be shown again.\n"
            + "=" * 72
        )

    reaped = await store.reap_orphaned_runs()
    if reaped:
        log.warning("marked interrupted runs as failed", extra={"count": reaped})
    # Expired sessions are already refused by ``resolve``; this only stops the
    # table growing without bound. Startup is the whole schedule -- a process
    # that never restarts is a problem this sweep would not fix anyway.
    expired = await app.state.auth.purge_expired_sessions()
    if expired:
        log.info("purged expired sessions", extra={"count": expired})
    reclaimed = await app.state.queue.reclaim_expired()
    if reclaimed:
        log.warning("requeued jobs from a stopped worker", extra={"count": reclaimed})

    app.state.bus = build_bus(
        settings,
        lambda run_id, seq: _fetch_event_for_bus(app, run_id, seq),
    )
    await app.state.bus.start()

    app.state.vault = Vault(settings.credentials_key or None)
    # One model, built on first use, shared by the two things that can spend a
    # token: repairing a locator mid-replay, and driving an authoring session.
    # Sharing it keeps Bedrock configured in one place, which is the reason
    # there is only one provider at all.
    app.state.repair_model = RepairModel(settings)

    # The healer is the only route from a replay to a model, and it is handed
    # over lazily and only when healing is switched on.
    app.state.replays = ReplayManager(
        store,
        settings,
        bus=app.state.bus,
        llm_factory=app.state.repair_model,
        queue=app.state.queue,
        vault=app.state.vault,
    )

    # The worker that actually runs queued batches. In this process by default,
    # which is what makes a single-machine install work with nothing else
    # started; set WORKER_ENABLED=false on an API pod that should only serve
    # HTTP and leave the batches to a worker Deployment.
    app.state.worker = None
    if settings.worker_enabled:
        app.state.worker = Worker(
            app.state.queue,
            {"batch": app.state.replays.run_batch_job},
            poll_interval=settings.worker_poll_seconds,
            workspace_concurrency=settings.worker_workspace_concurrency,
        )
        await app.state.worker.start()
        log.info(
            "job worker started",
            extra={
                "worker_id": app.state.queue.worker_id,
                "workspace_concurrency": settings.worker_workspace_concurrency,
            },
        )
    if not app.state.vault.available:
        log.warning(
            "credential storage is disabled: CREDENTIALS_KEY is not set. "
            "Use cases that need a login cannot be executed until it is."
        )
    # Recording is a person in front of a browser window, so the sessions live
    # in this process and are closed with it. See recorder.py.
    app.state.recorder = Recorder(
        enabled=settings.recorder_enabled,
        command=settings.recorder_command,
        browser=settings.recorder_browser,
        timeout_seconds=settings.recorder_timeout_seconds,
    )
    ok, reason = app.state.recorder.available()
    log.info("recorder", extra={"available": ok, "reason": reason or None})

    # The third role, beside the recorder and the worker. Sessions live in this
    # process for the same reason recordings do -- a browser is open and a
    # person is watching it -- and the events they produce are persisted as
    # they happen, so the transcript survives even when the live session does
    # not.
    app.state.agent_sessions = AgentSessions(
        store, app.state.bus, settings, app.state.repair_model
    )
    log.info(
        "agent",
        extra={"enabled": settings.agent_enabled, "provider": settings.agent_browser_provider},
    )

    # Said at startup rather than when the first run fails. On Windows only a
    # ProactorEventLoop can spawn the Playwright driver, and uvicorn picks the
    # other one whenever --reload or --workers is set -- so the usual
    # development command is the one that cannot replay. See browser.py.
    if sys.platform == "win32" and not isinstance(
        asyncio.get_running_loop(), asyncio.ProactorEventLoop
    ):
        log.warning(
            "this event loop cannot start a browser, so replaying will fail. "
            "uvicorn picks it whenever --reload or --workers is set; add "
            "--loop none to the command.",
            extra={"loop": type(asyncio.get_running_loop()).__name__},
        )

    try:
        yield
    finally:
        # Stopped first: a job still in flight releases its lease on the way
        # out and is picked straight back up, whereas one whose store has
        # already closed fails for a reason that has nothing to do with it.
        if app.state.worker is not None:
            await app.state.worker.stop()
        # Before the rest: a headed browser that outlives its backend is a
        # window nobody owns, very possibly signed into something.
        await app.state.recorder.shutdown()
        # Same reason, one layer along: a Node subprocess and the Chromium
        # behind it must not outlive the API that started them.
        await app.state.agent_sessions.shutdown()
        await app.state.bus.stop()
        await store.close()
        log.info("backend stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    A factory rather than a module-level singleton, so a test can construct an
    app against its own database and its own configuration without touching
    the process environment. This is the same reason ``config`` no longer
    exposes a module-level ``settings`` object.
    """
    settings = settings or get_settings()

    application = FastAPI(
        title="TRACE",
        version="1.0.0",
        description=(
            "Record a browser workflow by doing it once, then replay it over a "
            "spreadsheet without an LLM."
        ),
        lifespan=lifespan,
    )
    application.state.settings = settings

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        # Bearer tokens, not cookies: nothing is sent automatically by the
        # browser, so CSRF has no purchase here and credentials stay off.
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in ALL_ROUTERS:
        application.include_router(router)

    return application


app = create_app()


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    import uvicorn

    _settings = get_settings()
    configure_logging(
        _settings.log_level,
        log_dir=_settings.log_path if _settings.log_to_file else None,
        file_name=_settings.log_file_name,
    )
    uvicorn.run("main:app", host=_settings.host, port=_settings.port, reload=False)
