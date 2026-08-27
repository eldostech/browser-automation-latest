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
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.service import AuthService
from bus import build_bus
from checkpoints import Checkpointer
from config import Settings, get_settings
from credentials import Vault
from jobs import JobQueue
from logging_setup import configure_logging
from mcp_client import MCPConfig, probe
from routers import ALL_ROUTERS
from stash import SecretStash
from runner import ReplayManager, RunManager
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
    configure_logging(settings.log_level)
    log.info(
        "starting backend",
        extra={
            "models": settings.models_in_use,
            "mcp_transport": settings.mcp_transport,
            "allowed_domains": settings.agent_allowed_domains,
            "db_schema": settings.db_schema,
        },
    )

    store = Store(settings)
    await store.connect()

    app.state.settings = settings
    app.state.store = store
    app.state.auth = AuthService(store.sessions, settings)
    app.state.queue = JobQueue(store.sessions)
    # Credentials a recording used, held only until the user decides whether
    # to keep them. In memory, with a TTL, and never written down -- see
    # stash.py for what that costs and why it is the right trade.
    app.state.stash = SecretStash()

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
    reclaimed = await app.state.queue.reclaim_expired()
    if reclaimed:
        log.warning("requeued jobs from a stopped worker", extra={"count": reclaimed})

    app.state.bus = build_bus(
        settings,
        lambda run_id, seq: _fetch_event_for_bus(app, run_id, seq),
    )
    await app.state.bus.start()

    # Opened before the manager, which hands the saver to every graph it
    # compiles. Held for the life of the process.
    checkpointer = Checkpointer(settings)
    await checkpointer.__aenter__()
    app.state.checkpointer = checkpointer
    log.info("agent checkpointing", extra={"backend": checkpointer.backend})

    app.state.manager = RunManager(
        store, settings, bus=app.state.bus, checkpointer=checkpointer.saver
    )
    app.state.vault = Vault(settings.credentials_key or None)
    # The healer is the only route from a replay to a model, and it is handed
    # over lazily and only when healing is switched on.
    app.state.replays = ReplayManager(
        store, settings, bus=app.state.bus, llm_factory=lambda: app.state.manager.repair_llm
    )
    if not app.state.vault.available:
        log.warning(
            "credential storage is disabled: CREDENTIALS_KEY is not set. "
            "Use cases that need a login cannot be executed until it is."
        )
    app.state.health = {"checked_at": 0.0, "result": None}

    # Probe MCP once at startup so the tool list is visible in the logs and
    # /healthz can answer without spawning a browser on every request.
    app.state.startup_probe = asyncio.create_task(_startup_probe(app, settings))

    try:
        yield
    finally:
        app.state.startup_probe.cancel()
        await app.state.manager.shutdown()
        await app.state.bus.stop()
        await checkpointer.__aexit__(None, None, None)
        await store.close()
        log.info("backend stopped")


async def _startup_probe(app: FastAPI, settings: Settings) -> None:
    try:
        result = await probe(MCPConfig.from_settings(settings), timeout=90.0)
    except asyncio.CancelledError:
        return
    app.state.health = {"checked_at": time.time(), "result": result}
    if result.get("ok"):
        log.info(
            "MCP server reachable",
            extra={"tool_count": result.get("tool_count"), "tools": result.get("tools")},
        )
    else:
        log.error("MCP server unreachable at startup", extra={"error": result.get("error")})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    A factory rather than a module-level singleton, so a test can construct an
    app against its own database and its own configuration without touching
    the process environment. This is the same reason ``config`` no longer
    exposes a module-level ``settings`` object.
    """
    settings = settings or get_settings()

    application = FastAPI(
        title="Browser Agent",
        version="1.0.0",
        description=(
            "An LLM agent that drives a real browser through the Playwright MCP "
            "server, and replays what it learned without further LLM calls."
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
    configure_logging(_settings.log_level)
    uvicorn.run("main:app", host=_settings.host, port=_settings.port, reload=False)
