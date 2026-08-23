"""FastAPI application: the only thing the frontend talks to.

The LLM API key never leaves this process and never appears in a URL. The
frontend sends a task; the backend decides what the browser does.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from agent import RunOptions
from config import Settings, settings
from events import TERMINAL_STATUSES, dump_event
from llm import llm_health
from logging_setup import configure_logging
from mcp_client import MCPConfig, probe
from runner import EventBus, RunManager, RunRequest
from store import Store

log = logging.getLogger(__name__)

#: Non-event transport frame used to keep idle proxies from closing the socket.
#: Clients ignore any message whose ``type`` starts with ``__``.
HEARTBEAT = {"type": "__heartbeat__"}
HEARTBEAT_INTERVAL = 20.0

#: How long a cached MCP connectivity result is considered fresh.
HEALTH_CACHE_TTL = 60.0


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CreateRunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=8000)
    start_url: str | None = None

    # Guardrail overrides; anything omitted falls back to the server defaults.
    max_steps: int | None = Field(default=None, ge=1, le=200)
    timeout_seconds: float | None = Field(default=None, ge=10, le=3600)
    allowed_domains: list[str] | None = None
    require_approval: bool | None = None
    screenshot_every_step: bool | None = None

    # Browser overrides.
    headless: bool | None = None
    browser: str | None = None

    #: Values to keep out of the event log, the database and the logs. Anything
    #: listed here is replaced with a placeholder wherever it appears -- in the
    #: task text, in a tool argument, in a tool result echoing it back, or in
    #: the model's own prose. Write-only: never returned by any endpoint.
    secrets: list[str] | None = None

    @field_validator("start_url")
    @classmethod
    def _validate_url(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError("start_url must begin with http:// or https://")
        return value

    @field_validator("allowed_domains")
    @classmethod
    def _clean_domains(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return [d.strip() for d in value if d and d.strip()]


class ApprovalRequest(BaseModel):
    decision: Literal["approve", "reject"]
    approval_id: str | None = None
    note: str | None = Field(default=None, max_length=1000)


class CreateRunResponse(BaseModel):
    run_id: str
    status: str


# ---------------------------------------------------------------------------
# Application wiring
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    log.info(
        "starting backend",
        extra={
            "llm_provider": settings.llm_provider,
            "model": settings.llm_model,
            "mcp_transport": settings.mcp_transport,
            "allowed_domains": settings.agent_allowed_domains,
        },
    )

    store = Store(settings.db_path, settings.artifacts_path)
    await store.connect()
    reaped = await store.reap_orphaned_runs()
    if reaped:
        log.warning("marked interrupted runs as failed", extra={"count": reaped})

    app.state.settings = settings
    app.state.store = store
    app.state.bus = EventBus()
    app.state.manager = RunManager(store, settings, bus=app.state.bus)
    app.state.health = {"checked_at": 0.0, "result": None}

    # Probe MCP once at startup so the tool list is visible in the logs and
    # /healthz can answer without spawning a browser on every request.
    app.state.startup_probe = asyncio.create_task(_startup_probe(app))

    try:
        yield
    finally:
        app.state.startup_probe.cancel()
        await app.state.manager.shutdown()
        await store.close()
        log.info("backend stopped")


async def _startup_probe(app: FastAPI) -> None:
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


app = FastAPI(
    title="Browser Agent",
    version="0.1.0",
    description="An LLM agent that drives a real browser through the Playwright MCP server.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- dependencies -----------------------------------------------------------


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_manager(request: Request) -> RunManager:
    return request.app.state.manager


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


# ---------------------------------------------------------------------------
# Health & config
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz(request: Request, deep: bool = Query(default=False)) -> JSONResponse:
    """Liveness plus MCP connectivity.

    The shallow check (default) reports the last known MCP state, refreshed at
    startup and after every deep probe. ``?deep=1`` forces a live connect,
    which spawns a real browser -- fine for a manual check, too heavy for a
    container healthcheck loop.
    """
    app_state = request.app.state
    store: Store = app_state.store
    cache = app_state.health

    fresh = (time.time() - cache["checked_at"]) < HEALTH_CACHE_TTL
    if deep or cache["result"] is None:
        result = await probe(MCPConfig.from_settings(app_state.settings), timeout=60.0)
        app_state.health = {"checked_at": time.time(), "result": result}
        cache = app_state.health
        fresh = True

    mcp_result = cache["result"] or {"ok": None, "error": "not probed yet"}
    db_ok = await store.ping()
    llm = llm_health(app_state.settings)

    healthy = db_ok and llm["configured"] and mcp_result.get("ok") is not False
    body = {
        "status": "ok" if healthy else "degraded",
        "database": {"ok": db_ok, "path": str(app_state.settings.db_path)},
        "llm": llm,
        "mcp": {
            **mcp_result,
            "checked_at": cache["checked_at"],
            "stale": not fresh,
            "command": MCPConfig.from_settings(app_state.settings).command_line()
            if app_state.settings.mcp_transport == "stdio"
            else app_state.settings.mcp_server_url,
        },
        "active_runs": sum(1 for _ in app_state.manager._tasks),  # noqa: SLF001
    }
    return JSONResponse(body, status_code=200 if healthy else 503)


@app.get("/api/config")
async def get_config(settings_dep: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    """Defaults the task composer pre-fills. Contains no secrets."""
    return {
        "defaults": {
            "max_steps": settings_dep.agent_max_steps,
            "timeout_seconds": settings_dep.agent_timeout_seconds,
            "allowed_domains": settings_dep.agent_allowed_domains,
            "require_approval": settings_dep.agent_require_approval,
            "screenshot_every_step": settings_dep.agent_screenshot_every_step,
            "headless": settings_dep.mcp_headless,
            "browser": settings_dep.mcp_browser,
        },
        "model": settings_dep.llm_model,
        "provider": settings_dep.llm_provider,
        "transport": settings_dep.mcp_transport,
    }


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@app.post("/api/runs", response_model=CreateRunResponse, status_code=201)
async def create_run(
    body: CreateRunRequest,
    manager: RunManager = Depends(get_manager),
) -> CreateRunResponse:
    options: RunOptions = manager.default_options()
    if body.max_steps is not None:
        options.max_steps = body.max_steps
    if body.timeout_seconds is not None:
        options.timeout_seconds = body.timeout_seconds
    if body.allowed_domains is not None:
        options.allowed_domains = body.allowed_domains
    if body.require_approval is not None:
        options.require_approval = body.require_approval
    if body.screenshot_every_step is not None:
        options.screenshot_every_step = body.screenshot_every_step

    run_id = await manager.start_run(
        RunRequest(
            task=body.task,
            start_url=body.start_url,
            options=options,
            headless=body.headless,
            browser=body.browser,
            secrets=list(body.secrets or []),
        )
    )
    return CreateRunResponse(run_id=run_id, status="pending")


@app.get("/api/runs")
async def list_runs(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    store: Store = Depends(get_store),
) -> dict[str, Any]:
    runs = await store.list_runs(status=status, limit=limit, offset=offset)
    return {
        "runs": [run.to_dict() for run in runs],
        "total": await store.count_runs(status),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/runs/{run_id}")
async def get_run(
    run_id: str,
    store: Store = Depends(get_store),
    manager: RunManager = Depends(get_manager),
) -> dict[str, Any]:
    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        **run.to_dict(),
        "active": manager.is_active(run_id),
        "pending_approval": manager.pending_approval(run_id),
        "artifacts": [
            {"id": a.id, "kind": a.kind, "mime": a.mime, "url": f"/api/artifacts/{a.id}"}
            for a in await store.list_artifacts(run_id)
        ],
    }


@app.get("/api/runs/{run_id}/events")
async def get_run_events(
    run_id: str,
    after_seq: int = Query(default=0, ge=0),
    store: Store = Depends(get_store),
) -> dict[str, Any]:
    """Full event history. Used to replay a finished run without a WebSocket."""
    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    events = await store.get_events(run_id, after_seq=after_seq)
    return {"run_id": run_id, "events": [dump_event(e) for e in events]}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    store: Store = Depends(get_store),
    manager: RunManager = Depends(get_manager),
) -> dict[str, Any]:
    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    if run.status in TERMINAL_STATUSES:
        return {"run_id": run_id, "cancelled": False, "reason": f"run already {run.status}"}

    cancelled = await manager.cancel_run(run_id)
    if not cancelled:
        raise HTTPException(status_code=409, detail="run is not active on this backend")
    return {"run_id": run_id, "cancelled": True}


@app.post("/api/runs/{run_id}/approve")
async def approve_action(
    run_id: str,
    body: ApprovalRequest,
    store: Store = Depends(get_store),
    manager: RunManager = Depends(get_manager),
) -> dict[str, Any]:
    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    decision = "approved" if body.decision == "approve" else "rejected"
    resolved = manager.resolve_approval(run_id, body.approval_id, decision, body.note)
    if not resolved:
        raise HTTPException(
            status_code=409,
            detail="no approval is pending for this run (it may have timed out or been resolved)",
        )
    return {"run_id": run_id, "decision": decision}


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


@app.get("/api/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str, store: Store = Depends(get_store)) -> FileResponse:
    record = await store.get_artifact(artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(
        record.path,
        media_type=record.mime,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# ---------------------------------------------------------------------------
# Live stream
# ---------------------------------------------------------------------------


@app.websocket("/api/runs/{run_id}/stream")
async def stream_run(websocket: WebSocket, run_id: str, after_seq: int = Query(default=0)) -> None:
    """Replay everything after ``after_seq``, then stream live events.

    The client reconnects with the highest ``seq`` it has seen, which makes
    reconnection lossless without any server-side session state.
    """
    store: Store = websocket.app.state.store
    bus: EventBus = websocket.app.state.bus

    await websocket.accept()
    run = await store.get_run(run_id)
    if run is None:
        await websocket.close(code=4404, reason="run not found")
        return

    # Subscribe before reading history so nothing produced during the replay is
    # missed; duplicates are filtered by seq below.
    queue = bus.subscribe(run_id)
    last_seq = after_seq
    finished = False

    try:
        for event in await store.get_events(run_id, after_seq=after_seq):
            await websocket.send_json(dump_event(event))
            last_seq = max(last_seq, event.seq)
            if event.type == "run_finished":
                finished = True

        if finished:
            await websocket.close(code=1000, reason="run already finished")
            return

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_INTERVAL)
            except asyncio.TimeoutError:
                await websocket.send_json(HEARTBEAT)
                continue

            if event.seq <= last_seq:
                continue
            await websocket.send_json(dump_event(event))
            last_seq = event.seq
            if event.type == "run_finished":
                await websocket.close(code=1000, reason="run finished")
                return

    except WebSocketDisconnect:
        log.debug("websocket client disconnected", extra={"run_id": run_id})
    except Exception as exc:  # noqa: BLE001 - never leave the socket half-open
        log.warning("websocket stream error", extra={"run_id": run_id, "error": str(exc)})
        try:
            await websocket.close(code=1011, reason="stream error")
        except Exception:  # noqa: BLE001 - socket may already be gone
            pass
    finally:
        bus.unsubscribe(run_id, queue)


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    import uvicorn

    configure_logging(settings.log_level)
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)
