"""Health probes and the client-visible configuration.

``/healthz`` is the only endpoint besides login that does not require a token:
a load balancer cannot hold a session, and a probe that needs credentials is a
probe that reports "unhealthy" the moment authentication breaks -- which is
precisely when you need the rest of the signal to still be readable.

It reveals no data: counts, booleans, and the configured model names.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from config import Settings
from deps import CurrentUser, get_config
from llm import PROVIDER, llm_health
from mcp_client import MCPConfig, probe

log = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

#: How long a cached MCP connectivity result is considered fresh.
HEALTH_CACHE_TTL = 60.0


@router.get("/healthz")
async def healthz(request: Request, deep: bool = Query(default=False)) -> JSONResponse:
    """Liveness plus dependency status.

    The shallow check (default) reports the last known MCP state, refreshed at
    startup and after every deep probe. ``?deep=1`` forces a live connect,
    which spawns a real browser -- fine for a manual check, too heavy for a
    container healthcheck loop.
    """
    app_state = request.app.state
    store = app_state.store
    settings: Settings = app_state.settings
    cache = app_state.health

    fresh = (time.time() - cache["checked_at"]) < HEALTH_CACHE_TTL
    if deep or cache["result"] is None:
        result = await probe(MCPConfig.from_settings(settings), timeout=60.0)
        app_state.health = {"checked_at": time.time(), "result": result}
        cache = app_state.health
        fresh = True

    mcp_result = cache["result"] or {"ok": None, "error": "not probed yet"}
    db_ok = await store.ping()
    llm = llm_health(settings)

    # A deep probe checks the models can actually be called. A model the
    # account lacks otherwise only shows up one step into a run, as a 403.
    if deep:
        manager = app_state.manager
        roles = {
            "driver": manager.llm,
            "distiller": manager.distill_llm,
            "repair": manager.repair_llm,
        }
        checks = await asyncio.gather(*(client.check_access() for client in roles.values()))
        llm = {**llm, "access": dict(zip(roles, checks))}
        if any(not check["ok"] for check in checks):
            llm = {**llm, "configured": False}

    queue = await app_state.queue.depth()
    healthy = db_ok and llm["configured"] and mcp_result.get("ok") is not False
    body = {
        "status": "ok" if healthy else "degraded",
        "database": {
            "ok": db_ok,
            # Host and schema, never the credentials.
            "host": settings.db_host,
            "name": settings.db_name,
            "schema": settings.db_schema,
        },
        "llm": llm,
        "mcp": {
            **mcp_result,
            "checked_at": cache["checked_at"],
            "stale": not fresh,
            "command": MCPConfig.from_settings(settings).command_line()
            if settings.mcp_transport == "stdio"
            else settings.mcp_server_url,
        },
        "queue": {"queued": queue.get("queued", 0), "running": queue.get("running", 0)},
        "events": {"cross_process": getattr(app_state.bus, "connected", False)},
        "active_runs": len(app_state.manager._tasks),  # noqa: SLF001
    }
    return JSONResponse(body, status_code=200 if healthy else 503)


@router.get("/api/config")
async def get_config_endpoint(
    _: CurrentUser, settings: Annotated[Settings, Depends(get_config)]
) -> dict[str, Any]:
    """Defaults the task composer pre-fills. Contains no secrets."""
    return {
        "defaults": {
            "max_steps": settings.agent_max_steps,
            "timeout_seconds": settings.agent_timeout_seconds,
            "allowed_domains": settings.agent_allowed_domains,
            "require_approval": settings.agent_require_approval,
            "screenshot_every_step": settings.agent_screenshot_every_step,
            "headless": settings.mcp_headless,
            "browser": settings.mcp_browser,
        },
        "model": settings.llm_model,
        "models": settings.models_in_use,
        "provider": PROVIDER,
        "transport": settings.mcp_transport,
    }
