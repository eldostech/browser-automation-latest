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

log = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

@router.get("/healthz")
async def healthz(request: Request, deep: bool = Query(default=False)) -> JSONResponse:
    """Liveness plus dependency status.

    There is no browser probe any more. The MCP server was a separate process
    that could be down while everything else was up, so its reachability was
    worth caching and reporting; Playwright is a library in this process, and
    "can it launch a browser" is answered by launching one -- too heavy for a
    healthcheck loop and answered anyway by the first execution.

    ``?deep=1`` still checks the model can be called, because a model the
    account lacks otherwise shows up one step into a run, as a 403.
    """
    app_state = request.app.state
    store = app_state.store
    settings: Settings = app_state.settings

    db_ok = await store.ping()
    llm = llm_health(settings)

    # A deep probe checks the models can actually be called. A model the
    # account lacks otherwise only shows up one step into a run, as a 403.
    if deep:
        # One role left. The driver and the distiller went with the agent: a
        # workflow is recorded by watching someone do it, and a codegen script
        # is parsed rather than interpreted.
        check = await app_state.repair_model.client.check_access()
        llm = {**llm, "access": {"repair": check}}
        if not check["ok"]:
            llm = {**llm, "configured": False}

    queue = await app_state.queue.depth()
    healthy = db_ok and llm["configured"]
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
        "browser": {
            "engine": settings.browser_engine,
            "headless": settings.browser_headless,
            "recorder": app_state.recorder.available()[0],
        },
        "queue": {"queued": queue.get("queued", 0), "running": queue.get("running", 0)},
        "events": {"cross_process": getattr(app_state.bus, "connected", False)},
        "active_execution": app_state.replays.active,
    }
    return JSONResponse(body, status_code=200 if healthy else 503)


@router.get("/api/config")
async def get_config_endpoint(
    _: CurrentUser, settings: Annotated[Settings, Depends(get_config)]
) -> dict[str, Any]:
    """What the dashboard needs to know about this deployment. No secrets.

    The task composer these defaults were for is gone with the agent. What a
    client still asks is what kind of browser it will get, whether this
    deployment can record at all, and which model is behind the one thing that
    still costs tokens.
    """
    return {
        "defaults": {
            "browser": settings.browser_engine,
            "headless": settings.browser_headless,
            "trace": settings.browser_trace,
            "screenshots": settings.replay_screenshots,
            "healing": settings.replay_healing_enabled,
        },
        "environment": settings.environment,
        "recorder": {"enabled": settings.recorder_enabled},
        "model": settings.llm_repair_model,
        "provider": PROVIDER,
    }
