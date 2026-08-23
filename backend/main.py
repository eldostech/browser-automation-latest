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
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from agent import RunOptions
from batch import BatchInputError, parse_csv, results_csv, rows_from_json, validate_rows
from config import Settings, settings
from credentials import (
    NO_KEY_MESSAGE,
    Vault,
    VaultError,
    VaultUnavailable,
    new_credential_id,
)
from distill import DistillationError, distill
from events import TERMINAL_STATUSES, dump_event
from llm import llm_health
from logging_setup import configure_logging
from mcp_client import MCPConfig, probe
from runner import (
    BatchRequest,
    EventBus,
    ExecutionBusy,
    ExecutionRequest,
    ReplayManager,
    RunManager,
    RunRequest,
)
from store import Store
from usecase import UseCase

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
    app.state.vault = Vault(settings.credentials_key or None)
    # The healer is the only route from a replay to a model, and it is handed
    # over lazily and only when healing is switched on.
    app.state.replays = ReplayManager(
        store, settings, bus=app.state.bus, llm_factory=lambda: app.state.manager.llm
    )
    if not app.state.vault.available:
        log.warning(
            "credential storage is disabled: CREDENTIALS_KEY is not set. "
            "Use cases that need a login cannot be executed until it is."
        )
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


# ---------------------------------------------------------------------------
# Use cases
# ---------------------------------------------------------------------------


@app.post("/api/runs/{run_id}/distill", status_code=201)
async def distill_run(
    run_id: str,
    store: Store = Depends(get_store),
    manager: RunManager = Depends(get_manager),
) -> dict[str, Any]:
    """Promote a successful run into a reusable use case.

    This is the one LLM call in the whole replay feature. Everything the use
    case is later executed with costs nothing.
    """
    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    if run.status != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=(
                f"only a succeeded run can be recorded as a use case; this one is {run.status!r}. "
                "A failed run has no reliable sequence of working steps to learn from."
            ),
        )

    events = await store.get_events(run_id)
    try:
        use_case = await distill(events, task=run.task, llm=manager.llm, source_run_id=run_id)
    except DistillationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    definition = use_case.model_dump(mode="json", by_alias=True)
    usecase_id, version = await store.save_usecase(definition)

    log.info(
        "distilled a run into a use case",
        extra={"run_id": run_id, "usecase_id": usecase_id, "version": version},
    )
    return {
        "usecase_id": usecase_id,
        "version": version,
        "name": use_case.name,
        "status": use_case.status,
        "warnings": use_case.warnings,
        "setup_steps": len(use_case.setup_steps),
        "row_steps": len(use_case.row_steps),
        "inputs": [spec.name for spec in use_case.inputs],
        "secrets": [spec.name for spec in use_case.secrets],
        "blocked_scripts": use_case.blocked_scripts,
    }


@app.get("/api/usecases")
async def list_usecases(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    store: Store = Depends(get_store),
) -> dict[str, Any]:
    rows = await store.list_usecases(status=status, limit=limit, offset=offset)
    return {"usecases": rows, "limit": limit, "offset": offset}


@app.get("/api/usecases/{usecase_id}")
async def get_usecase(
    usecase_id: str,
    version: int | None = Query(default=None),
    store: Store = Depends(get_store),
) -> dict[str, Any]:
    definition = await store.get_usecase(usecase_id, version)
    if definition is None:
        raise HTTPException(status_code=404, detail="use case not found")
    return {
        "definition": definition,
        "versions": await store.list_usecase_versions(usecase_id),
    }


@app.put("/api/usecases/{usecase_id}", status_code=201)
async def update_usecase(
    usecase_id: str,
    body: dict[str, Any],
    store: Store = Depends(get_store),
) -> dict[str, Any]:
    """Save reviewer edits as a new version.

    Never rewrites the version in place: a batch already running is reading
    from a specific version and must not have it changed underneath it.
    """
    if await store.get_usecase(usecase_id) is None:
        raise HTTPException(status_code=404, detail="use case not found")

    body = {**body, "id": usecase_id}
    try:
        use_case = UseCase.model_validate(body)
    except Exception as exc:  # noqa: BLE001 - surfaced to the editing UI verbatim
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _, version = await store.save_usecase(use_case.model_dump(mode="json", by_alias=True))
    return {"usecase_id": usecase_id, "version": version, "status": use_case.status}


@app.post("/api/usecases/{usecase_id}/publish")
async def publish_usecase(
    usecase_id: str,
    store: Store = Depends(get_store),
) -> dict[str, Any]:
    """Move a reviewed draft to ``ready`` so it can be executed.

    Re-validates at ``ready``, which is where the stricter rules bite -- most
    notably that a use case carrying raw JavaScript cannot be published until
    someone has read the code and turned ``allow_scripts`` on.
    """
    definition = await store.get_usecase(usecase_id)
    if definition is None:
        raise HTTPException(status_code=404, detail="use case not found")

    try:
        UseCase.model_validate({**definition, "status": "ready"})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    await store.set_usecase_status(usecase_id, "ready")
    return {"usecase_id": usecase_id, "status": "ready"}


@app.delete("/api/usecases/{usecase_id}")
async def archive_usecase(
    usecase_id: str,
    store: Store = Depends(get_store),
) -> dict[str, Any]:
    """Archive rather than delete -- execution history references the id."""
    if not await store.delete_usecase(usecase_id):
        raise HTTPException(status_code=404, detail="use case not found")
    return {"usecase_id": usecase_id, "status": "archived"}


# ---------------------------------------------------------------------------
# Credentials (write-only)
# ---------------------------------------------------------------------------


class CredentialRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    #: ``{slot: value}`` matching the use case's declared secrets. Write-only:
    #: no endpoint returns these, and nothing in the dashboard needs them back.
    values: dict[str, str] = Field(min_length=1)


def get_vault(request: Request) -> Vault:
    return request.app.state.vault


def get_replays(request: Request) -> ReplayManager:
    return request.app.state.replays


@app.post("/api/credentials", status_code=201)
async def create_credential(
    body: CredentialRequest,
    store: Store = Depends(get_store),
    vault: Vault = Depends(get_vault),
) -> dict[str, Any]:
    if not vault.available:
        raise HTTPException(status_code=503, detail=NO_KEY_MESSAGE)
    try:
        ciphertext = vault.seal(body.values)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    credential_id = await store.save_credential(
        new_credential_id(), body.name, Vault.slots_of(body.values), ciphertext
    )
    log.info("stored a credential", extra={"credential_id": credential_id, "slots": len(body.values)})
    return {"id": credential_id, "name": body.name, "slots": Vault.slots_of(body.values)}


@app.get("/api/credentials")
async def list_credentials(
    store: Store = Depends(get_store), vault: Vault = Depends(get_vault)
) -> dict[str, Any]:
    """Names and slot lists. Never a value."""
    return {"credentials": await store.list_credentials(), "vault_available": vault.available}


@app.delete("/api/credentials/{credential_id}")
async def delete_credential(
    credential_id: str, store: Store = Depends(get_store)
) -> dict[str, Any]:
    if not await store.delete_credential(credential_id):
        raise HTTPException(status_code=404, detail="credential not found")
    return {"id": credential_id, "deleted": True}


# ---------------------------------------------------------------------------
# Executing a use case (zero LLM calls)
# ---------------------------------------------------------------------------


class ExecuteRequest(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)
    #: Bind stored credentials by id, or pass values inline for a one-off.
    credential_id: str | None = None
    secrets: dict[str, str] | None = None
    version: int | None = None
    headless: bool | None = None
    browser: str | None = None


async def _resolve_secrets(
    body: ExecuteRequest, store: Store, vault: Vault
) -> dict[str, str]:
    """Decrypt the bound credential, or take inline values for a one-off.

    Whatever comes back is registered with the run's redactor before anything
    is emitted, so a value cannot reach the event log even if a tool echoes it.
    """
    if body.credential_id:
        ciphertext = await store.get_credential_ciphertext(body.credential_id)
        if ciphertext is None:
            raise HTTPException(status_code=404, detail="credential not found")
        try:
            values = vault.open(ciphertext)
        except VaultUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except VaultError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        await store.touch_credential(body.credential_id)
        return {**values, **(body.secrets or {})}
    return dict(body.secrets or {})


@app.get("/api/executions/active")
async def active_execution(replays: ReplayManager = Depends(get_replays)) -> dict[str, Any]:
    """What holds the single execution slot, if anything."""
    return {"active": replays.active}


@app.post("/api/usecases/{usecase_id}/execute", status_code=201)
async def execute_usecase(
    usecase_id: str,
    body: ExecuteRequest,
    store: Store = Depends(get_store),
    vault: Vault = Depends(get_vault),
    replays: ReplayManager = Depends(get_replays),
) -> dict[str, Any]:
    """Run one input row against a stored use case. **No LLM call is made.**"""
    definition = await store.get_usecase(usecase_id, body.version)
    if definition is None:
        raise HTTPException(status_code=404, detail="use case not found")

    try:
        use_case = UseCase.model_validate(definition)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"stored use case is invalid: {exc}") from exc

    if use_case.status != "ready":
        raise HTTPException(
            status_code=409,
            detail=(
                f"this use case is {use_case.status!r}. Review it and publish it before "
                "running it -- a distilled recording is a best guess until a person has "
                "checked it."
            ),
        )

    secrets = await _resolve_secrets(body, store, vault)

    missing_secrets = use_case.missing_secrets(secrets)
    if missing_secrets:
        raise HTTPException(
            status_code=422,
            detail=f"missing required credential slot(s): {', '.join(missing_secrets)}",
        )

    values = use_case.with_defaults(body.inputs)
    missing_inputs = use_case.missing_inputs(values)
    if missing_inputs:
        raise HTTPException(
            status_code=422,
            detail=f"missing required input(s): {', '.join(missing_inputs)}",
        )

    try:
        return await replays.execute_once(
            ExecutionRequest(
                usecase=use_case,
                version=int(definition.get("version") or 1),
                inputs=values,
                secrets=secrets,
                headless=body.headless,
                browser=body.browser,
            )
        )
    except ExecutionBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/usecases/{usecase_id}/executions")
async def list_usecase_executions(
    usecase_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    store: Store = Depends(get_store),
) -> dict[str, Any]:
    return {"executions": await store.list_executions(usecase_id=usecase_id, limit=limit)}


# ---------------------------------------------------------------------------
# Batches: one use case over many rows, one shared session
# ---------------------------------------------------------------------------


class BatchRequestBody(BaseModel):
    """Rows arrive either as raw CSV text or as a JSON array of objects."""

    csv: str | None = None
    rows: list[dict[str, Any]] | None = None
    credential_id: str | None = None
    secrets: dict[str, str] | None = None
    version: int | None = None
    headless: bool | None = None
    browser: str | None = None


async def _load_usecase_for_execution(
    usecase_id: str, version: int | None, store: Store
) -> tuple[UseCase, int]:
    definition = await store.get_usecase(usecase_id, version)
    if definition is None:
        raise HTTPException(status_code=404, detail="use case not found")
    try:
        use_case = UseCase.model_validate(definition)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"stored use case is invalid: {exc}") from exc
    if use_case.status != "ready":
        raise HTTPException(
            status_code=409,
            detail=(
                f"this use case is {use_case.status!r}. Review it and publish it before "
                "running it against a file."
            ),
        )
    return use_case, int(definition.get("version") or 1)


@app.post("/api/usecases/{usecase_id}/batch", status_code=202)
async def start_batch(
    usecase_id: str,
    body: BatchRequestBody,
    store: Store = Depends(get_store),
    vault: Vault = Depends(get_vault),
    replays: ReplayManager = Depends(get_replays),
) -> dict[str, Any]:
    """Run a use case over a file of input rows. **No LLM call is made.**

    Every row is validated against the input schema before a browser opens, so
    a bad column fails in a millisecond rather than on record 700.
    """
    use_case, version = await _load_usecase_for_execution(usecase_id, body.version, store)

    try:
        parsed = parse_csv(body.csv) if body.csv is not None else rows_from_json(body.rows)
    except BatchInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    problems = validate_rows(use_case, parsed)
    if problems:
        raise HTTPException(
            status_code=422,
            detail={"message": "the input file does not match this use case", "problems": problems},
        )

    secrets = await _resolve_secrets(
        ExecuteRequest(credential_id=body.credential_id, secrets=body.secrets), store, vault
    )
    missing = use_case.missing_secrets(secrets)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"missing required credential slot(s): {', '.join(missing)}",
        )

    try:
        batch_id = await replays.start_batch(
            BatchRequest(
                usecase=use_case,
                version=version,
                rows=parsed.rows,
                secrets=secrets,
                credential_id=body.credential_id,
                headless=body.headless,
                browser=body.browser,
            )
        )
    except ExecutionBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "batch_id": batch_id,
        "total": len(parsed),
        "columns": parsed.columns,
        "warnings": parsed.warnings,
    }


@app.get("/api/batches/{batch_id}")
async def get_batch(
    batch_id: str,
    store: Store = Depends(get_store),
    replays: ReplayManager = Depends(get_replays),
) -> dict[str, Any]:
    batch = await store.get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="batch not found")

    executions = await store.list_executions(batch_id=batch_id)
    active = replays.active
    return {
        "batch": batch,
        "executions": executions,
        "running": bool(active and active.get("batch_id") == batch_id),
        "pending": sum(1 for row in executions if row["status"] == "pending"),
    }


@app.post("/api/batches/{batch_id}/resume", status_code=202)
async def resume_batch(
    batch_id: str,
    body: BatchRequestBody,
    store: Store = Depends(get_store),
    vault: Vault = Depends(get_vault),
    replays: ReplayManager = Depends(get_replays),
) -> dict[str, Any]:
    """Re-run only the rows that are not ``succeeded``.

    Covers all three ways a batch ends early -- re-login failure, the circuit
    breaker, and a process restart -- identically.
    """
    batch = await store.get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="batch not found")

    outstanding = await replays.pending_row_indices(batch_id)
    if not outstanding:
        raise HTTPException(status_code=409, detail="every row in this batch already succeeded")

    use_case, version = await _load_usecase_for_execution(
        batch["usecase_id"], batch["version"], store
    )

    executions = await store.list_executions(batch_id=batch_id)
    by_index = {int(row["row_index"]): row["inputs"] for row in executions if row["row_index"] is not None}
    highest = max(by_index) if by_index else -1
    rows = [by_index.get(index, {}) for index in range(highest + 1)]

    secrets = await _resolve_secrets(
        ExecuteRequest(
            credential_id=body.credential_id or batch.get("credential_id"),
            secrets=body.secrets,
        ),
        store,
        vault,
    )
    missing = use_case.missing_secrets(secrets)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"missing required credential slot(s): {', '.join(missing)}",
        )

    try:
        new_id = await replays.start_batch(
            BatchRequest(
                usecase=use_case,
                version=version,
                rows=rows,
                secrets=secrets,
                credential_id=body.credential_id or batch.get("credential_id"),
                only_rows=outstanding,
                headless=body.headless,
                browser=body.browser,
            )
        )
    except ExecutionBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {"batch_id": new_id, "resumed_from": batch_id, "rows": len(outstanding)}


@app.post("/api/batches/{batch_id}/cancel")
async def cancel_batch(
    batch_id: str, replays: ReplayManager = Depends(get_replays)
) -> dict[str, Any]:
    """Stop after the row in flight finishes."""
    active = replays.active
    if not active or active.get("batch_id") != batch_id:
        raise HTTPException(status_code=409, detail="that batch is not running")
    return {"batch_id": batch_id, "cancelled": await replays.cancel_active()}


@app.get("/api/batches/{batch_id}/results.csv")
async def batch_results_csv(
    batch_id: str, store: Store = Depends(get_store)
) -> PlainTextResponse:
    """One row out per row in, in a stable column order so files diff cleanly."""
    batch = await store.get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="batch not found")

    definition = await store.get_usecase(batch["usecase_id"], batch["version"])
    if definition is None:
        raise HTTPException(status_code=404, detail="the use case this batch ran is gone")

    body = results_csv(
        UseCase.model_validate(definition), await store.list_executions(batch_id=batch_id)
    )
    return PlainTextResponse(
        body,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="batch-{batch_id[:8]}.csv"'},
    )


@app.get("/api/usecases/{usecase_id}/batches")
async def list_usecase_batches(
    usecase_id: str, store: Store = Depends(get_store)
) -> dict[str, Any]:
    return {"batches": await store.list_batches(usecase_id=usecase_id)}


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
