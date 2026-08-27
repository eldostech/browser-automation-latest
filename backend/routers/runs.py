"""Recording runs: start, watch, approve, cancel, and the live stream."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse

from agent import RunOptions
from auth.rbac import Permission
from auth.service import Principal
from deps import (
    WorkspaceData,
    get_manager,
    get_store,
    principal_from_websocket,
    require,
    run_or_404,
)
from events import TERMINAL_STATUSES, dump_event
from fields import FieldSet
from routers.schemas import ApprovalRequest, CreateRunRequest, CreateRunResponse
from runner import RunManager, RunRequest

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["runs"])

#: Non-event transport frame used to keep idle proxies from closing the socket.
#: Clients ignore any message whose ``type`` starts with ``__``.
HEARTBEAT = {"type": "__heartbeat__"}
HEARTBEAT_INTERVAL = 20.0

Manager = Annotated[RunManager, Depends(get_manager)]


@router.post("/runs", response_model=CreateRunResponse, status_code=201)
async def create_run(
    body: CreateRunRequest,
    request: Request,
    manager: Manager,
    principal: Annotated[Principal, Depends(require(Permission.RUN_CREATE))],
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

    try:
        fields = FieldSet.from_payload([f.model_dump() for f in body.fields])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    run_id = await manager.start_run(
        RunRequest(
            task=body.task,
            start_url=body.start_url,
            options=options,
            headless=body.headless,
            browser=body.browser,
            secrets=list(body.secrets or []),
            fields=fields,
            workspace_id=principal.workspace_id,
            owner_id=principal.user_id,
        )
    )

    # Held in memory only, and only until the user decides whether to keep
    # them. Nothing about this reaches disk. See stash.py.
    if fields.secret_values:
        request.app.state.stash.put(
            run_id, fields.secret_values, workspace_id=principal.workspace_id
        )

    return CreateRunResponse(run_id=run_id, status="pending")


@router.get("/runs")
async def list_runs(
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.RUN_READ))],
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    runs = await data.list_runs(status=status, limit=limit, offset=offset)
    return {
        "runs": [run.to_dict() for run in runs],
        "total": await data.count_runs(status),
        "limit": limit,
        "offset": offset,
    }


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    data: WorkspaceData,
    manager: Manager,
    _: Annotated[Principal, Depends(require(Permission.RUN_READ))],
) -> dict[str, Any]:
    run = await run_or_404(run_id, data)
    return {
        **run.to_dict(),
        "active": manager.is_active(run_id),
        "pending_approval": manager.pending_approval(run_id),
        "artifacts": [
            {"id": a.id, "kind": a.kind, "mime": a.mime, "url": f"/api/artifacts/{a.id}"}
            for a in await data.list_artifacts(run_id)
        ],
    }


@router.get("/runs/{run_id}/credential-slots")
async def held_credential_slots(
    run_id: str,
    request: Request,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.RUN_READ))],
) -> dict[str, Any]:
    """Which credential slots this recording still has values for.

    Names only -- there is no endpoint that returns a held value. The UI asks
    this when offering "save these credentials with the use case", so that it
    only offers when there is something to save and can say which slots.

    An empty list is the normal answer for a run that used no credentials, and
    also for one whose values have expired. The two are deliberately not
    distinguished: either way there is nothing to save.
    """
    await run_or_404(run_id, data)
    return {
        "run_id": run_id,
        "slots": request.app.state.stash.slots(run_id, workspace_id=principal.workspace_id),
    }


@router.get("/runs/{run_id}/events")
async def get_run_events(
    run_id: str,
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.RUN_READ))],
    after_seq: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Full event history. Used to replay a finished run without a WebSocket."""
    await run_or_404(run_id, data)
    events = await data.get_events(run_id, after_seq=after_seq)
    return {"run_id": run_id, "events": [dump_event(e) for e in events]}


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    data: WorkspaceData,
    manager: Manager,
    _: Annotated[Principal, Depends(require(Permission.RUN_CANCEL))],
) -> dict[str, Any]:
    run = await run_or_404(run_id, data)
    if run.status in TERMINAL_STATUSES:
        return {"run_id": run_id, "cancelled": False, "reason": f"run already {run.status}"}

    if not await manager.cancel_run(run_id):
        raise HTTPException(status_code=409, detail="That run is not active on this backend.")
    return {"run_id": run_id, "cancelled": True}


@router.post("/runs/{run_id}/approve")
async def approve_action(
    run_id: str,
    body: ApprovalRequest,
    data: WorkspaceData,
    manager: Manager,
    principal: Annotated[Principal, Depends(require(Permission.RUN_APPROVE))],
) -> dict[str, Any]:
    """Let a paused run proceed, or refuse it.

    The approval is recorded against the person who gave it. Before this, the
    only trace an approval left was a boolean saying one had happened.
    """
    await run_or_404(run_id, data)

    decision = "approved" if body.decision == "approve" else "rejected"
    if not manager.resolve_approval(run_id, body.approval_id, decision, body.note):
        raise HTTPException(
            status_code=409,
            detail=(
                "No approval is pending for this run "
                "(it may have timed out or already been resolved)."
            ),
        )

    await data.audit(
        f"run.{decision}",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="run",
        resource_id=run_id,
        detail={"approval_id": body.approval_id, "note": body.note},
    )
    return {"run_id": run_id, "decision": decision, "by": principal.email}


@router.get("/artifacts/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.RUN_READ))],
) -> FileResponse:
    record = await data.get_artifact(artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No such artifact.")
    return FileResponse(
        record.path,
        media_type=record.mime,
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


# ---------------------------------------------------------------------------
# Live stream
# ---------------------------------------------------------------------------


@router.websocket("/runs/{run_id}/stream")
async def stream_run(
    websocket: WebSocket, run_id: str, after_seq: int = Query(default=0)
) -> None:
    """Replay everything after ``after_seq``, then stream live events.

    The client reconnects with the highest ``seq`` it has seen, which makes
    reconnection lossless without any server-side session state.

    Authentication happens before the socket is accepted. A browser cannot set
    an Authorization header on a WebSocket handshake, so the token arrives as a
    query parameter -- see ``deps.principal_from_websocket`` for the trade-off.
    """
    principal = await principal_from_websocket(websocket, websocket.query_params.get("token"))
    if principal is None:
        # 1008 is "policy violation". Closing before accept() would give the
        # browser no code at all, so accept and then close with a reason the
        # client can actually distinguish from a network failure.
        await websocket.accept()
        await websocket.close(code=1008, reason="authentication required")
        return
    if not principal.can(Permission.RUN_READ):
        await websocket.accept()
        await websocket.close(code=1008, reason="not permitted")
        return

    store = websocket.app.state.store
    bus = websocket.app.state.bus
    data = store.workspace(principal.workspace_id)

    await websocket.accept()
    if await data.get_run(run_id) is None:
        await websocket.close(code=4404, reason="run not found")
        return

    # Subscribe before reading history so nothing produced during the replay is
    # missed; duplicates are filtered by seq below.
    queue = bus.subscribe(run_id)
    last_seq = after_seq
    finished = False

    try:
        for event in await data.get_events(run_id, after_seq=after_seq):
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
