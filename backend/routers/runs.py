"""Recording runs: start, watch, approve, cancel, and the live stream."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from urllib.parse import quote
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
from fastapi.responses import FileResponse, RedirectResponse, Response

from auth.rbac import Permission
from auth.service import Principal
from deps import (
    WorkspaceData,
    get_replays,
    get_store,
    principal_from_websocket,
    require,
    run_or_404,
)
from events import TERMINAL_STATUSES, dump_event
from fields import FieldSet
from imagediff import describe
from storage import S3_SCHEME, StorageError
from runner import ReplayManager

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["runs"])

#: How much of a page has to change before it is worth pointing at. Below this
#: is a clock, a cursor, or an animation; above it something moved.
DIVERGENCE = 0.02

#: Non-event transport frame used to keep idle proxies from closing the socket.
#: Clients ignore any message whose ``type`` starts with ``__``.
HEARTBEAT = {"type": "__heartbeat__"}
HEARTBEAT_INTERVAL = 20.0

Replays = Annotated[ReplayManager, Depends(get_replays)]


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
    replays: Replays,
    _: Annotated[Principal, Depends(require(Permission.RUN_READ))],
) -> dict[str, Any]:
    run = await run_or_404(run_id, data)
    return {
        **run.to_dict(),
        "active": replays.is_active(run_id),
        "artifacts": [
            {"id": a.id, "kind": a.kind, "mime": a.mime, "url": f"/api/artifacts/{a.id}"}
            for a in await data.list_artifacts(run_id)
        ],
    }


@router.get("/runs/{run_id}/steps")
async def get_run_steps(
    run_id: str,
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.RUN_READ))],
) -> dict[str, Any]:
    """Every step of this run, with its screenshot and the baseline's.

    The event log can answer this too, by scanning JSON and joining nothing.
    This is the projection written alongside it, which is what makes the
    side-by-side comparison a read rather than an image operation per render.

    ``pixel_diff`` is the fraction of pixels that changed. ``null`` means there
    was no baseline -- a first run, or a step that has never succeeded before
    -- which is a different thing from ``0.0`` meaning nothing moved.
    """
    await run_or_404(run_id, data)
    steps = await data.list_run_steps(run_id)
    for step in steps:
        for key, field in (("screenshot_url", "screenshot_id"), ("baseline_url", "baseline_id")):
            artifact = step.get(field)
            step[key] = f"/api/artifacts/{artifact}" if artifact else None
        step["diff"] = describe(step.get("pixel_diff"))

    # The first step that visibly diverged, so the UI can open there instead of
    # asking somebody to scroll a thousand rows looking for it.
    diverged = next(
        (
            s["seq"]
            for s in steps
            if s["status"] == "failed" or (s.get("pixel_diff") or 0) >= DIVERGENCE
        ),
        None,
    )
    return {"run_id": run_id, "steps": steps, "first_divergence": diverged}


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
    replays: Replays,
    _: Annotated[Principal, Depends(require(Permission.RUN_CANCEL))],
) -> dict[str, Any]:
    run = await run_or_404(run_id, data)
    if run.status in TERMINAL_STATUSES:
        return {"run_id": run_id, "cancelled": False, "reason": f"run already {run.status}"}

    if not await replays.cancel_run(run_id):
        raise HTTPException(status_code=409, detail="That run is not active on this backend.")
    return {"run_id": run_id, "cancelled": True}



def _artifact_headers(record) -> dict[str, str]:
    """Caching, plus the download's own name when it has one.

    A screenshot is rendered inline and needs no name -- it is identified by
    the step it belongs to. A downloaded document does: the file is the
    deliverable, and saving it as an opaque id is how a migration ends up with
    a bucket nobody can join to anything. The name also travels to whatever
    system it gets uploaded into next.
    """
    headers = {"Cache-Control": "private, max-age=31536000, immutable"}
    name = (getattr(record, "filename", "") or "").strip()
    if not name:
        return headers

    # A quote or a newline here would let a stored filename forge extra header
    # content, so neither survives. The RFC 5987 form carries anything
    # non-ASCII; the plain one is the fallback for older clients.
    strip = str.maketrans({chr(92): '_', chr(34): '_', chr(13): '', chr(10): ''})
    safe = name.translate(strip)
    ascii_name = safe.encode("ascii", "replace").decode("ascii")
    headers["Content-Disposition"] = (
        f'attachment; filename="{ascii_name}"; '
        f"filename*=UTF-8''{quote(safe, safe='')}"
    )
    return headers


@router.get("/artifacts/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.RUN_READ))],
):
    """Serve a screenshot from whichever backend holds it.

    Object storage gets a redirect to a short-lived presigned URL, so the bytes
    travel from S3 to the browser rather than through this process. A local
    file is streamed from disk.

    Caching is `private` because an artifact is workspace-scoped: a shared
    proxy must not hold a copy that it could hand to a different tenant.
    """
    record = await data.get_artifact(artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No such artifact.")

    direct = data.artifact_url(record)
    if direct:
        return RedirectResponse(direct, status_code=307)

    if record.path.startswith(S3_SCHEME):
        # Object storage that could not presign: stream it rather than fail.
        try:
            payload = await data.read_artifact(record)
        except StorageError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return Response(
            payload,
            media_type=record.mime,
            headers=_artifact_headers(record),
        )

    if not Path(record.path).exists():
        # The row outlived the file -- an artifacts directory cleared by hand,
        # or a backend switch that left the old files behind.
        raise HTTPException(
            status_code=404,
            detail="That artifact is recorded but its file is missing from storage.",
        )
    return FileResponse(
        record.path,
        media_type=record.mime,
        headers=_artifact_headers(record),
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
