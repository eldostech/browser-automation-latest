"""Executing a stored use case: one row, or a file of them. **No LLM calls.**

Everything here runs against a use case that was distilled once. The cost of a
thousand rows is a thousand browser sessions and zero tokens, which is the
entire point of the feature.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

from auth.rbac import Permission
from auth.service import Principal
from batch import BatchInputError, results_csv, validate_rows
from deps import WorkspaceData, batch_or_404, get_replays, get_vault, require
from credentials import Vault
from routers.schemas import BatchRequestBody, ExecuteRequest
from runner import BatchRequest, ExecutionBusy, ExecutionRequest, ReplayManager
from services import (
    load_runnable_usecase,
    require_missing_nothing,
    require_scripts_permitted,
    resolve_secrets,
    rows_from_body,
)
from usecase import UseCase

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["executions"])

Replays = Annotated[ReplayManager, Depends(get_replays)]
VaultDep = Annotated[Vault, Depends(get_vault)]


@router.get("/executions/active")
async def active_execution(
    replays: Replays, _: Annotated[Principal, Depends(require(Permission.BATCH_READ))]
) -> dict[str, Any]:
    """What holds the execution slot, if anything."""
    return {"active": replays.active}


@router.post("/usecases/{usecase_id}/execute", status_code=201)
async def execute_usecase(
    usecase_id: str,
    body: ExecuteRequest,
    data: WorkspaceData,
    vault: VaultDep,
    replays: Replays,
    principal: Annotated[Principal, Depends(require(Permission.BATCH_CREATE))],
) -> dict[str, Any]:
    """Run one input row against a stored use case. **No LLM call is made.**"""
    use_case, version, _ = await load_runnable_usecase(usecase_id, body.version, data)
    await require_scripts_permitted(usecase_id, use_case, data)

    secrets = await resolve_secrets(body, data, vault)
    values = use_case.with_defaults(body.inputs)
    require_missing_nothing(use_case, secrets, values)

    try:
        return await replays.execute_once(
            ExecutionRequest(
                usecase=use_case,
                version=version,
                inputs=values,
                secrets=secrets,
                headless=body.headless,
                browser=body.browser,
                workspace_id=principal.workspace_id,
                owner_id=principal.user_id,
            )
        )
    except ExecutionBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/usecases/{usecase_id}/executions")
async def list_usecase_executions(
    usecase_id: str,
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.BATCH_READ))],
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    return {"executions": await data.list_executions(usecase_id=usecase_id, limit=limit)}


# ---------------------------------------------------------------------------
# Batches: one use case over many rows, one shared session
# ---------------------------------------------------------------------------


@router.post("/usecases/{usecase_id}/batch", status_code=202)
async def start_batch(
    usecase_id: str,
    body: BatchRequestBody,
    data: WorkspaceData,
    vault: VaultDep,
    replays: Replays,
    principal: Annotated[Principal, Depends(require(Permission.BATCH_CREATE))],
) -> dict[str, Any]:
    """Run a use case over a file of input rows. **No LLM call is made.**

    Every row is validated against the input schema before a browser opens, so
    a bad column fails in a millisecond rather than on record 700.
    """
    use_case, version, _ = await load_runnable_usecase(usecase_id, body.version, data)
    await require_scripts_permitted(usecase_id, use_case, data)

    try:
        parsed = rows_from_body(body)
    except BatchInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    problems = validate_rows(use_case, parsed)
    if problems:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "The input file does not match this use case.",
                "problems": problems,
            },
        )

    secrets = await resolve_secrets(
        ExecuteRequest(credential_id=body.credential_id, secrets=body.secrets), data, vault
    )
    require_missing_nothing(use_case, secrets)

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
                workspace_id=principal.workspace_id,
                owner_id=principal.user_id,
            )
        )
    except ExecutionBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    await data.audit(
        "batch.start",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="batch",
        resource_id=batch_id,
        detail={"usecase_id": usecase_id, "rows": len(parsed)},
    )
    return {
        "batch_id": batch_id,
        "total": len(parsed),
        "columns": parsed.columns,
        "warnings": parsed.warnings,
    }


@router.get("/batches/{batch_id}")
async def get_batch(
    batch_id: str,
    data: WorkspaceData,
    replays: Replays,
    _: Annotated[Principal, Depends(require(Permission.BATCH_READ))],
) -> dict[str, Any]:
    batch = await batch_or_404(batch_id, data)
    executions = await data.list_executions(batch_id=batch_id)
    active = replays.active
    return {
        "batch": batch,
        "executions": executions,
        "running": bool(active and active.get("batch_id") == batch_id),
        "pending": sum(1 for row in executions if row["status"] == "pending"),
    }


@router.post("/batches/{batch_id}/resume", status_code=202)
async def resume_batch(
    batch_id: str,
    body: BatchRequestBody,
    data: WorkspaceData,
    vault: VaultDep,
    replays: Replays,
    principal: Annotated[Principal, Depends(require(Permission.BATCH_CREATE))],
) -> dict[str, Any]:
    """Re-run only the rows that are not ``succeeded``.

    Covers all three ways a batch ends early -- re-login failure, the circuit
    breaker, and a process restart -- identically.
    """
    batch = await batch_or_404(batch_id, data)

    outstanding = await replays.pending_row_indices(batch_id, principal.workspace_id)
    if not outstanding:
        raise HTTPException(status_code=409, detail="Every row in this batch already succeeded.")

    use_case, version, _ = await load_runnable_usecase(
        batch["usecase_id"], batch["version"], data
    )
    await require_scripts_permitted(batch["usecase_id"], use_case, data)

    executions = await data.list_executions(batch_id=batch_id)
    by_index = {
        int(row["row_index"]): row["inputs"]
        for row in executions
        if row["row_index"] is not None
    }
    highest = max(by_index) if by_index else -1
    rows = [by_index.get(index, {}) for index in range(highest + 1)]

    secrets = await resolve_secrets(
        ExecuteRequest(
            credential_id=body.credential_id or batch.get("credential_id"),
            secrets=body.secrets,
        ),
        data,
        vault,
    )
    require_missing_nothing(use_case, secrets)

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
                workspace_id=principal.workspace_id,
                owner_id=principal.user_id,
            )
        )
    except ExecutionBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {"batch_id": new_id, "resumed_from": batch_id, "rows": len(outstanding)}


@router.post("/batches/{batch_id}/cancel")
async def cancel_batch(
    batch_id: str,
    data: WorkspaceData,
    replays: Replays,
    _: Annotated[Principal, Depends(require(Permission.RUN_CANCEL))],
) -> dict[str, Any]:
    """Stop after the row in flight finishes.

    The order of these two checks is deliberate. "Not running" is answered
    first, so a batch id that does not exist gets the same 409 as one that
    exists but is idle -- cancelling is about the execution slot, not about the
    record. Ownership is checked only once we know the batch *is* running,
    which is the point at which the answer could otherwise leak: without it,
    one tenant could stop another's batch by guessing its id.
    """
    active = replays.active
    if not active or active.get("batch_id") != batch_id:
        raise HTTPException(status_code=409, detail="That batch is not running.")
    await batch_or_404(batch_id, data)
    return {"batch_id": batch_id, "cancelled": await replays.cancel_active()}


@router.get("/batches/{batch_id}/results.csv")
async def batch_results_csv(
    batch_id: str,
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.BATCH_READ))],
) -> PlainTextResponse:
    """One row out per row in, in a stable column order so files diff cleanly."""
    batch = await batch_or_404(batch_id, data)

    definition = await data.get_usecase(batch["usecase_id"], batch["version"])
    if definition is None:
        raise HTTPException(status_code=404, detail="The use case this batch ran is gone.")

    body = results_csv(
        UseCase.model_validate(definition), await data.list_executions(batch_id=batch_id)
    )
    return PlainTextResponse(
        body,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="batch-{batch_id[:8]}.csv"'},
    )


@router.get("/usecases/{usecase_id}/batches")
async def list_usecase_batches(
    usecase_id: str,
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.BATCH_READ))],
) -> dict[str, Any]:
    return {"batches": await data.list_batches(usecase_id=usecase_id)}
