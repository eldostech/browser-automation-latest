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
from batch import results_csv, validate_rows
from ingest import BatchInputError, Dataset, rows_from_json
from mapping import apply_mapping
from store import WorkspaceStore
from config import Settings
from deps import (
    WorkspaceData,
    batch_or_404,
    get_config,
    get_replays,
    get_vault,
    require,
)
from credentials import Vault
from routers.schemas import BatchRequestBody, ExecuteRequest
from runner import BatchRequest, ExecutionRequest, ReplayManager
from services import (
    load_runnable_usecase,
    require_missing_nothing,
    require_scripts_permitted,
    resolve_secrets,
    rows_from_body,
)
from usecase import TargetMissing, UseCase

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
        result = await replays.execute_once(
            ExecutionRequest(
                usecase=use_case,
                version=version,
                inputs=values,
                secrets=secrets,
                base_url=body.base_url,
                headless=body.headless,
                browser=body.browser,
                workspace_id=principal.workspace_id,
                owner_id=principal.user_id,
                owner_email=principal.email,
            )
        )
    except TargetMissing as exc:
        # A named target this deployment has no address for. Said plainly to
        # whoever pressed the button; guessing at one is how a workflow ends
        # up run against the wrong site.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Batches were audited from the start; a single row was not, so the most
    # common way to run a use case left no trace of who did it. The inputs go
    # in the entry because "what was it run with" is half the question -- and
    # they are safe to record: a declared secret is a credential slot, never
    # an input.
    await data.audit(
        "usecase.execute",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="usecase",
        resource_id=usecase_id,
        detail={
            "execution_id": result.get("execution_id"),
            "run_id": result.get("run_id"),
            "status": result.get("status"),
            "version": version,
            "inputs": values,
            "credential_id": body.credential_id,
        },
    )
    return result


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


async def rows_for_batch(body: BatchRequestBody, data: WorkspaceStore) -> Dataset:
    """The rows this batch will run, however the caller named them.

    A stored dataset is read here rather than in ``services.rows_from_body``
    because it needs the request scope, and reaching the store from a function
    that otherwise only parses bytes would make it much harder to see that one
    tenant cannot read another tenant's file.

    The mapping is applied at this point -- before validation, before the batch
    row is written, before a browser exists -- so everything downstream works in
    declared field names and never has to know what the spreadsheet called its
    columns.
    """
    if body.dataset_id is None:
        return rows_from_body(body)

    stored = await data.get_dataset(body.dataset_id, sample=0)
    if stored is None:
        raise HTTPException(status_code=404, detail="No such dataset.")
    rows = await data.get_dataset_rows(body.dataset_id) or []

    if body.mapping:
        known = {column["name"] for column in stored["columns"]}
        unknown = sorted(set(body.mapping.values()) - known)
        if unknown:
            raise BatchInputError(
                "the mapping names column(s) that are not in this dataset: "
                + ", ".join(unknown)
            )
        rows = apply_mapping(rows, body.mapping)

    if not rows:
        raise BatchInputError("that dataset has no rows")
    return rows_from_json(rows)


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
        parsed = await rows_for_batch(body, data)
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

    if body.secrets:
        # A batch is claimed by a worker that may be another process, and the
        # only thing it is given is the credential id. Inline values would have
        # to be written into the job payload to survive that hop, and writing a
        # password into a table a batch listing reads is precisely what the
        # vault exists to avoid. See stash.py for the same argument at the
        # other end of the recording flow.
        raise HTTPException(
            status_code=422,
            detail=(
                "A batch cannot take inline secrets. Save the login as a credential "
                "and pass its credential_id instead."
            ),
        )

    secrets = await resolve_secrets(
        ExecuteRequest(credential_id=body.credential_id, secrets=None), data, vault
    )
    require_missing_nothing(use_case, secrets)

    try:
        batch_id = await replays.start_batch(
            BatchRequest(
                usecase=use_case,
                version=version,
                rows=parsed.rows,
                secrets=secrets,
                base_url=body.base_url,
                credential_id=body.credential_id,
                dataset_id=body.dataset_id,
                headless=body.headless,
                browser=body.browser,
                workspace_id=principal.workspace_id,
                owner_id=principal.user_id,
                owner_email=principal.email,
            )
        )
    except TargetMissing as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    await data.audit(
        "batch.start",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="batch",
        resource_id=batch_id,
        detail={
            "usecase_id": usecase_id,
            "rows": len(parsed),
            "dataset_id": body.dataset_id,
        },
    )
    return {
        "batch_id": batch_id,
        "total": len(parsed),
        "columns": parsed.column_names,
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

    # The rows the batch was queued with, not a reconstruction from the
    # executions it managed to create. Rebuilding from `executions.inputs`
    # yielded an empty row for anything never attempted -- which is exactly the
    # set a resume exists to run -- so a resume after the circuit breaker
    # tripped used to replay blanks.
    rows = await data.get_batch_rows(batch_id)
    if rows is None:
        raise HTTPException(status_code=404, detail="No such batch.")
    if not rows:
        raise HTTPException(
            status_code=409,
            detail=(
                "This batch was queued before its rows were stored with it, so there is "
                "nothing to resume from. Start it again from the input file."
            ),
        )

    outstanding = await replays.pending_row_indices(
        batch_id, principal.workspace_id, total=len(rows)
    )
    if not outstanding:
        raise HTTPException(status_code=409, detail="Every row in this batch already succeeded.")

    use_case, version, _ = await load_runnable_usecase(
        batch["usecase_id"], batch["version"], data
    )
    await require_scripts_permitted(batch["usecase_id"], use_case, data)

    if body.secrets:
        raise HTTPException(
            status_code=422,
            detail=(
                "A batch cannot take inline secrets. Save the login as a credential "
                "and pass its credential_id instead."
            ),
        )

    secrets = await resolve_secrets(
        ExecuteRequest(
            credential_id=body.credential_id or batch.get("credential_id"),
            secrets=None,
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
                # A resume goes back to the address the original ran against unless
                # the caller deliberately names another. Re-resolving would let a
                # target edited in between move the remaining rows to a different
                # deployment from the ones already done.
                base_url=body.base_url or str(batch.get("base_url") or ""),
                only_rows=outstanding,
                headless=body.headless,
                browser=body.browser,
                workspace_id=principal.workspace_id,
                owner_id=principal.user_id,
                owner_email=principal.email,
            )
        )
    except TargetMissing as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"batch_id": new_id, "resumed_from": batch_id, "rows": len(outstanding)}


@router.post("/batches/{batch_id}/cancel")
async def cancel_batch(
    batch_id: str,
    data: WorkspaceData,
    replays: Replays,
    principal: Annotated[Principal, Depends(require(Permission.RUN_CANCEL))],
) -> dict[str, Any]:
    """Stop a batch, queued or in flight.

    Ownership is established first now. It used to be checked only after "is
    this running?", so that a batch id which did not exist and one that existed
    but was idle gave the same answer -- when cancelling meant taking the
    in-process slot, that was the whole story. A queued batch lives in a table
    a tenant either can or cannot see, so the scoped lookup is both the 404 and
    the tenancy check, and it has to come first.

    A running batch stops after the row in flight finishes. A queued one is
    cancelled in the queue and never starts.
    """
    await batch_or_404(batch_id, data)
    cancelled = await replays.cancel_batch(batch_id, principal.workspace_id)
    if not cancelled:
        raise HTTPException(
            status_code=409, detail="That batch is not running and is not queued."
        )
    return {"batch_id": batch_id, "cancelled": True}


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


@router.get("/usecases/{usecase_id}/estimate")
async def estimate(
    usecase_id: str,
    data: WorkspaceData,
    settings: Annotated[Settings, Depends(get_config)],
    _: Annotated[Principal, Depends(require(Permission.BATCH_READ))],
    rows: int = Query(default=1, ge=1, le=1_000_000),
) -> dict[str, Any]:
    """What running this many rows would cost, before anybody commits to it.

    A limit is what stops a mistake; an estimate is what prevents one. They are
    different jobs, and this is the cheaper of the two -- an agent loop over a
    spreadsheet is the easiest way to spend a lot of money here, and the number
    that matters is visible before the button rather than after the bill.
    """
    from estimates import estimate_batch

    definition = await data.get_usecase(usecase_id)
    if definition is None:
        raise HTTPException(status_code=404, detail="No such use case.")
    use_case = UseCase.model_validate(definition)
    spend = await data.spend_this_month()

    return estimate_batch(
        use_case,
        rows,
        model=settings.llm_repair_model,
        healing_enabled=settings.replay_healing_enabled,
        remaining_usd=spend.get("remaining_usd"),
    ).as_dict()
