"""Uploaded files of input rows, and how their columns line up with a use case.

Two things live here because they are two halves of one step the user takes:
"here is my spreadsheet" and "yes, that column is the email address".

**Why a dataset outlives the request that uploaded it.** Mapping is a
conversation -- upload, look at what is in the file, agree the columns, then
run -- and that is three round trips over the same rows. Without a resource the
browser has to hold the file and post it each time, which also means the
mapping suggestion is computed against something the server has not seen.

**Uploads are read before they are stored.** ``read_table`` raises on anything
it cannot parse, so a file that would fail on row 700 fails here instead, in a
millisecond, with a message about the file rather than about a browser.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from auth.rbac import Permission
from auth.service import Principal
from deps import WorkspaceData, require
from ingest import BatchInputError, ColumnProfile, read_table
from mapping import suggest, unresolved
from services import load_runnable_usecase
from routers.schemas import MappingRequest

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["datasets"])

#: A cap on what one upload may hold. Not a security boundary -- the reverse
#: proxy is -- but the point past which a JSONB column and a browser preview
#: both stop being the right shape, and failing here with a sentence beats
#: failing later with a timeout.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ROWS = 50_000


@router.post("/datasets", status_code=201)
async def upload_dataset(
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.BATCH_CREATE))],
    file: UploadFile = File(...),
    name: str = Form(""),
) -> dict[str, Any]:
    """Parse an uploaded CSV, spreadsheet or text file and keep it."""
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"that file is {len(raw) // (1024 * 1024)}MB; the limit is "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB. Split it, or run it in parts."
            ),
        )

    try:
        dataset = read_table(raw, file.filename or "upload.csv")
    except BatchInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if len(dataset) > MAX_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"that file has {len(dataset)} rows; the limit is {MAX_ROWS}.",
        )

    dataset_id = uuid.uuid4().hex
    await data.create_dataset(
        dataset_id,
        name=(name or file.filename or "Untitled").strip()[:200],
        filename=(file.filename or "")[:400],
        source=dataset.source,
        rows=dataset.rows,
        columns=[column.to_dict() for column in dataset.columns],
        warnings=dataset.warnings,
        owner_id=principal.user_id,
        owner_email=principal.email,
    )
    await data.audit(
        "dataset.upload",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="dataset",
        resource_id=dataset_id,
        detail={"filename": file.filename, "rows": len(dataset)},
    )
    log.info(
        "dataset uploaded",
        extra={"dataset_id": dataset_id, "rows": len(dataset), "source": dataset.source},
    )
    return {"dataset_id": dataset_id, **(await data.get_dataset(dataset_id))}


@router.get("/datasets")
async def list_datasets(
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.BATCH_READ))],
) -> dict[str, Any]:
    return {"datasets": await data.list_datasets()}


@router.get("/datasets/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.BATCH_READ))],
    sample: int = 20,
) -> dict[str, Any]:
    dataset = await data.get_dataset(dataset_id, sample=max(0, min(sample, 200)))
    if dataset is None:
        raise HTTPException(status_code=404, detail="No such dataset.")
    return dataset


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.BATCH_CREATE))],
) -> dict[str, Any]:
    if not await data.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="No such dataset.")
    await data.audit(
        "dataset.delete",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="dataset",
        resource_id=dataset_id,
    )
    # Batches that ran from it keep their own copy of the rows, so deleting a
    # dataset does not rewrite history.
    return {"deleted": dataset_id}


@router.post("/usecases/{usecase_id}/mapping")
async def suggest_mapping(
    usecase_id: str,
    body: MappingRequest,
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.BATCH_CREATE))],
) -> dict[str, Any]:
    """Propose a column for each declared input.

    Nothing here commits: the response is a ranking with reasons, and the user
    confirms it. An automatic mapping that is wrong and unreviewed does not
    fail -- it succeeds a thousand times into the wrong fields.

    ``unresolved`` names the subset a model could usefully improve. It is
    reported rather than acted on, so the cost of asking is a decision the
    caller makes with the ambiguity in front of them.
    """
    use_case, _version, _definition = await load_runnable_usecase(usecase_id, body.version, data)

    dataset = await data.get_dataset(body.dataset_id, sample=0)
    if dataset is None:
        raise HTTPException(status_code=404, detail="No such dataset.")

    columns = [ColumnProfile.from_dict(column) for column in dataset["columns"]]
    suggestions = suggest(((spec.name, spec.type) for spec in use_case.inputs), columns)

    return {
        "usecase_id": usecase_id,
        "dataset_id": body.dataset_id,
        "suggestions": [suggestion.to_dict() for suggestion in suggestions],
        "unresolved": [suggestion.field for suggestion in unresolved(suggestions)],
        "columns": [column.name for column in columns],
    }
