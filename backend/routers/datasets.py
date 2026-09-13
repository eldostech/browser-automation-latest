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
from ingest import rows_from_json, BatchInputError, ColumnProfile, read_table
from mapping import suggest, unresolved
from services import load_runnable_usecase
from routers.schemas import DatasetFromRunRequest, MappingRequest

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




async def _declared_column_order(
    data: Any, executions: list[dict[str, Any]], output: str
) -> list[str]:
    """The column names of the ``extract_rows`` step that produced this output.

    Best effort: a definition that has since been edited, or an output that
    came from somewhere else, simply gives no order and the rows keep whatever
    order they arrived in.
    """
    first = executions[0]
    definition = await data.get_usecase(first.get("usecase_id"), first.get("version"))
    if not definition:
        return []
    for phase in ("setup_steps", "row_steps", "teardown_steps"):
        for step in definition.get(phase) or []:
            if step.get("action") == "extract_rows" and step.get("output") == output:
                return [c.get("name", "") for c in (step.get("columns") or []) if c.get("name")]
    return []


@router.post("/datasets/from-run", status_code=201)
async def dataset_from_run(
    body: DatasetFromRunRequest,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.BATCH_CREATE))],
) -> dict[str, Any]:
    """Turn what a discovery run found into rows a second use case can run on.

    This is the join between the two passes of a migration. The first walks the
    vendor's list pages and reads identifiers out of them with ``extract_rows``;
    this makes those rows a dataset; the second runs once per row to pull the
    detail. Without it the first pass produces a list nobody can act on.

    It accepts a batch as well as a single execution, because discovery is
    often itself a batch -- one row per page of a paginated list -- and the
    rows of all of them are one list. They are concatenated in row order, which
    is the order the pages were walked.

    The rows are profiled on the way in, exactly as an uploaded spreadsheet is,
    so the column mapper works the same on both and a person can look at what
    was found before committing to four thousand detail runs.
    """
    if bool(body.execution_id) == bool(body.batch_id):
        raise HTTPException(
            status_code=422,
            detail="Name exactly one of execution_id or batch_id.",
        )

    if body.execution_id:
        executions = [
            e
            for e in await data.list_executions(usecase_id=None)
            if e.get("id") == body.execution_id
        ]
        if not executions:
            raise HTTPException(status_code=404, detail="No such execution.")
    else:
        executions = await data.list_executions(batch_id=body.batch_id)
        if not executions:
            raise HTTPException(status_code=404, detail="No such batch, or it ran no rows.")

    rows: list[dict[str, Any]] = []
    seen_output = False
    for execution in sorted(executions, key=lambda e: e.get("row_index") or 0):
        outputs = execution.get("outputs") or {}
        if body.output not in outputs:
            continue
        seen_output = True
        found = outputs[body.output]
        if isinstance(found, list):
            rows.extend(item for item in found if isinstance(item, dict))
        elif isinstance(found, dict):
            rows.append(found)

    if not seen_output:
        available = sorted(
            {k for e in executions for k in (e.get("outputs") or {})}
        )
        raise HTTPException(
            status_code=422,
            detail=(
                f"Nothing was extracted under {body.output!r}. "
                + (
                    f"This run produced: {', '.join(available)}."
                    if available
                    else "This run produced no outputs at all."
                )
            ),
        )
    if not rows:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{body.output!r} was extracted but held no rows, so there is nothing to "
                "run a second pass against. The list page may have been empty, or the "
                "row locator may have matched nothing."
            ),
        )
    if len(rows) > MAX_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"that run found {len(rows)} rows; the limit is {MAX_ROWS}.",
        )

    # Put the columns back in the order they were declared. JSONB does not
    # preserve key order -- Postgres sorts keys by length, so "name" comes out
    # before "account_id" -- and a person reading the dataset should see the
    # columns in the order they wrote them, not in an order the storage engine
    # chose. The step that produced them is the authority.
    declared = await _declared_column_order(data, executions, body.output)
    if declared:
        rows = [
            {name: row.get(name, "") for name in declared if name in row} | {
                k: v for k, v in row.items() if k not in declared
            }
            for row in rows
        ]

    parsed = rows_from_json(rows)
    dataset_id = uuid.uuid4().hex
    await data.create_dataset(
        dataset_id,
        name=(body.name or f"Found by {body.output}").strip()[:200],
        # Not a file. Recording where it came from is what makes the second
        # pass traceable back to the crawl that produced its rows.
        filename=f"run:{body.execution_id or body.batch_id}#{body.output}"[:400],
        source="discovery",
        rows=parsed.rows,
        columns=[column.to_dict() for column in parsed.columns],
        warnings=parsed.warnings,
        owner_id=principal.user_id,
        owner_email=principal.email,
    )
    await data.audit(
        "dataset.from_run",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="dataset",
        resource_id=dataset_id,
        detail={
            "output": body.output,
            "rows": len(parsed.rows),
            "execution_id": body.execution_id,
            "batch_id": body.batch_id,
        },
    )
    log.info(
        "dataset built from a run",
        extra={"dataset_id": dataset_id, "rows": len(parsed.rows), "output": body.output},
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
