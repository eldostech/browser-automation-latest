"""Logic that sits between the HTTP layer and the store.

These functions used to live inline in ``main.py``, where several of them were
written out twice -- once for the single-row execute path and once for the
batch path. Duplicated request handling is where two endpoints quietly stop
agreeing about what "ready" or "missing credential" means.

They raise ``HTTPException`` because every caller is an HTTP handler and
translating a bespoke exception at each call site would be ceremony without
benefit. If a non-HTTP caller ever appears -- a queue worker, say -- that is the
moment to introduce a domain exception and map it once at the boundary.
"""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Any

from fastapi import HTTPException

from batch import BatchInputError, parse_csv, parse_workbook, rows_from_json
from credentials import Vault, VaultError, VaultUnavailable
from routers.schemas import BatchRequestBody, ExecuteRequest, RepairRequest
from store import WorkspaceStore
from usecase import UseCase

log = logging.getLogger(__name__)


async def resolve_secrets(
    body: ExecuteRequest, data: WorkspaceStore, vault: Vault
) -> dict[str, str]:
    """Decrypt the bound credential, or take inline values for a one-off.

    Whatever comes back is registered with the run's redactor before anything
    is emitted, so a value cannot reach the event log even if a tool echoes it.
    """
    if body.credential_id:
        ciphertext = await data.get_credential_ciphertext(body.credential_id)
        if ciphertext is None:
            # Also the answer when the credential belongs to another
            # workspace: the scoped store simply cannot see it.
            raise HTTPException(status_code=404, detail="No such credential.")
        try:
            values = vault.open(ciphertext)
        except VaultUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except VaultError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        await data.touch_credential(body.credential_id)
        return {**values, **(body.secrets or {})}
    return dict(body.secrets or {})


async def load_runnable_usecase(
    usecase_id: str, version: int | None, data: WorkspaceStore
) -> tuple[UseCase, int, dict[str, Any]]:
    """Fetch a use case and refuse it unless it is fit to execute.

    Returns ``(use_case, version, raw_definition)``.
    """
    definition = await data.get_usecase(usecase_id, version)
    if definition is None:
        raise HTTPException(status_code=404, detail="No such use case.")

    try:
        use_case = UseCase.model_validate(definition)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the editor
        raise HTTPException(
            status_code=422, detail=f"The stored use case is invalid: {exc}"
        ) from exc

    if use_case.status != "ready":
        raise HTTPException(
            status_code=409,
            detail=(
                f"This use case is {use_case.status!r}. Review it and publish it before "
                "running it -- a distilled recording is a best guess until a person has "
                "checked it."
            ),
        )
    return use_case, int(definition.get("version") or 1), definition


async def require_scripts_permitted(usecase_id: str, use_case: UseCase, data: WorkspaceStore) -> None:
    """Refuse to execute script steps unless an admin has enabled them here.

    ``allow_scripts`` inside the definition says the *author* wants scripts; it
    is part of a document a user can edit. This checks the separate flag on the
    resource, which only someone holding ``script:enable`` can set. Both must
    agree, so an author cannot grant themselves code execution by editing JSON.
    """
    if not any(step.action == "script" for step in use_case.all_steps):
        return

    row = await data.get_usecase_row(usecase_id)
    if not row or not row.get("scripts_enabled"):
        raise HTTPException(
            status_code=403,
            detail=(
                "This use case contains a script step. An administrator must review the "
                "code and enable scripts for it before it can run, because a script runs "
                "arbitrary JavaScript inside a browser session that may be signed in."
            ),
        )


def rows_from_body(body: BatchRequestBody):
    """Whichever way the rows arrived, one shape comes out."""
    if body.xlsx_base64 is not None:
        try:
            data = base64.b64decode(body.xlsx_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise BatchInputError(f"the workbook was not valid base64: {exc}") from exc
        return parse_workbook(data, body.sheet)
    if body.csv is not None:
        return parse_csv(body.csv)
    return rows_from_json(body.rows)


async def find_failed_execution(
    usecase_id: str, body: RepairRequest, data: WorkspaceStore
) -> dict[str, Any]:
    """The failure to repair: the one named, or the most recent."""
    executions = await data.list_executions(usecase_id=usecase_id, limit=200)
    if body.execution_id:
        match = next((e for e in executions if e["id"] == body.execution_id), None)
    elif body.run_id:
        match = next((e for e in executions if e["run_id"] == body.run_id), None)
    else:
        match = next(
            (
                e
                for e in sorted(executions, key=lambda e: e["created_at"] or "", reverse=True)
                if e["status"] == "failed"
            ),
            None,
        )

    if match is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "no failed run found for this use case; run it once so there is a "
                "failure to look at"
            ),
        )
    if match["status"] != "failed":
        raise HTTPException(
            status_code=409, detail="that run did not fail, so there is nothing to repair"
        )
    return match


def require_missing_nothing(use_case: UseCase, secrets: dict[str, str], values: dict[str, Any] | None = None) -> None:
    """Refuse the request if a declared secret or input has no value.

    Checked before a browser opens, so a missing column fails in a millisecond
    rather than on record 700.
    """
    missing_secrets = use_case.missing_secrets(secrets)
    if missing_secrets:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required credential slot(s): {', '.join(missing_secrets)}",
        )
    if values is not None:
        missing_inputs = use_case.missing_inputs(values)
        if missing_inputs:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required input(s): {', '.join(missing_inputs)}",
            )


__all__ = [
    "find_failed_execution",
    "load_runnable_usecase",
    "require_missing_nothing",
    "require_scripts_permitted",
    "resolve_secrets",
    "rows_from_body",
]
