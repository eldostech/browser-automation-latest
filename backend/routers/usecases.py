"""Use cases: distil, review, publish, rename, repair, archive."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import ValidationError

from auth.rbac import Permission
from auth.service import Principal
from credentials import NO_KEY_MESSAGE, Vault, new_credential_id
from deps import (
    WorkspaceData,
    get_manager,
    get_replays,
    get_vault,
    require,
    run_or_404,
    usecase_or_404,
)
from distill import DistillationError, distill
from repair import (
    RepairError,
    UseCaseDoctor,
    apply_fixes,
    candidates,
    gather_context,
    is_unchanged,
    validate_patched,
)
from routers.schemas import DistillRequest, RenameRequest, RepairRequest, ScriptsRequest
from runner import ReplayManager, RunManager
from services import find_failed_execution
from usecase import UseCase

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["usecases"])

Manager = Annotated[RunManager, Depends(get_manager)]
Replays = Annotated[ReplayManager, Depends(get_replays)]


@router.post("/runs/{run_id}/distill", status_code=201)
async def distill_run(
    run_id: str,
    request: Request,
    data: WorkspaceData,
    manager: Manager,
    vault: Annotated[Vault, Depends(get_vault)],
    principal: Annotated[Principal, Depends(require(Permission.USECASE_CREATE))],
    body: DistillRequest | None = None,
) -> dict[str, Any]:
    """Promote a successful run into a reusable use case.

    This is the one LLM call in the whole replay feature. Everything the use
    case is later executed with costs nothing.
    """
    run = await run_or_404(run_id, data)
    if run.status != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Only a succeeded run can be recorded as a use case; this one is "
                f"{run.status!r}. A failed run has no reliable sequence of working "
                "steps to learn from."
            ),
        )

    # A replay is a use case being *executed*. Distilling one would spend an
    # LLM call to derive a use case from a use case -- a copy of the original
    # with its parameters already substituted into the steps, so the "inputs"
    # would be whichever row happened to run. The button that offered this is
    # gone from the UI, but the guard belongs here: the API is the contract.
    options = run.options or {}
    if options.get("replay"):
        source = options.get("usecase_id")
        raise HTTPException(
            status_code=409,
            detail=(
                "This run executed an existing use case rather than recording a new "
                "one, so there is nothing here to distil"
                + (f" -- it ran use case {source}." if source else ".")
                + " To change that use case, edit it or use Fix with AI on a failure."
            ),
        )

    events = await data.get_events(run_id)
    # What the user named before recording. Present, parameterisation is a
    # lookup rather than a judgement -- see fields.py.
    declared = (run.options or {}).get("declared") or {}

    try:
        use_case = await distill(
            events,
            task=run.task,
            llm=manager.distill_llm,
            source_run_id=run_id,
            declared=declared,
        )
    except DistillationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValidationError as exc:
        # A recording the schema cannot express is the user's problem to see,
        # not a server fault. Name the step rather than returning a 500.
        raise HTTPException(
            status_code=422,
            detail=f"This run could not be turned into a use case: {exc}",
        ) from exc

    body = body or DistillRequest()
    if body.name:
        use_case.name = body.name.strip() or use_case.name

    definition = use_case.model_dump(mode="json", by_alias=True)
    usecase_id, version = await data.save_usecase(
        definition,
        created_by="distilled",
        created_by_id=principal.user_id,
        owner_id=principal.user_id,
    )

    # The credentials the recording used have been sitting in memory since it
    # started. This is the moment they are either kept or forgotten -- there is
    # no third state, and doing nothing means forgetting.
    credential = await _resolve_recorded_credential(
        request.app.state.stash, run_id, body, data, vault, principal
    )

    await data.audit(
        "usecase.distill",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="usecase",
        resource_id=usecase_id,
        detail={"run_id": run_id, "version": version},
    )
    log.info(
        "distilled a run into a use case",
        extra={"run_id": run_id, "usecase_id": usecase_id, "version": version},
    )
    return {
        "usecase_id": usecase_id,
        "version": version,
        # A suggestion. The caller is expected to confirm or replace it via
        # PATCH before moving on.
        "name": use_case.name,
        "suggested_name": use_case.name,
        "status": use_case.status,
        "warnings": use_case.warnings,
        "setup_steps": len(use_case.setup_steps),
        "row_steps": len(use_case.row_steps),
        "inputs": [spec.name for spec in use_case.inputs],
        "secrets": [spec.name for spec in use_case.secrets],
        "blocked_scripts": use_case.blocked_scripts,
        "credential": credential,
    }


async def _resolve_recorded_credential(
    stash, run_id: str, body: DistillRequest, data, vault: Vault, principal: Principal
) -> dict[str, Any] | None:
    """Save the recording's credentials, or discard them. Never neither.

    ``take`` removes them from memory whichever way this goes, so an aborted
    save does not leave a password sitting in the process.
    """
    values = stash.take(run_id, workspace_id=principal.workspace_id)
    if not values:
        return None

    if not body.save_credential_as:
        log.info(
            "discarded the credentials a recording used",
            extra={"run_id": run_id, "slots": len(values)},
        )
        return {"saved": False, "slots": sorted(values)}

    if not vault.available:
        # Refusing beats pretending: the user asked for them to be kept and
        # they cannot be, so say so rather than silently dropping them.
        raise HTTPException(
            status_code=503,
            detail=(
                NO_KEY_MESSAGE + " The credentials from this recording have been "
                "discarded; set the key and re-record, or add them by hand later."
            ),
        )

    credential_id = await data.save_credential(
        new_credential_id(),
        body.save_credential_as.strip(),
        Vault.slots_of(values),
        vault.seal(values),
        owner_id=principal.user_id,
    )
    await data.audit(
        "credential.save",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="credential",
        resource_id=credential_id,
        detail={"name": body.save_credential_as, "from_run": run_id},
    )
    return {
        "saved": True,
        "id": credential_id,
        "name": body.save_credential_as.strip(),
        "slots": Vault.slots_of(values),
    }


@router.get("/usecases")
async def list_usecases(
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.USECASE_READ))],
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    rows = await data.list_usecases(status=status, limit=limit, offset=offset)
    return {"usecases": rows, "limit": limit, "offset": offset}


@router.get("/usecases/{usecase_id}")
async def get_usecase(
    usecase_id: str,
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.USECASE_READ))],
    version: int | None = Query(default=None),
) -> dict[str, Any]:
    definition = await data.get_usecase(usecase_id, version)
    if definition is None:
        raise HTTPException(status_code=404, detail="No such use case.")
    return {
        "definition": definition,
        "versions": await data.list_usecase_versions(usecase_id),
        "meta": await data.get_usecase_row(usecase_id),
    }


@router.get("/usecases/{usecase_id}/activity")
async def usecase_activity(
    usecase_id: str,
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.USECASE_READ))],
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """Who did what to this use case, and when.

    Deliberately *not* behind ``audit:read``. That permission guards the
    workspace-wide log, which carries account changes and credential activity
    and is properly an administrator's view. Knowing who published a use case
    and who last ran it is ordinary operational context for anyone allowed to
    see the use case at all -- withholding it would make the audit trail
    something only an admin can benefit from, which defeats the point.
    """
    await usecase_or_404(usecase_id, data)
    return {
        "usecase_id": usecase_id,
        "entries": await data.list_audit(
            resource_type="usecase", resource_id=usecase_id, limit=limit
        ),
    }


@router.put("/usecases/{usecase_id}", status_code=201)
async def update_usecase(
    usecase_id: str,
    body: dict[str, Any],
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_CREATE))],
) -> dict[str, Any]:
    """Save reviewer edits as a new version.

    Never rewrites the version in place: a batch already running is reading
    from a specific version and must not have it changed underneath it.
    """
    await usecase_or_404(usecase_id, data)

    body = {**body, "id": usecase_id}
    try:
        use_case = UseCase.model_validate(body)
    except Exception as exc:  # noqa: BLE001 - surfaced to the editing UI verbatim
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _, version = await data.save_usecase(
        use_case.model_dump(mode="json", by_alias=True),
        created_by="edited",
        created_by_id=principal.user_id,
    )
    await data.audit(
        "usecase.edit",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="usecase",
        resource_id=usecase_id,
        detail={"version": version},
    )
    return {"usecase_id": usecase_id, "version": version, "status": use_case.status}


@router.patch("/usecases/{usecase_id}")
async def rename_usecase(
    usecase_id: str,
    body: RenameRequest,
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.USECASE_CREATE))],
) -> dict[str, Any]:
    """Rename a use case in place.

    No new version: a name is a label, not part of what executes, so renaming
    must not appear in a history that exists to record behaviour. Use PUT to
    change the steps.
    """
    if not await data.rename_usecase(usecase_id, body.name.strip(), body.description):
        raise HTTPException(status_code=404, detail="No such use case.")
    return {"usecase_id": usecase_id, "name": body.name.strip()}


@router.post("/usecases/{usecase_id}/publish")
async def publish_usecase(
    usecase_id: str,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_PUBLISH))],
) -> dict[str, Any]:
    """Move a reviewed draft to ``ready`` so it can be executed.

    Re-validates at ``ready``, which is where the stricter rules bite -- most
    notably that a use case carrying raw JavaScript cannot be published until
    someone has read the code and turned ``allow_scripts`` on.
    """
    definition = await usecase_or_404(usecase_id, data)

    try:
        UseCase.model_validate({**definition, "status": "ready"})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    await data.set_usecase_status(usecase_id, "ready")
    await data.audit(
        "usecase.publish",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="usecase",
        resource_id=usecase_id,
        detail={"version": definition.get("version")},
    )
    return {"usecase_id": usecase_id, "status": "ready", "published_by": principal.email}


@router.post("/usecases/{usecase_id}/scripts")
async def set_scripts(
    usecase_id: str,
    body: ScriptsRequest,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.SCRIPT_ENABLE))],
) -> dict[str, Any]:
    """Permit script steps on one use case. Administrators only.

    A script step runs arbitrary JavaScript inside a browser session that may
    be signed in with somebody else's credentials, so this is a separate
    authority from writing or publishing a use case -- an author cannot grant
    themselves code execution by editing the definition's ``allow_scripts``
    field, because execution checks *this* flag as well.
    """
    await usecase_or_404(usecase_id, data)
    if not await data.set_scripts_enabled(usecase_id, body.enabled, actor_id=principal.user_id):
        raise HTTPException(status_code=404, detail="No such use case.")

    await data.audit(
        "usecase.scripts_enabled" if body.enabled else "usecase.scripts_disabled",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="usecase",
        resource_id=usecase_id,
        detail={"reason": body.reason},
    )
    log.warning(
        "script execution permission changed",
        extra={
            "usecase_id": usecase_id,
            "enabled": body.enabled,
            "actor": principal.email,
        },
    )
    return {"usecase_id": usecase_id, "scripts_enabled": body.enabled, "by": principal.email}


@router.delete("/usecases/{usecase_id}")
async def archive_usecase(
    usecase_id: str,
    data: WorkspaceData,
    replays: Replays,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_DELETE))],
    purge: bool = Query(default=False),
) -> dict[str, Any]:
    """Archive a use case, or delete it outright with ``?purge=true``.

    Archiving is the default because it is reversible and keeps every
    reference intact. Purging removes the use case, its versions and its
    execution records permanently; the runs and events they produced are kept,
    since those record what actually happened to a browser.
    """
    active = replays.active
    if active and active.get("usecase_id") == usecase_id:
        raise HTTPException(
            status_code=409,
            detail=(
                "That use case is running right now; wait for it to finish or cancel it first."
            ),
        )

    if purge:
        removed = await data.purge_usecase(usecase_id)
        if removed is None:
            raise HTTPException(status_code=404, detail="No such use case.")
        await data.audit(
            "usecase.purge",
            actor_id=principal.user_id,
            actor_email=principal.email,
            resource_type="usecase",
            resource_id=usecase_id,
            detail=removed,
        )
        return {"usecase_id": usecase_id, "deleted": True, "removed": removed}

    if not await data.delete_usecase(usecase_id):
        raise HTTPException(status_code=404, detail="No such use case.")
    await data.audit(
        "usecase.archive",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="usecase",
        resource_id=usecase_id,
    )
    return {"usecase_id": usecase_id, "status": "archived"}


@router.post("/usecases/{usecase_id}/repair", status_code=201)
async def repair_usecase(
    usecase_id: str,
    body: RepairRequest,
    data: WorkspaceData,
    manager: Manager,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_REPAIR))],
) -> dict[str, Any]:
    """Mend a use case that failed, using the page as it was when it broke.

    One LLM call. The result is saved as a new **draft** version -- existing
    versions are untouched and a person publishes it, which is the same gate a
    freshly distilled use case passes through.
    """
    definition = await usecase_or_404(usecase_id, data)

    execution = await find_failed_execution(usecase_id, body, data)
    events = await data.get_events(execution["run_id"]) if execution.get("run_id") else []

    try:
        use_case = UseCase.model_validate({**definition, "status": "draft"})
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail=f"The stored use case is invalid: {exc}"
        ) from exc

    context = gather_context(use_case, execution, events)
    doctor = UseCaseDoctor(manager.repair_llm)
    try:
        proposal = await doctor.diagnose(context)
    except RepairError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not proposal.actionable:
        return {
            "usecase_id": usecase_id,
            "repaired": False,
            "diagnosis": proposal.diagnosis,
            "unfixable_reason": proposal.unfixable_reason
            or "The model had no edit to suggest for this failure.",
            "confidence": proposal.confidence,
            "llm_tokens": proposal.tokens,
        }

    patched, applied = apply_fixes(definition, proposal, candidates(context.snapshot))

    # A repair that changes nothing must not be reported as one, and must not
    # leave a version behind. Otherwise pressing the button appears to work
    # while the use case stays exactly as broken as it was.
    if is_unchanged(definition, patched):
        skipped = [line for line in applied if line.startswith("SKIPPED")]
        log.info(
            "repair proposed nothing that could be applied",
            extra={
                "usecase_id": usecase_id,
                "fixes": len(proposal.fixes),
                "skipped": len(skipped),
            },
        )
        return {
            "usecase_id": usecase_id,
            "repaired": False,
            "diagnosis": proposal.diagnosis,
            "confidence": proposal.confidence,
            "applied": applied,
            "unfixable_reason": (
                "None of the proposed edits could be applied, so nothing was saved: "
                + "; ".join(skipped)
                if skipped
                else "The proposed edits would leave the use case exactly as it is, so "
                "nothing was saved. It may already carry this repair."
            ),
            "llm_tokens": proposal.tokens,
        }

    try:
        validate_patched(patched)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "The proposed repair does not produce a valid use case, so nothing was "
                f"saved: {exc}"
            ),
        ) from exc

    patched["status"] = "draft"
    _, version = await data.save_usecase(
        patched, created_by="repair", created_by_id=principal.user_id
    )
    await data.set_usecase_status(usecase_id, "draft")

    await data.audit(
        "usecase.repair",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="usecase",
        resource_id=usecase_id,
        detail={"version": version, "fixes": len(applied), "tokens": proposal.tokens},
    )
    log.info(
        "repaired a use case",
        extra={"usecase_id": usecase_id, "version": version, "fixes": len(applied)},
    )
    return {
        "usecase_id": usecase_id,
        "repaired": True,
        "version": version,
        "diagnosis": proposal.diagnosis,
        "confidence": proposal.confidence,
        "applied": applied,
        "llm_tokens": proposal.tokens,
    }
