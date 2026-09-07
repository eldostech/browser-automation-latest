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
    get_replays,
    get_vault,
    require,
    run_or_404,
    usecase_or_404,
)
from repair import (
    locator_changes,
    RepairError,
    UseCaseDoctor,
    apply_fixes,
    candidates,
    gather_context,
    is_unchanged,
    validate_patched,
)
from routers.schemas import DistillRequest, RenameRequest, RepairRequest, ScriptsRequest
from runner import ReplayManager
from services import find_failed_execution
from usecase import Step, UseCase

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["usecases"])

Replays = Annotated[ReplayManager, Depends(get_replays)]


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


@router.post("/usecases/import", status_code=201)
async def import_usecase(
    body: dict[str, Any],
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_CREATE))],
) -> dict[str, Any]:
    """Take a definition exported from another environment and land it here.

    This is how a use case reaches UAT and production: it is recorded once,
    against dev, and the document moves. Re-recording in each environment would
    produce three different documents that drift apart, which is the thing this
    exists to prevent.

    Nothing sensitive travels. Secrets are *slots* in a definition -- names,
    never values -- so each environment supplies its own under its own
    ``CREDENTIALS_KEY``, and the addresses are bound to ``{{env.base_url}}``,
    which this deployment answers for itself. See section 8.4 of the design
    document.

    Two things are deliberately not carried across, both for the same reason:
    an approval given in one environment is not an approval in another.

    ``status`` always lands at ``draft``. Publishing is per-environment, and it
    re-validates; arriving as a draft is what forces UAT to look at this rather
    than inherit dev's decision.

    ``allow_scripts`` always lands false. It is the flag that lets a use case
    run arbitrary JavaScript against a live page, and it is granted by a person
    who has read the code. Carrying it across would let code approved against
    dev's data execute against production's, which nobody would have agreed to.

    The id *is* preserved, so one use case is the same use case everywhere and
    a run in UAT can be lined up against the run in dev it came from.
    Re-importing appends a version rather than duplicating, which makes
    promoting a revision the same gesture as promoting it the first time.
    """
    incoming = {**body}
    incoming["status"] = "draft"
    incoming["allow_scripts"] = False
    # A run id from the source environment names a run that does not exist in
    # this database. Keeping it would be a reference that resolves to nothing,
    # or worse, to something unrelated.
    incoming.pop("source_run_id", None)
    # An absent id is a definition built by hand rather than exported; the
    # model mints one.

    try:
        use_case = UseCase.model_validate(incoming)
    except ValidationError as exc:
        # Surfaced verbatim: a definition that will not validate here is the
        # most useful thing to show whoever is doing the promotion.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        usecase_id, version = await data.save_usecase(
            use_case.model_dump(mode="json", by_alias=True),
            created_by=f"imported by {principal.email}",
            created_by_id=principal.user_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    await data.audit(
        "usecase.import",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="usecase",
        resource_id=usecase_id,
        detail={"version": version, "name": use_case.name},
    )
    log.info(
        "use case imported",
        extra={"usecase_id": usecase_id, "version": version},
    )
    return {
        "usecase_id": usecase_id,
        "version": version,
        "status": use_case.status,
        "imported_by": principal.email,
    }


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


async def _remember_repair(
    request: Request,
    principal: Principal,
    *,
    usecase_id: str,
    before: dict[str, Any],
    after: dict[str, Any],
    context: Any,
    proposal: Any,
) -> int:
    """Write each landed locator change to healing memory. Returns how many.

    Never raises into the repair. A use case that was successfully mended must
    not be reported as failed because the thing that remembers it is switched
    off, unreachable, or out of embedding quota -- the repair is the point and
    the memory is the bonus.
    """
    try:
        memory = request.app.state.replays.make_memory(principal.workspace_id)
        if memory is None or not memory.available:
            return 0

        page = "\n".join(
            f'{index}. {node.role} "{node.name or node.text}"'
            for index, node in enumerate(candidates(context.snapshot))
        )
        written = 0
        for change in locator_changes(before, after):
            await memory.remember(
                usecase_id=usecase_id,
                step_id=change.step_id,
                page_url=context.page_url or "",
                page=page,
                step_summary=_summarise_step(after, change.step_id),
                wanted=change.field_name or _wanted_from(change.old_locator),
                old_locator=change.old_locator,
                new_locator=change.new_locator,
                explanation=proposal.diagnosis,
                confirmed_by=principal.email,
            )
            written += 1
        return written
    except Exception:  # noqa: BLE001 - see the docstring
        log.warning(
            "the repair was saved but could not be remembered",
            extra={"usecase_id": usecase_id},
            exc_info=True,
        )
        return 0


def _summarise_step(definition: dict[str, Any], step_id: str) -> str:
    """How the step reads, for the text a recall is matched against."""
    for phase in ("setup_steps", "row_steps", "teardown_steps"):
        for step in definition.get(phase) or []:
            if step.get("id") == step_id:
                try:
                    return Step.model_validate(step).summary()
                except ValidationError:
                    return f"{step.get('action', 'step')} {step_id}"
    return step_id


def _wanted_from(locator: dict[str, Any] | None) -> str:
    """What the step was looking for before the repair, in words."""
    if not locator:
        return ""
    return str(locator.get("name") or locator.get("text") or locator.get("selector") or "")


@router.post("/usecases/{usecase_id}/repair", status_code=201)
async def repair_usecase(
    usecase_id: str,
    body: RepairRequest,
    request: Request,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_REPAIR))],
) -> dict[str, Any]:
    """Mend a use case that failed, using the page as it was when it broke.

    One LLM call. The result is saved as a new **draft** version -- existing
    versions are untouched and a person publishes it, which is the same gate a
    recorded use case passes through.
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

    # A script step has no locators, so there is nothing for a repair to
    # retarget. Saying so costs nothing and beats spending an LLM call to be
    # told the same thing in vaguer words.
    failed_id = execution.get("failed_step_id")
    failed_step = next((s for s in use_case.all_steps if s.id == failed_id), None)
    if failed_step is not None and failed_step.action == "script":
        return {
            "usecase_id": usecase_id,
            "repaired": False,
            "diagnosis": (
                f"Step {failed_id!r} is raw JavaScript, so there is no locator to repair."
            ),
            "unfixable_reason": (
                "A script step carries code rather than a target, so nothing here can be "
                "retargeted at a changed page -- and a script that finds nothing usually "
                "returns quietly rather than failing, which is why the error surfaced at a "
                "later step instead. Re-record this task: the recorder no longer offers the "
                "model a JavaScript tool, so it will produce ordinary click and fill steps "
                "that can be reviewed and repaired."
            ),
            "confidence": "high",
            "llm_tokens": 0,
        }

    context = gather_context(use_case, execution, events)
    # Handed the memory so the button reads what it has already learned, not
    # only writes to it. Built the same way the healer's is, and None when
    # healing memory is switched off.
    try:
        recall = request.app.state.replays.make_memory(principal.workspace_id)
    except Exception:  # noqa: BLE001 - a repair must not fail for want of recall
        log.warning("healing memory is unavailable to this repair", exc_info=True)
        recall = None
    doctor = UseCaseDoctor(request.app.state.repair_model.client, memory=recall)
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

    patched, applied = apply_fixes(
        definition, proposal, candidates(context.snapshot), context.snapshot
    )

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

    # Learn from it. Without this the button mends one use case and teaches the
    # system nothing: the in-run healer wrote every high-confidence fix to
    # healing memory, and this path -- the one a person actually presses -- did
    # not, so the same page change was diagnosed from scratch every time, at the
    # cost of an LLM call each.
    #
    # Stamped with the person's email rather than "model". They chose to repair
    # this, looked at the result and published it, which is a stronger signal
    # than an unattended heal, and `as_prompt` sorts on exactly that.
    remembered = await _remember_repair(
        request,
        principal,
        usecase_id=usecase_id,
        before=definition,
        after=patched,
        context=context,
        proposal=proposal,
    )

    await data.audit(
        "usecase.repair",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="usecase",
        resource_id=usecase_id,
        detail={
            "version": version,
            "fixes": len(applied),
            "tokens": proposal.tokens,
            "remembered": remembered,
        },
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
        "remembered": remembered,
        "llm_tokens": proposal.tokens,
    }
