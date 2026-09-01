"""What this workspace has learned about broken locators.

Healing writes here on its own when it is sure. These endpoints are the other
half: what it learned, whether that was right, and taking it back out when it
was not.

**Why a person can write here too.** When the model is not confident enough to
repair a step, the run stops and somebody looks at it. What they know at that
moment -- "they moved the button into the dialog" -- is worth more than what
the model guessed, and there was previously nowhere to put it. Recording it
means the next run has a human-confirmed precedent, which ``memory.as_prompt``
ranks above anything the model worked out alone.

**Why forgetting matters as much as remembering.** A fix that was right last
month and wrong now is exactly what makes healing confidently incorrect. It has
to be removable, by someone who can see what is in there.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from auth.rbac import Permission
from auth.service import Principal
from deps import WorkspaceData, require
from memory import domain_of
from routers.schemas import RememberFixRequest

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("")
async def list_fixes(
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.USECASE_READ))],
    domain: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Every fix this workspace remembers, newest first."""
    return {"fixes": await data.list_fixes(domain=domain, limit=max(1, min(limit, 200)))}


@router.post("", status_code=201)
async def remember_fix(
    body: RememberFixRequest,
    request: Request,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_REPAIR))],
) -> dict[str, Any]:
    """Record what a person worked out, so the next run does not have to.

    Stamped with their email rather than ``model``: a fix somebody looked at
    and confirmed outranks one nobody checked, and that distinction is what
    ``as_prompt`` sorts on.
    """
    replays = request.app.state.replays
    memory = replays.make_memory(principal.workspace_id)
    if memory is None or not memory.available:
        raise HTTPException(
            status_code=503,
            detail=(
                "Healing memory is switched off for this deployment, so there is "
                "nowhere to record that. Set HEALING_MEMORY_ENABLED=true."
            ),
        )
    if not domain_of(body.page_url):
        raise HTTPException(
            status_code=422,
            detail="page_url must be an absolute URL: the domain is what scopes recall.",
        )

    await memory.remember(
        usecase_id=body.usecase_id,
        step_id=body.step_id,
        page_url=body.page_url,
        page=body.page,
        step_summary=body.step_summary,
        wanted=body.wanted,
        old_locator=body.old_locator,
        new_locator=body.new_locator,
        explanation=body.explanation,
        confirmed_by=principal.email,
        error_kind=body.error_kind,
    )
    await data.audit(
        "memory.remember",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="usecase",
        resource_id=body.usecase_id or "",
        detail={"step_id": body.step_id, "domain": domain_of(body.page_url)},
    )
    return {"remembered": True, "domain": domain_of(body.page_url)}


@router.delete("/{fix_id}")
async def forget_fix(
    fix_id: str,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_REPAIR))],
) -> dict[str, Any]:
    """Take a fix back out.

    A fix that has gone stale does not fail loudly -- it gets recalled, put in
    front of the model as precedent, and quietly makes the next repair worse.
    """
    if not await data.forget_fix(fix_id):
        raise HTTPException(status_code=404, detail="No such remembered fix.")
    await data.audit(
        "memory.forget",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="memory",
        resource_id=fix_id,
    )
    return {"forgotten": fix_id}
