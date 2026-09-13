"""Where a use case actually points, per deployment.

A use case names a target -- ``schemora``, ``bank`` -- and each deployment
answers that name with its own address. Twenty workflows against one site share
one row; moving that site's UAT host is one edit; onboarding a new site is a
row rather than a release.

This is deliberately *not* configuration. The base URL used to come from a map
in the environment, which works while every workflow in an environment shares
one site and needs a variable and a redeploy for each one after that. Targets
are edited by the people who run the workflows, in the environment they are
running them in.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException

from auth.rbac import Permission
from auth.service import Principal
from deps import WorkspaceData, require
from routers.schemas import TargetRequest

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/targets", tags=["targets"])


@router.get("")
async def list_targets(
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.USECASE_READ))],
) -> dict[str, Any]:
    return {"targets": await data.list_targets()}


@router.put("/{name}", status_code=201)
async def save_target(
    name: str,
    body: TargetRequest,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_CREATE))],
) -> dict[str, Any]:
    """Create or update the address this deployment gives one target.

    An origin, not a URL with a path: a use case's steps carry their own paths,
    and a target holding ``/login`` would put it in front of every one of them.
    """
    parsed = urlparse(body.base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(
            status_code=422,
            detail=(
                "A target's address must be an absolute http or https URL, "
                "for example https://uat.example.com"
            ),
        )
    if parsed.path.strip("/") or parsed.query or parsed.fragment:
        raise HTTPException(
            status_code=422,
            detail=(
                "A target is an origin, not a page. Give it "
                f"{parsed.scheme}://{parsed.netloc} and let each use case's steps "
                "carry their own paths."
            ),
        )

    saved = await data.save_target(
        name.strip(),
        f"{parsed.scheme}://{parsed.netloc}",
        description=body.description.strip(),
        updated_by=principal.email,
    )
    await data.audit(
        "target.save",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="target",
        resource_id=saved["name"],
        detail={"base_url": saved["base_url"]},
    )
    log.info("target saved", extra={"target": saved["name"]})
    return saved


@router.delete("/{name}")
async def delete_target(
    name: str,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.USECASE_CREATE))],
) -> dict[str, Any]:
    """Remove a target.

    Use cases naming it are left alone rather than rewritten: the next run
    refuses with a message saying which target is missing, which is the right
    outcome. Silently repointing a workflow at the address it was recorded
    against is the accident targets exist to prevent.
    """
    if not await data.delete_target(name):
        raise HTTPException(status_code=404, detail="No such target.")
    await data.audit(
        "target.delete",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="target",
        resource_id=name,
        detail={},
    )
    return {"deleted": name}
