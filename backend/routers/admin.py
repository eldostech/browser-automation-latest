"""Account management and the audit trail. Administrators only."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from auth.rbac import Permission, ROLE_PERMISSIONS
from auth.service import AuthError, AuthService, Principal
from deps import WorkspaceData, get_auth, require
from routers.schemas import CreateUserRequest, SpendLimitRequest, UpdateUserRequest

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])

Admin = Annotated[Principal, Depends(require(Permission.USER_MANAGE))]


@router.get("/roles")
async def list_roles(_: Admin) -> dict[str, Any]:
    """The role/permission matrix, so the UI does not hard-code a second copy."""
    return {
        "roles": [
            {"name": role.value, "permissions": sorted(p.value for p in perms)}
            for role, perms in ROLE_PERMISSIONS.items()
        ]
    }


@router.get("/users")
async def list_users(
    admin: Admin, auth: Annotated[AuthService, Depends(get_auth)]
) -> dict[str, Any]:
    return {"users": await auth.list_users(admin.workspace_id)}


@router.post("/users", status_code=201)
async def create_user(
    body: CreateUserRequest,
    admin: Admin,
    auth: Annotated[AuthService, Depends(get_auth)],
    data: WorkspaceData,
) -> dict[str, Any]:
    """Add an account to the caller's own workspace.

    The workspace is taken from the caller, never from the request body -- an
    admin of one tenant must not be able to create users in another by naming
    its id.
    """
    try:
        user = await auth.create_user(
            workspace_id=admin.workspace_id,
            email=body.email,
            password=body.password,
            role=body.role,
            display_name=body.display_name,
        )
    except AuthError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    await data.audit(
        "user.create",
        actor_id=admin.user_id,
        actor_email=admin.email,
        resource_type="user",
        resource_id=user["id"],
        detail={"email": user["email"], "role": user["role"]},
    )
    return user


@router.patch("/users/{user_id}")
async def update_user(
    user_id: str,
    body: UpdateUserRequest,
    admin: Admin,
    auth: Annotated[AuthService, Depends(get_auth)],
    data: WorkspaceData,
) -> dict[str, Any]:
    """Change a role, enable/disable an account, or reset a password."""
    changed: dict[str, Any] = {}
    try:
        if body.role is not None:
            user = await auth.set_role(admin.workspace_id, user_id, body.role)
            if user is None:
                raise HTTPException(status_code=404, detail="No such user.")
            changed["role"] = body.role
        if body.is_active is not None:
            user = await auth.set_active(admin.workspace_id, user_id, body.is_active)
            if user is None:
                raise HTTPException(status_code=404, detail="No such user.")
            changed["is_active"] = body.is_active
        if body.new_password is not None:
            await auth.reset_password(admin.workspace_id, user_id, body.new_password)
            changed["password"] = "reset"
    except AuthError as exc:
        # Includes "this is the only administrator", which is a refusal rather
        # than a failure and reads better as a 409.
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if not changed:
        raise HTTPException(status_code=422, detail="Nothing to change.")

    await data.audit(
        "user.update",
        actor_id=admin.user_id,
        actor_email=admin.email,
        resource_type="user",
        resource_id=user_id,
        detail=changed,
    )
    return {"id": user_id, "changed": changed}


@router.get("/audit")
async def read_audit(
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.AUDIT_READ))],
    resource_type: str | None = Query(default=None),
    resource_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    """Who did what, most recent first."""
    return {
        "entries": await data.list_audit(
            resource_type=resource_type, resource_id=resource_id, limit=limit
        )
    }


@router.get("/spend")
async def get_spend(
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.RUN_READ))],
) -> dict[str, Any]:
    """What this workspace has spent with a model this month, and its ceiling.

    Readable by anyone who can read runs, not just an administrator. A person
    about to start an agent session needs to know whether there is room, and
    finding out by being refused is a worse way to learn it.
    """
    return await data.spend_this_month()


@router.put("/spend/limit")
async def set_spend_limit(
    body: SpendLimitRequest,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.USER_MANAGE))],
) -> dict[str, Any]:
    """Set or clear the monthly ceiling. Administrators only.

    Setting a budget is the same kind of authority as managing accounts, and
    deliberately not the same as being able to spend against it -- the person
    who can raise the limit should not be every person who can hit it.
    """
    await data.set_spend_limit(body.limit_usd)
    await data.audit(
        "workspace.spend_limit",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="workspace",
        resource_id=principal.workspace_id,
        detail={"limit_usd": body.limit_usd},
    )
    return await data.spend_this_month()
