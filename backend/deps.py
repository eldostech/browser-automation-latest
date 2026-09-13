"""FastAPI dependencies: configuration, services, identity and authorization.

Everything a handler needs arrives through this module, which is what lets a
test build an app with a different database, a fake LLM or a fixed principal by
overriding one dependency instead of monkeypatching an import.

**Authorization is a dependency, not an ``if``.** ``require(Permission.X)``
returns something a router can declare, so the check happens before the handler
body runs and is visible in the route definition -- and in the generated
OpenAPI -- rather than buried three lines into a function. A handler that
forgets to check is a handler that never receives a request.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Callable

from fastapi import Depends, HTTPException, Query, Request, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth.rbac import Permission
from auth.service import AuthService, Principal
from config import Settings
from credentials import Vault
from jobs import JobQueue
from store import Store, WorkspaceStore

log = logging.getLogger(__name__)

#: auto_error=False so a missing header produces our 401 with a WWW-Authenticate
#: challenge rather than FastAPI's bare 403, which misreports the situation.
bearer_scheme = HTTPBearer(auto_error=False)

UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Authentication required.",
    headers={"WWW-Authenticate": "Bearer"},
)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


def get_config(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_auth(request: Request) -> AuthService:
    return request.app.state.auth


def get_vault(request: Request) -> Vault:
    return request.app.state.vault


def get_replays(request: Request):
    return request.app.state.replays


def get_queue(request: Request) -> JobQueue:
    return request.app.state.queue


def get_bus(request: Request):
    return request.app.state.bus


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


async def current_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)] = None,
) -> Principal:
    """The authenticated caller, or 401.

    This is the only place a token becomes an identity. Handlers take a
    ``Principal`` and never see the token, so there is no way for one to
    accidentally trust a client-supplied user id.
    """
    if credentials is None or not credentials.credentials:
        raise UNAUTHENTICATED

    auth: AuthService = request.app.state.auth
    principal = await auth.resolve(credentials.credentials)
    if principal is None:
        raise UNAUTHENTICATED
    return principal


CurrentUser = Annotated[Principal, Depends(current_principal)]


async def principal_from_websocket(
    websocket: WebSocket, token: str | None = Query(default=None)
) -> Principal | None:
    """Authenticate a WebSocket.

    Browsers cannot set headers on a WebSocket handshake, so the token arrives
    as a query parameter. That is a real downside -- query strings turn up in
    access logs and proxy logs -- and it is mitigated by these being short-lived
    session tokens that can be revoked, rather than long-lived API keys.

    Returns None rather than raising: the caller must close the socket with a
    policy-violation code, which is not something an HTTPException can do.
    """
    if not token:
        return None
    auth: AuthService = websocket.app.state.auth
    return await auth.resolve(token)


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


def require(*permissions: Permission) -> Callable[..., Any]:
    """A dependency asserting the caller holds every listed permission.

    Declared on the route:

        @router.post("/usecases", dependencies=[Depends(require(Permission.USECASE_CREATE))])

    or taken as an argument when the handler also wants the principal:

        principal: Principal = Depends(require(Permission.USECASE_CREATE))
    """

    async def dependency(principal: CurrentUser) -> Principal:
        missing = [p for p in permissions if not principal.can(p)]
        if missing:
            log.info(
                "permission denied",
                extra={
                    "user_id": principal.user_id,
                    "role": principal.role,
                    "missing": [p.value for p in missing],
                },
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Your role does not allow this action "
                    f"(requires {', '.join(p.value for p in missing)})."
                ),
            )
        return principal

    return dependency


# ---------------------------------------------------------------------------
# Tenant-scoped data access
# ---------------------------------------------------------------------------


async def get_workspace_store(
    principal: CurrentUser, store: Annotated[Store, Depends(get_store)]
) -> WorkspaceStore:
    """The store, confined to the caller's workspace.

    Handlers depend on this rather than on ``Store``, so the tenant filter is
    applied by construction and a handler has no object capable of reaching
    another workspace's rows.
    """
    return store.workspace(principal.workspace_id)


WorkspaceData = Annotated[WorkspaceStore, Depends(get_workspace_store)]


# ---------------------------------------------------------------------------
# Shared lookups
# ---------------------------------------------------------------------------


async def usecase_or_404(usecase_id: str, data: WorkspaceData) -> dict[str, Any]:
    """Fetch a use case definition or raise 404.

    Twenty-one hand-written ``raise HTTPException(404)`` blocks used to say
    this. Note that "belongs to another workspace" and "does not exist" produce
    the same 404 -- deliberately, since a distinguishable 403 would confirm
    that an id exists to someone who should not know it.
    """
    definition = await data.get_usecase(usecase_id)
    if definition is None:
        raise HTTPException(status_code=404, detail="No such use case.")
    return definition


async def run_or_404(run_id: str, data: WorkspaceData):
    record = await data.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No such run.")
    return record


async def batch_or_404(batch_id: str, data: WorkspaceData) -> dict[str, Any]:
    batch = await data.get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="No such batch.")
    return batch


__all__ = [
    "CurrentUser",
    "Permission",
    "Principal",
    "WorkspaceData",
    "batch_or_404",
    "bearer_scheme",
    "current_principal",
    "get_auth",
    "get_bus",
    "get_config",
    "get_queue",
    "get_replays",
    "get_store",
    "get_vault",
    "get_workspace_store",
    "principal_from_websocket",
    "require",
    "run_or_404",
    "usecase_or_404",
]
