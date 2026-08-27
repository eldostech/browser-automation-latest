"""Login, logout, and the caller's own account.

The only unauthenticated endpoints in the application are ``POST /login`` and
the health probe. Everything else requires a bearer token.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from auth.service import AuthError, AuthService
from config import Settings
from deps import CurrentUser, get_auth, get_config
from routers.schemas import ChangePasswordRequest, LoginRequest, LoginResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    request: Request,
    auth: Annotated[AuthService, Depends(get_auth)],
    settings: Annotated[Settings, Depends(get_config)],
) -> LoginResponse:
    """Exchange an email and password for a session token.

    Every failure returns the same 401 with the same message. Distinguishing
    "no such account" from "wrong password" would turn this endpoint into a
    tool for discovering who has an account here.
    """
    try:
        token, principal = await auth.authenticate(
            body.email,
            body.password,
            user_agent=request.headers.get("user-agent", ""),
        )
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return LoginResponse(
        access_token=token,
        expires_in=settings.auth_session_ttl_hours * 3600,
        user=principal.to_dict(),
    )


@router.post("/logout")
async def logout(
    request: Request,
    principal: CurrentUser,
    auth: Annotated[AuthService, Depends(get_auth)],
) -> dict[str, Any]:
    """Revoke the token this request was made with.

    Revocation takes effect on the next request rather than at token expiry,
    which is the whole reason sessions are opaque rows instead of JWTs.
    """
    header = request.headers.get("authorization", "")
    token = header[7:] if header.lower().startswith("bearer ") else ""
    return {"logged_out": await auth.logout(token)}


@router.get("/me")
async def me(principal: CurrentUser) -> dict[str, Any]:
    """Who the caller is and what they may do.

    The frontend uses ``permissions`` to decide which controls to render. That
    is a convenience, not a security boundary -- every one of those actions is
    checked again on the server.
    """
    return principal.to_dict()


@router.post("/password")
async def change_password(
    body: ChangePasswordRequest,
    principal: CurrentUser,
    auth: Annotated[AuthService, Depends(get_auth)],
) -> dict[str, Any]:
    """Change your own password, which signs out every other session."""
    try:
        await auth.change_password(
            principal.user_id,
            current_password=body.current_password,
            new_password=body.new_password,
        )
    except AuthError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"changed": True, "note": "All other sessions have been signed out."}
