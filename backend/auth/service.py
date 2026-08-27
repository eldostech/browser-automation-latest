"""Accounts, logins and sessions.

The whole of "who is this" lives here. Everything downstream consumes a
:class:`Principal` and never learns how it was obtained -- which is what makes
adding OIDC later an addition rather than a rewrite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from auth.passwords import (
    PasswordError,
    generate_password,
    hash_password,
    hash_token,
    new_session_token,
    verify_password,
)
from auth.rbac import Permission, Role, is_valid_role, permissions_for
from config import Settings
from db.base import iso, utcnow
from db.models import User, UserSession, Workspace

log = logging.getLogger(__name__)

#: Verified when the email is unknown, so that a login attempt costs the same
#: whether or not the account exists. Without this, response time answers "is
#: this address registered?" for anyone who asks. The value is a real bcrypt
#: hash of a random string; nothing can match it.
_DUMMY_HASH = "$2b$12$C6UzMDM.H6dfI/f/IKcEe.PjWZKMEMMbaBlF/9BLTKzD.YNa1PVSy"


class AuthError(Exception):
    """Authentication or account management failed, with a user-safe message."""


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller.

    Immutable, and carries its permission set already resolved, so an
    authorization check is a frozenset lookup rather than a database round trip
    on every request.
    """

    user_id: str
    workspace_id: str
    email: str
    display_name: str
    role: str
    permissions: frozenset[Permission] = field(default_factory=frozenset)

    def can(self, permission: Permission) -> bool:
        return permission in self.permissions

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.user_id,
            "workspace_id": self.workspace_id,
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role,
            "permissions": sorted(p.value for p in self.permissions),
        }


def _principal_from(user: User) -> Principal:
    return Principal(
        user_id=user.id,
        workspace_id=user.workspace_id,
        email=user.email,
        display_name=user.display_name,
        role=user.role,
        permissions=permissions_for(user.role),
    )


def _user_dict(user: User) -> dict[str, Any]:
    """A user as the API returns them. ``password_hash`` is structurally absent
    rather than filtered, so a future field cannot leak it by omission."""
    return {
        "id": user.id,
        "workspace_id": user.workspace_id,
        "email": user.email,
        "display_name": user.display_name,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": iso(user.created_at),
        "last_login_at": iso(user.last_login_at),
    }


class AuthService:
    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], settings: Settings
    ) -> None:
        self._sessions = session_factory
        self._settings = settings

    # -- bootstrap ----------------------------------------------------------
    async def bootstrap(self) -> str | None:
        """Ensure the deployment has a workspace and at least one admin.

        Runs on every startup and does nothing once any user exists, so it is
        safe to leave enabled. Returns a generated password exactly once, on
        the boot that created the account, so the caller can print it.

        The "no users at all" condition is what makes this safe: it cannot
        resurrect a deliberately deleted admin, and it cannot reset a password.
        """
        async with self._sessions() as session:
            existing = await session.scalar(select(func.count()).select_from(User))
            if existing:
                return None

            workspace = await session.scalar(select(Workspace).limit(1))
            if workspace is None:
                name = self._settings.bootstrap_workspace_name
                workspace = Workspace(name=name, slug=_slugify(name))
                session.add(workspace)
                await session.flush()

            password = self._settings.bootstrap_admin_password
            generated = not password
            if generated:
                password = generate_password()

            user = User(
                workspace_id=workspace.id,
                email=self._settings.bootstrap_admin_email.strip().lower(),
                display_name="Administrator",
                password_hash=hash_password(password, rounds=self._settings.auth_bcrypt_rounds),
                role=Role.ADMIN.value,
            )
            session.add(user)
            await session.commit()

            log.warning(
                "created bootstrap administrator",
                extra={"user_email": user.email, "generated_password": generated},
            )
            return password if generated else None

    # -- login --------------------------------------------------------------
    async def authenticate(
        self, email: str, password: str, *, user_agent: str = ""
    ) -> tuple[str, Principal]:
        """Exchange an email and password for a session token.

        Raises :class:`AuthError` with one deliberately vague message for every
        failure mode -- unknown address, wrong password, disabled account. The
        caller does not get to learn which.
        """
        normalized = email.strip().lower()
        async with self._sessions() as session:
            user = await session.scalar(select(User).where(User.email == normalized))

            # Spend the same time on a miss as on a hit.
            stored = user.password_hash if user is not None else _DUMMY_HASH
            ok = verify_password(password, stored)

            if user is None or not ok or not user.is_active:
                log.info(
                    "failed login",
                    extra={"user_email": normalized, "reason": _why(user, ok)},
                )
                raise AuthError("Incorrect email or password.")

            token = new_session_token()
            expires = utcnow() + timedelta(hours=self._settings.auth_session_ttl_hours)
            session.add(
                UserSession(
                    user_id=user.id,
                    token_hash=hash_token(token),
                    expires_at=expires,
                    user_agent=user_agent[:300],
                )
            )
            user.last_login_at = utcnow()
            await session.commit()

            log.info("login", extra={"user_id": user.id, "user_email": user.email})
            return token, _principal_from(user)

    async def resolve(self, token: str) -> Principal | None:
        """The principal behind a bearer token, or None.

        Called on every authenticated request, so it is one indexed query
        joining the session to its user. Four conditions can invalidate a
        session, and all four are checked here rather than trusted from the
        token: expiry, explicit revocation, account deactivation, and a
        password change after the session was issued.
        """
        if not token:
            return None

        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(UserSession, User)
                    .join(User, User.id == UserSession.user_id)
                    .where(UserSession.token_hash == hash_token(token))
                )
            ).first()
            if row is None:
                return None

            user_session, user = row
            now = utcnow()
            if user_session.expires_at <= now or user_session.revoked_at is not None:
                return None
            if not user.is_active:
                return None
            if (
                user.credentials_changed_at is not None
                and user_session.created_at < user.credentials_changed_at
            ):
                return None

            return _principal_from(user)

    async def logout(self, token: str) -> bool:
        async with self._sessions() as session:
            result = await session.execute(
                update(UserSession)
                .where(UserSession.token_hash == hash_token(token))
                .where(UserSession.revoked_at.is_(None))
                .values(revoked_at=utcnow())
            )
            await session.commit()
            return bool(result.rowcount)

    async def purge_expired_sessions(self) -> int:
        """Housekeeping. Expired rows are already refused by ``resolve``; this
        only stops the table growing without bound."""
        async with self._sessions() as session:
            result = await session.execute(
                delete(UserSession).where(UserSession.expires_at <= utcnow())
            )
            await session.commit()
            return int(result.rowcount or 0)

    # -- account management -------------------------------------------------
    async def create_user(
        self,
        *,
        workspace_id: str,
        email: str,
        password: str,
        role: str,
        display_name: str = "",
    ) -> dict[str, Any]:
        if not is_valid_role(role):
            raise AuthError(f"Unknown role {role!r}.")
        normalized = email.strip().lower()
        if not normalized:
            raise AuthError("An email address is required.")

        try:
            password_hash = hash_password(password, rounds=self._settings.auth_bcrypt_rounds)
        except PasswordError as exc:
            raise AuthError(str(exc)) from exc

        async with self._sessions() as session:
            user = User(
                workspace_id=workspace_id,
                email=normalized,
                display_name=display_name or normalized.split("@")[0],
                password_hash=password_hash,
                role=role,
            )
            session.add(user)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                # The unique constraint is the authority on this, not a
                # prior SELECT -- two simultaneous signups would both pass a
                # check-then-insert.
                raise AuthError(f"An account already exists for {normalized}.") from exc
            return _user_dict(user)

    async def list_users(self, workspace_id: str) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            users = (
                await session.scalars(
                    select(User)
                    .where(User.workspace_id == workspace_id)
                    .order_by(User.created_at)
                )
            ).all()
            return [_user_dict(u) for u in users]

    async def set_role(self, workspace_id: str, user_id: str, role: str) -> dict[str, Any] | None:
        if not is_valid_role(role):
            raise AuthError(f"Unknown role {role!r}.")
        async with self._sessions() as session:
            user = await self._scoped_user(session, workspace_id, user_id)
            if user is None:
                return None
            if user.role == Role.ADMIN.value and role != Role.ADMIN.value:
                await self._refuse_last_admin(session, workspace_id, user_id)
            user.role = role
            await session.commit()
            return _user_dict(user)

    async def set_active(
        self, workspace_id: str, user_id: str, is_active: bool
    ) -> dict[str, Any] | None:
        async with self._sessions() as session:
            user = await self._scoped_user(session, workspace_id, user_id)
            if user is None:
                return None
            if not is_active:
                if user.role == Role.ADMIN.value:
                    await self._refuse_last_admin(session, workspace_id, user_id)
                # Deactivation has to end live sessions, or the account keeps
                # working until its token happens to expire.
                await session.execute(
                    update(UserSession)
                    .where(UserSession.user_id == user_id)
                    .where(UserSession.revoked_at.is_(None))
                    .values(revoked_at=utcnow())
                )
            user.is_active = is_active
            await session.commit()
            return _user_dict(user)

    async def change_password(
        self, user_id: str, *, current_password: str, new_password: str
    ) -> None:
        async with self._sessions() as session:
            user = await session.get(User, user_id)
            if user is None or not verify_password(current_password, user.password_hash):
                raise AuthError("Current password is incorrect.")
            try:
                user.password_hash = hash_password(
                    new_password, rounds=self._settings.auth_bcrypt_rounds
                )
            except PasswordError as exc:
                raise AuthError(str(exc)) from exc
            # Every session issued before this moment stops working, including
            # any an attacker holds. This is the reason the column exists.
            user.credentials_changed_at = utcnow()
            await session.commit()

    async def reset_password(
        self, workspace_id: str, user_id: str, new_password: str
    ) -> None:
        """Admin-initiated reset. No current password required; same session
        invalidation applies."""
        async with self._sessions() as session:
            user = await self._scoped_user(session, workspace_id, user_id)
            if user is None:
                raise AuthError("No such user.")
            try:
                user.password_hash = hash_password(
                    new_password, rounds=self._settings.auth_bcrypt_rounds
                )
            except PasswordError as exc:
                raise AuthError(str(exc)) from exc
            user.credentials_changed_at = utcnow()
            await session.commit()

    # -- internals ----------------------------------------------------------
    async def _scoped_user(
        self, session: AsyncSession, workspace_id: str, user_id: str
    ) -> User | None:
        """Fetch a user only if they belong to this workspace.

        Every admin operation goes through here, so a workspace admin cannot
        act on an account in another tenant by guessing its id.
        """
        return await session.scalar(
            select(User).where(User.id == user_id, User.workspace_id == workspace_id)
        )

    async def _refuse_last_admin(
        self, session: AsyncSession, workspace_id: str, user_id: str
    ) -> None:
        """Stop a workspace locking itself out.

        Demoting or disabling the only administrator leaves nobody who can
        manage accounts -- recoverable only by editing the database by hand.
        """
        remaining = await session.scalar(
            select(func.count())
            .select_from(User)
            .where(
                User.workspace_id == workspace_id,
                User.role == Role.ADMIN.value,
                User.is_active.is_(True),
                User.id != user_id,
            )
        )
        if not remaining:
            raise AuthError(
                "This is the only active administrator; promote another account first."
            )


def _slugify(name: str) -> str:
    cleaned = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned or "workspace"


def _why(user: User | None, password_ok: bool) -> str:
    """Reason for the log only. The HTTP response never distinguishes these."""
    if user is None:
        return "unknown-email"
    if not password_ok:
        return "bad-password"
    if not user.is_active:
        return "inactive"
    return "unknown"


__all__ = ["AuthError", "AuthService", "Principal"]
