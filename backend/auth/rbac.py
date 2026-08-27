"""Roles and permissions.

This module is deliberately free of FastAPI, SQLAlchemy and HTTP. It answers
one question -- *may this role do this thing* -- as data plus a pure function,
which means the policy can be unit-tested exhaustively and read in one sitting
by someone deciding whether to grant an account a role.

The split from authentication is the important part. There is no SSO here yet,
so ``auth/service.py`` checks passwords; when an OIDC provider replaces it,
this file does not change. Identity is pluggable, authority is not.

**Permissions are additive and coarse.** Every attempt at fine-grained
per-resource ACLs in a system this size ends as a permission table nobody can
reason about. Ownership is handled separately and structurally: the workspace
filter in the Store means a query cannot return another tenant's row at all,
so permissions only decide what you may do with rows you can already see.
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Ordered least to most privileged, though the mapping below is what counts."""

    VIEWER = "viewer"
    OPERATOR = "operator"
    AUTHOR = "author"
    ADMIN = "admin"


class Permission(StrEnum):
    # -- runs (the recording agent) -----------------------------------------
    RUN_READ = "run:read"
    RUN_CREATE = "run:create"
    RUN_APPROVE = "run:approve"
    RUN_CANCEL = "run:cancel"

    # -- use cases ----------------------------------------------------------
    USECASE_READ = "usecase:read"
    USECASE_CREATE = "usecase:create"
    USECASE_PUBLISH = "usecase:publish"
    USECASE_DELETE = "usecase:delete"
    USECASE_REPAIR = "usecase:repair"

    # -- execution ----------------------------------------------------------
    BATCH_READ = "batch:read"
    BATCH_CREATE = "batch:create"

    # -- credentials --------------------------------------------------------
    #: Listing names and slots. There is no permission to *read* a secret,
    #: because no code path returns one.
    CREDENTIAL_READ = "credential:read"
    CREDENTIAL_WRITE = "credential:write"
    CREDENTIAL_DELETE = "credential:delete"

    #: Turning on script steps for a use case. Its own permission rather than
    #: part of USECASE_PUBLISH: a script step is arbitrary JavaScript running
    #: in a session that may hold someone else's credentials, so the authority
    #: to write a use case and the authority to let it run code are different
    #: authorities.
    SCRIPT_ENABLE = "script:enable"

    # -- administration -----------------------------------------------------
    USER_MANAGE = "user:manage"
    AUDIT_READ = "audit:read"


#: Read-only access to everything the workspace holds.
_VIEWER: frozenset[Permission] = frozenset(
    {
        Permission.RUN_READ,
        Permission.USECASE_READ,
        Permission.BATCH_READ,
        Permission.CREDENTIAL_READ,
    }
)

#: Can execute published work, and approve a step mid-run -- but cannot change
#: what a use case *is*. This is the role for the person feeding a thousand
#: rows through a process somebody else designed.
_OPERATOR: frozenset[Permission] = _VIEWER | {
    Permission.RUN_CREATE,
    Permission.RUN_APPROVE,
    Permission.RUN_CANCEL,
    Permission.BATCH_CREATE,
    Permission.CREDENTIAL_WRITE,
}

#: Can design, publish and repair use cases.
_AUTHOR: frozenset[Permission] = _OPERATOR | {
    Permission.USECASE_CREATE,
    Permission.USECASE_PUBLISH,
    Permission.USECASE_DELETE,
    Permission.USECASE_REPAIR,
    Permission.CREDENTIAL_DELETE,
}

#: Everything, plus the two authorities nobody else gets: managing accounts,
#: and enabling script execution.
_ADMIN: frozenset[Permission] = (
    _AUTHOR
    | {
        Permission.SCRIPT_ENABLE,
        Permission.USER_MANAGE,
        Permission.AUDIT_READ,
    }
)

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.VIEWER: _VIEWER,
    Role.OPERATOR: _OPERATOR,
    Role.AUTHOR: _AUTHOR,
    Role.ADMIN: _ADMIN,
}


def permissions_for(role: str | Role) -> frozenset[Permission]:
    """What a role may do. An unknown role gets nothing.

    Failing closed matters here: a typo in a database row, or a role removed in
    a later version, must not grant access. It also means a downgrade that
    drops a role leaves those accounts locked out rather than elevated.
    """
    try:
        return ROLE_PERMISSIONS[Role(role)]
    except ValueError:
        return frozenset()


def has_permission(role: str | Role, permission: Permission) -> bool:
    return permission in permissions_for(role)


def is_valid_role(role: str) -> bool:
    return role in {r.value for r in Role}


__all__ = [
    "Permission",
    "ROLE_PERMISSIONS",
    "Role",
    "has_permission",
    "is_valid_role",
    "permissions_for",
]
