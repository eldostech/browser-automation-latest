"""Identity and authorization.

``rbac`` decides what a role may do and knows nothing about how anyone logged
in; ``service`` establishes who the caller is and knows nothing about what they
may then do. Keeping those two halves ignorant of each other is what lets an
OIDC provider replace the second without touching the first.
"""

from auth.passwords import (
    MIN_PASSWORD_LENGTH,
    PasswordError,
    generate_password,
    hash_password,
    verify_password,
)
from auth.rbac import Permission, ROLE_PERMISSIONS, Role, has_permission, permissions_for
from auth.service import AuthError, AuthService, Principal

__all__ = [
    "AuthError",
    "AuthService",
    "MIN_PASSWORD_LENGTH",
    "Permission",
    "PasswordError",
    "Principal",
    "ROLE_PERMISSIONS",
    "Role",
    "generate_password",
    "has_permission",
    "hash_password",
    "permissions_for",
    "verify_password",
]
