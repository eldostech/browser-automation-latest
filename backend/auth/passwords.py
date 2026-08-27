"""Password hashing and session-token minting.

bcrypt directly rather than passlib: passlib has been unmaintained since 2020
and its bcrypt backend breaks on each new bcrypt release, which is a poor
dependency to put underneath the login path. The API needed here is two
functions wide.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

import bcrypt

#: bcrypt truncates silently at 72 *bytes*, so a longer passphrase would have
#: its tail ignored -- and two passwords sharing a 72-byte prefix would be
#: interchangeable. Refuse instead of quietly weakening the password.
MAX_PASSWORD_BYTES = 72

MIN_PASSWORD_LENGTH = 12

#: 32 bytes from the OS CSPRNG, base64url-encoded. Long enough that online
#: guessing is hopeless and offline guessing has nothing to chew on.
TOKEN_BYTES = 32


class PasswordError(ValueError):
    """A password that cannot be used, with a reason safe to show the user."""


def validate_password(password: str) -> None:
    """Raise :class:`PasswordError` if this password cannot be stored safely.

    Length only. Composition rules ("one number, one symbol") shrink the search
    space more often than they enlarge it, and NIST has recommended against
    them since SP 800-63B.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise PasswordError(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes "
            "(bcrypt ignores anything beyond that)."
        )


def hash_password(password: str, *, rounds: int = 12) -> str:
    validate_password(password)
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds)).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time check. Never raises -- a malformed stored hash is a
    failed login, not a 500 that tells an attacker the account is special."""
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False


def generate_password(length: int = 20) -> str:
    """A password for the bootstrap admin when the operator supplied none."""
    return secrets.token_urlsafe(length)[:length]


def new_session_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(token: str) -> str:
    """SHA-256 of a session token, hex.

    Plain SHA-256 rather than bcrypt, deliberately, and the reason is the
    opposite of the one for passwords: this input is 256 bits of CSPRNG output,
    so there is no dictionary to attack and no work factor worth paying on
    every single authenticated request. What the hash buys is that a database
    leak yields no usable sessions.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)


__all__ = [
    "MIN_PASSWORD_LENGTH",
    "PasswordError",
    "generate_password",
    "hash_password",
    "hash_token",
    "new_session_token",
    "tokens_equal",
    "validate_password",
    "verify_password",
]
