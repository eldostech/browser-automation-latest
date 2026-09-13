"""Encrypted storage for the credentials a use case signs in with.

Design decisions worth stating, because each one is a place this could have
been quietly weaker:

**No plaintext fallback.** With ``CREDENTIALS_KEY`` unset the vault refuses to
store anything rather than falling back to writing values in the clear. A
feature that half-works by writing passwords to disk is worse than a feature
that tells you to configure a key.

**Write-only over HTTP.** Values go in and are never returned. The API can list
which slots a credential defines and when it was last used; it cannot read a
value back. Nothing in the dashboard needs to, and an endpoint that could would
be the obvious thing to attack.

**Decrypted only into a local variable.** :meth:`Vault.resolve` hands values
straight to the executor, which renders them into a tool argument and drops
them. They are registered with the run's :class:`redaction.Redactor` first, so
even if a value reaches an event it is rewritten before persistence.

Fernet is AES-128-CBC with an HMAC and a timestamp, from ``cryptography``. It
is authenticated, so a tampered ciphertext fails loudly instead of decrypting
to garbage.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class VaultUnavailable(RuntimeError):
    """No encryption key is configured, so credentials cannot be stored."""


class VaultError(RuntimeError):
    """A stored credential could not be decrypted."""


NO_KEY_MESSAGE = (
    "Credential storage is disabled because CREDENTIALS_KEY is not set. "
    "Generate one with:\n"
    "    python -c \"from cryptography.fernet import Fernet; "
    'print(Fernet.generate_key().decode())"\n'
    "and put it in .env as CREDENTIALS_KEY. Losing the key makes existing "
    "stored credentials unreadable, so keep it with your other secrets."
)


def generate_key() -> str:
    """A fresh Fernet key, for the setup instructions and the tests."""
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Vault:
    """Encrypts and decrypts credential bundles.

    A bundle is ``{slot_name: value}`` -- the slots a use case's ``secrets``
    declare. One credential record therefore covers a whole login rather than a
    single field, which is what lets a batch bind "the IXL account" in one go.
    """

    __slots__ = ("_fernet",)

    def __init__(self, key: str | None) -> None:
        if not key:
            self._fernet = None
            return
        from cryptography.fernet import Fernet

        try:
            self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
        except (ValueError, TypeError) as exc:
            raise VaultUnavailable(
                f"CREDENTIALS_KEY is not a valid Fernet key ({exc}). {NO_KEY_MESSAGE}"
            ) from exc

    @property
    def available(self) -> bool:
        return self._fernet is not None

    def _require(self):
        if self._fernet is None:
            raise VaultUnavailable(NO_KEY_MESSAGE)
        return self._fernet

    def seal(self, values: dict[str, str]) -> bytes:
        """Encrypt a ``{slot: value}`` bundle."""
        fernet = self._require()
        cleaned = {str(k): str(v) for k, v in values.items() if v not in (None, "")}
        if not cleaned:
            raise ValueError("a credential must define at least one non-empty slot")
        return fernet.encrypt(json.dumps(cleaned).encode("utf-8"))

    def open(self, ciphertext: bytes) -> dict[str, str]:
        """Decrypt a bundle. Raises rather than returning a partial result."""
        fernet = self._require()
        from cryptography.fernet import InvalidToken

        try:
            return json.loads(fernet.decrypt(ciphertext).decode("utf-8"))
        except InvalidToken as exc:
            raise VaultError(
                "A stored credential could not be decrypted. This almost always means "
                "CREDENTIALS_KEY has changed since it was saved; re-enter the credential."
            ) from exc
        except (ValueError, UnicodeDecodeError) as exc:
            raise VaultError(f"Stored credential is corrupt: {exc}") from exc

    @staticmethod
    def slots_of(values: dict[str, str]) -> list[str]:
        return sorted(values)


def new_credential_id() -> str:
    return uuid.uuid4().hex

