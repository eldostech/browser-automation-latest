"""Credentials held between a recording and the decision to keep it.

A recording that signs in needs its credentials again at exactly one later
moment: when the user says "save this as a use case" and asks for the login to
be saved with it. Between those two points the values have to live somewhere.

Nowhere on disk is the requirement. So they live here: a dict in the backend
process, with an expiry, and nothing else. The consequences are deliberate and
worth stating plainly, because they are the price of not writing a password
down:

* A backend restart loses them. The use case can still be saved; its credential
  slots are known from the recording, and the operator binds a stored
  credential to it later.
* A second worker cannot see them. Saving must happen on the process that ran
  the recording -- which it does, because the user is looking at that run.
* They expire. A credential typed an hour ago and never saved is gone, rather
  than sitting in memory until the process ends.

The alternative -- sealing them into the vault immediately and deleting them if
unsaved -- was rejected because it makes the common case (a recording the user
throws away) write a credential to durable storage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: How long an unsaved credential survives after its run finishes. Long enough
#: to review a recording and decide; short enough that walking away discards it.
DEFAULT_TTL_SECONDS = 60 * 30


@dataclass(slots=True)
class _Entry:
    values: dict[str, str]
    expires_at: float
    workspace_id: str


@dataclass
class SecretStash:
    """Per-run credential values, in memory, with a TTL.

    Not a cache: there is no backing store to fall back to, and a miss is a
    final answer rather than a reason to go and look somewhere else.
    """

    ttl_seconds: float = DEFAULT_TTL_SECONDS
    _entries: dict[str, _Entry] = field(default_factory=dict, repr=False)

    def put(self, run_id: str, values: dict[str, str], *, workspace_id: str) -> None:
        """Hold a run's credentials. Storing nothing stores nothing."""
        if not values:
            return
        self._sweep()
        self._entries[run_id] = _Entry(
            values=dict(values),
            expires_at=time.monotonic() + self.ttl_seconds,
            workspace_id=workspace_id,
        )
        log.debug("held credentials for a run", extra={"run_id": run_id, "slots": len(values)})

    def take(self, run_id: str, *, workspace_id: str) -> dict[str, str] | None:
        """Read and remove a run's credentials.

        Removing on read is the point: this is called when the values are being
        sealed into the vault, and holding them any longer than that serves
        nothing. The workspace check means a run id from another tenant cannot
        be used to pull credentials out of this process.
        """
        self._sweep()
        entry = self._entries.get(run_id)
        if entry is None or entry.workspace_id != workspace_id:
            return None
        del self._entries[run_id]
        return entry.values

    def discard(self, run_id: str) -> None:
        """Forget a run's credentials. Safe to call when there are none."""
        if self._entries.pop(run_id, None) is not None:
            log.debug("discarded held credentials", extra={"run_id": run_id})

    def slots(self, run_id: str, *, workspace_id: str) -> list[str]:
        """Which slots are still held, so the UI can offer to save them.

        Names only. There is no method on this class that returns a value
        without also removing it.
        """
        self._sweep()
        entry = self._entries.get(run_id)
        if entry is None or entry.workspace_id != workspace_id:
            return []
        return sorted(entry.values)

    def _sweep(self) -> None:
        now = time.monotonic()
        expired = [run_id for run_id, e in self._entries.items() if e.expires_at <= now]
        for run_id in expired:
            del self._entries[run_id]
        if expired:
            log.debug("expired held credentials", extra={"count": len(expired)})

    def __len__(self) -> int:
        self._sweep()
        return len(self._entries)


__all__ = ["DEFAULT_TTL_SECONDS", "SecretStash"]
