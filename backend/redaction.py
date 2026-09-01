"""Keeps secret values out of the event log, the database and the logs.

Why this exists as its own layer rather than "be careful at the call site":
credentials reach an event through at least five paths -- the task text a
person pasted, a ``browser_fill_form`` argument, a tool result echoing the
value back, the model's own prose repeating it, and an error message quoting
the failing call. Auditing five paths forever is a losing game. One pass over
every event on its way out is not.

The pass is uniform by construction: an event is dumped to a plain dict, every
string anywhere in it is rewritten, and the dict is re-validated back into an
event. A new event type, or a new field on an existing one, is covered the day
it is added, with no change here.

Scope and honesty about it
--------------------------
This redacts **text**. It cannot redact a screenshot -- an artifact is a PNG,
and a password typed into a visible field will be in it. Use cases that carry
secrets should run with ``screenshot_every_step`` off; the code cannot enforce
that for you.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from events import AgentEvent, dump_event, parse_event

log = logging.getLogger(__name__)

#: What a redacted value is replaced with. Deliberately obvious in a timeline:
#: a run that shows this is working correctly, not malfunctioning.
PLACEHOLDER = "«redacted»"

#: Values shorter than this are not redacted. A two-character "secret" appears
#: inside ordinary words and redacting it would shred every event in the run
#: while protecting nothing worth protecting.
MIN_SECRET_LENGTH = 4


class Redactor:
    """Rewrites known secret values out of arbitrary data.

    Immutable in spirit -- :meth:`add` exists because secrets are bound to a
    run at different moments (some from the vault at start, some discovered in
    the task text), but the set only ever grows within one run.
    """

    __slots__ = ("_secrets",)

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets: list[str] = []
        for secret in secrets:
            self.add(secret)

    def add(self, value: str | None) -> None:
        """Register a value. Short, blank and duplicate values are ignored."""
        if not value or not isinstance(value, str):
            return
        if len(value) < MIN_SECRET_LENGTH or value in self._secrets:
            return
        self._secrets.append(value)
        # Longest first, so a secret that contains another does not leave the
        # shorter one's tail behind as a readable fragment.
        self._secrets.sort(key=len, reverse=True)

    def update(self, values: Iterable[str | None]) -> None:
        for value in values:
            self.add(value)

    @property
    def active(self) -> bool:
        return bool(self._secrets)

    def __len__(self) -> int:
        return len(self._secrets)

    # -- application --------------------------------------------------------
    def text(self, value: str) -> str:
        """Redact every registered secret out of one string."""
        if not self._secrets or not value:
            return value
        for secret in self._secrets:
            if secret in value:
                value = value.replace(secret, PLACEHOLDER)
        return value

    def structure(self, node: Any) -> Any:
        """Deep-copy ``node``, redacting every string it contains.

        Dictionary *keys* are rewritten too. A secret is far more likely to be
        a value, but nothing stops a form field from being keyed by one.
        """
        if not self._secrets:
            return node
        if isinstance(node, str):
            return self.text(node)
        if isinstance(node, dict):
            return {self.structure(k): self.structure(v) for k, v in node.items()}
        if isinstance(node, list):
            return [self.structure(item) for item in node]
        if isinstance(node, tuple):
            return tuple(self.structure(item) for item in node)
        return node

    def event(self, event: AgentEvent) -> AgentEvent:
        """Redact an event wherever a secret might be hiding in it.

        Falls back to the original event if the round trip fails for any
        reason. Losing an event to a redaction bug would be a worse outcome
        than the one this guards against, and the failure is logged loudly.
        """
        if not self._secrets:
            return event
        try:
            return parse_event(self.structure(dump_event(event)))
        except Exception as exc:  # noqa: BLE001 - redaction must never drop an event
            log.error(
                "event redaction failed; emitting unredacted",
                extra={"event_type": getattr(event, "type", "?"), "error": str(exc)},
            )
            return event


#: A redactor with nothing registered. Every method is a pass-through, so code
#: paths with no secrets pay nothing and need no ``if redactor is not None``.
NULL_REDACTOR = Redactor()
