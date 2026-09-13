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
import re
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Credentials somebody typed into prose
# ---------------------------------------------------------------------------
#
# Everything above protects values this system was *told* about. The other half
# of the problem is the value nobody declared: a task pasted in as
#
#     Login with below credentials
#     User: nitin
#     Password : HappyLearning@123
#
# which is then the run row's `task` column, the `run_started` event, the use
# case's own description, and the first message sent to a model. No amount of
# redaction downstream helps, because nothing downstream knows that string is a
# password. It has to be recognised here, at the boundary, before the first
# write.

#: Labels that introduce a value which must not be stored in prose.
SECRET_LABELS: tuple[str, ...] = (
    "password", "passcode", "passphrase", "pwd", "pass",
    "secret", "api key", "api-key", "api_key", "apikey",
    "token", "otp", "pin", "credential", "credentials",
)

#: Labels that introduce *who* to sign in as. Not a secret on its own, and left
#: alone unless a bound credential has a slot for it -- in which case rewriting
#: it is what lets the same task run for a different account.
IDENTIFIER_LABELS: tuple[str, ...] = (
    "user", "username", "user id", "userid", "user name",
    "login", "email", "e-mail", "account",
)

#: `label` `separator` `value`, where the separator is required.
#:
#: Requiring ``:`` or ``=`` is what keeps this from firing on ordinary prose:
#: "the password field is on the right" has no separator and is not touched,
#: while "Password : hunter2" plainly is a credential. The value is one token,
#: because taking the rest of the line would swallow the sentence after it.
#: Longest label first, so "password" wins over the "pass" inside it.
_LABELS: tuple[str, ...] = tuple(
    sorted((*SECRET_LABELS, *IDENTIFIER_LABELS), key=len, reverse=True)
)

_LABELLED = re.compile(
    r"(?i)\b(?P<label>"
    + "|".join(re.escape(label) for label in _LABELS)
    + r")\b\s*[:=]\s*[\"']?(?P<value>[^\s\"']{3,200}?)[\"']?(?=\s|$)"
)


@dataclass
class Found:
    """One credential-looking value in free text."""

    label: str
    value: str

    @property
    def is_secret(self) -> bool:
        return self.label.casefold() in {label.casefold() for label in SECRET_LABELS}


@dataclass
class Scrubbed:
    """Text with credential values taken out, and what was taken."""

    text: str
    found: list[Found] = field(default_factory=list)

    @property
    def secrets(self) -> list[Found]:
        """The ones that are genuinely credentials, not just an account name."""
        return [item for item in self.found if item.is_secret]

    @property
    def labels(self) -> list[str]:
        return sorted({item.label.casefold() for item in self.secrets})


def find_credentials(text: str) -> list[Found]:
    """Credential-looking values in free text, in the order they appear."""
    if not text:
        return []
    found: list[Found] = []
    for match in _LABELLED.finditer(text):
        # Sentence punctuation is not part of the value. Stripping it matters
        # for more than tidiness: the value is what gets replaced everywhere it
        # appears, and "hunter2." does not match the "hunter2" two sentences
        # later. A password that genuinely ends in a full stop still loses its
        # first seven characters here, which is enough.
        value = match.group("value").rstrip(".,;:!?)")
        if len(value) < 3:
            continue
        found.append(Found(label=match.group("label"), value=value))
    return found


def scrub_credentials(text: str, slots: Iterable[str] = ()) -> Scrubbed:
    """``text`` with credential values replaced, and a record of what was found.

    Each value becomes ``{{secret.<slot>}}`` when a bound credential has a slot
    that plainly matches the label, and the placeholder otherwise. The
    substitution is the useful case and not a nicety: the task then reads
    "Password : {{secret.password}}", which is exactly the literal the recorder
    is instructed to type and the tool layer substitutes at the moment of
    typing. So the task keeps working, the value is never written anywhere, and
    the same task runs for a different account tomorrow.

    Nothing is guessed about which slot: only a name containing the label, or a
    label containing the name, is accepted.
    """
    found = find_credentials(text)
    if not found:
        return Scrubbed(text=text)

    known = [slot for slot in slots if slot]
    out = text
    for item in found:
        slot = _slot_for(item.label, known)
        replacement = f"{{{{secret.{slot}}}}}" if slot else PLACEHOLDER
        out = out.replace(item.value, replacement)
    return Scrubbed(text=out, found=found)


def _slot_for(label: str, slots: list[str]) -> str:
    """The bound slot this label refers to, or "".

    Substring either way, deliberately: a label of "password" should find a
    slot called "password" or "login_password", and a slot called "user" should
    be found by a label of "user name".
    """
    wanted = re.sub(r"[^a-z0-9]", "", label.casefold())
    if not wanted:
        return ""
    for slot in slots:
        plain = re.sub(r"[^a-z0-9]", "", slot.casefold())
        if plain and (plain in wanted or wanted in plain):
            return slot
    return ""
