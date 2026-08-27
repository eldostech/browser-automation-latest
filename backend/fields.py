"""Declared fields: the values a recording uses, named before it starts.

The problem this solves
-----------------------
A task used to be one block of prose with the values embedded in it::

    Go to https://example.com/contact/ and enter Full Name: Nitin Asati,
    Work Email: nitin@example.com, password hunter2

Three things go wrong with that, and all three are the same mistake -- the
values are mixed in with the instruction, so nothing downstream can tell them
apart:

* **The password is in clear text** in the task, which is stored on the run,
  shown in the timeline, and echoed back by any tool that repeats its argument.
* **Nothing knows which words are parameters.** At distillation time the model
  is shown a list of literals the recording typed and asked to guess which ones
  vary per row. "Nitin Asati" is a good guess; "Submit" is not; a product code
  is anyone's guess. Guessing is the wrong mechanism for something the user
  already knows.
* **The user finds out what the parameters are afterwards**, from whatever the
  model decided, rather than declaring them.

Separating the two fixes all three. The prose says *what to do*; the fields say
*what to do it with*. A field marked secret never appears in clear text
anywhere, and a field not marked secret becomes a use case input with the name
the user gave it -- deterministically, by exact value match, with no model
judgement involved.

Why matching on the value is sound
----------------------------------
The user supplies both the name and the value, so at distillation time we are
not inferring that "Nitin Asati" *might* be a parameter -- we are looking up a
string we were told to look for. The only way to get a false positive is for a
declared value to coincide with an unrelated literal on the page, which is why
:data:`MIN_MATCH_LENGTH` exists: a field whose value is ``1`` or ``ok`` would
match half the recording, so short values are matched only where the surrounding
argument makes them unambiguous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Iterable

#: A field name has to survive being written as ``{{input.name}}`` in a use
#: case template and as a CSV column header, so it is an identifier.
NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Below this length, a declared value is not substituted into recorded step
#: arguments by exact match -- "1" or "ok" would match text that has nothing to
#: do with the field. Such fields still become declared inputs; they simply
#: keep their recorded literal unless the argument matches the value exactly
#: and entirely.
MIN_MATCH_LENGTH = 4


def secret_placeholder(slot: str) -> str:
    """What a secret's value is replaced with in the event log.

    Labelled with the slot rather than a bare marker, which is what makes the
    round trip work: the timeline shows «secret:password» so a reader knows
    *which* credential was used, and distillation turns that same token into
    ``{{secret.password}}`` without ever seeing the value. The plaintext does
    not need to survive the run for the use case to be parameterised.
    """
    return f"«secret:{slot}»"


#: Matches what the function above produces, for the reverse direction.
SECRET_PLACEHOLDER_RE = re.compile(r"«secret:([A-Za-z_][A-Za-z0-9_]*)»")


@dataclass(slots=True)
class DeclaredField:
    """One named value the recording will use.

    ``value`` is what to type during *this* recording. For a secret it is held
    only in memory, for the length of the run; see ``SecretStash``.
    """

    name: str
    value: str
    secret: bool = False
    description: str = ""
    #: Free-form hint carried into the use case's input spec, so a later
    #: operator filling a CSV knows what the column means.
    example: str = ""

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        if not NAME_PATTERN.match(self.name):
            raise ValueError(
                f"{self.name!r} is not a usable field name. Use letters, digits and "
                "underscores, starting with a letter -- it becomes a column header "
                "and a template name."
            )

    @property
    def matchable(self) -> bool:
        """Whether this value is distinctive enough to substitute by search."""
        return len(self.value) >= MIN_MATCH_LENGTH

    @property
    def template(self) -> str:
        """How a use case refers to this field."""
        kind = "secret" if self.secret else "input"
        return f"{{{{{kind}.{self.name}}}}}"


@dataclass(slots=True)
class FieldSet:
    """Every field declared for one run, split by kind."""

    fields: list[DeclaredField] = dataclass_field(default_factory=list)

    @classmethod
    def from_payload(cls, raw: Iterable[dict[str, Any]] | None) -> "FieldSet":
        seen: set[str] = set()
        declared: list[DeclaredField] = []
        for item in raw or []:
            entry = DeclaredField(
                name=str(item.get("name", "")),
                value=str(item.get("value", "")),
                secret=bool(item.get("secret", False)),
                description=str(item.get("description", "") or ""),
                example=str(item.get("example", "") or ""),
            )
            if entry.name in seen:
                raise ValueError(f"{entry.name!r} is declared twice.")
            seen.add(entry.name)
            declared.append(entry)
        return cls(declared)

    # -- views --------------------------------------------------------------
    @property
    def inputs(self) -> list[DeclaredField]:
        return [f for f in self.fields if not f.secret]

    @property
    def secrets(self) -> list[DeclaredField]:
        return [f for f in self.fields if f.secret]

    @property
    def secret_values(self) -> dict[str, str]:
        """``{slot: value}``. In memory only -- never written anywhere."""
        return {f.name: f.value for f in self.secrets}

    def __bool__(self) -> bool:
        return bool(self.fields)

    # -- what is safe to persist -------------------------------------------
    def persistable(self) -> dict[str, Any]:
        """The part of this that may be stored on the run.

        Input names *and values* -- they are ordinary data, and distillation
        needs the values to find them in the recorded arguments. Secret
        names only: the slot is useful (it names the credential the use case
        will need) and the value must not be written down.
        """
        return {
            "inputs": {f.name: f.value for f in self.inputs},
            "input_meta": {
                f.name: {"description": f.description, "example": f.example}
                for f in self.inputs
            },
            "secret_slots": [f.name for f in self.secrets],
        }

    # -- how the agent is told about them ----------------------------------
    def prompt_block(self) -> str:
        """The field table appended to the user's instruction.

        Secrets are given to the model as their *placeholder*, not their value.
        The model never sees a password; the executor substitutes the real one
        when the tool call is made. That keeps the credential out of the
        message history, which is the one place redaction cannot reach after
        the fact -- history is replayed to the model on every turn.
        """
        if not self.fields:
            return ""

        lines = ["", "Values to use (supplied by the user):"]
        for entry in self.inputs:
            note = f"  -- {entry.description}" if entry.description else ""
            lines.append(f"  {entry.name}: {entry.value}{note}")
        for entry in self.secrets:
            note = f"  -- {entry.description}" if entry.description else ""
            lines.append(
                f"  {entry.name}: {secret_placeholder(entry.name)} "
                f"(a credential; type it exactly as written){note}"
            )
        lines.append("")
        lines.append(
            "Use each value where the page asks for it. Do not invent values, and "
            "do not use a value the list does not contain."
        )
        return "\n".join(lines)


def parameterise(value: Any, declared: dict[str, str]) -> Any:
    """Replace declared values in a recorded argument with their templates.

    ``declared`` maps a *value* to the template that should replace it, which
    is the direction the lookup needs -- we are scanning recorded arguments for
    strings we already know.

    Recurses into dicts and lists because a value can be nested arbitrarily
    deep inside a tool argument (``fill_form`` puts them inside a list of field
    objects, for instance).
    """
    if isinstance(value, str):
        for literal, template in declared.items():
            if literal and literal in value:
                value = value.replace(literal, template)
        return SECRET_PLACEHOLDER_RE.sub(lambda m: f"{{{{secret.{m.group(1)}}}}}", value)
    if isinstance(value, dict):
        return {k: parameterise(v, declared) for k, v in value.items()}
    if isinstance(value, list):
        return [parameterise(v, declared) for v in value]
    return value


def substitution_map(persisted: dict[str, Any]) -> dict[str, str]:
    """``{recorded literal: template}`` from what was stored on the run.

    Only values long enough to be distinctive are included; a field whose value
    is "1" would otherwise rewrite every digit in the recording.
    """
    mapping: dict[str, str] = {}
    for name, value in (persisted.get("inputs") or {}).items():
        if value and len(str(value)) >= MIN_MATCH_LENGTH:
            mapping[str(value)] = f"{{{{input.{name}}}}}"
    # Longest first: a field whose value contains another's must win, or the
    # shorter one rewrites a fragment and leaves a broken template behind.
    return dict(sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True))


__all__ = [
    "DeclaredField",
    "FieldSet",
    "MIN_MATCH_LENGTH",
    "NAME_PATTERN",
    "SECRET_PLACEHOLDER_RE",
    "parameterise",
    "secret_placeholder",
    "substitution_map",
]
