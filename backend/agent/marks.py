"""The tools that turn *doing* a task into *recording* one.

Playwright MCP can drive a browser. It cannot produce a use case, and the gap
between those two is this file.

**`describe_element` is the bridge.** An MCP ref is an index into one snapshot
and is meaningless in any other; a `UseCase` needs a locator that still works
next year. Resolving one to the other is not new work here -- `snapshot.py` was
written for exactly this and says so in its own docstring -- but two things are
added on top, and both come from failures this codebase has already had:

* **`exact` is set from what else is on the page.** Playwright matches an
  accessible name as a case-insensitive substring, so a button named "Invite"
  also finds "+ Invite User". A run failed on precisely that, spending its
  whole step budget waiting for a button the dialog was covering.
* **Ambiguity is reported at the moment of recording.** If a rung matches three
  elements, the person watching the agent can see it now, rather than the
  batch discovering it four thousand rows later.

**The marking tools replace inference with declaration.** The design sized loop
detection as the hard part of distillation: recognise the repeated shape across
four hundred near-identical sub-trajectories and work out where a row begins
and which value varies. Do not infer what the agent can declare.
`mark_setup_complete`, `begin_row` and `end_row` turn that into bookkeeping,
and `mark_as_input` / `mark_as_output` / `mark_as_secret` name the values while
the page they came from is still on screen.

These are ours, not the server's: they never reach the browser, and they are
dispatched here rather than over MCP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from snapshot import Node, Snapshot

# Imported for the shape of what we produce. `usecase` is a schema module with
# no I/O, so this does not couple the agent to the application.
from usecase import Locator

#: Marking tools, and the schema each takes. These are advertised to the model
#: beside the server's own, and they are the only tools in the registry that
#: never touch the browser.
MARK_TOOLS: dict[str, dict[str, Any]] = {
    "describe_element": {
        "description": (
            "Resolve an element reference from the current snapshot into the "
            "durable locator a recorded step would use, and report how many "
            "elements each way of finding it matches. Call this before marking "
            "anything: an ambiguous locator is far cheaper to fix now."
        ),
        "properties": {
            "ref": {"type": "string", "description": "A ref such as 'e12'."},
        },
        "required": ["ref"],
    },
    "mark_setup_complete": {
        "description": (
            "Everything done so far was setup -- signing in, choosing a "
            "workspace -- and runs once per batch rather than once per row. "
            "Call this exactly once, when the per-record work is about to start."
        ),
        "properties": {},
        "required": [],
    },
    "begin_row": {
        "description": (
            "The work for one record starts here. Everything until end_row "
            "becomes the steps that repeat, once per row of the spreadsheet."
        ),
        "properties": {
            "key": {
                "type": "string",
                "description": "What identifies this record, e.g. 'A-1001'.",
            }
        },
        "required": ["key"],
    },
    "end_row": {
        "description": "The work for one record is finished.",
        "properties": {},
        "required": [],
    },
    "mark_as_input": {
        "description": (
            "This value changes per record and should come from a spreadsheet "
            "column. The step that types it becomes a template."
        ),
        "properties": {
            "ref": {"type": "string"},
            "name": {"type": "string", "description": "The column name."},
        },
        "required": ["ref", "name"],
    },
    "mark_as_output": {
        "description": (
            "Read this element's value into the results file. Becomes an "
            "extract step at this point in the run, on the page it was seen on."
        ),
        "properties": {
            "ref": {"type": "string"},
            "column": {"type": "string"},
        },
        "required": ["ref", "column"],
    },
    "mark_as_secret": {
        "description": (
            "This is a credential. It binds to a stored credential slot; the "
            "value itself is never written into the use case."
        ),
        "properties": {
            "ref": {"type": "string"},
            "slot": {"type": "string", "description": "The credential slot name."},
        },
        "required": ["ref", "slot"],
    },
}


# ---------------------------------------------------------------------------
# ref -> a locator ladder
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Described:
    """What an element is, durably, and how safely it can be found again."""

    ref: str
    role: str
    name: str
    #: Ranked, semantic first. The same shape a recorded step carries.
    ladder: list[Locator] = field(default_factory=list)
    #: How many elements the leading rung matches. One is what you want.
    matches: int = 1
    #: Set when the accessible name is a substring of another element's, which
    #: is when `exact` stops being optional.
    shadowed_by: tuple[str, ...] = ()

    @property
    def ambiguous(self) -> bool:
        return self.matches != 1

    def describe_first(self) -> str:
        """The leading rung, as a person would read it."""
        return self.ladder[0].describe() if self.ladder else "(nothing durable)"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "role": self.role,
            "name": self.name,
            "locators": [loc.model_dump(mode="json", exclude_none=True) for loc in self.ladder],
            "describe": self.ladder[0].describe() if self.ladder else "",
            "matches": self.matches,
            "ambiguous": self.ambiguous,
            "shadowed_by": list(self.shadowed_by),
        }

    def as_text(self) -> str:
        """What the model is told. Short, and honest about the risk."""
        if not self.ladder:
            return f"{self.ref} is not on the page as it now stands."
        lines = [f"{self.ref} is {self.describe_first()}"]
        if self.ambiguous:
            lines.append(
                f"WARNING: that matches {self.matches} elements, so a recorded "
                "step could act on the wrong one. Point at something more "
                "specific, or accept that this step will need a person."
            )
        elif self.shadowed_by:
            others = ", ".join(repr(name) for name in self.shadowed_by)
            lines.append(
                f"Its name is contained in {others}, so the locator is recorded "
                "as an exact match."
            )
        return "\n".join(lines)


def describe_element(snapshot: Snapshot, ref: str) -> Described:
    """One ref, as the locator ladder a recorded step would carry.

    The ladder is built the way the codegen parser builds one: a semantic rung
    first, and a text rung behind it that costs nothing and survives a redesign
    keeping a control's wording while changing its role.
    """
    node = snapshot.get(ref)
    if node is None:
        return Described(ref=ref, role="", name="", matches=0)

    exact = _needs_exact(snapshot, node)
    ladder: list[Locator] = []
    if node.role:
        ladder.append(
            Locator(strategy="role", role=node.role, name=node.name or None, exact=exact)
        )
    if node.name:
        ladder.append(Locator(strategy="text", text=node.name, exact=exact))
    if not ladder:
        # Nothing durable to say about it. Better an empty ladder the caller
        # can refuse than a css guess nobody can review.
        return Described(ref=ref, role=node.role, name=node.name, matches=0)

    return Described(
        ref=ref,
        role=node.role,
        name=node.name,
        ladder=ladder,
        matches=_count(snapshot, node, exact),
        shadowed_by=_shadowing(snapshot, node),
    )


def _needs_exact(snapshot: Snapshot, node: Node) -> bool:
    """Whether the recorded name has to be the *whole* accessible name.

    A run failed on this: a page with "+ Invite User" and, in the dialog it
    opens, "Invite". Playwright reads a name as a substring, so the dialog's
    locator found both and acted on the one behind the dialog. Deciding it here
    means the recording carries the answer rather than the replay discovering
    the question.
    """
    return bool(node.name) and bool(_shadowing(snapshot, node))


def _shadowing(snapshot: Snapshot, node: Node) -> tuple[str, ...]:
    """Other elements whose accessible name *contains* this one's."""
    if not node.name:
        return ()
    wanted = node.name.casefold()
    return tuple(
        dict.fromkeys(
            other.name
            for other in snapshot
            if other.ref != node.ref
            and other.name
            and other.name.casefold() != wanted
            and wanted in other.name.casefold()
        )
    )


def _count(snapshot: Snapshot, node: Node, exact: bool) -> int:
    """How many elements the leading rung would match on this page."""
    if not node.name:
        return sum(1 for other in snapshot if other.role == node.role)
    wanted = node.name.casefold()
    if exact:
        return sum(
            1
            for other in snapshot
            if other.role == node.role and (other.name or "").casefold() == wanted
        )
    return sum(
        1
        for other in snapshot
        if other.role == node.role and wanted in (other.name or "").casefold()
    )


# ---------------------------------------------------------------------------
# What the marks accumulate into
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Mark:
    """One declaration, tied to the tool call it was made after.

    ``after_call`` is how many calls had happened when this was made, and it is
    correctness rather than bookkeeping: a value has to be read on the page it
    was pointed at. Appending every reading to the end would read the first
    page's field after the browser had already moved to the third.
    """

    kind: str
    after_call: int
    ref: str = ""
    name: str = ""
    described: Described | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "after_call": self.after_call,
            "ref": self.ref,
            "name": self.name,
            "element": self.described.as_dict() if self.described else None,
        }


class Marks:
    """Everything the agent declared about the shape of the use case.

    Deliberately dumb: it records and refuses the obviously wrong, and it does
    not try to make a use case. Distillation reads it in a later phase, and
    keeping the two apart means this can be tested by asserting on a list.
    """

    def __init__(self) -> None:
        self.entries: list[Mark] = []
        #: Call index at which setup ended. None means it never did, which is
        #: refused at the end -- a workflow whose sign-in runs per row would
        #: sign in four thousand times.
        self.setup_ended_at: int | None = None
        self.rows: list[tuple[int, int | None, str]] = []
        self._open_row: tuple[int, str] | None = None

    # -- boundaries -------------------------------------------------------
    def setup_complete(self, after_call: int) -> str:
        if self.setup_ended_at is not None:
            return "Setup was already marked complete; it can only happen once."
        if self._open_row is not None:
            return "A row is open. Setup cannot end in the middle of one."
        self.setup_ended_at = after_call
        self.entries.append(Mark("setup_complete", after_call))
        return ""

    def begin_row(self, after_call: int, key: str) -> str:
        if self._open_row is not None:
            return f"Row {self._open_row[1]!r} is still open. Call end_row first."
        self._open_row = (after_call, key)
        self.entries.append(Mark("begin_row", after_call, name=key))
        return ""

    def end_row(self, after_call: int) -> str:
        if self._open_row is None:
            return "No row is open, so there is nothing to end."
        started, key = self._open_row
        self.rows.append((started, after_call, key))
        self._open_row = None
        self.entries.append(Mark("end_row", after_call, name=key))
        return ""

    # -- values -----------------------------------------------------------
    def mark_value(
        self, kind: str, after_call: int, ref: str, name: str, described: Described
    ) -> str:
        if described.matches == 0:
            return (
                f"{ref} is not on the page as it now stands, so there is nothing "
                "to mark. Take a fresh snapshot."
            )
        if described.ambiguous:
            return (
                f"{ref} resolves to {described.describe_first()} which matches "
                f"{described.matches} elements. Marking it would record a step "
                "that can act on the wrong one; point at something more specific."
            )
        if any(
            entry.kind == kind and entry.name == name for entry in self.entries
        ):
            return f"{name!r} has already been marked as {kind.replace('_', ' ')}."
        self.entries.append(Mark(kind, after_call, ref=ref, name=name, described=described))
        return ""

    # -- what a session must have said before it can finish ---------------
    def unfinished(self) -> str:
        """Why this session cannot be distilled yet, or "".

        Enforced by code at the point ``finish`` is called, rather than asked
        for in a prompt. An agent that forgets is told to go back.
        """
        if self._open_row is not None:
            return f"Row {self._open_row[1]!r} was never ended. Call end_row."
        if not self.rows:
            return (
                "No row was recorded. Call begin_row before the work for one "
                "record and end_row after it, so the steps that repeat can be "
                "told apart from the sign-in that must not."
            )
        return ""

    def as_dicts(self) -> list[dict[str, Any]]:
        return [entry.as_dict() for entry in self.entries]


__all__ = [
    "MARK_TOOLS",
    "Described",
    "Mark",
    "Marks",
    "describe_element",
]
