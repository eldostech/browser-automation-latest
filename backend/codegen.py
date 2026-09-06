"""Turning a ``playwright codegen`` script into a replayable use case.

This replaces the LLM pass in ``distill.py``. The old recorder was a model
driving a browser, and most of what it did was not the workflow -- it failed a
third of its tool calls, took ten snapshots so it could see, and addressed
elements by refs that meant nothing a second later. Distillation existed to
find the workflow inside all of that.

A person recording their own task produces none of that noise. They do not
record their failures, they need no snapshots, and ``codegen`` emits semantic
locators because that is what its own locator generator prefers. So the four
hard problems of distillation simply do not arise here, and what is left is a
parse.

**The script is data, never code.** It is parsed with :mod:`ast` and never
executed, never ``eval``'d, and never imported. That matters more than it
might look: the file arrives from a subprocess driving a browser the user
pointed at a website, and "run whatever the recorder wrote" would make any page
that can influence codegen's output into arbitrary code execution here.

**Anything unrecognised is reported, not guessed at.** ``codegen`` emits a
narrow subset of the Playwright API, but it is not a stable contract and a new
release can add to it. An unsupported line becomes a
:class:`Unsupported` entry the user is shown, rather than a step that silently
does nothing or -- far worse -- a step that does something adjacent.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import ValidationError

from usecase import Assertion, Locator, Step

log = logging.getLogger(__name__)

#: ``get_by_*`` calls and the locator strategy each becomes. These are the ones
#: codegen actually emits; the mapping is one-to-one and deliberately dumb,
#: because the moment this starts reinterpreting a recorded locator it is
#: guessing about an element it has never seen.
_GET_BY: dict[str, str] = {
    "get_by_role": "role",
    "get_by_label": "label",
    "get_by_placeholder": "placeholder",
    "get_by_test_id": "test_id",
    "get_by_alt_text": "alt_text",
    "get_by_text": "text",
    "get_by_title": "text",
}

#: Actions on a locator, and the ``Step.action`` each becomes.
_ACTIONS: dict[str, str] = {
    "click": "click",
    "dblclick": "click",
    "fill": "fill",
    "type": "fill",
    "press": "press",
    "check": "click",
    "uncheck": "click",
    "select_option": "select",
    "hover": "hover",
    "set_input_files": "upload",
}

#: Assertions codegen writes when the user records one.
_EXPECTATIONS: dict[str, tuple[str, bool]] = {
    "to_be_visible": ("element_visible", False),
    "to_be_hidden": ("element_visible", True),
    "to_have_text": ("text_present", False),
    "to_contain_text": ("text_present", False),
    # The recorder's "Assert value" button, for an input. It was not handled,
    # so pointing at a filled-in field produced a dropped line and nothing
    # else -- which is exactly the gesture somebody makes when they mean
    # "read this one".
    "to_have_value": ("field_value", False),
    "to_have_url": ("url_contains", False),
}

#: Boilerplate around the recording proper. Everything the launcher does is
#: this module's caller's business, not a step.
_IGNORED_PAGE_CALLS = frozenset(
    {"close", "wait_for_timeout", "wait_for_load_state", "set_default_timeout", "pause"}
)


@dataclass(slots=True)
class Unsupported:
    """A line the parser recognised as an action but cannot represent."""

    line: int
    source: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"line": self.line, "source": self.source, "reason": self.reason}


@dataclass(slots=True)
class Recording:
    """What a codegen script contained."""

    steps: list[Step] = field(default_factory=list)
    assertions: list[Assertion] = field(default_factory=list)
    #: Every URL the recording visited, which is what the allowlist is built
    #: from. Taken from pages visited rather than only from ``goto``, because a
    #: click that navigates is still a navigation.
    urls: list[str] = field(default_factory=list)
    unsupported: list[Unsupported] = field(default_factory=list)
    #: Every literal value typed during the recording, in order, each with what
    #: it was typed into. The value alone is a poor thing to name a field
    #: after -- "test" and a UUID say nothing -- whereas the control's
    #: accessible name is what the person saw on screen when they typed it.
    values: list["TypedValue"] = field(default_factory=list)
    #: Elements a person pointed at with the recorder's assert buttons. Each is
    #: offered afterwards as either a check or a value to put in the results.
    captured: list["CapturedValue"] = field(default_factory=list)

    @property
    def typed(self) -> list[str]:
        """Just the values, in order. What the parameterisation matches on."""
        return [entry.value for entry in self.values]

    @property
    def start_url(self) -> str | None:
        return self.urls[0] if self.urls else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.model_dump(mode="json") for step in self.steps],
            "assertions": [check.model_dump(mode="json") for check in self.assertions],
            "urls": list(self.urls),
            "typed": list(self.typed),
            "values": [entry.to_dict() for entry in self.values],
            "captured": [entry.to_dict() for entry in self.captured],
            "unsupported": [entry.to_dict() for entry in self.unsupported],
        }



@dataclass
class CapturedValue:
    """An element a person pointed at while recording, and what it held.

    Produced by the recorder's **Assert text** and **Assert value** buttons:
    you click the button, then click the field. That is the whole gesture, and
    it is the only point-and-click way codegen offers to name an element
    without typing a selector.

    codegen writes it as an assertion, because that is what those buttons are
    for. Whether it is really a *check* ("this should still say Approved") or a
    *reading* ("put this in the spreadsheet") is a question only the person can
    answer, so both are offered afterwards and neither is guessed.

    ``after_step`` is how many steps had been recorded when this was seen. An
    extraction has to happen where the value was on screen -- appending them
    all at the end would read page one's field after navigating to page three.
    """

    locator: "Locator"
    value: str
    #: "text" for Assert text, "value" for Assert value on an input.
    kind: str
    after_step: int
    line: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "locator": self.locator.model_dump(mode="json", exclude_none=True),
            "describe": self.locator.describe(),
            "label": self.locator.name or self.locator.text or "",
            "value": self.value,
            "kind": self.kind,
            "after_step": self.after_step,
            "line": self.line,
        }



@dataclass
class TypedValue:
    """One value the person typed or chose, and what they typed it into.

    ``label`` is the control's accessible name as the recording saw it -- the
    label, placeholder or role name codegen wrote into the locator. It is
    frequently the only human-readable thing about a value: a dropdown records
    the option's ``value`` attribute, which on a real application is as likely
    to be a UUID as a word.

    It can be empty. ``get_by_role("combobox")`` with nothing to name it is
    exactly the case that has no label to offer, and inventing one would be
    worse than admitting it.
    """

    value: str
    action: str
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "action": self.action, "label": self.label}


class CodegenError(ValueError):
    """The script could not be parsed at all."""



def _first_message(exc: ValidationError) -> str:
    """The one useful sentence out of a pydantic error report."""
    for error in exc.errors():
        message = str(error.get("msg", "")).removeprefix("Value error, ").strip()
        if message:
            return message
    return "this action cannot be represented as a step"


def parse(script: str) -> Recording:
    """Read a ``codegen --target=python-async`` script into a recording."""
    try:
        tree = ast.parse(script)
    except SyntaxError as exc:
        raise CodegenError(f"the recorded script is not valid Python: {exc}") from exc

    body = _recording_body(tree)
    if body is None:
        raise CodegenError(
            "the recorded script has no run function; it may have been closed before "
            "anything was recorded"
        )

    lines = script.splitlines()
    recording = Recording()
    counter = 0

    for node in body:
        call = _awaited_call(node)
        if call is None:
            continue

        source = lines[call.lineno - 1].strip() if call.lineno - 1 < len(lines) else ""
        counter += 1
        try:
            _consume(recording, call, source, f"s{counter}")
        except ValidationError as exc:
            # One line the schema will not accept must not cost a person the
            # whole session. It joins the other things this line could not
            # represent, and the review screen shows it alongside them.
            recording.unsupported.append(
                Unsupported(call.lineno, source, _first_message(exc))
            )

    # A recording of nothing is a mistake worth naming, rather than an empty
    # use case that runs a thousand times and does nothing.
    if not recording.steps:
        raise CodegenError(
            "no actions were recorded. Perform the task in the browser window before "
            "closing it."
        )
    return recording


def _recording_body(tree: ast.Module) -> list[ast.stmt] | None:
    """The statements of codegen's ``run`` coroutine.

    Codegen wraps the recording in ``async def run(playwright)``. Taking the
    function rather than the module keeps the launcher's own statements -- and
    anything a future release adds at module scope -- out of the parse.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
            return node.body
    return None


def _awaited_call(node: ast.stmt) -> ast.Call | None:
    """The call in ``await <call>``, whether or not its result is assigned."""
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Await):
        inner = node.value.value
        return inner if isinstance(inner, ast.Call) else None
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Await):
        inner = node.value.value
        return inner if isinstance(inner, ast.Call) else None
    return None


def _consume(recording: Recording, call: ast.Call, source: str, step_id: str) -> None:
    """Turn one awaited call into a step, an assertion, or a complaint."""
    if not isinstance(call.func, ast.Attribute):
        return

    method = call.func.attr
    receiver = call.func.value

    # expect(...).to_be_visible() and friends.
    if _is_expect(receiver):
        _consume_expect(recording, call, method, source)
        return

    root = _root_name(receiver)
    if root is None:
        return

    # page.goto("...") and the launcher boilerplate around it.
    if isinstance(receiver, ast.Name):
        if method == "goto":
            url = _string_arg(call, 0)
            if url:
                recording.urls.append(url)
                recording.steps.append(
                    Step(id=step_id, action="navigate", url=url)
                )
            return
        if method in _IGNORED_PAGE_CALLS or root != "page":
            return
        recording.unsupported.append(
            Unsupported(call.lineno, source, f"page.{method}() is not supported")
        )
        return

    if method not in _ACTIONS:
        # A locator chain ending in something we do not perform.
        if root == "page":
            recording.unsupported.append(
                Unsupported(call.lineno, source, f"{method}() is not supported")
            )
        return

    locators = _locator_chain(receiver)
    if locators is None:
        recording.unsupported.append(
            Unsupported(
                call.lineno,
                source,
                "the element is addressed in a way this cannot record "
                "(a frame, a filter, or a chained locator)",
            )
        )
        return
    if not locators:
        return

    action = _ACTIONS[method]
    if action == "upload":
        # Refused before the step is built, not after: the schema requires a
        # file path, and there is no honest one to supply.
        recording.unsupported.append(
            Unsupported(
                call.lineno,
                source,
                "file uploads cannot be replayed from a recording: the file that was "
                "chosen is on the machine that recorded it",
            )
        )
        return

    # Read the argument before building the step, not after. ``Step`` checks
    # that an action has what it needs the moment it is constructed, so a
    # ``press`` built without its key raises there -- and the assignment that
    # would have supplied the key is on the next line, never reached.
    value: str | None = None
    if action == "fill":
        value = _string_arg(call, 0)
        if value is None:
            recording.unsupported.append(
                Unsupported(call.lineno, source, "the typed value is not a literal")
            )
            return
        recording.values.append(TypedValue(value, action, _label_of(locators)))
    elif action == "press":
        value = _string_arg(call, 0)
        if value is None:
            recording.unsupported.append(
                Unsupported(call.lineno, source, "the key pressed is not a literal")
            )
            return
    elif action == "select":
        value = _string_arg(call, 0) or _keyword_string(call, "value")
        if value is None:
            # codegen writes a list when more than one option was selected at
            # once, and a step holds one value. Worth naming, because "not a
            # literal" describes a list badly enough to send someone looking
            # for the wrong problem.
            arguments = call.args[0] if call.args else None
            if isinstance(arguments, (ast.List, ast.Tuple)):
                reason = (
                    "more than one option was selected at once, and a step records "
                    "a single choice"
                )
            else:
                reason = "the selected option is not a literal"
            recording.unsupported.append(Unsupported(call.lineno, source, reason))
            return
        recording.values.append(TypedValue(value, action, _label_of(locators)))

    recording.steps.append(
        Step(id=step_id, action=action, locators=locators, value=value)
    )


def _consume_expect(
    recording: Recording, call: ast.Call, method: str, source: str
) -> None:
    if method not in _EXPECTATIONS:
        recording.unsupported.append(
            Unsupported(call.lineno, source, f"expect(...).{method}() is not supported")
        )
        return

    kind, negate = _EXPECTATIONS[method]
    subject = _expect_subject(call.func.value)  # type: ignore[union-attr]

    if kind == "url_contains":
        value = _string_arg(call, 0)
        if value is None:
            # codegen writes a compiled regex for a pattern assertion, which is
            # not something this schema holds.
            recording.unsupported.append(
                Unsupported(call.lineno, source, "the asserted URL is not a literal")
            )
            return
        recording.assertions.append(
            Assertion(kind="url_contains", value=value, negate=negate)
        )
        return

    if subject is None:
        recording.unsupported.append(
            Unsupported(call.lineno, source, "the asserted element could not be read")
        )
        return

    if kind in {"text_present", "field_value"}:
        value = _string_arg(call, 0)
        if value is not None:
            # Kept with its locator, which the text case used to discard. A
            # value with no element is only ever "this text is somewhere on the
            # page"; with one it can also be "read this field", and which of
            # those the person meant is asked rather than assumed.
            recording.captured.append(
                CapturedValue(
                    locator=subject[0],
                    value=value,
                    kind="value" if kind == "field_value" else "text",
                    after_step=len(recording.steps),
                    line=call.lineno,
                )
            )
            if kind == "text_present":
                recording.assertions.append(
                    Assertion(kind="text_present", value=value, negate=negate)
                )
            return

    recording.assertions.append(
        Assertion(kind="element_visible", locator=subject[0], negate=negate)
    )


def _expect_subject(node: ast.expr) -> list[Locator] | None:
    """The locator inside ``expect(<locator>)``."""
    if not isinstance(node, ast.Call) or not node.args:
        return None
    chain = _locator_chain(node.args[0])
    return chain or None


def _is_expect(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "expect"
    )


def _root_name(node: ast.expr) -> str | None:
    """The identifier a locator chain starts from -- ``page``, normally."""
    while isinstance(node, (ast.Call, ast.Attribute, ast.Subscript)):
        node = node.func if isinstance(node, ast.Call) else node.value  # type: ignore[assignment]
    return node.id if isinstance(node, ast.Name) else None


#: Locator properties -- written without parentheses, so they reach the AST as
#: an attribute rather than a call.
_POSITIONS = frozenset({"first", "last"})


def _label_of(locators: list[Locator]) -> str:
    """What the person saw on the control they just used, if anything did.

    The first rung is the one codegen chose, so it is the one that named the
    element. A css rung names nothing, and neither does a role with no
    accessible name -- both return "" rather than a guess.
    """
    for locator in locators:
        if locator.strategy == "role" and locator.name:
            return locator.name
        if locator.strategy in {"label", "placeholder", "alt_text"} and locator.text:
            return locator.text
    return ""


def _locator_chain(node: ast.expr) -> list[Locator] | None:
    """The ladder for ``page.get_by_role(...).first``, ``.nth(1)`` and friends.

    Returns ``None`` when the chain contains something that cannot be recorded
    -- a frame, a filter, or one locator scoped inside another. Those are not
    failures of the recording; they are shapes this schema has no rung for, and
    inventing an approximation would produce a use case that clicks something
    else on row one.
    """
    # ``.first`` and ``.last`` are properties, not calls, so they arrive as an
    # ast.Attribute wrapping the chain rather than as another ast.Call. Walking
    # only calls made the whole chain unreadable, and codegen writes ``.first``
    # whenever a locator matched more than one element -- which is most of the
    # time on a real page. Every step addressed that way was being dropped.
    steps: list[tuple[str, ast.Call | None]] = []
    current = node
    while True:
        if isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
            steps.append((current.func.attr, current))
            current = current.func.value
        elif isinstance(current, ast.Attribute) and current.attr in _POSITIONS:
            steps.append((current.attr, None))
            current = current.value
        else:
            break

    if not isinstance(current, ast.Name) or current.id != "page":
        return None

    steps.reverse()
    locator: Locator | None = None
    nth = 0

    for name, call in steps:
        if name in _GET_BY:
            assert call is not None
            if locator is not None:
                # A second get_by_* means one locator scoped inside another.
                return None
            locator = _locator_from(name, call)
            if locator is None:
                return None
        elif name == "locator":
            if locator is not None or call is None:
                return None
            selector = _string_arg(call, 0)
            if selector is None:
                return None
            locator = Locator(strategy="css", selector=selector)
        elif name == "nth":
            if call is None:
                return None
            index = _int_arg(call, 0)
            if index is None:
                return None
            nth = index
        elif name == "first":
            nth = 0
        elif name == "last":
            # Held as -1 and performed as Playwright's own ``.last``, so it
            # stays "whichever is last when this runs" rather than becoming a
            # fixed index guessed from the recording -- which is how a batch
            # ends up clicking the wrong row.
            nth = -1
        else:
            return None

    if locator is None:
        return None
    locator.nth = nth
    return _ladder(locator)


def _locator_from(name: str, call: ast.Call) -> Locator | None:
    strategy = _GET_BY[name]
    value = _string_arg(call, 0)
    if value is None:
        return None
    if strategy == "role":
        accessible_name = _keyword_string(call, "name")
        return Locator(strategy="role", role=value, name=accessible_name)
    return Locator(strategy=strategy, text=value)


def _ladder(primary: Locator) -> list[Locator]:
    """One recorded locator, plus any free fallback it implies.

    Codegen gives one way to find the element. A ``role`` rung with an
    accessible name also implies a text rung, which costs nothing to record and
    catches the redesign that keeps a control's wording while changing its role
    -- a link that becomes a button.

    ``text`` and not ``label``: every rung is executed as the Playwright call
    it names, and ``get_by_label`` matches only form controls that have a
    label. A button named "Sign in" is found by its text, not by a label it
    does not have, so a ``label`` fallback for one is a rung that can never
    match. This was not visible while rungs were resolved by searching a
    snapshot for an accessible name, which matched either.
    """
    ladder = [primary]
    if primary.strategy == "role" and primary.name:
        ladder.append(Locator(strategy="text", text=primary.name, nth=primary.nth))
    return ladder


def _string_arg(call: ast.Call, index: int) -> str | None:
    if len(call.args) <= index:
        return None
    node = call.args[index]
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _int_arg(call: ast.Call, index: int) -> int | None:
    if len(call.args) <= index:
        return None
    node = call.args[index]
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, int) else None


def _keyword_string(call: ast.Call, keyword: str) -> str | None:
    for item in call.keywords:
        if item.arg == keyword and isinstance(item.value, ast.Constant):
            value = item.value.value
            if isinstance(value, str):
                return value
    return None


def urls_of(recording: Recording) -> list[str]:
    """Distinct hosts visited, for the domain allowlist."""
    hosts: list[str] = []
    for url in recording.urls:
        match = re.match(r"^[a-z][a-z0-9+.\-]*://([^/:]+)", url, re.IGNORECASE)
        host = match.group(1).lower() if match else None
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def summarise(recording: Recording) -> str:
    parts = [f"{len(recording.steps)} step(s)"]
    if recording.assertions:
        parts.append(f"{len(recording.assertions)} check(s)")
    if recording.unsupported:
        parts.append(f"{len(recording.unsupported)} line(s) not understood")
    return ", ".join(parts)


__all__ = [
    "CodegenError",
    "Recording",
    "Unsupported",
    "parse",
    "summarise",
    "urls_of",
]
