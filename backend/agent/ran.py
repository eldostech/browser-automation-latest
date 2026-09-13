"""The Playwright expression the MCP server actually ran, as a locator.

Every acting call comes back with it::

    ### Ran Playwright code
    ```js
    await page.locator('label').filter({ hasText: 'Nayra Asati' }).click();
    ```

and until now this codebase read the page out of that reply and threw the code
away. `providers/base.ToolResult` even says both are worth keeping.

**Why it is worth more than the snapshot rung.** The agent acts on a ref, and
`marks.describe_element` turns that ref into a locator by reading the
*accessibility tree*. The server resolves the same ref to a **DOM element**.
Those are not always the same thing, and the gap is not academic: a real
recording of a profile picker produced

    a11y tree  ->  role=radio name="Nayra Asati"
    server ran ->  page.locator('label').filter({ hasText: 'Nayra Asati' })

because the site draws a styled radio whose input is not clickable and whose
label is. The recording replayed, found the radio, and spent thirty seconds
failing to click something a person clicks every day. The same session typed a
secret word into ``input[name="secretWord"]`` -- unambiguous, exactly right --
while the tree-derived rung matched three elements and the mark was refused.

So the expression the server ran is evidence of the one thing the snapshot
cannot tell us: which DOM element received the action. It goes into the ladder
behind the semantic rungs, because a CSS path is less durable than a role and
name across a redesign -- but ahead of nothing, which is what it had before.

This parses **JavaScript**, which is what the server emits. `codegen.py` parses
the Python that `playwright codegen` writes, with Python's own `ast`. The
vocabulary is the same set of calls in camelCase, so this maps onto the same
`Locator` model and deliberately makes the same choices; what it does not share
is a parser, because feeding JavaScript to `ast.parse` fails on the first
object literal.
"""

from __future__ import annotations

import logging
import re

from usecase import Locator

log = logging.getLogger(__name__)

#: The fenced block the server puts its code in.
_BLOCK = re.compile(r"###\s*Ran Playwright code\s*```(?:js|javascript)?\s*(.+?)```", re.S)

#: One link of the chain: a name, then its argument text up to the matching
#: bracket. Depth-counted rather than regex-matched, because an argument can
#: itself hold brackets and quotes.
_LINK = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*(\()?")

#: `getByRole('button', ...)` and friends, to the strategy each one means.
_FACTORIES: dict[str, str] = {
    "getByRole": "role",
    "getByText": "text",
    "getByLabel": "label",
    "getByPlaceholder": "placeholder",
    "getByAltText": "alt_text",
    "getByTestId": "test_id",
    "getByTitle": "text",
}

#: What ends a chain. Everything before it describes the element.
ACTIONS: frozenset[str] = frozenset(
    {
        "click",
        "dblclick",
        "fill",
        "type",
        "press",
        "check",
        "uncheck",
        "selectOption",
        "hover",
        "setInputFiles",
        "focus",
        "clear",
        "tap",
    }
)


def ran_code(text: str) -> str:
    """The code block out of a tool result, or ``""``."""
    found = _BLOCK.search(text or "")
    return found.group(1).strip() if found else ""


def locator_from(code: str) -> Locator | None:
    """The element a Playwright statement acts on, as a `Locator`.

    ``None`` when the statement does not describe an element -- a `goto`, a
    keyboard press, a snapshot -- or when it uses something this does not
    model. Returning nothing is the honest answer and costs only the rung:
    a half-understood chain would be a locator nobody could review.
    """
    if not code:
        return None
    statement = _first_statement(code)
    if not statement or "page" not in statement:
        return None

    locator: Locator | None = None
    frames: list[str] = []

    for name, argument in _links(statement):
        if name in _FACTORIES:
            made = _from_factory(name, argument)
            if made is None:
                return None
            made.within = locator
            made.frames = frames
            frames = []
            locator = made
        elif name == "locator":
            selector = _string(argument)
            if not selector:
                return None
            locator = Locator(
                strategy="css", selector=selector, within=locator, frames=frames
            )
            frames = []
        elif name == "frameLocator":
            selector = _string(argument)
            if not selector or locator is not None:
                return None
            frames.append(selector)
        elif name == "contentFrame":
            # The newer spelling: the frame element is found as a locator and
            # then stepped into. Only a plain CSS rung converts, matching what
            # `codegen.py` accepts, because anything else would mean guessing
            # which of several frames was meant.
            if locator is None or locator.strategy != "css" or locator.within is not None:
                return None
            if locator.nth or locator.has_text:
                return None
            frames.append(locator.selector or "")
            locator = None
        elif name == "filter":
            has_text = _option(argument, "hasText")
            if not has_text or locator is None:
                return None
            locator.has_text = has_text
        elif name == "nth":
            if locator is None:
                return None
            try:
                locator.nth = int((argument or "").strip())
            except ValueError:
                return None
        elif name == "first":
            # 0 is this schema's "no position given", so `.first()` records no
            # position -- the same choice `codegen.py` makes, and for the same
            # reason: a resolver that acts on whichever element comes first is
            # the behaviour the ladder exists to refuse.
            if locator is None:
                return None
        elif name == "last":
            if locator is None:
                return None
            locator.nth = -1
        elif name in ACTIONS:
            break
        else:
            # An unmodelled link -- `scrollIntoViewIfNeeded`, `waitFor`, a
            # method added by a later Playwright. The chain is no longer fully
            # understood, so it produces nothing.
            return None

    if locator is None or frames:
        return None
    try:
        # Through the schema rather than trusted: this text came off the wire.
        return Locator.model_validate(locator.model_dump(mode="json", exclude_none=True))
    except Exception:  # noqa: BLE001 - a rung nobody can validate is no rung
        log.info("the server's own locator did not validate", extra={"code": code[:200]})
        return None


def _first_statement(code: str) -> str:
    """The first statement of the block, without `await` or a trailing `;`.

    One statement is what an acting call produces. A block with several is a
    `browser_fill_form`, whose fields are recorded per field from the marks --
    so the first is the only one this is asked about.
    """
    for line in code.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        line = line.removeprefix("await ").strip().rstrip(";")
        return line
    return ""


def _links(statement: str):
    """``(name, argument_text)`` for each link after `page`, in order."""
    start = statement.find("page")
    if start < 0:
        return
    index = start + len("page")
    while True:
        match = _LINK.match(statement, index)
        if match is None:
            return
        name = match.group(1)
        if match.group(2) is None:
            # A property, not a call: `.first`, `.contentFrame` are written
            # both ways depending on the version that emitted them.
            yield name, None
            index = match.end()
            continue
        argument, index = _balanced(statement, match.end() - 1)
        yield name, argument


def _balanced(text: str, opening: int) -> tuple[str, int]:
    """The text inside the bracket at ``opening``, and where it ends.

    Depth-counted and quote-aware, because ``{ hasText: 'a)b' }`` is an
    ordinary thing for a page to contain and a regex would stop inside it.
    """
    depth = 0
    quote = ""
    for position in range(opening, len(text)):
        character = text[position]
        if quote:
            if character == quote and text[position - 1] != "\\":
                quote = ""
            continue
        if character in "\"'`":
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : position], position + 1
    return "", len(text)


def _from_factory(name: str, argument: str | None) -> Locator | None:
    value = _string(argument)
    if not value:
        return None
    strategy = _FACTORIES[name]
    exact = _option(argument, "exact") == "true"
    if strategy == "role":
        return Locator(
            strategy="role",
            role=value,
            name=_option(argument, "name") or None,
            exact=exact and bool(_option(argument, "name")),
        )
    return Locator(strategy=strategy, text=value, exact=exact)


def _string(argument: str | None) -> str:
    """The first quoted string in an argument list."""
    if not argument:
        return ""
    found = re.search(r"""(['"`])(.*?)(?<!\\)\1""", argument, re.S)
    return _unescape(found.group(2)) if found else ""


def _option(argument: str | None, key: str) -> str:
    """One value out of a `{ name: 'x', exact: true }` options object."""
    if not argument:
        return ""
    found = re.search(
        rf"""\b{re.escape(key)}\s*:\s*(?:(['"`])(.*?)(?<!\\)\1|(true|false|\d+))""",
        argument,
        re.S,
    )
    if not found:
        return ""
    return _unescape(found.group(2)) if found.group(2) is not None else found.group(3)


def _unescape(value: str) -> str:
    return value.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")


__all__ = ["ACTIONS", "locator_from", "ran_code"]
