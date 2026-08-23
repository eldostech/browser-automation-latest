"""Parser for the accessibility snapshot returned by Playwright MCP.

This module is the linchpin of deterministic replay, and it exists because of
one property of the snapshot format: it is regular enough to parse exactly.

A snapshot looks like this::

    ### Page
    - Page URL: https://www.ixl.com/signin
    - Page Title: Sign in
    ### Snapshot
    ```yaml
    - generic [ref=e2]:
      - navigation "Shortcuts menu" [ref=e3]:
        - heading "Skip to" [level=2] [ref=e4]
        - link "main content" [ref=e7] [cursor=pointer]:
          - /url: "#skippedLink"
          - text: Main content
    ```

Every interactive node carries a **role**, an optional **accessible name**, and
a **ref**. That gives two lookups, and the whole replay design rests on them:

``by_ref`` -- used at *distillation* time
    ``ref=e17`` is an index into one snapshot and is meaningless in any other.
    Resolving it to ``(role="textbox", name="Username")`` against the snapshot
    captured immediately before the step turns an ephemeral handle into a
    durable description.

``locate`` -- used at *replay* time
    The reverse: given ``(role, name)``, find the ref in a *fresh* snapshot.
    Snapshots cost tokens only when they enter an LLM context, so a replay with
    no model in the loop can take one before every step for free.

Only ref-bearing lines are indexed. That is not a limitation but a filter: it
excludes the property lines (``- /url:``, ``- text:``) and the ``### Page``
header, none of which describe an element you can act on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

#: The fenced block holding the tree. Older/newer servers may omit the fence,
#: in which case the whole body is parsed.
_FENCE_RE = re.compile(r"```(?:yaml|yml)?\s*\n(?P<body>.*?)```", re.DOTALL)

_PAGE_URL_RE = re.compile(r"^-\s*Page URL:\s*(?P<url>\S+)\s*$", re.MULTILINE)
_PAGE_TITLE_RE = re.compile(r"^-\s*Page Title:\s*(?P<title>.*?)\s*$", re.MULTILINE)

#: One node line. Roles are single tokens; the accessible name is a quoted
#: string that may contain escaped quotes; attributes are bracketed pairs in
#: any order; a trailing ``:`` may be followed by the node's text content.
_NODE_RE = re.compile(
    r"""
    ^(?P<indent>[ \t]*)
    -[ ]+
    (?P<role>[A-Za-z][A-Za-z0-9_-]*)
    (?:[ ]+"(?P<name>(?:[^"\\]|\\.)*)")?
    (?P<attrs>(?:[ ]*\[[^\]]*\])*)
    (?:[ ]*:(?P<text>.*))?
    [ ]*$
    """,
    re.VERBOSE,
)

_ATTR_RE = re.compile(r"\[(?P<key>[A-Za-z][A-Za-z0-9_-]*)(?:=(?P<value>[^\]]*))?\]")

#: Roles that never describe something a person interacts with. Kept out of
#: `locate` results so a wrapper `generic` never shadows the real control.
_STRUCTURAL_ROLES: frozenset[str] = frozenset({"generic", "group", "none", "presentation"})


@dataclass(slots=True)
class Node:
    """One ref-bearing line of the accessibility tree."""

    ref: str
    role: str
    name: str = ""
    depth: int = 0
    text: str = ""
    attrs: dict[str, str] = field(default_factory=dict)
    line_no: int = 0

    @property
    def interactive(self) -> bool:
        return self.role not in _STRUCTURAL_ROLES

    def describe(self) -> str:
        """Human-readable identity, for review UIs and failure messages."""
        return f'{self.role} "{self.name}"' if self.name else self.role


def _unescape(value: str) -> str:
    return value.replace('\\"', '"').replace("\\\\", "\\")


def _normalise(value: str) -> str:
    """Fold whitespace and case, for the tolerant tiers of name matching."""
    return re.sub(r"\s+", " ", value or "").strip().casefold()


@dataclass(slots=True)
class Snapshot:
    """A parsed accessibility snapshot.

    ``by_ref`` is exact. ``locate`` is deliberately tolerant, because the name
    recorded months ago may differ from today's by casing or whitespace while
    still denoting the same control.
    """

    nodes: list[Node] = field(default_factory=list)
    page_url: str | None = None
    page_title: str | None = None

    # -- lookups ------------------------------------------------------------
    @property
    def by_ref(self) -> dict[str, Node]:
        return {node.ref: node for node in self.nodes}

    def get(self, ref: str) -> Node | None:
        """The node a ref points at, or ``None`` if this snapshot has no such ref."""
        for node in self.nodes:
            if node.ref == ref:
                return node
        return None

    def find(self, role: str, name: str | None = None) -> list[Node]:
        """Every node matching ``role`` (and ``name``, when given), in document order.

        Matching runs in three tiers and stops at the first that yields
        anything: exact name, then case/whitespace-insensitive, then substring.
        Tiering rather than always-substring keeps an exact match from being
        outvoted by a longer label that merely contains it.
        """
        role_key = _normalise(role)
        candidates = [n for n in self.nodes if _normalise(n.role) == role_key]
        if name is None:
            return candidates

        exact = [n for n in candidates if n.name == name]
        if exact:
            return exact

        wanted = _normalise(name)
        loose = [n for n in candidates if _normalise(n.name) == wanted]
        if loose:
            return loose

        return [n for n in candidates if wanted and wanted in _normalise(n.name)]

    def locate(self, role: str, name: str | None = None, nth: int = 0) -> Node | None:
        """The ``nth`` node matching ``role``/``name``, or ``None``.

        Structural wrappers are skipped when anything interactive matches, so a
        ``generic`` container never shadows the button inside it.
        """
        matches = self.find(role, name)
        interactive = [n for n in matches if n.interactive]
        pool = interactive or matches
        if not pool:
            return None
        if nth < 0 or nth >= len(pool):
            return None
        return pool[nth]

    def roles(self) -> dict[str, int]:
        """Role histogram. Used in failure messages to say what *was* on the page."""
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node.role] = counts.get(node.role, 0) + 1
        return counts

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)


def parse(text: str) -> Snapshot:
    """Parse snapshot text. Never raises -- unparseable input yields an empty snapshot.

    Tolerance is deliberate: this runs against whatever the MCP server emitted,
    including truncated tool results. A caller that gets no nodes falls through
    to the next locator strategy, which is a far better failure than an
    exception taking down a 1,000-row batch.
    """
    snapshot = Snapshot()
    if not text:
        return snapshot

    url_match = _PAGE_URL_RE.search(text)
    if url_match:
        snapshot.page_url = url_match.group("url")
    title_match = _PAGE_TITLE_RE.search(text)
    if title_match:
        snapshot.page_title = title_match.group("title") or None

    fence = _FENCE_RE.search(text)
    body = fence.group("body") if fence else text

    for line_no, line in enumerate(body.splitlines(), start=1):
        if not line.strip():
            continue
        match = _NODE_RE.match(line)
        if match is None:
            continue

        attrs = {
            m.group("key"): (m.group("value") or "")
            for m in _ATTR_RE.finditer(match.group("attrs") or "")
        }
        ref = attrs.pop("ref", "")
        if not ref:
            # Property lines and unreferenced decoration. Nothing to act on.
            continue

        indent = match.group("indent") or ""
        snapshot.nodes.append(
            Node(
                ref=ref,
                role=match.group("role"),
                name=_unescape(match.group("name") or ""),
                depth=len(indent.expandtabs(2)) // 2,
                text=(match.group("text") or "").strip(),
                attrs=attrs,
                line_no=line_no,
            )
        )

    return snapshot


def is_ref(target: str) -> bool:
    """True when a recorded target is an ephemeral ref rather than a selector.

    Both spellings the model produces are recognised: ``ref=e15`` and ``[ref=e15]``.
    """
    value = (target or "").strip()
    return bool(re.fullmatch(r"\[?ref=[A-Za-z0-9]+\]?", value))


def extract_ref(target: str) -> str | None:
    """Pull the ref id out of ``ref=e15`` / ``[ref=e15]``. ``None`` if absent."""
    match = re.search(r"ref=([A-Za-z0-9]+)", target or "")
    return match.group(1) if match else None
