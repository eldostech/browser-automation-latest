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

Refs are optional, and which source produced the tree decides whether they are
there. Playwright MCP adds ``[ref=eN]`` to every node; Playwright's own
``locator.aria_snapshot()`` emits the same YAML **without** them. Both are
parsed, because both are used: MCP during the transition, and ``aria_snapshot``
by the engine that replaced it.

That is why the ref filter this parser used to apply is gone. Requiring a ref
made every node of a real ``aria_snapshot`` invisible -- the tree parsed
cleanly and yielded nothing, which is the most expensive kind of wrong. Lines
that genuinely describe no element (``- /url: ...``) fail the role pattern and
are skipped on their own merits.
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
    #: The text this was parsed from.
    #:
    #: Kept because a repair proposed later reads the page *as text* and parses
    #: it again -- rebuilding it from the nodes loses the shape the parser
    #: expects, and the repair then has nothing to match against.
    raw: str = ""

    # -- lookups ------------------------------------------------------------
    @property
    def by_ref(self) -> dict[str, Node]:
        """Ref-bearing nodes only.

        Playwright's own ``aria_snapshot()`` emits no refs, so on that input
        this is empty and every ref-based lookup correctly finds nothing --
        rather than collecting the whole tree under the empty string.
        """
        return {node.ref: node for node in self.nodes if node.ref}

    def get(self, ref: str) -> Node | None:
        """The node a ref points at, or ``None`` if this snapshot has no such ref.

        An empty ref matches nothing. Nodes parsed from an aria snapshot carry
        no ref at all, and without this guard ``get("")`` would return the
        first of them -- an arbitrary element, confidently.
        """
        if not ref:
            return None
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

        An *unnamed* structural role refuses to match at all. ``generic`` with
        no accessible name describes half the wrappers on any real page, so
        "the first one" is a near-arbitrary element -- and clicking the wrong
        thing silently is strictly worse than failing loudly. A recording can
        end up with such a locator when the page itself exposes nothing better;
        distillation warns about those steps so a reviewer sees it before a
        batch does.
        """
        if role in _STRUCTURAL_ROLES and not name:
            return None
        matches = self.find(role, name)
        interactive = [n for n in matches if n.interactive]
        pool = interactive or matches
        if not pool:
            return None
        if nth < 0 or nth >= len(pool):
            return None
        return pool[nth]

    def by_name(self, name: str, nth: int = 0) -> "Node | None":
        """The ``nth`` node whose accessible name matches, whatever its role.

        A label, a placeholder and an image's alt text are all the same thing
        once a page is rendered: they become the control's accessible name. So
        a recorded ``get_by_label("Password")`` is answered here rather than by
        guessing which ARIA role the control turned out to have -- guessing
        wrong means falling through to a weaker rung for no reason.

        Interactive nodes win over structural ones for the same reason
        :meth:`locate` prefers them: a ``generic`` wrapper carrying the same
        name as the input inside it must not shadow the input.
        """
        wanted = _normalise(name)
        if not wanted:
            return None

        exact = [n for n in self.nodes if n.name == name]
        loose = [n for n in self.nodes if _normalise(n.name) == wanted]
        partial = [n for n in self.nodes if wanted in _normalise(n.name)]
        pool = exact or loose or partial
        interactive = [n for n in pool if n.interactive]
        pool = interactive or pool
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
    snapshot = Snapshot(raw=text or "")
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


#: A ref with no ``ref=`` prefix, which is the spelling Playwright MCP's own
#: tools use for their ``ref`` argument -- and which the model therefore copies
#: into ``target`` as well.
BARE_REF_RE = re.compile(r"e\d+")


def is_ref(target: str) -> bool:
    """True when a recorded target is an ephemeral ref rather than a selector.

    Three spellings, because the model produces all three: ``ref=e15``,
    ``[ref=e15]``, and a bare ``e15``.

    The bare form was missing, and the consequence was not a missed
    optimisation. An unrecognised ref fell through to the text fallback and
    became ``text=e15`` -- a locator that cannot match anything, on every step
    of the recording, discovered only when the use case was replayed. The
    caller resolves against the snapshot before trusting this, so a genuine
    piece of text that happens to look like ``e15`` is still handled correctly.
    """
    value = (target or "").strip()
    return bool(
        re.fullmatch(r"\[?ref=[A-Za-z0-9]+\]?", value) or BARE_REF_RE.fullmatch(value)
    )


def extract_ref(target: str) -> str | None:
    """Pull the ref id out of ``ref=e15``, ``[ref=e15]`` or a bare ``e15``."""
    value = (target or "").strip()
    match = re.search(r"ref=([A-Za-z0-9]+)", value)
    if match:
        return match.group(1)
    return value if BARE_REF_RE.fullmatch(value) else None
