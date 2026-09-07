"""What has to be true before a proposed tool call reaches the browser.

Two decisions live here, and the second one is the important one.

**What is offered.** Playwright MCP advertises two dozen tools. `offered()`
takes the ones that map onto something a `UseCase` can express, drops the
ones that only help it *find* the way, and refuses the rest by removing them
from the list the model is shown -- not by asking it in a prompt not to use
them.

**What may run without a person, and what has to wait for one.** :func:`guard`
refuses any ``target`` that is not ``eN``-shaped and present in what the page
last reported -- see ``catalog.py``'s docstring on why that check exists at
all -- and separately decides whether an *allowed* call still needs a human
to say yes, which is what the interrupt in ``graph.py`` waits on.

Nothing here imports FastAPI or ``store``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from policy import check_navigation, classify

from ..providers.base import REF_FORMAT, ToolSpec
from .catalog import IRREVERSIBLE, KNOWN_NAMES, NOT_WORTH_THE_TOKENS, REFUSED, TARGET_KEYS, WRITES


@dataclass(slots=True)
class Guarded:
    """The verdict on one proposed call.

    A refusal is handed back to the model as a tool error rather than raised:
    an agent told *why* it may not do something can choose differently, and an
    agent that crashes cannot. The audit entry is written either way.
    """

    allowed: bool
    reason: str = ""
    #: True when a human has to say yes before this runs. The rendezvous itself
    #: arrives with the graph; the classification is decided here so that the
    #: decision is in code rather than in the model's opinion of its own action.
    needs_approval: bool = False
    category: str = ""


@dataclass(slots=True)
class GuardContext:
    """What the guard needs to know that is not in the call itself."""

    #: Hosts this session may touch. Deny-by-default: empty blocks everything.
    allowed_domains: tuple[str, ...] = ()
    #: Refs the page has reported so far. A target outside this set is either
    #: stale or invented, and both are refusals.
    known_refs: frozenset[str] = frozenset()
    #: Whether the session was started with permission to change anything.
    may_write: bool = False
    #: Tools the server actually advertised, so a typo is refused as a typo.
    available: frozenset[str] = frozenset()
    #: Annotations for tools outside `KNOWN_NAMES` -- a tool from a
    #: registered MCP server, keyed by its (already-prefixed) name. Playwright
    #: and this codebase's own tools do not need an entry here: they are
    #: classified by name, not by annotation.
    annotations: dict[str, dict[str, bool]] = field(default_factory=dict)


def offered(specs: Iterable[ToolSpec]) -> list[ToolSpec]:
    """The tools the model is shown: everything advertised, minus the refused.

    Removal rather than instruction. A tool absent from the list cannot be
    called by a model that decides the rules do not apply to it, and a prompt
    saying "do not use browser_evaluate" is a request.
    """
    return [
        spec
        for spec in specs
        if spec.name not in REFUSED and spec.name not in NOT_WORTH_THE_TOKENS
    ]


def guard(name: str, arguments: dict[str, Any], ctx: GuardContext) -> Guarded:
    """Everything that must be true before a tool call reaches the browser.

    Ordered cheapest-first, and by how badly the answer is wanted: an unknown
    tool is a mistake worth naming plainly, a refused one deserves its reason,
    and the allowlist is the hard gate that must run before anything touches
    the network.
    """
    if name in REFUSED:
        return Guarded(False, f"{name} is not available. {REFUSED[name]}")
    if ctx.available and name not in ctx.available:
        known = ", ".join(sorted(ctx.available))
        return Guarded(False, f"There is no tool called {name!r}. Available: {known}.")

    # The hard gate, and it is applied to every call rather than to a
    # navigation tool by name: a tool we have never seen could still take a URL.
    navigation = check_navigation(name, arguments, ctx.allowed_domains)
    if not navigation.allowed:
        return Guarded(False, navigation.reason)

    stale = _bad_targets(arguments, ctx.known_refs)
    if stale:
        return Guarded(False, stale)

    if name not in KNOWN_NAMES:
        # A tool from a registered MCP server. Nobody here has read its
        # source, so it is classified from what it chose to declare about
        # itself rather than from a name this file recognises -- and the
        # moment that declaration is silent, the answer is the conservative
        # one, the same way `available` above is deny-by-default.
        return _classify_unannotated(name, ctx.annotations.get(name, {}), ctx.may_write)

    if name in WRITES and not ctx.may_write:
        return Guarded(
            False,
            f"{name} would change the page, and this session was started "
            "read-only. Ask for write access if the task needs it.",
        )

    # A soft gate, and the last one: it does not refuse, it marks. `classify`
    # was written for the agent that used to live here and has been unused
    # since; it reads the arguments for submits, payments, deletions and
    # credentials rather than asking the model what it thinks of its own next
    # action, which is the only version of this check worth having.
    decision = classify(name, arguments)
    categories = decision.category_values
    return Guarded(
        True,
        needs_approval=any(c in IRREVERSIBLE for c in categories),
        category=", ".join(categories),
    )


def _classify_unannotated(
    name: str, annotations: dict[str, bool], may_write: bool
) -> Guarded:
    """Whether a tool this file has never heard of may run, and whether it
    needs a person first.

    The MCP protocol's tool annotations are hints a server *may* declare, not
    a guarantee, so absence means "unknown" and unknown is treated as the
    worst case: a write that also needs approval. Only a server that
    explicitly says ``readOnlyHint: true`` skips both gates. This is the same
    "deny-by-default: empty blocks everything" rule `GuardContext` already
    documents for the allowlist -- applied here to the one thing about a new
    tool nobody in this codebase has read the source of.
    """
    if annotations.get("readOnlyHint") is True:
        return Guarded(True)
    if not may_write:
        return Guarded(
            False,
            f"{name} is not known to be read-only, and this session was "
            "started read-only. Ask for write access if the task needs it.",
        )
    return Guarded(True, needs_approval=True, category="unannotated_write")


def _bad_targets(arguments: dict[str, Any], known: frozenset[str]) -> str:
    """Why this call's element references are not usable, or "".

    Two distinct failures with one message each, because they need different
    fixes: a selector means "take a snapshot and point at what you find", and
    an unknown ref means "your snapshot is stale".
    """
    for key in TARGET_KEYS:
        value = arguments.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not REF_FORMAT.match(value):
            return (
                f"{key}={value!r} is not an element reference. This server also "
                "accepts a raw selector there and it is not allowed here: a "
                "selector a model composed is the invented locator this platform "
                "exists to avoid, and only a ref makes the server report the "
                "durable locator it used. Take a snapshot and pass a ref such "
                "as 'e12'."
            )
        if known and value not in known:
            return (
                f"{key}={value!r} is not on the page as it now stands. Refs are "
                "only valid for the snapshot they came from; take a fresh one."
            )
    return ""


__all__ = ["Guarded", "GuardContext", "guard", "offered"]
