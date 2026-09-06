"""Which tools the agent may call, and what has to be true before it does.

Two decisions live here, and the second one is the important one.

**What is offered.** Playwright MCP advertises two dozen tools. The registry
takes the ones that map onto something a `UseCase` can express, drops the ones
that only help it *find* the way, and refuses the rest by removing them from
the list the model is shown -- not by asking it in a prompt not to use them.

**What "the model cannot invent a locator" actually costs.** The design
document claimed that property came free with Playwright MCP, because the
model picks an opaque handle out of a snapshot. That was wrong, and probing a
live server is what showed it: ``browser_click``'s ``target`` is documented as
"Exact target element reference from the page snapshot, **or a unique element
selector**", and passing ``#o`` clicks the element. The model can author a
selector.

So the property is restored by enforcement. :func:`guard` refuses any ``target``
that is not ``eN``-shaped and present in what the page last reported. That
matters beyond tidiness in two ways:

* A model-authored selector is the failure mode this whole platform is built
  against -- an invented locator that matches something today and something
  else, or several things, tomorrow.
* Given a ref, the server replies with the Playwright code it ran --
  ``page.getByRole('button', { name: '+ Invite User' })`` -- which is a durable
  locator handed over for free. Given a raw selector it echoes the selector
  back. Forcing refs is therefore also what makes distillation possible later.

Nothing here imports FastAPI or ``store``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from policy import check_navigation, classify

from .provider import REF_FORMAT, ToolSpec


# ---------------------------------------------------------------------------
# What each tool becomes
# ---------------------------------------------------------------------------

#: MCP tool -> the ``Step.action`` it distils into.
#:
#: That this mapping is close to an identity is not a coincidence. Read the
#: comment above ``ELEMENT_ACTIONS`` in ``usecase.py``: it explains ``press``
#: and ``upload`` in terms of ``browser_press_key`` and ``browser_file_upload``.
#: The action vocabulary in the schema is a fossil of the original MCP design,
#: so distilling a trajectory into steps is close to a rename.
DISTILS_TO: dict[str, str] = {
    "browser_navigate": "navigate",
    "browser_navigate_back": "navigate",
    "browser_click": "click",
    "browser_type": "fill",
    "browser_fill_form": "fill_form",
    "browser_select_option": "select",
    "browser_press_key": "press",
    "browser_hover": "hover",
    "browser_file_upload": "upload",
    "browser_wait_for": "wait",
}

#: Offered, but never a step. These are how the agent *finds* the way rather
#: than the way, and carrying them into a use case would make a replay redo an
#: exploration nobody needs repeating four thousand times.
PERCEPTION: frozenset[str] = frozenset(
    {
        "browser_snapshot",
        "browser_find",
        "browser_take_screenshot",
        "browser_console_messages",
        "browser_network_requests",
        "browser_network_request",
        "browser_tabs",
        "browser_handle_dialog",
        "browser_resize",
    }
)

#: Removed from the list the model is shown, with the reason kept so a refusal
#: can say something better than "unknown tool".
REFUSED: dict[str, str] = {
    "browser_evaluate": (
        "Arbitrary JavaScript. `allow_scripts` is a privileged grant to a "
        "recording a person has read; an agent that can write JS at run time "
        "would make that gate decorative."
    ),
    "browser_run_code_unsafe": (
        "Arbitrary Playwright code against a live session that may hold "
        "someone else's credentials. Same reason as browser_evaluate."
    ),
    "browser_close": (
        "The session's lifetime belongs to the caller. An agent that can close "
        "the browser can end a run in a way nothing else expects."
    ),
    "browser_drag": "Not expressible as a step yet, so it could not be distilled.",
    "browser_drop": "Not expressible as a step yet, so it could not be distilled.",
}

#: Tools that change the page rather than read it. Gated behind an explicit
#: "let it write" on the session: an agent sent to find out how a form works
#: must not submit it on the way.
WRITES: frozenset[str] = frozenset(DISTILS_TO) - {"browser_wait_for"}

#: Arguments naming an element. All of them must hold a ref.
TARGET_KEYS: tuple[str, ...] = ("target", "startTarget", "endTarget")

#: The categories that stop and ask a person. Deliberately narrower than
#: "sensitive".
#:
#: `policy.classify` also flags credentials, file uploads and dialogs. Those
#: are worth *recording* -- they are redacted and audited either way -- but
#: they are not irreversible, and an approval prompt on every password typed
#: into a sign-in form trains people to click Allow without reading. That is
#: how an approval gate stops working, and a gate nobody reads is worse than
#: no gate because it looks like protection.
#:
#: So: what cannot be undone. Submitting, paying, deleting.
IRREVERSIBLE: frozenset[str] = frozenset(
    {"form_submit", "payment", "destructive"}
)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


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


def offered(specs: Iterable[ToolSpec]) -> list[ToolSpec]:
    """The tools the model is shown: everything advertised, minus the refused.

    Removal rather than instruction. A tool absent from the list cannot be
    called by a model that decides the rules do not apply to it, and a prompt
    saying "do not use browser_evaluate" is a request.
    """
    return [spec for spec in specs if spec.name not in REFUSED]


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


__all__ = [
    "DISTILS_TO",
    "GuardContext",
    "Guarded",
    "PERCEPTION",
    "REFUSED",
    "IRREVERSIBLE",
    "TARGET_KEYS",
    "WRITES",
    "guard",
    "offered",
]
