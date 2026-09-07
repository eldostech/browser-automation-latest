"""What each Playwright MCP tool is, for the purposes of distillation and
approval -- classified by name, since nobody here implements them.

**What "the model cannot invent a locator" actually costs.** The design
document claimed that property came free with Playwright MCP, because the
model picks an opaque handle out of a snapshot. That was wrong, and probing a
live server is what showed it: ``browser_click``'s ``target`` is documented as
"Exact target element reference from the page snapshot, **or a unique element
selector**", and passing ``#o`` clicks the element. The model can author a
selector. So the property is restored by enforcement in ``guard.py``, and
this file is the data that enforcement reads.
"""

from __future__ import annotations

from ..tools import TOOLS

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
        "browser_tabs",
        "browser_handle_dialog",
    }
)

#: Advertised by the server, and deliberately not offered.
#:
#: Not refused -- there is nothing wrong with them -- just not worth their
#: weight. **Every tool schema is re-sent on every model call**, and a real
#: session measured 7,600 tokens per call with most of it in the tool list
#: rather than in the page. These five carry large schemas and answer
#: questions an agent recording a workflow does not have: a person is watching
#: the browser, so it does not need to screenshot it, and network and console
#: dumps are for debugging a site rather than recording one.
NOT_WORTH_THE_TOKENS: frozenset[str] = frozenset(
    {
        "browser_take_screenshot",
        "browser_console_messages",
        "browser_network_requests",
        "browser_network_request",
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

#: Every tool name this file has a hand-written opinion about: Playwright's,
#: and the tools this package answers itself (`agent/tools/`). A name outside
#: this set did not come from either -- it came from a server registered
#: through the tool-server registry, and nobody here has read its source.
#: `_classify_unannotated` in `guard.py` handles those from their own
#: advertised annotations rather than by guessing at a name.
KNOWN_NAMES: frozenset[str] = (
    frozenset(DISTILS_TO) | PERCEPTION | frozenset(REFUSED)
    | NOT_WORTH_THE_TOKENS | frozenset(TOOLS)
)

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

__all__ = [
    "DISTILS_TO",
    "IRREVERSIBLE",
    "KNOWN_NAMES",
    "NOT_WORTH_THE_TOKENS",
    "PERCEPTION",
    "REFUSED",
    "TARGET_KEYS",
    "WRITES",
]
