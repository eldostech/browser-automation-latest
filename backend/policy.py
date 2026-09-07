"""Action safety policy -- the single place where "is this action sensitive?"
is decided.

The agent loop never hardcodes a tool name. It asks :func:`classify` and acts
on the answer, so tuning what needs a human in the loop means editing the
constants at the top of this file and nothing else.

Two independent checks live here:

``domain_allowed`` / ``check_navigation``
    A hard gate. A navigation outside the allowlist is refused and the refusal
    is handed back to the model as a tool error, so it can adapt rather than
    crash. This is deny-by-default: an empty allowlist blocks everything.

``classify``
    A soft gate. Sensitive actions are paused for human approval but are not
    forbidden -- the human decides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable
from fnmatch import fnmatch
from urllib.parse import urlparse

from prompt_loader import NAVIGATION_BLOCKED, render

# ---------------------------------------------------------------------------
# Tunable policy -- edit these, not the agent loop.
# ---------------------------------------------------------------------------


class Category(str, Enum):
    FORM_SUBMIT = "form_submit"
    CREDENTIALS = "credentials"
    PAYMENT = "payment"
    DESTRUCTIVE = "destructive"
    OFF_ALLOWLIST = "off_allowlist"
    CODE_EXECUTION = "code_execution"
    FILE_UPLOAD = "file_upload"
    DIALOG = "dialog"


#: Tool-name fragments that are sensitive regardless of arguments. Matched as
#: substrings against the lowercased MCP tool name, because tool names differ
#: between MCP server versions and we discover them at runtime.
SENSITIVE_TOOL_FRAGMENTS: dict[str, Category] = {
    "evaluate": Category.CODE_EXECUTION,
    "file_upload": Category.FILE_UPLOAD,
    "upload_file": Category.FILE_UPLOAD,
    "handle_dialog": Category.DIALOG,
    "install": Category.CODE_EXECUTION,
}

#: Tool-name fragments that submit something. Combined with the text/element
#: heuristics below rather than being sensitive on their own.
SUBMITTING_TOOL_FRAGMENTS: tuple[str, ...] = ("click", "press_key", "select_option", "submit")

#: Words that make a click a form submission.
#:
#: Deliberately **not** ``sign in`` / ``log in``: authenticating with a
#: credential the session was already given is not a new commitment to
#: approve, it is the use of one approved when the credential was bound to
#: the session in the first place. Typing that credential is not gated for
#: exactly this reason (see ``CREDENTIALS`` below, and ``tools.py``'s
#: ``IRREVERSIBLE`` set); gating the click that submits it right afterward
#: was the same action, split into two calls and treated inconsistently --
#: stopping a session to ask "is it OK to sign in with the login you just
#: told it to use" trains a person to click Allow without reading, which is
#: how an approval gate stops working. ``sign up`` (registration) stays
#: gated: creating a new account is a real commitment a bound credential
#: does not already cover.
SUBMIT_PATTERNS = re.compile(
    r"\b(submit|send|continue|next|save|apply|sign\s?up|"
    r"register|subscribe|post|publish|book|reserve)\b",
    re.IGNORECASE,
)

#: Words that mean money is about to move.
PAYMENT_PATTERNS = re.compile(
    r"\b(pay|payment|checkout|check\s?out|purchase|buy|order|billing|"
    r"credit\s?card|card\s?number|cvv|cvc|iban|paypal|place\s+order|"
    r"complete\s+purchase|subscribe\s+now)\b",
    re.IGNORECASE,
)

#: Words that mean data is about to be destroyed.
DESTRUCTIVE_PATTERNS = re.compile(
    r"\b(delete|remove|destroy|erase|drop|wipe|deactivate|close\s+account|"
    r"cancel\s+subscription|unsubscribe|revoke|reset)\b",
    re.IGNORECASE,
)

#: Words that mean a secret is being typed.
CREDENTIAL_PATTERNS = re.compile(
    r"\b(password|passwd|passphrase|secret|api[\s_-]?key|token|otp|"
    r"one[\s-]?time\s?code|2fa|mfa|pin|ssn|social\s+security)\b",
    re.IGNORECASE,
)

#: Argument keys that are scanned for the patterns above.
TEXT_ARG_KEYS: tuple[str, ...] = ("element", "text", "value", "ref", "selector", "name", "key", "values")

#: Argument keys that may contain a URL.
URL_ARG_KEYS: tuple[str, ...] = ("url", "href", "link", "target")

#: A value under a URL key only counts as a URL if it is shaped like one: an
#: explicit scheme, a protocol-relative URL, or a bare ``host.tld``. Models
#: routinely put element refs and CSS selectors in these keys (``ref=e15``,
#: ``button[name="play"]``) and those cannot navigate anywhere.
_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*:", re.IGNORECASE)
_BARE_HOST_RE = re.compile(
    r"^[a-z0-9\-]+(\.[a-z0-9\-]+)+(:\d+)?([/?#].*)?$", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Decision:
    """Outcome of :func:`classify`."""

    sensitive: bool
    categories: list[Category] = field(default_factory=list)
    reason: str = ""

    @property
    def category_values(self) -> list[str]:
        return [c.value for c in self.categories]


@dataclass(slots=True)
class NavigationCheck:
    allowed: bool
    url: str | None = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Domain allowlist
# ---------------------------------------------------------------------------


def normalise_domain(value: str) -> str:
    value = value.strip().lower()
    if "://" in value:
        value = urlparse(value).netloc or value
    return value.split("/")[0].split(":")[0]


def domain_allowed(url: str, allowlist: Iterable[str]) -> bool:
    """True when ``url`` is reachable under ``allowlist``.

    Rules:
      * ``*`` anywhere in the list disables the check entirely.
      * ``example.com`` matches the exact host only.
      * ``*.example.com`` matches any subdomain **and** the apex domain, which
        is what people mean in practice when they write it.
      * Any other pattern containing ``*`` or ``?`` is matched as a glob, so
        ``*localhost*`` and ``dev-*.internal`` work. This used to match
        nothing at all: only the two forms above were understood, so a pattern
        like ``*localhost*`` was silently inert while the refusal message
        listed it back verbatim -- which reads as the allowlist contradicting
        itself.
      * Non-http(s) schemes (``about:``, ``data:``, ``file:``) are refused.
      * An empty allowlist refuses everything (deny by default).

    A glob matches substrings, so ``*localhost*`` also permits
    ``notlocalhost.example.com``. That is what the pattern asks for; prefer a
    bare ``localhost`` when an exact host is what you mean.
    """
    patterns = [p.strip().lower() for p in allowlist if p and p.strip()]
    if not patterns:
        return False

    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
    except ValueError:
        # Unparseable (unbalanced brackets, bad IPv6 literal, ...). Deny rather
        # than raise -- a malformed URL must never take the whole run down.
        return False

    # The scheme is checked BEFORE the "*" escape hatch, not after. "*" means
    # "any host", not "any URL": it used to short-circuit first, so a wildcard
    # allowlist -- which people reach for while getting something working --
    # also permitted file:///etc/passwd and data: URLs. Widening which *sites*
    # are reachable should never widen what a page can be.
    if parsed.scheme not in ("http", "https"):
        return False

    if "*" in patterns:
        return True

    host = (parsed.hostname or "").lower()
    if not host:
        return False

    for pattern in patterns:
        if pattern.startswith("*."):
            base = pattern[2:]
            if host == base or host.endswith("." + base):
                return True
        elif "*" in pattern or "?" in pattern:
            if fnmatch(host, pattern):
                return True
        elif host == normalise_domain(pattern):
            return True
    return False


def looks_like_url(value: str) -> bool:
    """True when ``value`` could actually address a page.

    Deliberately permissive: anything with a scheme (``javascript:``, ``data:``)
    still counts so :func:`domain_allowed` gets to refuse it. What this filters
    out is the non-navigable junk models put in URL-named arguments -- element
    refs, CSS selectors, ``_blank`` -- which would otherwise be reported as
    allowlist violations.
    """
    if value.startswith("//"):  # protocol-relative
        return True
    return bool(_SCHEME_RE.match(value) or _BARE_HOST_RE.match(value))


def extract_urls(arguments: dict[str, Any]) -> list[str]:
    """Pull every URL-looking value out of a tool-call argument object."""
    found: list[str] = []

    def walk(node: Any, key_hint: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, key.lower())
        elif isinstance(node, list):
            for item in node:
                walk(item, key_hint)
        elif isinstance(node, str):
            candidate = node.strip()
            if key_hint in URL_ARG_KEYS and looks_like_url(candidate):
                found.append(candidate)
            elif candidate.startswith(("http://", "https://")):
                found.append(candidate)

    walk(arguments)
    return found


def check_navigation(
    tool_name: str, arguments: dict[str, Any], allowlist: Iterable[str]
) -> NavigationCheck:
    """Refuse any tool call that would leave the allowlist.

    Applied to *every* tool call rather than to a navigation tool by name --
    a tool we have never seen could still accept a URL.
    """
    for url in extract_urls(arguments):
        if not domain_allowed(url, allowlist):
            return NavigationCheck(
                allowed=False,
                url=url,
                reason=render(
                    NAVIGATION_BLOCKED,
                    url=repr(url),
                    allowlist=", ".join(allowlist) or "empty",
                ),
            )
    return NavigationCheck(allowed=True)


# ---------------------------------------------------------------------------
# Sensitivity classification
# ---------------------------------------------------------------------------


def _argument_text(arguments: dict[str, Any]) -> str:
    """Flatten the human-meaningful parts of an argument object into one string."""
    chunks: list[str] = []

    def walk(node: Any, key_hint: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, key.lower())
        elif isinstance(node, list):
            for item in node:
                walk(item, key_hint)
        elif isinstance(node, (str, int, float)):
            if key_hint in TEXT_ARG_KEYS or not key_hint:
                chunks.append(str(node))
            else:
                # Keep the key too: `password: "hunter2"` should trip the
                # credential rule even though the value looks innocuous.
                chunks.append(f"{key_hint} {node}")

    walk(arguments)
    return " ".join(chunks)


def classify(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    allowlist: Iterable[str] = (),
) -> Decision:
    """Decide whether a proposed tool call needs a human to approve it."""
    name = (tool_name or "").lower()
    haystack = _argument_text(arguments or {})
    categories: list[Category] = []
    reasons: list[str] = []

    for fragment, category in SENSITIVE_TOOL_FRAGMENTS.items():
        if fragment in name:
            categories.append(category)
            reasons.append(f"tool '{tool_name}' is always gated ({category.value})")

    is_interaction = any(fragment in name for fragment in SUBMITTING_TOOL_FRAGMENTS)
    is_typing = "type" in name or "fill" in name

    if CREDENTIAL_PATTERNS.search(haystack):
        categories.append(Category.CREDENTIALS)
        reasons.append("the arguments look like credentials or a one-time code")

    if PAYMENT_PATTERNS.search(haystack):
        categories.append(Category.PAYMENT)
        reasons.append("the arguments mention payment or checkout")

    if DESTRUCTIVE_PATTERNS.search(haystack) and (is_interaction or is_typing):
        categories.append(Category.DESTRUCTIVE)
        reasons.append("the action appears to delete or destroy data")

    if is_interaction and SUBMIT_PATTERNS.search(haystack):
        categories.append(Category.FORM_SUBMIT)
        reasons.append("the action appears to submit a form")

    # Pressing Enter is the other way to submit a form.
    if "press_key" in name and str(arguments.get("key", "")).lower() in ("enter", "return"):
        categories.append(Category.FORM_SUBMIT)
        reasons.append("pressing Enter can submit the focused form")

    navigation = check_navigation(tool_name, arguments or {}, allowlist)
    if not navigation.allowed:
        categories.append(Category.OFF_ALLOWLIST)
        reasons.append(navigation.reason)

    deduped: list[Category] = []
    for category in categories:
        if category not in deduped:
            deduped.append(category)

    return Decision(
        sensitive=bool(deduped),
        categories=deduped,
        reason="; ".join(dict.fromkeys(reasons)) if reasons else "",
    )
