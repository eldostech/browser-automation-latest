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
    "hover", "upload", "wait", "assert", "extract", "script",
]

#: Actions whose target is an element and therefore need a locator.
ELEMENT_ACTIONS: frozenset[str] = frozenset(
    {"click", "fill", "select", "press", "hover", "upload", "extract"}
)

Status = Literal["draft", "ready", "archived"]
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


# ---------------------------------------------------------------------------
# Leaf models
# ---------------------------------------------------------------------------


class Locator(BaseModel):
    """One rung of the locator ladder.

    ``role`` is resolved against a live snapshot at replay time and is the
    durable option; the rest are recorded fallbacks in decreasing order of how
    much site churn they survive.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["role", "css", "text", "nth"]
    #: strategy="role"
    role: str | None = None
    name: str | None = None
    #: Disambiguates when several nodes share role+name, and indexes `nth`.
    nth: int = 0
    #: strategy="css"
    selector: str | None = None
    #: strategy="text"
    text: str | None = None

    @model_validator(mode="after")
    def _requires_its_own_field(self) -> "Locator":
        required = {"role": "role", "css": "selector", "text": "text", "nth": None}[self.strategy]
        if required and not getattr(self, required):
            raise ValueError(f"locator strategy {self.strategy!r} requires {required!r}")
        if self.strategy == "nth" and self.nth < 0:
            raise ValueError("locator strategy 'nth' requires a non-negative nth")
        return self

    def describe(self) -> str:
        if self.strategy == "role":
            base = f'role={self.role}' + (f' name="{self.name}"' if self.name else "")
        elif self.strategy == "css":
            base = f"css={self.selector}"
        elif self.strategy == "text":
            base = f"text={self.text!r}"
        else:
            base = f"nth={self.nth}"
        return base if self.nth == 0 or self.strategy == "nth" else f"{base} [{self.nth}]"

    @property
    def brittle(self) -> bool:
        """True for the rungs that break on cosmetic change; flagged in the UI."""
        return self.strategy in {"text", "nth"}


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
        if self.action == "wait" and self.wait_for is None:
            raise ValueError(f"step {self.id!r}: 'wait' requires wait_for")
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
    warnings: list[str] = Field(default_factory=list)

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

    @model_validator(mode="after")
    def _declared_outputs_match_extract_steps(self) -> "UseCase":
        produced = {s.output for s in self.all_steps if s.action == "extract" and s.output}
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

    def brittle_steps(self) -> list[Step]:
        """Steps whose best locator is one of the fragile rungs. Flagged in review."""
        return [s for s in self.all_steps if s.locators and s.locators[0].brittle]

    def bump(self) -> "UseCase":
        """A copy at the next version, stamped now. Versions are immutable."""
        return self.model_copy(update={"version": self.version + 1, "updated_at": _now()})
