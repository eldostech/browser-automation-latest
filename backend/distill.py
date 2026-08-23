"""Turns a successful run into a replayable :class:`usecase.UseCase`.

Distillation is two passes, and the split is the whole point:

**Pass 1 -- the pre-filter, in this module, with no LLM.** Everything that can
be decided by looking at the data is decided here: which calls actually
succeeded, which were pure observation, which retries were attempts at the same
thing, and what each ephemeral ``ref`` pointed at. On the sample run this turns
34 tool calls into about 8 candidate steps.

**Pass 2 -- one LLM call**, in :func:`distill`, over that short list. The model
only does what genuinely needs judgement: naming things, deciding which
literals are per-row inputs and which are credentials, splitting setup from
per-row work, and proposing assertions.

Doing it the other way round -- handing the model the raw event log -- would
cost roughly what the original run cost, which would defeat the purpose of the
feature at the moment of creating it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import ValidationError

from events import AgentEvent
from prompt_loader import DISTILL, load
from snapshot import Snapshot, extract_ref, is_ref, parse as parse_snapshot
from usecase import (
    Assertion,
    FormField,
    InputSpec,
    Locator,
    SecretSpec,
    Step,
    UseCase,
    WaitFor,
)

log = logging.getLogger(__name__)

#: Tools that only ever *observe*. They exist to feed the model a view of the
#: page; with no model in the loop they carry no action at all.
OBSERVATION_TOOLS: frozenset[str] = frozenset(
    {
        "browser_snapshot",
        "browser_take_screenshot",
        "browser_screenshot",
        "browser_console_messages",
        "browser_network_requests",
        "browser_network_request",
        "browser_find",
    }
)

#: Tool name -> use-case action. Anything absent is carried through as a
#: `script`-like unknown and flagged for review rather than silently dropped.
TOOL_ACTIONS: dict[str, str] = {
    "browser_navigate": "navigate",
    "browser_navigate_back": "navigate_back",
    "browser_click": "click",
    "browser_type": "fill",
    "browser_fill_form": "fill_form",
    "browser_select_option": "select",
    "browser_press_key": "press",
    "browser_hover": "hover",
    "browser_file_upload": "upload",
    "browser_wait_for": "wait",
    "browser_evaluate": "extract",
    "browser_run_code_unsafe": "script",
    "browser_drag": "drag",
    "browser_drop": "drop",
    "browser_handle_dialog": "dialog",
    "browser_tabs": "tabs",
    "browser_resize": "resize",
}

#: Argument keys that have held an element target across playwright-mcp
#: versions. Checked in order.
TARGET_KEYS: tuple[str, ...] = ("target", "selector", "ref", "element")

#: Argument keys that have held a typed value.
VALUE_KEYS: tuple[str, ...] = ("text", "value", "key", "values", "paths")

#: A tool result that starts with one of these is a failure the server reported
#: without setting isError -- observed in real runs, so it must be caught.
_SOFT_FAILURE_RE = re.compile(
    r"^\s*(?:###\s*)?(?:Error|Failed|TimeoutError|locator\.\w+:)", re.IGNORECASE
)

#: Ladder order, most durable first. `role` is resolved against a live snapshot
#: and survives markup churn; `nth` breaks the moment anything is reordered.
_STRATEGY_RANK: dict[str, int] = {"role": 0, "css": 1, "text": 2, "nth": 3}


def ladder(locators: Iterable[Locator]) -> list[Locator]:
    """De-duplicate and order locators by durability.

    Recording order is *arrival* order, which reflects how long the model
    flailed rather than which locator is best. The ladder has to be ordered by
    how much site churn each rung survives, or the executor would try the
    fragile rung first and only fall back to the durable one.
    """
    unique: list[Locator] = []
    for locator in locators:
        if locator is not None and locator not in unique:
            unique.append(locator)
    return sorted(unique, key=lambda loc: _STRATEGY_RANK.get(loc.strategy, 9))


@dataclass(slots=True)
class RecordedStep:
    """One surviving tool call, with its ref already resolved to role + name."""

    call_id: str
    step: int
    seq: int
    tool: str
    action: str
    arguments: dict[str, Any]
    description: str = ""
    locators: list[Locator] = field(default_factory=list)
    rejected_locators: list[Locator] = field(default_factory=list)
    value: str | None = None
    url: str | None = None
    #: ``fill_form`` only: one entry per field, each with its own locator ladder.
    fields: list[dict[str, Any]] = field(default_factory=list)
    #: Set when a ref could not be resolved against any snapshot.
    unresolved_ref: str | None = None
    result_text: str = ""

    @staticmethod
    def _dump(locator: Locator) -> dict[str, Any]:
        return locator.model_dump(exclude_none=True, exclude_defaults=True) or {
            "strategy": locator.strategy
        }

    def to_prompt_dict(self) -> dict[str, Any]:
        """The compact shape handed to the model. Deliberately small."""
        payload: dict[str, Any] = {"id": f"s{self.step}", "action": self.action}
        if self.description:
            payload["describes"] = self.description
        if self.locators:
            payload["locators"] = [self._dump(loc) for loc in self.locators]
        if self.url:
            payload["url"] = self.url
        if self.value is not None:
            payload["value"] = self.value
        if self.fields:
            payload["fields"] = [
                {
                    "name": f["name"],
                    "value": f["value"],
                    "type": f.get("type", "textbox"),
                    "locators": [self._dump(loc) for loc in f["locators"]],
                }
                for f in self.fields
            ]
        if self.action == "script":
            payload["code"] = (self.arguments.get("code") or "")[:600]
        if self.unresolved_ref:
            payload["warning"] = f"ref {self.unresolved_ref} could not be resolved"
        return payload

    def all_values(self) -> list[str]:
        """Every literal this step typed into the page."""
        values = [self.value] if self.value is not None else []
        values += [f["value"] for f in self.fields if f.get("value")]
        return [v for v in values if v]


@dataclass(slots=True)
class PreFilterResult:
    steps: list[RecordedStep]
    warnings: list[str]
    #: Literal values worth parameterising, in the order first seen.
    literals: list[str]
    start_url: str | None
    domains: list[str]
    #: Counters for the "34 calls -> 8 steps" line in the UI.
    stats: dict[str, int]

    def to_prompt_payload(self) -> list[dict[str, Any]]:
        return [step.to_prompt_dict() for step in self.steps]


# ---------------------------------------------------------------------------
# Pre-filter
# ---------------------------------------------------------------------------


def _argument(arguments: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in arguments and arguments[key] not in (None, ""):
            return arguments[key]
    return None


def _result_failed(event: Any) -> bool:
    """True when a tool result represents a failure.

    ``ok`` is the primary signal, but Playwright MCP also reports some failures
    as ordinary text -- a locator timeout comes back as prose with ``ok`` true.
    Both are treated as failures, because replaying an action that did not work
    is worse than dropping one that did.
    """
    if not getattr(event, "ok", True):
        return True
    return bool(_SOFT_FAILURE_RE.match(getattr(event, "text", "") or ""))


def _locators_for(target: Any, snapshot: Snapshot | None) -> tuple[list[Locator], str | None, str]:
    """Build the locator ladder for one recorded target.

    Returns ``(locators, unresolved_ref, description)``. A ref resolves to a
    role locator via the snapshot captured just before the step; a selector is
    kept as a css rung; anything else is treated as text.
    """
    if not isinstance(target, str) or not target.strip():
        return [], None, ""

    target = target.strip()

    if is_ref(target):
        ref = extract_ref(target)
        node = snapshot.get(ref) if (snapshot and ref) else None
        if node is None:
            return [], ref, ""
        return (
            [Locator(strategy="role", role=node.role, name=node.name or None)],
            None,
            node.describe(),
        )

    # A CSS selector, which is what the model converges on once refs fail.
    if re.search(r"[#.\[\]>:]|^[a-z]+$", target):
        return [Locator(strategy="css", selector=target)], None, ""

    return [Locator(strategy="text", text=target)], None, ""


def _same_target(a: RecordedStep, b: RecordedStep) -> bool:
    """Whether two steps are attempts at the same thing.

    Same action, and either the same resolved element or the same human
    description. This is what collapses the five-attempt radio-button cluster
    in the sample run down to the one call that worked.
    """
    if a.action != b.action:
        return False
    if a.description and a.description == b.description:
        return True
    if a.locators and b.locators:
        first, second = a.locators[0], b.locators[0]
        if first.strategy == second.strategy == "role":
            return (first.role, first.name) == (second.role, second.name)
    # Typing the same value into the same field, reached two different ways.
    return bool(a.value and a.value == b.value and a.action in {"fill", "select"})


def pre_filter(events: list[AgentEvent]) -> PreFilterResult:
    """Reduce a run's events to the calls that actually did something.

    Deterministic and cheap. Everything decidable from the data is decided
    here so the LLM call that follows stays small.
    """
    results: dict[str, Any] = {}
    for event in events:
        if event.type == "tool_result":
            results[event.call_id] = event

    # Snapshots indexed by the seq at which they were captured, so a step can
    # find the most recent view of the page that preceded it.
    snapshots: list[tuple[int, Snapshot]] = []
    for event in events:
        if event.type == "tool_result" and event.name in {"browser_snapshot", "browser_navigate"}:
            parsed = parse_snapshot(event.text or "")
            if len(parsed):
                snapshots.append((event.seq, parsed))

    def snapshot_before(seq: int) -> Snapshot | None:
        best: Snapshot | None = None
        for captured_at, snap in snapshots:
            if captured_at <= seq:
                best = snap
            else:
                break
        return best

    warnings: list[str] = []
    literals: list[str] = []
    domains: list[str] = []
    start_url: str | None = None
    stats = {"tool_calls": 0, "failed": 0, "observation": 0, "collapsed": 0, "kept": 0}

    kept: list[RecordedStep] = []

    for event in events:
        if event.type != "tool_call":
            continue
        stats["tool_calls"] += 1

        if event.name in OBSERVATION_TOOLS:
            stats["observation"] += 1
            continue

        result = results.get(event.call_id)
        if result is None or _result_failed(result):
            stats["failed"] += 1
            continue

        arguments = dict(event.arguments or {})
        action = TOOL_ACTIONS.get(event.name, "script" if "code" in arguments else "unknown")
        if action == "unknown":
            warnings.append(
                f"step {event.step}: no action mapping for tool {event.name!r}; review it by hand"
            )

        target = _argument(arguments, TARGET_KEYS)
        locators, unresolved, described = _locators_for(target, snapshot_before(event.seq))
        if unresolved:
            warnings.append(
                f"step {event.step}: ref {unresolved!r} did not appear in any preceding "
                "snapshot, so no durable locator could be derived"
            )

        raw_value = _argument(arguments, VALUE_KEYS)
        value = None if raw_value is None else str(raw_value)

        # fill_form carries one target per field rather than one per call, so
        # each field needs its own ladder. This is the shape a login actually
        # takes in a recorded run, so getting it wrong loses the sign-in.
        form_fields: list[dict[str, Any]] = []
        if action == "fill_form":
            snapshot_here = snapshot_before(event.seq)
            for raw_field in arguments.get("fields") or []:
                if not isinstance(raw_field, dict):
                    continue
                field_locators, field_unresolved, field_described = _locators_for(
                    _argument(raw_field, TARGET_KEYS), snapshot_here
                )
                if field_unresolved:
                    warnings.append(
                        f"step {event.step}: form field "
                        f"{raw_field.get('name') or '?'!r} used ref {field_unresolved!r}, "
                        "which did not appear in any preceding snapshot"
                    )
                form_fields.append(
                    {
                        "name": str(raw_field.get("name") or field_described or "field"),
                        "value": str(raw_field.get("value") or ""),
                        "type": str(raw_field.get("type") or "textbox"),
                        "locators": field_locators,
                    }
                )

        url = arguments.get("url") if action == "navigate" else None
        if url:
            if start_url is None:
                start_url = url
            host = re.sub(r"^https?://", "", url).split("/")[0].split(":")[0]
            if host and host not in domains:
                domains.append(host)

        step = RecordedStep(
            call_id=event.call_id,
            step=event.step,
            seq=event.seq,
            tool=event.name,
            action=action,
            arguments=arguments,
            description=(arguments.get("element") or described or "").strip(),
            locators=locators,
            value=value,
            url=url,
            fields=form_fields,
            unresolved_ref=unresolved,
            result_text=(getattr(result, "text", "") or "")[:400],
        )

        # Collapse a retry cluster: one goal, attempted several ways, every
        # attempt having succeeded. The later action wins, but the locators
        # merge -- each surviving attempt is a free extra rung on the ladder.
        if kept and _same_target(kept[-1], step):
            previous = kept.pop()
            stats["collapsed"] += 1
            step.locators = ladder([*previous.locators, *step.locators])
            step.rejected_locators = ladder(
                [
                    loc
                    for loc in (*previous.rejected_locators, *step.rejected_locators)
                    if loc not in step.locators
                ]
            )
            if not step.description:
                step.description = previous.description

        kept.append(step)

        for candidate in [*step.all_values(), url]:
            if candidate and candidate not in literals:
                literals.append(candidate)

    # Order every ladder by durability rather than by the order the model
    # happened to arrive at each locator.
    for step in kept:
        step.locators = ladder(step.locators)
        for item in step.fields:
            item["locators"] = ladder(item["locators"])

    stats["kept"] = len(kept)
    log.info(
        "pre-filter complete",
        extra={k: v for k, v in stats.items()},
    )
    return PreFilterResult(
        steps=kept,
        warnings=warnings,
        literals=literals,
        start_url=start_url,
        domains=domains,
        stats=stats,
    )


def summarise(result: PreFilterResult) -> str:
    """One line for logs and the review UI."""
    s = result.stats
    return (
        f"{s['tool_calls']} tool calls -> {s['kept']} steps "
        f"({s['failed']} failed, {s['observation']} observation-only, "
        f"{s['collapsed']} retries collapsed)"
    )


def as_prompt_json(result: PreFilterResult, task: str) -> str:
    """The compact JSON document the distiller prompt embeds."""
    return json.dumps(
        {
            "task": task,
            "start_url": result.start_url,
            "observed_domains": result.domains,
            "literal_values": result.literals,
            "steps": result.to_prompt_payload(),
        },
        indent=2,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Pass 2: the single LLM call
# ---------------------------------------------------------------------------

#: The tool the model must call, exactly once. Deliberately narrow: the model
#: selects and parameterises recorded steps *by id* and cannot express a
#: locator at all. That makes it structurally impossible for a hallucinated
#: selector to reach a use case -- every locator in the output came from a call
#: that actually succeeded during the recording.
BUILD_TOOL: dict[str, Any] = {
    "name": "build_usecase",
    "description": (
        "Assemble the reusable use case from the recording. Call exactly once. "
        "Reference recorded steps by their id; never invent locators."
    ),
    "input_schema": {
        "type": "object",
        "required": ["name", "row_step_ids"],
        "properties": {
            "name": {"type": "string", "description": "Short, specific name for the task."},
            "description": {"type": "string"},
            "inputs": {
                "type": "array",
                "description": "Values that change from row to row.",
                "items": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
                        "type": {"enum": ["string", "url", "number", "boolean", "file"]},
                        "required": {"type": "boolean"},
                        "description": {"type": "string"},
                        "example": {"type": "string"},
                    },
                },
            },
            "secrets": {
                "type": "array",
                "description": "Credential slots. Never include the value itself.",
                "items": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
                        "required": {"type": "boolean"},
                        "description": {"type": "string"},
                    },
                },
            },
            "setup_step_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Recorded step ids that run ONCE per batch (sign-in).",
            },
            "row_step_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Recorded step ids that run once per input row.",
            },
            "teardown_step_ids": {"type": "array", "items": {"type": "string"}},
            "values": {
                "type": "object",
                "description": (
                    "Replacement for a step's typed value, keyed by step id. Use "
                    "{{input.name}} or {{secret.name}}. For a fill_form step, key by "
                    "'<step_id>.<field name>'."
                ),
                "additionalProperties": {"type": "string"},
            },
            "urls": {
                "type": "object",
                "description": "Replacement URL for a navigate step, keyed by step id.",
                "additionalProperties": {"type": "string"},
            },
            "descriptions": {
                "type": "object",
                "description": "Human-readable description for a step, keyed by step id.",
                "additionalProperties": {"type": "string"},
            },
            "assertions": {
                "type": "array",
                "description": "Checks to insert. Placed immediately after the named step.",
                "items": {
                    "type": "object",
                    "required": ["after_step_id", "kind", "value"],
                    "properties": {
                        "after_step_id": {"type": "string"},
                        "kind": {"enum": ["url_contains", "text_present", "title_contains"]},
                        "value": {"type": "string"},
                        "negate": {"type": "boolean"},
                    },
                },
            },
            "session_check": {
                "type": "object",
                "description": "Proof the shared session is still signed in.",
                "required": ["kind", "value"],
                "properties": {
                    "kind": {"enum": ["url_contains", "text_present", "title_contains"]},
                    "value": {"type": "string"},
                    "negate": {"type": "boolean"},
                },
            },
            "row_reset_url": {
                "type": "string",
                "description": "URL opened before each row. May be templated.",
            },
            "outputs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of values extracted per row, if any.",
            },
        },
    },
}


class DistillationError(RuntimeError):
    """The model did not return a usable plan."""


#: Recorded actions the executor knows how to perform. Anything else is
#: reported for human attention rather than quietly dropped.
ACTIONS_WE_CAN_REPLAY: frozenset[str] = frozenset(
    {"navigate", "click", "fill", "fill_form", "select", "press", "hover",
     "upload", "wait", "extract", "script"}
)


def _frozen_literals(steps: Iterable[Step], literals: list[str]) -> list[str]:
    """Recorded values baked into script code, and therefore fixed for every row.

    A `script` step is opaque: nothing can substitute into JavaScript, so any
    literal inside it is frozen at whatever the recording happened to do. When
    those literals were *answers* rather than settings, the task is not
    replayable at all -- and that is much better said out loud than discovered
    on row 400.
    """
    code = "\n".join(step.code or "" for step in steps if step.action == "script")
    if not code:
        return []
    found: list[str] = []
    for literal in literals:
        # Short values match too eagerly inside code; a bare "2" is noise.
        if len(literal) >= 2 and literal in code and literal not in found:
            found.append(literal)
    return found


def _assertion_from(spec: dict[str, Any]) -> Assertion:
    return Assertion(
        kind=spec["kind"],
        value=str(spec.get("value") or ""),
        negate=bool(spec.get("negate", False)),
    )


def _build_step(recorded: RecordedStep, plan: dict[str, Any], *, step_id: str) -> Step | None:
    """Turn one recorded call into a :class:`usecase.Step`, applying the plan.

    Returns ``None`` for a recorded action with no executable equivalent; the
    caller reports that as a warning rather than dropping it silently.
    """
    values: dict[str, str] = plan.get("values") or {}
    urls: dict[str, str] = plan.get("urls") or {}
    descriptions: dict[str, str] = plan.get("descriptions") or {}

    action = recorded.action
    if action not in ACTIONS_WE_CAN_REPLAY:
        return None

    description = descriptions.get(step_id) or recorded.description or ""

    if action == "fill_form":
        fields = [
            FormField(
                name=item["name"],
                # A per-field override is keyed "<step id>.<field name>"; a
                # whole-step override applies to a single-field form.
                value=values.get(f"{step_id}.{item['name']}", values.get(step_id, item["value"])),
                type=item.get("type", "textbox"),
                locators=list(item["locators"]),
            )
            for item in recorded.fields
        ]
        return Step(
            id=step_id,
            action="fill_form",
            description=description,
            fields=fields,
            locators=list(recorded.locators),
        )

    kwargs: dict[str, Any] = {
        "id": step_id,
        "action": action,
        "description": description,
        "locators": list(recorded.locators),
        "rejected_locators": list(recorded.rejected_locators),
    }

    if action == "navigate":
        kwargs["url"] = urls.get(step_id, recorded.url)
    elif action == "script":
        kwargs["code"] = recorded.arguments.get("code") or ""
    elif action == "wait":
        seconds = recorded.arguments.get("time")
        kwargs["wait_for"] = (
            WaitFor(kind="time", seconds=float(seconds))
            if isinstance(seconds, (int, float))
            else WaitFor(kind="text", value=str(recorded.value or ""))
        )
    elif action == "extract":
        kwargs["output"] = recorded.arguments.get("output") or f"{step_id}_value"
    elif recorded.value is not None or step_id in values:
        kwargs["value"] = values.get(step_id, recorded.value)

    return Step(**kwargs)


def build_usecase(
    plan: dict[str, Any],
    pre: PreFilterResult,
    *,
    source_run_id: str | None = None,
    task: str = "",
) -> UseCase:
    """Assemble a :class:`UseCase` from the model's plan and the recording.

    Deterministic and separately testable: the same plan over the same
    recording always yields the same use case. It is also the only path from a
    plan to a use case, which is what guarantees every locator in the result
    came from a call that actually succeeded -- the model cannot express one.
    """
    by_id = {f"s{step.step}": step for step in pre.steps}
    warnings = list(pre.warnings)

    def collect(key: str) -> list[tuple[str, RecordedStep]]:
        chosen: list[tuple[str, RecordedStep]] = []
        for raw_id in plan.get(key) or []:
            recorded = by_id.get(str(raw_id))
            if recorded is None:
                warnings.append(f"the plan referenced unknown step {raw_id!r}; ignored")
                continue
            chosen.append((str(raw_id), recorded))
        return chosen

    setup_pairs = collect("setup_step_ids")
    row_pairs = collect("row_step_ids")
    teardown_pairs = collect("teardown_step_ids")

    claimed = {sid for sid, _ in (*setup_pairs, *row_pairs, *teardown_pairs)}
    dropped = [sid for sid in by_id if sid not in claimed]
    if dropped:
        warnings.append(
            f"{len(dropped)} recorded step(s) dropped by the plan: " + ", ".join(dropped)
        )

    assertions_by_step: dict[str, list[dict[str, Any]]] = {}
    for spec in plan.get("assertions") or []:
        assertions_by_step.setdefault(str(spec.get("after_step_id")), []).append(spec)

    counter = {"n": 0}

    def build(pairs: list[tuple[str, RecordedStep]]) -> list[Step]:
        built: list[Step] = []
        for step_id, recorded in pairs:
            try:
                step = _build_step(recorded, plan, step_id=step_id)
            except ValidationError as exc:
                # One step the schema will not accept must not cost the whole
                # recording. Drop it, say so loudly, and let the reviewer decide
                # whether what is left is still worth publishing.
                reasons = "; ".join(
                    str(error.get("msg", "")).removeprefix("Value error, ")
                    for error in exc.errors()
                )
                warnings.append(
                    f"{step_id}: recorded {recorded.tool!r} could not be turned into a "
                    f"replayable step and was left out ({reasons})"
                )
                log.warning(
                    "dropped an unbuildable step",
                    extra={"step_id": step_id, "tool": recorded.tool, "reasons": reasons},
                )
                continue
            if step is None:
                warnings.append(
                    f"{step_id}: recorded action {recorded.action!r} (tool {recorded.tool!r}) "
                    "has no replayable equivalent and was left out"
                )
                continue
            built.append(step)
            # An assertion is woven in immediately after the step it verifies.
            for spec in assertions_by_step.get(step_id, []):
                counter["n"] += 1
                built.append(
                    Step(
                        id=f"{step_id}_check{counter['n']}",
                        action="assert",
                        description="verify " + str(spec.get("value", "")),
                        assertion=_assertion_from(spec),
                    )
                )
        return built

    setup_steps = build(setup_pairs)
    row_steps = build(row_pairs)
    teardown_steps = build(teardown_pairs)

    row_reset = None
    if plan.get("row_reset_url"):
        row_reset = Step(
            id="row_reset",
            action="navigate",
            description="return to a known state before each row",
            url=str(plan["row_reset_url"]),
        )

    session_check = _assertion_from(plan["session_check"]) if plan.get("session_check") else None

    # An assertion the allowlist makes impossible would fail every row while
    # blaming the page, so drop it here rather than storing a use case that
    # cannot succeed. Dropping loses nothing: it could never have passed.
    #
    # This runs *before* the "no assertions" check below, so a use case that
    # loses its only check is reported as having none -- which is the truth.
    def usable(steps: list[Step]) -> list[Step]:
        kept: list[Step] = []
        for step in steps:
            reason = (
                step.assertion.unsatisfiable_reason(pre.domains)
                if step.assertion is not None
                else None
            )
            if reason:
                warnings.append(f"dropped assertion {step.id}: it {reason}")
                continue
            kept.append(step)
        return kept

    setup_steps = usable(setup_steps)
    row_steps = usable(row_steps)
    teardown_steps = usable(teardown_steps)

    if session_check is not None:
        reason = session_check.unsatisfiable_reason(pre.domains)
        if reason:
            warnings.append(
                f"dropped the session check: it {reason}. Without one, a session that drops "
                "part-way through a batch will not be noticed."
            )
            session_check = None

    if not any(step.action == "assert" for step in (*setup_steps, *row_steps)):
        warnings.append(
            "no assertions were proposed. A replay has no judgement, so a batch will report "
            "success even when a row silently did nothing. Add at least one before publishing."
        )

    all_steps = (*setup_steps, *row_steps, *teardown_steps)

    scripts = [s.id for s in all_steps if s.action == "script"]
    if scripts:
        warnings.append(
            "contains raw-JavaScript step(s) "
            + ", ".join(scripts)
            + ". Read the code, then set allow_scripts if they are genuinely needed."
        )

    # An input nothing reads is worse than useless: it demands a value per row
    # and then ignores it. This happens when the model parameterises a literal
    # that lives inside `script` code, which nothing can substitute into.
    declared = [InputSpec(**spec) for spec in (plan.get("inputs") or [])]
    referenced = {name for step in all_steps for kind, name in step.references() if kind == "input"}
    if row_reset is not None:
        referenced |= {name for kind, name in row_reset.references() if kind == "input"}

    used = [spec for spec in declared if spec.name in referenced]
    unused = [spec.name for spec in declared if spec.name not in referenced]
    if unused:
        warnings.append(
            "dropped input(s) that no step reads: "
            + ", ".join(unused)
            + ". They were removed rather than asked for on every row. If these values really "
            "do change per record, the step that uses them has to read them -- which a "
            "raw-JavaScript step cannot do, because nothing substitutes into code."
        )

    # Literals frozen inside script code, so a reviewer can see what will be
    # identical on every single row.
    frozen = _frozen_literals(all_steps, pre.literals)
    if frozen:
        warnings.append(
            "these recorded values are hard-coded inside script step(s) and will be "
            "IDENTICAL on every row: "
            + ", ".join(repr(v) for v in frozen[:8])
            + ". If they should vary per record, this task cannot be replayed as recorded."
        )

    usecase = UseCase(
        name=str(plan.get("name") or task[:80] or "Untitled use case"),
        description=str(plan.get("description") or ""),
        status="draft",
        source_run_id=source_run_id,
        allowed_domains=list(pre.domains),
        allow_scripts=False,
        inputs=used,
        secrets=[SecretSpec(**spec) for spec in (plan.get("secrets") or [])],
        setup_steps=setup_steps,
        session_check=session_check,
        row_reset=row_reset,
        row_steps=row_steps,
        teardown_steps=teardown_steps,
        outputs=[o for o in (plan.get("outputs") or []) if isinstance(o, str)],
        warnings=warnings,
    )
    log.info(
        "distilled use case",
        extra={
            # Not "name": that is a reserved LogRecord attribute and collides.
            "usecase_name": usecase.name,
            "setup_steps": len(setup_steps),
            "row_steps": len(row_steps),
            "inputs": len(usecase.inputs),
            "secrets": len(usecase.secrets),
            "warnings": len(warnings),
        },
    )
    return usecase


async def distill(
    events: list[AgentEvent],
    *,
    task: str,
    llm: Any,
    source_run_id: str | None = None,
    timeout: float = 120.0,
) -> UseCase:
    """Pre-filter a run, then spend exactly one LLM call turning it into a use case.

    This is the only function in the replay feature that touches a model. Its
    cost is paid once and amortised over every future execution.
    """
    pre = pre_filter(events)
    if not pre.steps:
        raise DistillationError(
            "This run contains no successful actions to record -- every tool call either "
            "failed or was observation-only."
        )

    turn = await llm.run_turn(
        system=load(DISTILL),
        messages=[{"role": "user", "content": as_prompt_json(pre, task)}],
        tools=[BUILD_TOOL],
        timeout=timeout,
    )

    call = next((c for c in turn.tool_calls if c.name == BUILD_TOOL["name"]), None)
    if call is None:
        raise DistillationError(
            "The model did not return a use case plan. It replied: "
            + (turn.text or "(nothing)")[:400]
        )

    usecase = build_usecase(call.input, pre, source_run_id=source_run_id, task=task)
    usecase.warnings.insert(0, summarise(pre))
    return usecase
