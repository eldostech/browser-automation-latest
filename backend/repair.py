"""Repairing a use case that failed, without re-recording it.

Healing (``healing.py``) fixes one locator mid-run, for a step that asked for
it. This is the other half: a use case has already failed, a person is looking
at the failure, and they want it mended. It differs in three ways that matter.

**It works from history, not from a live browser.** The executor records the
page as it was when a step failed, so a repair can be proposed with no second
browser session and nothing to re-drive.

**It fixes more than locators.** A run fails for many reasons -- an element
renamed, an assertion that can never hold, a value that no longer applies, a
step the site dropped. Each has a different repair, so the model chooses the
*kind* of fix as well as its content.

**It never applies anything unreviewed.** A repair produces a new draft
version. The existing versions are untouched, and a person publishes it, which
is the same gate every distilled use case passes through.

The safety property from distillation is preserved exactly: **the model cannot
invent a locator.** It picks an element by index from the controls that were
actually on the page, so a hallucinated selector has no route in.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from prompt_loader import REPAIR, REPAIR_REQUEST, load, render
from snapshot import Node, Snapshot, parse as parse_snapshot
from usecase import Assertion, Locator, UseCase

log = logging.getLogger(__name__)

#: Roles worth offering as repair candidates. Structural wrappers are noise.
_INTERESTING = frozenset(
    {
        "button", "link", "textbox", "checkbox", "radio", "combobox", "listbox",
        "option", "menuitem", "tab", "switch", "slider", "searchbox", "spinbutton",
        "heading", "status", "alert", "cell", "columnheader", "img", "paragraph",
    }
)

#: Cap on candidates offered. A huge page must not put the original run's
#: token cost straight back into the context window.
MAX_CANDIDATES = 80

FIX_KINDS = (
    "replace_locator",
    "fix_assertion",
    "change_value",
    "make_optional",
    "drop_step",
    "fix_session_check",
)

PROPOSE_TOOL: dict[str, Any] = {
    "name": "propose_repair",
    "description": (
        "Explain why the use case failed and list the smallest set of fixes that "
        "would make it work. Call exactly once."
    ),
    "input_schema": {
        "type": "object",
        "required": ["diagnosis", "fixes"],
        "properties": {
            "diagnosis": {
                "type": "string",
                "description": "One or two sentences: what actually went wrong.",
            },
            "confidence": {"enum": ["high", "medium", "low"]},
            "unfixable_reason": {
                "type": "string",
                "description": (
                    "Set this INSTEAD of fixes when the failure cannot be repaired by "
                    "editing steps -- for example the task needs judgement per row, or "
                    "the site now requires something the recording never did."
                ),
            },
            "fixes": {
                "type": "array",
                "description": "Smallest set of edits that would fix it. May be empty.",
                "items": {
                    "type": "object",
                    "required": ["kind"],
                    "properties": {
                        "kind": {"enum": list(FIX_KINDS)},
                        "step_id": {
                            "type": "string",
                            "description": "Which step to edit. Omit only for fix_session_check.",
                        },
                        "element_index": {
                            "type": "integer",
                            "description": (
                                "replace_locator only: index from the candidate list. "
                                "You may not describe an element any other way."
                            ),
                        },
                        "assertion_kind": {
                            "enum": ["url_contains", "text_present", "title_contains"],
                            "description": "fix_assertion / fix_session_check.",
                        },
                        "value": {
                            "type": "string",
                            "description": "New value, for fix_assertion / change_value.",
                        },
                        "negate": {"type": "boolean"},
                        "reason": {"type": "string", "description": "One sentence."},
                    },
                },
            },
        },
    },
}


class RepairError(RuntimeError):
    """The failure could not be repaired, with the reason to show a person."""


@dataclass(slots=True)
class FailureContext:
    """Everything known about one failure, gathered from stored history."""

    usecase: UseCase
    failed_step_id: str | None
    error: str
    snapshot: Snapshot
    page_url: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)

    @property
    def failed_step(self):
        return next((s for s in self.usecase.all_steps if s.id == self.failed_step_id), None)


def candidates(snapshot: Snapshot) -> list[Node]:
    """Named, interactive controls that were on the page when it failed."""
    seen: set[tuple[str, str]] = set()
    chosen: list[Node] = []
    for node in snapshot:
        if node.role not in _INTERESTING:
            continue
        label = node.name or node.text
        if not label:
            continue
        key = (node.role, label)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(node)
        if len(chosen) >= MAX_CANDIDATES:
            break
    return chosen


def gather_context(
    usecase: UseCase,
    execution: dict[str, Any],
    events: list[Any],
) -> FailureContext:
    """Reconstruct what the page looked like when this execution failed.

    Reads the ``step_failed`` error the executor records at the moment of
    failure. Falls back to any snapshot text in the run's tool results, so a
    run from before that was recorded can still be repaired -- just with less
    to go on.
    """
    snapshot_text = ""
    page_url: str | None = None

    for event in events:
        if getattr(event, "type", None) == "error" and getattr(event, "kind", "") == "step_failed":
            detail = getattr(event, "detail", {}) or {}
            if detail.get("snapshot"):
                snapshot_text = str(detail["snapshot"])
                page_url = detail.get("page_url") or page_url
        elif getattr(event, "type", None) == "tool_result" and not snapshot_text:
            text = getattr(event, "text", "") or ""
            if "### Snapshot" in text:
                snapshot_text = text

    parsed = parse_snapshot(snapshot_text)
    return FailureContext(
        usecase=usecase,
        failed_step_id=execution.get("failed_step_id"),
        error=str(execution.get("error") or "the run failed without recording a reason"),
        snapshot=parsed,
        page_url=page_url or parsed.page_url,
        inputs=execution.get("inputs") or {},
    )


def _describe_steps(usecase: UseCase) -> str:
    lines: list[str] = []
    for phase, steps in (
        ("setup", usecase.setup_steps),
        ("row", usecase.row_steps),
        ("teardown", usecase.teardown_steps),
    ):
        for step in steps:
            locator = step.locators[0].describe() if step.locators else "-"
            value = f" value={step.value!r}" if step.value else ""
            lines.append(f"  [{phase}] {step.id}: {step.action} {locator}{value}")
    return "\n".join(lines) or "  (none)"


@dataclass(slots=True)
class RepairProposal:
    diagnosis: str
    fixes: list[dict[str, Any]]
    confidence: str = "medium"
    unfixable_reason: str = ""
    tokens: int = 0

    @property
    def actionable(self) -> bool:
        return bool(self.fixes) and not self.unfixable_reason


class UseCaseDoctor:
    """One LLM call: look at a failure and propose the smallest set of edits."""

    def __init__(self, llm: Any, max_tokens: int = 30_000) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    async def diagnose(self, context: FailureContext) -> RepairProposal:
        options = candidates(context.snapshot)
        listing = (
            "\n".join(
                f'{index}. {node.role} "{node.name or node.text}"'
                for index, node in enumerate(options)
            )
            or "(the page was not captured, so no elements can be offered)"
        )
        step = context.failed_step

        turn = await self.llm.run_turn(
            system=load(REPAIR),
            messages=[
                {
                    "role": "user",
                    "content": render(
                        REPAIR_REQUEST,
                        name=context.usecase.name,
                        error=context.error[:1500],
                        failed_step=step.summary() if step else "(unknown)",
                        failed_step_id=context.failed_step_id or "(unknown)",
                        failed_action=step.action if step else "(unknown)",
                        wanted=(
                            step.locators[0].describe()
                            if step and step.locators
                            else "(no locator recorded)"
                        ),
                        page_url=context.page_url or "(unknown)",
                        allowed_domains=", ".join(context.usecase.allowed_domains) or "(none)",
                        steps=_describe_steps(context.usecase),
                        candidates=listing,
                    ),
                }
            ],
            tools=[PROPOSE_TOOL],
            timeout=90.0,
        )

        tokens = int(turn.usage.get("input_tokens", 0)) + int(turn.usage.get("output_tokens", 0))
        call = next((c for c in turn.tool_calls if c.name == PROPOSE_TOOL["name"]), None)
        if call is None:
            raise RepairError(
                "The model did not propose a repair. It said: "
                + (turn.text or "(nothing)")[:400]
            )

        payload = call.input
        return RepairProposal(
            diagnosis=str(payload.get("diagnosis") or ""),
            fixes=[f for f in (payload.get("fixes") or []) if isinstance(f, dict)],
            confidence=str(payload.get("confidence") or "medium"),
            unfixable_reason=str(payload.get("unfixable_reason") or ""),
            tokens=tokens,
        )


def apply_fixes(
    definition: dict[str, Any], proposal: RepairProposal, options: list[Node]
) -> tuple[dict[str, Any], list[str]]:
    """Apply a proposal to a stored definition. Returns ``(patched, applied)``.

    A fix the model got wrong -- an unknown step, an out-of-range element -- is
    skipped and reported rather than guessed at.
    """
    patched = json.loads(json.dumps(definition))
    applied: list[str] = []

    def find(step_id: str):
        for phase in ("setup_steps", "row_steps", "teardown_steps"):
            for step in patched.get(phase) or []:
                if step.get("id") == step_id:
                    return phase, step
        if (patched.get("row_reset") or {}).get("id") == step_id:
            return "row_reset", patched["row_reset"]
        return None, None

    for fix in proposal.fixes:
        kind = fix.get("kind")
        step_id = str(fix.get("step_id") or "")
        reason = str(fix.get("reason") or "")

        if kind == "fix_session_check":
            check = Assertion(
                kind=fix.get("assertion_kind") or "url_contains",
                value=str(fix.get("value") or ""),
                negate=bool(fix.get("negate", False)),
            )
            patched["session_check"] = check.model_dump(mode="json")
            applied.append(f"session check -> {check.describe()}. {reason}")
            continue

        phase, step = find(step_id)
        if step is None:
            applied.append(f"SKIPPED: no step named {step_id!r}")
            continue

        if kind == "replace_locator":
            index = fix.get("element_index")
            if not isinstance(index, int) or index < 0 or index >= len(options):
                applied.append(f"SKIPPED {step_id}: element index {index!r} is not on the page")
                continue
            node = options[index]
            healed = Locator(strategy="role", role=node.role, name=node.name or None)
            existing = [
                loc
                for loc in (step.get("locators") or [])
                if (loc.get("role"), loc.get("name")) != (healed.role, healed.name)
            ]
            # Prepended, not replacing: a wrong repair degrades to what the
            # recording already knew rather than losing it.
            step["locators"] = [healed.model_dump(exclude_none=True), *existing]
            applied.append(f"{step_id} now looks for {healed.describe()}. {reason}")

        elif kind == "fix_assertion":
            check = Assertion(
                kind=fix.get("assertion_kind") or "text_present",
                value=str(fix.get("value") or ""),
                negate=bool(fix.get("negate", False)),
            )
            step["assert"] = check.model_dump(mode="json")
            step["action"] = "assert"
            applied.append(f"{step_id} now asserts {check.describe()}. {reason}")

        elif kind == "change_value":
            step["value"] = str(fix.get("value") or "")
            applied.append(f"{step_id} value -> {step['value']!r}. {reason}")

        elif kind == "make_optional":
            step["optional"] = True
            step["on_failure"] = "continue"
            applied.append(f"{step_id} is now optional -- a failure will not stop the row. {reason}")

        elif kind == "drop_step":
            if phase == "row_reset":
                patched["row_reset"] = None
            else:
                patched[phase] = [s for s in patched[phase] if s.get("id") != step_id]
            applied.append(f"{step_id} removed. {reason}")

        else:
            applied.append(f"SKIPPED {step_id}: unknown fix {kind!r}")

    return patched, applied


def validate_patched(patched: dict[str, Any]) -> UseCase:
    """Re-validate a repaired definition as a draft.

    A repair that produces something the schema refuses is a failed repair, not
    a new stored use case -- so this raising is the point.
    """
    return UseCase.model_validate({**patched, "status": "draft"})
