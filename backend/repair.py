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

from memory import as_prompt
from prompt_loader import REPAIR, REPAIR_REQUEST, load, render
from snapshot import (
    Node,
    Snapshot,
    count_matches,
    locator_for,
    parse as parse_snapshot,
)
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
                        "field_name": {
                            "type": "string",
                            "description": (
                                "replace_locator on a step that fills a FORM: which field to "
                                "change. Required there -- each field has its own locator, and "
                                "the step itself has none."
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
    #: Whether the run got far enough to fail *on a step*. False means it died
    #: before that -- the browser would not start, or setup never completed --
    #: and no page was ever on screen to capture. The two cases need different
    #: things from the person reading the refusal, so they are told apart.
    reached_a_step: bool = False

    @property
    def failed_step(self):
        return next((s for s in self.usecase.all_steps if s.id == self.failed_step_id), None)


#: Within one identical (role, label) group, how many distinct instances to
#: offer as separate candidates. This used to be 1 -- every instance past the
#: first was silently dropped, which meant a page with 11 identical per-row
#: "Chat" buttons offered exactly one candidate no matter which row actually
#: failed, and a repair could never point at a specific one. Capped rather
#: than unlimited: this exists to let a dozen near-identical rows be told
#: apart, not to let a doctor pick "the 340th delete icon" out of a wall of
#: them -- past this many, the rest collapse the way they always did.
MAX_PER_GROUP = 20


def candidates(snapshot: Snapshot) -> list[Node]:
    """Named, interactive controls that were on the page when it failed."""
    counts: dict[tuple[str, str], int] = {}
    chosen: list[Node] = []
    for node in snapshot:
        if node.role not in _INTERESTING:
            continue
        label = node.name or node.text
        if not label:
            continue
        key = (node.role, label)
        seen = counts.get(key, 0)
        if seen >= MAX_PER_GROUP:
            continue
        counts[key] = seen + 1
        chosen.append(node)
        if len(chosen) >= MAX_CANDIDATES:
            break
    return chosen


def _nearby_context(nodes: list[Node], index: int) -> str:
    """The nearest ancestor's own name or text, for telling apart two
    otherwise-identical candidates -- the title of the card a button sits in,
    the row of a table a cell belongs to.

    Cheap on purpose: the snapshot is already a top-down walk with a depth per
    node, so an element's nearest labelled ancestor is found by walking
    backward and stepping the depth ceiling up one level at a time, stopping
    at the first ancestor that has something to say for itself (skipping bare
    wrapper `generic`/`group` nodes, which are common and say nothing).
    """
    node = nodes[index]
    ceiling = node.depth
    for i in range(index - 1, -1, -1):
        other = nodes[i]
        if other.depth >= ceiling:
            continue
        ceiling = other.depth
        if other.name or other.text:
            return other.name or other.text
        if ceiling == 0:
            break
    return ""


def _listing(options: list[Node], snapshot: Snapshot) -> str:
    """The candidates, numbered for ``element_index`` -- with enough context
    to tell two identically-named ones apart.

    Plain role+name is shown for anything that is not part of an ambiguous
    group. Where it is -- the ordinary shape of a list, a card, a table row --
    the nearest ancestor's own text is appended, because that is usually
    exactly what a person or a diagnosis already uses to tell them apart
    ("the row for account A-1001"), and it is what lets a fix's `nth` land on
    the *right* one instead of always the first.
    """
    if not options:
        return "(the page was not captured, so no elements can be offered)"
    by_ref = {node.ref: i for i, node in enumerate(snapshot.nodes)}
    lines: list[str] = []
    for index, node in enumerate(options):
        line = f'{index}. {node.role} "{node.name or node.text}"'
        exact = False  # repair has never distinguished exact/loose matching
        if count_matches(snapshot, node, exact) > 1:
            page_index = by_ref.get(node.ref)
            context = (
                _nearby_context(snapshot.nodes, page_index)
                if page_index is not None
                else ""
            )
            note = f'inside "{context}"' if context else "position on the page only"
            line += f" -- one of several identical; {note}"
        lines.append(line)
    return "\n".join(lines)


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
    reached_a_step = False

    for event in events:
        if getattr(event, "type", None) == "error" and getattr(event, "kind", "") == "step_failed":
            reached_a_step = True
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
        reached_a_step=reached_a_step,
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
    #: The input/output split behind `tokens`, kept separately because a
    #: caller pricing this call (an agent recovery folding it into its own
    #: budget, say) needs the split -- input and output are priced
    #: differently, and collapsing them first would make that caller's own
    #: cost estimate wrong rather than merely approximate.
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def actionable(self) -> bool:
        return bool(self.fixes) and not self.unfixable_reason


@dataclass(slots=True)
class PendingRepair:
    """A diagnosis produced automatically, when an agent recovery gave up on
    a row rather than by a person pressing "Fix it with AI" -- carrying the
    one extra thing `apply_fixes` needs that a stored execution does not
    keep: the actual snapshot the diagnosis was read from.

    Produced by `agent/operate.py`, which has no store access by design (the
    same reason `engine.py` cannot import `llm`); a caller that does have one
    turns this into a draft version the way `repair.py`'s own router does,
    or discards it if it turns out to change nothing. Never applied without
    that step -- the rule that a repair is only ever a reviewable draft holds
    here exactly as it does for the button a person presses.
    """

    proposal: RepairProposal
    snapshot: Snapshot


class UseCaseDoctor:
    """One LLM call: look at a failure and propose the smallest set of edits."""

    def __init__(self, llm: Any, max_tokens: int = 30_000, memory: Any = None) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        #: What has already been worked out on this site. The healer has always
        #: consulted this; the repair button did not, so pressing it a second
        #: time on a failure somebody had already solved re-derived the answer
        #: from scratch at the cost of another call. Optional: healing memory
        #: can be switched off, and a repair still has to work without it.
        self.memory = memory

    async def diagnose(self, context: FailureContext) -> RepairProposal:
        options = candidates(context.snapshot)
        if not options:
            # Without the page there is nothing to point at, so every fix the
            # model could name would be a guess. Say so instead of paying for
            # a refusal.
            if not context.reached_a_step:
                # It never got as far as a step, so there was no page to record
                # and re-running changes nothing until the underlying error is
                # dealt with. Repeating "run it once more" here would send a
                # person round the same loop.
                raise RepairError(
                    "This run failed before any step could be attempted, so no page was "
                    "ever on screen to repair against. Nothing in the use case is "
                    f"necessarily wrong. The run failed with: {context.error}"
                )
            if any(True for _ in context.snapshot):
                # There *is* a page; it just has nothing on it. Almost always a
                # step that ran before the page finished rendering, which is a
                # timing problem rather than a locator problem -- and proposing
                # a new locator for it would be a confident wrong answer.
                raise RepairError(
                    "The page was captured, but it was empty apart from its frame -- "
                    "nothing had rendered yet when the step ran. That is a timing "
                    "problem, not a locator problem, so there is nothing here to repair. "
                    "The step gave up before the page finished arriving; re-run it, and "
                    "if it happens again the page needs longer than the step timeout "
                    "allows."
                )
            raise RepairError(
                "The page was not recorded for this failure, so there is nothing to match "
                "against and any repair would be guesswork. This affects runs from before "
                "the page was captured at failure time -- run it once more and the repair "
                "will have the page to work from."
            )
        listing = _listing(options, context.snapshot)
        step = context.failed_step

        # Evidence for the model, never an instruction: whatever it picks still
        # has to be one of the candidates above.
        past: list[Any] = []
        if self.memory is not None:
            past = await self.memory.recall(
                step_summary=step.summary() if step else context.error[:200],
                wanted=(
                    step.locators[0].describe() if step and step.locators else ""
                ),
                page_url=context.page_url or "",
                page=listing,
            )

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
                        past_fixes=as_prompt(past),
                    ),
                }
            ],
            tools=[PROPOSE_TOOL],
            timeout=90.0,
        )

        usage = {
            "input_tokens": int(turn.usage.get("input_tokens", 0)),
            "output_tokens": int(turn.usage.get("output_tokens", 0)),
        }
        tokens = usage["input_tokens"] + usage["output_tokens"]
        call = next((c for c in turn.tool_calls if c.name == PROPOSE_TOOL["name"]), None)
        if call is None:
            # Answering in prose rather than calling the tool is usually the
            # model explaining why it cannot help. That is information, not an
            # error -- report it as an unrepairable failure so the person reads
            # the reasoning instead of a stack of API noise.
            return RepairProposal(
                diagnosis=(turn.text or "").strip()[:1200] or "The model gave no answer.",
                fixes=[],
                confidence="low",
                unfixable_reason="the model did not propose a concrete edit",
                tokens=tokens,
                usage=usage,
            )

        payload = call.input
        return RepairProposal(
            diagnosis=str(payload.get("diagnosis") or ""),
            fixes=[f for f in (payload.get("fixes") or []) if isinstance(f, dict)],
            confidence=str(payload.get("confidence") or "medium"),
            unfixable_reason=str(payload.get("unfixable_reason") or ""),
            tokens=tokens,
            usage=usage,
        )



@dataclass(slots=True)
class LocatorChange:
    """One ladder that a repair gave a new head, and what it had before."""

    step_id: str
    #: The form field whose ladder changed, when the step fills a form and each
    #: field carries its own. Empty for an ordinary step.
    field_name: str
    old_locator: dict[str, Any] | None
    new_locator: dict[str, Any]


def locator_changes(before: dict[str, Any], after: dict[str, Any]) -> list[LocatorChange]:
    """What a repair changed, read off the result rather than the proposal.

    Taken by comparing the two definitions instead of by instrumenting
    ``apply_fixes``, for one reason: a fix that was proposed is not a fix that
    landed. The model can name a step that does not exist or an element index
    off the end of the page, and those are skipped. Reading the outcome means
    only real changes are learned from, and it cannot drift out of step with
    the code that applies them.
    """
    changes: list[LocatorChange] = []

    def head(holder: dict[str, Any]) -> dict[str, Any] | None:
        ladder = holder.get("locators") or []
        return ladder[0] if ladder else None

    def compare(step_id: str, field_name: str, was: dict, now: dict) -> None:
        old, new = head(was), head(now)
        if new is not None and new != old:
            changes.append(LocatorChange(step_id, field_name, old, new))

    for phase in ("setup_steps", "row_steps", "teardown_steps"):
        older = {s.get("id"): s for s in (before.get(phase) or [])}
        for step in after.get(phase) or []:
            was = older.get(step.get("id"))
            if was is None:
                continue
            step_id = str(step.get("id") or "")
            compare(step_id, "", was, step)
            # A fill_form step holds no ladder of its own; each field has one,
            # and that is what both the executor and a repair actually touch.
            was_fields = {f.get("name"): f for f in (was.get("fields") or [])}
            for field in step.get("fields") or []:
                previous = was_fields.get(field.get("name"))
                if previous is not None:
                    compare(step_id, str(field.get("name") or ""), previous, field)

    return changes


def apply_fixes(
    definition: dict[str, Any],
    proposal: RepairProposal,
    options: list[Node],
    snapshot: Snapshot | list[Node] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Apply a proposal to a stored definition. Returns ``(patched, applied)``.

    A fix the model got wrong -- an unknown step, an out-of-range element -- is
    skipped and reported rather than guessed at.

    ``snapshot`` is the *whole* page a picked element is counted against, not
    just the candidate list -- an element can be part of a duplicate group even
    when ``MAX_PER_GROUP`` trimmed some of its siblings out of ``options``.
    Optional and falls back to ``options`` itself (fine for a page with no
    trimming, which is every test fixture and most real pages) so existing
    callers are not forced to thread a snapshot through for no benefit.
    """
    counted_against: Snapshot | list[Node] = snapshot if snapshot is not None else options
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
            group_size = count_matches(counted_against, node, False)
            # `locator_for` narrows an ambiguous element by *where it sits*
            # before it resorts to counting, so the commonest shape of this
            # refusal is gone: the first "Chat" button of three cards is now
            # "the Chat button on the Alpha Project card" rather than an
            # element with no spelling. What is left is the case where nothing
            # around it has a name either, and that is still refused.
            healed = locator_for(counted_against, node)
            if healed is None:
                applied.append(
                    f"SKIPPED {step_id}: {node.role} {node.name!r} matches {group_size} "
                    "elements, this is the first of them, and nothing around it is named "
                    "either -- so no locator can single it out (only the 2nd match onward "
                    "can be pinned by position). Point at a different one of the matches, "
                    "or re-record this step against something that names the element."
                )
                continue
            if group_size > 1 and healed.within is not None:
                applied.append(
                    f"{step_id}: {node.role} {node.name!r} matches {group_size} elements on "
                    f"the page the doctor saw; narrowed to the one in "
                    f"{healed.within.describe()}."
                )
            elif group_size > 1:
                applied.append(
                    f"{step_id}: {node.role} {node.name!r} matches {group_size} elements on "
                    f"the page the doctor saw, and nothing on the page tells them apart; "
                    f"pinned to position {healed.nth} of them. Positional, not a name -- if "
                    "this list can reorder between runs, treat this as a stopgap and "
                    "re-record the step properly."
                )

            def ladder_for(holder: dict[str, Any]) -> list[dict[str, Any]]:
                existing = [
                    loc
                    for loc in (holder.get("locators") or [])
                    if (loc.get("role"), loc.get("name")) != (healed.role, healed.name)
                ]
                # Prepended, not replacing: a wrong repair degrades to what the
                # recording already knew rather than losing it.
                return [healed.model_dump(exclude_none=True), *existing]

            # A fill_form step holds no locator of its own -- each field has
            # its own ladder, and that is what the executor reads. Writing the
            # step-level list would look applied and do nothing.
            if step.get("action") == "fill_form":
                wanted = str(fix.get("field_name") or "")
                fields = step.get("fields") or []
                target = next((f for f in fields if f.get("name") == wanted), None)
                if target is None:
                    names = ", ".join(repr(f.get("name")) for f in fields) or "(none)"
                    applied.append(
                        f"SKIPPED {step_id}: it fills a form, so the fix must name which field "
                        f"to change. field_name was {wanted!r}; the fields are {names}"
                    )
                    continue
                target["locators"] = ladder_for(target)
                applied.append(
                    f"{step_id} field {wanted!r} now looks for {healed.describe()}. {reason}"
                )
            else:
                step["locators"] = ladder_for(step)
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


def is_unchanged(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Did a repair actually change the use case?

    Metadata that moves on every save is ignored, so this answers the question
    a person is really asking -- did any step change? Saving a version that
    differs only by a timestamp is how a repair comes to report success while
    nothing about the use case is different.
    """
    def strip(definition: dict[str, Any]) -> str:
        trimmed = {
            key: value
            for key, value in definition.items()
            if key not in {"version", "updated_at", "created_at", "status", "warnings"}
        }
        return json.dumps(trimmed, sort_keys=True, default=str)

    return strip(before) == strip(after)


def validate_patched(patched: dict[str, Any]) -> UseCase:
    """Re-validate a repaired definition as a draft.

    A repair that produces something the schema refuses is a failed repair, not
    a new stored use case -- so this raising is the point.
    """
    return UseCase.model_validate({**patched, "status": "draft"})
