"""The two passes that bracket a recording: a brief before, a walkthrough after.

Neither one touches a browser and neither can add a step. They exist because
the recording loop was being asked to do two jobs it is badly placed for.

**Before.** A session starts from whatever sentence a person typed. A vague one
produces a recording that does something adjacent to what was meant, and nobody
finds out until a replay. Worse, it produces a recording with values baked into
it: what decides whether a use case runs for four thousand customers or only
for the customer it was recorded with is whether the per-row values were
*declared*, and an undeclared value cannot be marked. So one call turns the
request into a goal, the per-row values, and what proves a row worked -- and
says out loud what the request did not say, which is the cheapest possible
moment for a person to correct an assumption.

The rule that makes this help rather than hurt: **the brief plans no clicks.**
A model that has never seen the site inventing an "Advanced search" link sends
the recorder hunting for something that does not exist, and this loop's most
expensive failure mode is a confident wrong first guess. Goal, data, proof. The
route is the recorder's to find, on the page, with the page in front of it.

**After.** The step list says what happens. It does not say what the workflow is
*for*, and that is what a reviewer needs before publishing and what a repair
needs before it can judge whether a replacement control makes sense. One call
over the distilled steps writes the overview and a purpose per step.

Both passes are failure-tolerant by construction. A brief that does not come
back leaves the session exactly as it was before this module existed; a
walkthrough that does not come back leaves a draft with no prose on it. The
recording is the expensive part and neither of these is allowed to cost one.

Both are *injected*, like a healer: a caller that passes no client gets no extra
model calls at all. That keeps every existing caller and every scripted test
unchanged, and it is how a deployment turns the passes off.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from prompt_loader import (
    BRIEF,
    BRIEF_REQUEST,
    WALKTHROUGH,
    WALKTHROUGH_REQUEST,
    load,
    render,
)
from usecase import UseCase

log = logging.getLogger(__name__)

#: Long enough for a considered answer, short enough that a stalled provider
#: cannot hold a browser session open waiting for prose.
TIMEOUT = 60.0

#: Caps. Both of these end up in a document that is read on a screen and sent
#: to a model on every repair, so neither may grow without limit.
MAX_OVERVIEW = 4_000
MAX_PURPOSE = 400


# ---------------------------------------------------------------------------
# Before: the brief
# ---------------------------------------------------------------------------

BRIEF_TOOL: dict[str, Any] = {
    "name": "brief",
    "description": (
        "State the goal, what varies per row, how a row proves it worked, and "
        "what the request left unsaid. No clicks, no invented controls. Call "
        "exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The outcome in one sentence, not the route to it.",
            },
            "per_row": {
                "type": "array",
                "description": (
                    "The values that change from row to row. Each becomes a "
                    "spreadsheet column."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short identifier-shaped column name.",
                        },
                        "means": {"type": "string", "description": "What it is."},
                    },
                    "required": ["name"],
                },
            },
            "done_when": {
                "type": "array",
                "description": "What the page shows only once a row's work is done.",
                "items": {"type": "string"},
            },
            "cautions": {
                "type": "array",
                "description": "Only what the request itself implies. Usually empty.",
                "items": {"type": "string"},
            },
            "unclear": {
                "type": "array",
                "description": (
                    "What the request does not say and the recorder will have to "
                    "decide on the page."
                ),
                "items": {"type": "string"},
            },
        },
        "required": ["goal"],
    },
}


@dataclass
class TaskBrief:
    """The request, restated as an outcome, some data and a proof."""

    goal: str = ""
    per_row: list[dict[str, str]] = field(default_factory=list)
    done_when: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    unclear: list[str] = field(default_factory=list)
    tokens: int = 0
    #: The provider's own input/output split, for costing. A single total
    #: would have to be charged as one or the other, and output tokens cost
    #: several times what input tokens do.
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.goal.strip()

    def as_prompt(self) -> str:
        """The brief as the recorder is shown it.

        Headed "alongside the task, never instead of it", because the task is
        what was actually asked for and this is one reading of it. A brief that
        contradicts the request has to lose, and saying so costs three lines.
        """
        if self.empty:
            return ""
        lines = [
            "## The brief",
            "",
            "One reading of the task, written before anybody looked at the site.",
            "Read it alongside the task, never instead of it.",
            "Where the two disagree, the task wins.",
            "Where the page disagrees with both, the page wins.",
            "",
            "**What this has to achieve:** " + self.goal,
        ]
        if self.per_row:
            lines += ["", "**Expected to vary per record**, so mark each as an input:"]
            for item in self.per_row:
                means = item.get("means") or ""
                lines.append(
                    "- " + item.get("name", "") + (" -- " + means if means else "")
                )
        if self.done_when:
            lines += ["", "**A record is done when:**"]
            lines += ["- " + item for item in self.done_when]
        if self.cautions:
            lines += ["", "**Care needed:**"]
            lines += ["- " + item for item in self.cautions]
        if self.unclear:
            lines += [
                "",
                "**Not stated in the task.** Decide these on the page, and say in",
                "finish what you chose:",
            ]
            lines += ["- " + item for item in self.unclear]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "per_row": self.per_row,
            "done_when": self.done_when,
            "cautions": self.cautions,
            "unclear": self.unclear,
            "tokens": self.tokens,
        }


async def write_brief(
    llm: Any,
    *,
    task: str,
    start_url: str = "",
    allowed_domains: tuple[str, ...] | list[str] = (),
    secrets: tuple[str, ...] | list[str] = (),
    sample: dict[str, Any] | None = None,
) -> TaskBrief | None:
    """One call, before the browser opens. ``None`` when it did not work out.

    Never raises. A brief is an improvement on the request, and losing a
    session because an improvement failed would be the wrong trade -- the
    caller carries on with the request exactly as it did before this existed.
    """
    try:
        turn = await llm.run_turn(
            system=load(BRIEF),
            messages=[
                {
                    "role": "user",
                    "content": render(
                        BRIEF_REQUEST,
                        task=task or "(nothing was said)",
                        start_url=start_url or "(none given)",
                        allowed_domains=", ".join(allowed_domains) or "(nothing configured)",
                        secrets=", ".join(secrets) or "(none)",
                        sample=json.dumps(sample) if sample else "(none given)",
                    ),
                }
            ],
            tools=[BRIEF_TOOL],
            timeout=TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 - see the docstring
        log.warning("the brief pass did not run: %s", exc)
        return None

    usage = getattr(turn, "usage", {}) or {}
    call = next((c for c in turn.tool_calls if c.name == BRIEF_TOOL["name"]), None)
    if call is None:
        # Prose instead of the tool is a model saying it has nothing to add,
        # which is a legitimate answer to a request that was already precise.
        # Not an error, and not worth a second call.
        log.info("the brief pass answered in prose; carrying on with the task as given")
        return None

    given = call.input if isinstance(call.input, dict) else {}
    brief = TaskBrief(
        goal=str(given.get("goal") or "").strip()[:1_000],
        per_row=_pairs(given.get("per_row")),
        done_when=_lines(given.get("done_when")),
        cautions=_lines(given.get("cautions")),
        unclear=_lines(given.get("unclear")),
        tokens=int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0)),
        usage={k: int(v) for k, v in usage.items() if isinstance(v, (int, float))},
    )
    return None if brief.empty else brief


# ---------------------------------------------------------------------------
# After: the walkthrough
# ---------------------------------------------------------------------------

WALKTHROUGH_TOOL: dict[str, Any] = {
    "name": "walkthrough",
    "description": (
        "Describe the recorded workflow in plain language, and say what each "
        "step is for. Use only the step ids given. Call exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "overview": {
                "type": "string",
                "description": (
                    "What the workflow achieves, what varies per row, and how a "
                    "row shows it worked. A short paragraph or a few."
                ),
            },
            "steps": {
                "type": "array",
                "description": "A purpose per step. Omit any step you cannot tell.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "One of the step ids you were given.",
                        },
                        "purpose": {
                            "type": "string",
                            "description": "Why this step exists, in one clause.",
                        },
                    },
                    "required": ["id", "purpose"],
                },
            },
        },
        "required": ["overview"],
    },
}


@dataclass
class Walkthrough:
    """Prose about a recording: the flow, and a purpose per step."""

    overview: str = ""
    purposes: dict[str, str] = field(default_factory=dict)
    tokens: int = 0
    #: The provider's own input/output split. See `TaskBrief.usage`.
    usage: dict[str, int] = field(default_factory=dict)
    #: Ids the model returned that are not in the use case. Reported rather
    #: than silently dropped: a model naming steps that do not exist was
    #: describing something other than this recording.
    unknown_steps: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.overview.strip() and not self.purposes


async def write_walkthrough(
    llm: Any,
    use_case: UseCase,
    *,
    task: str = "",
    brief: TaskBrief | None = None,
) -> Walkthrough | None:
    """One call, after distillation. ``None`` when it did not work out.

    Never raises, for the same reason `write_brief` does not, only more so: by
    the time this runs a model has driven a browser for minutes and a person
    has watched it.
    """
    steps = use_case.all_steps
    if not steps:
        return None
    try:
        turn = await llm.run_turn(
            system=load(WALKTHROUGH),
            messages=[
                {
                    "role": "user",
                    "content": render(
                        WALKTHROUGH_REQUEST,
                        task=task or use_case.description or "(nothing was said)",
                        brief=brief.as_prompt() if brief and not brief.empty else "",
                        setup=_listing(use_case.setup_steps),
                        row=_listing(use_case.row_steps),
                        inputs=", ".join(spec.name for spec in use_case.inputs) or "(nothing)",
                    ),
                }
            ],
            tools=[WALKTHROUGH_TOOL],
            timeout=TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 - see the docstring
        log.warning("the walkthrough pass did not run: %s", exc)
        return None

    usage = getattr(turn, "usage", {}) or {}
    call = next((c for c in turn.tool_calls if c.name == WALKTHROUGH_TOOL["name"]), None)
    if call is None:
        log.info("the walkthrough pass answered in prose; the draft keeps no overview")
        return None

    given = call.input if isinstance(call.input, dict) else {}
    known = {step.id for step in steps}
    purposes: dict[str, str] = {}
    unknown: list[str] = []
    for item in given.get("steps") or ():
        if not isinstance(item, dict):
            continue
        step_id = str(item.get("id") or "").strip()
        purpose = str(item.get("purpose") or "").strip()[:MAX_PURPOSE]
        if not step_id or not purpose:
            continue
        if step_id not in known:
            unknown.append(step_id)
            continue
        purposes[step_id] = purpose

    walkthrough = Walkthrough(
        overview=str(given.get("overview") or "").strip()[:MAX_OVERVIEW],
        purposes=purposes,
        tokens=int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0)),
        usage={k: int(v) for k, v in usage.items() if isinstance(v, (int, float))},
        unknown_steps=unknown,
    )
    return None if walkthrough.empty else walkthrough


def apply_walkthrough(use_case: UseCase, walkthrough: Walkthrough) -> list[str]:
    """Write the prose onto the draft. Returns warnings for the reviewer.

    Two deliberate restrictions.

    A purpose is written only where the step has none. The authoring agent's
    own sentence was written with the page in front of it, one call before the
    action; this pass is reading a step list afterwards. Where both exist the
    first is the better evidence, and overwriting it would quietly trade a
    first-hand account for a plausible reconstruction.

    Nothing here can change what runs. `intent` and `instructions` are read by
    people and by the two prompts that ask where a control went, and by nothing
    at all on the replay path.
    """
    warnings: list[str] = []
    if walkthrough.overview:
        use_case.instructions = walkthrough.overview
    filled = 0
    for step in use_case.all_steps:
        purpose = walkthrough.purposes.get(step.id)
        if purpose and not step.intent:
            step.intent = purpose
            filled += 1
    if walkthrough.unknown_steps:
        warnings.append(
            "The written walkthrough described "
            + str(len(walkthrough.unknown_steps))
            + " step(s) that are not in this use case ("
            + ", ".join(sorted(walkthrough.unknown_steps)[:5])
            + "). Read the overview against the steps before publishing."
        )
    log.info("walkthrough applied: %d step purpose(s) filled in", filled)
    return warnings


# ---------------------------------------------------------------------------
# Shaping what came back
# ---------------------------------------------------------------------------
#
# A tool schema is a request, not a guarantee -- some provider will hand back a
# string where a list was asked for, and a session must not die of it here, one
# call after the browser closed.


def _lines(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:10]


def _pairs(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        value = [value] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []
    out: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict) and str(item.get("name") or "").strip():
            out.append(
                {
                    "name": str(item["name"]).strip()[:60],
                    "means": str(item.get("means") or "").strip()[:200],
                }
            )
        elif isinstance(item, str) and item.strip():
            out.append({"name": item.strip()[:60], "means": ""})
    return out[:20]


def _listing(steps: list[Any]) -> str:
    """The steps as the writer is shown them: id, action, what it acts on."""
    if not steps:
        return "  (none)"
    lines: list[str] = []
    for step in steps:
        parts = ["  " + step.id + ": " + step.action]
        if step.locators:
            parts.append(step.locators[0].describe())
        if step.url:
            parts.append(step.url)
        if step.value:
            parts.append("value=" + repr(step.value))
        if step.assertion is not None:
            parts.append("check " + step.assertion.describe())
        lines.append(" ".join(parts))
    return "\n".join(lines)


__all__ = [
    "BRIEF_TOOL",
    "TaskBrief",
    "WALKTHROUGH_TOOL",
    "Walkthrough",
    "apply_walkthrough",
    "write_brief",
    "write_walkthrough",
]
