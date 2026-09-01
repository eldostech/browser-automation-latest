"""Repairing a single broken step, at a cost you can see.

A recorded use case eventually breaks: a site ships a redesign and the
accessible name a step was recorded against no longer exists. Healing asks the
model to find the control again -- once, for that one step -- and writes the
answer back as a new use case version, so the next thousand rows are free
again.

This is the one part of the replay feature that can spend tokens, so it is
built to make that impossible to do by accident:

**It lives here, not in replay.py.** The executor never imports an LLM client;
it accepts a healer that satisfies a small protocol. With no healer injected
there is no code path to a model, which is why "zero tokens" stays a
structural property rather than a configuration setting.

**Off unless a step asks for it.** Only a step whose ``on_failure`` is
``heal`` is ever offered, and only when the caller passed a healer.

**Budgeted.** :class:`HealingBudget` caps both attempts and tokens for a whole
batch. When it is exhausted, healing stops and the batch carries on failing
rows normally -- a broken selector must not quietly turn a free batch into an
expensive one.

**It cannot invent a locator.** The model chooses from the controls actually
present in the current snapshot, by index. A hallucinated selector has no route
into a use case, exactly as in distillation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from memory import HealingMemory, as_prompt
from prompt_loader import HEAL, HEAL_REQUEST, load, render
from snapshot import Node, Snapshot
from usecase import Locator, Step

log = logging.getLogger(__name__)

#: Roles worth offering as candidates. Structural wrappers are noise.
_INTERESTING = frozenset(
    {
        "button", "link", "textbox", "checkbox", "radio", "combobox", "listbox",
        "option", "menuitem", "tab", "switch", "slider", "searchbox", "spinbutton",
        "heading", "status", "alert", "cell", "columnheader", "img",
    }
)

#: Never offer more than this many candidates: a huge page would otherwise put
#: the original snapshot cost straight back into the context window.
MAX_CANDIDATES = 60

CHOOSE_TOOL: dict[str, Any] = {
    "name": "choose_element",
    "description": (
        "Pick the element the broken step was meant to act on, by its index in "
        "the candidate list. Use index -1 if none of them is the right element."
    ),
    "input_schema": {
        "type": "object",
        "required": ["index"],
        "properties": {
            "index": {
                "type": "integer",
                "description": "Index from the candidate list, or -1 if none matches.",
            },
            "confidence": {"enum": ["high", "medium", "low"]},
            "reason": {"type": "string", "description": "One sentence."},
            "explanation": {
                "type": "string",
                "description": (
                    "What changed on the page, in plain words, for somebody who "
                    "was not watching and may not know the site. This is stored "
                    "and shown the next time it breaks."
                ),
            },
        },
    },
}


@dataclass(slots=True)
class HealingBudget:
    """Caps what a whole batch may spend repairing itself."""

    max_attempts: int = 3
    max_tokens: int = 20_000
    attempts_used: int = 0
    tokens_used: int = 0

    @property
    def exhausted(self) -> bool:
        return self.attempts_used >= self.max_attempts or self.tokens_used >= self.max_tokens

    def record(self, tokens: int) -> None:
        self.attempts_used += 1
        self.tokens_used += tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts_used": self.attempts_used,
            "max_attempts": self.max_attempts,
            "tokens_used": self.tokens_used,
            "max_tokens": self.max_tokens,
            "exhausted": self.exhausted,
        }


@dataclass(slots=True)
class Repair:
    """A proposed replacement locator for one step."""

    step_id: str
    locator: Locator
    reason: str = ""
    confidence: str = "medium"
    tokens: int = 0
    #: Plain words for somebody who was not watching. Shown in the UI, stored
    #: in healing memory, and put in front of the model next time.
    explanation: str = ""
    #: What it used to look for, kept so the memory can record the change
    #: rather than just the destination.
    old_locator: Locator | None = None
    #: Whether this was recalled rather than worked out. A recalled fix costs
    #: nothing and is worth counting separately from one that did.
    recalled: bool = False


@dataclass(slots=True)
class StepHealer:
    """Asks the model to re-find one control on the page in front of it.

    Satisfies :class:`replay.Healer`. Constructed only where healing is
    deliberately enabled.
    """

    llm: Any
    budget: HealingBudget = field(default_factory=HealingBudget)
    #: Every repair accepted this session, for the version bump afterwards.
    repairs: list[Repair] = field(default_factory=list)
    #: What this workspace has learned. Left None, healing works exactly as it
    #: did before there was a memory -- which is what makes the memory an
    #: optimisation rather than a dependency.
    memory: "HealingMemory | None" = None
    #: Stamped onto anything remembered, so a fix can be traced to its recipe.
    usecase_id: str | None = None

    @property
    def tokens_used(self) -> int:
        return self.budget.tokens_used

    def candidates(self, snapshot: Snapshot) -> list[Node]:
        """Interactive, named controls currently on the page."""
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

    async def repair(self, step: Step, snapshot: Snapshot | None) -> Repair | None:
        """Propose a replacement locator, or ``None``.

        Returns ``None`` rather than raising for every ordinary failure --
        budget exhausted, nothing on the page, the model declining. A row that
        cannot be healed is just a failed row.
        """
        if self.budget.exhausted:
            log.info("healing budget exhausted; not attempting", extra={"step_id": step.id})
            return None
        if snapshot is None or not len(snapshot):
            return None

        options = self.candidates(snapshot)
        if not options:
            return None

        listing = "\n".join(
            f"{index}. {node.role} \"{node.name or node.text}\"" for index, node in enumerate(options)
        )
        wanted = step.locators[0].describe() if step.locators else "(no locator recorded)"

        # What was done about this before, on this site. Evidence for the
        # model, never an instruction: whatever it picks still has to be one of
        # the candidates above.
        past = []
        if self.memory is not None:
            past = await self.memory.recall(
                step_summary=step.summary(),
                wanted=wanted,
                page_url=snapshot.page_url or "",
                page=listing,
            )

        try:
            turn = await self.llm.run_turn(
                system=load(HEAL),
                messages=[
                    {
                        "role": "user",
                        "content": render(
                            HEAL_REQUEST,
                            step=step.summary(),
                            action=step.action,
                            wanted=wanted,
                            page_url=snapshot.page_url or "(unknown)",
                            candidates=listing,
                            past_fixes=as_prompt(past),
                        ),
                    }
                ],
                tools=[CHOOSE_TOOL],
                timeout=60.0,
            )
        except Exception as exc:  # noqa: BLE001 - a failed repair is just a failed row
            log.warning("healing call failed", extra={"step_id": step.id, "error": str(exc)})
            return None

        tokens = int(turn.usage.get("input_tokens", 0)) + int(turn.usage.get("output_tokens", 0))
        self.budget.record(tokens)

        call = next((c for c in turn.tool_calls if c.name == CHOOSE_TOOL["name"]), None)
        if call is None:
            return None

        index = call.input.get("index")
        if not isinstance(index, int) or index < 0 or index >= len(options):
            log.info("healer declined to pick an element", extra={"step_id": step.id})
            return None

        node = options[index]
        repair = Repair(
            step_id=step.id,
            locator=Locator(strategy="role", role=node.role, name=node.name or None),
            reason=str(call.input.get("reason") or ""),
            confidence=str(call.input.get("confidence") or "medium"),
            explanation=str(call.input.get("explanation") or call.input.get("reason") or ""),
            old_locator=step.locators[0] if step.locators else None,
            tokens=tokens,
        )
        self.repairs.append(repair)

        # Remembered only when the model was sure. A low-confidence guess is
        # exactly the answer not to give next time, and writing every attempt
        # down would fill the table with the ones that were wrong.
        if self.memory is not None and repair.confidence == "high":
            await self.memory.remember(
                usecase_id=self.usecase_id,
                step_id=step.id,
                page_url=snapshot.page_url or "",
                page=listing,
                step_summary=step.summary(),
                wanted=wanted,
                old_locator=repair.old_locator.model_dump() if repair.old_locator else None,
                new_locator=repair.locator.model_dump(),
                explanation=repair.explanation,
                confirmed_by="model",
            )
        log.info(
            "healed a step",
            extra={
                "step_id": step.id,
                "locator": repair.locator.describe(),
                "tokens": tokens,
                "confidence": repair.confidence,
            },
        )
        return repair


def _identity(locator: dict[str, Any]) -> tuple:
    """What makes two stored locators the same one.

    Raw dict equality is not enough: a locator dumped from the model carries
    defaults (``nth: 0``) that a hand-written or older stored one may omit, so
    comparing dicts would treat them as different and let the ladder grow a
    near-duplicate rung on every repair.
    """
    return (
        locator.get("strategy"),
        locator.get("role"),
        locator.get("name"),
        locator.get("selector"),
        locator.get("text"),
        locator.get("nth", 0),
    )


def apply_repairs(definition: dict[str, Any], repairs: list[Repair]) -> dict[str, Any]:
    """Return a copy of a stored use case with the repaired locators in front.

    The healed locator is *prepended* rather than replacing the ladder: the
    original stays as a fallback, so a repair that turns out to be wrong
    degrades to what the recording already knew instead of losing it.
    """
    if not repairs:
        return definition

    by_step: dict[str, Repair] = {repair.step_id: repair for repair in repairs}
    patched = {**definition}

    for phase in ("setup_steps", "row_steps", "teardown_steps"):
        steps = []
        for step in patched.get(phase) or []:
            repair = by_step.get(step.get("id"))
            if repair is None:
                steps.append(step)
                continue
            healed = repair.locator.model_dump(exclude_none=True)
            key = _identity(healed)
            existing = [
                loc for loc in (step.get("locators") or []) if _identity(loc) != key
            ]
            steps.append({**step, "locators": [healed, *existing]})
        patched[phase] = steps

    return patched
