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
import re
from dataclasses import dataclass, field
from typing import Any

from memory import HealingMemory, as_prompt
from pricing import price_of
from prompt_loader import HEAL, HEAL_REQUEST, load, render
from snapshot import (
    Node,
    Snapshot,
    parse as parse_snapshot,
    container_of,
    count_matches,
    index_among,
    locator_for,
    named_controls,
    ordinal_suffix,
)
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
#:
#: What changed under this number is which candidates survive the cut. It used
#: to be whichever came first in the document, so on a long form the list ended
#: somewhere above the submit button. They are ranked by resemblance to the
#: broken locator now, so the cut falls at the end of what is plausible.
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
    #: What those tokens cost, priced from each turn's own input/output split
    #: rather than from a total. Recorded because "this replay was free" and
    #: "this replay healed itself twice" have to be tellable apart, and until
    #: now they were not: the tokens were counted here and never left.
    usd_used: float = 0.0

    @property
    def exhausted(self) -> bool:
        return self.attempts_used >= self.max_attempts or self.tokens_used >= self.max_tokens

    def record(self, tokens: int, usd: float = 0.0) -> None:
        self.attempts_used += 1
        self.tokens_used += tokens
        self.usd_used += usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts_used": self.attempts_used,
            "max_attempts": self.max_attempts,
            "tokens_used": self.tokens_used,
            "max_tokens": self.max_tokens,
            "usd_used": round(self.usd_used, 4),
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
    #: What `remember` will need if this repair turns out to have worked.
    #:
    #: Held rather than written immediately because "worked" is not known here.
    #: The retry happens in `engine.py`, which then calls :meth:`StepHealer.
    #: confirm`. Writing at proposal time meant the memory recorded what the
    #: model *believed*, and a confident wrong answer was then recalled as
    #: evidence every time that site broke again -- which is the one way a
    #: memory makes healing worse rather than cheaper.
    pending: dict[str, Any] = field(default_factory=dict)


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

    @property
    def usd_used(self) -> float:
        return self.budget.usd_used

    @property
    def calls_used(self) -> int:
        return self.budget.attempts_used

    def candidates(self, snapshot: Snapshot, wanted: str = "") -> list[Node]:
        """Interactive, named controls currently on the page.

        Two rules here used to lose the answer before the model ever saw the
        question, and both are gone.

        **Nothing is deduplicated.** This kept one entry per ``(role, name)``
        pair, so a table with an "Edit" link on every row offered exactly one
        "Edit" -- and the model could not say it meant the second even when
        the second was plainly right. Duplicates are kept and numbered by
        :func:`_listing` instead, which is what makes "the third one" an
        answer that can be given at all.

        **The cut is by relevance, not by document order.** The list was
        truncated wherever the page happened to reach the cap, which on a long
        form is somewhere above the submit button. Candidates are now ordered
        by how much they look like the locator that broke, so the cut falls at
        the end of what is plausible rather than at the end of the header.
        Document order is preserved among equally plausible ones, because a
        page reads top to bottom and so should the list.
        """
        chosen = [
            node
            for node in snapshot
            if node.role in _INTERESTING and (node.name or node.text)
        ]
        if len(chosen) <= MAX_CANDIDATES:
            return chosen

        order = {id(node): index for index, node in enumerate(chosen)}
        ranked = sorted(
            chosen,
            key=lambda node: (-_resemblance(node, wanted), order[id(node)]),
        )
        # Back into document order once the cut is made: the model reads this
        # as a page, and a page whose controls arrive in relevance order is
        # harder to reason about than one that arrives in the order it renders.
        return sorted(ranked[:MAX_CANDIDATES], key=lambda node: order[id(node)])

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

        wanted = step.locators[0].describe() if step.locators else "(no locator recorded)"
        options = self.candidates(snapshot, wanted)
        if not options:
            return None

        listing = _listing(snapshot, options)

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
                            purpose=_purpose(step),
                            page_url=snapshot.page_url or "(unknown)",
                            candidates=listing,
                            was_working=_as_it_was(step),
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
        self.budget.record(tokens, price_of(getattr(self.llm, "model", ""), turn.usage))

        call = next((c for c in turn.tool_calls if c.name == CHOOSE_TOOL["name"]), None)
        if call is None:
            return None

        index = call.input.get("index")
        if not isinstance(index, int) or index < 0 or index >= len(options):
            log.info("healer declined to pick an element", extra={"step_id": step.id})
            return None

        node = options[index]
        chosen = locator_for(snapshot, node)
        if chosen is None:
            # The model picked an element no locator can single out: the first
            # of several identical controls with nothing named around it.
            # Declining costs a failed row; accepting would write a locator
            # into the use case that is refused as ambiguous on every future
            # row, which is the same failure made permanent.
            log.info(
                "healer picked an element that cannot be named uniquely",
                extra={"step_id": step.id, "element": node.describe()},
            )
            return None

        repair = Repair(
            step_id=step.id,
            locator=chosen,
            reason=str(call.input.get("reason") or ""),
            confidence=str(call.input.get("confidence") or "medium"),
            explanation=str(call.input.get("explanation") or call.input.get("reason") or ""),
            old_locator=step.locators[0] if step.locators else None,
            tokens=tokens,
        )
        self.repairs.append(repair)

        # Everything `remember` needs, kept until the retry says whether this
        # was right. See `Repair.pending` and :meth:`confirm`.
        repair.pending = {
            "usecase_id": self.usecase_id,
            "step_id": step.id,
            "page_url": snapshot.page_url or "",
            "page": listing,
            "step_summary": step.summary(),
            "wanted": wanted,
            "old_locator": repair.old_locator.model_dump() if repair.old_locator else None,
            "new_locator": repair.locator.model_dump(),
            "explanation": repair.explanation,
            "confirmed_by": "model",
        }
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

    async def confirm(self, repair: Repair, worked: bool) -> None:
        """Record the fix, now that the retry has said whether it was right.

        Called by ``engine.py`` after it re-runs the step. Two conditions, and
        the second is the one that was missing:

        **The model was sure.** A low-confidence guess is exactly the answer
        not to give next time.

        **It actually worked.** This used to be written at proposal time, so
        the memory recorded what the model *believed* about a page rather than
        what turned out to be true of it. A confident wrong answer was then
        recalled as evidence every time that site broke again -- and since
        recall puts past fixes in front of the model as context, a wrong one
        does not merely fail to help, it argues for repeating itself. That is
        the single way a memory can make healing worse rather than cheaper,
        and it was the way this one was built.
        """
        if not worked or self.memory is None or repair.confidence != "high":
            return
        if not repair.pending:
            return
        await self.memory.remember(**repair.pending)


def _purpose(step: Step) -> str:
    """What the step was for, as a prompt line, or nothing.

    A whole line rather than a bare value so a step that has no recorded
    purpose leaves no empty bullet behind -- every recording made before
    `Step.intent` existed, and every codegen recording, takes that path.
    """
    return f"- what it is for: {step.intent}" if step.intent else ""


def _as_it_was(step: Step) -> str:
    """The controls that were on the page when this step worked.

    The whole reason `Step.recorded_page` exists. Given only the current page,
    a repair is choosing between forty plausible controls; given both, most
    breakages are a renamed control that is obvious side by side and close to
    invisible from the new page alone.

    Rendered as the same kind of list as the candidates, so the two can be read
    against each other rather than one being a tree and the other a list. No
    indices on this one: nothing here is selectable, because none of it is on
    the page any more.
    """
    lines = named_controls(step.recorded_page, _INTERESTING, MAX_CANDIDATES)
    if not lines:
        return ""
    return (
        "Controls that were on this page when the step was recorded and "
        "working:\n\n" + "\n".join(lines)
    )


def _words(value: str) -> set[str]:
    return {word for word in re.split(r"[^a-z0-9]+", (value or "").casefold()) if word}


def _resemblance(node: Node, wanted: str) -> float:
    """How much this candidate looks like the locator that stopped matching.

    Deliberately crude: a shared role counts for something, and shared words in
    the name count for more. This orders a truncation, it does not make the
    choice -- the model still picks, from a list it can see all of.

    Crude is also the right amount of clever. A better similarity would be an
    embedding, and an embedding here would mean a network call on the failure
    path of every healed step, to rank a list the model is about to read in
    full anyway.
    """
    if not wanted:
        return 0.0
    wanted_words = _words(wanted)
    if not wanted_words:
        return 0.0
    score = 0.0
    if node.role and node.role.casefold() in wanted.casefold():
        score += 1.0
    shared = _words(node.name or node.text) & wanted_words
    if shared:
        score += 2.0 * len(shared) / len(wanted_words)
    return score


def _listing(snapshot: Snapshot, options: list[Node]) -> str:
    """The candidate list as the model reads it.

    Each line says which of several identical controls this one is, and where
    on the page it sits. Both exist so an answer can be *given*: with neither,
    a page holding six "Edit" links offers six lines that read identically, and
    every one of them is an equally good answer to a question that has one.
    """
    lines: list[str] = []
    for index, node in enumerate(options):
        line = f'{index}. {node.role} "{node.name or node.text}"'
        total = count_matches(snapshot, node, True)
        if total > 1:
            position = index_among(snapshot, node, True) + 1
            line += f" ({position}{ordinal_suffix(position)} of {total} like this)"
        container = container_of(snapshot, node)
        if container is not None:
            line += f' in {container.role} "{container.name or container.text}"'
        lines.append(line)
    return "\n".join(lines)


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
