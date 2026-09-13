"""Does telling a model what a step was *for* make it pick the right control?

The claim behind `Step.intent` is that it turns "which of these four controls
resembles a button named Submit" into "which of these submits the expense claim
for approval", and that the second question has one answer. That is a claim
about model behaviour, so scripted turns cannot test it -- a fake that returns
index 2 returns index 2 whatever the prompt says.

This measures it. The same broken step, the same page, asked both ways, several
times, against a real model.

    RUN_LLM=1 ../.venv/Scripts/python -m pytest tests/test_eval_intent.py -q -s

Opt-in, like every other live-model test here, and for the same reason: it
spends money. A run is a few cents.

Each case is built so the *wording* of the recorded locator points at the wrong
answer and the purpose points at the right one. That is deliberate and it is
the situation this is for: where the label alone settles it, nothing needed
fixing.

What is asserted, and what is only reported
-------------------------------------------
Asserted: the purpose must not make accuracy *worse*, and it must never turn a
correct answer into a destructive one. Those are not judgement calls.

Reported only: how much better. A threshold on that would fail on model
variance rather than on a regression, and an instrument that cries wolf gets
switched off -- which is how a measurement stops being one.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field

import pytest

from healing import HealingBudget, StepHealer
from snapshot import parse as parse_snapshot
from usecase import Locator, Step

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.llm,
    pytest.mark.eval,
    pytest.mark.skipif(
        os.environ.get("RUN_LLM") != "1",
        reason="set RUN_LLM=1 to measure this against a real model (it spends money)",
    ),
]

#: How many times each case is asked each way. Three is the smallest number
#: that can show a model answering the same question differently twice.
REPEATS = int(os.environ.get("EVAL_REPEATS", "3"))


def page(*controls: str) -> str:
    body = "\n".join(f"- {control} [ref=e{i + 1}]" for i, control in enumerate(controls))
    return '### Page\n- Page URL: https://vendor.test/record\n### Snapshot\n```yaml\n' + body + "\n```"


@dataclass
class Case:
    """One broken step, one changed page, and the two ways of asking."""

    name: str
    #: What the recording looked for, and no longer finds.
    wanted: Locator
    #: The page as it is now.
    snapshot: str
    #: The recorder's own sentence about what the step was for.
    intent: str
    #: The control the purpose identifies, by its accessible name.
    right: str
    #: The control the recorded *wording* points at instead. This is what a
    #: model matching on labels alone tends to pick.
    plausible: str
    #: True where picking wrongly does damage rather than failing a row. A
    #: purpose that causes one of these is worse than no purpose at all.
    destructive: bool = False


def cases() -> list[Case]:
    return [
        Case(
            name="submit became three different verbs",
            wanted=Locator(strategy="role", role="button", name="Submit"),
            snapshot=page(
                'button "Save draft"',
                'button "Send for approval"',
                'button "Discard"',
            ),
            intent="submits the expense claim for approval",
            right="Send for approval",
            plausible="Save draft",
        ),
        Case(
            name="one name field became three",
            wanted=Locator(strategy="role", role="textbox", name="Name"),
            snapshot=page(
                'textbox "Given name"',
                'textbox "Family name"',
                'textbox "Company name"',
            ),
            intent="types the company the invoice is billed to",
            right="Company name",
            plausible="Given name",
        ),
        Case(
            name="remove and delete are not the same thing",
            wanted=Locator(strategy="role", role="button", name="Delete"),
            snapshot=page(
                'button "Remove from list"',
                'button "Delete permanently"',
                'button "Archive"',
            ),
            intent="takes the row off the working list without destroying it",
            right="Remove from list",
            plausible="Delete permanently",
            destructive=True,
        ),
        Case(
            name="export became three ways out",
            wanted=Locator(strategy="role", role="link", name="Export"),
            snapshot=page(
                'link "Print"',
                'link "Download CSV"',
                'link "Share a link"',
            ),
            intent="downloads the spreadsheet of results for the row",
            right="Download CSV",
            plausible="Print",
        ),
    ]


@dataclass
class Tally:
    asked: int = 0
    right: int = 0
    #: Picked the control the old wording pointed at.
    plausible: int = 0
    other: int = 0
    #: Answered "none of these", which is honest and is not a wrong click.
    refused: int = 0
    destructive: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.right / self.asked if self.asked else 0.0

    def as_dict(self) -> dict:
        return {
            "asked": self.asked,
            "right": self.right,
            "chose_the_old_wording": self.plausible,
            "chose_something_else": self.other,
            "refused": self.refused,
            "destructive_choices": self.destructive,
            "accuracy": round(self.accuracy, 3),
        }


async def _ask(case: Case, *, with_intent: bool) -> str:
    """What the healer picked, by accessible name. "" means it refused."""
    from config import Settings
    from llm import build_llm

    step = Step(
        id="s1",
        action="click" if case.wanted.role != "textbox" else "fill",
        locators=[case.wanted],
        value="Acme Ltd" if case.wanted.role == "textbox" else None,
        intent=case.intent if with_intent else "",
        on_failure="heal",
    )
    healer = StepHealer(
        build_llm(Settings()),
        HealingBudget(max_attempts=2, max_tokens=40_000),
    )
    repair = await healer.repair(step, parse_snapshot(case.snapshot))
    if repair is None:
        return ""
    return repair.locator.name or ""


def _score(case: Case, chosen: str, tally: Tally) -> None:
    tally.asked += 1
    if not chosen:
        tally.refused += 1
    elif chosen == case.right:
        tally.right += 1
    elif chosen == case.plausible:
        tally.plausible += 1
        if case.destructive:
            tally.destructive.append(case.name)
    else:
        tally.other += 1
        if case.destructive:
            tally.destructive.append(case.name)


async def test_a_purpose_makes_a_model_pick_better():
    blind = Tally()
    told = Tally()
    per_case: dict[str, dict] = {}

    for case in cases():
        one_blind, one_told = Tally(), Tally()
        for _ in range(REPEATS):
            _score(case, await _ask(case, with_intent=False), one_blind)
            _score(case, await _ask(case, with_intent=True), one_told)
        per_case[case.name] = {
            "without_purpose": one_blind.as_dict(),
            "with_purpose": one_told.as_dict(),
        }
        for source, into in ((one_blind, blind), (one_told, told)):
            into.asked += source.asked
            into.right += source.right
            into.plausible += source.plausible
            into.other += source.other
            into.refused += source.refused
            into.destructive += source.destructive

    report = {
        "repeats": REPEATS,
        "cases": len(cases()),
        "without_purpose": blind.as_dict(),
        "with_purpose": told.as_dict(),
        "gain": round(told.accuracy - blind.accuracy, 3),
        "per_case": per_case,
    }
    print("\n" + json.dumps(report, indent=2))

    # The two that are not judgement calls.
    assert told.accuracy >= blind.accuracy, (
        "telling the model what the step was for made it worse: " + json.dumps(report)
    )
    assert told.destructive == [], (
        "a purpose led to a destructive choice, which is worse than no purpose"
    )


async def test_a_purpose_does_not_make_a_model_answer_differently_each_time():
    """Consistency, separately from accuracy. A prompt addition that raises the
    mean while widening the spread has not made the system more trustworthy,
    and "I cannot trust it" was the actual complaint."""
    case = cases()[0]
    answers = [await _ask(case, with_intent=True) for _ in range(REPEATS)]

    print("\nanswers with the purpose: " + json.dumps(answers))
    assert len(set(answers)) == 1, "the same question gave different answers"
    assert statistics.mode(answers) == case.right
