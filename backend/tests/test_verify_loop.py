"""The verify loop: mend a draft that does not replay, then prove the mend.

Verification used to end at a verdict. A draft that did not replay came back
with a warning saying so, and a person fixed it by hand -- which is the gap
between "the agent did the task perfectly" and "the recording does not work",
and the one users actually feel.

The asymmetry behind that gap is worth stating, because every test here is
about it: an agent acts on ``ref=e12``, an index into a snapshot seconds old
that always names exactly one element. A replay acts on a *description*. So
recording cannot fail the way replay fails, and the first time anybody finds
out is the first replay -- which is exactly where this now intervenes.
"""

from __future__ import annotations

import pytest

from agent.verify import Verification, verify
from usecase import Locator, Step, UseCase

pytestmark = pytest.mark.anyio


def draft() -> UseCase:
    return UseCase(
        name="open a record",
        status="draft",
        allowed_domains=["example.com"],
        inputs=[{"name": "record_url", "type": "url"}],
        row_steps=[
            Step(id="s1", action="navigate", url="{{input.record_url}}"),
            Step(
                id="s2",
                action="click",
                locators=[Locator(strategy="role", role="generic", nth=24)],
            ),
        ],
    )


SAMPLE = {"record_url": "https://example.com/record/1"}


class Repair:
    """What a healer reports having mended."""

    def __init__(self, step_id: str, locator: Locator) -> None:
        self.step_id = step_id
        self.locator = locator


class Healer:
    """Mends the first pass, and records that it was asked to."""

    def __init__(self, repairs: list[Repair] | None = None) -> None:
        self.repairs = list(repairs or [])
        self.asked = 0


def replayer(*verdicts: Verification):
    """A replayer that answers each pass in turn, recording how it was called."""
    calls: list[dict] = []
    answers = list(verdicts)

    async def replay(use_case, inputs, secrets, **kwargs):
        calls.append(
            {
                "healer": kwargs.get("healer"),
                "leading": [
                    step.locators[0].describe() if step.locators else ""
                    for step in use_case.row_steps
                ],
            }
        )
        return answers.pop(0) if answers else Verification(ran=True, ok=True)

    replay.calls = calls  # type: ignore[attr-defined]
    return replay


# --- the old behaviour, unchanged when nothing can heal --------------------


async def test_with_no_healer_there_is_one_pass_and_one_verdict():
    """What a deployment with healing switched off gets, and what every test
    written before this existed gets."""
    replay = replayer(Verification(ran=True, ok=False, failed_step="s2", error="no match"))

    report = await verify(draft(), SAMPLE, {}, replay=replay)

    assert len(replay.calls) == 1
    assert not report.ok
    assert report.repairs == {} and not report.reproved
    assert "Did not replay" in report.as_text()


async def test_a_draft_that_replays_first_time_is_not_touched():
    healer = Healer()
    replay = replayer(Verification(ran=True, ok=True, outputs={"score": "92%"}))

    report = await verify(draft(), SAMPLE, {}, replay=replay, healer=healer)

    assert len(replay.calls) == 1, "no second pass when the first one worked"
    assert report.ok and report.repairs == {}
    assert "Replayed cleanly" in report.as_text()


# --- the loop closing ------------------------------------------------------


async def test_a_mended_draft_is_replayed_again_without_a_healer():
    """The second pass is the whole point. The first proves a *model* can get
    through; only a pass with nothing model-shaped in it proves the draft can.
    """
    healer = Healer([Repair("s2", Locator(strategy="role", role="radio", name="Nayra"))])
    replay = replayer(
        Verification(ran=True, ok=False, failed_step="s2", error="no match"),
        Verification(ran=True, ok=True),
    )

    report = await verify(draft(), SAMPLE, {}, replay=replay, healer=healer)

    assert len(replay.calls) == 2
    assert replay.calls[0]["healer"] is healer, "the first pass may heal"
    assert replay.calls[1]["healer"] is None, "the second pass may not"
    assert report.ok and report.reproved
    assert report.repairs == {"s2": 'role=radio name="Nayra"'}


async def test_the_second_pass_runs_the_repaired_locator():
    """Not the recorded one. A second pass on the original would prove nothing
    and would fail for the same reason the first did."""
    healer = Healer([Repair("s2", Locator(strategy="role", role="radio", name="Nayra"))])
    replay = replayer(
        Verification(ran=True, ok=False, failed_step="s2", error="no match"),
        Verification(ran=True, ok=True),
    )

    await verify(draft(), SAMPLE, {}, replay=replay, healer=healer)

    assert replay.calls[0]["leading"][1] == "role=generic [24]"
    assert replay.calls[1]["leading"][1] == 'role=radio name="Nayra"'


async def test_the_mended_definition_comes_back_for_the_caller_to_keep():
    """A repair proved by the second pass and then thrown away would be the
    most expensive possible way to learn nothing."""
    healer = Healer([Repair("s2", Locator(strategy="role", role="radio", name="Nayra"))])
    replay = replayer(
        Verification(ran=True, ok=False, failed_step="s2", error="no match"),
        Verification(ran=True, ok=True),
    )

    report = await verify(draft(), SAMPLE, {}, replay=replay, healer=healer)

    assert report.patched is not None
    leading = report.patched["row_steps"][1]["locators"][0]
    assert (leading["role"], leading["name"]) == ("radio", "Nayra")
    # Prepended, not replacing: a repair that turns out to be wrong degrades to
    # what the recording already knew.
    assert len(report.patched["row_steps"][1]["locators"]) == 2


async def test_a_draft_the_healer_could_not_mend_keeps_its_verdict():
    """A step that failed for a reason healing cannot touch -- an assertion
    that can never hold, a page that will not load -- must come back as "did
    not replay" rather than as a second pass nobody could improve."""
    healer = Healer()  # ran, mended nothing
    replay = replayer(
        Verification(ran=True, ok=False, failed_step="s2", error="the page 404'd")
    )

    report = await verify(draft(), SAMPLE, {}, replay=replay, healer=healer)

    assert len(replay.calls) == 1
    assert not report.ok and not report.reproved
    assert "404" in report.as_text()


async def test_a_mend_that_does_not_validate_leaves_the_first_verdict_standing():
    """Better a truthful failure than a draft nobody can construct."""

    class Bad:
        step_id = "s2"
        locator = "not a locator at all"

    healer = Healer()
    healer.repairs = [Bad()]
    replay = replayer(Verification(ran=True, ok=False, failed_step="s2", error="no match"))

    report = await verify(draft(), SAMPLE, {}, replay=replay, healer=healer)

    assert not report.ok and not report.reproved


async def test_a_repaired_draft_still_says_it_needed_repairing():
    """It replays, and a reviewer still has to know the recording did not work
    as recorded -- because the next recording of the same site probably will
    not either."""
    healer = Healer([Repair("s2", Locator(strategy="role", role="radio", name="Nayra"))])
    replay = replayer(
        Verification(ran=True, ok=False, failed_step="s2", error="no match"),
        Verification(ran=True, ok=True),
    )

    report = await verify(draft(), SAMPLE, {}, replay=replay, healer=healer)
    text = report.as_text()

    assert "after mending 1 step" in text
    assert "replayed again with no model involved" in text


async def test_an_older_injected_replayer_still_works():
    """A replayer written before healing was threaded through here takes three
    positional arguments and no keywords. It keeps working, on the verdict
    alone, because a verdict is worth more than a repair."""

    async def old_style(use_case, inputs, secrets):
        return Verification(ran=True, ok=True)

    report = await verify(draft(), SAMPLE, {}, replay=old_style, healer=Healer())

    assert report.ok
