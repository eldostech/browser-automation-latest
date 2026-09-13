"""Two gates on a step: does it still find the right element, and should it run.

Both come from reading what other recorders do and finding a gap here.

**The continuity check.** Locator caches elsewhere validate a stored path
before trusting it: the element there must still say what it said. This
codebase had no equivalent, so a rung that says only *where* to look -- a CSS
path, a bare role, a test id -- could match exactly one element, be acted on,
and report success against a completely different control. That is the worst
outcome the eval harness measures, because it is recorded as a success and
repeats on every row.

**The condition.** `optional` says a failure is survivable. It does not say
"this step is not always needed", so a cookie banner that appears on the first
row and not the fourth still ran, still waited out its timeout, and still left
a failure for somebody to read. `when` is that missing statement, and it is an
`Assertion` -- the same check the executor already evaluates locally -- rather
than an expression language, because an expression language here would be a
script step wearing a smaller name.
"""

from __future__ import annotations

import pytest

from conftest import RecordingSink
from engine import UseCaseExecutor
from fake_browser import session_serving
from usecase import Assertion, Locator, Step, UseCase

pytestmark = pytest.mark.anyio


#: One button, and it is not the one that was recorded. A bare role rung
#: matches it: `role=button` says where to look and nothing about what it says.
RENAMED = """### Page
- Page URL: https://example.com/record
### Snapshot
```yaml
- button "Delete permanently" [ref=e1]
```"""

#: The same page as it was recorded against.
AS_RECORDED = """### Page
- Page URL: https://example.com/record
### Snapshot
```yaml
- button "Remove from list" [ref=e1]
```"""

WITH_A_BANNER = """### Page
- Page URL: https://example.com/record
### Snapshot
```yaml
- button "Accept cookies" [ref=e1]
- button "Save" [ref=e2]
```"""

WITHOUT_A_BANNER = """### Page
- Page URL: https://example.com/record
### Snapshot
```yaml
- button "Save" [ref=e2]
```"""


def running(use_case: UseCase, pages: list[str]):
    return UseCaseExecutor(
        use_case,
        session_serving(pages)(None),
        RecordingSink(),
        run_id="r1",
        step_timeout=0.4,
    )


def one_step(**overrides) -> UseCase:
    step = Step(
        id="s1",
        action="click",
        locators=[Locator(strategy="role", role="button")],
        **overrides,
    )
    return UseCase(name="x", allowed_domains=["example.com"], row_steps=[step])


# --- the continuity check ---------------------------------------------------


async def test_a_positional_rung_that_finds_a_different_control_does_not_act():
    """The failure this exists for. One button on the page, the rung matches
    it, and it is not the button that was recorded."""
    result = await running(one_step(expect_text="Remove from list"), [RENAMED]).run_row({})

    assert not result.ok
    assert "not the one that was recorded" in result.error
    assert "Delete permanently" in result.error, "what it found"
    assert "Remove from list" in result.error, "and what it wanted"


async def test_the_same_rung_acts_when_the_control_still_says_the_same_thing():
    result = await running(one_step(expect_text="Remove from list"), [AS_RECORDED]).run_row({})

    assert result.ok, result.error


async def test_a_step_with_no_recorded_text_behaves_exactly_as_it_did_before():
    """Every recording made before this existed, and every codegen recording,
    takes this path. It must not start refusing."""
    result = await running(one_step(), [RENAMED]).run_row({})

    assert result.ok, result.error


async def test_a_rung_that_matched_on_the_name_is_not_checked_twice():
    """A rung that found its element *by* accessible name has already proved
    the wording. Checking again could only produce a false refusal -- and would
    produce one here, where the recorded text is a trimmed form of the name."""
    use_case = UseCase(
        name="x",
        allowed_domains=["example.com"],
        row_steps=[
            Step(
                id="s1",
                action="click",
                locators=[Locator(strategy="role", role="button", name="Delete permanently")],
                expect_text="Delete",
            )
        ],
    )

    result = await running(use_case, [RENAMED]).run_row({})

    assert result.ok, result.error


async def test_containment_either_way_is_enough():
    """A wrapper's text includes its children's, a button may have gained a
    count beside it, and a recorded name is often a trimmed version of what the
    DOM holds. Equality here would refuse correct steps, which is the failure
    this must not introduce."""
    result = await running(one_step(expect_text="Delete"), [RENAMED]).run_row({})

    assert result.ok, result.error


# --- the condition ----------------------------------------------------------


def banner_step(**overrides) -> Step:
    return Step(
        id="s1",
        action="click",
        locators=[Locator(strategy="role", role="button", name="Accept cookies")],
        when=Assertion(
            kind="element_visible",
            locator=Locator(strategy="role", role="button", name="Accept cookies"),
        ),
        **overrides,
    )


def with_a_condition() -> UseCase:
    return UseCase(
        name="x",
        allowed_domains=["example.com"],
        row_steps=[
            banner_step(),
            Step(
                id="s2",
                action="click",
                locators=[Locator(strategy="role", role="button", name="Save")],
            ),
        ],
    )


async def test_a_conditional_step_runs_when_its_condition_holds():
    result = await running(with_a_condition(), [WITH_A_BANNER]).run_row({})

    assert result.ok, result.error
    assert [outcome.skipped for outcome in result.steps] == [False, False]


async def test_it_is_skipped_when_the_condition_does_not_hold():
    """And the row still succeeds. That is the whole difference from
    `optional`: nothing failed, so nothing has to be survived."""
    result = await running(with_a_condition(), [WITHOUT_A_BANNER]).run_row({})

    assert result.ok, result.error
    first, second = result.steps
    assert first.skipped and first.ok
    assert "did not hold" in first.message
    assert not second.skipped, "the step after it still ran"


async def test_the_skip_says_which_condition_did_not_hold():
    """A row that quietly did less than it was recorded doing is harder to
    notice than a failure, so the reason is recorded rather than implied."""
    result = await running(with_a_condition(), [WITHOUT_A_BANNER]).run_row({})

    assert 'name="Accept cookies" is visible' in result.steps[0].message


async def test_a_condition_is_not_waited_on():
    """Evaluated once, against the page as it is. Retrying would make "the
    banner is absent" cost the step timeout on every row of a batch, which is
    the cheapest possible way to make a thousand rows slow."""
    import time

    executor = running(with_a_condition(), [WITHOUT_A_BANNER])
    executor.step_timeout = 5.0
    started = time.monotonic()

    await executor.run_row({})

    assert time.monotonic() - started < 2.0, "it waited for the condition"


async def test_a_negated_condition_reads_the_other_way_round():
    use_case = UseCase(
        name="x",
        allowed_domains=["example.com"],
        row_steps=[
            Step(
                id="s1",
                action="click",
                locators=[Locator(strategy="role", role="button", name="Save")],
                when=Assertion(
                    kind="element_visible",
                    locator=Locator(strategy="role", role="button", name="Accept cookies"),
                    negate=True,
                ),
            )
        ],
    )

    with_banner = await running(use_case, [WITH_A_BANNER]).run_row({})
    without = await running(use_case, [WITHOUT_A_BANNER]).run_row({})

    assert with_banner.steps[0].skipped, "the banner is there, so this one does not run"
    assert not without.steps[0].skipped


async def test_a_condition_that_can_never_hold_is_refused_at_publish():
    """A `when` that can never hold does not fail a row, it skips the step on
    every row -- so the symptom is a workflow that quietly does less than it
    was recorded doing. Caught by the same gate that catches an impossible
    assertion."""
    with pytest.raises(Exception) as raised:
        UseCase(
            name="x",
            status="ready",
            allowed_domains=["example.com"],
            row_steps=[
                Step(
                    id="s1",
                    action="click",
                    locators=[Locator(strategy="role", role="button", name="Save")],
                    when=Assertion(kind="url_contains", value="example.com", negate=True),
                )
            ],
        )

    assert "s1 condition" in str(raised.value)


# --- a check about the row being processed ---------------------------------
#
# `Step.references` has always counted an assertion's value as a real template
# reference, so the schema said this worked. The executor compared the template
# text against the page, so it could never hold. `when` is what made it matter:
# the obvious condition to write is one about the record in hand.


PER_ROW = """### Page
- Page URL: https://example.com/record
### Snapshot
```yaml
- heading "Acme Ltd" [ref=e1]
- button "Save" [ref=e2]
```"""


def checking_the_row(kind: str = "text_present") -> UseCase:
    return UseCase(
        name="x",
        allowed_domains=["example.com"],
        inputs=[{"name": "customer"}],
        row_steps=[
            Step(
                id="s1",
                action="assert",
                **{
                    "assert": Assertion(
                        kind=kind, value="{{input.customer}}", timeout_ms=200
                    )
                },
            )
        ],
    )


async def test_an_assertion_about_a_row_value_is_rendered_before_it_is_checked():
    result = await running(checking_the_row(), [PER_ROW]).run_row({"customer": "Acme Ltd"})

    assert result.ok, result.error


async def test_and_fails_honestly_for_a_row_the_page_does_not_show():
    result = await running(checking_the_row(), [PER_ROW]).run_row({"customer": "Globex"})

    assert not result.ok
    assert "Globex" in result.error, "the failure names the value, not the template"


async def test_a_condition_can_be_about_the_row_being_processed():
    """The obvious thing to write, and the reason the rendering gap had to be
    closed rather than documented."""
    use_case = UseCase(
        name="x",
        allowed_domains=["example.com"],
        inputs=[{"name": "customer"}],
        row_steps=[
            Step(
                id="s1",
                action="click",
                locators=[Locator(strategy="role", role="button", name="Save")],
                when=Assertion(
                    kind="element_visible",
                    locator=Locator(strategy="role", role="heading", name="{{input.customer}}"),
                ),
            )
        ],
    )

    theirs = await running(use_case, [PER_ROW]).run_row({"customer": "Acme Ltd"})
    somebody_elses = await running(use_case, [PER_ROW]).run_row({"customer": "Globex"})

    assert not theirs.steps[0].skipped, "this row's heading is on the page"
    assert somebody_elses.steps[0].skipped, "this one's is not"
