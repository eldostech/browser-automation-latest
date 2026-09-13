"""The two passes that bracket a recording.

What each one is worth, stated as the failure it removes:

* **the brief** — a session used to start from whatever sentence somebody
  typed, and an undeclared per-row value cannot be marked, so it gets baked in
  as a constant. That is the difference between a use case that runs for four
  thousand customers and one that only ever works for the customer it was
  recorded with.
* **the walkthrough** — a step list says what happens and never what it is
  for, which is what a reviewer needs before publishing and what a repair
  needs months later before it can judge a replacement control.

Both must be incapable of costing a recording. Every failure path here is
tested for that specifically: a scribe that raises, that answers in prose, or
that names steps which do not exist must all leave a usable draft behind.
"""

from __future__ import annotations

import pytest

from agent.brief import (
    BRIEF_TOOL,
    WALKTHROUGH_TOOL,
    TaskBrief,
    apply_walkthrough,
    write_brief,
    write_walkthrough,
)
from llm import LLMTurn, ToolCallRequest
from usecase import Locator, Step, UseCase

pytestmark = pytest.mark.anyio


class Scribe:
    """Answers one turn with a tool call, and records what it was asked."""

    model = "scripted-scribe"

    def __init__(self, payload: dict | None, *, tool: str = "", tokens: int = 400) -> None:
        self.payload = payload
        self.tool = tool
        self.tokens = tokens
        self.calls = 0
        self.last_system = ""
        self.last_message = ""
        self.last_tools: list[dict] = []

    async def run_turn(self, *, system, messages, tools, on_text_delta=None, timeout=None):
        self.calls += 1
        self.last_system = system
        self.last_message = messages[-1]["content"]
        self.last_tools = tools
        usage = {"input_tokens": self.tokens, "output_tokens": 0}
        if self.payload is None:
            return LLMTurn(text="I would rather not", usage=usage)
        name = self.tool or tools[0]["name"]
        return LLMTurn(
            tool_calls=[ToolCallRequest(id="t1", name=name, input=self.payload)],
            stop_reason="tool_use",
            usage=usage,
        )


class Broken:
    model = "broken-scribe"

    async def run_turn(self, **kwargs):
        raise RuntimeError("the provider refused the request")


BRIEF = {
    "goal": "Each customer's billing address matches the address in the row.",
    "per_row": [
        {"name": "customer_number", "means": "the number typed into the search box"},
        {"name": "address", "means": "the new billing address"},
    ],
    "done_when": ["the page shows 'Address updated'"],
    "unclear": ["whether a customer with two addresses should have both changed"],
}


# --- the brief -------------------------------------------------------------


async def test_the_brief_names_the_values_that_vary():
    """The most valuable thing it produces. A value nobody declared cannot be
    marked, and an unmarked value is recorded as a constant."""
    brief = await write_brief(Scribe(BRIEF), task="update billing addresses")

    assert brief is not None
    assert [item["name"] for item in brief.per_row] == ["customer_number", "address"]
    assert brief.done_when == ["the page shows 'Address updated'"]


async def test_the_brief_says_the_task_wins_where_they_disagree():
    """It is one reading of the request, written by something that has never
    seen the site. Presented as an instruction it would override the thing it
    was derived from."""
    brief = await write_brief(Scribe(BRIEF), task="update billing addresses")

    text = brief.as_prompt()
    assert "alongside the task, never instead of it" in text
    assert "the task wins" in text
    assert "the page wins" in text


async def test_the_brief_surfaces_what_the_request_did_not_say():
    """Before the recording rather than after, which is the only point at
    which correcting an assumption is cheap."""
    brief = await write_brief(Scribe(BRIEF), task="update billing addresses")

    assert "Not stated in the task" in brief.as_prompt()
    assert "two addresses" in brief.as_prompt()


async def test_the_brief_pass_is_told_not_to_plan_clicks():
    """The rule that makes this help rather than hurt. A model that has never
    seen the site inventing an 'Advanced search' link sends the recorder
    hunting for something that does not exist."""
    scribe = Scribe(BRIEF)
    await write_brief(scribe, task="update billing addresses")

    assert "You are not planning clicks" in scribe.last_system
    assert "Name no control" in scribe.last_system


async def test_a_brief_with_no_goal_is_no_brief():
    """An empty goal renders an empty section, and an empty section in the
    recorder's prompt is worse than none: it reads as "there is nothing to
    achieve here"."""
    assert await write_brief(Scribe({"goal": "   "}), task="x") is None


async def test_a_scribe_that_answers_in_prose_is_not_an_error():
    """A legitimate answer to a request that was already precise."""
    assert await write_brief(Scribe(None), task="x") is None


async def test_a_scribe_that_raises_loses_the_brief_and_nothing_else():
    assert await write_brief(Broken(), task="x") is None


async def test_a_brief_whose_lists_come_back_as_strings_still_works():
    """A tool schema is a request, not a guarantee. This must not raise one
    call before a browser opens."""
    brief = await write_brief(
        Scribe({"goal": "g", "per_row": "customer_number", "done_when": "a banner"}),
        task="x",
    )

    assert brief is not None
    assert brief.per_row == [{"name": "customer_number", "means": ""}]
    assert brief.done_when == ["a banner"]


# --- the walkthrough -------------------------------------------------------


def recorded(**overrides) -> UseCase:
    base = dict(
        name="Update billing addresses",
        status="draft",
        allowed_domains=["vendor.test"],
        inputs=[{"name": "customer_number"}],
        setup_steps=[Step(id="s1", action="navigate", url="https://vendor.test/in")],
        row_steps=[
            Step(
                id="s2",
                action="fill",
                locators=[Locator(strategy="role", role="textbox", name="Search")],
                value="{{input.customer_number}}",
                intent="types the customer number the row gives",
            ),
            Step(
                id="s3",
                action="click",
                locators=[Locator(strategy="role", role="link", name="Billing")],
            ),
        ],
    )
    base.update(overrides)
    return UseCase(**base)


WALKTHROUGH = {
    "overview": (
        "Signs in once, then for each row searches for the customer by number "
        "and updates the billing address. A row is done when the page shows "
        "'Address updated'."
    ),
    "steps": [
        {"id": "s2", "purpose": "finds the customer the row names"},
        {"id": "s3", "purpose": "opens the customer's billing tab"},
    ],
}


async def test_the_flow_is_written_onto_the_draft_in_plain_language():
    use_case = recorded()
    walkthrough = await write_walkthrough(Scribe(WALKTHROUGH), use_case, task="t")

    apply_walkthrough(use_case, walkthrough)

    assert "Signs in once" in use_case.instructions
    assert "Address updated" in use_case.instructions


async def test_a_purpose_fills_a_step_that_has_none():
    """The gap this closes: a step whose description is a rendering of its own
    locator tells a repair nothing it does not already have."""
    use_case = recorded()
    walkthrough = await write_walkthrough(Scribe(WALKTHROUGH), use_case, task="t")

    apply_walkthrough(use_case, walkthrough)

    assert use_case.row_steps[1].intent == "opens the customer's billing tab"


async def test_it_never_overwrites_what_the_recorder_itself_said():
    """The agent's sentence was written with the page in front of it, one call
    before the action. This pass is reading a step list afterwards. Where both
    exist the first is the better evidence."""
    use_case = recorded()
    walkthrough = await write_walkthrough(Scribe(WALKTHROUGH), use_case, task="t")

    apply_walkthrough(use_case, walkthrough)

    assert use_case.row_steps[0].intent == "types the customer number the row gives"


async def test_a_purpose_for_a_step_that_does_not_exist_is_reported():
    """A model naming steps this recording does not have was describing
    something else, and the overview it wrote is then suspect too."""
    use_case = recorded()
    payload = {
        "overview": "does things",
        "steps": [{"id": "s99", "purpose": "invented"}],
    }
    walkthrough = await write_walkthrough(Scribe(payload), use_case, task="t")

    warnings = apply_walkthrough(use_case, walkthrough)

    assert walkthrough.unknown_steps == ["s99"]
    assert any("not in this use case" in w for w in warnings)
    assert all(step.intent != "invented" for step in use_case.all_steps)


async def test_nothing_written_can_change_what_runs():
    """The whole safety argument for both passes in one assertion."""
    use_case = recorded()
    before = [
        [rung.model_dump() for rung in step.locators] for step in use_case.all_steps
    ]
    actions = [step.action for step in use_case.all_steps]

    walkthrough = await write_walkthrough(Scribe(WALKTHROUGH), use_case, task="t")
    apply_walkthrough(use_case, walkthrough)

    assert [
        [rung.model_dump() for rung in step.locators] for step in use_case.all_steps
    ] == before
    assert [step.action for step in use_case.all_steps] == actions


async def test_a_recording_with_no_steps_is_not_described():
    """Nothing to say, and a model asked to describe nothing will say
    something."""
    scribe = Scribe(WALKTHROUGH)

    assert await write_walkthrough(scribe, UseCase(name="empty"), task="t") is None
    assert scribe.calls == 0


async def test_a_scribe_that_raises_after_the_recording_loses_only_the_prose():
    """By now a model has driven a browser for minutes and a person has
    watched it. Nothing here is allowed to cost that."""
    assert await write_walkthrough(Broken(), recorded(), task="t") is None


async def test_the_writer_is_shown_the_steps_it_must_describe():
    scribe = Scribe(WALKTHROUGH)
    await write_walkthrough(scribe, recorded(), task="update billing addresses")

    assert "s2: fill" in scribe.last_message
    assert 'role=link name="Billing"' in scribe.last_message
    assert "update billing addresses" in scribe.last_message
    assert "customer_number" in scribe.last_message, "the declared inputs"


async def test_the_brief_is_carried_into_the_walkthrough():
    """So the prose says what the workflow was *trying* to achieve, not only
    what the steps do."""
    scribe = Scribe(WALKTHROUGH)
    brief = TaskBrief(goal="Each customer's billing address matches the row.")

    await write_walkthrough(scribe, recorded(), task="t", brief=brief)

    assert "billing address matches the row" in scribe.last_message


def test_the_two_tools_ask_for_prose_and_never_for_a_locator():
    """Neither pass may express a locator, an index or a selector. That is the
    boundary that keeps them unable to affect a replay even by accident."""
    for tool in (BRIEF_TOOL, WALKTHROUGH_TOOL):
        fields = str(tool["input_schema"]["properties"]).lower()
        assert "locator" not in fields
        assert "selector" not in fields
        assert "element_index" not in fields


# --- what the token ceiling is for -----------------------------------------
#
# A real session recorded 405,305 tokens, cost 44 cents, and was stopped
# mid-record by a 400,000-token ceiling with over half its money unspent.
# Nearly all of those tokens were one long prompt prefix re-read on every
# turn, which is what prompt caching exists to make cheap. Counting a cache
# read against a runaway ceiling is budget starvation wearing the costume of a
# limit working.


def test_a_cache_read_does_not_count_against_the_token_ceiling():
    from agent.budget import Budget, Spend

    spend = Spend(Budget(tokens=400_000))
    spend.turn(
        {"input_tokens": 395_000, "output_tokens": 10_305, "cache_read_tokens": 360_000}
    )

    assert spend.tokens == 405_305, "the total is still reported truthfully"
    assert spend.fresh_tokens == 45_305
    assert spend.exceeded() is None, "this session had not run away"


def test_a_session_that_really_does_run_away_is_still_stopped():
    from agent.budget import Budget, Spend

    spend = Spend(Budget(tokens=100_000))
    spend.turn({"input_tokens": 120_000, "output_tokens": 0})

    hit = spend.exceeded()
    assert hit is not None and hit[0] == "tokens"


def test_the_message_says_which_number_stopped_it():
    """"Stopped at 405,305 tokens" beside a bill for 44 cents reads as a
    contradiction, and somebody then raises the wrong limit."""
    from agent.budget import Budget, Spend

    spend = Spend(Budget(tokens=50_000))
    spend.turn({"input_tokens": 300_000, "output_tokens": 0, "cache_read_tokens": 240_000})

    message = spend.exceeded()[1]
    assert "60,000 new tokens" in message
    assert "300,000 in total" in message
    assert "240,000 of them read from cache" in message
