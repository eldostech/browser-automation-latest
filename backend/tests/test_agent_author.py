"""What's left of the authoring session's own module: budgets, the system
prompt, and the message-pruning `agent/operate.py`'s recover/explore loop
still uses.

The node functions this file used to test directly (`decide`, `act`, `ask`,
`resolve`, `finish`, `stop_for_budget`, `load_context`, `plan`) were deleted
from `agent/author.py` once `create_agent` (`agent/graph.py`) made them
entirely dead -- nothing called them outside their own tests. What they
proved is covered against the real graph in `test_create_agent_graph.py`:
budget stops cleanly and keeps the trajectory, a step budget is actually
counted, an incomplete `finish` is refused, an irreversible action suspends
and resumes exactly once.

`ScriptedLLM`/`turn_calling` stay here rather than moving: `agent/operate.py`
(mid-replay recovery) still talks to a model through the older `LLMClient`
protocol these implement, and `test_agent_operate.py` imports them from this
file.
"""

from __future__ import annotations

import pytest

from agent import Budget, BudgetExhausted, Spend
from agent.author import system_prompt, AuthorRequest
from pricing import price_of
from llm import LLMTurn, ToolCallRequest

pytestmark = pytest.mark.anyio


# --- budgets ---------------------------------------------------------------


def test_each_limit_fails_differently_because_each_catches_something_else():
    """Four limits, and none of them is redundant.

    Tokens is the bill. Steps catches the loop that is making progress in its
    own opinion and none in anybody else's. Seconds catches the page that never
    finishes loading, which spends no tokens at all.
    """
    assert Spend(budget=Budget(steps=2), steps=2).exceeded()[0] == "steps"
    assert Spend(budget=Budget(tokens=10), tokens=10).exceeded()[0] == "tokens"
    assert Spend(budget=Budget(usd=0.5), usd=0.5).exceeded()[0] == "usd"

    slow = Spend(budget=Budget(seconds=0.0))
    assert slow.exceeded()[0] == "seconds"


def test_a_budget_that_has_room_left_says_nothing():
    assert Spend(budget=Budget(steps=40), steps=3).exceeded() is None


def test_an_unpriced_model_does_not_look_free():
    """A model missing from the table must not cost zero: an estimate of $0
    for a four-thousand-row batch is the most expensive kind of wrong.
    """
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    assert price_of("some-model-nobody-listed", usage) > 0


def test_a_bedrock_model_id_is_matched_despite_its_prefix_and_suffix():
    """Real ids carry a region prefix and a version suffix, so an exact lookup
    would fall through to the default for every actual deployment."""
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    assert price_of("us.anthropic.claude-haiku-4-5-20251001-v1:0", usage) == pytest.approx(0.80)


def test_reaching_a_limit_is_a_stop_that_keeps_what_it_did():
    """Often the recording is complete and the agent was about to tidy up."""
    exhausted = Spend(budget=Budget(steps=1), steps=1)
    with pytest.raises(BudgetExhausted) as caught:
        exhausted.check()
    assert caught.value.limit == "steps"
    assert "raise the step budget" in str(caught.value)


def test_the_system_prompt_names_the_allowlist():
    """The guard enforces it either way; an agent that knows the rule wastes
    fewer turns discovering it."""
    request = AuthorRequest(task="x", start_url="u", allowed_domains=("vendor.test",))
    assert "vendor.test" in system_prompt(request)


# --- a model that is not a model --------------------------------------------
#
# Used by `test_agent_operate.py`: `agent/operate.py` still talks to a model
# through `LLMClient`/`run_turn`, unaffected by the `create_agent` migration.


class ScriptedLLM:
    """Replays a list of turns. Records what it was asked, so a test can
    assert on the tools the model was actually offered."""

    model = "claude-sonnet-5"

    def __init__(self, turns: list[LLMTurn]) -> None:
        self.turns = list(turns)
        self.asked: list[dict] = []

    async def run_turn(self, *, system, messages, tools, **kwargs) -> LLMTurn:
        self.asked.append({"system": system, "messages": list(messages), "tools": tools})
        return self.turns.pop(0) if self.turns else LLMTurn(text="")

    def describe(self):
        return {"model": self.model}


def turn_calling(tool: str, **arguments) -> LLMTurn:
    """A turn that calls one tool.

    The parameter is `tool` and not `name` because `mark_as_input` takes a
    `name` of its own, and a helper that cannot express the call it is meant to
    make is a helper that quietly narrows the tests.
    """
    return LLMTurn(
        tool_calls=[ToolCallRequest(id=f"c{tool}", name=tool, input=arguments)],
        usage={"input_tokens": 100, "output_tokens": 20},
    )


# --- what a session costs --------------------------------------------------


def test_stale_pages_are_dropped_from_the_model_context():
    """Found by a real session, at real cost.

    Every tool result is a full accessibility tree, and they stayed in the
    history forever -- so turn N carried N snapshots and a session's cost grew
    with the square of its length. One measured 7,700 tokens per call and ran
    out of budget after doing the task correctly but before calling end_row,
    which lost the recording.
    """
    from agent.author import SNAPSHOTS_KEPT, for_model

    def page(n: int) -> dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": f"c{n}",
                    "content": f"### Page\n- Page URL: /p{n}\n" + "x" * 4000,
                }
            ],
        }

    history = [{"role": "user", "content": "do the task"}]
    for index in range(5):
        history.append({"role": "assistant", "content": f"turn {index}"})
        history.append(page(index))

    pruned = for_model(history)

    assert len(pruned) == len(history), "messages are replaced, never removed"
    full = [m for m in pruned if "xxxx" in str(m["content"])]
    assert len(full) == SNAPSHOTS_KEPT, "only the current pages stay in full"
    assert "/p4" in str(pruned[-1]["content"]), "and the newest is one of them"


def test_a_dropped_page_says_why_it_is_gone():
    """A model reading back must not conclude the page went blank -- and the
    reason is the useful part: those refs are stale and the guard refuses
    them, so keeping the page was paying to send something unusable."""
    from agent.author import for_model

    history = []
    for index in range(4):
        history.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"c{index}",
                     "content": "### Page\nold"},
                ],
            }
        )

    text = str(for_model(history)[0]["content"])
    assert "refs are stale" in text
    assert "Take a snapshot" in text


def test_anything_that_is_not_a_page_is_left_alone():
    """A refusal, a mark's answer, the task itself: all small and all worth
    keeping. Pruning by size rather than by kind would have eaten them."""
    from agent.author import for_model

    history = [
        {"role": "user", "content": "the task"},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1",
             "content": "e3 recorded as 'balance'."}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c2",
             "content": "not an element reference"}]},
    ]

    assert for_model(history) == history
