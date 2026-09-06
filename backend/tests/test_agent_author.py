"""The authoring loop: budgets, nodes, and the graph that wires them.

Split the way the code is. The node functions are plain and are tested here
with no graph library, no browser and no model -- which is most of the
behaviour. The graph's own contribution, the parts a `while` loop would not
give you, is tested separately and skipped where LangGraph is not installed:
the conditional edges, the checkpoint, and the interrupt.
"""

from __future__ import annotations

import pytest

from agent import Budget, BudgetExhausted, Spend
from agent.author import (
    AuthorRequest,
    AuthorState,
    FINISH,
    Wiring,
    initial_state,
    act,
    decide,
    finish,
    load_context,
    needs_approval,
    plan,
    resolve,
    stop_for_budget,
    system_prompt,
)
from agent.budget import price_of
from agent.session import AgentToolSession
from llm import LLMTurn, ToolCallRequest
from test_agent_marks import INVITE
from test_agent_tools import FakeMCP

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


# --- a model that is not a model ------------------------------------------


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


async def build_state(*turns: LLMTurn, **kwargs) -> tuple[AuthorState, Wiring]:
    """State and wiring, kept apart the way the graph keeps them.

    The live objects -- the tool session, the model -- are not in the state,
    because the state is what a checkpoint holds and a browser session cannot
    be written to Postgres.
    """
    provider = FakeMCP({"browser_snapshot": INVITE})
    session = AgentToolSession(
        provider, allowed_domains=("vendor.test",), may_write=kwargs.pop("may_write", True)
    )
    await session.__aenter__()
    request = AuthorRequest(
        task=kwargs.pop("task", "Read the balance for the account in this row."),
        start_url="https://vendor.test/users",
        allowed_domains=("vendor.test",),
        run_id="r1",
        budget=kwargs.pop("budget", Budget()),
        **kwargs,
    )
    events: list = []
    wiring = Wiring(
        request=request,
        tools=session,
        llm=ScriptedLLM(list(turns)),
        spend=Spend(budget=request.budget),
        emit=lambda event: collect(events, event),
    )
    wiring.events = events  # type: ignore[attr-defined]
    return initial_state(), wiring


async def collect(bucket, event):
    bucket.append(event)


# --- the nodes -------------------------------------------------------------


async def test_the_system_prompt_names_the_allowlist():
    """The guard enforces it either way; an agent that knows the rule wastes
    fewer turns discovering it."""
    request = AuthorRequest(task="x", start_url="u", allowed_domains=("vendor.test",))
    assert "vendor.test" in system_prompt(request)


async def test_starting_a_session_announces_it_on_the_existing_stream():
    """An agent run appears in the run view a replay uses, with no new event
    types: the vocabulary was already there from the agent that was deleted."""
    state, w = await build_state()

    state = await load_context(state, w)

    started = w.events[0]
    assert started.type == "run_started"
    assert started.task == w.request.task
    assert FINISH in started.tools
    assert "browser_snapshot" in started.tools


async def test_planning_costs_one_call_and_produces_prose_not_steps():
    """An outline is a map to notice deviation against. Steps guessed before
    seeing the page would be exactly the invented locators this system exists
    to prevent."""
    state, w = await build_state(
        LLMTurn(text="Sign in, search, open the record, read the balance.",
                usage={"input_tokens": 200, "output_tokens": 30})
    )

    state = await plan(state, w)

    assert state["outline"].startswith("Sign in")
    assert w.spend.llm_calls == 1
    assert w.spend.tokens == 230
    assert w.llm.asked[0]["tools"] == [], "planning has no tools to call"


async def test_deciding_offers_the_browser_tools_the_marks_and_finish():
    state, w = await build_state(turn_calling("browser_snapshot"))

    state = await decide(state, w)

    offered = {tool["name"] for tool in w.llm.asked[0]["tools"]}
    assert "browser_snapshot" in offered
    assert "mark_as_output" in offered
    assert FINISH in offered
    assert "browser_evaluate" not in offered, "refusals are removals, everywhere"


async def test_a_turn_with_no_tool_call_is_nudged_rather_than_ended():
    """Prose alone does not move the browser. Said once it is harmless; said
    repeatedly it is a loop that spends the whole budget producing text."""
    state, w = await build_state(LLMTurn(text="Let me think about this."))

    state = await decide(state, w)

    assert state["pending"] is None
    assert not state["done"]
    assert "Call a tool" in state["messages"][-1]["content"]


async def test_acting_runs_the_call_and_hands_the_result_back_to_the_model():
    state, w = await build_state(turn_calling("browser_snapshot"))

    state = await decide(state, w)
    state = await act(state, w)

    kinds = [e.type for e in w.events]
    assert "tool_call" in kinds and "tool_result" in kinds
    answer = state["messages"][-1]["content"][0]
    assert answer["type"] == "tool_result"
    assert "Invite" in answer["content"], "the model sees the page it just took"


async def test_a_refused_call_is_answered_as_an_error_the_model_can_act_on():
    state, w = await build_state(turn_calling("browser_click", target="#invite"))

    state = await decide(state, w)
    state = await act(state, w)

    answer = state["messages"][-1]["content"][0]
    assert answer["is_error"] is True
    assert "not an element reference" in answer["content"]


# --- finishing -------------------------------------------------------------


async def test_finishing_without_a_row_boundary_is_refused_and_the_session_goes_on():
    """A recording with no row boundary cannot be distilled at all.

    Accepting it here would mean discovering that on the review screen with
    nothing left to do about it, so the refusal happens while the browser is
    still open and the agent can still fix it.
    """
    state, w = await build_state(turn_calling(FINISH, summary="done"))

    state = await decide(state, w)
    state = await act(state, w)

    assert not state["done"]
    answer = state["messages"][-1]["content"][0]
    assert answer["is_error"] is True
    assert "No row was recorded" in answer["content"]


async def test_finishing_after_marking_a_row_ends_the_session():
    state, w = await build_state(
        turn_calling("browser_snapshot"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="Read the balance for A-1001."),
    )

    for _ in range(4):
        state = await decide(state, w)
        state = await act(state, w)

    assert state["done"]
    assert state["status"] == "succeeded"
    assert state["summary"].startswith("Read the balance")


async def test_an_incomplete_finish_is_recorded_as_partial():
    """"I could not do it and here is why" is a useful session. It is not a
    successful one, and saying so is the difference."""
    state, w = await build_state(
        turn_calling("browser_snapshot"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="The export button is disabled.", complete=False),
    )

    for _ in range(4):
        state = await decide(state, w)
        state = await act(state, w)

    assert state["status"] == "partial"


async def test_a_budget_stop_keeps_everything_and_says_which_limit():
    state, w = await build_state(budget=Budget(steps=1))
    state["step"] = 3

    state = await stop_for_budget(
        state, BudgetExhausted("steps", "Stopped after 1 steps."), w
    )

    assert state["status"] == "partial"
    assert "Stopped after 1 steps" in state["stopped_by"]
    error = next(e for e in w.events if e.type == "error")
    assert error.kind == "budget_steps"
    assert error.recoverable, "the session ended cleanly; it did not crash"
    assert w.events[-1].type == "run_finished"


# --- approval --------------------------------------------------------------


async def test_an_irreversible_call_stops_and_asks():
    """Classified from what the call does, not from the model's opinion of it."""
    state, w = await build_state(
        turn_calling("browser_snapshot"),
        turn_calling("browser_click", target="e3", element="Delete this account"),
    )

    state = await decide(state, w)
    state = await act(state, w)
    state = await decide(state, w)

    assert await needs_approval(state, w)


async def test_an_ordinary_call_does_not():
    state, w = await build_state(turn_calling("browser_snapshot"))
    state = await decide(state, w)

    assert not await needs_approval(state, w)


async def test_a_declined_action_tells_the_agent_not_to_retry_it():
    """One that only sees a failure retries the same thing until the budget is
    gone; one told a person declined can take the read-only route."""
    state, w = await build_state(
        turn_calling("browser_snapshot"),
        turn_calling("browser_click", target="e3", element="Delete this account"),
    )

    state = await decide(state, w)
    state = await act(state, w)
    state = await decide(state, w)
    state = await resolve(state, "rejected", w, "ap1")

    answer = state["messages"][-1]["content"][0]
    assert answer["is_error"] is True
    assert "Do not try it again" in answer["content"]
    assert w.tools.provider.calls[-1][0] == "browser_snapshot", "it never ran"


async def test_an_approved_action_runs():
    state, w = await build_state(
        turn_calling("browser_snapshot"),
        turn_calling("browser_click", target="e3", element="Delete this account"),
    )

    state = await decide(state, w)
    state = await act(state, w)
    state = await decide(state, w)
    state = await resolve(state, "approved", w, "ap1")

    assert w.tools.provider.calls[-1][0] == "browser_click"
    assert [e.type for e in w.events if e.type == "approval_resolved"]
