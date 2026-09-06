"""What the graph gives you that a `while` loop would not.

The nodes are tested in ``test_agent_author.py`` without LangGraph, because
that is most of the behaviour and it should not need a graph library. What is
left is the part that is genuinely LangGraph's: the conditional edges, the
checkpoint written per node, and a real interrupt that survives being resumed
later rather than a poll on a flag.

Skipped where the optional extra is not installed, which is the same
compatibility contract the rest of the agent package holds to.
"""

from __future__ import annotations

import pytest

from agent import Budget, run_agent_session
from agent.author import AuthorRequest, FINISH
from agent.graph import available
from test_agent_author import ScriptedLLM, turn_calling
from test_agent_marks import INVITE
from test_agent_tools import FakeMCP
from llm import LLMTurn

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not available(), reason="LangGraph is an optional extra"),
]


def a_session(*turns):
    """A provider and a model, scripted. Returns both so a test can look."""
    return FakeMCP({"browser_snapshot": INVITE}), ScriptedLLM(list(turns))


def request(**kwargs) -> AuthorRequest:
    return AuthorRequest(
        task=kwargs.pop("task", "Read the balance for one account."),
        start_url="https://vendor.test/users",
        allowed_domains=("vendor.test",),
        may_write=True,
        run_id=kwargs.pop("run_id", "graph-run"),
        budget=kwargs.pop("budget", Budget()),
        **kwargs,
    )


PLAN = LLMTurn(text="Sign in, open the record, read the balance.",
               usage={"input_tokens": 100, "output_tokens": 20})


def a_complete_session():
    """The shortest thing that counts as a recording: a row, opened and closed."""
    return [
        PLAN,
        turn_calling("browser_snapshot"),
        turn_calling("mark_setup_complete"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("mark_as_output", ref="e10", column="status"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="Read the status for A-1001."),
    ]


# --- the loop runs ---------------------------------------------------------


async def test_a_session_runs_end_to_end_through_the_graph():
    provider, llm = a_session(*a_complete_session())
    events = []

    result = await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep(events, e)
    )

    assert result.status == "succeeded", result.stopped_by
    assert result.summary == "Read the status for A-1001."
    assert result.unfinished == "", "it marked a row, so it can be distilled"
    assert [e.type for e in events][0] == "run_started"
    assert [e.type for e in events][-1] == "run_finished"


async def test_the_trajectory_is_what_distillation_will_read():
    """Ordered tool calls with what each would become. Phase E reads this."""
    provider, llm = a_session(*a_complete_session())

    result = await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep([], e)
    )

    assert [call["tool"] for call in result.trajectory] == [
        "browser_snapshot",
        "mark_setup_complete",
        "begin_row",
        "mark_as_output",
        "end_row",
    ]
    assert [mark["kind"] for mark in result.marks] == [
        "setup_complete",
        "begin_row",
        "mark_as_output",
        "end_row",
    ]


async def test_the_browser_is_closed_when_the_session_ends():
    provider, llm = a_session(*a_complete_session())

    await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep([], e)
    )

    assert provider.closed


# --- the edges -------------------------------------------------------------


async def test_a_step_budget_ends_the_session_rather_than_running_forever():
    """The model here never calls finish. Without a budget this is the loop
    that spends everything, which is why the limit is checked before each call
    rather than reported after."""
    provider, llm = a_session(PLAN, *[turn_calling("browser_snapshot")] * 50)

    result = await run_agent_session(
        request(budget=Budget(steps=4, tokens=None, seconds=None, usd=None)),
        llm=llm,
        provider=provider,
        emit=lambda e: _keep([], e),
    )

    assert result.status == "partial"
    assert "Stopped after 4 steps" in result.stopped_by
    assert result.trajectory, "everything it did before the limit is kept"


async def test_prose_goes_round_again_rather_than_ending_the_session():
    """The conditional edge that would be a `continue`."""
    provider, llm = a_session(
        PLAN,
        LLMTurn(text="Thinking about it."),
        *a_complete_session()[1:],
    )

    result = await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep([], e)
    )

    assert result.status == "succeeded"


# --- the interrupt ---------------------------------------------------------


async def test_an_irreversible_action_suspends_the_graph_and_waits():
    """Not a poll on a flag: the graph stops, the state is persisted, and the
    session is resumable from exactly here. This is the rendezvous
    `RUN_APPROVE` was defined for and never got."""
    from agent import graph as graph_module

    provider, llm = a_session(
        PLAN,
        turn_calling("browser_snapshot"),
        turn_calling("browser_click", target="e3", element="Delete this account"),
    )
    events = []
    checkpointer = graph_module.memory_checkpointer()

    result = await run_agent_session(
        request(),
        llm=llm,
        provider=provider,
        emit=lambda e: _keep(events, e),
        checkpointer=checkpointer,
    )

    # Suspension is a status, not an exception. A person being needed is a fact
    # about the session, and a caller that has to catch an error to learn it
    # cannot easily hold the session open and come back.
    assert result.status == "awaiting_approval"
    assert result.awaiting is not None
    assert result.awaiting["call"]["name"] == "browser_click"

    asked = next(e for e in events if e.type == "approval_required")
    assert asked.name == "browser_click"
    assert asked.categories, "it says why it stopped"
    assert provider.calls[-1][0] == "browser_snapshot", "the click did not run"
    assert not any(e.type == "run_finished" for e in events), "it is not over"


async def test_resuming_with_an_approval_continues_from_where_it_stopped():
    """The whole point of the checkpoint. Same thread id, a decision, and the
    graph carries on -- four seconds later in a tab, or the next morning."""
    from langgraph.types import Command

    from agent import graph as graph_module
    from agent.author import Wiring, initial_state
    from agent.budget import Spend
    from agent.session import AgentToolSession

    provider, llm = a_session(
        PLAN,
        turn_calling("browser_snapshot"),
        turn_calling("browser_click", target="e3", element="Delete this account"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="Deleted, as approved."),
    )
    events = []
    checkpointer = graph_module.memory_checkpointer()
    config = {"configurable": {"thread_id": "resumable"}, "recursion_limit": 400}

    session = AgentToolSession(provider, allowed_domains=("vendor.test",), may_write=True)
    async with session:
        wiring = Wiring(
            request=request(run_id="resumable"),
            tools=session,
            llm=llm,
            spend=Spend(budget=Budget()),
            emit=lambda e: _keep(events, e),
        )
        compiled = graph_module.build(wiring, checkpointer)
        async for _ in compiled.astream(
            initial_state(), config=config, stream_mode="values"
        ):
            pass

        assert any(e.type == "approval_required" for e in events)
        assert provider.calls[-1][0] == "browser_snapshot"

        async for _ in compiled.astream(
            Command(resume={"decision": "approved"}), config=config, stream_mode="values"
        ):
            pass

    assert any(e.type == "approval_resolved" for e in events)
    assert "browser_click" in [name for name, _ in provider.calls], "it ran after approval"
    assert any(e.type == "run_finished" for e in events)


async def _keep(bucket, event):
    bucket.append(event)


# --- the compatibility contract -------------------------------------------


def test_the_graph_module_imports_langgraph_only_inside_a_function():
    """Importing it at module scope would make the whole agent package
    unimportable where only requirements.txt was installed."""
    import inspect

    from agent import graph as graph_module

    source = inspect.getsource(graph_module)
    top_level = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    assert not any("langgraph" in line for line in top_level), top_level


async def test_the_budget_is_counted_where_it_is_checked():
    """A limit that is always satisfied is not a limit.

    An earlier version incremented the step count in the state and checked it
    on the Spend, which was never incremented -- so the step budget was
    present, checked before every call, and could not trip. The symptom was a
    loop that ran to LangGraph's recursion limit.
    """
    provider, llm = a_session(PLAN, *[turn_calling("browser_snapshot")] * 10)

    result = await run_agent_session(
        request(budget=Budget(steps=3, tokens=None, seconds=None, usd=None)),
        llm=llm,
        provider=provider,
        emit=lambda e: _keep([], e),
    )

    assert result.spend["steps"] == 3
    assert result.status == "partial"


async def test_a_session_that_stops_at_its_budget_still_says_it_is_over():
    """`finish` is a node like any other, and an earlier wrapper skipped every
    node once the session was done -- including the node whose whole job is to
    announce that it is done. The run view would have shown a run that never
    ended."""
    provider, llm = a_session(PLAN, *[turn_calling("browser_snapshot")] * 10)
    events = []

    await run_agent_session(
        request(budget=Budget(steps=2, tokens=None, seconds=None, usd=None)),
        llm=llm,
        provider=provider,
        emit=lambda e: _keep(events, e),
    )

    assert [e.type for e in events][-1] == "run_finished"
