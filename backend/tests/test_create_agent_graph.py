"""The rebuilt authoring graph: `create_agent` + `BudgetMiddleware` +
`FinishMiddleware` + `HumanInTheLoopMiddleware`, replacing the hand-rolled
`StateGraph` this file used to test.

Same shape as the old `test_agent_graph.py`: a fake browser and a scripted
model, driving `run_agent_session`/`AgentSession` for real, with nothing
mocked below the graph itself.
"""

from __future__ import annotations

import pytest

from agent import AgentSession, AuthorRequest, Budget, run_agent_session
from agent.tools.finish import NAME as FINISH
from agent.graph import available
from agent.verify import Verification
from llm import LangChainLLM
from scripted_chat_model import ScriptedChatModel, turn_calling
from test_agent_marks import INVITE
from test_agent_tools import FakeMCP

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not available(), reason="langchain is an optional extra"),
]


def a_session(*turns):
    """A provider and a model, scripted. `LangChainLLM` wraps the fake model
    the same way it wraps a real one -- `Wiring.llm` is always this shape,
    and `.raw`/`.model` are what the graph actually reads from it."""
    llm = LangChainLLM(ScriptedChatModel(responses=list(turns)), "scripted-model")
    return FakeMCP({"browser_snapshot": INVITE}), llm


async def _replays(use_case, inputs, secrets):
    return Verification(ran=True, ok=True)


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


def a_complete_session():
    return [
        turn_calling("browser_snapshot"),
        turn_calling("mark_setup_complete"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("mark_as_output", ref="e10", column="status"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="Read the status for A-1001."),
    ]


async def _keep(bucket, event):
    bucket.append(event)


# --- the loop runs -----------------------------------------------------


async def test_a_session_runs_end_to_end_through_the_graph():
    provider, llm = a_session(*a_complete_session())
    events = []

    result = await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep(events, e),
        replay=_replays,
    )

    assert result.status == "succeeded", result.stopped_by
    assert result.summary == "Read the status for A-1001."
    assert result.unfinished == "", "it marked a row, so it can be distilled"
    assert [e.type for e in events][0] == "run_started"


async def test_the_trajectory_is_what_distillation_will_read():
    provider, llm = a_session(*a_complete_session())

    result = await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep([], e),
        replay=_replays,
    )

    assert [call["tool"] for call in result.trajectory] == [
        "browser_snapshot",
        "mark_setup_complete",
        "begin_row",
        "mark_as_output",
        "end_row",
    ]


async def test_the_browser_is_closed_when_the_session_ends():
    provider, llm = a_session(*a_complete_session())

    await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep([], e),
        replay=_replays,
    )

    assert provider.closed


async def test_a_call_to_a_registered_servers_tool_never_becomes_a_step():
    """Phase 1's invariant, proved end to end against the new graph: a tool
    from a registered MCP server is for the agent's own use while it works,
    and the distilled use case must never carry a step derived from it --
    only Playwright's own tool names distil into anything at all."""
    from agent.providers import ToolSpec
    from test_agent_tools import FakeExtraServer

    provider, llm = a_session(
        turn_calling("browser_snapshot"),
        turn_calling("mark_setup_complete"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("crm.lookup_account", id="A-1001"),
        turn_calling("mark_as_output", ref="e10", column="status"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="Read the status for A-1001, checked against the CRM."),
    )
    extra = FakeExtraServer([
        ToolSpec("lookup_account", "Look up a CRM account", annotations={"readOnlyHint": True})
    ])

    result = await run_agent_session(
        request(), llm=llm, provider=provider, extra={"crm": extra},
        emit=lambda e: _keep([], e), replay=_replays,
    )

    assert result.status == "succeeded", result.stopped_by
    assert extra.calls == [("lookup_account", {"id": "A-1001"})], "the call did happen"
    assert "crm.lookup_account" in [call["tool"] for call in result.trajectory]

    steps = result.use_case["row_steps"] + result.use_case["setup_steps"]
    assert not any("crm" in str(step) or "lookup_account" in str(step) for step in steps), (
        "a tool call to a registered server leaked into the replayable use case"
    )


# --- the budget ----------------------------------------------------------


async def test_a_step_budget_ends_the_session_rather_than_running_forever():
    provider, llm = a_session(*[turn_calling("browser_snapshot")] * 50)

    result = await run_agent_session(
        request(budget=Budget(steps=4, tokens=None, seconds=None, usd=None)),
        llm=llm, provider=provider, emit=lambda e: _keep([], e), replay=_replays,
    )

    assert result.status == "partial"
    assert "Stopped after 4 steps" in result.stopped_by
    assert result.trajectory, "everything it did before the limit is kept"


async def test_the_budget_is_counted_where_it_is_checked():
    provider, llm = a_session(*[turn_calling("browser_snapshot")] * 10)

    result = await run_agent_session(
        request(budget=Budget(steps=3, tokens=None, seconds=None, usd=None)),
        llm=llm, provider=provider, emit=lambda e: _keep([], e), replay=_replays,
    )

    assert result.spend["steps"] == 3
    assert result.status == "partial"


async def test_a_session_that_stops_at_its_budget_still_says_it_is_over():
    provider, llm = a_session(*[turn_calling("browser_snapshot")] * 10)
    events = []

    await run_agent_session(
        request(budget=Budget(steps=2, tokens=None, seconds=None, usd=None)),
        llm=llm, provider=provider, emit=lambda e: _keep(events, e), replay=_replays,
    )

    assert [e.type for e in events][-1] == "run_finished"
    assert len([e for e in events if e.type == "run_finished"]) == 1


# --- the interrupt ---------------------------------------------------------


async def test_an_irreversible_action_suspends_the_graph_and_waits():
    provider, llm = a_session(
        turn_calling("browser_snapshot"),
        turn_calling("browser_click", target="e3", element="Delete this account"),
    )
    events = []

    result = await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep(events, e),
        replay=_replays,
    )

    assert result.status == "awaiting_approval"
    assert result.awaiting is not None
    assert result.awaiting["call"]["name"] == "browser_click"

    asked = next(e for e in events if e.type == "approval_required")
    assert asked.name == "browser_click"
    assert asked.categories, "it says why it stopped"
    assert provider.calls[-1][0] == "browser_snapshot", "the click did not run"
    assert not any(e.type == "run_finished" for e in events), "it is not over"


async def test_resuming_with_an_approval_continues_from_where_it_stopped():
    provider, llm = a_session(
        turn_calling("browser_snapshot"),
        turn_calling("browser_click", target="e3", element="Delete this account"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="Deleted, as approved."),
    )
    events = []

    async with AgentSession(
        request(run_id="resumable"), llm=llm, provider=provider,
        emit=lambda e: _keep(events, e), replay=_replays,
    ) as session:
        paused = await session.start()
        assert paused.status == "awaiting_approval"
        assert provider.calls[-1][0] == "browser_snapshot", "the click did not run"
        assert not provider.closed, "the browser stays open while a person decides"

        done = await session.resume("approved")

    assert "browser_click" in [name for name, _ in provider.calls], "it ran after approval"
    assert done.status in {"succeeded", "partial"}
    assert any(e.type == "approval_resolved" for e in events)
    assert any(e.type == "run_finished" for e in events)
    assert provider.closed


async def test_answering_an_approval_does_not_ask_a_second_time():
    """The bug this whole rewrite exists to fix by construction rather than
    by hand: `HumanInTheLoopMiddleware`'s `after_model` recomputes its
    interrupt payload from already-checkpointed state on every resume, with
    no id it mints and no side effect -- so exactly one `approval_required`
    and one `approval_resolved` come out no matter how the resume happens."""
    provider, llm = a_session(
        turn_calling("browser_snapshot"),
        turn_calling("browser_click", target="e3", element="Delete this account"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="Deleted, as approved."),
    )
    events = []

    async with AgentSession(
        request(run_id="one-ask-only"), llm=llm, provider=provider,
        emit=lambda e: _keep(events, e), replay=_replays,
    ) as session:
        await session.start()
        await session.resume("approved")

    asked = [e for e in events if e.type == "approval_required"]
    resolved = [e for e in events if e.type == "approval_resolved"]
    finished = [e for e in events if e.type == "run_finished"]

    assert len(asked) == 1, "asked a second time on resume"
    assert len(resolved) == 1
    assert asked[0].approval_id == resolved[0].approval_id
    assert len(finished) == 1


def test_the_graph_module_imports_the_optional_extra_only_inside_a_function():
    """`langchain_core` is a base dependency (already required by `llm.py`),
    so importing `tool_adapter`'s `as_langchain_tools` at module scope is
    fine. `langchain` itself -- and anything that imports it, like
    `.middleware` -- is the optional extra and must stay deferred: importing
    this module at all must not require it installed, or `available()` could
    never safely be called to check."""
    import inspect

    from agent import graph as graph_module

    source = inspect.getsource(graph_module)
    top_level = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    assert not any(
        line == "from .middleware import" or "import langchain.agents" in line or "from .middleware" in line
        for line in top_level
    ), top_level
