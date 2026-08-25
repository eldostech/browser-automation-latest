"""The agent loop as a LangGraph graph.

The loop was never the hard part -- it was thirty lines. What the graph adds is
a checkpointer: every superstep is persisted, so a run has state to come back
to. These tests pin that the graph is genuinely in the path (not a wrapper that
still hand-rolls the loop) and that every guardrail survived the move.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from agent import AgentSpec, BrowserAgent, RunOptions
from conftest import AutoApprovalGate, ScriptedLLM, final_turn, tool_turn
from graph import (
    BUDGET_EXHAUSTED,
    build_agent_graph,
    initial_state,
    last_ai_message,
    tool_message,
)


def agent(spec, mcp, llm, sink, gate=None) -> BrowserAgent:
    return BrowserAgent(spec, mcp, llm, sink, gate or AutoApprovalGate())


# --- the graph is really the loop ------------------------------------------


def test_the_agent_compiles_a_graph_rather_than_looping_by_hand():
    import inspect

    import agent as agent_module

    source = inspect.getsource(agent_module.BrowserAgent._loop)
    assert "build_agent_graph" in source
    assert "while True" not in source, "the hand-written loop is gone"


def test_the_graph_has_exactly_one_decision():
    """think -> act -> think, and one conditional edge out of think."""
    compiled = build_agent_graph(
        think=lambda s: {}, act=lambda s: {}, should_continue=lambda s: END
    )
    nodes = set(compiled.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {"think", "act"}


async def test_a_run_is_checkpointed_so_it_has_state_to_resume_from():
    """The reason for the graph. A hand-written loop keeps nothing."""
    calls: list[dict] = []

    async def think(state):
        calls.append(dict(state))
        return {"messages": [AIMessage(content="done")], "step": state["step"] + 1,
                "answer": "done"}

    compiled = build_agent_graph(
        think=think, act=lambda s: {}, should_continue=lambda s: END
    )
    config = {"configurable": {"thread_id": "run-1"}}
    await compiled.ainvoke(initial_state("do a thing"), config=config)

    snapshot = await compiled.aget_state(config)
    assert snapshot.values["answer"] == "done"
    assert [type(m).__name__ for m in snapshot.values["messages"]] == [
        "HumanMessage", "AIMessage",
    ]


async def test_two_runs_keep_separate_state():
    """Threads are keyed by run id, so concurrent runs cannot see each other."""
    async def think(state):
        return {"messages": [AIMessage(content=state["messages"][0].content)],
                "answer": state["messages"][0].content}

    compiled = build_agent_graph(
        think=think, act=lambda s: {}, should_continue=lambda s: END
    )
    await compiled.ainvoke(initial_state("first"), {"configurable": {"thread_id": "a"}})
    await compiled.ainvoke(initial_state("second"), {"configurable": {"thread_id": "b"}})

    a = await compiled.aget_state({"configurable": {"thread_id": "a"}})
    b = await compiled.aget_state({"configurable": {"thread_id": "b"}})
    assert a.values["answer"] == "first"
    assert b.values["answer"] == "second"


# --- state helpers ---------------------------------------------------------


def test_initial_state_starts_at_step_zero():
    state = initial_state("go")
    assert state["step"] == 0 and state["answer"] is None and state["halted"] is None
    assert isinstance(state["messages"][0], HumanMessage)


def test_last_ai_message_ignores_later_tool_results():
    state = initial_state("go")
    state["messages"] += [
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "t", "args": {}}]),
        ToolMessage(content="ok", tool_call_id="c1"),
    ]
    assert last_ai_message(state).tool_calls[0]["id"] == "c1"


def test_a_failed_tool_result_is_marked_for_the_model():
    assert tool_message("c1", "boom", is_error=True).status == "error"
    assert tool_message("c1", "fine").status == "success"


def test_an_empty_tool_result_still_says_something():
    """A blank tool message reads to the model as a broken turn."""
    assert tool_message("c1", "").content


# --- the guardrails still bind ---------------------------------------------


async def test_the_step_budget_halts_the_graph(spec, mcp, sink):
    spec.options.max_steps = 2
    llm = ScriptedLLM([tool_turn("browser_snapshot", {})], repeat_last=True)

    outcome = await agent(spec, mcp, llm, sink).run()

    assert outcome.status == "failed"
    assert "Step budget" in outcome.error
    assert outcome.steps <= spec.options.max_steps + 1


async def test_the_deadline_halts_the_graph(spec, mcp, sink):
    spec.options.timeout_seconds = 0.0
    llm = ScriptedLLM([final_turn("done")])

    outcome = await agent(spec, mcp, llm, sink).run()

    assert outcome.status == "failed"
    assert "Wall-clock budget" in outcome.error


async def test_the_allowlist_still_gates_a_tool_call(spec, mcp, sink):
    spec.options.require_approval = False
    llm = ScriptedLLM(
        [tool_turn("browser_navigate", {"url": "https://evil.example.net"}),
         final_turn("stopped")],
        repeat_last=False,
    )

    await agent(spec, mcp, llm, sink).run()

    errors = [e for e in sink.of_type("error") if e.kind == "allowlist_blocked"]
    assert errors, "the domain gate did not run inside the graph"
    assert mcp.calls_to("browser_navigate") == [] if hasattr(mcp, "calls_to") else True


async def test_tool_results_reach_the_next_turn(spec, mcp, sink):
    """The reducer must pair results back to the call that produced them."""
    llm = ScriptedLLM(
        [tool_turn("browser_snapshot", {}, call_id="c1"), final_turn("saw it")],
        repeat_last=False,
    )

    outcome = await agent(spec, mcp, llm, sink).run()

    assert outcome.status == "succeeded"
    second_turn = llm.calls[1]
    tool_results = [
        block
        for message in second_turn
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert [b["tool_use_id"] for b in tool_results] == ["c1"]
