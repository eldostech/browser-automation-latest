"""The agent loop, driven by a scripted LLM and a fake MCP session.

No browser, no network, no model calls.
"""

from __future__ import annotations

import asyncio

import pytest

from agent import AgentSpec, BrowserAgent
from conftest import (
    AutoApprovalGate,
    FakeMCPSession,
    FakeTool,
    NeverApprovalGate,
    RecordingSink,
    ScriptedLLM,
    final_turn,
    tool_turn,
)
from mcp_client import ToolOutcome


def build(spec, mcp, sink, turns, gate=None):
    return BrowserAgent(spec, mcp, ScriptedLLM(turns, repeat_last=False), sink, gate or AutoApprovalGate())


# --- happy path ------------------------------------------------------------


async def test_single_tool_call_then_answer(spec, mcp, sink):
    agent = build(
        spec, mcp, sink,
        [
            tool_turn("browser_snapshot", {}, text="Let me look at the page."),
            final_turn('Found it.\n```json\n{"url": "https://example.com/pricing"}\n```'),
        ],
    )

    outcome = await agent.run()

    assert outcome.status == "succeeded"
    assert outcome.result["data"] == {"url": "https://example.com/pricing"}
    assert "Found it." in outcome.result["answer"]
    assert ("browser_snapshot", {}) in mcp.calls

    types = [event.type for event in sink.events]
    assert types[0] == "run_started"
    assert "thinking" in types and "tool_call" in types and "tool_result" in types


async def test_events_carry_strictly_increasing_sequence_numbers(spec, mcp, sink):
    agent = build(spec, mcp, sink, [tool_turn("browser_snapshot", {}), final_turn("done")])
    await agent.run()
    seqs = [event.seq for event in sink.events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs)), "seq must be unique per run"


async def test_streaming_thinking_collapses_to_one_event(spec, mcp, sink):
    """A thinking block occupies exactly one seq no matter how many deltas."""
    agent = build(spec, mcp, sink, [final_turn("thinking out loud")])
    await agent.run()
    thinking = sink.of_type("thinking")
    assert len(thinking) == 1
    assert thinking[0].done is True
    assert thinking[0].text == "thinking out loud"


async def test_start_url_is_navigated_before_the_first_llm_turn(mcp, sink, options):
    spec = AgentSpec(
        run_id="r", task="check the page", start_url="https://example.com/start", options=options
    )
    agent = build(spec, mcp, sink, [final_turn("ok")])
    await agent.run()
    assert mcp.calls[0] == ("browser_navigate", {"url": "https://example.com/start"})


async def test_start_url_outside_the_allowlist_fails_the_run(mcp, sink, options):
    spec = AgentSpec(run_id="r", task="t", start_url="https://evil.net", options=options)
    agent = build(spec, mcp, sink, [final_turn("ok")])

    outcome = await agent.run()

    assert outcome.status == "failed"
    assert "evil.net" in outcome.error
    assert mcp.calls == [] or all(name != "browser_navigate" for name, _ in mcp.calls)


# --- guardrails ------------------------------------------------------------


async def test_step_budget_is_enforced(spec, mcp, sink):
    spec.options.max_steps = 3
    # Never finishes: always asks for another tool call, with varying args so
    # the loop detector does not fire first.
    turns = [tool_turn("browser_snapshot", {"n": i}, call_id=f"c{i}") for i in range(10)]
    agent = BrowserAgent(spec, mcp, ScriptedLLM(turns, repeat_last=False), sink, AutoApprovalGate())

    outcome = await agent.run()

    assert outcome.status == "failed"
    assert "Step budget" in outcome.error
    assert outcome.steps == 3


async def test_wall_clock_budget_is_enforced(spec, sink):
    spec.options.timeout_seconds = 0.25
    spec.options.max_steps = 100

    slow = FakeMCPSession([
        FakeTool("browser_snapshot", handler=None),
        FakeTool("browser_take_screenshot"),
    ])

    async def slow_call(name, arguments=None, *, timeout=None):
        await asyncio.sleep(0.1)
        return ToolOutcome(name=name, text="ok")

    slow.call_tool = slow_call  # type: ignore[assignment]

    turns = [tool_turn("browser_snapshot", {"n": i}, call_id=f"c{i}") for i in range(50)]
    agent = BrowserAgent(spec, slow, ScriptedLLM(turns, repeat_last=False), sink, AutoApprovalGate())

    outcome = await agent.run()

    assert outcome.status == "failed"
    assert "Wall-clock" in outcome.error


async def test_repeated_identical_actions_are_nudged_then_aborted(spec, mcp, sink):
    spec.options.max_steps = 20
    identical = [tool_turn("browser_click", {"ref": "e1"}, call_id="same") for _ in range(12)]
    agent = BrowserAgent(spec, mcp, ScriptedLLM(identical, repeat_last=False), sink, AutoApprovalGate())

    outcome = await agent.run()

    assert outcome.status == "failed"
    assert "identical" in outcome.error
    # The nudge is delivered as a tool error before the loop is aborted.
    nudged = [e for e in sink.of_type("tool_result") if not e.ok]
    assert nudged, "the agent should have been nudged before being aborted"


async def test_off_allowlist_tool_call_is_refused_when_approval_is_disabled(spec, mcp, sink):
    spec.options.require_approval = False
    agent = build(
        spec, mcp, sink,
        [tool_turn("browser_navigate", {"url": "https://evil.net"}), final_turn("blocked")],
    )

    outcome = await agent.run()

    assert outcome.status == "succeeded"  # the model was told, and stopped cleanly
    assert ("browser_navigate", {"url": "https://evil.net"}) not in mcp.calls
    errors = sink.of_type("error")
    assert any(e.kind == "allowlist_blocked" for e in errors)


# --- approvals -------------------------------------------------------------


async def test_sensitive_action_waits_for_approval_then_runs(spec, mcp, sink):
    gate = AutoApprovalGate("approved")
    agent = build(
        spec, mcp, sink,
        [tool_turn("browser_click", {"element": "Place order"}), final_turn("ordered")],
        gate=gate,
    )

    outcome = await agent.run()

    assert outcome.status == "succeeded"
    assert len(gate.requests) == 1
    assert gate.paused == 1 and gate.resumed == 1
    assert ("browser_click", {"element": "Place order"}) in mcp.calls

    required = sink.of_type("approval_required")
    assert len(required) == 1
    assert "payment" in required[0].categories
    assert sink.of_type("approval_resolved")[0].decision == "approved"


async def test_rejected_approval_blocks_the_tool_call(spec, mcp, sink):
    gate = AutoApprovalGate("rejected")
    agent = build(
        spec, mcp, sink,
        [tool_turn("browser_click", {"element": "Delete everything"}), final_turn("stopped")],
        gate=gate,
    )

    outcome = await agent.run()

    assert outcome.status == "succeeded"
    assert ("browser_click", {"element": "Delete everything"}) not in mcp.calls
    assert sink.of_type("approval_resolved")[0].decision == "rejected"


async def test_approval_timeout_is_treated_as_a_rejection(spec, mcp, sink):
    spec.options.approval_timeout_seconds = 0.1
    agent = build(
        spec, mcp, sink,
        [tool_turn("browser_click", {"element": "Submit payment"}), final_turn("gave up")],
        gate=NeverApprovalGate(),
    )

    outcome = await agent.run()

    assert outcome.status == "succeeded"
    assert sink.of_type("approval_resolved")[0].decision == "timeout"
    assert ("browser_click", {"element": "Submit payment"}) not in mcp.calls


async def test_no_approval_requested_when_the_gate_is_disabled(spec, mcp, sink):
    spec.options.require_approval = False
    gate = AutoApprovalGate()
    agent = build(
        spec, mcp, sink,
        [tool_turn("browser_click", {"element": "Submit form"}), final_turn("done")],
        gate=gate,
    )

    await agent.run()

    assert gate.requests == []
    assert ("browser_click", {"element": "Submit form"}) in mcp.calls


# --- failure handling ------------------------------------------------------


async def test_transient_tool_failure_is_retried(spec, sink, monkeypatch):
    import agent as agent_module
    from mcp_client import MCPToolError

    monkeypatch.setattr(agent_module, "RETRY_BACKOFF", (0.0, 0.0, 0.0))

    attempts = {"count": 0}

    def flaky(_args):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise MCPToolError("browser disconnected")
        return ToolOutcome(name="browser_click", text="clicked at last")

    mcp = FakeMCPSession([FakeTool("browser_click", handler=flaky), FakeTool("browser_take_screenshot")])
    agent = build(spec, mcp, sink, [tool_turn("browser_click", {"ref": "e1"}), final_turn("ok")])

    outcome = await agent.run()

    assert outcome.status == "succeeded"
    assert attempts["count"] == 3
    result = sink.of_type("tool_result")[0]
    assert result.ok is True and result.attempts == 3


async def test_dead_mcp_session_fails_the_run_without_hanging(spec, sink, monkeypatch):
    import agent as agent_module
    from mcp_client import MCPToolError

    monkeypatch.setattr(agent_module, "RETRY_BACKOFF", (0.0, 0.0, 0.0))

    def always_dead(_args):
        raise MCPToolError("connection closed")

    mcp = FakeMCPSession([
        FakeTool("browser_click", handler=always_dead),
        FakeTool("browser_take_screenshot", handler=always_dead),
    ])
    turns = [tool_turn("browser_click", {"n": i}, call_id=f"c{i}") for i in range(6)]
    agent = BrowserAgent(spec, mcp, ScriptedLLM(turns, repeat_last=False), sink, AutoApprovalGate())

    outcome = await asyncio.wait_for(agent.run(), timeout=5)

    assert outcome.status == "failed"
    assert "stopped responding" in outcome.error
    assert sink.of_type("run_started"), "the run must still have emitted its opening event"


async def test_tool_reported_error_is_fed_back_to_the_model(spec, sink):
    def failing(_args):
        return ToolOutcome(name="browser_click", text="element not found", is_error=True)

    mcp = FakeMCPSession([FakeTool("browser_click", handler=failing), FakeTool("browser_snapshot")])
    llm = ScriptedLLM(
        [tool_turn("browser_click", {"ref": "gone"}), final_turn("I could not find it.")],
        repeat_last=False,
    )
    agent = BrowserAgent(spec, mcp, llm, sink, AutoApprovalGate())

    outcome = await agent.run()

    assert outcome.status == "succeeded"
    result = sink.of_type("tool_result")[0]
    assert result.ok is False
    # The second LLM call must have seen the error as a tool_result.
    last_messages = llm.calls[-1]
    tool_results = [
        block
        for message in last_messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert any(block["is_error"] for block in tool_results)


async def test_long_tool_output_is_truncated_before_reaching_the_model(spec, sink):
    spec.options.max_tool_result_chars = 100

    def huge(_args):
        return ToolOutcome(name="browser_snapshot", text="x" * 5000)

    mcp = FakeMCPSession([FakeTool("browser_snapshot", handler=huge)])
    agent = build(spec, mcp, sink, [tool_turn("browser_snapshot", {}), final_turn("done")])

    await agent.run()

    result = sink.of_type("tool_result")[0]
    assert result.truncated is True
    assert len(result.text) < 400


# --- observation -----------------------------------------------------------


async def test_screenshots_are_emitted_but_never_sent_to_the_model(spec, sink):
    spec.options.screenshot_every_step = True
    mcp = FakeMCPSession()
    llm = ScriptedLLM([tool_turn("browser_snapshot", {}), final_turn("done")], repeat_last=False)
    agent = BrowserAgent(spec, mcp, llm, sink, AutoApprovalGate())

    await agent.run()

    assert sink.of_type("screenshot"), "a screenshot event should have been emitted"
    assert sink.screenshots, "the image bytes should have been persisted"

    # No image block may appear anywhere in the model's history.
    for messages in llm.calls:
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                assert all(
                    not (isinstance(block, dict) and block.get("type") == "image")
                    for block in content
                )


async def test_history_is_trimmed_in_valid_tool_pairs(spec, sink):
    spec.options.max_history_messages = 5
    spec.options.max_steps = 8
    mcp = FakeMCPSession()
    turns = [tool_turn("browser_snapshot", {"n": i}, call_id=f"c{i}") for i in range(8)]
    llm = ScriptedLLM(turns, repeat_last=False)
    agent = BrowserAgent(spec, mcp, llm, sink, AutoApprovalGate())

    await agent.run()

    final_messages = llm.calls[-1]
    assert len(final_messages) <= 6
    # The task prompt is never dropped.
    assert final_messages[0]["role"] == "user"
    assert isinstance(final_messages[0]["content"], str)
    # Every tool_use has a matching tool_result in the following message.
    for index, message in enumerate(final_messages):
        if message["role"] != "assistant" or not isinstance(message["content"], list):
            continue
        tool_use_ids = {
            block["id"] for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use"
        }
        if not tool_use_ids:
            continue
        following = final_messages[index + 1]["content"]
        result_ids = {
            block["tool_use_id"] for block in following
            if isinstance(block, dict) and block.get("type") == "tool_result"
        }
        assert tool_use_ids <= result_ids


async def test_cancellation_propagates(spec, sink):
    async def hang(name, arguments=None, *, timeout=None):
        await asyncio.sleep(30)
        return ToolOutcome(name=name)

    mcp = FakeMCPSession()
    mcp.call_tool = hang  # type: ignore[assignment]
    turns = [tool_turn("browser_snapshot", {})]
    agent = BrowserAgent(spec, mcp, ScriptedLLM(turns, repeat_last=False), sink, AutoApprovalGate())

    task = asyncio.create_task(agent.run())
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_final_answer_without_json_still_succeeds(spec, mcp, sink):
    agent = build(spec, mcp, sink, [final_turn("The pricing page is at /pricing.")])
    outcome = await agent.run()
    assert outcome.status == "succeeded"
    assert outcome.result["data"] is None
    assert outcome.result["answer"] == "The pricing page is at /pricing."


async def test_run_started_lists_the_discovered_tools(spec, mcp, sink):
    agent = build(spec, mcp, sink, [final_turn("done")])
    await agent.run()
    started = sink.of_type("run_started")[0]
    assert started.tools == mcp.tool_names
    assert "browser_navigate" in started.tools


async def test_tools_are_taken_from_the_server_not_hardcoded(spec, sink):
    """A server exposing unusual tool names must still work end to end."""
    mcp = FakeMCPSession([FakeTool("page_open"), FakeTool("page_capture_image")])
    llm = ScriptedLLM([tool_turn("page_open", {"url": "https://example.com"}), final_turn("ok")],
                      repeat_last=False)
    agent = BrowserAgent(spec, mcp, llm, sink, AutoApprovalGate())

    outcome = await agent.run()

    assert outcome.status == "succeeded"
    assert [t["name"] for t in llm.tool_schemas[0]] == ["page_open", "page_capture_image"]
