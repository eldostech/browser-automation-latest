"""Shared fixtures and fakes.

Nothing in the default test run touches the network, spawns a browser, or calls
an LLM. The end-to-end test in ``test_e2e_static.py`` is the single exception
and is opt-in via ``RUN_E2E=1``.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import pytest

# The backend is a flat module tree, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import AgentSpec, RunOptions  # noqa: E402
from events import AgentEvent  # noqa: E402
from llm import LLMTurn, ToolCallRequest  # noqa: E402
from mcp_client import MCPConfig, ToolOutcome  # noqa: E402
from store import Store  # noqa: E402


# ---------------------------------------------------------------------------
# Fake MCP session
# ---------------------------------------------------------------------------


@dataclass
class FakeTool:
    name: str
    description: str = "a fake browser tool"
    schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    #: Called with the tool arguments; returns a ToolOutcome or raises.
    handler: Callable[[dict[str, Any]], ToolOutcome] | None = None


class FakeMCPSession:
    """Implements the slice of :class:`mcp_client.MCPBrowserSession` the agent uses."""

    def __init__(self, tools: list[FakeTool] | None = None) -> None:
        self.config = MCPConfig(tool_timeout=5.0)
        self._tools = tools or [
            FakeTool("browser_navigate"),
            FakeTool("browser_snapshot"),
            FakeTool("browser_click"),
            FakeTool("browser_type"),
            FakeTool("browser_take_screenshot"),
        ]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    # -- discovery ------------------------------------------------------
    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self._tools]

    def anthropic_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.schema}
            for t in self._tools
        ]

    def find_tool(self, *candidates: str, contains: tuple[str, ...] = ()) -> str | None:
        names = self.tool_names
        for candidate in candidates:
            if candidate in names:
                return candidate
        for fragment in contains:
            for name in names:
                if fragment in name.lower():
                    return name
        return None

    # -- invocation -----------------------------------------------------
    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> ToolOutcome:
        self.calls.append((name, dict(arguments or {})))
        tool = next((t for t in self._tools if t.name == name), None)
        if tool is None:
            return ToolOutcome(name=name, text=f"unknown tool {name}", is_error=True)
        if tool.handler is not None:
            return tool.handler(dict(arguments or {}))
        if "screenshot" in name:
            return ToolOutcome(name=name, images=[("image/png", b"\x89PNG-fake")], duration_ms=3)
        return ToolOutcome(name=name, text=f"- Page URL: https://example.com\n- ok: {name}", duration_ms=5)


# ---------------------------------------------------------------------------
# Scripted LLM
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Replays a fixed list of turns; the last turn repeats if the loop runs on."""

    model = "scripted"

    def __init__(self, turns: list[LLMTurn], repeat_last: bool = True) -> None:
        self.turns = list(turns)
        self.repeat_last = repeat_last
        #: Message history as seen by the model on each turn.
        self.calls: list[list[dict[str, Any]]] = []
        #: Tool schema handed to the model on each turn.
        self.tool_schemas: list[list[dict[str, Any]]] = []

    async def run_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text_delta=None,
        timeout: float | None = None,
    ) -> LLMTurn:
        self.calls.append([dict(m) for m in messages])
        self.tool_schemas.append(list(tools))
        if self.turns:
            turn = self.turns.pop(0) if len(self.turns) > 1 or not self.repeat_last else self.turns[0]
        else:
            turn = LLMTurn(text="done", raw_content=[{"type": "text", "text": "done"}])
        if on_text_delta and turn.text:
            await on_text_delta(turn.text)
        return turn


def tool_turn(name: str, arguments: dict[str, Any], call_id: str = "call-1", text: str = "") -> LLMTurn:
    """An assistant turn that requests one tool call."""
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    content.append({"type": "tool_use", "id": call_id, "name": name, "input": arguments})
    return LLMTurn(
        text=text,
        tool_calls=[ToolCallRequest(id=call_id, name=name, input=arguments)],
        stop_reason="tool_use",
        raw_content=content,
    )


def final_turn(text: str) -> LLMTurn:
    return LLMTurn(text=text, stop_reason="end_turn", raw_content=[{"type": "text", "text": text}])


# ---------------------------------------------------------------------------
# Fake sink / approval gate
# ---------------------------------------------------------------------------


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self._seq = 0
        self.screenshots: list[bytes] = []

    def reserve_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def emit(self, event: AgentEvent) -> None:
        # Mirror the store's upsert semantics so streamed `thinking` blocks
        # collapse to one entry, exactly as a client would see after replay.
        for index, existing in enumerate(self.events):
            if existing.seq == event.seq:
                self.events[index] = event
                return
        self.events.append(event)

    async def save_screenshot(self, data: bytes, *, seq: int, mime: str = "image/png"):
        self.screenshots.append(data)
        return f"artifact-{seq}", f"/api/artifacts/artifact-{seq}"

    def of_type(self, event_type: str) -> list[AgentEvent]:
        return [e for e in self.events if e.type == event_type]


class AutoApprovalGate:
    """Answers every approval request with a fixed decision."""

    def __init__(self, decision: Literal["approved", "rejected", "timeout"] = "approved") -> None:
        self.decision = decision
        self.requests: list[str] = []
        self.paused = 0
        self.resumed = 0

    async def request(self, approval_id: str, timeout: float):
        self.requests.append(approval_id)
        return self.decision, None

    async def on_pause(self) -> None:
        self.paused += 1

    async def on_resume(self) -> None:
        self.resumed += 1


class NeverApprovalGate:
    """Blocks until cancelled -- used to test the approval timeout path."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    async def request(self, approval_id: str, timeout: float):
        self.requests.append(approval_id)
        try:
            await asyncio.wait_for(asyncio.Event().wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return "timeout", None
        return "approved", None

    async def on_pause(self) -> None:
        pass

    async def on_resume(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def options() -> RunOptions:
    return RunOptions(
        max_steps=6,
        timeout_seconds=30.0,
        allowed_domains=["example.com", "*.example.com"],
        require_approval=True,
        approval_timeout_seconds=2.0,
        screenshot_every_step=False,
    )


@pytest.fixture
def spec(options: RunOptions) -> AgentSpec:
    return AgentSpec(run_id="run-test", task="Find the pricing page", options=options)


@pytest.fixture
def sink() -> RecordingSink:
    return RecordingSink()


@pytest.fixture
def mcp() -> FakeMCPSession:
    return FakeMCPSession()


@pytest.fixture
async def store(tmp_path: Path):
    store = Store(tmp_path / "test.db", tmp_path / "artifacts")
    await store.connect()
    try:
        yield store
    finally:
        await store.close()
