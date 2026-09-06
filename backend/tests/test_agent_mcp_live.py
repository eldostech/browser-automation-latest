"""The tool layer against a real `npx @playwright/mcp`, with nothing faked.

Every other agent test drives a fake that answers the way the server does. That
is the right default -- the guard should be testable without Node -- but a fake
is a claim about somebody else's software, and this file is what stops the
claim rotting silently.

It is what caught the mistake the design was built on. `browser_click`'s
`target` is documented as "Exact target element reference from the page
snapshot, **or a unique element selector**", and a raw `#o` really does click
the element. "The model cannot invent a locator" is not a property of the
protocol; it is a property of our guard.

Opt in with `RUN_MCP=1`. It needs Node on PATH and downloads the server on
first run, which is slower than the rest of the suite put together.
"""

from __future__ import annotations

import os

import pytest

from agent import AgentToolSession, LocalPlaywrightMCP
from agent.tools import DISTILS_TO, PERCEPTION, REFUSED

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.mcp,
    pytest.mark.skipif(
        os.environ.get("RUN_MCP") != "1",
        reason="set RUN_MCP=1 to run against a real Playwright MCP server",
    ),
]

#: A page as a data URL, so this needs no server and cannot be flaky because of
#: one. The shape is the one that broke a real run: two controls whose
#: accessible names contain each other.
PAGE = (
    "data:text/html,<h1>Users</h1>"
    "<button id=o onclick=\"document.title='opened'\">+ Invite User</button>"
    "<input type=email placeholder=Email>"
)


@pytest.fixture
async def tools():
    provider = LocalPlaywrightMCP(headless=True)
    # `data:` is not a host, so the allowlist has to permit it explicitly for
    # these to run at all. That is the deny-by-default gate working.
    session = AgentToolSession(
        provider, allowed_domains=("data",), may_write=True
    )
    async with session:
        yield session


async def test_the_server_advertises_the_tools_the_registry_is_written_against():
    """A rename upstream should fail here, loudly, rather than in a session."""
    provider = LocalPlaywrightMCP(headless=True)
    session = await provider.open()
    try:
        advertised = {spec.name for spec in await session.list_tools()}
    finally:
        await provider.close()

    missing = (set(DISTILS_TO) | PERCEPTION) - advertised
    assert not missing, f"the registry names tools this server does not have: {missing}"

    still_there = set(REFUSED) & advertised
    assert "browser_evaluate" in still_there, (
        "browser_evaluate is refused because it exists. If it has gone, the "
        "refusal is dead code and should say so."
    )


async def test_a_ref_makes_the_server_report_a_durable_locator(tools):
    """The reason forcing refs is worth more than safety alone.

    Given a ref the server replies with the Playwright code it ran, which is a
    role-and-name locator handed over for free -- and it is what distillation
    will read. Given a selector it echoes the selector back, which is worth
    nothing later.
    """
    await tools.call("browser_navigate", {"url": PAGE})
    snapshot = await tools.call("browser_snapshot", {})
    assert tools.known_refs, snapshot.text

    ref = next(r for r in sorted(tools.known_refs, key=lambda x: int(x[1:])))
    result = await tools.call("browser_click", {"target": ref, "element": "a control"})

    assert not result.is_error, result.text
    assert "Ran Playwright code" in result.text


async def test_the_selector_the_protocol_allows_is_the_one_we_refuse(tools):
    """The finding this file exists for, asserted from both sides."""
    await tools.call("browser_navigate", {"url": PAGE})
    await tools.call("browser_snapshot", {})

    refused = await tools.call("browser_click", {"target": "#o", "element": "button"})
    assert refused.is_error
    assert "not an element reference" in refused.text

    # And the server really would have accepted it, which is why the guard is
    # not ceremony: the same call, made underneath the guard, works.
    raw = await tools.provider._session.call(  # noqa: SLF001 - deliberately beneath it
        "browser_click", {"target": "#o", "element": "button"}
    )
    assert not raw.is_error, raw.text


async def test_leaving_the_allowlist_never_reaches_the_browser(tools):
    result = await tools.call("browser_navigate", {"url": "https://example.com/"})

    assert result.is_error
    assert "example.com" in result.text


#: The page that broke a real run: two controls whose accessible names contain
#: each other, one of them inside a dialog covering the other.
INVITE_PAGE = (
    "data:text/html,<h1>Users</h1>"
    "<button>+ Invite User</button>"
    "<div role=dialog aria-label=Invite>"
    "<input type=email placeholder=Email><button>Invite</button></div>"
)


async def test_describe_element_reads_a_real_snapshot(tools):
    """The bridge, against the format the server actually emits.

    `snapshot.py` was written for Playwright MCP's tree and this is what stops
    that claim rotting: a change to the rendering breaks here rather than
    silently producing use cases whose locators describe nothing.
    """
    await tools.call("browser_navigate", {"url": INVITE_PAGE})
    await tools.call("browser_snapshot", {})

    assert tools.snapshot is not None, "the reply did not parse as a snapshot"
    named = {node.name: node.ref for node in tools.snapshot if node.name}
    assert "+ Invite User" in named
    assert "Invite" in named

    described = await tools.call("describe_element", {"ref": named["Invite"]})
    assert not described.is_error, described.text
    assert "exact" in described.text, (
        "'Invite' is a substring of '+ Invite User', so the recorded locator "
        "has to require the whole accessible name"
    )
