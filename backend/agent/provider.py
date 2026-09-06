"""Where the agent's browser comes from.

The agent drives a browser over Playwright MCP; the replay engine drives
Playwright directly. That looks like a contradiction and is not: they are
different jobs with different couplings.

`engine.py` wants the library. It needs auto-waiting, strict locators and
traces, and a JSON-RPC hop to a Node process would cost all three -- which is
why the MCP hop was removed from the replay path and must not come back.

The agent wants a *tool protocol*. It needs a schema per tool, a dispatcher, an
audit point, and -- when this runs on AgentCore -- a browser it does not own,
reached over CDP, with third-party tools arriving through Gateway as MCP. All
of that is MCP-shaped already.

**The consequence this file exists for: the agent must not assume it owns a
browser process.** Locally that is ``npx @playwright/mcp`` launching its own
Chromium. On AgentCore it is a managed session reached over a CDP endpoint, and
Playwright MCP can *attach* to one rather than launch one -- which is the seam
that makes the second implementation a different argv rather than a rewrite.
Get this interface right now and AgentCore is configuration; get it wrong and
it is a port.

Nothing here imports FastAPI or ``store``. The agent package is a library that
takes a request and yields results, so the HTTP surface -- ours today,
AgentCore's later -- is a thin adapter over it.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)

#: How Playwright MCP names an element in a snapshot: ``[ref=e12]``.
#:
#: Parsed out of *every* tool result rather than only out of ``browser_snapshot``
#: results, because an action's result carries an updated page section too, and
#: a ref that has just appeared is exactly the one the next call needs.
REF_IN_SNAPSHOT = re.compile(r"\[ref=(e\d+)\]")

#: What a valid ``target`` looks like. See ``tools.py`` -- the fact that the
#: server also accepts a raw CSS selector here is the thing the guard exists to
#: take back.
REF_FORMAT = re.compile(r"^e\d+$")


@dataclass(slots=True)
class ToolSpec:
    """One tool a server advertises, as the model will be shown it."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    """What a tool call produced.

    ``text`` is the server's own rendering, kept whole: Playwright MCP replies
    with the Playwright code it ran and the page as it now stands, and both are
    worth more than any summary of them. ``refs`` is what was parsed out of it.
    """

    text: str = ""
    is_error: bool = False
    refs: tuple[str, ...] = ()

    @classmethod
    def failed(cls, reason: str) -> "ToolResult":
        return cls(text=reason, is_error=True)


@runtime_checkable
class MCPSession(Protocol):
    """A live connection to one MCP server."""

    async def list_tools(self) -> list[ToolSpec]: ...

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


class BrowserProvider(Protocol):
    """Something that can hand out a browser-driving MCP session."""

    async def open(self) -> MCPSession: ...

    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Availability:
    """Whether this deployment can run an agent at all, and why not."""

    available: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"enabled": self.available, "reason": self.reason or None}


def local_availability() -> Availability:
    """Can this machine start a local Playwright MCP server?

    Reported rather than discovered at the moment somebody presses a button.
    ``RECORDER_ENABLED`` exists for exactly this reason -- a spawn failing with
    a message about a missing executable is not something a user can act on --
    and the agent gets the same treatment.
    """
    try:
        import mcp  # noqa: F401  - the client library, an optional extra
    except ImportError:
        return Availability(
            False,
            "The MCP client is not installed here. It is an optional extra: "
            "pip install -r backend/requirements-agent.txt",
        )
    if _npx_path() is None:
        return Availability(
            False,
            "Node is not on PATH, and the local browser provider starts "
            "@playwright/mcp with npx. Install Node 18 or newer.",
        )
    return Availability(True)


def _npx_path() -> str | None:
    """npx, under whichever name this platform gives it.

    On Windows the executable is ``npx.cmd``; asyncio's subprocess machinery
    will not find a bare ``npx`` there, and the failure it produces names a
    file rather than the missing dependency.
    """
    for candidate in ("npx.cmd", "npx") if os.name == "nt" else ("npx",):
        found = shutil.which(candidate)
        if found:
            return found
    return None


# ---------------------------------------------------------------------------
# The local provider
# ---------------------------------------------------------------------------


class LocalPlaywrightMCP:
    """`npx @playwright/mcp` over stdio, owning its own Chromium.

    ``--isolated`` keeps the profile in memory. A batch that signed in as one
    tenant must not leave a cookie behind for the next one, and a profile on
    disk is exactly how that happens.

    The version is pinned for the same reason ``playwright`` is pinned in
    ``requirements.txt``: this server's tool names and argument shapes are the
    contract the tool registry is written against, and ``@latest`` would let a
    release change them inside somebody's batch rather than in CI.
    """

    #: Bumping this is a deliberate act. `tools.py` is written against it.
    VERSION = "0.0.80"

    def __init__(
        self,
        *,
        headless: bool = True,
        version: str | None = None,
        cdp_endpoint: str = "",
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self.headless = headless
        self.version = version or self.VERSION
        #: When set, attach to a browser somebody else runs rather than launch
        #: one. This is the AgentCore path, and it is a flag rather than a
        #: subclass because it is genuinely the only difference.
        self.cdp_endpoint = cdp_endpoint
        self.extra_args = extra_args
        self._stack: Any = None
        self._session: Any = None

    def argv(self) -> list[str]:
        args = [f"@playwright/mcp@{self.version}", "--isolated"]
        if self.cdp_endpoint:
            args += ["--cdp-endpoint", self.cdp_endpoint]
        elif self.headless:
            args.append("--headless")
        args.extend(self.extra_args)
        return args

    async def open(self) -> MCPSession:
        availability = local_availability()
        if not availability.available:
            raise RuntimeError(availability.reason)

        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        npx = _npx_path()
        assert npx is not None  # local_availability() just checked

        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(
            stdio_client(StdioServerParameters(command=npx, args=["-y", *self.argv()]))
        )
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._session = _RealMCPSession(session)
        log.info(
            "playwright mcp started",
            extra={"version": self.version, "cdp": bool(self.cdp_endpoint)},
        )
        return self._session

    async def close(self) -> None:
        stack, self._stack, self._session = self._stack, None, None
        if stack is not None:
            # A server that has already died makes teardown raise, and a
            # failure to close a subprocess must not be what a caller sees
            # instead of whatever actually went wrong.
            try:
                await stack.aclose()
            except Exception as exc:  # noqa: BLE001
                log.warning("mcp server did not close cleanly", extra={"error": str(exc)})


class _RealMCPSession:
    """Adapts the MCP SDK's session to the two methods this package uses."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def list_tools(self) -> list[ToolSpec]:
        listing = await self._session.list_tools()
        return [
            ToolSpec(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.inputSchema or {}),
            )
            for tool in listing.tools
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        outcome = await self._session.call_tool(name, arguments)
        text = "\n".join(
            block.text for block in outcome.content if getattr(block, "text", None)
        )
        return ToolResult(
            text=text,
            is_error=bool(getattr(outcome, "isError", False)),
            refs=tuple(dict.fromkeys(REF_IN_SNAPSHOT.findall(text))),
        )


__all__ = [
    "Availability",
    "BrowserProvider",
    "LocalPlaywrightMCP",
    "MCPSession",
    "REF_FORMAT",
    "REF_IN_SNAPSHOT",
    "ToolResult",
    "ToolSpec",
    "local_availability",
]
