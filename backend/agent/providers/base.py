"""The protocol every browser/MCP connection speaks, and the bootstrapping
every stdio-based one shares.

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

#: What a valid ``target`` looks like. See ``guardrails/guard.py`` -- the fact
#: that the server also accepts a raw CSS selector here is the thing the guard
#: exists to take back.
REF_FORMAT = re.compile(r"^e\d+$")


@dataclass(slots=True)
class ToolSpec:
    """One tool a server advertises, as the model will be shown it."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    #: The MCP protocol's own tool annotations (``readOnlyHint``,
    #: ``destructiveHint``, ...), where the server bothers to declare them.
    #: Playwright MCP does not, and that is fine -- browser tools are
    #: classified by name in ``guardrails/catalog.py`` regardless. This is
    #: what a tool from some *other* server is classified from instead, since
    #: this codebase has no hand-written knowledge of what that server's
    #: tools do.
    annotations: dict[str, bool] = field(default_factory=dict)


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
# Shared stdio bootstrapping -- every concrete provider is a command to run
# ---------------------------------------------------------------------------


async def _open_stdio(
    command: str, args: list[str], env: dict[str, str] | None
) -> tuple[Any, "_RealMCPSession"]:
    """Bring up one stdio MCP server. Shared because the only thing that
    differs between the browser and any other server is what gets spawned --
    the three calls that bring a client session up are the same either way.
    """
    from contextlib import AsyncExitStack

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    stack = AsyncExitStack()
    read, write = await stack.enter_async_context(
        stdio_client(StdioServerParameters(command=command, args=args, env=env))
    )
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return stack, _RealMCPSession(session)


async def _close_stack(stack: Any, *, what: str) -> None:
    if stack is None:
        return
    # A server that has already died makes teardown raise, and a failure to
    # close a subprocess must not be what a caller sees instead of whatever
    # actually went wrong.
    try:
        await stack.aclose()
    except Exception as exc:  # noqa: BLE001
        log.warning("mcp server did not close cleanly", extra={"server_name": what, "error": str(exc)})


def _annotations_of(tool: Any) -> dict[str, bool]:
    """The MCP protocol's optional tool annotations, off the SDK's object.

    Most servers do not declare these -- the spec calls them hints, not a
    guarantee -- so a missing key means "unknown", never "safe". See
    ``guardrails/catalog.py``'s classification for a tool that did not come
    from the browser, which is the thing that reads these.
    """
    raw = getattr(tool, "annotations", None)
    if raw is None:
        return {}
    out: dict[str, bool] = {}
    for key in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"):
        value = getattr(raw, key, None)
        if isinstance(value, bool):
            out[key] = value
    return out


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
                annotations=_annotations_of(tool),
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
    "MCPSession",
    "REF_FORMAT",
    "REF_IN_SNAPSHOT",
    "ToolResult",
    "ToolSpec",
    "local_availability",
]
