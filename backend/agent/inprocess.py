"""The agent's tools, over a browser the engine already owns.

Authoring and operating differ in one way that decides everything else: **who
owns the browser**.

When an agent records a workflow it owns the browser, so Playwright MCP starts
one and the agent drives it. When an agent *rescues* a replay it does not own
anything -- the engine is three steps into a row on a page it opened, and the
recovery has to happen on that page. Starting a second browser would be a
second, blank tab looking at nothing.

So this is an ``MCPSession`` over a live ``PlaywrightSession``: the same tool
names, the same ``eN`` addressing, the same replies. Everything above it --
the guard, the ref discipline, the redaction, the audit, the graph -- cannot
tell the difference and does not have to. That is what ``BrowserProvider``
was for, and this is the implementation that proves it was worth having.

Two details that are not incidental:

**Refs are ours, and they are honest about it.** Playwright's own
``aria_snapshot`` has no ``[ref=eN]`` markers -- the MCP server adds them for
its own addressing. They are injected here, numbered in document order, and
valid only for the snapshot that produced them. That is exactly the contract
the real server offers, so the staleness check in the guard means the same
thing on both.

**Acting goes through the engine's ladder.** A ref resolves to a role and an
accessible name, and from there through ``UseCaseExecutor.locate`` rather than
through a locator this module composes. The recovery agent therefore inherits
the ambiguity refusal a recorded step gets: if the thing it points at matches
three elements, it is told, rather than acting on whichever came first.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from snapshot import Snapshot, parse as parse_snapshot
from usecase import Locator

from .marks import describe_element
from .provider import ToolResult, ToolSpec

log = logging.getLogger(__name__)

#: A line of an aria snapshot that names an element: `- button "Save":` etc.
#: Structural lines (`- /url: ...`, `- text: ...`) carry no role and get no ref,
#: which matches what the MCP server does.
ROLE_LINE = re.compile(r"^(\s*)-\s+(?P<role>[A-Za-z][\w-]*)(?P<rest>[\s\"\[].*|:?)$")

#: Lines that look like a role but are the snapshot's own structure.
NOT_ROLES = frozenset({"text", "url", "img"})

#: What a recovery agent may do. A deliberately small subset of the authoring
#: surface: it is putting a page back where a recorded step expects it, not
#: exploring, and every tool here maps to something a step could have done.
TOOLS: dict[str, dict[str, Any]] = {
    "browser_snapshot": {
        "description": "The page as an accessibility tree, with a ref per element.",
        "properties": {},
        "required": [],
    },
    "browser_click": {
        "description": "Click the element named by `target`.",
        "properties": {
            "target": {"type": "string", "description": "A ref such as 'e12'."},
            "element": {"type": "string", "description": "What it is, for the record."},
        },
        "required": ["target"],
    },
    "browser_type": {
        "description": "Type text into the element named by `target`.",
        "properties": {
            "target": {"type": "string"},
            "text": {"type": "string"},
            "element": {"type": "string"},
        },
        "required": ["target", "text"],
    },
    "browser_select_option": {
        "description": "Choose an option in the dropdown named by `target`.",
        "properties": {
            "target": {"type": "string"},
            "values": {"type": "array", "items": {"type": "string"}},
            "element": {"type": "string"},
        },
        "required": ["target", "values"],
    },
    "browser_press_key": {
        "description": "Send a key to whatever has focus.",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
    "browser_navigate": {
        "description": "Go to a URL. Subject to the same allowlist as everything else.",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
    "browser_navigate_back": {
        "description": "Go back one page.",
        "properties": {},
        "required": [],
    },
    "browser_wait_for": {
        "description": "Wait for text to appear, or for a number of seconds.",
        "properties": {
            "text": {"type": "string"},
            "time": {"type": "number"},
        },
        "required": [],
    },
}


def with_refs(text: str) -> str:
    """Inject ``[ref=eN]`` into an aria snapshot, numbered in document order.

    The MCP server's snapshots carry these and Playwright's own do not, so
    everything downstream -- the parser, the guard's staleness check, the
    ladder builder -- expects them. Adding them here rather than reimplementing
    all of that is why one snapshot format serves both.
    """
    lines: list[str] = []
    index = 0
    for line in text.splitlines():
        match = ROLE_LINE.match(line)
        role = match.group("role") if match else ""
        if not role or role in NOT_ROLES or "[ref=" in line:
            lines.append(line)
            continue
        index += 1
        trailing = ":" if line.rstrip().endswith(":") else ""
        body = line.rstrip()[: -1] if trailing else line.rstrip()
        lines.append(f"{body} [ref=e{index}]{trailing}")
    return "\n".join(lines)


class EngineBrowser:
    """An MCP-shaped session over the browser a replay is already using.

    Implements the two methods :class:`agent.provider.MCPSession` needs, so it
    can be handed to ``AgentToolSession`` in place of a real server.
    """

    def __init__(self, executor: Any) -> None:
        #: The live ``UseCaseExecutor``. Held rather than just its browser
        #: because acting goes through its locator ladder, not around it.
        self.executor = executor
        self._snapshot: Snapshot | None = None

    # -- as a provider ------------------------------------------------------
    async def open(self) -> "EngineBrowser":
        return self

    async def close(self) -> None:
        """Nothing. The browser belongs to the replay and outlives this."""

    # -- as an MCP session --------------------------------------------------
    async def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=name,
                description=spec["description"],
                input_schema={
                    "type": "object",
                    "properties": spec["properties"],
                    "required": spec["required"],
                },
            )
            for name, spec in TOOLS.items()
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            return ToolResult.failed(f"{name} is not available during a replay.")
        try:
            return await handler(arguments)
        except Exception as exc:  # noqa: BLE001 - a tool error, not a crash
            log.info("recovery tool failed", extra={"tool": name, "error": str(exc)})
            return ToolResult.failed(f"{name} failed: {exc}")

    # -- perception ---------------------------------------------------------
    async def _browser_snapshot(self, _: dict[str, Any]) -> ToolResult:
        return await self._page_now()

    async def _page_now(self, prefix: str = "") -> ToolResult:
        """The page as it now stands, in the shape the real server replies in.

        Every action answers with this, exactly as Playwright MCP does, because
        the agent's next call needs refs for the page it is now on and asking
        for a snapshot after every action would double the round trips.
        """
        browser = self.executor.browser
        raw = ""
        try:
            raw = await browser.page.locator("body").aria_snapshot()
        except Exception:  # noqa: BLE001 - mid-navigation
            pass
        text = with_refs(raw)
        self._snapshot = parse_snapshot(text)
        try:
            self._snapshot.page_url = browser.url
        except Exception:  # noqa: BLE001
            pass
        body = (
            f"{prefix}### Page\n- Page URL: {browser.url}\n"
            f"### Snapshot\n```yaml\n{text}\n```"
        )
        return ToolResult(
            text=body, refs=tuple(node.ref for node in self._snapshot if node.ref)
        )

    # -- acting -------------------------------------------------------------
    async def _locator_for(self, ref: str) -> tuple[list[Locator], str]:
        """The durable ladder for a ref, or an empty one.

        Reuses ``describe_element``, so a recovery aims at an element the same
        way a recorded step does -- including deciding ``exact`` from what else
        is on the page, which is the difference between clicking "Invite" and
        clicking "+ Invite User".
        """
        if self._snapshot is None:
            return [], ""
        described = describe_element(self._snapshot, ref)
        return list(described.ladder), described.describe_first()

    async def _act_on(self, ref: str, what: str, perform: Any) -> ToolResult:
        ladder, described = await self._locator_for(ref)
        if not ladder:
            return ToolResult.failed(
                f"{ref} is not on the page as it now stands. Take a snapshot."
            )
        found = await self.executor.locate(ladder, step_id=f"recover:{ref}")
        if found is None:
            return ToolResult.failed(
                f"{described} did not resolve to exactly one element, so acting "
                "on it would be a guess. Point at something more specific."
            )
        locator, _rung, _describe = found
        await perform(locator)
        return await self._page_now(f"### Did\n{what} {described}\n\n")

    async def _browser_click(self, arguments: dict[str, Any]) -> ToolResult:
        return await self._act_on(
            str(arguments.get("target") or ""),
            "clicked",
            lambda found: found.click(timeout=self._timeout()),
        )

    async def _browser_type(self, arguments: dict[str, Any]) -> ToolResult:
        text = str(arguments.get("text") or "")
        return await self._act_on(
            str(arguments.get("target") or ""),
            "typed into",
            lambda found: found.fill(text, timeout=self._timeout()),
        )

    async def _browser_select_option(self, arguments: dict[str, Any]) -> ToolResult:
        values = arguments.get("values") or []
        choice = str(values[0]) if values else ""
        return await self._act_on(
            str(arguments.get("target") or ""),
            "chose in",
            lambda found: found.select_option(choice, timeout=self._timeout()),
        )

    async def _browser_press_key(self, arguments: dict[str, Any]) -> ToolResult:
        await self.executor.browser.page.keyboard.press(str(arguments.get("key") or ""))
        return await self._page_now("### Did\npressed a key\n\n")

    async def _browser_navigate(self, arguments: dict[str, Any]) -> ToolResult:
        # The allowlist is enforced by the guard above, on every call, before
        # this is reached. Nothing is re-checked here on purpose: two places
        # that decide the same thing drift.
        await self.executor.browser.page.goto(str(arguments.get("url") or ""))
        return await self._page_now("### Did\nnavigated\n\n")

    async def _browser_navigate_back(self, _: dict[str, Any]) -> ToolResult:
        await self.executor.browser.page.go_back()
        return await self._page_now("### Did\nwent back\n\n")

    async def _browser_wait_for(self, arguments: dict[str, Any]) -> ToolResult:
        text = str(arguments.get("text") or "")
        if text:
            await self.executor.browser.page.get_by_text(text).first.wait_for(
                timeout=self._timeout()
            )
        else:
            seconds = min(float(arguments.get("time") or 1.0), 30.0)
            await self.executor.browser.page.wait_for_timeout(seconds * 1000)
        return await self._page_now("### Did\nwaited\n\n")

    def _timeout(self) -> int:
        return int(getattr(self.executor, "step_timeout", 30.0) * 1000)


__all__ = ["EngineBrowser", "TOOLS", "with_refs"]
