"""Any other MCP server a workspace registers, reached over stdio."""

from __future__ import annotations

import logging
from typing import Any

from .base import MCPSession, _close_stack, _open_stdio

log = logging.getLogger(__name__)


class StdioMCPProvider:
    """Any other MCP server, reached over stdio.

    Playwright's is special enough to keep its own class
    (``local_playwright.py``) -- a pinned version, ``--isolated``, a
    CDP-attach mode for AgentCore. Everything else just needs a command to
    run: the client-side bootstrapping is identical, which is what
    ``base.py``'s ``_open_stdio``/``_close_stack`` exist to share rather than
    duplicate.
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: tuple[str, ...] = (),
        env: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.args = args
        self.env = dict(env or {})
        self._stack: Any = None
        self._session: Any = None

    async def open(self) -> MCPSession:
        self._stack, self._session = await _open_stdio(
            self.command, list(self.args), self.env or None
        )
        log.info("mcp server started", extra={"server_name": self.name, "command": self.command})
        return self._session

    async def close(self) -> None:
        stack, self._stack, self._session = self._stack, None, None
        await _close_stack(stack, what=self.name)


__all__ = ["StdioMCPProvider"]
