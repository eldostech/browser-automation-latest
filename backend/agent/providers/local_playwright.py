"""The default provider: `npx @playwright/mcp` over stdio, on this machine."""

from __future__ import annotations

import logging
from typing import Any

from .base import MCPSession, _close_stack, _npx_path, _open_stdio, local_availability

log = logging.getLogger(__name__)


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

    #: Bumping this is a deliberate act. `guardrails/catalog.py` is written
    #: against it.
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

        npx = _npx_path()
        assert npx is not None  # local_availability() just checked

        self._stack, self._session = await _open_stdio(npx, ["-y", *self.argv()], None)
        log.info(
            "playwright mcp started",
            extra={"version": self.version, "cdp": bool(self.cdp_endpoint)},
        )
        return self._session

    async def close(self) -> None:
        stack, self._stack, self._session = self._stack, None, None
        await _close_stack(stack, what="playwright")


__all__ = ["LocalPlaywrightMCP"]
