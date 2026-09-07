"""Where the agent's browser/MCP connection comes from. See `base.py` for why
this shape exists and what every provider promises; `local_playwright.py`,
`stdio_mcp.py` and `inprocess.py` are the concrete providers -- one per way of
reaching a browser, never more than one file each.
"""

from __future__ import annotations

from .base import (
    Availability,
    BrowserProvider,
    MCPSession,
    REF_FORMAT,
    REF_IN_SNAPSHOT,
    ToolResult,
    ToolSpec,
    local_availability,
)
from .local_playwright import LocalPlaywrightMCP
from .stdio_mcp import StdioMCPProvider

__all__ = [
    "Availability",
    "BrowserProvider",
    "LocalPlaywrightMCP",
    "MCPSession",
    "REF_FORMAT",
    "REF_IN_SNAPSHOT",
    "StdioMCPProvider",
    "ToolResult",
    "ToolSpec",
    "local_availability",
]
