"""MCP client layer: the only place in this codebase that talks to a browser.

Design notes
------------
**All browser control goes through MCP.** There is deliberately no raw
Playwright-Python anywhere in the backend. The agent and a human debugging a
run therefore see exactly the same tool surface, and a tool that works in the
dashboard works identically when driven by the model.

**Transport: stdio by default, HTTP optional.** stdio makes the browser a
child process of the backend, so its lifetime is bounded by ours and a crashed
backend cannot leak a browser -- the right default for local development and
single-node deploys. HTTP/SSE is supported for the case where the browser must
live elsewhere (a dedicated container, a machine with a display, a shared
session pool); there, lifetime management becomes the operator's problem.

**Tools are discovered, never hardcoded.** ``list_tools()`` is called on
connect and the LLM tool schema is generated from that response, so a new
Playwright MCP release that adds or renames a tool needs no code change here.
Helper lookups like :meth:`find_tool` degrade to ``None`` rather than raising.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from config import Settings

log = logging.getLogger(__name__)


class MCPConnectionError(RuntimeError):
    """The MCP server could not be reached, spawned, or handshaken with."""


class MCPToolError(RuntimeError):
    """A tool call failed at the transport level (not a tool-reported error)."""


@dataclass(slots=True)
class DiscoveredTool:
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_anthropic(self) -> dict[str, Any]:
        schema = self.input_schema or {"type": "object", "properties": {}}
        # Anthropic requires a top-level object schema.
        if schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}
        return {
            "name": self.name,
            "description": (self.description or self.name)[:1024],
            "input_schema": schema,
        }


@dataclass(slots=True)
class ToolOutcome:
    """Normalised result of one MCP tool call."""

    name: str
    text: str = ""
    #: ``(mime_type, raw_bytes)`` for every image the tool returned.
    images: list[tuple[str, bytes]] = field(default_factory=list)
    is_error: bool = False
    structured: dict[str, Any] | None = None
    duration_ms: int = 0


@dataclass(slots=True)
class MCPConfig:
    """Per-run MCP settings, derived from :class:`config.Settings` plus run options."""

    transport: str = "stdio"
    npx_package: str = "@playwright/mcp@latest"
    npx_path: str = "npx"
    browser: str = "chromium"
    headless: bool = True
    isolated: bool = True
    storage_state: str | None = None
    extra_args: list[str] = field(default_factory=list)
    server_url: str = "http://localhost:8931/sse"
    handshake_timeout: float = 45.0
    tool_timeout: float = 60.0
    output_dir: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> "MCPConfig":
        config = cls(
            transport=settings.mcp_transport,
            npx_package=settings.mcp_npx_package,
            npx_path=settings.resolve_npx(),
            browser=settings.mcp_browser,
            headless=settings.mcp_headless,
            isolated=settings.mcp_isolated,
            storage_state=settings.mcp_storage_state,
            extra_args=list(settings.extra_mcp_args),
            server_url=settings.mcp_server_url,
            handshake_timeout=settings.mcp_handshake_timeout,
            tool_timeout=settings.mcp_tool_timeout,
        )
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        return config

    def command_line(self) -> list[str]:
        """The argv the stdio transport will spawn (also shown in /healthz)."""
        args = ["-y", self.npx_package]
        if self.headless:
            args.append("--headless")
        if self.isolated:
            args.append("--isolated")
        if self.browser:
            args += ["--browser", self.browser]
        if self.storage_state:
            args += ["--storage-state", self.storage_state]
        if self.output_dir:
            args += ["--output-dir", self.output_dir]
        args += self.extra_args
        return [self.npx_path, *args]


class MCPBrowserSession:
    """One MCP session == one browser session == one run.

    Use as an async context manager; teardown is guaranteed on success, error
    and cancellation alike::

        async with MCPBrowserSession(config) as mcp:
            await mcp.call_tool("browser_navigate", {"url": "https://example.com"})
    """

    def __init__(self, config: MCPConfig) -> None:
        self.config = config
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._tools: list[DiscoveredTool] = []
        self._stderr_file: Any = None
        self._stderr_path: Path | None = None
        self._closed = False

    # -- lifecycle ----------------------------------------------------------
    async def __aenter__(self) -> "MCPBrowserSession":
        try:
            await self.connect()
        except BaseException:
            await self.aclose()
            raise
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def connect(self) -> None:
        try:
            if self.config.transport == "stdio":
                read, write = await self._open_stdio()
            else:
                read, write = await self._open_http()

            session = await self._stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), timeout=self.config.handshake_timeout)
            self._session = session
        except asyncio.TimeoutError as exc:
            raise MCPConnectionError(
                "Timed out waiting for the MCP initialize handshake after "
                f"{self.config.handshake_timeout}s. {self._stderr_tail()}"
            ) from exc
        except MCPConnectionError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            raise MCPConnectionError(
                f"Could not connect to the Playwright MCP server: {exc}. {self._stderr_tail()}"
            ) from exc

        await self.refresh_tools()

    async def _open_stdio(self) -> tuple[Any, Any]:
        argv = self.config.command_line()
        log.info("spawning MCP server", extra={"argv": argv})

        # Keep the server's stderr so a failed handshake can explain itself
        # (missing browser binaries, bad flag, npm download failure).
        self._stderr_file = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".log", prefix="mcp-stderr-", delete=False, encoding="utf-8"
        )
        self._stderr_path = Path(self._stderr_file.name)

        params = StdioServerParameters(
            command=argv[0],
            args=argv[1:],
            env={**os.environ},
        )
        return await self._stack.enter_async_context(
            stdio_client(params, errlog=self._stderr_file)
        )

    async def _open_http(self) -> tuple[Any, Any]:
        url = self.config.server_url
        log.info("connecting to MCP server over HTTP", extra={"url": url})
        if url.rstrip("/").endswith("/sse"):
            read, write = await self._stack.enter_async_context(
                sse_client(url, timeout=self.config.handshake_timeout)
            )
            return read, write
        # Streamable HTTP yields a third element (a session-id getter) we don't need.
        read, write, _ = await self._stack.enter_async_context(streamablehttp_client(url))
        return read, write

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._stack.aclose()
        except Exception as exc:  # noqa: BLE001 - teardown must not mask the real error
            log.warning("error while closing MCP session", extra={"error": str(exc)})
        finally:
            self._session = None
            if self._stderr_file is not None:
                try:
                    self._stderr_file.close()
                except Exception:  # noqa: BLE001
                    pass
            if self._stderr_path and self._stderr_path.exists():
                try:
                    self._stderr_path.unlink()
                except OSError:
                    pass

    def _stderr_tail(self, lines: int = 15) -> str:
        if not self._stderr_path or not self._stderr_path.exists():
            return ""
        try:
            content = self._stderr_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""
        if not content:
            return ""
        tail = "\n".join(content.splitlines()[-lines:])
        return f"Server stderr:\n{tail}"

    # -- discovery ----------------------------------------------------------
    async def refresh_tools(self) -> list[DiscoveredTool]:
        result = await self.session.list_tools()
        self._tools = [
            DiscoveredTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.inputSchema or {}),
            )
            for tool in result.tools
        ]
        log.info(
            "discovered MCP tools",
            extra={"tool_count": len(self._tools), "tools": [t.name for t in self._tools]},
        )
        return self._tools

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise MCPConnectionError("MCP session is not connected")
        return self._session

    @property
    def tools(self) -> list[DiscoveredTool]:
        return list(self._tools)

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self._tools]

    def anthropic_tools(self) -> list[dict[str, Any]]:
        """The tool schema handed to the LLM, built from the live server response."""
        return [tool.to_anthropic() for tool in self._tools]

    def find_tool(self, *candidates: str, contains: Iterable[str] = ()) -> str | None:
        """Best-effort lookup used for the few tools the runner calls itself.

        Tries exact names first, then substring matches. Returns ``None`` when
        the server does not expose anything suitable, so callers can degrade.
        """
        names = self.tool_names
        for candidate in candidates:
            if candidate in names:
                return candidate
        for fragment in contains:
            for name in names:
                if fragment in name.lower():
                    return name
        return None

    # -- invocation ---------------------------------------------------------
    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> ToolOutcome:
        """Call one MCP tool and normalise the response.

        A tool that reports failure (``isError``) is *not* an exception: that
        is ordinary feedback the model should see and react to. Only transport
        failures raise.
        """
        loop = asyncio.get_running_loop()
        started = loop.time()
        effective_timeout = timeout or self.config.tool_timeout
        try:
            result = await asyncio.wait_for(
                self.session.call_tool(name, arguments or {}), timeout=effective_timeout
            )
        except asyncio.TimeoutError as exc:
            raise MCPToolError(
                f"Tool {name!r} did not return within {effective_timeout}s"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - transport-level failure
            raise MCPToolError(f"Tool {name!r} failed at the transport level: {exc}") from exc

        duration_ms = int((loop.time() - started) * 1000)
        return self._normalise(name, result, duration_ms)

    @staticmethod
    def _normalise(name: str, result: Any, duration_ms: int) -> ToolOutcome:
        texts: list[str] = []
        images: list[tuple[str, bytes]] = []

        for block in getattr(result, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                texts.append(getattr(block, "text", "") or "")
            elif block_type == "image":
                raw = getattr(block, "data", "") or ""
                try:
                    images.append((getattr(block, "mimeType", "image/png"), base64.b64decode(raw)))
                except (ValueError, TypeError):
                    log.warning("could not decode image block", extra={"tool": name})
            elif block_type == "resource":
                resource = getattr(block, "resource", None)
                text = getattr(resource, "text", None)
                if text:
                    texts.append(text)

        structured = getattr(result, "structuredContent", None)
        return ToolOutcome(
            name=name,
            text="\n".join(t for t in texts if t).strip(),
            images=images,
            is_error=bool(getattr(result, "isError", False)),
            structured=structured if isinstance(structured, dict) else None,
            duration_ms=duration_ms,
        )


async def probe(config: MCPConfig, timeout: float = 20.0) -> dict[str, Any]:
    """Connect, list tools, disconnect. Used by ``/healthz``.

    Never raises -- the health endpoint reports the failure instead.
    """
    try:
        async with asyncio.timeout(timeout):
            async with MCPBrowserSession(config) as session:
                return {
                    "ok": True,
                    "transport": config.transport,
                    "tool_count": len(session.tools),
                    "tools": session.tool_names,
                }
    except asyncio.TimeoutError:
        return {
            "ok": False,
            "transport": config.transport,
            "error": f"MCP probe timed out after {timeout}s",
        }
    except Exception as exc:  # noqa: BLE001 - health probe must never raise
        return {"ok": False, "transport": config.transport, "error": str(exc)}


def summarise_tools(tools: Sequence[DiscoveredTool], limit: int = 400) -> str:
    """One-line-per-tool summary, used in startup logs."""
    lines = [f"  - {tool.name}: {(tool.description or '').splitlines()[0][:limit]}" for tool in tools]
    return "\n".join(lines)
