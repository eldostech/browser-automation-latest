"""MCP servers a workspace's agent may reach for, beside its browser.

Registering one here is the whole point of Phase 1's tool layer: an agent
session used to have exactly one tool source, hardcoded. This is what lets a
workspace add more without anyone touching `agent/tools/`. See
`agent/session.py`'s `AgentToolSession.extra` for how a session actually opens
these, and `agent/manager.py` for where an enabled row becomes a provider.

None of this reaches the replay engine. A registered server's tools are for
the agent's own use while it works -- see the migration's docstring for why a
tool from here can never become a step.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from agent.providers import StdioMCPProvider
from auth.rbac import Permission
from auth.service import Principal
from deps import WorkspaceData, require
from routers.schemas import ToolServerPreviewRequest, ToolServerRequest

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent-tool-servers", tags=["agent-tool-servers"])

#: A server that never finishes its MCP handshake must not hang the request
#: that is only trying to find out what it offers. Fixed rather than a
#: setting: this is a debugging aid for the person registering a server, not
#: a knob a deployment has ever needed to turn.
PREVIEW_TIMEOUT_SECONDS = 15.0


@router.post("", status_code=201)
async def create_tool_server(
    body: ToolServerRequest,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.AGENT_TOOLS_WRITE))],
) -> dict[str, Any]:
    server_id = uuid.uuid4().hex
    stored_id = await data.save_tool_server(
        server_id,
        body.name,
        body.transport,
        body.connection.model_dump(),
        enabled=body.enabled,
        owner_id=principal.user_id,
    )
    await data.audit(
        "agent_tool_server.save",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="agent_tool_server",
        resource_id=stored_id,
        detail={"name": body.name, "transport": body.transport, "enabled": body.enabled},
    )
    log.info(
        "registered an agent tool server",
        extra={"server_id": stored_id, "server_name": body.name},
    )
    return {
        "id": stored_id,
        "name": body.name,
        "transport": body.transport,
        "enabled": body.enabled,
    }


@router.get("")
async def list_tool_servers(
    data: WorkspaceData,
    _: Annotated[Principal, Depends(require(Permission.AGENT_TOOLS_READ))],
) -> dict[str, Any]:
    return {"servers": await data.list_tool_servers()}


@router.delete("/{server_id}")
async def delete_tool_server(
    server_id: str,
    data: WorkspaceData,
    principal: Annotated[Principal, Depends(require(Permission.AGENT_TOOLS_WRITE))],
) -> dict[str, Any]:
    if not await data.delete_tool_server(server_id):
        raise HTTPException(status_code=404, detail="No such tool server.")
    await data.audit(
        "agent_tool_server.delete",
        actor_id=principal.user_id,
        actor_email=principal.email,
        resource_type="agent_tool_server",
        resource_id=server_id,
    )
    return {"id": server_id, "deleted": True}


@router.post("/preview")
async def preview_tool_server(
    body: ToolServerPreviewRequest,
    _: Annotated[Principal, Depends(require(Permission.AGENT_TOOLS_WRITE))],
) -> dict[str, Any]:
    """What a server offers, without registering it.

    Opens the given connection just long enough to list its tools, then
    closes it. Nothing here is saved -- this is the answer to "is this worth
    turning on" that a person needs *before* deciding that.
    """
    provider = StdioMCPProvider(
        "preview",
        body.connection.command,
        tuple(body.connection.args),
        body.connection.env,
    )
    try:
        session = await asyncio.wait_for(provider.open(), timeout=PREVIEW_TIMEOUT_SECONDS)
        try:
            specs = await asyncio.wait_for(session.list_tools(), timeout=PREVIEW_TIMEOUT_SECONDS)
        finally:
            await provider.close()
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{body.connection.command} did not answer within "
            f"{PREVIEW_TIMEOUT_SECONDS:.0f}s.",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - an arbitrary command, arbitrary failure
        raise HTTPException(status_code=422, detail=f"Could not start this server: {exc}") from exc

    return {
        "tools": [
            {"name": spec.name, "description": spec.description, "annotations": spec.annotations}
            for spec in specs
        ]
    }


__all__ = ["router"]
