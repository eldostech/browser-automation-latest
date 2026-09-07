"""Registering, listing, deleting and previewing an agent's MCP servers.

The same shape as `credentials.py`'s tests: this is a workspace-scoped
resource behind RBAC and an audit trail, and the router is a thin adapter over
`WorkspaceStore` -- see `test_store.py` for the tenancy and CRUD guarantees
this router inherits rather than re-implements.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from credentials import generate_key

pytestmark = pytest.mark.anyio


@pytest.fixture
def client(db_settings, db_engine, tmp_path, monkeypatch):
    from conftest import authenticate, build_app
    from test_api_execute import FakeReplaySession

    app = build_app(
        db_settings, tmp_path, monkeypatch,
        session_cls=FakeReplaySession,
        credentials_key=generate_key(),
        agent_enabled=True,
    )
    with TestClient(app) as test_client:
        yield authenticate(test_client)


CONNECTION = {"command": "node", "args": ["server.js"], "env": {"API_KEY": "x"}}


def test_registering_a_server_returns_its_id_and_lists_it(client: TestClient):
    created = client.post(
        "/api/agent-tool-servers",
        json={"name": "crm", "connection": CONNECTION},
    ).json()
    assert created["name"] == "crm"
    assert created["enabled"] is True

    rows = client.get("/api/agent-tool-servers").json()["servers"]
    assert [r["name"] for r in rows] == ["crm"]
    assert rows[0]["connection"] == CONNECTION


def test_registering_the_same_name_twice_replaces_it(client: TestClient):
    client.post("/api/agent-tool-servers", json={"name": "crm", "connection": CONNECTION})
    client.post(
        "/api/agent-tool-servers",
        json={"name": "crm", "connection": {**CONNECTION, "command": "python"}, "enabled": False},
    )

    rows = client.get("/api/agent-tool-servers").json()["servers"]
    assert len(rows) == 1
    assert rows[0]["connection"]["command"] == "python"
    assert rows[0]["enabled"] is False


def test_deleting_a_server_removes_it(client: TestClient):
    created = client.post(
        "/api/agent-tool-servers", json={"name": "crm", "connection": CONNECTION}
    ).json()

    response = client.delete(f"/api/agent-tool-servers/{created['id']}")
    assert response.status_code == 200

    assert client.get("/api/agent-tool-servers").json()["servers"] == []


def test_deleting_an_unknown_server_is_a_404(client: TestClient):
    assert client.delete("/api/agent-tool-servers/does-not-exist").status_code == 404


async def test_registering_needs_write_access_not_just_authoring(client: TestClient):
    """Registering a server is choosing what a workspace's agent may connect
    to -- previewing one runs an arbitrary command. Both are gated ahead of
    the ordinary authoring authority, the same way SCRIPT_ENABLE is."""
    from conftest import TEST_ADMIN_PASSWORD, login_as, make_user

    await make_user(client.app, "author@test.invalid", "author")
    token = login_as(client, "author@test.invalid", TEST_ADMIN_PASSWORD)
    headers = {"Authorization": f"Bearer {token}"}

    response = client.post(
        "/api/agent-tool-servers",
        json={"name": "crm", "connection": CONNECTION},
        headers=headers,
    )
    assert response.status_code == 403

    read = client.get("/api/agent-tool-servers", headers=headers)
    assert read.status_code == 200, "authoring still sees what is registered"


async def test_previewing_a_server_lists_its_tools_without_saving_it(client: TestClient, monkeypatch):
    """The answer to "is this worth turning on" -- nothing here is persisted."""
    from agent.providers import ToolResult, ToolSpec
    import routers.agent_tool_servers as router_module

    class FakePreviewProvider:
        def __init__(self, name, command, args, env):
            self.command = command

        async def open(self):
            return self

        async def close(self):
            return None

        async def list_tools(self):
            return [ToolSpec("lookup_account", "Look up a CRM account", annotations={"readOnlyHint": True})]

    monkeypatch.setattr(router_module, "StdioMCPProvider", FakePreviewProvider)

    result = client.post(
        "/api/agent-tool-servers/preview",
        json={"connection": CONNECTION},
    ).json()

    assert result["tools"] == [
        {
            "name": "lookup_account",
            "description": "Look up a CRM account",
            "annotations": {"readOnlyHint": True},
        }
    ]
    assert client.get("/api/agent-tool-servers").json()["servers"] == [], (
        "a preview must never register anything"
    )


async def test_a_server_that_fails_to_open_is_a_clean_422_not_a_500(client: TestClient, monkeypatch):
    import routers.agent_tool_servers as router_module

    class FailingProvider:
        def __init__(self, name, command, args, env):
            pass

        async def open(self):
            raise RuntimeError("no such command")

        async def close(self):
            return None

    monkeypatch.setattr(router_module, "StdioMCPProvider", FailingProvider)

    response = client.post(
        "/api/agent-tool-servers/preview",
        json={"connection": CONNECTION},
    )
    assert response.status_code == 422
