"""Executing a stored use case over HTTP, with a real browser session faked.

The headline test is ``test_executing_a_use_case_makes_no_llm_call``: the run
manager's LLM is replaced with an object that raises on any attribute access,
so any accidental model use anywhere in the execute path fails the test loudly.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from conftest import FakeTool
from credentials import generate_key
from mcp_client import ToolOutcome
from test_api import _fake_probe  # noqa: F401
from test_replay import SIGNED_IN, SIGNED_OUT, ScriptedMCP


class ExplodingLLM:
    """Any use at all is a test failure."""

    def __getattr__(self, name: str):
        raise AssertionError(f"the replay path touched the LLM client ({name!r})")


class FakeReplaySession:
    """Async-context wrapper around the scripted MCP session."""

    #: Set by each test so assertions can inspect the calls afterwards.
    current: ScriptedMCP | None = None

    def __init__(self, config: Any) -> None:
        self.config = config

    async def __aenter__(self) -> ScriptedMCP:
        FakeReplaySession.current = ScriptedMCP(
            [SIGNED_OUT, SIGNED_IN],
            routes={"/signin": SIGNED_OUT},
            advance_on={"browser_click"},
        )
        return FakeReplaySession.current

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False


USE_CASE = {
    "name": "Sign in and read the score",
    "status": "ready",
    "allowed_domains": ["example.com"],
    "inputs": [{"name": "record_url", "type": "url", "required": True}],
    "secrets": [{"name": "username"}, {"name": "password"}],
    "setup_steps": [
        {"id": "u1", "action": "navigate", "url": "https://example.com/signin"},
        {
            "id": "u2",
            "action": "fill_form",
            "fields": [
                {"name": "Username", "value": "{{secret.username}}",
                 "locators": [{"strategy": "role", "role": "textbox", "name": "Username"}]},
                {"name": "Password", "value": "{{secret.password}}",
                 "locators": [{"strategy": "role", "role": "textbox", "name": "Password"}]},
            ],
        },
        {"id": "u3", "action": "click",
         "locators": [{"strategy": "role", "role": "button", "name": "Sign in"}]},
    ],
    "session_check": {"kind": "url_contains", "value": "/signin", "negate": True},
    "row_reset": {"id": "reset", "action": "navigate", "url": "{{input.record_url}}"},
    "row_steps": [
        {"id": "s1", "action": "assert",
         "assert": {"kind": "url_contains", "value": "/signin", "negate": True}},
        {"id": "s2", "action": "extract", "output": "score",
         "locators": [{"strategy": "role", "role": "status", "name": "Score"}]},
    ],
    "outputs": ["score"],
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    import main
    import runner as runner_module

    monkeypatch.setattr(main.settings, "database_path", str(tmp_path / "api.db"))
    monkeypatch.setattr(main.settings, "artifacts_dir", str(tmp_path / "artifacts"))
    monkeypatch.setattr(main.settings, "llm_provider", "anthropic")
    monkeypatch.setattr(main.settings, "anthropic_api_key", "unused")
    monkeypatch.setattr(main.settings, "credentials_key", generate_key())
    monkeypatch.setattr(main, "probe", _fake_probe)
    monkeypatch.setattr(runner_module, "MCPBrowserSession", FakeReplaySession)

    with TestClient(main.app) as test_client:
        test_client.app.state.manager._llm = ExplodingLLM()  # noqa: SLF001 - test seam
        yield test_client


async def seed(client: TestClient, definition: dict | None = None) -> str:
    definition = {**USE_CASE, "id": "uc-1", **(definition or {})}
    usecase_id, _ = await client.app.state.store.save_usecase(definition)
    return usecase_id


# --- the headline guarantee ------------------------------------------------


async def test_executing_a_use_case_makes_no_llm_call(client: TestClient):
    usecase_id = await seed(client)
    credential = client.post(
        "/api/credentials",
        json={"name": "Example account", "values": {"username": "u", "password": "pw"}},
    ).json()

    response = client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={"inputs": {"record_url": "https://example.com/record/1"},
              "credential_id": credential["id"]},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "succeeded", body.get("error")
    assert body["llm_calls"] == 0
    assert body["llm_tokens"] == 0
    assert body["outputs"] == {"score": "92%"}


async def test_the_run_record_reports_zero_tokens(client: TestClient):
    usecase_id = await seed(client)
    credential = client.post(
        "/api/credentials",
        json={"name": "Example account", "values": {"username": "u", "password": "pw"}},
    ).json()
    body = client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={"inputs": {"record_url": "https://example.com/record/1"},
              "credential_id": credential["id"]},
    ).json()

    run = client.get(f"/api/runs/{body['run_id']}").json()
    assert run["status"] == "succeeded"
    assert run["result"]["llm_tokens"] == 0


async def test_the_timeline_renders_with_no_frontend_changes(client: TestClient):
    """A replay emits the same events the dashboard already understands."""
    usecase_id = await seed(client)
    credential = client.post(
        "/api/credentials",
        json={"name": "Example account", "values": {"username": "u", "password": "pw"}},
    ).json()
    body = client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={"inputs": {"record_url": "https://example.com/record/1"},
              "credential_id": credential["id"]},
    ).json()

    events = client.get(f"/api/runs/{body['run_id']}/events").json()["events"]
    kinds = {e["type"] for e in events}
    assert {"run_started", "step_started", "step_finished", "tool_call", "tool_result",
            "run_finished"} <= kinds


# --- credentials over HTTP -------------------------------------------------


async def test_credentials_are_write_only(client: TestClient):
    created = client.post(
        "/api/credentials", json={"name": "Example", "values": {"password": "s3cret"}}
    )
    assert created.status_code == 201

    listing = client.get("/api/credentials").json()
    assert listing["credentials"][0]["slots"] == ["password"]
    assert "s3cret" not in listing["credentials"][0].get("name", "")
    assert "s3cret" not in str(listing), "no endpoint returns a stored value"


async def test_secrets_never_reach_the_event_log(client: TestClient):
    usecase_id = await seed(client)
    credential = client.post(
        "/api/credentials",
        json={"name": "Example", "values": {"username": "u", "password": "s3cret-Example-Pw!"}},
    ).json()
    body = client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={"inputs": {"record_url": "https://example.com/record/1"},
              "credential_id": credential["id"]},
    ).json()

    events = client.get(f"/api/runs/{body['run_id']}/events").text
    assert "s3cret-Example-Pw!" not in events
    assert "«redacted»" in events, "the placeholder proves the pass ran"


async def test_the_real_secret_still_reaches_the_page(client: TestClient):
    usecase_id = await seed(client)
    credential = client.post(
        "/api/credentials",
        json={"name": "Example", "values": {"username": "u", "password": "s3cret-Example-Pw!"}},
    ).json()
    client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={"inputs": {"record_url": "https://example.com/record/1"},
              "credential_id": credential["id"]},
    )

    filled = [args for name, args in FakeReplaySession.current.calls if name == "browser_fill_form"]
    values = [f["value"] for f in filled[0]["fields"]]
    assert "s3cret-Example-Pw!" in values, "redaction must not reach the browser call itself"


async def test_an_unknown_credential_is_a_404(client: TestClient):
    usecase_id = await seed(client)
    response = client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={"inputs": {"record_url": "https://example.com/r"}, "credential_id": "nope"},
    )
    assert response.status_code == 404


async def test_the_vault_reports_itself_disabled_without_a_key(client: TestClient, monkeypatch):
    from credentials import Vault

    client.app.state.vault = Vault(None)
    response = client.post("/api/credentials", json={"name": "x", "values": {"a": "b"}})

    assert response.status_code == 503
    assert "CREDENTIALS_KEY" in response.json()["detail"]


# --- guards ----------------------------------------------------------------


async def test_a_draft_use_case_cannot_be_executed(client: TestClient):
    usecase_id = await seed(client, {"status": "draft"})
    response = client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={"inputs": {"record_url": "https://example.com/r"}, "secrets": {"username": "u", "password": "p"}},
    )
    assert response.status_code == 409
    assert "publish" in response.json()["detail"]


async def test_a_missing_input_is_refused_before_the_browser_opens(client: TestClient):
    usecase_id = await seed(client)
    FakeReplaySession.current = None

    response = client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={"inputs": {}, "secrets": {"username": "u", "password": "p"}},
    )

    assert response.status_code == 422
    assert "record_url" in response.json()["detail"]
    assert FakeReplaySession.current is None, "no session was opened"


async def test_a_missing_credential_slot_is_refused(client: TestClient):
    usecase_id = await seed(client)
    response = client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={"inputs": {"record_url": "https://example.com/r"}, "secrets": {"username": "u"}},
    )
    assert response.status_code == 422
    assert "password" in response.json()["detail"]


async def test_executing_an_unknown_use_case_is_a_404(client: TestClient):
    assert client.post("/api/usecases/nope/execute", json={}).status_code == 404


async def test_the_execution_slot_is_reported(client: TestClient):
    assert client.get("/api/executions/active").json()["active"] is None


async def test_executions_are_listed_for_a_use_case(client: TestClient):
    usecase_id = await seed(client)
    client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={"inputs": {"record_url": "https://example.com/r"},
              "secrets": {"username": "u", "password": "p"}},
    )

    executions = client.get(f"/api/usecases/{usecase_id}/executions").json()["executions"]
    assert len(executions) == 1
    assert executions[0]["llm_tokens"] == 0
    assert executions[0]["inputs"] == {"record_url": "https://example.com/r"}


# --- watching the browser --------------------------------------------------


async def test_headless_false_reaches_the_browser_config(client: TestClient):
    """The toggle must actually open a visible window, not just look like it."""
    seen: list[Any] = []

    class RecordingSession(FakeReplaySession):
        def __init__(self, config: Any) -> None:
            seen.append(config)
            super().__init__(config)

    import runner as runner_module

    runner_module.MCPBrowserSession = RecordingSession
    try:
        usecase_id = await seed(client)
        client.post(
            f"/api/usecases/{usecase_id}/execute",
            json={
                "inputs": {"record_url": "https://example.com/r"},
                "secrets": {"username": "u", "password": "p"},
                "headless": False,
            },
        )
        assert seen and seen[0].headless is False
        # And the flag the MCP server actually receives.
        assert "--headless" not in seen[0].command_line()
    finally:
        runner_module.MCPBrowserSession = FakeReplaySession


async def test_headless_defaults_to_the_server_setting(client: TestClient):
    seen: list[Any] = []

    class RecordingSession(FakeReplaySession):
        def __init__(self, config: Any) -> None:
            seen.append(config)
            super().__init__(config)

    import runner as runner_module

    runner_module.MCPBrowserSession = RecordingSession
    try:
        usecase_id = await seed(client)
        client.post(
            f"/api/usecases/{usecase_id}/execute",
            json={
                "inputs": {"record_url": "https://example.com/r"},
                "secrets": {"username": "u", "password": "p"},
            },
        )
        assert seen and seen[0].headless is True
        assert "--headless" in seen[0].command_line()
    finally:
        runner_module.MCPBrowserSession = FakeReplaySession
