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
    # Bedrock is the only provider, so /healthz is made deterministic with
    # fake AWS credentials rather than by pinning a different one.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATESTONLY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-not-used")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
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


# --- repairing a failure instead of re-recording ---------------------------


class RepairLLM:
    """Proposes a fixed repair and counts the calls."""

    model = "fake"

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    async def run_turn(self, *, system, messages, tools, on_text_delta=None, timeout=None):
        self.calls += 1
        from llm import LLMTurn, ToolCallRequest

        return LLMTurn(
            tool_calls=[ToolCallRequest(id="t1", name="propose_repair", input=self.payload)],
            stop_reason="tool_use",
            usage={"input_tokens": 800, "output_tokens": 40},
        )


BROKEN = {
    **USE_CASE,
    "id": "uc-broken",
    "row_steps": [
        {
            "id": "s1",
            "action": "click",
            "description": "Submit button",
            # Not on the signed-in page, so the row fails.
            "locators": [{"strategy": "role", "role": "button", "name": "Nonexistent"}],
        }
    ],
    "outputs": [],
}


async def test_a_failed_run_can_be_repaired_without_re_recording(client: TestClient):
    store = client.app.state.store
    usecase_id, _ = await store.save_usecase(BROKEN)

    failure = client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={"inputs": {"record_url": "https://example.com/r"},
              "secrets": {"username": "u", "password": "p"}},
    ).json()
    assert failure["status"] == "failed"

    llm = RepairLLM(
        {
            "diagnosis": "The button is now called Submit.",
            "confidence": "high",
            # The candidate list is in document order: 0 is the "Welcome back"
            # heading, 1 is the Submit button.
            "fixes": [
                {"kind": "replace_locator", "step_id": "s1", "element_index": 1,
                 "reason": "same control, new label"}
            ],
        }
    )
    client.app.state.manager._llm = llm  # noqa: SLF001 - test seam

    repaired = client.post(
        f"/api/usecases/{usecase_id}/repair", json={"execution_id": failure["execution_id"]}
    )

    assert repaired.status_code == 201, repaired.text
    body = repaired.json()
    assert body["repaired"] is True
    assert body["llm_tokens"] == 840
    assert llm.calls == 1, "one call to mend it"
    assert any("s1" in line for line in body["applied"])

    # Saved as a new draft version; the original is untouched and review is
    # required before it can run again.
    detail = client.get(f"/api/usecases/{usecase_id}").json()
    assert detail["definition"]["status"] == "draft"
    assert detail["definition"]["version"] == 2
    assert detail["definition"]["row_steps"][0]["locators"][0]["name"] == "Submit"
    assert detail["definition"]["row_steps"][0]["locators"][1]["name"] == "Nonexistent"
    original = client.get(f"/api/usecases/{usecase_id}", params={"version": 1}).json()
    assert original["definition"]["row_steps"][0]["locators"][0]["name"] == "Nonexistent"


async def test_a_repair_the_model_declines_reports_why_and_changes_nothing(client: TestClient):
    store = client.app.state.store
    usecase_id, _ = await store.save_usecase({**BROKEN, "id": "uc-unfixable"})

    failure = client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={"inputs": {"record_url": "https://example.com/r"},
              "secrets": {"username": "u", "password": "p"}},
    ).json()

    client.app.state.manager._llm = RepairLLM(  # noqa: SLF001
        {
            "diagnosis": "The per-row work needs a value worked out from the page.",
            "fixes": [],
            "unfixable_reason": "this task needs reasoning on every row",
            "confidence": "high",
        }
    )

    body = client.post(
        f"/api/usecases/{usecase_id}/repair", json={"execution_id": failure["execution_id"]}
    ).json()

    assert body["repaired"] is False
    assert "reasoning on every row" in body["unfixable_reason"]
    assert client.get(f"/api/usecases/{usecase_id}").json()["definition"]["version"] == 1


async def test_repairing_with_no_failure_to_look_at_is_a_404(client: TestClient):
    usecase_id = await seed(client)
    response = client.post(f"/api/usecases/{usecase_id}/repair", json={})
    assert response.status_code == 404
    assert "no failed run" in response.json()["detail"]


async def test_repairing_an_unknown_use_case_is_a_404(client: TestClient):
    assert client.post("/api/usecases/nope/repair", json={}).status_code == 404


async def test_a_repair_that_changes_nothing_is_not_reported_as_repaired(client: TestClient):
    """Regression: pressing Fix twice saved two identical versions and said
    'repaired' both times, which looks exactly like a change that did not
    persist."""
    store = client.app.state.store
    usecase_id, _ = await store.save_usecase({**BROKEN, "id": "uc-noop"})

    failure = client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={"inputs": {"record_url": "https://example.com/r"},
              "secrets": {"username": "u", "password": "p"}},
    ).json()

    # A fix pointing at a step that does not exist: nothing can be applied.
    client.app.state.manager._llm = RepairLLM(  # noqa: SLF001
        {
            "diagnosis": "Something moved.",
            "confidence": "medium",
            "fixes": [
                {"kind": "replace_locator", "step_id": "not-a-step", "element_index": 0}
            ],
        }
    )

    body = client.post(
        f"/api/usecases/{usecase_id}/repair", json={"execution_id": failure["execution_id"]}
    ).json()

    assert body["repaired"] is False
    assert "nothing was saved" in body["unfixable_reason"]
    assert any("SKIPPED" in line for line in body["applied"])

    # And crucially, no version was created.
    detail = client.get(f"/api/usecases/{usecase_id}").json()
    assert detail["definition"]["version"] == 1
    assert len(detail["versions"]) == 1
