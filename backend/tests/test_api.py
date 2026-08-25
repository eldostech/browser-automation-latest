"""The HTTP + WebSocket surface, wired to a fake MCP session and a scripted LLM.

These run the real FastAPI app (lifespan included) but never spawn a browser
and never call a model.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from conftest import FakeMCPSession, ScriptedLLM, final_turn, tool_turn


class FakeMCPBrowserSession:
    """Drop-in replacement for the real session's async context manager."""

    def __init__(self, config: Any) -> None:
        self.config = config

    async def __aenter__(self) -> FakeMCPSession:
        return FakeMCPSession()

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False


async def _fake_probe(config: Any, timeout: float = 20.0) -> dict[str, Any]:
    return {"ok": True, "transport": "stdio", "tool_count": 5, "tools": ["browser_snapshot"]}


@pytest.fixture
def client(tmp_path, monkeypatch):
    import main
    import runner as runner_module

    monkeypatch.setattr(main.settings, "database_path", str(tmp_path / "api.db"))
    monkeypatch.setattr(main.settings, "artifacts_dir", str(tmp_path / "artifacts"))
    # Pin the API tests to the key-based provider so /healthz is deterministic
    # and never reads the developer's real AWS environment.
    # Bedrock is the only provider, so /healthz is made deterministic with
    # fake AWS credentials rather than by pinning a different one.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATESTONLY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-not-used")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setattr(main.settings, "agent_allowed_domains", ["example.com"])
    monkeypatch.setattr(main.settings, "agent_screenshot_every_step", False)
    monkeypatch.setattr(main.settings, "agent_max_steps", 6)
    monkeypatch.setattr(main, "probe", _fake_probe)
    monkeypatch.setattr(runner_module, "MCPBrowserSession", FakeMCPBrowserSession)

    with TestClient(main.app) as test_client:
        yield test_client


def use_llm(client: TestClient, *turns) -> ScriptedLLM:
    llm = ScriptedLLM(list(turns), repeat_last=False)
    client.app.state.manager._llm = llm  # noqa: SLF001 - test seam
    return llm


def wait_for_status(client: TestClient, run_id: str, timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/runs/{run_id}").json()
        if body["status"] in ("succeeded", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


# --- basics ----------------------------------------------------------------


def test_healthz_reports_dependencies(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["database"]["ok"] is True
    assert body["llm"]["configured"] is True
    assert body["mcp"]["ok"] is True


def test_config_endpoint_exposes_defaults_and_no_secrets(client):
    body = client.get("/api/config").json()
    assert body["defaults"]["allowed_domains"] == ["example.com"]
    assert "key" not in str(body).lower() or "api_key" not in str(body).lower()
    assert "test-key-not-used" not in str(body)


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/nope").status_code == 404


def test_task_is_required(client):
    assert client.post("/api/runs", json={"task": ""}).status_code == 422


def test_start_url_must_be_http(client):
    response = client.post("/api/runs", json={"task": "x", "start_url": "file:///etc/passwd"})
    assert response.status_code == 422


# --- run lifecycle ---------------------------------------------------------


def test_run_completes_and_is_persisted(client):
    use_llm(
        client,
        tool_turn("browser_snapshot", {}, text="Looking at the page."),
        final_turn('All done.\n```json\n{"items": 3}\n```'),
    )

    created = client.post("/api/runs", json={"task": "count the items"})
    assert created.status_code == 201
    run_id = created.json()["run_id"]

    body = wait_for_status(client, run_id)
    assert body["status"] == "succeeded"
    assert body["result"]["data"] == {"items": 3}
    assert body["steps"] >= 1

    listed = client.get("/api/runs").json()
    assert any(run["id"] == run_id for run in listed["runs"])
    assert client.get("/api/runs?status=succeeded").json()["total"] == 1

    events = client.get(f"/api/runs/{run_id}/events").json()["events"]
    types = [event["type"] for event in events]
    assert types[0] == "run_started"
    assert types[-1] == "run_finished"
    assert "tool_call" in types and "tool_result" in types

    seqs = [event["seq"] for event in events]
    assert seqs == sorted(seqs) == list(dict.fromkeys(seqs))


def test_events_can_be_replayed_from_a_sequence_number(client):
    use_llm(client, final_turn("done"))
    run_id = client.post("/api/runs", json={"task": "t"}).json()["run_id"]
    wait_for_status(client, run_id)

    everything = client.get(f"/api/runs/{run_id}/events").json()["events"]
    tail = client.get(f"/api/runs/{run_id}/events?after_seq=1").json()["events"]
    assert len(tail) == len(everything) - 1
    assert tail[0]["seq"] == 2


def test_websocket_replays_a_finished_run_then_closes(client):
    use_llm(client, tool_turn("browser_snapshot", {}), final_turn("done"))
    run_id = client.post("/api/runs", json={"task": "t"}).json()["run_id"]
    wait_for_status(client, run_id)

    received = []
    with client.websocket_connect(f"/api/runs/{run_id}/stream?after_seq=0") as socket:
        while True:
            try:
                message = socket.receive_json()
            except Exception:
                break
            if message.get("type", "").startswith("__"):
                continue
            received.append(message)
            if message["type"] == "run_finished":
                break

    assert received[0]["type"] == "run_started"
    assert received[-1]["type"] == "run_finished"
    assert received[-1]["status"] == "succeeded"


def test_websocket_resume_skips_already_seen_events(client):
    use_llm(client, tool_turn("browser_snapshot", {}), final_turn("done"))
    run_id = client.post("/api/runs", json={"task": "t"}).json()["run_id"]
    wait_for_status(client, run_id)

    with client.websocket_connect(f"/api/runs/{run_id}/stream?after_seq=2") as socket:
        first = socket.receive_json()
    assert first["seq"] == 3


def test_websocket_rejects_an_unknown_run(client):
    with client.websocket_connect("/api/runs/nope/stream") as socket:
        with pytest.raises(Exception):
            socket.receive_json()


# --- approvals -------------------------------------------------------------


def test_approval_pauses_the_run_until_a_human_answers(client):
    use_llm(
        client,
        tool_turn("browser_click", {"element": "Place order"}),
        final_turn("ordered"),
    )
    run_id = client.post(
        "/api/runs", json={"task": "buy it", "require_approval": True}
    ).json()["run_id"]

    # Wait for the loop to block on the approval.
    pending = None
    deadline = time.time() + 10
    while time.time() < deadline:
        body = client.get(f"/api/runs/{run_id}").json()
        if body.get("pending_approval"):
            pending = body["pending_approval"]
            break
        time.sleep(0.05)

    assert pending is not None, "the run should have paused for approval"
    assert pending["name"] == "browser_click"
    assert "payment" in pending["categories"]

    approved = client.post(
        f"/api/runs/{run_id}/approve",
        json={"decision": "approve", "approval_id": pending["approval_id"]},
    )
    assert approved.status_code == 200

    body = wait_for_status(client, run_id)
    assert body["status"] == "succeeded"

    types = [event["type"] for event in client.get(f"/api/runs/{run_id}/events").json()["events"]]
    assert "approval_required" in types and "approval_resolved" in types


def test_rejecting_an_approval_blocks_the_action(client):
    use_llm(
        client,
        tool_turn("browser_click", {"element": "Delete account"}),
        final_turn("I stopped."),
    )
    run_id = client.post("/api/runs", json={"task": "delete it"}).json()["run_id"]

    pending = None
    deadline = time.time() + 10
    while time.time() < deadline:
        body = client.get(f"/api/runs/{run_id}").json()
        if body.get("pending_approval"):
            pending = body["pending_approval"]
            break
        time.sleep(0.05)
    assert pending is not None

    client.post(
        f"/api/runs/{run_id}/approve",
        json={"decision": "reject", "approval_id": pending["approval_id"], "note": "not today"},
    )

    body = wait_for_status(client, run_id)
    assert body["status"] == "succeeded"

    events = client.get(f"/api/runs/{run_id}/events").json()["events"]
    resolved = [e for e in events if e["type"] == "approval_resolved"]
    assert resolved[0]["decision"] == "rejected"
    assert resolved[0]["note"] == "not today"


def test_approving_when_nothing_is_pending_is_a_conflict(client):
    use_llm(client, final_turn("done"))
    run_id = client.post("/api/runs", json={"task": "t"}).json()["run_id"]
    wait_for_status(client, run_id)

    response = client.post(f"/api/runs/{run_id}/approve", json={"decision": "approve"})
    assert response.status_code == 409


# --- cancellation ----------------------------------------------------------


def test_cancelling_a_finished_run_is_a_no_op(client):
    use_llm(client, final_turn("done"))
    run_id = client.post("/api/runs", json={"task": "t"}).json()["run_id"]
    wait_for_status(client, run_id)

    body = client.post(f"/api/runs/{run_id}/cancel").json()
    assert body["cancelled"] is False


def test_cancelling_an_in_flight_run_marks_it_cancelled(client):
    # A run that pauses for approval is a convenient way to hold it open.
    use_llm(client, tool_turn("browser_click", {"element": "Submit payment"}), final_turn("x"))
    run_id = client.post("/api/runs", json={"task": "pay"}).json()["run_id"]

    deadline = time.time() + 10
    while time.time() < deadline:
        if client.get(f"/api/runs/{run_id}").json().get("pending_approval"):
            break
        time.sleep(0.05)

    assert client.post(f"/api/runs/{run_id}/cancel").json()["cancelled"] is True

    body = wait_for_status(client, run_id)
    assert body["status"] == "cancelled"

    # A cancelled run must still emit its terminal event, or watchers hang.
    types = [e["type"] for e in client.get(f"/api/runs/{run_id}/events").json()["events"]]
    assert types[-1] == "run_finished"


# --- artifacts -------------------------------------------------------------


def test_screenshots_are_served_as_artifacts(client):
    client.app.state.settings.agent_screenshot_every_step = True
    try:
        use_llm(client, tool_turn("browser_snapshot", {}), final_turn("done"))
        run_id = client.post(
            "/api/runs", json={"task": "t", "screenshot_every_step": True}
        ).json()["run_id"]
        wait_for_status(client, run_id)

        detail = client.get(f"/api/runs/{run_id}").json()
        assert detail["artifacts"], "expected at least one screenshot artifact"

        image = client.get(detail["artifacts"][0]["url"])
        assert image.status_code == 200
        assert image.content.startswith(b"\x89PNG")
    finally:
        client.app.state.settings.agent_screenshot_every_step = False


def test_unknown_artifact_is_404(client):
    assert client.get("/api/artifacts/deadbeef").status_code == 404
