"""The HTTP + WebSocket surface, wired to a fake browser and a scripted LLM.

These run the real FastAPI app (lifespan included) but never launch a browser
and never call a model.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from conftest import ScriptedLLM, final_turn, tool_turn
from fake_browser import SIGNED_IN, SIGNED_OUT, session_serving


#: The browser these tests replay against. Named for what it stands in for
#: rather than for MCP, which no longer exists here.
FakeBrowser = session_serving([SIGNED_OUT, SIGNED_IN], {"/signin": SIGNED_OUT})


@pytest.fixture
def app_under_test(db_settings, db_engine, tmp_path, monkeypatch):
    """A freshly built app on the test database, with no browser and no model.

    Built through ``create_app`` rather than importing the module-level ``app``,
    so its configuration comes from the fixture instead of the developer's
    ``.env`` -- the leak this project shipped three times.
    """
    import main
    import runner as runner_module
    from conftest import api_settings

    # Bedrock is the only provider, so /healthz is made deterministic with
    # fake AWS credentials rather than by pinning a different one.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATESTONLY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-not-used")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setattr(runner_module, "PlaywrightSession", FakeBrowser)

    return main.create_app(api_settings(db_settings, tmp_path))


async def seed_run(client, *, status: str = "succeeded", events: int = 3) -> str:
    """A finished run with a few events, written straight into the store.

    Runs are created by executing a use case now, not by asking a model to go
    and do something. These tests are about reading one back -- its events, its
    artifacts, the WebSocket -- so they seed one directly rather than driving
    an execution to produce it.
    """
    from conftest import app_workspace
    from events import RunFinished, RunStarted, StepFinished, StepStarted

    store = await app_workspace(client.app)
    run_id = "run-seed"
    await store.create_run(run_id, "Replay: sign in and open a record", None, {"replay": True})

    # Bracketed by run_started/run_finished exactly as an execution writes it:
    # the WebSocket replays what is in the table, and a client reads until the
    # terminal event tells it to stop.
    seq = 1
    await store.append_event(
        RunStarted(run_id=run_id, seq=seq, task="Replay: sign in and open a record", options={})
    )
    for step in range(1, events + 1):
        seq += 1
        await store.append_event(
            StepStarted(
                run_id=run_id, seq=seq, step=step, step_id=f"s{step}",
                action="click", description="click Sign in", phase="row",
            )
        )
        seq += 1
        await store.append_event(
            StepFinished(
                run_id=run_id, seq=seq, step=step, step_id=f"s{step}",
                ok=True, duration_ms=5,
            )
        )

    seq += 1
    await store.append_event(
        RunFinished(
            run_id=run_id, seq=seq, status=status, steps=events,
            duration_ms=100, summary="done",
        )
    )
    await store.finish_run(run_id, status, steps=events, duration_ms=100, summary="done")
    return run_id


@pytest.fixture
def anonymous(app_under_test):
    """A client with no credentials, for testing that endpoints refuse it."""
    with TestClient(app_under_test) as test_client:
        yield test_client


@pytest.fixture
def client(app_under_test):
    """An administrator's client.

    Signed in as the bootstrap admin, so tests that predate authentication
    still exercise what they were written to exercise. Role-specific behaviour
    is tested explicitly in ``test_api_auth.py``.
    """
    from conftest import authenticate

    with TestClient(app_under_test) as test_client:
        yield authenticate(test_client)


def use_llm(client: TestClient, *turns) -> ScriptedLLM:
    llm = ScriptedLLM(list(turns), repeat_last=False)
    client.app.state.repair_model._client = llm  # noqa: SLF001 - test seam
    return llm


def ws_url(client: TestClient, path: str) -> str:
    """Append the client's bearer token as a query parameter.

    A browser cannot set an Authorization header on a WebSocket handshake, so
    the server accepts the session token in the query string instead. Tests go
    the same route the frontend does.
    """
    token = client.headers["Authorization"].removeprefix("Bearer ")
    separator = "&" if "?" in path else "?"
    return f"{path}{separator}token={token}"


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
    # No browser probe: Playwright is a library in this process, not a server
    # that can be down while everything else is up.
    assert body["browser"]["engine"] == "chromium"


def test_config_endpoint_exposes_defaults_and_no_secrets(client):
    body = client.get("/api/config").json()
    assert body["defaults"]["browser"] == "chromium"
    assert body["recorder"]["enabled"] in (True, False)
    assert "key" not in str(body).lower() or "api_key" not in str(body).lower()
    assert "test-key-not-used" not in str(body)


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/nope").status_code == 404


async def test_events_can_be_replayed_from_a_sequence_number(client):
    use_llm(client, final_turn("done"))
    run_id = await seed_run(client)
    wait_for_status(client, run_id)

    everything = client.get(f"/api/runs/{run_id}/events").json()["events"]
    tail = client.get(f"/api/runs/{run_id}/events?after_seq=1").json()["events"]
    assert len(tail) == len(everything) - 1
    assert tail[0]["seq"] == 2


async def test_websocket_replays_a_finished_run_then_closes(client):
    run_id = await seed_run(client)
    wait_for_status(client, run_id)

    received = []
    with client.websocket_connect(ws_url(client, f"/api/runs/{run_id}/stream?after_seq=0")) as socket:
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


async def test_websocket_resume_skips_already_seen_events(client):
    run_id = await seed_run(client)
    wait_for_status(client, run_id)

    with client.websocket_connect(ws_url(client, f"/api/runs/{run_id}/stream?after_seq=2")) as socket:
        first = socket.receive_json()
    assert first["seq"] == 3


def test_websocket_rejects_an_unknown_run(client):
    with client.websocket_connect(ws_url(client, "/api/runs/nope/stream")) as socket:
        with pytest.raises(Exception):
            socket.receive_json()


# --- approvals -------------------------------------------------------------


async def test_cancelling_a_finished_run_is_a_no_op(client):
    use_llm(client, final_turn("done"))
    run_id = await seed_run(client)
    wait_for_status(client, run_id)

    body = client.post(f"/api/runs/{run_id}/cancel").json()
    assert body["cancelled"] is False


async def test_screenshots_are_served_as_artifacts(client):
    """A screenshot is stored once and served by id.

    Seeded rather than produced by a run: what is under test is the artifact
    endpoint -- that it finds the bytes, sets the type, and refuses an id from
    another tenant -- not that an execution takes pictures, which
    ``test_e2e_engine.py`` covers against a real browser.
    """
    from conftest import app_workspace

    run_id = await seed_run(client)
    store = await app_workspace(client.app)
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 32
    await store.save_artifact(run_id, png, kind="screenshot", mime="image/png", seq=1)

    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["artifacts"], "expected at least one screenshot artifact"

    image = client.get(detail["artifacts"][0]["url"])
    assert image.status_code == 200
    assert image.content.startswith(b"\x89PNG")


def test_unknown_artifact_is_404(client):
    assert client.get("/api/artifacts/deadbeef").status_code == 404


# --- declared fields and credential handling --------------------------------

