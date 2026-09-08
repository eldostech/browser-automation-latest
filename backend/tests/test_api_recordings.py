"""The recording endpoints, end to end over HTTP.

The subprocess is faked the same way ``test_recorder.py`` fakes it: a script
that writes what codegen would write and exits. What is under test here is the
route from "start a recording" to "there is a draft use case", which is the
path replacing an agent run costing ~247,000 tokens with one costing none.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap

import pytest
from fastapi.testclient import TestClient

from credentials import generate_key
from test_api_execute import ExplodingLLM, FakeReplaySession
from test_recorder import RECORDED

pytestmark = pytest.mark.anyio


@pytest.fixture
def fake_command(tmp_path):
    program = tmp_path / "fake_codegen.py"
    program.write_text(
        textwrap.dedent(
            f"""
            import sys

            SCRIPT = {RECORDED!r}
            for arg in sys.argv[1:]:
                if arg.startswith("--output="):
                    with open(arg.split("=", 1)[1], "w", encoding="utf-8") as handle:
                        handle.write(SCRIPT)
            """
        ).strip(),
        encoding="utf-8",
    )
    return f'"{sys.executable}" "{program}"'


@pytest.fixture
def client(db_settings, db_engine, tmp_path, monkeypatch, fake_command):
    from conftest import authenticate, build_app

    app = build_app(
        db_settings,
        tmp_path,
        monkeypatch,
        session_cls=FakeReplaySession,
        credentials_key=generate_key(),
        recorder_command=fake_command,
    )
    with TestClient(app) as test_client:
        test_client.app.state.repair_model._client = ExplodingLLM()  # noqa: SLF001 - test seam
        yield authenticate(test_client)


async def settled(client: TestClient, recording_id: str, timeout: float = 15.0) -> dict:
    """Poll until the recording leaves the `recording` state.

    Polling the endpoint rather than awaiting the recorder's task: the app runs
    in the TestClient's own event loop, so that task belongs to a different
    loop than the test. Polling is also exactly what the UI does, which makes
    this the path under test rather than a shortcut around it.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        body = client.get(f"/api/recordings/{recording_id}").json()
        if body["status"] != "recording":
            return body
        await asyncio.sleep(0.05)
    raise AssertionError(f"recording {recording_id} never finished")


async def record(client: TestClient) -> str:
    """Start a recording and wait for the fake window to close."""
    started = client.post(
        "/api/recordings", json={"start_url": "https://example.com/login", "name": "Orders"}
    )
    assert started.status_code == 201, started.text
    recording_id = started.json()["recording_id"]
    body = await settled(client, recording_id)
    assert body["status"] == "ready", body.get("error")
    return recording_id


# --- starting and finishing -------------------------------------------------


async def test_a_recording_finishes_and_reports_what_it_saw(client: TestClient):
    started = client.post(
        "/api/recordings", json={"start_url": "https://example.com/login", "name": "Orders"}
    )
    assert started.status_code == 201
    recording_id = started.json()["recording_id"]
    assert started.json()["status"] == "recording"

    body = await settled(client, recording_id)
    assert body["status"] == "ready", body.get("error")
    assert body["summary"].startswith("6 step(s)")
    assert body["domains"] == ["example.com"]
    # The values typed during the recording, offered so the user can say which
    # are per-row inputs and which are credentials.
    assert "s3cret-Example-Pw" in body["typed"]


async def test_a_recording_is_not_visible_to_another_tenant(client: TestClient):
    """The scoped lookup is the tenancy check: an id from elsewhere is simply
    not found."""
    started = client.post(
        "/api/recordings", json={"start_url": "https://example.com/", "name": "x"}
    ).json()
    assert (
        client.app.state.recorder.get(started["recording_id"], "some-other-workspace") is None
    )


async def test_an_unknown_recording_is_a_404(client: TestClient):
    assert client.get("/api/recordings/nope").status_code == 404
    assert client.post("/api/recordings/nope/save", json={}).status_code == 404


async def test_recording_can_be_switched_off_for_a_deployment(
    db_settings, db_engine, tmp_path, monkeypatch
):
    """A pod with no display answers 501 rather than failing at spawn time
    with an X11 error nobody can act on."""
    from conftest import authenticate, build_app

    app = build_app(db_settings, tmp_path, monkeypatch, recorder_enabled=False)
    with TestClient(app) as raw:
        client = authenticate(raw)
        response = client.post(
            "/api/recordings", json={"start_url": "https://example.com/", "name": "x"}
        )
        assert response.status_code == 501
        assert "display" in response.json()["detail"]
        assert client.get("/api/recordings").json()["available"] is False


# --- saving it as a use case ------------------------------------------------


async def test_a_recording_becomes_a_draft_use_case(client: TestClient):
    recording_id = await record(client)

    saved = client.post(
        f"/api/recordings/{recording_id}/save",
        json={
            "name": "Submit orders",
            "fields": [
                {"name": "username", "value": "nitin", "secret": True},
                {"name": "password", "value": "s3cret-Example-Pw", "secret": True},
                {"name": "reference", "value": "A-1024", "secret": False},
            ],
        },
    )
    assert saved.status_code == 201, saved.text
    assert saved.json()["status"] == "draft"

    use_case = client.get(f"/api/usecases/{saved.json()['usecase_id']}").json()["definition"]
    assert use_case["name"] == "Submit orders"
    assert [s["name"] for s in use_case["secrets"]] == ["username", "password"]
    assert [i["name"] for i in use_case["inputs"]] == ["reference"]
    # The login runs once per session, the rest once per row.
    assert len(use_case["setup_steps"]) == 4
    # The recording's own origin is held as a template so one document can run
    # in dev, UAT and production; base_url is what it resolves to when the
    # deployment names nothing.
    assert use_case["allowed_domains"] == ["{{env.base_url}}"]
    assert use_case["base_url"] == "https://example.com"


async def test_a_credential_does_not_survive_in_the_saved_steps(client: TestClient):
    recording_id = await record(client)
    saved = client.post(
        f"/api/recordings/{recording_id}/save",
        json={
            "fields": [{"name": "password", "value": "s3cret-Example-Pw", "secret": True}]
        },
    ).json()

    body = client.get(f"/api/usecases/{saved['usecase_id']}").text
    assert "s3cret-Example-Pw" not in body
    assert "{{secret.password}}" in body


async def test_a_recorded_credential_is_saved_to_the_vault_automatically(client: TestClient):
    """The value a person just typed while recording is exactly the value the
    vault needs; making them re-type it on a second screen was the gap."""
    recording_id = await record(client)

    saved = client.post(
        f"/api/recordings/{recording_id}/save",
        json={
            "name": "Submit orders",
            "fields": [
                {"name": "username", "value": "nitin", "secret": True},
                {"name": "password", "value": "s3cret-Example-Pw", "secret": True},
                {"name": "reference", "value": "A-1024", "secret": False},
            ],
        },
    ).json()

    assert saved["credential_saved"] is True
    assert saved["credential_name"] == "Submit orders"

    credentials = client.get("/api/credentials").json()["credentials"]
    stored = next(c for c in credentials if c["name"] == "Submit orders")
    assert stored["slots"] == ["password", "username"]


async def test_the_auto_saved_credential_value_is_never_in_the_response(client: TestClient):
    recording_id = await record(client)

    saved = client.post(
        f"/api/recordings/{recording_id}/save",
        json={
            "name": "Submit orders",
            "fields": [{"name": "password", "value": "s3cret-Example-Pw", "secret": True}],
        },
    )
    assert "s3cret-Example-Pw" not in saved.text


async def test_a_recording_with_no_secret_field_saves_no_credential(client: TestClient):
    recording_id = await record(client)

    saved = client.post(
        f"/api/recordings/{recording_id}/save",
        json={"fields": [{"name": "reference", "value": "A-1024", "secret": False}]},
    ).json()

    assert saved["credential_saved"] is False
    assert saved["credential_name"] is None


async def test_re_saving_under_the_same_name_replaces_the_credential_not_duplicates_it(
    client: TestClient,
):
    """Re-recording the same workflow is the ordinary reason to save under
    the same name twice; the vault entry should track the newest value, not
    accumulate one for every take."""
    first_recording = await record(client)
    client.post(
        f"/api/recordings/{first_recording}/save",
        json={
            "name": "Submit orders",
            "fields": [{"name": "password", "value": "old-pw", "secret": True}],
        },
    )

    second_recording = await record(client)
    client.post(
        f"/api/recordings/{second_recording}/save",
        json={
            "name": "Submit orders",
            "fields": [{"name": "password", "value": "new-pw", "secret": True}],
        },
    )

    credentials = client.get("/api/credentials").json()["credentials"]
    matching = [c for c in credentials if c["name"] == "Submit orders"]
    assert len(matching) == 1


async def test_saving_an_unfinished_recording_is_refused(client: TestClient):
    started = client.post(
        "/api/recordings", json={"start_url": "https://example.com/", "name": "x"}
    ).json()
    # Deliberately not waited on: the window is still "open".
    response = client.post(f"/api/recordings/{started['recording_id']}/save", json={})
    assert response.status_code == 409

    await settled(client, started["recording_id"])


async def test_a_saved_recording_is_discarded_afterwards(client: TestClient):
    recording_id = await record(client)
    client.post(
        f"/api/recordings/{recording_id}/save",
        json={"fields": [{"name": "reference", "value": "A-1024", "secret": False}]},
    )
    assert client.get(f"/api/recordings/{recording_id}").status_code == 404


async def test_a_recording_can_be_thrown_away(client: TestClient):
    recording_id = await record(client)
    assert client.delete(f"/api/recordings/{recording_id}").status_code == 200
    assert client.get(f"/api/recordings/{recording_id}").status_code == 404
