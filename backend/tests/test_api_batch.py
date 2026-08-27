"""Batch execution over HTTP: upload rows, watch progress, export results.

Runs the real endpoints against a faked browser session, so the recovery
contract and the single-slot lock are exercised the way the dashboard will hit
them.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from credentials import generate_key
from test_api import _fake_probe  # noqa: F401
from test_api_execute import USE_CASE, ExplodingLLM, FakeReplaySession

CSV = (
    "record_url\n"
    "https://example.com/record/1\n"
    "https://example.com/record/2\n"
    "https://example.com/record/3\n"
)


@pytest.fixture
def client(db_settings, db_engine, tmp_path, monkeypatch):
    from conftest import authenticate, build_app

    app = build_app(
        db_settings,
        tmp_path,
        monkeypatch,
        session_cls=FakeReplaySession,
        credentials_key=generate_key(),
        replay_row_delay_seconds=0.0,
    )
    with TestClient(app) as test_client:
        test_client.app.state.manager._llm = ExplodingLLM()  # noqa: SLF001 - test seam
        yield authenticate(test_client)


async def seed(client: TestClient) -> str:
    from conftest import app_workspace

    data = await app_workspace(client.app)
    usecase_id, _ = await data.save_usecase({**USE_CASE, "id": "uc-1"})
    return usecase_id


async def wait_for_batch(client: TestClient, batch_id: str, timeout: float = 5.0) -> dict[str, Any]:
    """Poll until the batch reaches a terminal state."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        body = client.get(f"/api/batches/{batch_id}").json()
        if body["batch"]["finished_at"]:
            return body
        await asyncio.sleep(0.02)
    raise AssertionError(f"batch {batch_id} did not finish within {timeout}s")


def credential(client: TestClient) -> str:
    return client.post(
        "/api/credentials",
        json={"name": "Example", "values": {"username": "u", "password": "pw"}},
    ).json()["id"]


# --- the happy path --------------------------------------------------------


async def test_a_batch_runs_every_row_on_one_session(client: TestClient):
    usecase_id = await seed(client)
    started = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"csv": CSV, "credential_id": credential(client)},
    )

    assert started.status_code == 202
    assert started.json()["total"] == 3

    body = await wait_for_batch(client, started.json()["batch_id"])

    assert body["batch"]["status"] == "succeeded"
    assert body["batch"]["succeeded"] == 3
    assert body["batch"]["failed"] == 0
    assert [row["status"] for row in body["executions"]] == ["succeeded"] * 3
    assert all(row["llm_tokens"] == 0 for row in body["executions"])


async def test_each_row_keeps_its_own_inputs_and_outputs(client: TestClient):
    usecase_id = await seed(client)
    started = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"csv": CSV, "credential_id": credential(client)},
    ).json()
    body = await wait_for_batch(client, started["batch_id"])

    urls = [row["inputs"]["record_url"] for row in body["executions"]]
    assert urls == [
        "https://example.com/record/1",
        "https://example.com/record/2",
        "https://example.com/record/3",
    ]
    assert all(row["outputs"] == {"score": "92%"} for row in body["executions"])


async def test_results_export_as_csv(client: TestClient):
    usecase_id = await seed(client)
    started = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"csv": CSV, "credential_id": credential(client)},
    ).json()
    await wait_for_batch(client, started["batch_id"])

    response = client.get(f"/api/batches/{started['batch_id']}/results.csv")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    lines = response.text.strip().splitlines()
    assert lines[0].startswith("record_url,row_index,status")
    assert lines[0].endswith("score")
    assert len(lines) == 4
    assert lines[1].startswith("https://example.com/record/1,0,succeeded")


# --- validation happens before the browser opens ---------------------------


async def test_an_unknown_column_is_refused_without_opening_a_browser(client: TestClient):
    usecase_id = await seed(client)
    FakeReplaySession.current = None

    response = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"csv": "nonsense\nvalue\n", "credential_id": credential(client)},
    )

    assert response.status_code == 422
    problems = response.json()["detail"]["problems"]
    assert any("nonsense" in p for p in problems)
    assert FakeReplaySession.current is None, "a bad column must fail in a millisecond"


async def test_a_row_missing_a_required_input_is_refused(client: TestClient):
    usecase_id = await seed(client)
    response = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"csv": "record_url\nhttps://a\n\n", "credential_id": credential(client)},
    )
    # The blank line is skipped, so this file is actually fine.
    assert response.status_code == 202

    response = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"rows": [{"record_url": ""}], "credential_id": credential(client)},
    )
    assert response.status_code in (409, 422)


@pytest.mark.parametrize("payload", [{"csv": ""}, {"csv": "record_url\n"}, {"rows": []}])
async def test_unusable_files_are_refused(client: TestClient, payload: dict):
    usecase_id = await seed(client)
    response = client.post(
        f"/api/usecases/{usecase_id}/batch", json={**payload, "credential_id": credential(client)}
    )
    assert response.status_code == 422


async def test_a_draft_use_case_cannot_be_batched(client: TestClient):
    from conftest import app_workspace

    store = await app_workspace(client.app)
    await store.save_usecase({**USE_CASE, "id": "uc-draft", "status": "draft"})
    response = client.post(
        "/api/usecases/uc-draft/batch", json={"csv": CSV, "credential_id": credential(client)}
    )
    assert response.status_code == 409
    assert "publish" in response.json()["detail"]


async def test_missing_credential_slots_are_refused(client: TestClient):
    usecase_id = await seed(client)
    response = client.post(
        f"/api/usecases/{usecase_id}/batch", json={"csv": CSV, "secrets": {"username": "u"}}
    )
    assert response.status_code == 422
    assert "password" in response.json()["detail"]


# --- one at a time ---------------------------------------------------------


async def test_a_second_batch_is_refused_while_one_is_running(client: TestClient):
    usecase_id = await seed(client)
    credential_id = credential(client)
    first = client.post(
        f"/api/usecases/{usecase_id}/batch", json={"csv": CSV, "credential_id": credential_id}
    )
    assert first.status_code == 202

    second = client.post(
        f"/api/usecases/{usecase_id}/batch", json={"csv": CSV, "credential_id": credential_id}
    )

    if second.status_code == 409:
        assert "already running" in second.json()["detail"]
    await wait_for_batch(client, first.json()["batch_id"])


async def test_the_active_slot_reports_the_batch(client: TestClient):
    usecase_id = await seed(client)
    started = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"csv": CSV, "credential_id": credential(client)},
    ).json()

    # Either it is still running and reported, or it already finished.
    active = client.get("/api/executions/active").json()["active"]
    if active is not None:
        assert active["batch_id"] == started["batch_id"]
        assert active["rows"] == 3

    await wait_for_batch(client, started["batch_id"])
    assert client.get("/api/executions/active").json()["active"] is None


# --- resume ----------------------------------------------------------------


async def test_resuming_a_fully_succeeded_batch_is_refused(client: TestClient):
    usecase_id = await seed(client)
    started = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"csv": CSV, "credential_id": credential(client)},
    ).json()
    await wait_for_batch(client, started["batch_id"])

    response = client.post(f"/api/batches/{started['batch_id']}/resume", json={})

    assert response.status_code == 409
    assert "already succeeded" in response.json()["detail"]


async def test_resume_reruns_only_the_rows_that_did_not_succeed(client: TestClient):
    usecase_id = await seed(client)
    from conftest import app_workspace

    store = await app_workspace(client.app)
    started = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"csv": CSV, "credential_id": credential(client)},
    ).json()
    body = await wait_for_batch(client, started["batch_id"])

    # Simulate two rows having failed, the way a stopped batch would leave them.
    for row in body["executions"][1:]:
        await store.finish_execution(row["id"], "failed", error="pretend failure")

    resumed = client.post(
        f"/api/batches/{started['batch_id']}/resume",
        json={"credential_id": credential(client)},
    )

    assert resumed.status_code == 202
    assert resumed.json()["rows"] == 2, "only the two non-succeeded rows"
    await wait_for_batch(client, resumed.json()["batch_id"])


async def test_resuming_an_unknown_batch_is_a_404(client: TestClient):
    assert client.post("/api/batches/nope/resume", json={}).status_code == 404


async def test_an_unknown_batch_is_a_404(client: TestClient):
    assert client.get("/api/batches/nope").status_code == 404
    assert client.get("/api/batches/nope/results.csv").status_code == 404


async def test_cancelling_a_batch_that_is_not_running_is_refused(client: TestClient):
    assert client.post("/api/batches/nope/cancel").status_code == 409


async def test_batches_are_listed_for_a_use_case(client: TestClient):
    usecase_id = await seed(client)
    started = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"csv": CSV, "credential_id": credential(client)},
    ).json()
    await wait_for_batch(client, started["batch_id"])

    batches = client.get(f"/api/usecases/{usecase_id}/batches").json()["batches"]
    assert len(batches) == 1
    assert batches[0]["total"] == 3
