"""Batch execution over HTTP: upload rows, watch progress, export results.

Runs the real endpoints against a faked browser session, so the recovery
contract and the queue hand-off are exercised the way the dashboard will hit
them. The worker runs inside the app under test, which is also the default
single-machine deployment -- so these cover the path an operator actually gets.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from credentials import generate_key
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
        test_client.app.state.repair_model._client = ExplodingLLM()  # noqa: SLF001 - test seam
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
    response = client.post(f"/api/usecases/{usecase_id}/batch", json={"csv": CSV})
    assert response.status_code == 422
    assert "password" in response.json()["detail"]


async def test_a_batch_will_not_take_inline_secrets(client: TestClient):
    """A batch is claimed by a worker that may be another process.

    Inline values would have to be written into the job payload to survive that
    hop, which puts a password in a table a batch listing reads. Saving the
    login as a credential is the supported route, and refusing here is what
    stops the unsupported one working by accident.
    """
    usecase_id = await seed(client)
    response = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"csv": CSV, "secrets": {"username": "u", "password": "p"}},
    )
    assert response.status_code == 422
    assert "credential_id" in response.json()["detail"]


# --- the queue hand-off ----------------------------------------------------


async def test_a_batch_carries_its_rows_and_its_job(client: TestClient):
    """The batch row, its rows and its job are written together.

    Rows used to live only in the memory of the process that accepted the
    upload, which is why nothing could run them but that process. Storing them
    is what lets a worker elsewhere claim the work -- and what lets a resume
    re-run rows that were never attempted.
    """
    from conftest import app_workspace

    usecase_id = await seed(client)
    started = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"csv": CSV, "credential_id": credential(client)},
    ).json()
    batch_id = started["batch_id"]

    data = await app_workspace(client.app)
    rows = await data.get_batch_rows(batch_id)
    assert [row["record_url"] for row in rows] == [
        "https://example.com/record/1",
        "https://example.com/record/2",
        "https://example.com/record/3",
    ]

    batch = await data.get_batch(batch_id)
    assert batch["job_id"], "a queued batch must name the job that will run it"

    await wait_for_batch(client, batch_id)


async def test_a_stored_row_never_holds_a_secret(client: TestClient):
    """Only the credential id makes the trip.

    The worker opens the credential itself. Putting the values in the batch
    would write a password into a table that a batch listing reads, which is
    the thing the vault exists to prevent.
    """
    from conftest import app_workspace

    password = "distinctive-Example-Pw!"
    credential_id = client.post(
        "/api/credentials",
        json={"name": "Distinctive", "values": {"username": "u", "password": password}},
    ).json()["id"]

    usecase_id = await seed(client)
    started = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"csv": CSV, "credential_id": credential_id},
    ).json()

    data = await app_workspace(client.app)
    stored = repr(await data.get_batch_rows(started["batch_id"]))
    assert password not in stored
    assert repr(await data.get_batch(started["batch_id"])).count(password) == 0

    await wait_for_batch(client, started["batch_id"])


# --- one at a time ---------------------------------------------------------


async def test_a_second_batch_is_queued_rather_than_refused(client: TestClient):
    """The single slot is gone; the queue holds the second one instead.

    This used to be a 409 telling the caller to try again later, which made
    "one at a time" the caller's problem. Per-workspace concurrency in the
    queue gives the same one-at-a-time guarantee for one tenant, without asking
    anyone to poll for a free slot -- and without one workspace's batch
    stopping another workspace running anything at all.
    """
    usecase_id = await seed(client)
    credential_id = credential(client)
    first = client.post(
        f"/api/usecases/{usecase_id}/batch", json={"csv": CSV, "credential_id": credential_id}
    )
    assert first.status_code == 202

    second = client.post(
        f"/api/usecases/{usecase_id}/batch", json={"csv": CSV, "credential_id": credential_id}
    )
    assert second.status_code == 202
    assert second.json()["batch_id"] != first.json()["batch_id"]

    for batch_id in (first.json()["batch_id"], second.json()["batch_id"]):
        await wait_for_batch(client, batch_id)


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


async def test_cancelling_a_batch_that_does_not_exist_is_a_404(client: TestClient):
    """Ownership is established before liveness now.

    Cancelling used to mean taking the in-process slot, so "no such batch" and
    "that batch is idle" were the same answer. A queued batch lives in a table
    a tenant either can or cannot see, so the scoped lookup is both the 404 and
    the tenancy check, and it comes first.
    """
    assert client.post("/api/batches/nope/cancel").status_code == 404


async def test_cancelling_a_finished_batch_is_refused(client: TestClient):
    usecase_id = await seed(client)
    started = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"csv": CSV, "credential_id": credential(client)},
    ).json()
    await wait_for_batch(client, started["batch_id"])

    response = client.post(f"/api/batches/{started['batch_id']}/cancel")
    assert response.status_code == 409
    assert "not running" in response.json()["detail"]


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


# --- a use case that runs one record but cannot run many -------------------


async def test_a_row_that_signs_out_is_refused_when_a_batch_is_asked_for(
    client: TestClient,
):
    """The failure behind "record one worked and everything after failed".

    Refused here rather than at publish, and that placement is the whole
    point: the use case runs a single record perfectly, so blocking it earlier
    left somebody with a recording they could only delete. Here the person has
    just asked for many records, which is exactly when it matters.
    """
    from conftest import app_workspace

    data = await app_workspace(client.app)
    signing_out = {
        **USE_CASE,
        "id": "uc-signs-out",
        "session_check": None,
        "row_steps": [
            *USE_CASE["row_steps"],
            {
                "id": "s-out",
                "action": "click",
                "locators": [{"strategy": "role", "role": "link", "name": "Sign out"}],
            },
        ],
    }
    usecase_id, _ = await data.save_usecase(signing_out)

    response = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"csv": CSV, "credential_id": credential(client)},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "signs out at the end of every record" in detail
    assert "single record runs fine" in detail, "says what does still work"
    assert "session check" in detail, "and names a way to fix it"


async def test_that_same_use_case_still_runs_one_record(client: TestClient):
    """The other half of the argument for where the check lives."""
    from conftest import app_workspace

    data = await app_workspace(client.app)
    signing_out = {
        **USE_CASE,
        "id": "uc-signs-out-single",
        "session_check": None,
        "row_steps": [
            *USE_CASE["row_steps"],
            {
                "id": "s-out",
                "action": "click",
                "locators": [{"strategy": "role", "role": "link", "name": "Sign out"}],
            },
        ],
    }
    usecase_id, _ = await data.save_usecase(signing_out)

    response = client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={
            "inputs": {"record_url": "https://example.com/record/1"},
            "credential_id": credential(client),
        },
    )

    # Accepted and run. Whether the run then succeeds is the page's business --
    # this double has no "Sign out" link on it -- and what matters here is that
    # the request was not refused. A single record is not the shape the batch
    # gate is about.
    assert response.status_code == 201, response.text
    assert "signs out at the end of every record" not in response.text
