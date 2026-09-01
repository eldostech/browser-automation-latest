"""Uploading a file, seeing what is in it, and agreeing how it maps.

The step this covers is the one a non-technical user actually performs, so the
assertions are about what they are shown and what they are stopped from doing
-- not about parsing, which ``test_ingest.py`` owns.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from credentials import generate_key
from test_api_execute import USE_CASE, ExplodingLLM, FakeReplaySession

CSV = b"Record URL,Answer\nhttps://example.com/1,alpha\nhttps://example.com/2,beta\n"


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


def upload(client: TestClient, data: bytes = CSV, filename: str = "records.csv"):
    return client.post(
        "/api/datasets",
        files={"file": (filename, io.BytesIO(data), "text/csv")},
        data={"name": "Weekly records"},
    )


def credential(client: TestClient) -> str:
    return client.post(
        "/api/credentials",
        json={"name": "Example", "values": {"username": "u", "password": "pw"}},
    ).json()["id"]


async def seed(client: TestClient) -> str:
    from conftest import app_workspace

    data = await app_workspace(client.app)
    usecase_id, _ = await data.save_usecase({**USE_CASE, "id": "uc-1"})
    return usecase_id


# --- uploading -------------------------------------------------------------


async def test_an_upload_comes_back_described(client: TestClient):
    body = upload(client).json()

    assert body["row_count"] == 2
    assert body["name"] == "Weekly records"
    assert [column["name"] for column in body["columns"]] == ["Record URL", "Answer"]
    # The preview the UI draws, rather than the whole file.
    assert len(body["sample"]) == 2


async def test_a_file_that_cannot_be_read_is_refused_with_a_reason(client: TestClient):
    response = upload(client, b"", "empty.csv")
    assert response.status_code == 422
    assert "empty" in response.json()["detail"]


async def test_an_unsupported_file_type_says_what_is_supported(client: TestClient):
    response = upload(client, b"%PDF-1.4 nonsense", "records.pdf")
    assert response.status_code == 422
    assert "CSV" in response.json()["detail"]


async def test_datasets_are_listed_without_their_rows(client: TestClient):
    upload(client)
    [listed] = client.get("/api/datasets").json()["datasets"]

    assert listed["row_count"] == 2
    assert listed["sample"] == [], "a listing must not ship every row of every file"


async def test_a_dataset_can_be_fetched_with_a_bigger_preview(client: TestClient):
    dataset_id = upload(client).json()["dataset_id"]
    body = client.get(f"/api/datasets/{dataset_id}?sample=1").json()
    assert len(body["sample"]) == 1


async def test_a_missing_dataset_is_a_404(client: TestClient):
    assert client.get("/api/datasets/nope").status_code == 404
    assert client.delete("/api/datasets/nope").status_code == 404


async def test_a_dataset_can_be_deleted(client: TestClient):
    dataset_id = upload(client).json()["dataset_id"]
    assert client.delete(f"/api/datasets/{dataset_id}").status_code == 200
    assert client.get("/api/datasets").json()["datasets"] == []


# --- mapping ---------------------------------------------------------------


async def test_mapping_suggests_a_column_for_each_declared_input(client: TestClient):
    usecase_id = await seed(client)
    dataset_id = upload(client).json()["dataset_id"]

    body = client.post(
        f"/api/usecases/{usecase_id}/mapping", json={"dataset_id": dataset_id}
    ).json()

    chosen = {s["field"]: s["column"] for s in body["suggestions"]}
    assert chosen["record_url"] == "Record URL"
    assert all(s["reason"] for s in body["suggestions"]), "every guess explains itself"


async def test_a_clean_file_needs_nothing_from_a_model(client: TestClient):
    """The cost claim, asserted rather than described.

    Well-named columns resolve on string handling alone, so `unresolved` is
    empty and there is nothing to spend tokens on.
    """
    usecase_id = await seed(client)
    dataset_id = upload(client).json()["dataset_id"]

    body = client.post(
        f"/api/usecases/{usecase_id}/mapping", json={"dataset_id": dataset_id}
    ).json()
    assert body["unresolved"] == []


async def test_mapping_against_a_missing_dataset_is_a_404(client: TestClient):
    usecase_id = await seed(client)
    response = client.post(
        f"/api/usecases/{usecase_id}/mapping", json={"dataset_id": "nope"}
    )
    assert response.status_code == 404


# --- running from a dataset ------------------------------------------------


async def test_a_batch_runs_from_a_dataset_and_a_mapping(client: TestClient):
    """The whole point of the step: the spreadsheet's words, not the use
    case's, and the mapping is what joins them."""
    usecase_id = await seed(client)
    dataset_id = upload(client).json()["dataset_id"]
    credential_id = credential(client)

    started = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={
            "dataset_id": dataset_id,
            "mapping": {"record_url": "Record URL"},
            "credential_id": credential_id,
        },
    )
    assert started.status_code == 202
    body = started.json()
    assert body["total"] == 2
    # Downstream everything speaks in declared field names.
    assert body["columns"] == ["record_url"]

    from test_api_batch import wait_for_batch

    await wait_for_batch(client, body["batch_id"])


async def test_a_batch_records_which_dataset_it_came_from(client: TestClient):
    from conftest import app_workspace

    usecase_id = await seed(client)
    dataset_id = upload(client).json()["dataset_id"]
    started = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={
            "dataset_id": dataset_id,
            "mapping": {"record_url": "Record URL"},
            "credential_id": credential(client),
        },
    ).json()

    data = await app_workspace(client.app)
    batch = await data.get_batch(started["batch_id"])
    assert batch["dataset_id"] == dataset_id

    from test_api_batch import wait_for_batch

    await wait_for_batch(client, started["batch_id"])


async def test_deleting_a_dataset_does_not_rewrite_a_batch_that_ran_from_it(
    client: TestClient,
):
    """A batch copies its rows. History has to stay readable after somebody
    tidies up their uploads."""
    from conftest import app_workspace

    usecase_id = await seed(client)
    dataset_id = upload(client).json()["dataset_id"]
    started = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={
            "dataset_id": dataset_id,
            "mapping": {"record_url": "Record URL"},
            "credential_id": credential(client),
        },
    ).json()

    from test_api_batch import wait_for_batch

    await wait_for_batch(client, started["batch_id"])
    client.delete(f"/api/datasets/{dataset_id}")

    data = await app_workspace(client.app)
    assert len(await data.get_batch_rows(started["batch_id"])) == 2


async def test_a_mapping_naming_a_column_that_is_not_there_is_refused(client: TestClient):
    usecase_id = await seed(client)
    dataset_id = upload(client).json()["dataset_id"]

    response = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={"dataset_id": dataset_id, "mapping": {"record_url": "Invented"}},
    )
    assert response.status_code == 422
    assert "Invented" in response.json()["detail"]


async def test_unmapped_columns_do_not_reach_validation(client: TestClient):
    """Without the mapping dropping them, "Answer" arrives as an undeclared
    input and the user is sent looking for a problem they already solved."""
    usecase_id = await seed(client)
    dataset_id = upload(client).json()["dataset_id"]

    response = client.post(
        f"/api/usecases/{usecase_id}/batch",
        json={
            "dataset_id": dataset_id,
            "mapping": {"record_url": "Record URL"},
            "credential_id": credential(client),
        },
    )
    assert response.status_code == 202

    from test_api_batch import wait_for_batch

    await wait_for_batch(client, response.json()["batch_id"])


async def test_an_unmapped_dataset_must_already_use_the_field_names(client: TestClient):
    usecase_id = await seed(client)
    dataset_id = upload(client).json()["dataset_id"]

    response = client.post(
        f"/api/usecases/{usecase_id}/batch", json={"dataset_id": dataset_id}
    )
    assert response.status_code == 422
    assert "Record URL" in str(response.json()["detail"])
