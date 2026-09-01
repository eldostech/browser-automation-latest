"""The step trail, and what it is compared against.

A recording made by `playwright codegen` has no screenshots of its own --
codegen owns that browser and we never see its pages -- so there is nothing
from "the day it was recorded" to diff against. The baseline is instead the
last run of the same version that *worked*, which is arguably the more useful
question: not "does this match the recording" but "what changed since it last
worked".
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from credentials import generate_key
from test_api_execute import USE_CASE, ExplodingLLM, FakeReplaySession

pytestmark = pytest.mark.anyio


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
        # Every step photographed, so there is something to compare.
        replay_screenshots="every_step",
    )
    with TestClient(app) as test_client:
        test_client.app.state.repair_model._client = ExplodingLLM()  # noqa: SLF001 - test seam
        yield authenticate(test_client)


async def seed(client: TestClient) -> str:
    from conftest import app_workspace

    data = await app_workspace(client.app)
    usecase_id, _ = await data.save_usecase({**USE_CASE, "id": "uc-1"})
    return usecase_id


def credential(client: TestClient) -> str:
    return client.post(
        "/api/credentials",
        json={"name": "Example", "values": {"username": "u", "password": "pw"}},
    ).json()["id"]


def execute(client: TestClient, usecase_id: str) -> dict:
    return client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={
            "inputs": {"record_url": "https://example.com/record/1"},
            "credential_id": credential(client),
        },
    ).json()


# --- the trail -------------------------------------------------------------


async def test_every_step_is_recorded_as_a_row(client: TestClient):
    usecase_id = await seed(client)
    result = execute(client, usecase_id)

    body = client.get(f"/api/runs/{result['run_id']}/steps").json()
    steps = body["steps"]

    assert steps, "an execution must leave a step trail"
    assert [s["step_id"] for s in steps] == sorted(
        [s["step_id"] for s in steps], key=lambda _: 0
    ), "steps come back in the order they ran"
    assert all(s["status"] in {"succeeded", "failed", "skipped", "healed"} for s in steps)
    assert all(s["action"] for s in steps)


async def test_a_step_carries_the_locator_that_matched(client: TestClient):
    """Which rung matched is how drift shows up before it becomes breakage."""
    usecase_id = await seed(client)
    result = execute(client, usecase_id)

    steps = client.get(f"/api/runs/{result['run_id']}/steps").json()["steps"]
    located = [s for s in steps if s["locator"]]
    assert located, "element steps record what they matched"
    assert all(s["locator_rung"] is not None for s in located)


async def test_a_step_points_at_its_screenshot(client: TestClient):
    usecase_id = await seed(client)
    result = execute(client, usecase_id)

    steps = client.get(f"/api/runs/{result['run_id']}/steps").json()["steps"]
    shot = next(s for s in steps if s["screenshot_url"])
    image = client.get(shot["screenshot_url"])
    assert image.status_code == 200
    assert image.content[:4] == b"\x89PNG"


# --- the baseline ----------------------------------------------------------


async def test_the_first_run_has_nothing_to_compare_against(client: TestClient):
    """`null`, not zero. A first run is not a perfect match with something
    that does not exist."""
    usecase_id = await seed(client)
    result = execute(client, usecase_id)

    steps = client.get(f"/api/runs/{result['run_id']}/steps").json()["steps"]
    assert all(s["baseline_id"] is None for s in steps)
    assert all(s["pixel_diff"] is None for s in steps)
    assert all(s["diff"] == "no baseline" for s in steps)


async def test_a_later_run_is_compared_against_the_last_one_that_worked(client: TestClient):
    usecase_id = await seed(client)
    execute(client, usecase_id)
    second = execute(client, usecase_id)

    steps = client.get(f"/api/runs/{second['run_id']}/steps").json()["steps"]
    compared = [s for s in steps if s["baseline_id"]]

    assert compared, "the second run should have a baseline to compare against"
    # The fake browser serves the same pages both times, so nothing moved.
    assert all(s["pixel_diff"] == 0.0 for s in compared)
    assert all(s["diff"] == "identical" for s in compared)


async def test_a_run_is_never_its_own_baseline(client: TestClient):
    usecase_id = await seed(client)
    result = execute(client, usecase_id)

    steps = client.get(f"/api/runs/{result['run_id']}/steps").json()["steps"]
    assert all(s["baseline_id"] != s["screenshot_id"] for s in steps)


async def test_the_first_divergence_is_named(client: TestClient):
    """So the UI opens where something changed, rather than asking somebody to
    scroll a thousand rows looking for it."""
    usecase_id = await seed(client)
    execute(client, usecase_id)
    second = execute(client, usecase_id)

    body = client.get(f"/api/runs/{second['run_id']}/steps").json()
    # Nothing moved between two identical runs, so there is nothing to open on.
    assert body["first_divergence"] is None


# --- scoping ---------------------------------------------------------------


async def test_steps_of_an_unknown_run_are_a_404(client: TestClient):
    assert client.get("/api/runs/nope/steps").status_code == 404
