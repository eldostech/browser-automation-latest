"""Which site a run points at, and who decides.

The base URL used to come from ``USECASE_ENV``, one map per deployment. That
holds only while every use case in an environment shares a base URL; run
workflows against several sites and it becomes a variable per site, set in the
environment, needing a release to add one. These tests pin the arrangement that
replaced it: the definition names a target, the deployment says where that
target is, and a single run may override both.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from credentials import generate_key
from usecase import TargetMissing, resolve_base_url
from test_api_execute import FakeReplaySession

pytestmark = pytest.mark.anyio


# --- the rule itself -------------------------------------------------------


def test_an_override_beats_everything():
    """For a branch deployment or one customer's tenant, where standing
    configuration would be ceremony for a single run."""
    assert (
        resolve_base_url(
            target="schemora",
            targets={"schemora": "https://uat.schemora.ai"},
            recorded="https://dev.schemora.ai",
            override="https://pr-482.schemora.dev/",
        )
        == "https://pr-482.schemora.dev"
    )


def test_the_target_is_the_ordinary_path():
    assert (
        resolve_base_url(
            target="schemora",
            targets={"schemora": "https://uat.schemora.ai", "bank": "https://uat.bank.test"},
            recorded="https://dev.schemora.ai",
        )
        == "https://uat.schemora.ai"
    )


def test_several_sites_coexist_which_one_map_could_not():
    """The failure that prompted this. One deployment, many sites."""
    targets = {
        "schemora": "https://uat.schemora.ai",
        "bank": "https://uat.bank.test",
        "vendor": "https://vendor-uat.example.com",
    }
    resolved = {
        name: resolve_base_url(target=name, targets=targets, recorded="https://dev.x")
        for name in targets
    }
    assert resolved == {
        "schemora": "https://uat.schemora.ai",
        "bank": "https://uat.bank.test",
        "vendor": "https://vendor-uat.example.com",
    }


def test_naming_no_target_runs_where_it_was_recorded():
    """What lets a single-environment install work with nothing configured."""
    assert (
        resolve_base_url(target="", targets={}, recorded="https://dev.schemora.ai/")
        == "https://dev.schemora.ai"
    )


def test_a_target_this_deployment_cannot_answer_is_refused():
    """Never a silent fallback.

    Dropping to the recorded URL would send a use case promoted to production
    at whatever host it happened to be recorded against -- quietly, on a run
    somebody had every reason to trust. The message names what is missing and
    what is defined.
    """
    with pytest.raises(TargetMissing) as caught:
        resolve_base_url(
            target="bank",
            targets={"schemora": "https://uat.schemora.ai"},
            recorded="https://dev.bank.test",
        )

    message = str(caught.value)
    assert "'bank'" in message
    assert "schemora" in message, "it says what this deployment does have"
    assert "dev.bank.test" not in message, "and never suggests the recorded host"


# --- through the API -------------------------------------------------------


@pytest.fixture
def client(db_settings, db_engine, tmp_path, monkeypatch):
    from conftest import authenticate, build_app

    app = build_app(
        db_settings, tmp_path, monkeypatch,
        session_cls=FakeReplaySession, credentials_key=generate_key(),
    )
    with TestClient(app) as test_client:
        yield authenticate(test_client)


async def test_targets_are_edited_rather_than_deployed(client: TestClient):
    """The point of the change: onboarding a site is a row, not a release."""
    created = client.put(
        "/api/targets/schemora",
        json={"base_url": "https://uat.schemora.ai", "description": "The product"},
    )
    assert created.status_code == 201, created.text

    client.put("/api/targets/bank", json={"base_url": "https://uat.bank.test"})

    listed = client.get("/api/targets").json()["targets"]
    assert {t["name"]: t["base_url"] for t in listed} == {
        "schemora": "https://uat.schemora.ai",
        "bank": "https://uat.bank.test",
    }


async def test_saving_the_same_name_moves_it_rather_than_duplicating(client: TestClient):
    """A target is identified by what a use case calls it."""
    client.put("/api/targets/schemora", json={"base_url": "https://uat.schemora.ai"})
    client.put("/api/targets/schemora", json={"base_url": "https://uat2.schemora.ai"})

    listed = client.get("/api/targets").json()["targets"]
    assert len(listed) == 1
    assert listed[0]["base_url"] == "https://uat2.schemora.ai"


async def test_a_target_is_an_origin_not_a_page(client: TestClient):
    """A path here would be prefixed onto every step's own path."""
    response = client.put(
        "/api/targets/schemora", json={"base_url": "https://uat.schemora.ai/login"}
    )

    assert response.status_code == 422
    assert "origin, not a page" in response.json()["detail"]


async def test_a_target_must_be_an_absolute_url(client: TestClient):
    response = client.put("/api/targets/schemora", json={"base_url": "uat.schemora.ai"})
    assert response.status_code == 422
    assert "absolute" in response.json()["detail"]


async def test_deleting_a_target_does_not_repoint_the_use_cases_naming_it(
    client: TestClient,
):
    """They fail loudly on the next run instead."""
    client.put("/api/targets/schemora", json={"base_url": "https://uat.schemora.ai"})
    assert client.delete("/api/targets/schemora").status_code == 200
    assert client.get("/api/targets").json()["targets"] == []
    assert client.delete("/api/targets/schemora").status_code == 404


async def test_one_use_case_reaches_two_different_sites(client: TestClient):
    """End to end, and the whole point.

    The same definition, run twice: once against the site its target names, and
    once against an address given for that run alone. Neither is written into
    the document, so promoting it carries no address at all.
    """
    from conftest import app_workspace
    from test_api_execute import BROKEN

    store = await app_workspace(client.app)
    definition = {
        **BROKEN,
        "target": "schemora",
        "base_url": "https://dev.schemora.ai",
        "allowed_domains": ["*"],
        "row_steps": [
            {"id": "s0", "action": "navigate", "url": "{{env.base_url}}/orders"},
            *BROKEN["row_steps"],
        ],
    }
    usecase_id, _ = await store.save_usecase(definition)
    client.put("/api/targets/schemora", json={"base_url": "https://uat.schemora.ai"})

    def run(**extra):
        return client.post(
            f"/api/usecases/{usecase_id}/execute",
            json={
                "inputs": {"record_url": "https://example.com/r"},
                "secrets": {"username": "u", "password": "p"},
                **extra,
            },
        )

    def orders_url() -> str:
        # The row step's own navigation. `visited[0]` is the setup sign-in,
        # which is not the URL under test.
        visited = FakeReplaySession.current.page.visited
        return next(url for url in visited if url.endswith("/orders"))

    assert run().status_code == 201
    visited_via_target = orders_url()

    assert run(base_url="https://pr-482.schemora.dev").status_code == 201
    visited_via_override = orders_url()

    assert visited_via_target == "https://uat.schemora.ai/orders"
    assert visited_via_override == "https://pr-482.schemora.dev/orders"


async def test_a_run_naming_an_unknown_target_is_refused_before_it_starts(
    client: TestClient,
):
    """Said to whoever pressed the button, not buried in a run's error."""
    from conftest import app_workspace
    from test_api_execute import BROKEN

    store = await app_workspace(client.app)
    usecase_id, _ = await store.save_usecase({**BROKEN, "target": "bank"})

    response = client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={
            "inputs": {"record_url": "https://example.com/r"},
            "secrets": {"username": "u", "password": "p"},
        },
    )

    assert response.status_code == 422
    assert "'bank'" in response.json()["detail"]


# --- getting unstuck -------------------------------------------------------


def test_a_recording_made_on_localhost_names_no_target():
    """Otherwise every locally recorded use case refuses to run.

    "localhost" is where the browser was, not what it was looking at. Naming a
    target after it means the use case will not run against the machine it was
    just recorded on until somebody defines a target called localhost.
    """
    from routers.recordings import _target_name

    assert _target_name("http://localhost:8002") == ""
    assert _target_name("http://127.0.0.1:5173") == ""
    assert _target_name("http://app.localhost:3000") == ""
    # A real site still gets a name.
    assert _target_name("https://uat.schemora.ai") == "schemora"


def test_the_refusal_says_both_ways_out():
    """A message that names the problem and not the remedy leaves you stuck."""
    with pytest.raises(TargetMissing) as caught:
        resolve_base_url(
            target="localhost",
            targets={"schemora-local": "http://localhost:8002"},
            recorded="http://localhost:8002",
        )

    message = str(caught.value)
    assert "'localhost'" in message
    assert "schemora-local" in message, "it lists what this deployment does have"
    assert "under Targets" in message, "one way out: define it"
    assert "beside Pace" in message, "the other: point the use case at an existing one"


async def test_a_use_case_can_be_pointed_at_a_different_target(client: TestClient):
    """The thing that was impossible: the target was in the definition with no
    endpoint or screen that could change it."""
    from conftest import app_workspace
    from test_api_execute import BROKEN

    store = await app_workspace(client.app)
    usecase_id, _ = await store.save_usecase({**BROKEN, "target": "localhost"})
    client.put("/api/targets/schemora-local", json={"base_url": "http://localhost:8002"})

    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    saved = client.put(
        f"/api/usecases/{usecase_id}", json={**definition, "target": "schemora-local"}
    )

    assert saved.status_code == 201, saved.text
    body = client.get(f"/api/usecases/{usecase_id}").json()
    assert body["definition"]["target"] == "schemora-local"
    assert body["meta"]["target"] == "schemora-local", (
        "the summary row mirrors it, so the list can show it without loading "
        "every definition"
    )
