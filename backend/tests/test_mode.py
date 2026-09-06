"""How a use case says whether a run may spend a token.

Whether a replay can reach a model used to be ``REPLAY_HEALING_ENABLED``: one
switch, per deployment, that nobody using the product could see. That is the
wrong place for the decision twice over -- it is invisible, and it is a
property of the *workflow* rather than of the installation.

These tests pin the move onto the use case, and in particular the part that is
easy to get wrong: an upgrade must not change how anything already published
behaves. A document written before ``mode`` existed never chose, and recording
that it chose ``strict`` would switch healing off underneath a deployment that
has it on today.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from credentials import generate_key
from usecase import UseCase, effective_mode
from test_api_execute import FakeReplaySession

pytestmark = pytest.mark.anyio


# --- resolving the mode ----------------------------------------------------


def test_a_use_case_that_never_chose_follows_the_deployment():
    """The compatibility case, and the only reason `mode` is nullable."""
    assert effective_mode(None, healing_enabled=True) == "guided"
    assert effective_mode(None, healing_enabled=False) == "strict"


def test_the_deployment_setting_is_a_ceiling_not_a_decision():
    """An operator who turned healing off across an installation meant it.

    A document arriving from another environment must not be able to switch it
    back on, so the setting can only ever restrict.
    """
    assert effective_mode("guided", healing_enabled=False) == "strict"


def test_choosing_strict_wins_over_a_deployment_that_allows_healing():
    """The choice is the use case's whenever the deployment permits one."""
    assert effective_mode("strict", healing_enabled=True) == "strict"


def test_only_the_two_implemented_modes_are_accepted():
    """A mode with nothing behind it is worse than no mode.

    "Explore" belongs to a later phase. Offering it now would take the choice
    and then not honour it.
    """
    assert UseCase(name="x", mode="guided").mode == "guided"
    with pytest.raises(ValueError):
        UseCase(name="x", mode="explore")


# --- what that means for a run --------------------------------------------


def build_manager(tmp_path, **settings_kwargs):
    from config import Settings
    from runner import EventBus, ReplayManager
    from store import Store
    from test_healing import ChoosingLLM

    return ReplayManager(
        Store(tmp_path / "x.db", tmp_path / "a"),
        Settings(_env_file=None, **settings_kwargs),
        EventBus(),
        llm_factory=lambda: ChoosingLLM(),
    )


def test_a_strict_use_case_gets_no_healer_even_where_healing_is_allowed(tmp_path):
    """The point of the whole change: the workflow decides, not the install."""
    manager = build_manager(tmp_path, replay_healing_enabled=True)

    assert manager.make_healer("", "uc", mode="strict") is None
    assert manager.make_healer("", "uc", mode="guided") is not None


def test_an_upgrade_does_not_change_how_a_published_use_case_runs(tmp_path):
    """A definition stored before `mode` existed keeps behaving as it did.

    This is the test that would have caught the tempting version of this
    change -- defaulting the field to "strict" and silently disabling healing
    for every use case in a deployment that had it switched on.
    """
    published_before = {
        "id": "a" * 32,
        "name": "Recorded last month",
        "status": "ready",
        "allowed_domains": ["vendor.test"],
        "row_steps": [{"id": "s1", "action": "navigate", "url": "https://vendor.test/"}],
    }
    use_case = UseCase.model_validate(published_before)
    assert use_case.mode is None, "it never chose, and must not be said to have"

    with_healing = build_manager(tmp_path, replay_healing_enabled=True)
    assert with_healing.make_healer("", use_case.id, mode=use_case.mode) is not None

    without = build_manager(tmp_path, replay_healing_enabled=False)
    assert without.make_healer("", use_case.id, mode=use_case.mode) is None


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


DEFINITION = {
    "name": "Pull statements",
    "status": "ready",
    "allowed_domains": ["vendor.test"],
    "row_steps": [{"id": "s1", "action": "navigate", "url": "https://vendor.test/"}],
}


async def test_choosing_a_mode_is_a_new_version(client: TestClient):
    """It changes what executes, so it belongs in the history like any edit."""
    created = client.post("/api/usecases/import", json={**DEFINITION, "id": "m" * 32})
    assert created.status_code == 201, created.text
    usecase_id = created.json()["usecase_id"]

    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    assert definition["mode"] is None, "a fresh import has not chosen either"

    saved = client.put(
        f"/api/usecases/{usecase_id}", json={**definition, "mode": "guided"}
    )
    assert saved.status_code == 201, saved.text
    assert saved.json()["version"] == definition["version"] + 1

    after = client.get(f"/api/usecases/{usecase_id}").json()
    assert after["definition"]["mode"] == "guided"


async def test_the_list_shows_the_mode_without_reading_definitions(client: TestClient):
    """Mirrored onto the row, like `target`, so a chip costs no extra query."""
    usecase_id = client.post(
        "/api/usecases/import", json={**DEFINITION, "id": "n" * 32, "mode": "guided"}
    ).json()["usecase_id"]

    rows = client.get("/api/usecases").json()["usecases"]
    row = next(r for r in rows if r["id"] == usecase_id)

    assert row["mode"] == "guided"
    assert row["authored_by"] == "person"


async def test_a_use_case_that_never_chose_reports_no_mode_rather_than_a_guess(
    client: TestClient,
):
    """The list has to be able to say "following the deployment"."""
    usecase_id = client.post(
        "/api/usecases/import", json={**DEFINITION, "id": "p" * 32}
    ).json()["usecase_id"]

    row = next(
        r for r in client.get("/api/usecases").json()["usecases"] if r["id"] == usecase_id
    )
    assert row["mode"] is None
