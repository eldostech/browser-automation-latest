"""The use-case HTTP surface: distil a run, review it, publish it.

Uses the real FastAPI app with a scripted LLM, so the "exactly one model call"
guarantee is exercised through the endpoint rather than only in unit tests.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from conftest import ScriptedLLM
from llm import LLMTurn, ToolCallRequest

# Reuse the app fixture wiring from the main API tests.
from test_api import FakeBrowser

SNAPSHOT = """### Page
- Page URL: https://example.com/signin
### Snapshot
```yaml
- textbox "Username" [ref=e1]
- textbox "Password" [ref=e2]
- button "Sign in" [ref=e3]
```"""

# Step ids are sequential over the SURVIVING steps, so the dropped snapshot
# leaves no gap: navigate=s1, fill_form=s2, click=s3, navigate=s4.
PLAN = {
    "name": "Sign in and open a record",
    "description": "Signs in once, then opens one record per row.",
    "inputs": [{"name": "record_url", "type": "url"}],
    "secrets": [{"name": "username"}, {"name": "password"}],
    "setup_step_ids": ["s1", "s2", "s3"],
    "row_step_ids": ["s4"],
    "values": {"s2.Username": "{{secret.username}}", "s2.Password": "{{secret.password}}"},
    "urls": {"s4": "{{input.record_url}}"},
    "assertions": [
        {"after_step_id": "s3", "kind": "url_contains", "value": "/signin", "negate": True}
    ],
    "session_check": {"kind": "url_contains", "value": "/signin", "negate": True},
    "row_reset_url": "{{input.record_url}}",
}


@pytest.fixture
def client(db_settings, db_engine, tmp_path, monkeypatch):
    from conftest import authenticate, build_app

    app = build_app(db_settings, tmp_path, monkeypatch, session_cls=FakeBrowser)
    with TestClient(app) as test_client:
        yield authenticate(test_client)


class PlanLLM:
    """Answers a distillation request with a fixed plan; counts the calls."""

    model = "fake"

    def __init__(self, plan: dict | None = PLAN) -> None:
        self.plan = plan
        self.calls = 0

    async def run_turn(self, *, system, messages, tools, on_text_delta=None, timeout=None):
        self.calls += 1
        if self.plan is None:
            return LLMTurn(text="no idea")
        return LLMTurn(
            tool_calls=[ToolCallRequest(id="t1", name="build_usecase", input=self.plan)],
            stop_reason="tool_use",
        )


async def seed_run(client: TestClient, *, status: str = "succeeded") -> str:
    """Write a finished run with a recording straight into the store."""
    from events import ToolCall, ToolResult

    from conftest import app_workspace

    store = await app_workspace(client.app)
    run_id = "run-seed"
    await store.create_run(run_id, "sign in and open a record", None, {})

    seq = 0

    async def pair(tool: str, arguments: dict[str, Any], ok: bool, text: str, step: int) -> None:
        nonlocal seq
        seq += 1
        await store.append_event(
            ToolCall(run_id=run_id, seq=seq, step=step, call_id=f"c{step}", name=tool,
                     arguments=arguments)
        )
        seq += 1
        await store.append_event(
            ToolResult(run_id=run_id, seq=seq, step=step, call_id=f"c{step}", name=tool,
                       ok=ok, duration_ms=1, text=text)
        )

    await pair("browser_navigate", {"url": "https://example.com/signin"}, True, "ok", 1)
    await pair("browser_snapshot", {}, True, SNAPSHOT, 2)
    await pair(
        "browser_fill_form",
        {"fields": [
            {"target": "ref=e1", "name": "Username", "value": "someone"},
            {"target": "ref=e2", "name": "Password", "value": "«redacted»"},
        ]},
        True, "filled", 3,
    )
    await pair("browser_click", {"target": "ref=e3", "element": "Sign in button"}, True, "ok", 4)
    await pair("browser_navigate", {"url": "https://example.com/record/1"}, True, "ok", 5)

    await store.finish_run(run_id, status, steps=5, duration_ms=100, summary="done")
    return run_id


async def seed_usecase(client: TestClient, **overrides) -> str:
    """A draft use case in the store, without going through a recording.

    These tests are about what happens to a use case *after* it exists --
    editing it, publishing it, renaming it, archiving it. How it came to exist
    is the recorder's business and is tested there.
    """
    from conftest import app_workspace

    store = await app_workspace(client.app)
    definition = {
        "id": "uc-seed",
        "name": "Sign in and open a record",
        "status": "draft",
        "allowed_domains": ["example.com"],
        "inputs": [{"name": "record_url", "type": "url", "required": True}],
        "secrets": [{"name": "username"}, {"name": "password"}],
        "setup_steps": [
            {"id": "u1", "action": "navigate", "url": "https://example.com/signin"},
            {
                "id": "u2",
                "action": "fill",
                "value": "{{secret.username}}",
                "locators": [{"strategy": "label", "text": "Username"}],
            },
        ],
        "row_steps": [
            {"id": "s1", "action": "navigate", "url": "{{input.record_url}}"},
        ],
        **overrides,
    }
    usecase_id, _ = await store.save_usecase(definition)
    return usecase_id


def use_plan_llm(client: TestClient, plan: dict | None = PLAN) -> PlanLLM:
    llm = PlanLLM(plan)
    client.app.state.repair_model._client = llm  # noqa: SLF001 - test seam
    return llm


# --- distilling ------------------------------------------------------------


async def test_use_cases_are_listed_newest_first(client: TestClient):
    await seed_usecase(client)

    rows = client.get("/api/usecases").json()["usecases"]
    assert len(rows) == 1
    assert rows[0]["status"] == "draft"


async def test_an_edit_appends_a_version_rather_than_rewriting_one(client: TestClient):
    usecase_id = await seed_usecase(client)

    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    definition["name"] = "Renamed by a reviewer"
    updated = client.put(f"/api/usecases/{usecase_id}", json=definition)

    assert updated.status_code == 201
    assert updated.json()["version"] == 2

    body = client.get(f"/api/usecases/{usecase_id}").json()
    assert body["definition"]["name"] == "Renamed by a reviewer"
    assert [v["version"] for v in body["versions"]] == [2, 1]

    original = client.get(f"/api/usecases/{usecase_id}", params={"version": 1}).json()
    assert original["definition"]["name"] == "Sign in and open a record", "v1 is immutable"


async def test_an_invalid_edit_is_rejected_with_the_reason(client: TestClient):
    usecase_id = await seed_usecase(client)

    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    definition["row_steps"][0]["url"] = "{{input.undeclared}}"

    response = client.put(f"/api/usecases/{usecase_id}", json=definition)
    assert response.status_code == 422
    assert "not declared" in response.json()["detail"]


async def test_editing_an_unknown_use_case_is_a_404(client: TestClient):
    assert client.put("/api/usecases/nope", json={"name": "x"}).status_code == 404


# --- publishing ------------------------------------------------------------


async def test_publishing_moves_a_draft_to_ready(client: TestClient):
    usecase_id = await seed_usecase(client)

    assert client.post(f"/api/usecases/{usecase_id}/publish").json()["status"] == "ready"
    assert client.get(f"/api/usecases/{usecase_id}").json()["definition"]["status"] == "ready"


async def test_a_use_case_with_scripts_cannot_be_published_without_the_opt_in(client: TestClient):
    usecase_id = await seed_usecase(client)

    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    definition["row_steps"].append(
        {"id": "danger", "action": "script", "code": "await page.evaluate('1')"}
    )
    assert client.put(f"/api/usecases/{usecase_id}", json=definition).status_code == 201

    response = client.post(f"/api/usecases/{usecase_id}/publish")
    assert response.status_code == 422
    assert "allow_scripts" in response.json()["detail"]

    # A reviewer who reads the code and opts in can then publish.
    definition["allow_scripts"] = True
    client.put(f"/api/usecases/{usecase_id}", json=definition)
    assert client.post(f"/api/usecases/{usecase_id}/publish").status_code == 200


async def test_archiving_keeps_the_record(client: TestClient):
    usecase_id = await seed_usecase(client)

    assert client.delete(f"/api/usecases/{usecase_id}").json()["status"] == "archived"
    assert client.get(f"/api/usecases/{usecase_id}").status_code == 200, "history still resolves"


async def test_publishing_an_unknown_use_case_is_a_404(client: TestClient):
    assert client.post("/api/usecases/nope/publish").status_code == 404


# --- archiving vs deleting -------------------------------------------------


async def test_archiving_is_reversible_and_keeps_the_record(client: TestClient):
    usecase_id = await seed_usecase(client)

    assert client.delete(f"/api/usecases/{usecase_id}").json()["status"] == "archived"
    assert client.get(f"/api/usecases/{usecase_id}").status_code == 200


async def test_purging_removes_the_use_case_for_good(client: TestClient):
    usecase_id = await seed_usecase(client)

    response = client.delete(f"/api/usecases/{usecase_id}", params={"purge": "true"})

    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert response.json()["removed"]["usecases"] == 1
    assert client.get(f"/api/usecases/{usecase_id}").status_code == 404
    assert client.get("/api/usecases").json()["usecases"] == []


async def test_purging_keeps_the_runs_it_produced(client: TestClient):
    """The timeline records what happened to a browser; deleting a recipe must
    not erase the history of things it actually did."""
    run_id = await seed_run(client)
    usecase_id = await seed_usecase(client)

    client.delete(f"/api/usecases/{usecase_id}", params={"purge": "true"})

    assert client.get(f"/api/runs/{run_id}").status_code == 200
    assert client.get(f"/api/runs/{run_id}/events").json()["events"]


async def test_purging_an_unknown_use_case_is_a_404(client: TestClient):
    assert client.delete("/api/usecases/nope", params={"purge": "true"}).status_code == 404


# --- naming ----------------------------------------------------------------
#
# The model's suggestion can only exist after distillation has read the
# recording, so the name is offered at that point and can be changed then or
# later. A rename is a label change, not a change to what runs.


async def test_renaming_changes_the_label_without_creating_a_version(client: TestClient):
    usecase_id = await seed_usecase(client)

    response = client.patch(f"/api/usecases/{usecase_id}", json={"name": "Nightly demo requests"})

    assert response.status_code == 200
    body = client.get(f"/api/usecases/{usecase_id}").json()
    assert body["definition"]["name"] == "Nightly demo requests"
    assert body["definition"]["version"] == 1, "a label change is not a new version"
    assert [v["version"] for v in body["versions"]] == [1]


async def test_a_rename_shows_up_in_the_list(client: TestClient):
    usecase_id = await seed_usecase(client)

    client.patch(f"/api/usecases/{usecase_id}", json={"name": "Renamed"})

    assert client.get("/api/usecases").json()["usecases"][0]["name"] == "Renamed"


async def test_renaming_leaves_the_steps_alone(client: TestClient):
    usecase_id = await seed_usecase(client)
    before = client.get(f"/api/usecases/{usecase_id}").json()["definition"]

    client.patch(f"/api/usecases/{usecase_id}", json={"name": "Something else"})
    after = client.get(f"/api/usecases/{usecase_id}").json()["definition"]

    assert after["row_steps"] == before["row_steps"]
    assert after["setup_steps"] == before["setup_steps"]
    assert after["status"] == before["status"]


async def test_a_description_can_be_changed_too(client: TestClient):
    usecase_id = await seed_usecase(client)

    client.patch(
        f"/api/usecases/{usecase_id}",
        json={"name": "Kept", "description": "One row per prospect."},
    )
    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    assert definition["description"] == "One row per prospect."


async def test_renaming_a_published_use_case_does_not_unpublish_it(client: TestClient):
    """A label change must not send something back for review."""
    usecase_id = await seed_usecase(client)
    client.post(f"/api/usecases/{usecase_id}/publish")

    client.patch(f"/api/usecases/{usecase_id}", json={"name": "Still ready"})

    assert client.get(f"/api/usecases/{usecase_id}").json()["definition"]["status"] == "ready"


@pytest.mark.parametrize("name", ["", "   ", "x" * 201])
async def test_an_unusable_name_is_refused(client: TestClient, name: str):
    usecase_id = await seed_usecase(client)

    response = client.patch(f"/api/usecases/{usecase_id}", json={"name": name})
    assert response.status_code == 422


async def test_renaming_an_unknown_use_case_is_a_404(client: TestClient):
    assert client.patch("/api/usecases/nope", json={"name": "x"}).status_code == 404


# ---------------------------------------------------------------------------
# Promotion: moving one document between environments
# ---------------------------------------------------------------------------


def promotable(**overrides: Any) -> dict[str, Any]:
    """A definition of the shape an export from dev produces."""
    definition = {
        "id": "9f3c2a1b4d5e6f708192a3b4c5d6e7f8",
        "name": "Create a project",
        "status": "ready",
        "base_url": "https://dev.example.com",
        "allowed_domains": ["{{env.base_url}}"],
        "inputs": [{"name": "reference"}],
        "secrets": [{"name": "password"}],
        "row_steps": [
            {"id": "s1", "action": "navigate", "url": "{{env.base_url}}/orders"},
            {
                "id": "s2",
                "action": "fill",
                "value": "{{input.reference}}",
                "locators": [{"strategy": "label", "text": "Reference"}],
            },
        ],
    }
    return {**definition, **overrides}


def test_a_definition_from_another_environment_can_be_imported(client: TestClient):
    """The gap that made promotion impossible.

    ``PUT /usecases/{id}`` calls ``usecase_or_404`` first, so it can only update
    something that already exists, and the only path that created a use case was
    saving a recording. A document that would run unchanged in UAT had no way to
    get there short of re-recording it -- which produces a different document --
    or writing to the database by hand.
    """
    response = client.post("/api/usecases/import", json=promotable())
    assert response.status_code == 201, response.text
    body = response.json()

    assert body["usecase_id"] == "9f3c2a1b4d5e6f708192a3b4c5d6e7f8", (
        "the id is preserved, so one use case is the same use case in every "
        "environment and its runs can be lined up"
    )
    assert body["version"] == 1

    stored = client.get(f"/api/usecases/{body['usecase_id']}").json()["definition"]
    assert stored["base_url"] == "https://dev.example.com"
    assert stored["allowed_domains"] == ["{{env.base_url}}"]
    assert [s["url"] for s in stored["row_steps"] if s["action"] == "navigate"] == [
        "{{env.base_url}}/orders"
    ]


def test_an_import_always_arrives_as_a_draft(client: TestClient):
    """An approval given in dev is not an approval in production."""
    response = client.post("/api/usecases/import", json=promotable(status="ready"))

    assert response.status_code == 201, response.text
    assert response.json()["status"] == "draft"


def test_an_import_never_carries_permission_to_run_javascript(client: TestClient):
    """The flag that matters most, and the one easiest to carry across silently.

    ``allow_scripts`` lets a use case run arbitrary JavaScript against a live
    page, and it is granted by a person who has read that code. Carrying it
    would let code approved against dev's data execute against production's,
    which nobody in production agreed to.
    """
    response = client.post(
        "/api/usecases/import", json=promotable(status="ready", allow_scripts=True)
    )
    assert response.status_code == 201, response.text

    stored = client.get(f"/api/usecases/{response.json()['usecase_id']}").json()
    assert stored["definition"]["allow_scripts"] is False


def test_re_importing_appends_a_version_rather_than_duplicating(client: TestClient):
    """Promoting a revision is the same gesture as promoting it the first time."""
    first = client.post("/api/usecases/import", json=promotable())
    assert first.status_code == 201, first.text

    revised = promotable()
    revised["name"] = "Create a project (revised)"
    second = client.post("/api/usecases/import", json=revised)

    assert second.status_code == 201, second.text
    assert second.json()["usecase_id"] == first.json()["usecase_id"]
    assert second.json()["version"] == 2

    listed = client.get("/api/usecases").json()["usecases"]
    assert len([u for u in listed if u["id"] == first.json()["usecase_id"]]) == 1


def test_a_run_id_from_the_source_environment_is_dropped(client: TestClient):
    """It names a run in another database, so it resolves to nothing here."""
    response = client.post(
        "/api/usecases/import",
        json=promotable(source_run_id="a-run-that-only-exists-in-dev"),
    )
    assert response.status_code == 201, response.text

    stored = client.get(f"/api/usecases/{response.json()['usecase_id']}").json()
    assert not stored["definition"].get("source_run_id")


def test_a_definition_this_environment_cannot_validate_is_refused(client: TestClient):
    """Said verbatim: it is the most useful thing to show whoever is promoting."""
    broken = promotable()
    broken["row_steps"][1]["locators"] = []

    response = client.post("/api/usecases/import", json=broken)

    assert response.status_code == 422
    assert "locator" in response.json()["detail"]


def test_importing_is_recorded_against_whoever_did_it(client: TestClient):
    """Each environment's approval trail is its own."""
    response = client.post("/api/usecases/import", json=promotable())
    assert response.status_code == 201, response.text

    activity = client.get(
        f"/api/usecases/{response.json()['usecase_id']}/activity"
    ).json()
    actions = [entry["action"] for entry in activity["entries"]]
    assert "usecase.import" in actions
