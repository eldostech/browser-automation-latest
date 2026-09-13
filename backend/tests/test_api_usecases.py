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


# ---------------------------------------------------------------------------
# Editing a locator, and checking it before saving
# ---------------------------------------------------------------------------


def editable(**overrides: Any) -> dict[str, Any]:
    """A use case whose locators a reviewer would want to change."""
    definition = {
        "id": "1a2b3c4d5e6f70819a2b3c4d5e6f7081",
        "name": "Edit a customer",
        "status": "draft",
        "base_url": "https://example.com",
        "allowed_domains": ["example.com"],
        "row_steps": [
            {"id": "s1", "action": "navigate", "url": "https://example.com/customers"},
            {
                "id": "s2",
                "action": "click",
                "locators": [{"strategy": "role", "role": "button", "name": "Edit"}],
            },
        ],
    }
    return {**definition, **overrides}


def test_a_locator_can_be_rewritten_as_a_scope_and_saved(client: TestClient):
    """The edit this whole surface exists for.

    A recorded ladder is what codegen happened to write and a healed one is
    what a model picked off the page. Both are usually right and neither is
    always right, and until this existed the only remedy for one wrong rung
    was re-recording the workflow -- throwing away every other step to fix one.
    """
    created = client.post("/api/usecases/import", json=editable())
    assert created.status_code == 201, created.text
    usecase_id = created.json()["usecase_id"]

    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    definition["row_steps"][1]["locators"] = [
        {
            "strategy": "role",
            "role": "button",
            "name": "Edit",
            "exact": True,
            "within": {"strategy": "role", "role": "row", "has_text": "Acme Ltd"},
        }
    ]

    saved = client.put(f"/api/usecases/{usecase_id}", json=definition)
    assert saved.status_code == 201, saved.text

    after = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    rung = after["row_steps"][1]["locators"][0]
    assert rung["within"]["has_text"] == "Acme Ltd"
    assert rung["exact"] is True


def test_editing_a_locator_writes_a_new_version_rather_than_changing_the_old_one(
    client: TestClient,
):
    """A batch already running is reading from a specific version, and must not
    have it changed underneath it."""
    created = client.post("/api/usecases/import", json=editable())
    usecase_id = created.json()["usecase_id"]
    first = created.json()["version"]

    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    definition["row_steps"][1]["locators"] = [
        {"strategy": "css", "selector": "tr.acme button"}
    ]
    saved = client.put(f"/api/usecases/{usecase_id}", json=definition)

    assert saved.json()["version"] > first


def test_a_locator_check_outside_the_allowlist_is_refused_before_a_browser_opens(
    client: TestClient,
):
    """The same gate a run passes, for the same reason: this endpoint opens a
    browser and visits a URL somebody typed into a form."""
    created = client.post("/api/usecases/import", json=editable())
    usecase_id = created.json()["usecase_id"]

    response = client.post(
        f"/api/usecases/{usecase_id}/locator-check",
        json={
            "url": "https://somewhere-else.test/page",
            "locators": [{"strategy": "role", "role": "button", "name": "Edit"}],
        },
    )

    assert response.status_code == 400
    assert "somewhere-else.test" in response.text


def test_a_malformed_locator_is_named_rather_than_rejected_wholesale(
    client: TestClient,
):
    """The editor has to map the complaint back to a field, so the reply says
    which rung -- not "422" against a body it cannot read."""
    created = client.post("/api/usecases/import", json=editable())
    usecase_id = created.json()["usecase_id"]

    response = client.post(
        f"/api/usecases/{usecase_id}/locator-check",
        json={
            "url": "https://example.com/customers",
            "locators": [
                {"strategy": "role", "role": "button", "name": "Edit"},
                {"strategy": "css"},
            ],
        },
    )

    assert response.status_code == 422
    assert "locator 1" in response.text


def test_a_scope_deeper_than_the_schema_allows_is_refused_on_save(client: TestClient):
    """The depth bound is on the model, so it holds for a hand-edited document
    exactly as it does for a recorded one."""
    created = client.post("/api/usecases/import", json=editable())
    usecase_id = created.json()["usecase_id"]

    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    scope: dict[str, Any] = {"strategy": "role", "role": "main"}
    for _ in range(4):
        scope = {"strategy": "role", "role": "group", "within": scope}
    definition["row_steps"][1]["locators"] = [
        {"strategy": "role", "role": "button", "name": "Edit", "within": scope}
    ]

    response = client.put(f"/api/usecases/{usecase_id}", json=definition)
    assert response.status_code == 422


# --- describing what a use case does ---------------------------------------
#
# The same pass an agent session runs at the end of its own recording, on
# demand. It exists for the two cases that one does not reach: a codegen
# recording, which has no model in it and therefore no account of itself, and
# anything recorded before this existed.


class WalkthroughLLM:
    """Answers a describe request, and counts the calls."""

    model = "fake"

    def __init__(self, payload: dict | None) -> None:
        self.payload = payload
        self.calls = 0

    async def run_turn(self, *, system, messages, tools, on_text_delta=None, timeout=None):
        self.calls += 1
        if self.payload is None:
            return LLMTurn(text="I would rather not")
        return LLMTurn(
            tool_calls=[ToolCallRequest(id="t1", name="walkthrough", input=self.payload)],
            stop_reason="tool_use",
        )


WALKTHROUGH = {
    "overview": "Signs in once, then opens the record each row names.",
    "steps": [{"id": "s1", "purpose": "opens the record the row names"}],
}


def use_walkthrough_llm(client: TestClient, payload: dict | None = WALKTHROUGH) -> WalkthroughLLM:
    llm = WalkthroughLLM(payload)
    client.app.state.repair_model._client = llm  # noqa: SLF001 - test seam
    return llm


async def test_a_use_case_can_be_described_in_plain_language(client: TestClient):
    usecase_id = await seed_usecase(client)
    use_walkthrough_llm(client)

    response = client.post(f"/api/usecases/{usecase_id}/describe", json={})

    assert response.status_code == 201
    assert "Signs in once" in response.json()["instructions"]
    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    assert "Signs in once" in definition["instructions"]


async def test_a_described_step_says_what_it_is_for(client: TestClient):
    """The point of the pass. A repair reads this, and a step's description is
    a rendering of its own locator -- which says nothing about why."""
    usecase_id = await seed_usecase(client)
    use_walkthrough_llm(client)

    client.post(f"/api/usecases/{usecase_id}/describe", json={})

    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    assert definition["row_steps"][0]["intent"] == "opens the record the row names"


async def test_describing_a_published_use_case_leaves_it_published(client: TestClient):
    """Unlike a repair, which changes what runs and must land as a draft.
    Forcing a republish to gain a description would mean nobody ever described
    a published use case, which is most of them."""
    usecase_id = await seed_usecase(client)
    client.post(f"/api/usecases/{usecase_id}/publish")
    use_walkthrough_llm(client)

    response = client.post(f"/api/usecases/{usecase_id}/describe", json={})

    assert response.status_code == 201
    assert response.json()["status"] == "ready"
    assert client.get(f"/api/usecases/{usecase_id}").json()["definition"]["status"] == "ready"


async def test_describing_appends_a_version_rather_than_rewriting_one(client: TestClient):
    usecase_id = await seed_usecase(client)
    use_walkthrough_llm(client)

    client.post(f"/api/usecases/{usecase_id}/describe", json={})

    body = client.get(f"/api/usecases/{usecase_id}").json()
    assert [v["version"] for v in body["versions"]] == [2, 1]
    original = client.get(f"/api/usecases/{usecase_id}", params={"version": 1}).json()
    # Absent rather than empty: a stored definition keeps exactly the keys it
    # was saved with, and the seed predates this field.
    assert not original["definition"].get("instructions"), "v1 is immutable"


async def test_a_model_that_declines_changes_nothing_and_says_so(client: TestClient):
    """`write_walkthrough` swallows its own failures so a recording session
    cannot be lost to one. Here there is no session to lose, and somebody is
    waiting on a button."""
    usecase_id = await seed_usecase(client)
    use_walkthrough_llm(client, None)

    response = client.post(f"/api/usecases/{usecase_id}/describe", json={})

    assert response.status_code == 502
    assert "Nothing was changed" in response.json()["detail"]
    body = client.get(f"/api/usecases/{usecase_id}").json()
    assert [v["version"] for v in body["versions"]] == [1]


async def test_describing_an_unknown_use_case_is_a_404(client: TestClient):
    use_walkthrough_llm(client)
    assert client.post("/api/usecases/nope/describe", json={}).status_code == 404


# --- exporting a script -----------------------------------------------------
#
# One way only, and read-only. The document is what TRACE runs and what a
# repair edits, so a file the platform read back would be a second source of
# truth that drifts from the first.


async def test_a_use_case_can_be_exported_as_a_playwright_script(client: TestClient):
    usecase_id = await seed_usecase(client)

    body = client.get(f"/api/usecases/{usecase_id}/export/python").json()

    assert body["filename"] == "sign_in_and_open_a_record.py"
    assert "def setup(page):" in body["script"]
    assert "def do_row(page, row):" in body["script"]
    assert "one-way export" in body["script"]


async def test_exporting_writes_no_version_and_changes_nothing(client: TestClient):
    usecase_id = await seed_usecase(client)

    client.get(f"/api/usecases/{usecase_id}/export/python")

    body = client.get(f"/api/usecases/{usecase_id}").json()
    assert [v["version"] for v in body["versions"]] == [1]


async def test_a_secret_is_exported_as_an_environment_read_never_a_value(client: TestClient):
    """The export is a file that gets committed and pasted into tickets."""
    usecase_id = await seed_usecase(client)

    script = client.get(f"/api/usecases/{usecase_id}/export/python").json()["script"]

    assert 'os.environ["TRACE_SECRET_USERNAME"]' in script


async def test_an_older_version_can_be_exported(client: TestClient):
    """So a script can be regenerated from whatever version is actually
    deployed, rather than only from the newest draft."""
    usecase_id = await seed_usecase(client)
    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    definition["name"] = "Renamed"
    client.put(f"/api/usecases/{usecase_id}", json=definition)

    first = client.get(f"/api/usecases/{usecase_id}/export/python", params={"version": 1}).json()
    latest = client.get(f"/api/usecases/{usecase_id}/export/python").json()

    assert "Sign in and open a record" in first["script"]
    assert "Renamed" in latest["script"]


async def test_exporting_an_unknown_use_case_is_a_404(client: TestClient):
    assert client.get("/api/usecases/nope/export/python").status_code == 404
