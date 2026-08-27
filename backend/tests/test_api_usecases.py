"""The use-case HTTP surface: distil a run, review it, publish it.

Uses the real FastAPI app with a scripted LLM, so the "exactly one model call"
guarantee is exercised through the endpoint rather than only in unit tests.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from conftest import FakeMCPSession, ScriptedLLM
from llm import LLMTurn, ToolCallRequest

# Reuse the app fixture wiring from the main API tests.
from test_api import FakeMCPBrowserSession, _fake_probe  # noqa: F401

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

    app = build_app(db_settings, tmp_path, monkeypatch, session_cls=FakeMCPBrowserSession)
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


def use_plan_llm(client: TestClient, plan: dict | None = PLAN) -> PlanLLM:
    llm = PlanLLM(plan)
    client.app.state.manager._llm = llm  # noqa: SLF001 - test seam
    return llm


# --- distilling ------------------------------------------------------------


async def test_distilling_a_run_creates_a_draft_use_case(client: TestClient):
    run_id = await seed_run(client)
    llm = use_plan_llm(client)

    response = client.post(f"/api/runs/{run_id}/distill")

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Sign in and open a record"
    assert body["status"] == "draft", "review is mandatory"
    assert body["setup_steps"] == 4, "three recorded steps plus the woven-in assertion"
    assert body["row_steps"] == 1
    assert body["inputs"] == ["record_url"]
    assert sorted(body["secrets"]) == ["password", "username"]
    assert llm.calls == 1, "the entire cost of the feature"


async def test_the_stored_definition_contains_no_ephemeral_refs(client: TestClient):
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]

    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    assert "ref=e" not in str(definition), "a ref would point at the wrong element next time"


async def test_credentials_do_not_reach_the_stored_definition(client: TestClient):
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]

    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    assert "someone" not in str(definition)
    assert "{{secret.username}}" in str(definition)


async def test_a_failed_run_cannot_be_distilled(client: TestClient):
    run_id = await seed_run(client, status="failed")
    llm = use_plan_llm(client)

    response = client.post(f"/api/runs/{run_id}/distill")

    assert response.status_code == 409
    assert "succeeded" in response.json()["detail"]
    assert llm.calls == 0, "a failed run must not cost a token"


async def test_distilling_an_unknown_run_is_a_404(client: TestClient):
    assert client.post("/api/runs/nope/distill").status_code == 404


async def test_a_model_that_returns_no_plan_is_reported_as_unprocessable(client: TestClient):
    run_id = await seed_run(client)
    use_plan_llm(client, plan=None)

    response = client.post(f"/api/runs/{run_id}/distill")

    assert response.status_code == 422
    assert "no idea" in response.json()["detail"]


# --- listing, reading, editing --------------------------------------------


async def test_use_cases_are_listed_newest_first(client: TestClient):
    run_id = await seed_run(client)
    use_plan_llm(client)
    client.post(f"/api/runs/{run_id}/distill")

    rows = client.get("/api/usecases").json()["usecases"]
    assert len(rows) == 1
    assert rows[0]["status"] == "draft"
    assert rows[0]["source_run_id"] == run_id


async def test_an_edit_appends_a_version_rather_than_rewriting_one(client: TestClient):
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]

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
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]

    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    definition["row_steps"][0]["url"] = "{{input.undeclared}}"

    response = client.put(f"/api/usecases/{usecase_id}", json=definition)
    assert response.status_code == 422
    assert "not declared" in response.json()["detail"]


async def test_editing_an_unknown_use_case_is_a_404(client: TestClient):
    assert client.put("/api/usecases/nope", json={"name": "x"}).status_code == 404


# --- publishing ------------------------------------------------------------


async def test_publishing_moves_a_draft_to_ready(client: TestClient):
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]

    assert client.post(f"/api/usecases/{usecase_id}/publish").json()["status"] == "ready"
    assert client.get(f"/api/usecases/{usecase_id}").json()["definition"]["status"] == "ready"


async def test_a_use_case_with_scripts_cannot_be_published_without_the_opt_in(client: TestClient):
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]

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
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]

    assert client.delete(f"/api/usecases/{usecase_id}").json()["status"] == "archived"
    assert client.get(f"/api/usecases/{usecase_id}").status_code == 200, "history still resolves"


async def test_publishing_an_unknown_use_case_is_a_404(client: TestClient):
    assert client.post("/api/usecases/nope/publish").status_code == 404


# --- archiving vs deleting -------------------------------------------------


async def test_archiving_is_reversible_and_keeps_the_record(client: TestClient):
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]

    assert client.delete(f"/api/usecases/{usecase_id}").json()["status"] == "archived"
    assert client.get(f"/api/usecases/{usecase_id}").status_code == 200


async def test_purging_removes_the_use_case_for_good(client: TestClient):
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]

    response = client.delete(f"/api/usecases/{usecase_id}", params={"purge": "true"})

    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert response.json()["removed"]["usecases"] == 1
    assert client.get(f"/api/usecases/{usecase_id}").status_code == 404
    assert client.get("/api/usecases").json()["usecases"] == []


async def test_purging_keeps_the_run_that_produced_it(client: TestClient):
    """The timeline records what happened to a browser; deleting a recipe
    must not erase the history of things it actually did."""
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]

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


async def test_distilling_returns_the_name_as_a_suggestion(client: TestClient):
    run_id = await seed_run(client)
    use_plan_llm(client)

    body = client.post(f"/api/runs/{run_id}/distill").json()

    assert body["suggested_name"] == "Sign in and open a record"
    assert body["name"] == body["suggested_name"]


async def test_renaming_changes_the_label_without_creating_a_version(client: TestClient):
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]

    response = client.patch(f"/api/usecases/{usecase_id}", json={"name": "Nightly demo requests"})

    assert response.status_code == 200
    body = client.get(f"/api/usecases/{usecase_id}").json()
    assert body["definition"]["name"] == "Nightly demo requests"
    assert body["definition"]["version"] == 1, "a label change is not a new version"
    assert [v["version"] for v in body["versions"]] == [1]


async def test_a_rename_shows_up_in_the_list(client: TestClient):
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]

    client.patch(f"/api/usecases/{usecase_id}", json={"name": "Renamed"})

    assert client.get("/api/usecases").json()["usecases"][0]["name"] == "Renamed"


async def test_renaming_leaves_the_steps_alone(client: TestClient):
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]
    before = client.get(f"/api/usecases/{usecase_id}").json()["definition"]

    client.patch(f"/api/usecases/{usecase_id}", json={"name": "Something else"})
    after = client.get(f"/api/usecases/{usecase_id}").json()["definition"]

    assert after["row_steps"] == before["row_steps"]
    assert after["setup_steps"] == before["setup_steps"]
    assert after["status"] == before["status"]


async def test_a_description_can_be_changed_too(client: TestClient):
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]

    client.patch(
        f"/api/usecases/{usecase_id}",
        json={"name": "Kept", "description": "One row per prospect."},
    )
    definition = client.get(f"/api/usecases/{usecase_id}").json()["definition"]
    assert definition["description"] == "One row per prospect."


async def test_renaming_a_published_use_case_does_not_unpublish_it(client: TestClient):
    """A label change must not send something back for review."""
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]
    client.post(f"/api/usecases/{usecase_id}/publish")

    client.patch(f"/api/usecases/{usecase_id}", json={"name": "Still ready"})

    assert client.get(f"/api/usecases/{usecase_id}").json()["definition"]["status"] == "ready"


@pytest.mark.parametrize("name", ["", "   ", "x" * 201])
async def test_an_unusable_name_is_refused(client: TestClient, name: str):
    run_id = await seed_run(client)
    use_plan_llm(client)
    usecase_id = client.post(f"/api/runs/{run_id}/distill").json()["usecase_id"]

    response = client.patch(f"/api/usecases/{usecase_id}", json={"name": name})
    assert response.status_code == 422


async def test_renaming_an_unknown_use_case_is_a_404(client: TestClient):
    assert client.patch("/api/usecases/nope", json={"name": "x"}).status_code == 404
