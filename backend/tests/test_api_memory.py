"""Reading, adding to, and correcting what the system has learned.

Healing writes here on its own when it is confident. These endpoints are the
other half: what it learned, whether that was right, and taking it out when it
was not. The human-written entry matters most -- when the model is not sure
enough to repair a step, somebody looks at it, and what they know at that
moment previously had nowhere to go.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from credentials import generate_key
from test_api_execute import ExplodingLLM, FakeReplaySession

pytestmark = pytest.mark.anyio

FIX = {
    "step_id": "s3",
    "page_url": "https://shop.test/checkout",
    "page": 'button "Confirm purchase"\ntextbox "Card number"',
    "step_summary": "click Submit order",
    "wanted": 'role=button name="Submit order"',
    "explanation": "they renamed Submit order to Confirm purchase and moved it into the dialog",
    "usecase_id": "uc-1",
    "old_locator": {"strategy": "role", "role": "button", "name": "Submit order"},
    "new_locator": {"strategy": "role", "role": "button", "name": "Confirm purchase"},
}


@pytest.fixture
def client(db_settings, db_engine, tmp_path, monkeypatch):
    from conftest import authenticate, build_app

    app = build_app(
        db_settings,
        tmp_path,
        monkeypatch,
        session_cls=FakeReplaySession,
        credentials_key=generate_key(),
        # Deterministic, so the suite needs no Bedrock.
        embedding_backend="hash",
    )
    with TestClient(app) as test_client:
        test_client.app.state.repair_model._client = ExplodingLLM()  # noqa: SLF001 - test seam
        yield authenticate(test_client)


async def test_a_person_can_record_what_they_worked_out(client: TestClient):
    response = client.post("/api/memory", json=FIX)
    assert response.status_code == 201, response.text
    assert response.json()["domain"] == "shop.test"

    [fix] = client.get("/api/memory").json()["fixes"]
    assert fix["new_locator"]["name"] == "Confirm purchase"
    assert fix["domain"] == "shop.test"


async def test_a_human_entry_is_attributed_to_them_not_to_the_model(client: TestClient):
    """A fix somebody confirmed outranks one nobody checked, and that ordering
    is what `as_prompt` sorts on."""
    client.post("/api/memory", json=FIX)
    [fix] = client.get("/api/memory").json()["fixes"]

    assert fix["confirmed_by"] not in ("", "model")
    assert "@" in fix["confirmed_by"]


async def test_the_explanation_is_required(client: TestClient):
    """The entry exists to be read by somebody who was not there. Without the
    sentence it is a locator swap with no account of why."""
    response = client.post("/api/memory", json={**FIX, "explanation": ""})
    assert response.status_code == 422


async def test_a_relative_url_is_refused_because_it_scopes_nothing(client: TestClient):
    response = client.post("/api/memory", json={**FIX, "page_url": "/checkout"})
    assert response.status_code == 422
    assert "domain" in response.json()["detail"]


async def test_fixes_can_be_filtered_by_site(client: TestClient):
    client.post("/api/memory", json=FIX)
    client.post("/api/memory", json={**FIX, "page_url": "https://other.test/x"})

    assert len(client.get("/api/memory").json()["fixes"]) == 2
    filtered = client.get("/api/memory", params={"domain": "shop.test"}).json()["fixes"]
    assert [f["domain"] for f in filtered] == ["shop.test"]


async def test_a_stale_fix_can_be_taken_back_out(client: TestClient):
    """One that was right last month and wrong now does not fail loudly -- it
    gets recalled as precedent and quietly makes the next repair worse."""
    client.post("/api/memory", json=FIX)
    [fix] = client.get("/api/memory").json()["fixes"]

    assert client.delete(f"/api/memory/{fix['id']}").status_code == 200
    assert client.get("/api/memory").json()["fixes"] == []


async def test_forgetting_something_that_is_not_there_is_a_404(client: TestClient):
    assert client.delete("/api/memory/nope").status_code == 404


async def test_recording_is_refused_when_the_memory_is_off(
    db_settings, db_engine, tmp_path, monkeypatch
):
    """With no embedder there is nowhere to put it, and saying so beats
    accepting the entry and silently dropping it."""
    from conftest import authenticate, build_app

    app = build_app(
        db_settings,
        tmp_path,
        monkeypatch,
        session_cls=FakeReplaySession,
        healing_memory_enabled=False,
    )
    with TestClient(app) as raw:
        client = authenticate(raw)
        response = client.post("/api/memory", json=FIX)
        assert response.status_code == 503
        assert "HEALING_MEMORY_ENABLED" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Learning from the repair button
# ---------------------------------------------------------------------------


async def test_repairing_with_ai_is_remembered(client: TestClient):
    """The regression that made the whole feature pointless in practice.

    The in-run healer wrote every high-confidence fix to healing memory. The
    repair endpoint -- the one behind the button a person actually presses --
    saved a new version, wrote an audit entry, and learned nothing. So the same
    page change was diagnosed from scratch on every run, at the cost of an LLM
    call each, and the table stayed empty however many times the button worked.
    """
    from conftest import app_workspace
    from test_api_execute import BROKEN, RepairLLM

    store = await app_workspace(client.app)
    usecase_id, _ = await store.save_usecase(BROKEN)

    failure = client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={
            "inputs": {"record_url": "https://example.com/r"},
            "secrets": {"username": "u", "password": "p"},
        },
    ).json()
    assert failure["status"] == "failed", failure

    client.app.state.repair_model._client = RepairLLM(  # noqa: SLF001 - test seam
        {
            "diagnosis": "The button is now called Submit.",
            "confidence": "high",
            "fixes": [
                {
                    "kind": "replace_locator",
                    "step_id": "s1",
                    "element_index": 1,
                    "reason": "same control, new label",
                }
            ],
        }
    )

    repaired = client.post(
        f"/api/usecases/{usecase_id}/repair",
        json={"execution_id": failure["execution_id"]},
    )
    assert repaired.status_code == 201, repaired.text
    assert repaired.json()["repaired"] is True
    assert repaired.json()["remembered"] == 1, "the fix has to be written down"

    learned = client.get("/api/memory").json()["fixes"]
    assert len(learned) == 1, learned
    entry = learned[0]
    assert entry["step_id"] == "s1"
    assert entry["new_locator"]["name"] == "Submit"
    assert entry["old_locator"]["name"] == "Nonexistent"
    assert "@" in entry["confirmed_by"], (
        "a person chose this repair and published it, which outranks an "
        "unattended heal -- as_prompt sorts on exactly that"
    )


async def test_a_repair_still_succeeds_when_it_cannot_be_remembered(
    client: TestClient, monkeypatch
):
    """The repair is the point; remembering it is the bonus.

    A use case that was successfully mended must not be reported as broken
    because the thing that writes the lesson down is switched off, unreachable,
    or out of embedding quota.
    """
    from conftest import app_workspace
    from test_api_execute import BROKEN, RepairLLM

    store = await app_workspace(client.app)
    usecase_id, _ = await store.save_usecase(BROKEN)

    failure = client.post(
        f"/api/usecases/{usecase_id}/execute",
        json={
            "inputs": {"record_url": "https://example.com/r"},
            "secrets": {"username": "u", "password": "p"},
        },
    ).json()
    assert failure["status"] == "failed", failure

    def explode(_workspace_id):
        raise RuntimeError("the embedding service is down")

    monkeypatch.setattr(client.app.state.replays, "make_memory", explode)
    client.app.state.repair_model._client = RepairLLM(  # noqa: SLF001 - test seam
        {
            "diagnosis": "The button is now called Submit.",
            "confidence": "high",
            "fixes": [
                {
                    "kind": "replace_locator",
                    "step_id": "s1",
                    "element_index": 1,
                    "reason": "same control, new label",
                }
            ],
        }
    )

    repaired = client.post(
        f"/api/usecases/{usecase_id}/repair",
        json={"execution_id": failure["execution_id"]},
    )

    assert repaired.status_code == 201, repaired.text
    assert repaired.json()["repaired"] is True
    assert repaired.json()["remembered"] == 0
    detail = client.get(f"/api/usecases/{usecase_id}").json()
    assert detail["definition"]["row_steps"][0]["locators"][0]["name"] == "Submit"


async def test_the_next_repair_is_told_what_the_last_one_worked_out(client: TestClient):
    """Writing it down is only half of not repeating a mistake.

    The repair path had no memory at all -- it neither wrote nor read -- so the
    same page change was re-derived on every press. This closes the loop: what
    the first repair learned is in the prompt the second one gets.
    """
    from conftest import app_workspace
    from test_api_execute import BROKEN, RepairLLM

    store = await app_workspace(client.app)
    usecase_id, _ = await store.save_usecase(BROKEN)

    proposal = {
        "diagnosis": "The button is now called Submit.",
        "confidence": "high",
        "fixes": [
            {
                "kind": "replace_locator",
                "step_id": "s1",
                "element_index": 1,
                "reason": "same control, new label",
            }
        ],
    }

    async def repair_once() -> RepairLLM:
        failure = client.post(
            f"/api/usecases/{usecase_id}/execute",
            json={
                "inputs": {"record_url": "https://example.com/r"},
                "secrets": {"username": "u", "password": "p"},
            },
        ).json()
        assert failure["status"] == "failed", failure
        llm = RepairLLM(proposal)
        client.app.state.repair_model._client = llm  # noqa: SLF001 - test seam
        response = client.post(
            f"/api/usecases/{usecase_id}/repair",
            json={"execution_id": failure["execution_id"]},
        )
        assert response.status_code == 201, response.text
        return llm

    first = await repair_once()
    assert "nothing similar has been fixed" in first.prompts[0], (
        "the first repair has nothing to go on, which is the point of the second"
    )

    # Put it back the way it was, so the same failure happens again.
    await store.save_usecase({**BROKEN, "id": usecase_id, "status": "ready"})
    second = await repair_once()

    assert "Submit" in second.prompts[0], (
        "the second repair is told what the first one worked out"
    )
