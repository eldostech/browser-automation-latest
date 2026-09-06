"""Starting, watching, answering and saving an agent session over HTTP.

The router is deliberately the same shape as the recordings one -- start,
watch, save -- because two ways of producing the same artifact should not be
two products. What it has that the other does not is the part that costs money
and can act on its own: a budget on the way in, a decision endpoint, and a
verification report on the way out.

Neither a browser nor a model appears here. The manager builds both, so a test
replaces the two factories on the manager and drives everything else for real:
the routes, the permissions, the audit entries, the event stream.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from agent.graph import available
from agent.verify import Verification
from credentials import generate_key
from test_agent_author import ScriptedLLM, turn_calling
from test_agent_distil import ACCOUNT, a_recording
from test_agent_tools import FakeMCP
from test_api_execute import FakeReplaySession

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not available(), reason="LangGraph is an optional extra"),
]


@pytest.fixture
def client(db_settings, db_engine, tmp_path, monkeypatch):
    from conftest import authenticate, build_app

    app = build_app(
        db_settings, tmp_path, monkeypatch,
        session_cls=FakeReplaySession,
        credentials_key=generate_key(),
        agent_enabled=True,
    )
    with TestClient(app) as test_client:
        yield authenticate(test_client)


def arm(client: TestClient, *turns, replay=None) -> FakeMCP:
    """Give the manager a browser and a model that are neither."""
    provider = FakeMCP({
        name: ACCOUNT
        for name in ("browser_snapshot", "browser_navigate", "browser_type", "browser_click")
    })
    manager = client.app.state.agent_sessions
    # Accepts the per-session headless override even though this fake ignores
    # it -- the production signature takes one, and a stub narrower than that
    # would silently pass every test that calls it the real way.
    manager.provider = lambda headless=None: provider
    manager.llm_factory = lambda: ScriptedLLM(list(turns or a_recording()))
    # Verification would otherwise start Chromium, which is not what this file
    # is about. `test_agent_mcp_live.py` covers the real one, end to end.
    manager.replay = replay or _replays_cleanly
    return provider


async def _replays_cleanly(use_case, inputs, secrets):
    return Verification(ran=True, ok=True, outputs={"balance": "1,240.55"})


async def settle(client: TestClient, session_id: str, want: set[str]) -> dict:
    """Poll until the session reaches one of `want`. Sessions run as tasks."""
    for _ in range(200):
        body = client.get(f"/api/agent-sessions/{session_id}").json()
        if body["status"] in want:
            return body
        await asyncio.sleep(0.02)
    raise AssertionError(f"session stayed {body['status']}")


START = {
    "task": "Read the balance for the account in this row.",
    "start_url": "https://vendor.test/accounts",
    "name": "Pull balances",
}


# --- availability ----------------------------------------------------------


async def test_a_deployment_with_the_agent_off_says_so_rather_than_failing(
    db_settings, db_engine, tmp_path, monkeypatch
):
    """501, like the recorder answers on a machine with no display. It is not
    the caller's mistake, and the message has to name the setting."""
    from conftest import authenticate, build_app

    app = build_app(db_settings, tmp_path, monkeypatch, session_cls=FakeReplaySession)
    with TestClient(app) as client:
        authenticate(client)
        response = client.post("/api/agent-sessions", json=START)

    assert response.status_code == 501
    assert "AGENT_ENABLED" in response.json()["detail"]


async def test_starting_without_somewhere_to_start_is_refused(client: TestClient):
    arm(client)
    response = client.post("/api/agent-sessions", json={"task": "Do a thing"})

    assert response.status_code == 422
    assert "somewhere to start" in response.json()["detail"]


# --- the session -----------------------------------------------------------


async def test_a_session_runs_and_produces_a_verified_draft(client: TestClient):
    arm(client)

    created = client.post("/api/agent-sessions", json=START)
    assert created.status_code == 201, created.text
    session_id = created.json()["id"]

    body = await settle(client, session_id, {"succeeded", "partial", "failed"})
    assert body["status"] == "succeeded", body.get("error") or body.get("stopped_by")
    assert body["use_case"] is not None
    assert body["use_case"]["authored_by"] == "agent"
    assert body["spend"]["llm_calls"] > 0, "an agent session costs tokens"


async def test_the_session_appears_on_the_stream_a_replay_uses(client: TestClient):
    """No second streaming path. Same events table, same sequence numbers, so
    the existing run view shows an agent session as it shows a replay."""
    arm(client)
    created = client.post("/api/agent-sessions", json=START).json()
    await settle(client, created["id"], {"succeeded", "partial", "failed"})

    run = client.get(f"/api/runs/{created['run_id']}")
    assert run.status_code == 200, run.text

    events = client.get(f"/api/runs/{created['run_id']}/events").json()["events"]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "run_started"
    assert "tool_call" in kinds and "tool_result" in kinds

    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), (
        "the sequence number is the client's resume token; it has to be "
        "unique and increasing or a reconnect replays or skips events"
    )


async def test_the_allowlist_is_derived_rather_than_asked_for(client: TestClient):
    """A person describing a task should not also have to write an allowlist,
    and one typed by hand is what would be wrong when a session wanders."""
    provider = arm(client)
    created = client.post("/api/agent-sessions", json=START).json()
    await settle(client, created["id"], {"succeeded", "partial", "failed"})

    saved = client.post(
        f"/api/agent-sessions/{created['id']}/save", json={}
    ).json()
    definition = client.get(f"/api/usecases/{saved['usecase_id']}").json()["definition"]
    assert definition["allowed_domains"] == ["vendor.test"]


# --- approval --------------------------------------------------------------


def a_session_that_asks():
    return [
        turn_calling("browser_snapshot"),
        turn_calling("browser_click", target="e3", element="Delete this account"),
        turn_calling("mark_setup_complete"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("end_row"),
        turn_calling("finish", summary="Done, once you allowed it."),
    ]


async def test_a_session_that_asks_waits_and_is_answered(client: TestClient):
    from agent.author import FINISH  # noqa: F401 - naming the tool it calls

    arm(client, *([_plan()] + a_session_that_asks()), replay=_replays_cleanly)

    created = client.post(
        "/api/agent-sessions", json={**START, "may_write": True}
    ).json()
    waiting = await settle(client, created["id"], {"awaiting_approval"})

    assert waiting["awaiting"]["call"]["name"] == "browser_click"

    answered = client.post(
        f"/api/agent-sessions/{created['id']}/decide", json={"decision": "approved"}
    )
    assert answered.status_code == 200, answered.text

    body = await settle(client, created["id"], {"succeeded", "partial", "failed"})
    assert body["status"] in {"succeeded", "partial"}


async def test_answering_a_session_that_is_not_waiting_is_a_conflict(client: TestClient):
    arm(client)
    created = client.post("/api/agent-sessions", json=START).json()
    await settle(client, created["id"], {"succeeded", "partial", "failed"})

    response = client.post(
        f"/api/agent-sessions/{created['id']}/decide", json={"decision": "approved"}
    )
    assert response.status_code == 409


def _plan():
    from llm import LLMTurn

    return LLMTurn(text="Open the record and read it.",
                   usage={"input_tokens": 100, "output_tokens": 20})


# --- saving ----------------------------------------------------------------


async def test_the_draft_lands_as_a_draft_use_case(client: TestClient):
    """A model chose these steps, so a person reviews them. The same gate a
    codegen recording passes through, for a stronger reason."""
    arm(client)
    created = client.post("/api/agent-sessions", json=START).json()
    await settle(client, created["id"], {"succeeded", "partial", "failed"})

    saved = client.post(
        f"/api/agent-sessions/{created['id']}/save", json={"name": "Balances"}
    )
    assert saved.status_code == 201, saved.text
    assert saved.json()["status"] == "draft"

    definition = client.get(
        f"/api/usecases/{saved.json()['usecase_id']}"
    ).json()["definition"]
    assert definition["name"] == "Balances"
    assert definition["authored_by"] == "agent"


async def test_saving_a_session_that_produced_nothing_says_what_it_is_doing(
    client: TestClient,
):
    arm(client)
    created = client.post("/api/agent-sessions", json=START).json()

    response = client.post(f"/api/agent-sessions/{created['id']}/save", json={})

    assert response.status_code == 409
    assert "no draft yet" in response.json()["detail"]


async def test_saving_records_what_it_cost_and_whether_it_replayed(client: TestClient):
    """The audit entry a reviewer or an auditor actually wants: which session,
    what it spent, and whether the thing being published was proved."""
    arm(client)
    created = client.post("/api/agent-sessions", json=START).json()
    await settle(client, created["id"], {"succeeded", "partial", "failed"})
    client.post(f"/api/agent-sessions/{created['id']}/save", json={})

    entries = client.get("/api/admin/audit").json()["entries"]
    recorded = next(e for e in entries if e["action"] == "usecase.record")
    assert recorded["detail"]["agent_session_id"] == created["id"]
    assert "spend" in recorded["detail"]
    assert "verified" in recorded["detail"]


# --- scoping ---------------------------------------------------------------


async def test_a_session_from_another_workspace_is_not_found(client: TestClient):
    """An id is not an authorisation. The manager scopes every read."""
    arm(client)
    created = client.post("/api/agent-sessions", json=START).json()

    manager = client.app.state.agent_sessions
    assert manager.get(created["id"], "some-other-workspace") is None


async def test_shutting_down_waits_for_what_it_cancels(
    db_settings, db_engine, tmp_path, monkeypatch
):
    """Cancelling without awaiting leaks a database connection.

    A cancelled session still has to run its `finally`: close the browser, and
    let RunLifecycle write the run's ending and hand its connection back. Walk
    away at that moment and one of ten connections is gone -- and the symptom
    is not an error here, it is the next thing that wants a connection waiting
    forever. It hung the suite a whole file later, inside a fixture.

    Driven through the app's own lifespan rather than by calling shutdown
    directly, because that is the path that actually runs in production and it
    is the one that has to be right.
    """
    from conftest import authenticate, build_app

    app = build_app(
        db_settings, tmp_path, monkeypatch,
        session_cls=FakeReplaySession,
        credentials_key=generate_key(),
        agent_enabled=True,
    )
    with TestClient(app) as client:
        authenticate(client)
        arm(client)
        client.post("/api/agent-sessions", json=START)
        manager = app.state.agent_sessions
        tasks = [record._task for record in manager._sessions.values()]
        assert tasks and any(not t.done() for t in tasks), "nothing was in flight"

    # Leaving the block ran lifespan shutdown.
    assert all(task.done() for task in tasks), (
        "shutdown returned while a session was still running, so its finally "
        "never ran and its connection was never handed back"
    )
    assert manager._sessions == {}


# --- watching the session ----------------------------------------------


def test_headless_defaults_to_the_deployment_setting():
    """Nobody asked, so the installation's own default decides -- same as it
    always has."""
    from agent_manager import AgentSessions

    manager = AgentSessions(None, None, object(), lambda: None)
    manager.settings = type("S", (), {"agent_headless": True, "agent_browser_provider": "local"})()

    assert manager.provider().headless is True
    assert manager.provider(headless=None).headless is True


def test_a_person_starting_a_session_can_ask_to_watch_it():
    """The same choice a replay already offers under "Show the browser while
    it runs" -- watching is how trust in this gets built the first few times,
    and nobody should have to ask an administrator to see it."""
    from agent_manager import AgentSessions

    manager = AgentSessions(None, None, object(), lambda: None)
    manager.settings = type("S", (), {"agent_headless": True, "agent_browser_provider": "local"})()

    assert manager.provider(headless=False).headless is False


def test_the_override_has_no_effect_on_a_managed_browser():
    """A CDP session's browser is somebody else's to configure, and this
    session did not start it -- headless is not this deployment's call to make
    there either way."""
    from agent_manager import AgentSessions

    manager = AgentSessions(None, None, object(), lambda: None)
    manager.settings = type(
        "S",
        (),
        {
            "agent_headless": True,
            "agent_browser_provider": "cdp",
            "agent_cdp_endpoint": "ws://managed/abc",
        },
    )()

    provider = manager.provider(headless=False)
    assert "--headless" not in provider.argv()
    assert "ws://managed/abc" in provider.argv()


async def test_starting_a_session_can_request_a_visible_browser(client: TestClient):
    """Through the whole path: the request body, the manager, the provider
    that is actually constructed."""
    arm(client)
    manager = client.app.state.agent_sessions
    seen = {}
    real_start = manager.start

    async def spy(**kwargs):
        seen["headless"] = kwargs.get("headless")
        return await real_start(**kwargs)

    manager.start = spy

    client.post("/api/agent-sessions", json={**START, "headless": False})

    assert seen["headless"] is False
