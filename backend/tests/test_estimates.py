"""What a batch will cost, before anybody commits to it.

A limit is what stops a mistake; an estimate is what prevents one. This is the
cheaper of the two and the one a person actually reads, so what it says matters
more than how precisely it says it: the difference somebody needs to see is
between "nothing", "a few dollars if the site has moved" and "three hundred and
twenty dollars", and those are three orders of magnitude apart.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from credentials import generate_key
from estimates import estimate_batch
from usecase import Locator, Step, UseCase
from test_api_execute import FakeReplaySession

pytestmark = pytest.mark.anyio

MODEL = "us.anthropic.claude-sonnet-5-20260514-v1:0"


def a_use_case(mode) -> UseCase:
    return UseCase(
        name="Pull balances",
        mode=mode,
        status="ready",
        allowed_domains=["vendor.test"],
        row_steps=[Step(id="s1", action="navigate", url="https://vendor.test/")],
    )


# --- the three answers -----------------------------------------------------


def test_strict_costs_nothing_and_says_why():
    """Not "about nothing". Nothing -- the engine cannot construct a model
    client, so this is a property of the code and the copy should say so
    rather than sounding like a forecast."""
    estimate = estimate_batch(a_use_case("strict"), 4000, model=MODEL)

    assert estimate.low_usd == 0 and estimate.high_usd == 0
    assert "property of the code" in estimate.note


def test_guided_starts_at_nothing_because_that_is_the_usual_answer():
    """Quoting an average would misrepresent the common case as the expected
    one. On a site that has not changed, a Guided batch costs zero."""
    estimate = estimate_batch(a_use_case("guided"), 4000, model=MODEL)

    assert estimate.low_usd == 0
    assert estimate.high_usd > 0
    assert "unless something on the site has moved" in estimate.note


def test_explore_is_a_range_and_the_range_is_wide():
    """How many turns a row takes depends on the site. A single number would
    imply an accuracy this cannot have."""
    estimate = estimate_batch(a_use_case("explore"), 4000, model=MODEL)

    assert estimate.low_usd > 0
    assert estimate.high_usd > estimate.low_usd * 2
    assert "4,000" in estimate.note


def test_explore_costs_orders_of_magnitude_more_than_guided():
    """The comparison is the whole point of showing an estimate at all."""
    rows = 4000
    guided = estimate_batch(a_use_case("guided"), rows, model=MODEL)
    explore = estimate_batch(a_use_case("explore"), rows, model=MODEL)

    assert explore.low_usd > guided.high_usd * 10


def test_a_large_explore_batch_is_offered_the_cheaper_route():
    """The product arguing for itself: explore a few, save what it learns,
    replay the rest for nothing."""
    note = estimate_batch(a_use_case("explore"), 4000, model=MODEL).note

    assert "twenty rows" in note
    assert "Strict" in note


def test_a_small_explore_batch_is_not_lectured():
    """Twenty rows is what the advice tells you to do. Repeating it there
    would be noise."""
    assert "twenty rows" not in estimate_batch(a_use_case("explore"), 5, model=MODEL).note


def test_the_deployment_ceiling_still_applies_to_the_estimate():
    """Healing off means every use case is Strict, and the estimate has to
    agree with what will actually happen rather than with the document."""
    estimate = estimate_batch(
        a_use_case("explore"), 4000, model=MODEL, healing_enabled=False
    )

    assert estimate.mode == "strict"
    assert estimate.high_usd == 0


def test_an_estimate_over_what_is_left_this_month_says_so():
    estimate = estimate_batch(
        a_use_case("explore"), 4000, model=MODEL, remaining_usd=5.0
    )

    assert estimate.over_budget


def test_an_estimate_within_budget_does_not():
    estimate = estimate_batch(
        a_use_case("guided"), 10, model=MODEL, remaining_usd=50.0
    )

    assert not estimate.over_budget


# --- through the API -------------------------------------------------------


@pytest.fixture
def client(db_settings, db_engine, tmp_path, monkeypatch):
    from conftest import authenticate, build_app

    app = build_app(
        db_settings, tmp_path, monkeypatch,
        session_cls=FakeReplaySession, credentials_key=generate_key(),
        replay_healing_enabled=True,
    )
    with TestClient(app) as test_client:
        yield authenticate(test_client)


DEFINITION = {
    "name": "Pull balances",
    "status": "ready",
    "mode": "explore",
    "allowed_domains": ["vendor.test"],
    "outputs": ["balance"],
    "description": "Open the account and read its balance.",
}


async def test_the_estimate_is_available_before_a_batch_starts(client: TestClient):
    created = client.post(
        "/api/usecases/import", json={**DEFINITION, "id": "e" * 32}
    )
    assert created.status_code == 201, created.text
    usecase_id = created.json()["usecase_id"]

    body = client.get(f"/api/usecases/{usecase_id}/estimate?rows=4000").json()

    assert body["mode"] == "explore"
    assert body["rows"] == 4000
    assert body["high_usd"] > body["low_usd"]
    assert body["note"]


async def test_an_explore_use_case_may_declare_outputs_no_step_produces(
    client: TestClient,
):
    """The schema exception that makes Explore expressible at all.

    There are no steps: the declared outputs are the *instruction* the agent is
    given, not a summary of what the steps do. A row that finishes without them
    is failed by the operate graph rather than returned with blank columns.
    """
    response = client.post("/api/usecases/import", json={**DEFINITION, "id": "f" * 32})

    assert response.status_code == 201, response.text
