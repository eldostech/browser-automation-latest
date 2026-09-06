"""What a run cost, and what a workspace may spend.

Two things here spend tokens: an agent authoring session, and a healer
re-finding a control mid-replay. Before this neither was recorded anywhere that
could be summed, and one of them -- the healer -- counted its tokens carefully
and then dropped them on the floor, so a Guided replay that repaired itself
twice still reported as free.

These tests pin the recording, the ceiling, and the one property that makes the
ceiling worth having: it lowers the session's own budget rather than adding a
second thing that can stop a session, so there is one mechanism and one message.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from agent import Budget
from credentials import generate_key
from db.base import utcnow
from pricing import price_of, price_of_total
from test_api_execute import FakeReplaySession

pytestmark = pytest.mark.anyio


# --- pricing ---------------------------------------------------------------


def test_an_unlisted_model_is_not_free():
    """An estimate of $0.00 for a four-thousand-row batch is the most
    expensive kind of wrong, and a model missing from the table is far more
    likely to be new than to be free."""
    assert price_of("something-nobody-listed", {"input_tokens": 1_000_000}) > 0


def test_a_real_bedrock_id_is_matched_through_its_prefix_and_suffix():
    """Every actual deployment uses one, so an exact lookup would fall through
    to the default always and the table would be decoration."""
    priced = price_of(
        "us.anthropic.claude-haiku-4-5-20251001-v1:0", {"input_tokens": 1_000_000}
    )
    assert priced == pytest.approx(0.80)


def test_pricing_a_total_costs_more_than_pricing_the_same_tokens_as_input():
    """The assumed output share is stated rather than hidden, and generous.

    Output tokens cost about five times input ones, so the split is most of the
    answer -- and a governance number that under-reports is the one that lets a
    bill through.
    """
    as_input = price_of("claude-sonnet-5", {"input_tokens": 100_000})
    as_total = price_of_total("claude-sonnet-5", 100_000)
    assert as_total > as_input


# --- what a healer spends is no longer invisible ---------------------------


def test_a_healer_records_what_its_repairs_cost():
    """It counted tokens and dropped them. Now the price goes with them, taken
    from each turn's own input/output split rather than from a total."""
    from healing import HealingBudget

    budget = HealingBudget()
    budget.record(1_000, 0.012)
    budget.record(500, 0.006)

    assert budget.tokens_used == 1_500
    assert budget.usd_used == pytest.approx(0.018)
    assert budget.to_dict()["usd_used"] == pytest.approx(0.018)


def test_a_strict_run_reports_a_measured_zero_rather_than_a_blank():
    """"Free" and "not measured" look identical on a dashboard.

    An executor with no healer cannot reach a model at all, so zero here is a
    fact about the code rather than a default nobody filled in.
    """
    from engine import UseCaseExecutor
    from usecase import UseCase

    executor = UseCaseExecutor(UseCase(name="x"), object(), object(), run_id="r")
    assert executor.spent == (0, 0, 0.0)


def test_a_replay_that_healed_no_longer_claims_to_have_been_free():
    """`replay_terminal` hardcoded zero tokens. That was true while healing was
    the only spender and nothing carried what it spent; it stopped being true
    the moment a use case could choose Guided mode."""
    from engine import RowResult
    from runner import replay_terminal

    terminal = replay_terminal(
        RowResult(ok=True, llm_calls=2, llm_tokens=3_400, llm_usd=0.021)
    )

    assert terminal.tokens == 3_400
    assert terminal.cost_usd == pytest.approx(0.021)
    assert terminal.result["llm_calls"] == 2


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


async def spend(client: TestClient, usd: float, *, days_ago: int = 0) -> None:
    """A finished run that cost something, as the lifecycle would write it."""
    from conftest import app_workspace

    store = await app_workspace(client.app)
    import uuid

    run_id = uuid.uuid4().hex
    await store.create_run(run_id, "a run", None, {})
    await store.finish_run(
        run_id, "succeeded", steps=1, duration_ms=10, tokens=int(usd * 100_000),
        cost_usd=usd,
    )
    if days_ago:
        from db.models import Run
        from sqlalchemy import update

        async with store._sessions() as session:  # noqa: SLF001 - fixing a timestamp
            await session.execute(
                update(Run)
                .where(Run.id == run_id)
                .values(created_at=utcnow() - timedelta(days=days_ago))
            )
            await session.commit()


async def test_an_installation_that_never_set_a_budget_is_not_told_it_has_one(
    client: TestClient,
):
    """Unlimited and limited-to-a-number are different states."""
    body = client.get("/api/admin/spend").json()

    assert body["limit_usd"] is None
    assert body["remaining_usd"] is None
    assert body["usd"] == 0


async def test_spend_is_summed_over_every_kind_of_run(client: TestClient):
    """An authoring session and a Guided replay write to the same two columns,
    so there is one number rather than two somebody has to remember to add."""
    await spend(client, 0.40)
    await spend(client, 0.35)

    body = client.get("/api/admin/spend").json()
    assert body["usd"] == pytest.approx(0.75)
    assert body["runs"] == 2


async def test_last_month_does_not_count_against_this_month(client: TestClient):
    """A calendar month, because that is how a person thinks about a budget --
    and because a rolling window changes the answer while nothing happens."""
    await spend(client, 5.00, days_ago=40)
    await spend(client, 0.25)

    body = client.get("/api/admin/spend").json()
    assert body["usd"] == pytest.approx(0.25)


async def test_setting_a_ceiling_is_an_administrator_action(client: TestClient):
    response = client.put("/api/admin/spend/limit", json={"limit_usd": 25})

    assert response.status_code == 200, response.text
    assert response.json()["limit_usd"] == 25
    assert response.json()["remaining_usd"] == 25


async def test_the_ceiling_can_be_removed_but_zero_still_means_zero(
    client: TestClient,
):
    """None is "no limit"; zero is "stop all spending", which is a perfectly
    reasonable thing to set deliberately. Conflating them would make the second
    unexpressible."""
    client.put("/api/admin/spend/limit", json={"limit_usd": 0})
    assert client.get("/api/admin/spend").json()["limit_usd"] == 0

    client.put("/api/admin/spend/limit", json={"limit_usd": None})
    assert client.get("/api/admin/spend").json()["limit_usd"] is None


async def test_setting_a_ceiling_is_audited(client: TestClient):
    client.put("/api/admin/spend/limit", json={"limit_usd": 50})

    entries = client.get("/api/admin/audit").json()["entries"]
    entry = next(e for e in entries if e["action"] == "workspace.spend_limit")
    assert entry["detail"]["limit_usd"] == 50


# --- the ceiling, applied to a session -------------------------------------
#
# Driven against a stub store rather than the app's. What is being tested is
# arithmetic over one dictionary, and reaching into a live connection pool from
# a different event loop than the one it was bound to tests asyncpg instead.


class StubStore:
    """Answers `spend_this_month` and nothing else."""

    def __init__(self, used: float, limit: float | None) -> None:
        self.used = used
        self.limit = limit

    def workspace(self, workspace_id: str) -> "StubStore":
        return self

    async def spend_this_month(self) -> dict:
        remaining = None if self.limit is None else max(0.0, self.limit - self.used)
        return {
            "usd": self.used,
            "tokens": 0,
            "runs": 0,
            "limit_usd": self.limit,
            "remaining_usd": remaining,
        }


def manager_over(used: float, limit: float | None):
    from agent_manager import AgentSessions

    return AgentSessions(StubStore(used, limit), None, object(), lambda: None)


async def test_the_ceiling_lowers_the_session_budget_rather_than_watching_it():
    """One mechanism, one message.

    The session budget is already checked before every model call and every
    tool call. Folding the workspace's remaining allowance into it means a
    session stops the same way whichever limit it reached, instead of a second
    watcher racing the first.
    """
    budget = await manager_over(0.80, 1.00)._within_the_ceiling("ws", Budget(usd=5.0))

    assert budget.usd == pytest.approx(0.20), "capped to what the month has left"
    assert budget.steps == Budget().steps, "the other limits are untouched"


async def test_a_session_asking_for_less_than_is_left_keeps_its_own_budget():
    budget = await manager_over(0.0, 100.0)._within_the_ceiling("ws", Budget(usd=0.50))
    assert budget.usd == 0.50


async def test_a_workspace_with_no_ceiling_is_not_capped():
    budget = await manager_over(9999.0, None)._within_the_ceiling("ws", Budget(usd=5.0))
    assert budget.usd == 5.0


async def test_a_workspace_at_its_ceiling_is_refused_before_a_browser_opens():
    """Starting with nothing to spend would open a browser, take a snapshot and
    stop -- which reads as a failure rather than as a budget."""
    from agent_manager import AgentUnavailable

    with pytest.raises(AgentUnavailable) as caught:
        await manager_over(1.50, 1.00)._within_the_ceiling("ws", Budget())

    assert "monthly limit" in str(caught.value)
    assert "$1.50" in str(caught.value), (
        "it says what was spent, not merely that it is over"
    )


# --- prompt caching, priced correctly --------------------------------------
#
# A real session burned 127,000 tokens in six ordinary turns against a real
# page, almost all of it the tool schema list resent verbatim every time.
# Bedrock's prompt cache fixes the repetition; these tests are what stop the
# cost model silently pretending it did not, once it is turned on.


def test_a_cache_read_costs_a_tenth_of_a_fresh_token():
    """Without this, turning caching on made every session look exactly as
    expensive as before -- the same token count, priced as if none of it had
    been a hit."""
    fresh = price_of("claude-sonnet-5", {"input_tokens": 10_000})
    cached = price_of(
        "claude-sonnet-5",
        {"input_tokens": 10_000, "cache_read_tokens": 10_000},
    )
    assert cached == pytest.approx(fresh * 0.1)


def test_a_cache_write_costs_a_little_more_than_a_fresh_token():
    written = price_of(
        "claude-sonnet-5",
        {"input_tokens": 10_000, "cache_creation_tokens": 10_000},
    )
    fresh = price_of("claude-sonnet-5", {"input_tokens": 10_000})
    assert written == pytest.approx(fresh * 1.25)


def test_cache_and_fresh_tokens_in_one_turn_are_priced_separately():
    """The ordinary shape of a real turn: most of the input was cached, a
    little of it -- the newest tool result -- was not."""
    usage = {
        "input_tokens": 10_000,
        "cache_read_tokens": 9_000,
        "cache_creation_tokens": 0,
    }
    cost = price_of("claude-sonnet-5", usage)
    expected = (1_000 * 3.0 + 9_000 * 3.0 * 0.1) / 1_000_000
    assert cost == pytest.approx(expected)


def test_usage_with_no_cache_fields_prices_exactly_as_before():
    """Every existing caller -- the healer, every scripted test -- passes
    usage with no cache keys at all, and must keep costing what it always
    did."""
    assert price_of("claude-sonnet-5", {"input_tokens": 1000, "output_tokens": 200}) == (
        pytest.approx((1000 * 3.0 + 200 * 15.0) / 1_000_000)
    )
