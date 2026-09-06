"""Replay first, and an agent only when the plan breaks.

Two things are being pinned here, and the first is the important one.

**The happy path costs nothing.** A row where every step passes must make no
model call at all -- not "few", none -- because that is the property the whole
product rests on, and it is the property that quietly stops being true the
first time somebody adds a "quick check" to the loop.

**A rescue resumes, it does not restart.** Re-running a partial row from the
top is how a form gets submitted twice, and a test is the only thing standing
between that and a live system.
"""

from __future__ import annotations

import pytest

from agent.graph import available
from agent.inprocess import EngineBrowser, with_refs
from agent.operate import GIVE_UP, RESUME, run_row_with_agent
from engine import RowResult
from test_agent_author import ScriptedLLM, turn_calling
from llm import LLMTurn

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not available(), reason="LangGraph is an optional extra"),
]


ARIA = '''- generic [active]:
  - heading "Users" [level=1]
  - button "Continue"
  - textbox "Email"'''


class FakeLocator:
    def __init__(self, page) -> None:
        self.page = page

    async def click(self, **_):
        self.page.did.append("click")

    async def fill(self, value, **_):
        self.page.did.append(f"fill:{value}")

    async def select_option(self, value, **_):
        self.page.did.append(f"select:{value}")


class FakePage:
    def __init__(self) -> None:
        self.url = "https://vendor.test/users"
        self.did: list[str] = []

    def locator(self, _selector):
        return self

    async def aria_snapshot(self):
        return ARIA

    async def goto(self, url):
        self.did.append(f"goto:{url}")
        self.url = url

    async def go_back(self):
        self.did.append("back")


class FakeBrowser:
    def __init__(self) -> None:
        self.page = FakePage()

    @property
    def url(self):
        return self.page.url


class FakeExecutor:
    """A replay that fails a fixed number of times, then succeeds.

    Records how it was called, because *how* it resumes is most of what these
    tests are about.
    """

    step_timeout = 5.0

    def __init__(self, failures: int = 0, fail_at: int = 2) -> None:
        self.browser = FakeBrowser()
        self.failures = failures
        self.fail_at = fail_at
        self.calls: list[dict] = []

    async def run_row(self, inputs, *, start_at=0, outputs=None):
        self.calls.append({"start_at": start_at, "outputs": dict(outputs or {})})
        if self.failures > 0:
            self.failures -= 1
            return RowResult(
                ok=False,
                outputs={**(outputs or {}), "seen": "before the failure"},
                failed_step_id="s7",
                failed_index=self.fail_at,
                error='step \'s7\' (click role=button name="Next") failed: TimeoutError',
            )
        return RowResult(ok=True, outputs={**(outputs or {}), "balance": "1,240.55"})

    async def locate(self, locators, step_id="ad-hoc"):
        return FakeLocator(self.browser.page), 0, "role=button"


async def _ignore(event):
    return None


def a_model(*turns):
    return ScriptedLLM(list(turns))


# --- the happy path --------------------------------------------------------


async def test_a_row_that_works_makes_no_model_call_at_all():
    """Not "few". None.

    This is the property the product rests on, and the one that quietly stops
    being true the first time somebody adds a check to the loop.
    """
    executor = FakeExecutor(failures=0)
    llm = a_model()

    result = await run_row_with_agent(
        executor, {"account": "A-1001"}, llm=llm, emit=_ignore,
        allowed_domains=("vendor.test",),
    )

    assert result.ok
    assert llm.asked == [], "the model was consulted on a row that worked"
    assert len(executor.calls) == 1


async def test_a_row_that_works_runs_the_plan_from_the_top():
    executor = FakeExecutor(failures=0)

    await run_row_with_agent(
        executor, {}, llm=a_model(), emit=_ignore, allowed_domains=("vendor.test",)
    )

    assert executor.calls == [{"start_at": 0, "outputs": {}}]


# --- rescue ----------------------------------------------------------------


async def test_a_rescue_resumes_from_the_step_that_failed():
    """Re-running a partial row from the top is how a form gets submitted
    twice. The resume index is not an optimisation; it is the only safe way to
    carry on at all."""
    executor = FakeExecutor(failures=1, fail_at=3)
    llm = a_model(
        turn_calling("browser_snapshot"),
        turn_calling(RESUME, note="dismissed a cookie banner"),
    )

    result = await run_row_with_agent(
        executor, {}, llm=llm, emit=_ignore, allowed_domains=("vendor.test",)
    )

    assert result.ok
    assert [call["start_at"] for call in executor.calls] == [0, 3]


async def test_a_rescue_carries_forward_what_was_already_read():
    """A value extracted before the failure is not extracted again -- the page
    it was on is three pages back by now."""
    executor = FakeExecutor(failures=1)
    llm = a_model(turn_calling(RESUME, note="waited"))

    result = await run_row_with_agent(
        executor, {}, llm=llm, emit=_ignore, allowed_domains=("vendor.test",)
    )

    assert executor.calls[1]["outputs"] == {"seen": "before the failure"}
    assert result.outputs["seen"] == "before the failure"
    assert result.outputs["balance"] == "1,240.55"


async def test_giving_up_fails_the_row_with_the_reason_attached():
    """A row that fails with a clear reason is worth more than one that
    succeeded by doing something nobody asked for."""
    executor = FakeExecutor(failures=1)
    llm = a_model(turn_calling(GIVE_UP, reason="the account is locked"))

    result = await run_row_with_agent(
        executor, {}, llm=llm, emit=_ignore, allowed_domains=("vendor.test",)
    )

    assert not result.ok
    assert "the account is locked" in result.error
    assert len(executor.calls) == 1, "it did not resume after giving up"


async def test_rescues_are_bounded_so_a_bad_row_cannot_spend_forever():
    """A row needing four rescues is a use case that needs re-recording, and
    paying to discover that once per row across four thousand rows is the bill
    this design exists to avoid."""
    executor = FakeExecutor(failures=99)
    llm = a_model(*[turn_calling(RESUME, note="tried")] * 20)

    result = await run_row_with_agent(
        executor, {}, llm=llm, emit=_ignore,
        allowed_domains=("vendor.test",), max_attempts=2,
    )

    assert not result.ok
    assert len(executor.calls) == 3, "one attempt, then two rescues, then it stops"


async def test_a_failure_with_nowhere_to_resume_is_not_rescued():
    """A row that failed before any step ran -- a missing input, say -- is not
    something an agent can help with, and asking it would be a model call spent
    on a certainty."""
    executor = FakeExecutor(failures=1)
    executor.fail_at = None  # type: ignore[assignment]
    llm = a_model(turn_calling(RESUME))

    result = await run_row_with_agent(
        executor, {}, llm=llm, emit=_ignore, allowed_domains=("vendor.test",)
    )

    assert not result.ok
    assert llm.asked == [], "no model call for a failure nothing can clear"


async def test_the_rescue_is_told_which_step_failed_and_why():
    executor = FakeExecutor(failures=1)
    llm = a_model(turn_calling(RESUME))

    await run_row_with_agent(
        executor, {}, llm=llm, emit=_ignore, allowed_domains=("vendor.test",)
    )

    system = llm.asked[0]["system"]
    assert "s7" in system
    assert "TimeoutError" in system
    assert "vendor.test" in system, "the allowlist is in the prompt as well as the guard"


async def test_the_rescue_is_told_not_to_finish_the_task_itself():
    """The remaining steps run again the moment it stands aside. An agent that
    submits the form leaves the workflow to submit it a second time."""
    executor = FakeExecutor(failures=1)
    llm = a_model(turn_calling(RESUME))

    await run_row_with_agent(
        executor, {}, llm=llm, emit=_ignore, allowed_domains=("vendor.test",)
    )

    assert "submit it again" in llm.asked[0]["system"]


# --- one browser -----------------------------------------------------------


async def test_the_rescue_acts_on_the_page_the_replay_was_already_on():
    """Not a second browser. Starting one would be a blank tab looking at
    nothing while the failure sits three pages away in the first."""
    executor = FakeExecutor(failures=1)
    llm = a_model(
        turn_calling("browser_snapshot"),
        turn_calling("browser_click", target="e3", element="Continue"),
        turn_calling(RESUME, note="clicked past the banner"),
    )

    await run_row_with_agent(
        executor, {}, llm=llm, emit=_ignore, allowed_domains=("vendor.test",)
    )

    assert "click" in executor.browser.page.did


async def test_the_adapter_numbers_elements_the_way_the_real_server_does():
    """Playwright's own snapshot has no refs -- the MCP server adds them. They
    are added here so the parser, the guard's staleness check and the ladder
    builder all work unchanged on both."""
    refs = with_refs(ARIA)

    assert '- button "Continue" [ref=e2]' in refs or '[ref=' in refs
    assert refs.count("[ref=") == 4, refs


async def test_a_structural_line_gets_no_ref():
    """`- text:` and `- /url:` are the snapshot's own structure, not elements,
    and the real server does not number them either."""
    refs = with_refs('- link "Open":\n  - /url: /a/1\n  - text: Open')

    assert refs.count("[ref=") == 1


async def test_the_adapter_refuses_a_ref_the_page_is_not_showing():
    executor = FakeExecutor()
    browser = EngineBrowser(executor)
    await browser.call("browser_snapshot", {})

    result = await browser.call("browser_click", {"target": "e99"})

    assert result.is_error
    assert "not on the page" in result.text


async def test_an_action_answers_with_the_page_as_it_now_stands():
    """Exactly as the real server does, so the agent's next call has refs for
    the page it is now on rather than for the one it just left."""
    executor = FakeExecutor()
    browser = EngineBrowser(executor)
    await browser.call("browser_snapshot", {})

    result = await browser.call("browser_click", {"target": "e3"})

    assert not result.is_error, result.text
    assert "### Page" in result.text
    assert result.refs, "the reply has to carry refs or the next call is blind"


# --- when the graph is used at all -----------------------------------------


def a_manager(tmp_path, **settings_kwargs):
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


def a_use_case(mode):
    from usecase import Step, UseCase

    return UseCase(
        name="x",
        mode=mode,
        allowed_domains=["vendor.test"],
        row_steps=[Step(id="s1", action="navigate", url="https://vendor.test/")],
    )


def test_a_strict_use_case_never_gets_the_graph(tmp_path):
    """Not even where the deployment could provide it. Strict means the model
    is unreachable, and a second path that could reach one would make that a
    promise rather than a property."""
    manager = a_manager(tmp_path, replay_healing_enabled=True, agent_enabled=True)

    assert manager.make_row_runner(None, a_use_case("strict"), "r", None, None) is None


def test_a_deployment_without_the_agent_runs_exactly_as_it_did(tmp_path):
    """Nothing changes underneath an installation that did not opt in: a
    Guided row still gets the engine and its healer, which is what it had."""
    manager = a_manager(tmp_path, replay_healing_enabled=True, agent_enabled=False)

    assert manager.make_row_runner(None, a_use_case("guided"), "r", None, None) is None


def test_a_guided_use_case_gets_the_graph_where_the_deployment_offers_it(tmp_path):
    manager = a_manager(tmp_path, replay_healing_enabled=True, agent_enabled=True)

    assert manager.make_row_runner(None, a_use_case("guided"), "r", None, None) is not None


def test_healing_switched_off_is_still_a_ceiling(tmp_path):
    """The deployment setting can only ever restrict, and that has to hold for
    the graph as well or the ceiling would have a hole in it."""
    manager = a_manager(tmp_path, replay_healing_enabled=False, agent_enabled=True)

    assert manager.make_row_runner(None, a_use_case("guided"), "r", None, None) is None


# --- explore ---------------------------------------------------------------


def an_explore_use_case(outputs=("balance",)):
    from usecase import UseCase

    return UseCase(
        name="Read balances",
        description="Open the account in this row and read its balance.",
        mode="explore",
        allowed_domains=["vendor.test"],
        outputs=list(outputs),
    )


async def explore_row(*turns, outputs=("balance",), inputs=None):
    executor = FakeExecutor(failures=0)
    llm = a_model(*turns)
    result = await run_row_with_agent(
        executor,
        inputs or {"account": "A-1001"},
        llm=llm,
        emit=_ignore,
        allowed_domains=("vendor.test",),
        explore=True,
        usecase=an_explore_use_case(outputs),
    )
    return result, llm, executor


async def test_explore_never_touches_the_engine():
    """There is no plan to replay. A use case in Explore that fell through to
    the engine would run an empty step list and report success, which is the
    worst possible outcome because nobody goes looking for it."""
    result, _llm, executor = await explore_row(
        turn_calling("record_value", name="balance", value="1,240.55"),
        turn_calling("finish"),
    )

    assert result.ok
    assert executor.calls == [], "the engine was asked to replay something"


async def test_a_value_is_banked_when_it_is_seen():
    """Not collected at the end. The page a value was on is three pages back
    by the time a row finishes."""
    result, _llm, _ = await explore_row(
        turn_calling("browser_snapshot"),
        turn_calling("record_value", name="balance", value="1,240.55"),
        turn_calling("finish"),
    )

    assert result.outputs == {"balance": "1,240.55"}


async def test_finishing_without_the_values_is_a_failed_row():
    """A results file with blank columns is worse than a row marked failed,
    because nobody goes looking for it."""
    result, _llm, _ = await explore_row(turn_calling("finish", note="all done"))

    assert not result.ok
    assert "balance" in result.error


async def test_giving_up_says_what_stopped_it():
    result, _llm, _ = await explore_row(
        turn_calling(GIVE_UP, reason="the account does not exist")
    )

    assert not result.ok
    assert "does not exist" in result.error


async def test_a_row_that_runs_out_of_steps_says_what_to_do_about_it():
    """"It stopped" is not enough: the two fixes are opposite -- raise the
    budget, or stop working it out every time and record it."""
    result, _llm, _ = await explore_row(*[turn_calling("browser_snapshot")] * 40)

    assert not result.ok
    assert "record this workflow" in result.error


async def test_the_row_is_told_its_own_values_and_what_to_report():
    result, llm, _ = await explore_row(
        turn_calling("record_value", name="balance", value="1"),
        turn_calling("finish"),
        inputs={"account": "A-1001"},
    )

    system = llm.asked[0]["system"]
    assert "A-1001" in system
    assert "balance" in system
    assert "Do not sign in" in system, (
        "the session is shared across rows, and signing in per row is the "
        "thing this whole design exists to avoid"
    )


async def test_explore_reports_what_the_row_cost():
    """Per-row, because that is the number that multiplies by four thousand."""
    result, _llm, _ = await explore_row(
        turn_calling("record_value", name="balance", value="1"),
        turn_calling("finish"),
    )

    assert result.llm_calls == 2
    assert result.llm_tokens > 0
    assert result.llm_usd > 0


def test_explore_without_an_agent_is_refused_rather_than_downgraded(tmp_path):
    """The one mode difference that is not about how much a run may spend.

    Falling back to the engine would replay nothing and mark every row
    succeeded. Refusing is the only honest answer.
    """
    from runner import ModeUnavailable

    manager = a_manager(tmp_path, replay_healing_enabled=True, agent_enabled=False)
    usecase = a_use_case("explore")

    with pytest.raises(ModeUnavailable) as caught:
        manager.make_row_runner(None, usecase, "r", None, None)

    assert "Explore mode" in str(caught.value)
    assert "AGENT_ENABLED" in str(caught.value)


def test_explore_gets_the_graph_where_the_agent_is_available(tmp_path):
    manager = a_manager(tmp_path, replay_healing_enabled=True, agent_enabled=True)

    assert manager.make_row_runner(None, a_use_case("explore"), "r", None, None)
