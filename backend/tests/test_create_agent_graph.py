"""The rebuilt authoring graph: `create_agent` + `BudgetMiddleware` +
`FinishMiddleware` + `HumanInTheLoopMiddleware`, replacing the hand-rolled
`StateGraph` this file used to test.

Same shape as the old `test_agent_graph.py`: a fake browser and a scripted
model, driving `run_agent_session`/`AgentSession` for real, with nothing
mocked below the graph itself.
"""

from __future__ import annotations

import pytest

from agent import AgentSession, AuthorRequest, Budget, run_agent_session
from agent.tools.finish import NAME as FINISH
from agent.graph import available
from agent.verify import Verification
from llm import LangChainLLM
from scripted_chat_model import ScriptedChatModel, turn_calling
from test_agent_marks import INVITE
from test_agent_tools import FakeMCP

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not available(), reason="langchain is an optional extra"),
]


def a_session(*turns):
    """A provider and a model, scripted. `LangChainLLM` wraps the fake model
    the same way it wraps a real one -- `Wiring.llm` is always this shape,
    and `.raw`/`.model` are what the graph actually reads from it."""
    llm = LangChainLLM(ScriptedChatModel(responses=list(turns)), "scripted-model")
    return FakeMCP({"browser_snapshot": INVITE}), llm


async def _replays(use_case, inputs, secrets):
    return Verification(ran=True, ok=True)


def request(**kwargs) -> AuthorRequest:
    return AuthorRequest(
        task=kwargs.pop("task", "Read the balance for one account."),
        start_url="https://vendor.test/users",
        allowed_domains=("vendor.test",),
        may_write=True,
        run_id=kwargs.pop("run_id", "graph-run"),
        budget=kwargs.pop("budget", Budget()),
        **kwargs,
    )


def a_complete_session():
    return [
        turn_calling("browser_snapshot"),
        turn_calling("mark_setup_complete"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("mark_as_output", ref="e10", column="status"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="Read the status for A-1001."),
    ]


async def _keep(bucket, event):
    bucket.append(event)


# --- the loop runs -----------------------------------------------------


async def test_a_session_runs_end_to_end_through_the_graph():
    provider, llm = a_session(*a_complete_session())
    events = []

    result = await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep(events, e),
        replay=_replays,
    )

    assert result.status == "succeeded", result.stopped_by
    assert result.summary == "Read the status for A-1001."
    assert result.unfinished == "", "it marked a row, so it can be distilled"
    assert [e.type for e in events][0] == "run_started"


async def test_the_trajectory_is_what_distillation_will_read():
    provider, llm = a_session(*a_complete_session())

    result = await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep([], e),
        replay=_replays,
    )

    assert [call["tool"] for call in result.trajectory] == [
        "browser_snapshot",
        "mark_setup_complete",
        "begin_row",
        "mark_as_output",
        "end_row",
    ]


async def test_the_browser_is_closed_when_the_session_ends():
    provider, llm = a_session(*a_complete_session())

    await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep([], e),
        replay=_replays,
    )

    assert provider.closed


async def test_a_call_to_a_registered_servers_tool_never_becomes_a_step():
    """Phase 1's invariant, proved end to end against the new graph: a tool
    from a registered MCP server is for the agent's own use while it works,
    and the distilled use case must never carry a step derived from it --
    only Playwright's own tool names distil into anything at all."""
    from agent.providers import ToolSpec
    from test_agent_tools import FakeExtraServer

    provider, llm = a_session(
        turn_calling("browser_snapshot"),
        turn_calling("mark_setup_complete"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("crm.lookup_account", id="A-1001"),
        turn_calling("mark_as_output", ref="e10", column="status"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="Read the status for A-1001, checked against the CRM."),
    )
    extra = FakeExtraServer([
        ToolSpec("lookup_account", "Look up a CRM account", annotations={"readOnlyHint": True})
    ])

    result = await run_agent_session(
        request(), llm=llm, provider=provider, extra={"crm": extra},
        emit=lambda e: _keep([], e), replay=_replays,
    )

    assert result.status == "succeeded", result.stopped_by
    assert extra.calls == [("lookup_account", {"id": "A-1001"})], "the call did happen"
    assert "crm.lookup_account" in [call["tool"] for call in result.trajectory]

    steps = result.use_case["row_steps"] + result.use_case["setup_steps"]
    assert not any("crm" in str(step) or "lookup_account" in str(step) for step in steps), (
        "a tool call to a registered server leaked into the replayable use case"
    )


# --- the budget ----------------------------------------------------------


async def test_a_step_budget_ends_the_session_rather_than_running_forever():
    provider, llm = a_session(*[turn_calling("browser_snapshot")] * 50)

    result = await run_agent_session(
        request(budget=Budget(steps=4, tokens=None, seconds=None, usd=None)),
        llm=llm, provider=provider, emit=lambda e: _keep([], e), replay=_replays,
    )

    assert result.status == "partial"
    assert "Stopped after 4 steps" in result.stopped_by
    assert result.trajectory, "everything it did before the limit is kept"


async def test_the_budget_is_counted_where_it_is_checked():
    provider, llm = a_session(*[turn_calling("browser_snapshot")] * 10)

    result = await run_agent_session(
        request(budget=Budget(steps=3, tokens=None, seconds=None, usd=None)),
        llm=llm, provider=provider, emit=lambda e: _keep([], e), replay=_replays,
    )

    assert result.spend["steps"] == 3
    assert result.status == "partial"


async def test_a_session_that_stops_at_its_budget_still_says_it_is_over():
    provider, llm = a_session(*[turn_calling("browser_snapshot")] * 10)
    events = []

    await run_agent_session(
        request(budget=Budget(steps=2, tokens=None, seconds=None, usd=None)),
        llm=llm, provider=provider, emit=lambda e: _keep(events, e), replay=_replays,
    )

    assert [e.type for e in events][-1] == "run_finished"
    assert len([e for e in events if e.type == "run_finished"]) == 1


async def test_a_budget_stop_says_how_far_it_actually_got():
    """`runs.status` maps a budget stop ("partial") to "succeeded" -- there is
    a draft either way, worth a look -- which means the status code alone can
    no longer tell a person "ran out having done nothing" apart from "ran out
    with three rows banked". A real session did the former and looked, from
    the status alone, exactly like the latter. This is the difference showing
    up in the one place left for it: the stop message itself.
    """
    provider, llm = a_session(*[turn_calling("browser_snapshot")] * 10)

    result = await run_agent_session(
        request(budget=Budget(steps=2, tokens=None, seconds=None, usd=None)),
        llm=llm, provider=provider, emit=lambda e: _keep([], e), replay=_replays,
    )

    assert result.status == "partial"
    assert "Stopped during setup, before any row began." in result.stopped_by


async def test_a_budget_stop_mid_row_says_the_row_was_left_open():
    provider, llm = a_session(
        turn_calling("browser_snapshot"),
        turn_calling("mark_setup_complete"),
        turn_calling("begin_row", key="A-1001"),
        *[turn_calling("browser_snapshot")] * 10,
    )

    result = await run_agent_session(
        request(budget=Budget(steps=4, tokens=None, seconds=None, usd=None)),
        llm=llm, provider=provider, emit=lambda e: _keep([], e), replay=_replays,
    )

    assert result.status == "partial"
    assert "'A-1001' was left open, unfinished" in result.stopped_by


# --- thinking ----------------------------------------------------------


async def test_what_the_model_says_reaches_the_transcript():
    """`create_agent`'s own nodes read a turn's tool calls and nothing else --
    without `ThinkingMiddleware`, a model's reasoning (ordinary commentary, or
    an extended-thinking block once one is configured) reached nobody. A real
    session went 62 events without a single word of it visible anywhere.
    """
    provider, llm = a_session(
        *a_complete_session()[:1],
        turn_calling(
            "mark_setup_complete",
            thinking="The sign-in form is not on this page; setup is already done.",
        ),
        *a_complete_session()[2:],
    )
    events = []

    await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep(events, e), replay=_replays,
    )

    thoughts = [e.text for e in events if e.type == "thinking"]
    assert "The sign-in form is not on this page; setup is already done." in thoughts


async def test_a_turn_with_nothing_to_say_emits_no_thinking_event():
    provider, llm = a_session(*a_complete_session())
    events = []

    await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep(events, e), replay=_replays,
    )

    assert not [e for e in events if e.type == "thinking"]


# --- the interrupt ---------------------------------------------------------


async def test_an_irreversible_action_suspends_the_graph_and_waits():
    provider, llm = a_session(
        turn_calling("browser_snapshot"),
        turn_calling("browser_click", target="e3", element="Delete this account"),
    )
    events = []

    result = await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep(events, e),
        replay=_replays,
    )

    assert result.status == "awaiting_approval"
    assert result.awaiting is not None
    assert result.awaiting["call"]["name"] == "browser_click"

    asked = next(e for e in events if e.type == "approval_required")
    assert asked.name == "browser_click"
    assert asked.categories, "it says why it stopped"
    assert provider.calls[-1][0] == "browser_snapshot", "the click did not run"
    assert not any(e.type == "run_finished" for e in events), "it is not over"


async def test_resuming_with_an_approval_continues_from_where_it_stopped():
    provider, llm = a_session(
        turn_calling("browser_snapshot"),
        turn_calling("browser_click", target="e3", element="Delete this account"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="Deleted, as approved."),
    )
    events = []

    async with AgentSession(
        request(run_id="resumable"), llm=llm, provider=provider,
        emit=lambda e: _keep(events, e), replay=_replays,
    ) as session:
        paused = await session.start()
        assert paused.status == "awaiting_approval"
        assert provider.calls[-1][0] == "browser_snapshot", "the click did not run"
        assert not provider.closed, "the browser stays open while a person decides"

        done = await session.resume("approved")

    assert "browser_click" in [name for name, _ in provider.calls], "it ran after approval"
    assert done.status in {"succeeded", "partial"}
    assert any(e.type == "approval_resolved" for e in events)
    assert any(e.type == "run_finished" for e in events)
    assert provider.closed


async def test_answering_an_approval_does_not_ask_a_second_time():
    """The bug this whole rewrite exists to fix by construction rather than
    by hand: `HumanInTheLoopMiddleware`'s `after_model` recomputes its
    interrupt payload from already-checkpointed state on every resume, with
    no id it mints and no side effect -- so exactly one `approval_required`
    and one `approval_resolved` come out no matter how the resume happens."""
    provider, llm = a_session(
        turn_calling("browser_snapshot"),
        turn_calling("browser_click", target="e3", element="Delete this account"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="Deleted, as approved."),
    )
    events = []

    async with AgentSession(
        request(run_id="one-ask-only"), llm=llm, provider=provider,
        emit=lambda e: _keep(events, e), replay=_replays,
    ) as session:
        await session.start()
        await session.resume("approved")

    asked = [e for e in events if e.type == "approval_required"]
    resolved = [e for e in events if e.type == "approval_resolved"]
    finished = [e for e in events if e.type == "run_finished"]

    assert len(asked) == 1, "asked a second time on resume"
    assert len(resolved) == 1
    assert asked[0].approval_id == resolved[0].approval_id
    assert len(finished) == 1


def test_the_graph_module_imports_the_optional_extra_only_inside_a_function():
    """`langchain_core` is a base dependency (already required by `llm.py`),
    so importing `tool_adapter`'s `as_langchain_tools` at module scope is
    fine. `langchain` itself -- and anything that imports it, like
    `.middleware` -- is the optional extra and must stay deferred: importing
    this module at all must not require it installed, or `available()` could
    never safely be called to check."""
    import inspect

    from agent import graph as graph_module

    source = inspect.getsource(graph_module)
    top_level = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    assert not any(
        line == "from .middleware import" or "import langchain.agents" in line or "from .middleware" in line
        for line in top_level
    ), top_level


# --- the prompt-cache hint belongs to exactly one provider -----------------
#
# Sending it to the other one did not miss a saving, it failed the call:
# `AsyncCompletions.create() got an unexpected keyword argument
# 'cache_control'`, on the first turn of every OpenRouter session, because
# `ChatOpenAI` hands its keyword arguments to the OpenAI SDK and the SDK
# refuses one it does not know.
#
# `LangChainLLM.run_turn` guards its own copy. The authoring loop reaches the
# model through `create_agent` instead, so it needed its own guard -- which is
# exactly the kind of thing a second code path is for and the reason this test
# covers the middleware rather than the client.


class Request:
    """Enough of `ModelRequest` for the middleware to work on."""

    def __init__(self) -> None:
        self.messages: list = []
        self.model_settings: dict = {}
        #: Read by the turn log, which runs in a `finally` and so runs even on
        #: the error path this exercises.
        self.tools: list = []


async def _settings_after(provider: str) -> dict:
    from agent.budget import Spend
    from agent.marks import Marks
    from agent.middleware import BudgetMiddleware

    middleware = BudgetMiddleware(
        Spend(Budget()), model_name="m", marks=Marks(), provider=provider
    )
    request = Request()

    async def handler(req):
        class Response:
            result = []

        return Response()

    await middleware.awrap_model_call(request, handler)
    return request.model_settings


async def test_bedrock_is_sent_the_prompt_cache_hint():
    """It is worth real money: a measured session spent 127,000 tokens over six
    turns, almost all of it the tool schemas resent verbatim."""
    assert (await _settings_after("bedrock"))["cache_control"] == {"ttl": "5m"}


async def test_openrouter_is_not():
    """The bug, as a test. Any provider but Bedrock must see nothing."""
    assert "cache_control" not in await _settings_after("openrouter")


async def test_an_access_error_names_the_provider_that_refused():
    """It named Bedrock whatever had actually refused the call, which sends
    somebody to check the wrong credentials."""
    from agent.budget import Spend
    from agent.marks import Marks
    from agent.middleware import BudgetMiddleware
    from llm import LLMAccessError

    middleware = BudgetMiddleware(
        Spend(Budget()), model_name="a/b", marks=Marks(), provider="openrouter"
    )

    async def handler(req):
        raise RuntimeError("Error code: 401 - invalid api key")

    with pytest.raises(LLMAccessError, match="openrouter"):
        await middleware.awrap_model_call(Request(), handler)


async def test_a_session_is_costed_at_the_providers_own_rate():
    """The USD ceiling is enforced against this number, so costing an
    OpenRouter model from a default that happens to be Claude Sonnet's would
    stop a cheap session early and let an expensive one run past its budget."""
    from agent.budget import Spend

    usage = {"input_tokens": 1_000_000, "output_tokens": 0}

    table = Spend(Budget())
    table.turn(usage, "meta-llama/llama-3.3-70b")

    published = Spend(Budget())
    published.turn(usage, "meta-llama/llama-3.3-70b", rates=(0.12, 0.30))

    assert table.usd == pytest.approx(3.0), "the table's default"
    assert published.usd == pytest.approx(0.12), "what the provider says"


# --- the two passes that bracket the session ------------------------------
#
# Injected, so a session given no scribe behaves exactly as every test above
# it does. These two prove the wiring, not the passes themselves --
# `test_agent_brief.py` owns those.


def a_scribe():
    """One fake answering both passes: the brief reads `goal`, the
    walkthrough reads `overview`, and each pass is offered only its own tool."""
    from test_agent_brief import Scribe

    return Scribe(
        {
            "goal": "Every account's status is read into the results file.",
            "per_row": [{"name": "account", "means": "the account number"}],
            "overview": "Signs in once, then reads the status for each account.",
        }
    )


async def test_the_brief_reaches_the_recorder_in_its_very_first_message():
    """Before the first snapshot. A per-row value nobody declared cannot be
    marked, and an unmarked value is recorded as a constant."""
    provider, llm = a_session(*a_complete_session())

    await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep([], e),
        replay=_replays, scribe=a_scribe(),
    )

    first_turn = llm.raw.calls[0]
    task_message = str(first_turn[-1].content)
    assert "## The brief" in task_message
    assert "account" in task_message
    assert "Read the balance for one account." in task_message, "the task is still there"


async def test_a_session_with_no_scribe_is_told_exactly_what_it_always_was():
    """Every caller that had no scribe before makes no extra model call and
    sees no empty section where the brief would have been."""
    provider, llm = a_session(*a_complete_session())

    await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep([], e),
        replay=_replays,
    )

    assert "## The brief" not in str(llm.raw.calls[0][-1].content)


async def test_the_brief_is_shown_to_the_person_watching_before_anything_moves():
    """The point of naming what the request left unsaid is that somebody reads
    it, and the only cheap moment to correct an assumption is before the
    browser starts."""
    provider, llm = a_session(*a_complete_session())
    events = []

    await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep(events, e),
        replay=_replays, scribe=a_scribe(),
    )

    kinds = [e.type for e in events]
    thinking = kinds.index("thinking")
    assert thinking < kinds.index("tool_call"), "before the first action"
    assert "Every account's status" in events[thinking].text


async def test_the_flow_is_written_onto_the_draft_the_reviewer_opens():
    provider, llm = a_session(*a_complete_session())

    result = await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep([], e),
        replay=_replays, scribe=a_scribe(),
    )

    assert "Signs in once" in result.use_case["instructions"]
    assert result.brief["goal"].startswith("Every account's status")


async def test_the_two_passes_are_charged_to_the_session_that_made_them():
    """Not free, and not hidden. Both are part of recording, so both land in
    the same spend the budget is enforced against."""
    provider, llm = a_session(*a_complete_session())

    with_scribe = await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep([], e),
        replay=_replays, scribe=a_scribe(),
    )
    provider, llm = a_session(*a_complete_session())
    without = await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep([], e),
        replay=_replays,
    )

    assert with_scribe.spend["tokens"] > without.spend["tokens"]


# --- when the draft is not replayed ---------------------------------------
#
# The replay spends no tokens -- it is the ordinary engine, which has no path
# to a model -- so what it costs is a browser launch and one pass through the
# flow. Two cases where paying that buys nothing.


def a_budget_stop():
    """A session that ran out mid-record, which is what four real sessions in
    a row did. `mark_setup_complete` and nothing after it: no row was ever
    opened, so there is no whole record to replay."""
    return [
        turn_calling("browser_snapshot"),
        turn_calling("mark_setup_complete"),
    ]


async def test_a_session_that_never_finished_a_record_is_not_replayed():
    provider, llm = a_session(*a_budget_stop())
    replayed = []

    async def replay(use_case, inputs, secrets, **kwargs):
        replayed.append(use_case)
        return Verification(ran=True, ok=True)

    result = await run_agent_session(
        request(budget=Budget(steps=2)), llm=llm, provider=provider,
        emit=lambda e: _keep([], e), replay=replay,
    )

    assert replayed == [], "nothing whole to replay"
    assert result.verification["ran"] is False
    assert "stopped before it finished a record" in result.verification["skipped"]


async def test_the_reason_it_was_not_replayed_is_carried_not_implied():
    """"Not verified" with no reason reads as a failure of the verification
    rather than as a session that did not get far enough to have one."""
    provider, llm = a_session(*a_budget_stop())

    result = await run_agent_session(
        request(budget=Budget(steps=2)), llm=llm, provider=provider,
        emit=lambda e: _keep([], e), replay=_replays,
    )

    assert result.verification["skipped"]


async def test_a_deployment_can_switch_verification_off():
    """And is told what it gave up, on the draft itself, rather than being
    left to notice that nothing checked the recording."""
    provider, llm = a_session(*a_complete_session())
    replayed = []

    async def replay(use_case, inputs, secrets, **kwargs):
        replayed.append(use_case)
        return Verification(ran=True, ok=True)

    result = await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep([], e),
        replay=replay, verify_draft=False,
    )

    assert replayed == [], "switched off means no browser is opened"
    assert "AGENT_VERIFY_DRAFT" in result.verification["skipped"]
    assert "first real run" in result.verification["skipped"]
    assert result.use_case is not None, "the draft is still produced"


async def test_a_finished_session_is_still_replayed_by_default():
    """The guarantee this whole phase exists for, unchanged."""
    provider, llm = a_session(*a_complete_session())
    replayed = []

    async def replay(use_case, inputs, secrets, **kwargs):
        replayed.append(use_case)
        return Verification(ran=True, ok=True)

    result = await run_agent_session(
        request(), llm=llm, provider=provider, emit=lambda e: _keep([], e), replay=replay,
    )

    assert len(replayed) == 1
    assert result.verification["ran"] is True
