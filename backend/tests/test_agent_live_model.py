"""One session, against a real model and a real browser. Nothing faked.

Every other agent test drives scripted turns, which is right: the graph, the
guard and the distiller should be testable without spending anything. But a
scripted turn is a claim about how a model behaves, and this file is what stops
that claim going unexamined.

It answers the question the rest of the suite cannot: **does a real model,
given these prompts and these tool schemas, actually record a workflow?** A
graph that works perfectly on turns written by the person who wrote the graph
proves less than it appears to.

Opt in with `RUN_LLM=1`. It **spends money** -- a small amount, a few cents,
bounded by a budget set below -- and needs AWS credentials for Bedrock, Node on
PATH, and Chromium. It is off by default for exactly that reason: no test
should be able to bill somebody by being run.

    RUN_LLM=1 python -m pytest tests/test_agent_live_model.py -q -s
"""

from __future__ import annotations

import os

import pytest

from agent import AuthorRequest, Budget, LocalPlaywrightMCP, run_agent_session
from agent.graph import available
from agent.verify import Verification

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.llm,
    pytest.mark.skipif(
        os.environ.get("RUN_LLM") != "1",
        reason="set RUN_LLM=1 to run against a real Bedrock model (this spends money)",
    ),
    pytest.mark.skipif(not available(), reason="LangGraph is an optional extra"),
]


#: A page as a data URL, so this needs no server and cannot fail because one is
#: down. Deliberately trivial: what is being tested is whether the model can
#: follow the *protocol* -- snapshot, act by ref, mark, finish -- not whether it
#: can solve anything hard.
PAGE = (
    "data:text/html,<h1>Account</h1>"
    "<label>Account number <input id=a></label>"
    "<button onclick=\"document.getElementById('bal').textContent='1,240.55'\">"
    "Look up</button>"
    "<p>Balance: <span id=bal></span></p>"
)


async def test_a_real_model_records_a_workflow_that_replays():
    """The whole arc, with nothing scripted.

    A model is given a task and the tools, drives a real browser through
    Playwright MCP, marks what varies and what to read, and the draft it
    produces is replayed by the engine. If the prompts are wrong, the tool
    schemas confusing, or the marking gesture unnatural to a model, this is
    where it shows.
    """
    from config import Settings
    from llm import build_llm

    settings = Settings()
    llm = build_llm(settings)

    events: list = []
    verified: dict = {}

    async def keep(event):
        events.append(event)

    async def replay(use_case, inputs, secrets):
        # Verification is stubbed so a failure here is unambiguous: it means
        # the *model* did not produce something replayable, not that a browser
        # would not start. The real replay is covered in test_agent_mcp_live.
        verified["use_case"] = use_case
        verified["inputs"] = inputs
        return Verification(ran=True, ok=True)

    result = await run_agent_session(
        AuthorRequest(
            task=(
                "Type the account number A-1001 into the account field, press "
                "Look up, and read the balance that appears."
            ),
            start_url=PAGE,
            allowed_domains=("data",),
            may_write=True,
            # Measured rather than guessed. A session of this shape takes
            # 13-15 calls at roughly 7,000 tokens each -- most of it the tool
            # schemas and the system prompt, which are re-sent every call --
            # so 90,000 stopped it *after* doing the task and *before* it
            # could close the row. Sized to finish, and still capped in
            # dollars, because the failure mode of a generous budget is a bill
            # nobody chose.
            budget=Budget(steps=30, tokens=200_000, seconds=300, usd=0.90),
        ),
        llm=llm,
        provider=LocalPlaywrightMCP(headless=True),
        emit=keep,
        replay=replay,
        name="Read a balance",
    )

    print(f"\nspent: {result.spend}")
    print(f"status: {result.status} — {result.summary or result.stopped_by}")
    for call in result.trajectory:
        print(f"  {call['tool']:24} {call.get('element') or ''}")
    for warning in result.draft_warnings:
        print(f"  ! {warning}")

    assert result.status in {"succeeded", "partial"}, result.stopped_by
    assert not result.unfinished, (
        f"the session could not be distilled: {result.unfinished}"
    )
    assert result.use_case is not None, "the session produced no draft at all"

    # The marking gesture is the part most likely to be unnatural to a model,
    # because it is ours rather than something it has seen a thousand times.
    kinds = {mark["kind"] for mark in result.marks}
    assert "begin_row" in kinds, (
        "the model never marked a row, so nothing could be distilled. If this "
        "fails the prompt is not landing, not the graph."
    )

    steps = result.use_case["row_steps"]
    assert steps, "a row with no steps is not a recording"
    assert result.spend["usd"] > 0, "a real session that cost nothing did not happen"


#: A page that proves, without ever showing the real value to anything that
#: would store or print it, that the value which reached the browser was the
#: real one. The title becomes "MATCH" only on an exact match, so seeing
#: "MATCH" is proof the substitution worked, and the assertion for "never
#: printed the secret" can be made on the very same trajectory.
CREDENTIAL_PAGE = (
    "data:text/html,<h1>Sign in</h1>"
    "<label>Login code <input id=a oninput=\""
    "document.title = (this.value === 'Live-Bedrock-Check-9f3') ? 'MATCH' : 'NOMATCH'"
    "\"></label>"
)
REAL_SECRET = "Live-Bedrock-Check-9f3"


async def test_a_real_model_types_a_bound_credential_it_never_sees():
    """The bug two real sessions hit before this fix: told to sign in and
    given nothing to type, a model fabricated "admin" / "password". It never
    saw a real value either time -- there was no route from a bound
    credential to a typed character at all.

    This proves the route exists against a real model: the credential is
    typed correctly, and the real value never appears anywhere this test can
    see -- not in the trajectory, not in the summary, not in what gets
    printed below.
    """
    from config import Settings
    from llm import build_llm

    events: list = []

    result = await run_agent_session(
        AuthorRequest(
            task=(
                "Type the login code into the field using the credential "
                "provided. Afterwards take a snapshot and mark_as_output "
                "whatever the page title says, calling the column 'check'."
            ),
            start_url=CREDENTIAL_PAGE,
            allowed_domains=("data",),
            secrets=("login",),
            may_write=True,
            budget=Budget(steps=15, tokens=200_000, seconds=180, usd=0.60),
        ),
        llm=build_llm(Settings()),
        provider=LocalPlaywrightMCP(headless=True),
        emit=lambda e: _keep(events, e),
        secrets={"login": REAL_SECRET},
        replay=_ok,
        name="Types a credential",
    )

    dump = str(result.trajectory) + str(result.summary) + str(result.marks)
    print(f"\nspent: {result.spend}")
    for call in result.trajectory:
        print(f"  {call['tool']:24} {call.get('detail','')[:80]!r}")

    assert REAL_SECRET not in dump, "the real value leaked into the trail"
    assert any(
        "MATCH" in str(call.get("detail", "")) and "NOMATCH" not in str(call.get("detail", ""))
        for call in result.trajectory
    ), "the page never confirmed the typed value matched the real credential"


async def _keep(bucket, event):
    bucket.append(event)


async def test_a_real_model_refuses_to_leave_the_allowlist():
    """The guard is enforcement, not instruction -- but a model that fights it
    for twenty turns is a prompt problem, and this is where that shows."""
    from config import Settings
    from llm import build_llm

    result = await run_agent_session(
        AuthorRequest(
            task="Go to https://example.com and read the heading.",
            start_url=PAGE,
            allowed_domains=("data",),
            may_write=True,
            budget=Budget(steps=8, tokens=40_000, seconds=120, usd=0.30),
        ),
        llm=build_llm(Settings()),
        provider=LocalPlaywrightMCP(headless=True),
        emit=_ignore,
        replay=_ok,
        name="Blocked",
    )

    refused = [call for call in result.trajectory if call["refused"]]
    print(f"\nrefused {len(refused)} call(s); spent {result.spend}")
    assert not any(
        "example.com" in str(call.get("detail", "")) and call["ok"]
        for call in result.trajectory
    ), "a call reached a domain outside the allowlist"


async def _ignore(event):
    return None


async def _ok(use_case, inputs, secrets):
    return Verification(ran=True, ok=True)
