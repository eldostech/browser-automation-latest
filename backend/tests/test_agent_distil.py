"""The bridge: a session becomes a document, and the document is proved.

Everything before this phase is a browser-using chatbot with an audit log.
These tests cover the two halves that change that.

**Distillation is bookkeeping**, and that is the achievement. The hard version
-- read four hundred near-identical sub-trajectories and work out where a
record's work begins -- is a question the agent was asked while it still had
the page in front of it, so this cuts on declared boundaries instead of
guessing at repeated shapes.

**Verification is what makes it trustworthy.** The draft is replayed by the
engine, from a cold start, before anybody sees it.
"""

from __future__ import annotations

import pytest

from agent import distil, verify
from agent.tools.finish import NAME as FINISH
from agent.verify import Verification
from llm import LangChainLLM
from scripted_chat_model import ScriptedChatModel, turn_calling
from test_agent_marks import INVITE
from test_agent_tools import FakeMCP
from test_create_agent_graph import request

pytestmark = pytest.mark.anyio


#: A page with a field to type into and a value to read, which is the smallest
#: thing that exercises an input and an output at once.
ACCOUNT = """### Page
- Page URL: https://vendor.test/accounts
### Snapshot
```yaml
- generic [ref=e1]:
  - textbox "Account" [ref=e2]
  - button "Open" [ref=e3]
  - text "Balance" [ref=e4]
```
"""


def a_recording():
    """Sign in, then one record: type an account, open it, read the balance."""
    return [
        turn_calling("browser_navigate", url="https://vendor.test/login"),
        turn_calling("browser_snapshot"),
        turn_calling("browser_type", target="e2", text="secret-sign-in-value"),
        turn_calling("mark_as_secret", ref="e2", slot="vendor_login"),
        turn_calling("mark_setup_complete"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("browser_type", target="e2", text="A-1001"),
        turn_calling("mark_as_input", ref="e2", name="account"),
        turn_calling("browser_click", target="e3"),
        turn_calling("mark_as_output", ref="e4", column="balance"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="Read the balance for A-1001."),
    ]


#: Two cards, each with a "Chat" button sharing the same role and name --
#: the shape that produced a real failure: an agent clicked the right one by
#: ref, and the durable locator distilled from that click matched both.
CHAT_LIST = """### Page
- Page URL: https://vendor.test/projects
### Snapshot
```yaml
- generic [ref=e1]:
  - generic [ref=e2]:
    - text "Project A" [ref=e3]
    - button "Chat" [ref=e4]
    - button "Open" [ref=e5]
  - generic [ref=e6]:
    - text "Project B" [ref=e7]
    - button "Chat" [ref=e8]
```
"""


async def test_an_ambiguous_click_resolved_by_position_gets_a_mild_note():
    """Found for real, in two parts. First: an agent clicked the right "Chat"
    button among several identical ones by ref -- that click always runs,
    since a ref is position-specific, not name-based. Second, found only
    after the first fix shipped: `describe_element` (agent/marks.py) had
    already learned to attach the click's position among the matches
    (`nth`), which is what actually lets a replay resolve this instead of
    refusing -- but this file's own warning still fired the *old*, scarier
    message ("will refuse to guess... re-record this step") regardless,
    telling a reviewer to redo a step that already worked. e8 is the *second*
    of the two "Chat" buttons, so this checks the corrected, milder note."""
    from agent import run_agent_session

    provider = FakeMCP({"browser_snapshot": CHAT_LIST})
    llm = LangChainLLM(
        ScriptedChatModel(responses=[
            turn_calling("browser_snapshot"),
            turn_calling("mark_setup_complete"),
            turn_calling("begin_row", key="project-b"),
            # Twice, because a click whose only distinguishing feature is
            # its position is now refused once at record time -- the page is
            # still on screen and something scoped is usually available. Here
            # there is not, so the repeat is the agent saying so, and the step
            # is recorded with the milder note this test is about.
            turn_calling("browser_click", target="e8", element="Chat button for Project B"),
            turn_calling("browser_click", target="e8", element="Chat button for Project B"),
            turn_calling("mark_as_output", ref="e5", column="open_label"),
            turn_calling("end_row"),
            turn_calling(FINISH, summary="Opened chat for Project B."),
        ]),
        "scripted-model",
    )

    result = await run_agent_session(
        request(), llm=llm, provider=provider, emit=_ignore,
        replay=_replays_cleanly, name="Open a project's chat",
    )

    assert result.use_case is not None
    # One click recorded, not two: the refused attempt never reached the
    # browser and never became a step.
    clicks = [s for s in result.use_case["row_steps"] if s["action"] == "click"]
    assert len(clicks) == 1
    click_step = clicks[0]
    assert click_step["locators"][0]["name"] == "Chat"
    assert click_step["locators"][0]["nth"] == 1, "e8 is the second of the two matches"
    warning = next(w for w in result.draft_warnings if click_step["id"] in w)
    assert "2nd one" in warning
    assert "by position on the page" in warning
    assert "will refuse to guess" not in warning, "this step will replay correctly"


async def test_the_first_of_an_ambiguous_pair_still_gets_the_loud_warning():
    """The other half of the same fix: `nth` cannot express "specifically the
    first" (it is indistinguishable from "no position given" -- see
    describe_element's own docstring), so a click on the *first* of the
    duplicates is genuinely still unresolved, and must keep the strong
    warning telling a reviewer to re-record it."""
    from agent import run_agent_session

    provider = FakeMCP({"browser_snapshot": CHAT_LIST})
    llm = LangChainLLM(
        ScriptedChatModel(responses=[
            turn_calling("browser_snapshot"),
            turn_calling("mark_setup_complete"),
            turn_calling("begin_row", key="project-a"),
            turn_calling("browser_click", target="e4", element="Chat button for Project A"),
            turn_calling("mark_as_output", ref="e5", column="open_label"),
            turn_calling("end_row"),
            turn_calling(FINISH, summary="Opened chat for Project A."),
        ]),
        "scripted-model",
    )

    result = await run_agent_session(
        request(), llm=llm, provider=provider, emit=_ignore,
        replay=_replays_cleanly, name="Open a project's chat",
    )

    assert result.use_case is not None
    click_step = next(s for s in result.use_case["row_steps"] if s["action"] == "click")
    assert click_step["locators"][0]["nth"] == 0, "e4 is the first of the two matches"
    warning = next(w for w in result.draft_warnings if click_step["id"] in w)
    assert "matched 2 elements" in warning
    assert "will refuse to guess" in warning
    assert "re-record this step" in warning


#: The shape of the real production failure: each project "card" is a bare
#: `generic` element with no accessible name at all -- no button, no link,
#: nothing to give position a name to be a fallback from. A real replay found
#: this same page exposing zero generic-role elements minutes later, proving
#: the match count was never a stable property of the page.
GENERIC_CARDS = """### Page
- Page URL: https://vendor.test/projects
### Snapshot
```yaml
- list [ref=e1]:
  - generic [ref=e2]
  - generic [ref=e3]
  - generic [ref=e4]
```
"""


async def test_a_click_on_an_unnamed_generic_wrapper_gets_the_unreliable_warning():
    """Distinct from, and worse than, the "2nd of the Chat buttons" case above:
    `nth` still gets computed, but resolving by position among anonymous,
    unnamed structural elements is not something a replay should be told is
    safe. This step must not get the mild "by position on the page" note even
    though its leading rung has `nth > 0` -- it must get the strong,
    differently-worded warning telling a reviewer to re-record against
    something with a real name."""
    from agent import run_agent_session

    provider = FakeMCP({"browser_snapshot": GENERIC_CARDS})
    llm = LangChainLLM(
        ScriptedChatModel(responses=[
            turn_calling("browser_snapshot"),
            turn_calling("mark_setup_complete"),
            turn_calling("begin_row", key="project-b"),
            # Refused the first time: the wrapper cannot be *described*, and
            # the agent is told so while the page is still in front of it.
            turn_calling("browser_click", target="e3", element="second project card"),
            # Repeated, which is the agent saying there is nothing better on
            # this page -- and on this page there genuinely is not. It goes
            # through, carrying the warning a reviewer has to act on.
            turn_calling("browser_click", target="e3", element="second project card"),
            turn_calling("end_row"),
            turn_calling(FINISH, summary="Opened the second project."),
        ]),
        "scripted-model",
    )

    result = await run_agent_session(
        request(), llm=llm, provider=provider, emit=_ignore,
        replay=_replays_cleanly, name="Open a project",
    )

    assert result.use_case is not None
    click_step = next(s for s in result.use_case["row_steps"] if s["action"] == "click")
    assert click_step["locators"][0]["role"] == "generic"
    assert click_step["locators"][0]["nth"] == 1, "e3 is the second of the three matches"
    warning = next(w for w in result.draft_warnings if click_step["id"] in w)
    assert "not a real control" in warning
    assert "is not something a replay can trust" in warning
    assert "by position on the page" not in warning, "the milder nth-resolved note must not fire here"
    assert "Re-record this step" in warning


async def a_run(*, replay=None, turns=None):
    from agent import run_agent_session

    provider = FakeMCP({name: ACCOUNT for name in
                        ("browser_snapshot", "browser_navigate", "browser_type", "browser_click")})
    llm = LangChainLLM(ScriptedChatModel(responses=list(turns or a_recording())), "scripted-model")
    return await run_agent_session(
        request(),
        llm=llm,
        provider=provider,
        emit=_ignore,
        replay=replay or _replays_cleanly,
        name="Pull balances",
    )


async def _ignore(event):
    return None


async def _replays_cleanly(use_case, inputs, secrets):
    return Verification(ran=True, ok=True, outputs={"balance": "1,240.55"}, duration_ms=6200)


# --- what a session becomes ------------------------------------------------


async def test_a_session_becomes_a_draft_use_case():
    result = await a_run()

    assert result.use_case is not None
    assert result.use_case["status"] == "draft", "a person reviews it, always"
    assert result.use_case["authored_by"] == "agent"
    assert result.use_case["name"] == "Pull balances"


async def test_setup_and_row_are_split_where_the_agent_said_they_were():
    """The boundary is declared, not inferred. Get it wrong and a batch signs
    in four thousand times."""
    result = await a_run()

    setup = [s["action"] for s in result.use_case["setup_steps"]]
    row = [s["action"] for s in result.use_case["row_steps"]]

    assert setup == ["navigate", "fill"], "the sign-in runs once per batch"
    assert row == ["fill", "click", "extract"], "the work runs once per row"


async def test_a_marked_value_becomes_a_template_and_a_declared_input():
    result = await a_run()

    fill = result.use_case["row_steps"][0]
    assert fill["value"] == "{{input.account}}"
    assert [i["name"] for i in result.use_case["inputs"]] == ["account"]


async def test_a_marked_credential_never_carries_its_value():
    """The slot travels; the secret does not. Same rule the replay path has."""
    result = await a_run()

    sign_in = result.use_case["setup_steps"][1]
    assert sign_in["value"] == "{{secret.vendor_login}}"
    assert "secret-sign-in-value" not in str(result.use_case)
    assert [s["name"] for s in result.use_case["secrets"]] == ["vendor_login"]


async def test_a_read_lands_where_it_was_pointed_at():
    """Position is correctness. Appending every reading to the end would read
    the first page's field after the browser had moved to the third -- the
    same mistake the codegen path had to fix, reached from the other side."""
    result = await a_run()

    assert [s["action"] for s in result.use_case["row_steps"]] == [
        "fill", "click", "extract",
    ]
    assert result.use_case["outputs"] == ["balance"]


async def test_the_steps_carry_durable_locators_not_refs():
    """A ref is an index into one snapshot. This is the whole reason the
    session resolves one before every call rather than after."""
    result = await a_run()

    click = result.use_case["row_steps"][1]
    assert click["locators"], "a click with no locator cannot replay"
    assert click["locators"][0]["role"] == "button"
    assert click["locators"][0]["name"] == "Open"
    assert "e3" not in str(click["locators"])


async def test_snapshots_and_marks_do_not_become_steps():
    """They are how it *found* the way, not the way. Replaying an exploration
    four thousand times is four thousand wasted page loads."""
    result = await a_run()

    actions = [s["action"] for s in result.use_case["setup_steps"] + result.use_case["row_steps"]]
    assert "snapshot" not in actions
    assert len(result.trajectory) > len(actions), "the trajectory keeps more than the steps"


# --- refusing to guess -----------------------------------------------------


async def test_one_record_is_recorded_and_said_to_be_one_record():
    """"These steps worked once" is not "these steps are the same every time",
    and a reviewer should be told which they have."""
    result = await a_run()

    assert any("Only one record" in w for w in result.draft_warnings)


async def test_a_second_record_taking_a_different_route_is_a_warning():
    """The gift of asking for two. A mismatch is a warning on the review
    screen, which beats a silent guess at which shape was meant."""
    turns = a_recording()[:-1] + [
        turn_calling("begin_row", key="A-1002"),
        turn_calling("browser_type", target="e2", text="A-1002"),
        # No click this time: a different route.
        turn_calling("end_row"),
        turn_calling(FINISH, summary="Two records."),
    ]
    result = await a_run(turns=turns)

    assert any("did not take the same route" in w for w in result.draft_warnings)


async def test_a_value_marked_on_an_element_nothing_typed_into_is_reported():
    """Rather than a use case with a declared input no step reads -- which the
    publish validator would reject later, with less to say about why."""
    turns = [
        turn_calling("browser_snapshot"),
        turn_calling("mark_setup_complete"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("mark_as_input", ref="e2", name="account"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="Nothing typed."),
    ]
    result = await a_run(turns=turns)

    assert any("nothing recorded typing" in w for w in result.draft_warnings)
    assert result.use_case["inputs"] == [], "an input nothing reads is not declared"


async def test_a_session_with_no_row_is_still_distilled_with_the_reason():
    """Throwing it away loses the expensive part. A person can read the steps
    and fix the boundary by hand."""
    turns = [
        turn_calling("browser_snapshot"),
        turn_calling(FINISH, summary="Never marked a row."),
    ]
    result = await a_run(turns=turns)

    assert result.use_case is not None
    assert any("No row was recorded" in w for w in result.draft_warnings)


# --- verification ----------------------------------------------------------


async def test_a_draft_that_replays_says_so_with_what_it_read():
    result = await a_run()

    assert result.verification["ran"] is True
    assert result.verification["ok"] is True
    assert result.verification["outputs"] == {"balance": "1,240.55"}


async def test_a_draft_that_does_not_replay_says_so_first():
    """This is the line a reviewer reads before anything else, so it goes at
    the top of the warnings rather than somewhere in the report."""
    async def broken(use_case, inputs, secrets):
        return Verification(
            ran=True, ok=False, failed_step="a9",
            error="no element matched. Tried: role=button name=\"Open\".",
        )

    result = await a_run(replay=broken)

    assert result.verification["ok"] is False
    assert "Did not replay" in result.draft_warnings[0]
    assert result.use_case is not None, "the recording is kept either way"


async def test_verification_is_given_the_values_that_were_actually_typed():
    """The record the agent worked through is the only row whose answer is
    known, so it is the one the draft is checked against."""
    seen = {}

    async def capture(use_case, inputs, secrets):
        seen.update(inputs)
        return Verification(ran=True, ok=True)

    await a_run(replay=capture)

    assert seen == {"account": "A-1001"}


async def test_a_draft_missing_a_value_is_not_verified_and_says_why():
    """Reporting "not verified, and here is the reason" beats reporting a
    failure that is really a gap in what was captured."""
    from usecase import InputSpec, Locator, Step, UseCase

    use_case = UseCase(
        name="x",
        inputs=[InputSpec(name="account")],
        row_steps=[
            Step(id="s1", action="fill", value="{{input.account}}",
                 locators=[Locator(strategy="role", role="textbox", name="Account")]),
        ],
    )

    report = await verify(use_case, {}, {})

    assert not report.ran
    assert "no value was captured" in report.skipped
    assert "Not verified" in report.as_text()


async def test_a_verification_that_cannot_run_is_a_result_not_a_crash():
    """A browser that will not start must not look like a use case that does
    not work."""
    async def explodes(use_case, inputs, secrets):
        raise RuntimeError("no browser on this machine")

    report = await verify(
        _one_step_use_case(), {"account": "A-1"}, {}, replay=explodes
    )

    assert not report.ran
    assert "no browser" in report.skipped


def _one_step_use_case():
    from usecase import InputSpec, Locator, Step, UseCase

    return UseCase(
        name="x",
        inputs=[InputSpec(name="account")],
        row_steps=[
            Step(id="s1", action="fill", value="{{input.account}}",
                 locators=[Locator(strategy="role", role="textbox", name="Account")]),
        ],
    )


async def test_a_credential_is_never_kept_as_a_sample():
    """The slot goes into the document; the value goes into the vault.

    Keeping it beside the draft "so verification can run" would be a plaintext
    password in a table nobody thinks of as a secret store.
    """
    seen = {}

    async def capture(use_case, inputs, secrets):
        seen.update(inputs)
        return Verification(ran=True, ok=True)

    result = await a_run(replay=capture)

    assert "vendor_login" not in seen
    assert "secret-sign-in-value" not in str(seen)
    assert "secret-sign-in-value" not in str(result.use_case)


async def test_going_back_records_where_it_went():
    """`browser_navigate_back` says nothing about its destination, and the page
    it landed on is the only thing that does. Without this, "back to the list"
    is a step a replay cannot perform."""
    turns = [
        turn_calling("browser_snapshot"),
        turn_calling("mark_setup_complete"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("browser_click", target="e3"),
        turn_calling("browser_navigate_back"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="Opened and went back."),
    ]
    result = await a_run(turns=turns)

    back = result.use_case["row_steps"][-1]
    assert back["action"] == "navigate"
    assert back["url"] == "https://vendor.test/users", "where it landed"


# --- what a real model actually did ----------------------------------------


def test_a_column_named_the_way_a_person_would_is_made_usable():
    """Found by the live test, at real cost.

    Asked to name the column for a field labelled "Account number", the model
    answered "Account number" -- the right answer to the question, and not an
    identifier. It reached InputSpec and raised mid-distillation, losing a
    session that had otherwise gone perfectly.
    """
    from agent.marks import as_name

    assert as_name("Account number") == "account_number"
    assert as_name("balance") == "balance"
    assert as_name("Ref #") == "ref"
    assert as_name("2024 Total") == "_2024_total", "an identifier cannot start with a digit"
    assert as_name("   ") == ""


async def test_the_agent_is_told_the_name_it_actually_got():
    """Answering "recorded" leaves it using its own spelling in the next call
    and in its summary, and then two names for one column are loose."""
    turns = [
        turn_calling("browser_snapshot"),
        turn_calling("mark_setup_complete"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("browser_type", target="e2", text="A-1001"),
        turn_calling("mark_as_input", ref="e2", name="Account number"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="done"),
    ]
    result = await a_run(turns=turns)

    assert [i["name"] for i in result.use_case["inputs"]] == ["account_number"]
    said = next(c for c in result.trajectory if c["tool"] == "mark_as_input")["detail"]
    assert "account_number" in said


async def test_distillation_never_loses_a_session_to_a_schema_error():
    """A session is the expensive part. A model drove a browser for two minutes
    and a person watched it; losing all of that at the last step is the worst
    possible way to spend it.

    A draft that will not validate is rebuilt without the parts that would not,
    and the reviewer is told which -- a missing column name is a minute's work,
    a vanished session is not.
    """
    from agent.distil import _build

    warnings: list[str] = []
    use_case = _build(
        {
            "name": "Salvageable",
            "status": "draft",
            "inputs": [],
            "outputs": ["never extracted by any step"],
            "warnings": warnings,
        },
        warnings,
    )

    assert use_case is not None, "it raised instead of degrading"
    assert use_case.name == "Salvageable"
    assert any("could not be turned into" in w for w in warnings)


async def test_typing_a_per_row_value_before_the_row_began_moves_into_it():
    """Found by a real session, and the fix is not leniency.

    A model works the way a person would: do the task, then say what the parts
    were. So the typing lands before `mark_setup_complete`, in setup -- and a
    setup step referencing {{input.x}} is refused by the schema, correctly,
    because setup runs once per batch and there is no row to take a value from.

    The mark is a statement of fact: this value changes per record. So the step
    that types it is row work by definition, and moving it acts on what the
    agent said rather than guessing at what it meant.
    """
    turns = [
        turn_calling("browser_snapshot"),
        turn_calling("browser_type", target="e2", text="A-1001"),
        turn_calling("browser_click", target="e3"),
        turn_calling("mark_setup_complete"),
        turn_calling("begin_row", key="A-1001"),
        turn_calling("mark_as_input", ref="e2", name="account"),
        turn_calling("end_row"),
        turn_calling(FINISH, summary="did it, then said what it was"),
    ]
    result = await a_run(turns=turns)

    assert result.use_case is not None
    row = [s["action"] for s in result.use_case["row_steps"]]
    setup = [s["action"] for s in result.use_case["setup_steps"]]

    assert "fill" in row, "the step that types a per-row value has to be in the row"
    assert "fill" not in setup
    assert any("before the row began" in w for w in result.draft_warnings)
    assert [i["name"] for i in result.use_case["inputs"]] == ["account"]


# ---------------------------------------------------------------------------
# A recorded wait
# ---------------------------------------------------------------------------
#
# `browser_wait_for` was mapped to a `wait` action and then nothing built the
# condition, so every wait an agent performed produced a step the schema
# refuses -- and the exception came out of `distil` and took the whole session
# with it. A real session was lost to `{"time": 2}`, twice.


def a_call(seq, name, action, arguments, locators=(), page_url="https://x.test/"):
    from agent.session import ToolCallRecord

    return ToolCallRecord(
        seq=seq,
        name=name,
        arguments=dict(arguments),
        ok=True,
        action=action,
        locators=list(locators),
        element="",
        page_url=page_url,
    )


def test_a_recorded_pause_becomes_a_timed_wait():
    from agent.distil import _steps

    steps = _steps([a_call(36, "browser_wait_for", "wait", {"time": 2})], {}, [])

    assert len(steps) == 1
    assert steps[0].wait_for is not None
    assert steps[0].wait_for.kind == "time"
    assert steps[0].wait_for.seconds == 2.0


def test_waiting_for_text_beats_waiting_for_a_number_of_seconds():
    """The agent sends both. Seen for real:
    `{"text": "architecture", "time": 2, "textGone": ""}`.

    A replay that waits for the text waits exactly as long as the page needs;
    one that sleeps for the recorded two seconds is guessing the next page is
    no slower than this one was, which is the guess every flaky replay rests
    on.
    """
    from agent.distil import _steps

    steps = _steps(
        [
            a_call(
                125,
                "browser_wait_for",
                "wait",
                {"text": "architecture", "time": 2, "textGone": ""},
            )
        ],
        {},
        [],
    )

    assert steps[0].wait_for.kind == "text"
    assert steps[0].wait_for.value == "architecture"


def test_waiting_for_text_to_go_is_recorded_as_that():
    from agent.distil import _steps

    steps = _steps(
        [a_call(1, "browser_wait_for", "wait", {"textGone": "Loading"})], {}, []
    )

    assert steps[0].wait_for.kind == "text_gone"
    assert steps[0].wait_for.value == "Loading"


def test_a_wait_that_named_no_condition_is_left_out_and_said_so():
    """Turning it into a sleep nobody asked for would be inventing a step."""
    from agent.distil import _steps

    warnings: list[str] = []
    steps = _steps([a_call(1, "browser_wait_for", "wait", {})], {}, warnings)

    assert steps == []
    assert len(warnings) == 1
    assert "named nothing to wait for" in warnings[0]


def test_an_absurdly_long_pause_is_clamped_rather_than_refused():
    """A recorded wait longer than the schema allows is a person's patience,
    not a requirement, and losing the step over it is the worse trade."""
    from agent.distil import _steps

    steps = _steps([a_call(1, "browser_wait_for", "wait", {"time": 900})], {}, [])

    assert steps[0].wait_for.seconds == 120.0


def test_one_call_the_schema_refuses_no_longer_destroys_the_session():
    """The codegen recorder has always reported an unrepresentable line and
    kept the rest. This path raised instead, so one bad call threw away a
    session somebody had just spent ten minutes driving -- and the only thing
    they could do with it was delete it.
    """
    from agent.distil import _steps

    warnings: list[str] = []
    calls = [
        a_call(
            1,
            "browser_click",
            "click",
            {"target": "e1"},
            [{"strategy": "role", "role": "button", "name": "Search"}],
        ),
        # `extract` requires an output name, and nothing here supplies one --
        # a stand-in for any call the schema will not take.
        a_call(2, "browser_snapshot", "extract", {}),
        a_call(
            3,
            "browser_click",
            "click",
            {"target": "e2"},
            [{"strategy": "role", "role": "button", "name": "Next"}],
        ),
    ]

    steps = _steps(calls, {}, warnings)

    assert [step.id for step in steps] == ["a1", "a3"], "the rest survives"
    assert any("could not be recorded as a step" in w for w in warnings)


def test_the_reason_a_call_was_left_out_describes_that_call():
    """"Nothing says where it went" is true of a navigation with no URL and
    nonsense about a wait. A warning that describes the wrong problem sends a
    reviewer looking in the wrong place."""
    from agent.distil import _steps

    warnings: list[str] = []
    _steps(
        [
            a_call(1, "browser_wait_for", "wait", {}),
            a_call(2, "browser_navigate_back", "navigate", {}, page_url=""),
        ],
        {},
        warnings,
    )

    assert "nothing to wait for" in warnings[0]
    assert "nowhere to go" in warnings[1]


# --- the session that came back with no steps at all ----------------------
#
# Found in production, and the worst shape a failure can take: the agent
# signed in, solved the task, signed out, and the draft had nothing in it. The
# person saw a perfect transcript beside an empty use case.
#
# The chain: `mark_as_secret` was refused because the ref resolved to a
# locator matching three elements, so the slot was never declared. The step
# that typed `{{secret.secretword}}` stayed. One undeclared reference fails
# validation -- and the recovery then dropped `secrets`, which left *every*
# `{{secret.x}}` undeclared, failed again, and ran out at an empty document.


def _typed_but_unmarked():
    """A value typed as a placeholder whose mark did not go through."""
    from agent.marks import Described, Marks
    from agent.session import ToolCallRecord
    from usecase import Locator

    ladder = [
        Locator(strategy="role", role="textbox", name="Username").model_dump(
            mode="json", exclude_none=True
        )
    ]

    def call(seq, action, name, **kw):
        return ToolCallRecord(seq=seq, name=name, ok=True, action=action, **kw)

    calls = [
        call(1, "navigate", "browser_navigate", arguments={"url": "https://ixl.test/signin"}),
        call(2, "fill", "browser_type", arguments={"target": "e1", "text": "{{secret.login}}"},
             locators=ladder, element='role=textbox name="Username"'),
        call(3, "", "mark_as_secret", arguments={"ref": "e1", "slot": "login"}),
        # Typed. The mark came back "matches 3 elements" and was refused, so
        # nothing declared the slot.
        call(4, "fill", "browser_type", arguments={"target": "e2", "text": "{{secret.secretword}}"},
             locators=ladder, element='role=textbox name="Secret word"'),
        call(5, "", "mark_setup_complete", arguments={}),
        call(6, "", "begin_row", arguments={"key": "row-1"}),
        call(7, "fill", "browser_type", arguments={"target": "e3", "text": "490"},
             locators=ladder, element='role=textbox name="answer"'),
        call(8, "click", "browser_click", arguments={"target": "e4"},
             locators=ladder, element='role=button name="Submit"'),
        call(9, "", "end_row", arguments={}),
    ]

    marks = Marks()
    described = Described(
        ref="e1", role="textbox", name="Username",
        ladder=[Locator.model_validate(ladder[0])],
    )
    marks.mark_value("mark_as_secret", 3, "e1", "login", described)
    marks.setup_complete(5)
    marks.begin_row(6, "row-1")
    marks.end_row(9)
    return calls, marks


def _draft_of(calls, marks):
    from agent.distil import distil

    return distil(
        calls, marks, name="IXL", task="solve one problem",
        start_url="https://ixl.test", allowed_domains=("ixl.test",),
    )


def test_a_typed_secret_whose_mark_was_refused_does_not_empty_the_draft():
    """The regression, stated as the thing the person actually lost."""
    draft = _draft_of(*_typed_but_unmarked())

    assert len(draft.use_case.setup_steps) == 3
    assert len(draft.use_case.row_steps) == 2


def test_the_slot_is_declared_from_the_step_that_types_it():
    """A step typing `{{secret.x}}` is the recording saying it needs a slot
    called x. A mark is refused for reasons that have nothing to do with
    whether the value is a credential."""
    draft = _draft_of(*_typed_but_unmarked())

    assert [s.name for s in draft.use_case.secrets] == ["login", "secretword"]


def test_and_the_reviewer_is_told_which_slot_to_bind():
    """A declared slot nobody fills is refused later by name, which is a
    minute's work. A draft with no steps is not."""
    draft = _draft_of(*_typed_but_unmarked())

    said = " ".join(draft.warnings)
    assert "secretword" in said
    assert "Bind a credential" in said


def test_dropping_secrets_can_never_be_the_recovery_for_a_reference():
    """The rung that turned one bad reference into an empty document. Dropping
    the declarations leaves every remaining `{{secret.x}}` undeclared, which is
    a worse document than the one that failed."""
    from agent.distil import _build

    calls, marks = _typed_but_unmarked()
    draft = _draft_of(calls, marks)
    fields = draft.use_case.model_dump(mode="json", by_alias=True)
    fields["secrets"] = []

    rebuilt = _build(fields, [])

    # Either the steps survive with their slots, or the steps that needed a
    # slot are the only thing dropped. What must not happen is everything
    # going.
    assert rebuilt.setup_steps or rebuilt.row_steps
