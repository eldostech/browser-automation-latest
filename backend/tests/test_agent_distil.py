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


async def test_an_ambiguous_click_becomes_a_step_with_a_loud_warning():
    """Found for real: an agent clicked the right "Chat" button among eleven
    identical ones by ref -- that click always runs, since a ref is
    position-specific, not name-based -- and the recording carried only
    `role=button name="Chat"`, which a replay days later correctly refused
    rather than guessing among them. The warning exists so a person sees this
    on the review screen instead of discovering it from a failed batch."""
    from agent import run_agent_session

    provider = FakeMCP({"browser_snapshot": CHAT_LIST})
    llm = LangChainLLM(
        ScriptedChatModel(responses=[
            turn_calling("browser_snapshot"),
            turn_calling("mark_setup_complete"),
            turn_calling("begin_row", key="project-b"),
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
    click_step = next(s for s in result.use_case["row_steps"] if s["action"] == "click")
    assert click_step["locators"][0]["name"] == "Chat"
    assert any(
        "matched 2 elements" in w and click_step["id"] in w for w in result.draft_warnings
    ), result.draft_warnings


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
