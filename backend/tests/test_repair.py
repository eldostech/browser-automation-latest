"""Repairing a failed use case instead of re-recording it.

Two properties carry the weight here, and they mirror distillation:

* the model cannot invent a locator -- it picks from what was on the page;
* nothing it proposes is applied without a person publishing it afterwards.
"""

from __future__ import annotations

import pytest

from events import ErrorEvent, ToolResult
from llm import LLMTurn, ToolCallRequest
from repair import (
    PROPOSE_TOOL,
    RepairError,
    RepairProposal,
    UseCaseDoctor,
    apply_fixes,
    candidates,
    gather_context,
    validate_patched,
)
from usecase import Assertion, FormField, InputSpec, Locator, Step, UseCase

PAGE = """### Page
- Page URL: https://example.com/contact
### Snapshot
```yaml
- generic "wrapper" [ref=e1]:
  - textbox "Your name" [ref=e2]
  - textbox "Work email" [ref=e3]
  - button "Request a demo" [ref=e4]
```"""


def use_case(**overrides) -> UseCase:
    base = dict(
        id="uc-1",
        name="Book a demo",
        allowed_domains=["example.com"],
        row_steps=[
            Step(
                id="s10",
                action="fill",
                description="Full name field",
                locators=[Locator(strategy="role", role="textbox", name="Full name")],
                value="Ada",
            ),
            Step(
                id="s11",
                action="click",
                locators=[Locator(strategy="role", role="button", name="Request a demo")],
            ),
        ],
    )
    base.update(overrides)
    return UseCase(**base)


def execution(**overrides) -> dict:
    base = {
        "id": "ex-1",
        "run_id": "run-1",
        "status": "failed",
        "failed_step_id": "s10",
        "error": "\"getByRole('textbox', { name: 'Full name' })\" does not match any elements.",
        "inputs": {},
        "created_at": "2026-08-23T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def failure_events(snapshot: str = PAGE) -> list:
    return [
        ErrorEvent(
            run_id="run-1",
            seq=1,
            step=1,
            kind="step_failed",
            message="no element matched",
            recoverable=True,
            detail={"step_id": "s10", "page_url": "https://example.com/contact", "snapshot": snapshot},
        )
    ]


class ProposingLLM:
    def __init__(self, payload: dict | None, tokens: int = 1200) -> None:
        self.payload = payload
        self.tokens = tokens
        self.calls = 0
        self.last_message = ""

    async def run_turn(self, *, system, messages, tools, on_text_delta=None, timeout=None):
        self.calls += 1
        self.last_message = messages[-1]["content"]
        usage = {"input_tokens": self.tokens, "output_tokens": 0}
        if self.payload is None:
            return LLMTurn(text="I have no idea", usage=usage)
        return LLMTurn(
            tool_calls=[ToolCallRequest(id="t1", name="propose_repair", input=self.payload)],
            stop_reason="tool_use",
            usage=usage,
        )


# --- reconstructing the failure from history -------------------------------


def test_the_page_at_failure_is_recovered_from_the_event_log():
    """No second browser session is needed to propose a repair."""
    context = gather_context(use_case(), execution(), failure_events())

    assert context.failed_step_id == "s10"
    assert context.page_url == "https://example.com/contact"
    assert len(context.snapshot) == 4
    assert context.failed_step.description == "Full name field"


def test_a_run_without_a_recorded_snapshot_falls_back_to_tool_results():
    """Runs from before the failure snapshot was recorded stay repairable."""
    events = [
        ToolResult(
            run_id="run-1", seq=1, step=1, call_id="c1", name="browser_navigate",
            ok=True, duration_ms=1, text=PAGE,
        )
    ]
    context = gather_context(use_case(), execution(), events)
    assert len(context.snapshot) == 4


def test_no_page_at_all_still_produces_a_context():
    context = gather_context(use_case(), execution(), [])
    assert len(context.snapshot) == 0
    assert context.error


def test_only_interactive_named_controls_are_offered():
    context = gather_context(use_case(), execution(), failure_events())
    offered = candidates(context.snapshot)
    assert [n.name for n in offered] == ["Your name", "Work email", "Request a demo"]
    assert all(n.role != "generic" for n in offered)


# --- the model cannot invent a locator -------------------------------------


def test_the_tool_schema_admits_no_selector():
    properties = PROPOSE_TOOL["input_schema"]["properties"]["fixes"]["items"]["properties"]
    assert "element_index" in properties
    assert "selector" not in properties and "locator" not in properties
    assert "css" not in str(properties).lower()


# --- proposing --------------------------------------------------------------


async def test_a_renamed_field_is_diagnosed_and_repaired():
    llm = ProposingLLM(
        {
            "diagnosis": "The field was renamed from 'Full name' to 'Your name'.",
            "confidence": "high",
            "fixes": [
                {"kind": "replace_locator", "step_id": "s10", "element_index": 0,
                 "reason": "same field, new label"}
            ],
        }
    )
    context = gather_context(use_case(), execution(), failure_events())
    proposal = await UseCaseDoctor(llm).diagnose(context)

    assert llm.calls == 1
    assert proposal.actionable
    assert proposal.tokens == 1200
    assert "renamed" in proposal.diagnosis


async def test_the_prompt_carries_the_error_the_steps_and_the_page():
    llm = ProposingLLM({"diagnosis": "d", "fixes": []})
    await UseCaseDoctor(llm).diagnose(gather_context(use_case(), execution(), failure_events()))

    assert "does not match any elements" in llm.last_message
    assert "s10" in llm.last_message
    assert 'textbox "Your name"' in llm.last_message
    assert "example.com" in llm.last_message


async def test_a_model_that_declines_is_not_an_error():
    llm = ProposingLLM(
        {"diagnosis": "The page wants a login first.", "fixes": [],
         "unfixable_reason": "the site now requires sign-in the recording never did"}
    )
    proposal = await UseCaseDoctor(llm).diagnose(
        gather_context(use_case(), execution(), failure_events())
    )
    assert proposal.actionable is False
    assert "sign-in" in proposal.unfixable_reason


# (a model that answers in prose is covered below, under "when there is no
#  page to look at" -- it is reported rather than raised.)


# --- the page as it was, beside the page as it is --------------------------
#
# `Step.recorded_page` reached healing first and repair second, and repair is
# the path that needs it more: healing runs seconds after the failure, a repair
# can happen weeks later and be driven by somebody who was never there when it
# was recorded.


AS_RECORDED = """### Page
- Page URL: https://example.com/contact
### Snapshot
```yaml
- textbox "Full name" [ref=e1]
- textbox "Work email" [ref=e2]
- button "Request a demo" [ref=e3]
```"""


def recorded(page: str) -> UseCase:
    """The same use case, with the failing step remembering its own page."""
    case = use_case()
    case.row_steps[0].recorded_page = page
    return case


async def test_the_doctor_is_shown_the_page_as_it_was_when_the_step_worked():
    llm = ProposingLLM({"diagnosis": "d", "fixes": []})

    await UseCaseDoctor(llm).diagnose(
        gather_context(recorded(AS_RECORDED), execution(), failure_events())
    )

    assert "when the step was recorded and working" in llm.last_message
    assert 'textbox "Full name"' in llm.last_message, "the label that is gone"
    assert 'textbox "Your name"' in llm.last_message, "and the one that replaced it"


async def test_the_recorded_controls_are_offered_without_indices():
    """The numbered list is the one a fix picks from. Numbering a control that
    is no longer on the page would invite `element_index` pointing at it."""
    llm = ProposingLLM({"diagnosis": "d", "fixes": []})

    await UseCaseDoctor(llm).diagnose(
        gather_context(recorded(AS_RECORDED), execution(), failure_events())
    )

    was = llm.last_message.split("when the step was recorded and working", 1)[1]
    assert '- textbox "Full name"' in was
    assert '0. textbox "Full name"' not in was


async def test_a_step_with_no_recorded_page_asks_as_it_did_before():
    """Every use case recorded before this existed takes this path, and the
    prompt must not grow an empty section for them."""
    llm = ProposingLLM({"diagnosis": "d", "fixes": []})

    await UseCaseDoctor(llm).diagnose(gather_context(use_case(), execution(), failure_events()))

    assert "when the step was recorded" not in llm.last_message


async def test_an_unparseable_recorded_page_is_ignored_rather_than_fatal():
    """Context is a bonus. Losing the repair over it would be the wrong trade."""
    llm = ProposingLLM({"diagnosis": "d", "fixes": []})

    proposal = await UseCaseDoctor(llm).diagnose(
        gather_context(recorded("this is not a snapshot at all"), execution(), failure_events())
    )

    assert proposal.diagnosis == "d"


# --- applying ---------------------------------------------------------------


def definition() -> dict:
    return use_case().model_dump(mode="json", by_alias=True)


def offered() -> list:
    return candidates(gather_context(use_case(), execution(), failure_events()).snapshot)


def test_replacing_a_locator_prepends_and_keeps_the_original():
    proposal = RepairProposal(
        diagnosis="renamed",
        fixes=[{"kind": "replace_locator", "step_id": "s10", "element_index": 0}],
    )
    patched, applied = apply_fixes(definition(), proposal, offered())

    locators = patched["row_steps"][0]["locators"]
    assert locators[0]["name"] == "Your name", "the repair leads"
    assert locators[1]["name"] == "Full name", "the recording survives as a fallback"
    assert "Your name" in applied[0]
    validate_patched(patched)


def test_an_element_index_off_the_page_is_skipped_not_guessed():
    proposal = RepairProposal(
        diagnosis="x", fixes=[{"kind": "replace_locator", "step_id": "s10", "element_index": 99}]
    )
    patched, applied = apply_fixes(definition(), proposal, offered())

    assert patched["row_steps"][0]["locators"][0]["name"] == "Full name", "unchanged"
    assert "SKIPPED" in applied[0]


def test_an_unknown_step_is_skipped():
    proposal = RepairProposal(
        diagnosis="x", fixes=[{"kind": "replace_locator", "step_id": "nope", "element_index": 0}]
    )
    _, applied = apply_fixes(definition(), proposal, offered())
    assert "SKIPPED" in applied[0] and "nope" in applied[0]


def test_fixing_an_assertion():
    base = use_case(
        row_steps=[
            Step(id="s1", action="assert",
                 assertion=Assertion(kind="url_contains", value="example.com", negate=True))
        ]
    ).model_dump(mode="json", by_alias=True)
    proposal = RepairProposal(
        diagnosis="impossible check",
        fixes=[
            {
                "kind": "fix_assertion",
                "step_id": "s1",
                "assertion_kind": "text_present",
                "value": "Thanks",
                "reason": "the page shows this once the form is sent",
            }
        ],
    )
    patched, applied = apply_fixes(base, proposal, offered())

    assert patched["row_steps"][0]["assert"]["kind"] == "text_present"
    assert patched["row_steps"][0]["assert"]["value"] == "Thanks"
    assert "Thanks" in applied[0]
    validate_patched(patched)


def test_changing_a_value():
    proposal = RepairProposal(
        diagnosis="x", fixes=[{"kind": "change_value", "step_id": "s10", "value": "Grace"}]
    )
    patched, _ = apply_fixes(definition(), proposal, offered())
    assert patched["row_steps"][0]["value"] == "Grace"


def test_making_a_step_optional_also_stops_it_aborting_the_row():
    proposal = RepairProposal(diagnosis="x", fixes=[{"kind": "make_optional", "step_id": "s10"}])
    patched, _ = apply_fixes(definition(), proposal, offered())

    assert patched["row_steps"][0]["optional"] is True
    assert patched["row_steps"][0]["on_failure"] == "continue"


def test_dropping_a_step():
    proposal = RepairProposal(diagnosis="x", fixes=[{"kind": "drop_step", "step_id": "s10"}])
    patched, applied = apply_fixes(definition(), proposal, offered())

    assert [s["id"] for s in patched["row_steps"]] == ["s11"]
    assert "removed" in applied[0]
    validate_patched(patched)


def test_fixing_the_session_check_needs_no_step_id():
    proposal = RepairProposal(
        diagnosis="x",
        fixes=[
            {"kind": "fix_session_check", "assertion_kind": "url_contains",
             "value": "/signin", "negate": True}
        ],
    )
    patched, applied = apply_fixes(definition(), proposal, offered())

    assert patched["session_check"]["value"] == "/signin"
    assert patched["session_check"]["negate"] is True
    assert "session check" in applied[0]


def test_an_unknown_fix_kind_is_skipped():
    proposal = RepairProposal(diagnosis="x", fixes=[{"kind": "reformat_everything", "step_id": "s10"}])
    _, applied = apply_fixes(definition(), proposal, offered())
    assert "SKIPPED" in applied[0]


def test_several_fixes_apply_together():
    proposal = RepairProposal(
        diagnosis="two things",
        fixes=[
            {"kind": "replace_locator", "step_id": "s10", "element_index": 0},
            {"kind": "make_optional", "step_id": "s11"},
        ],
    )
    patched, applied = apply_fixes(definition(), proposal, offered())

    assert patched["row_steps"][0]["locators"][0]["name"] == "Your name"
    assert patched["row_steps"][1]["optional"] is True
    assert len(applied) == 2


def test_the_original_definition_is_never_mutated():
    original = definition()
    proposal = RepairProposal(diagnosis="x", fixes=[{"kind": "drop_step", "step_id": "s10"}])
    apply_fixes(original, proposal, offered())
    assert [s["id"] for s in original["row_steps"]] == ["s10", "s11"]


def test_a_repair_that_breaks_the_schema_is_rejected():
    """A repair producing something invalid is a failed repair, not a new use case."""
    base = use_case(
        inputs=[InputSpec(name="who")],
        row_steps=[
            Step(id="s1", action="fill",
                 locators=[Locator(strategy="css", selector="#a")], value="{{input.who}}")
        ],
    ).model_dump(mode="json", by_alias=True)
    # Dropping the only step that reads `who` leaves an input nothing uses,
    # which is fine in a draft -- but a locator-less fill would not be.
    proposal = RepairProposal(
        diagnosis="x",
        fixes=[{"kind": "fix_assertion", "step_id": "s1", "assertion_kind": "text_present",
                "value": "ok"}],
    )
    patched, _ = apply_fixes(base, proposal, offered())
    # The step became an assert, so it no longer reads the input; still valid
    # as a draft.
    validate_patched(patched)


# --- a repeated card, the ordinary shape of a list page ---------------------

#: 11 identical "Chat" buttons, one per project card -- the page that broke a
#: real run: a step recorded against one specific card's button could not be
#: told apart from the other 10 by role and name alone.
CARDS = """### Page
- Page URL: https://example.com/dashboard
### Snapshot
```yaml
- generic [ref=e1]:
  - generic "Alpha Project" [ref=e2]:
    - heading "Alpha Project" [ref=e3]
    - button "Chat" [ref=e4]
  - generic "Beta Project" [ref=e5]:
    - heading "Beta Project" [ref=e6]
    - button "Chat" [ref=e7]
  - generic "Gamma Project" [ref=e8]:
    - heading "Gamma Project" [ref=e9]
    - button "Chat" [ref=e10]
```"""


def test_a_duplicate_group_offers_every_instance_not_just_the_first():
    """This used to collapse to one candidate no matter which card failed."""
    options = candidates(gather_context(use_case(), execution(), failure_events(CARDS)).snapshot)
    chat_buttons = [n for n in options if n.role == "button" and n.name == "Chat"]
    assert len(chat_buttons) == 3
    assert [n.ref for n in chat_buttons] == ["e4", "e7", "e10"]


def test_the_listing_says_which_card_each_duplicate_belongs_to():
    from repair import _listing

    snapshot = gather_context(use_case(), execution(), failure_events(CARDS)).snapshot
    listing = _listing(candidates(snapshot), snapshot)

    assert 'inside "Beta Project"' in listing
    assert 'inside "Gamma Project"' in listing


def test_picking_one_of_a_duplicate_group_names_the_card_it_sits_in():
    """The fix that would have unblocked the real incident: a repair can point
    at *the second* identical button.

    It says so by naming the card rather than by counting, which is the
    stronger of the two answers -- "the Chat button on Beta Project" survives a
    fourth project being added above it, and "the second Chat button" does not.
    """
    snapshot = gather_context(use_case(), execution(), failure_events(CARDS)).snapshot
    options = candidates(snapshot)
    beta_index = next(i for i, n in enumerate(options) if n.ref == "e7")

    proposal = RepairProposal(
        diagnosis="x",
        fixes=[{"kind": "replace_locator", "step_id": "s10", "element_index": beta_index}],
    )
    patched, applied = apply_fixes(definition(), proposal, options, snapshot)

    locator = patched["row_steps"][0]["locators"][0]
    assert locator["name"] == "Chat"
    assert locator["nth"] == 0, "scoped, so no position is needed"
    assert locator["within"]["name"] == "Beta Project"
    assert "Beta Project" in applied[0]
    validate_patched(patched)


def test_the_first_of_a_duplicate_group_can_be_repaired_once_its_card_is_named():
    """`nth=0` means "no position given", so the first of several identical
    controls has no positional spelling at all and used to be refused outright.
    Naming where it sits gives it one -- and it is the answer a person would
    have given anyway."""
    snapshot = gather_context(use_case(), execution(), failure_events(CARDS)).snapshot
    options = candidates(snapshot)
    alpha_index = next(i for i, n in enumerate(options) if n.ref == "e4")

    proposal = RepairProposal(
        diagnosis="x",
        fixes=[{"kind": "replace_locator", "step_id": "s10", "element_index": alpha_index}],
    )
    patched, applied = apply_fixes(definition(), proposal, options, snapshot)

    locator = patched["row_steps"][0]["locators"][0]
    assert locator["name"] == "Chat"
    assert locator["within"]["name"] == "Alpha Project"
    assert "SKIPPED" not in applied[0]
    validate_patched(patched)


def test_without_the_full_snapshot_ambiguity_still_degrades_safely():
    """A caller that does not thread the snapshot through (every existing one
    before this change) still gets a correct answer for anything the trimmed
    candidate list itself already contains -- it just cannot see duplicates
    MAX_PER_GROUP trimmed away."""
    snapshot = gather_context(use_case(), execution(), failure_events(CARDS)).snapshot
    options = candidates(snapshot)
    beta_index = next(i for i, n in enumerate(options) if n.ref == "e7")

    proposal = RepairProposal(
        diagnosis="x",
        fixes=[{"kind": "replace_locator", "step_id": "s10", "element_index": beta_index}],
    )
    patched, _ = apply_fixes(definition(), proposal, options)  # no snapshot passed

    assert patched["row_steps"][0]["locators"][0]["nth"] == 1


#: The same three cards, with nothing naming any of them. There is genuinely
#: no way to single out the first "Chat" button here, and the point of the
#: fixture is that this case still exists after scoping was added.
BARE_CARDS = """### Page
- Page URL: https://example.com/dashboard
### Snapshot
```yaml
- generic [ref=e1]:
  - generic [ref=e2]:
    - button "Chat" [ref=e4]
  - generic [ref=e5]:
    - button "Chat" [ref=e7]
  - generic [ref=e8]:
    - button "Chat" [ref=e10]
```"""


def test_the_first_of_a_duplicate_group_is_still_refused_when_nothing_names_it():
    """`nth=0` means "no position given" as far as the executor's resolver is
    concerned -- so a "fix" that pins the first of several identical elements
    to position 0 would look applied in the diff and still be refused, for
    the identical reason, the next time it runs.

    Scoping answers this whenever something around the element has a name. When
    nothing does, skipping and saying so is still more honest than a repair
    that appears to work and does not."""
    snapshot = gather_context(use_case(), execution(), failure_events(BARE_CARDS)).snapshot
    options = candidates(snapshot)
    alpha_index = next(i for i, n in enumerate(options) if n.ref == "e4")

    proposal = RepairProposal(
        diagnosis="x",
        fixes=[{"kind": "replace_locator", "step_id": "s10", "element_index": alpha_index}],
    )
    patched, applied = apply_fixes(definition(), proposal, options, snapshot)

    assert patched["row_steps"][0]["locators"][0]["name"] == "Full name", "unchanged"
    assert "SKIPPED" in applied[0]
    assert "nothing around it is named" in applied[0]


# --- when there is no page to look at --------------------------------------


async def test_a_run_that_never_reached_a_step_says_so_rather_than_advising_a_retry():
    """Two different refusals, because they need two different things done.

    No events at all means the run died before any step was attempted -- the
    browser would not start, or setup never finished. There was no page to
    record, and telling the person to run it again just sends them round the
    same loop; the useful thing is the error that actually stopped it.
    """
    llm = ProposingLLM({"diagnosis": "d", "fixes": []})
    context = gather_context(use_case(), execution(), [])

    with pytest.raises(RepairError) as caught:
        await UseCaseDoctor(llm).diagnose(context)

    message = str(caught.value)
    assert "before any step" in message
    assert "run it once more" not in message
    assert llm.calls == 0, "every fix would be a guess, so do not pay for a refusal"


async def test_a_snapshot_that_was_only_a_link_counts_as_no_page():
    """The spill bug produced exactly this: a result with nothing parseable."""
    events = failure_events("### Snapshot\n- [Snapshot](.playwright-mcp/page-1.yml)\n")
    llm = ProposingLLM({"diagnosis": "d", "fixes": []})

    with pytest.raises(RepairError, match="not recorded"):
        await UseCaseDoctor(llm).diagnose(gather_context(use_case(), execution(), events))
    assert llm.calls == 0


async def test_prose_instead_of_a_tool_call_is_reported_not_raised():
    """A model explaining itself is information, not an API failure."""
    llm = ProposingLLM(None)
    proposal = await UseCaseDoctor(llm).diagnose(
        gather_context(use_case(), execution(), failure_events())
    )

    assert proposal.actionable is False
    assert "no idea" in proposal.diagnosis
    assert proposal.confidence == "low"
    assert proposal.tokens == 1200


# --- a repair that changes nothing is not a repair -------------------------
#
# Regression: pressing "Fix it with AI" twice produced two new versions, the
# second byte-identical to the first, and reported success both times. From
# the outside that is indistinguishable from "the change did not persist".


def test_is_unchanged_ignores_metadata_that_moves_on_every_save():
    from repair import is_unchanged

    before = definition()
    after = {**before, "version": 9, "updated_at": "later", "warnings": ["new"]}
    assert is_unchanged(before, after) is True


def test_is_unchanged_sees_a_real_edit():
    from repair import is_unchanged

    before = definition()
    after, _ = apply_fixes(
        before,
        RepairProposal(diagnosis="x", fixes=[{"kind": "drop_step", "step_id": "s10"}]),
        offered(),
    )
    assert is_unchanged(before, after) is False


def test_a_proposal_whose_fixes_all_skip_leaves_the_definition_alone():
    from repair import is_unchanged

    before = definition()
    after, applied = apply_fixes(
        before,
        RepairProposal(
            diagnosis="x",
            fixes=[{"kind": "replace_locator", "step_id": "does-not-exist", "element_index": 0}],
        ),
        offered(),
    )
    assert all("SKIPPED" in line for line in applied)
    assert is_unchanged(before, after) is True


# --- a form's fields each carry their own locator --------------------------


def form_definition() -> dict:
    return UseCase(
        id="uc-form",
        name="Contact form",
        allowed_domains=["example.com"],
        row_steps=[
            Step(
                id="s4",
                action="fill_form",
                fields=[
                    FormField(name="Full Name", value="Ada",
                              locators=[Locator(strategy="css", selector="input#name")]),
                    FormField(name="Work Email", value="a@b.c",
                              locators=[Locator(strategy="css", selector="input#email")]),
                ],
            )
        ],
    ).model_dump(mode="json", by_alias=True)


def test_replacing_a_form_field_locator_edits_the_field_not_the_step():
    """The executor reads per-field locators; a step-level one is ignored."""
    patched, applied = apply_fixes(
        form_definition(),
        RepairProposal(
            diagnosis="renamed",
            fixes=[
                {"kind": "replace_locator", "step_id": "s4", "field_name": "Full Name",
                 "element_index": 0}
            ],
        ),
        offered(),
    )
    step = patched["row_steps"][0]

    assert step["fields"][0]["locators"][0]["name"] == "Your name", "the field was changed"
    assert step["fields"][0]["locators"][1]["selector"] == "input#name", "original kept"
    assert step["fields"][1]["locators"][0]["selector"] == "input#email", "other field untouched"
    assert step["locators"] == [], "the ignored step-level list is left empty"
    assert "Full Name" in applied[0]
    validate_patched(patched)


def test_a_form_fix_without_a_field_name_is_refused_with_the_field_list():
    """Silently writing a locator the executor ignores is the worst outcome."""
    before = form_definition()
    patched, applied = apply_fixes(
        before,
        RepairProposal(
            diagnosis="x",
            fixes=[{"kind": "replace_locator", "step_id": "s4", "element_index": 0}],
        ),
        offered(),
    )
    from repair import is_unchanged

    assert "SKIPPED" in applied[0]
    assert "'Full Name'" in applied[0] and "'Work Email'" in applied[0]
    assert is_unchanged(before, patched) is True


def test_an_unknown_field_name_is_refused():
    _, applied = apply_fixes(
        form_definition(),
        RepairProposal(
            diagnosis="x",
            fixes=[{"kind": "replace_locator", "step_id": "s4", "field_name": "Nope",
                    "element_index": 0}],
        ),
        offered(),
    )
    assert "SKIPPED" in applied[0] and "'Nope'" in applied[0]


# --- what each step was for -----------------------------------------------


async def test_the_doctor_is_told_what_the_failing_step_was_for():
    llm = ProposingLLM({"diagnosis": "d", "fixes": []})
    case = use_case()
    case.row_steps[0].intent = "finds the customer the row names"

    await UseCaseDoctor(llm).diagnose(gather_context(case, execution(), failure_events()))

    assert "what it is for: finds the customer the row names" in llm.last_message


async def test_the_purpose_of_the_steps_around_it_is_shown_too():
    """A repair judges one step against the flow it sits in. A list of
    mechanics with no purposes on it is what made "which of these forty
    controls" the only question available."""
    llm = ProposingLLM({"diagnosis": "d", "fixes": []})
    case = use_case()
    case.row_steps[1].intent = "saves the change"

    await UseCaseDoctor(llm).diagnose(gather_context(case, execution(), failure_events()))

    assert "      for: saves the change" in llm.last_message


async def test_a_use_case_with_no_purposes_reads_as_it_always_did():
    llm = ProposingLLM({"diagnosis": "d", "fixes": []})

    await UseCaseDoctor(llm).diagnose(gather_context(use_case(), execution(), failure_events()))

    assert "what it is for" not in llm.last_message
    assert "      for:" not in llm.last_message


# --- what the browser tried, and what stopped it --------------------------
#
# The failure that kept coming back. A recorded click on a column header
# failed with "TimeoutError: Locator.click: Timeout 30000ms exceeded" and
# nothing else, because the engine kept the first line of Playwright's error
# and dropped the call log. The element had been found; something was covering
# it. Given only a timeout and a locator, a repair proposed the same locator
# with `exact` turned off -- the only change it could express -- and the next
# run happened to work, so the diagnosis was never made.


COVERED = """Call log:
  - waiting for get_by_role("columnheader", name="Make")
  -   locator resolved to <th class="sortable">Make</th>
  - attempting click action
  -   <div id="onetrust-consent-sdk">…</div> intercepts pointer events
  - retrying click action"""


def covering_events(snapshot: str = PAGE) -> list:
    return [
        ErrorEvent(
            run_id="run-1",
            seq=1,
            step=1,
            kind="step_failed",
            message="TimeoutError: Locator.click: Timeout 30000ms exceeded.",
            recoverable=True,
            detail={
                "step_id": "s10",
                "page_url": "https://example.com/contact",
                "snapshot": snapshot,
                "call_log": COVERED,
            },
        )
    ]


def test_the_call_log_is_recovered_from_the_failure_event():
    context = gather_context(use_case(), execution(), covering_events())

    assert "intercepts pointer events" in context.call_log


async def test_the_doctor_is_shown_what_stopped_the_action(client=None):
    """So it can tell "cannot find it" from "found it and could not click
    it" -- which need different answers and got the same one."""
    llm = ProposingLLM({"diagnosis": "d", "fixes": []})

    await UseCaseDoctor(llm).diagnose(
        gather_context(use_case(), execution(), covering_events())
    )

    assert "What the browser tried" in llm.last_message
    assert "onetrust-consent-sdk" in llm.last_message


async def test_a_failure_with_no_call_log_asks_exactly_as_it_did_before():
    """Every run recorded before this existed, and every failure that was not
    an action -- an assertion, a URL off the allowlist."""
    llm = ProposingLLM({"diagnosis": "d", "fixes": []})

    await UseCaseDoctor(llm).diagnose(
        gather_context(use_case(), execution(), failure_events())
    )

    assert "What the browser tried" not in llm.last_message
