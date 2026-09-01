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
