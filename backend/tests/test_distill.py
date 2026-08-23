"""Distillation: a noisy recording in, a replayable use case out.

The pre-filter tests are built on synthetic events so each rule is isolated;
the end-to-end test replays the shape of the real IXL run (34 calls, 13 of them
failures) to prove the whole pipeline holds together.

The one thing every test here guards jointly: the model may choose *which*
recorded steps survive and *what* their values mean, but it can never
contribute a locator. Every locator in a finished use case came from a call
that actually succeeded.
"""

from __future__ import annotations

import pytest

from distill import (
    BUILD_TOOL,
    DistillationError,
    build_usecase,
    distill,
    pre_filter,
    summarise,
)
from events import Screenshot, ToolCall, ToolResult
from llm import LLMTurn, ToolCallRequest

SNAPSHOT = """### Page
- Page URL: https://example.com/signin
### Snapshot
```yaml
- textbox "Username" [ref=e17]
- textbox "Password" [ref=e21]
- button "Sign in" [ref=e30]
- radio "Nayra Asati" [ref=e40]
```"""


class _Seq:
    def __init__(self) -> None:
        self.n = 0

    def __call__(self) -> int:
        self.n += 1
        return self.n


def build_events(*specs) -> list:
    """``(tool, arguments, ok, text)`` tuples -> a paired call/result stream."""
    seq = _Seq()
    events: list = []
    for index, (tool, arguments, ok, text) in enumerate(specs, start=1):
        call_id = f"c{index}"
        events.append(
            ToolCall(
                run_id="r", seq=seq(), step=index, call_id=call_id, name=tool, arguments=arguments
            )
        )
        events.append(
            ToolResult(
                run_id="r", seq=seq(), step=index, call_id=call_id, name=tool,
                ok=ok, duration_ms=1, text=text,
            )
        )
    return events


# --- pre-filter: dropping the noise ----------------------------------------


def test_observation_only_calls_are_dropped():
    events = build_events(
        ("browser_snapshot", {}, True, SNAPSHOT),
        ("browser_take_screenshot", {}, True, ""),
        ("browser_console_messages", {}, True, "[]"),
        ("browser_navigate", {"url": "https://example.com"}, True, "ok"),
    )
    result = pre_filter(events)
    assert [s.action for s in result.steps] == ["navigate"]
    assert result.stats["observation"] == 3


def test_failed_calls_are_dropped():
    events = build_events(
        ("browser_click", {"target": "#a"}, False, "could not click"),
        ("browser_click", {"target": "#b"}, True, "clicked"),
    )
    result = pre_filter(events)
    assert len(result.steps) == 1
    assert result.steps[0].locators[0].selector == "#b"
    assert result.stats["failed"] == 1


def test_a_failure_reported_as_prose_is_still_treated_as_a_failure():
    """Playwright MCP returns some timeouts as text with ok=true."""
    events = build_events(
        ("browser_click", {"target": "#a"}, True, "Error: locator.click: Timeout 5000ms exceeded"),
        ("browser_click", {"target": "#b"}, True, "clicked"),
    )
    result = pre_filter(events)
    assert [s.locators[0].selector for s in result.steps] == ["#b"]


def test_a_call_with_no_result_is_dropped():
    """The run died mid-call; nothing proves the action landed."""
    events = [ToolCall(run_id="r", seq=1, step=1, call_id="c1", name="browser_click",
                       arguments={"target": "#a"})]
    assert pre_filter(events).steps == []


def test_non_tool_events_are_ignored():
    events = build_events(("browser_navigate", {"url": "https://example.com"}, True, "ok"))
    events.append(Screenshot(run_id="r", seq=99, step=1, artifact_id="a", url="/a"))
    assert len(pre_filter(events).steps) == 1


# --- pre-filter: the ref resolution that makes replay possible -------------


def test_an_ephemeral_ref_is_resolved_to_role_and_name():
    events = build_events(
        ("browser_snapshot", {}, True, SNAPSHOT),
        ("browser_click", {"target": "ref=e30", "element": "Sign in button"}, True, "clicked"),
    )
    step = pre_filter(events).steps[0]
    locator = step.locators[0]
    assert (locator.strategy, locator.role, locator.name) == ("role", "button", "Sign in")


def test_the_bracketed_ref_spelling_resolves_too():
    events = build_events(
        ("browser_snapshot", {}, True, SNAPSHOT),
        ("browser_click", {"target": "[ref=e30]"}, True, "clicked"),
    )
    assert pre_filter(events).steps[0].locators[0].name == "Sign in"


def test_a_ref_resolves_against_the_snapshot_that_preceded_it_not_a_later_one():
    later = SNAPSHOT.replace('button "Sign in" [ref=e30]', 'button "Continue" [ref=e30]')
    events = build_events(
        ("browser_snapshot", {}, True, SNAPSHOT),
        ("browser_click", {"target": "ref=e30"}, True, "clicked"),
        ("browser_snapshot", {}, True, later),
    )
    assert pre_filter(events).steps[0].locators[0].name == "Sign in"


def test_an_unresolvable_ref_is_warned_about_not_silently_kept():
    events = build_events(
        ("browser_snapshot", {}, True, SNAPSHOT),
        ("browser_click", {"target": "ref=e999"}, True, "clicked"),
    )
    result = pre_filter(events)
    assert result.steps[0].locators == []
    assert any("e999" in w for w in result.warnings)


def test_no_ref_survives_into_a_locator():
    """The whole point: a distilled use case must contain no ephemeral refs."""
    events = build_events(
        ("browser_snapshot", {}, True, SNAPSHOT),
        ("browser_click", {"target": "ref=e30"}, True, "clicked"),
        ("browser_type", {"target": "ref=e17", "text": "nitin"}, True, "typed"),
    )
    for step in pre_filter(events).steps:
        for locator in step.locators:
            assert "ref=" not in (locator.selector or "")
            assert locator.strategy == "role"


def test_a_css_selector_is_kept_as_a_css_rung():
    events = build_events(("browser_click", {"target": "button[type='submit']"}, True, "ok"))
    locator = pre_filter(events).steps[0].locators[0]
    assert (locator.strategy, locator.selector) == ("css", "button[type='submit']")


# --- pre-filter: form fields ----------------------------------------------


def test_fill_form_fields_each_get_their_own_locator():
    events = build_events(
        ("browser_snapshot", {}, True, SNAPSHOT),
        (
            "browser_fill_form",
            {
                "fields": [
                    {"target": "ref=e17", "name": "Username", "value": "nitin", "type": "textbox"},
                    {"target": "ref=e21", "name": "Password", "value": "pw", "type": "textbox"},
                ]
            },
            True,
            "filled",
        ),
    )
    step = pre_filter(events).steps[0]
    assert [f["name"] for f in step.fields] == ["Username", "Password"]
    assert step.fields[0]["locators"][0].name == "Username"
    assert step.fields[1]["locators"][0].name == "Password"


def test_form_field_values_are_offered_as_parameter_candidates():
    events = build_events(
        ("browser_snapshot", {}, True, SNAPSHOT),
        (
            "browser_fill_form",
            {"fields": [{"target": "ref=e17", "name": "Username", "value": "nitin"}]},
            True,
            "filled",
        ),
    )
    assert "nitin" in pre_filter(events).literals


# --- pre-filter: retries and metadata --------------------------------------


def test_consecutive_attempts_at_the_same_element_collapse_to_one():
    events = build_events(
        ("browser_snapshot", {}, True, SNAPSHOT),
        ("browser_click", {"target": "ref=e30", "element": "Sign in button"}, True, "ok"),
        ("browser_click", {"target": "#submit", "element": "Sign in button"}, True, "ok"),
    )
    result = pre_filter(events)
    assert len(result.steps) == 1, "one goal, one step"
    assert result.stats["collapsed"] == 1
    # The earlier successful attempt survives as a fallback rung.
    assert any(loc.strategy == "role" for loc in result.steps[0].locators)
    assert any(loc.selector == "#submit" for loc in result.steps[0].locators)


def test_the_start_url_and_domains_are_captured():
    events = build_events(
        ("browser_navigate", {"url": "https://www.ixl.com/signin"}, True, "ok"),
        ("browser_navigate", {"url": "https://www.ixl.com/math"}, True, "ok"),
    )
    result = pre_filter(events)
    assert result.start_url == "https://www.ixl.com/signin"
    assert result.domains == ["www.ixl.com"]


def test_summarise_reports_the_reduction():
    events = build_events(
        ("browser_snapshot", {}, True, SNAPSHOT),
        ("browser_click", {"target": "#a"}, False, "nope"),
        ("browser_click", {"target": "#b"}, True, "ok"),
    )
    assert summarise(pre_filter(events)) == (
        "3 tool calls -> 1 steps (1 failed, 1 observation-only, 0 retries collapsed)"
    )


# --- assembling the use case ----------------------------------------------


@pytest.fixture
def recording():
    """The shape of the real IXL run, minus the noise it also contained."""
    return pre_filter(
        build_events(
            ("browser_navigate", {"url": "https://www.ixl.com/signin"}, True, "ok"),
            ("browser_snapshot", {}, True, SNAPSHOT),
            (
                "browser_fill_form",
                {
                    "fields": [
                        {"target": "ref=e17", "name": "Username", "value": "nitin"},
                        {"target": "ref=e21", "name": "Password", "value": "«redacted»"},
                    ]
                },
                True,
                "filled",
            ),
            ("browser_click", {"target": "ref=e30", "element": "Sign in button"}, True, "ok"),
            ("browser_navigate", {"url": "https://www.ixl.com/math/g5"}, True, "ok"),
        )
    )


PLAN = {
    "name": "IXL practice",
    "description": "Sign in and open a skill.",
    "inputs": [{"name": "practice_url", "type": "url"}],
    "secrets": [{"name": "username"}, {"name": "password"}],
    "setup_step_ids": ["s1", "s3", "s4"],
    "row_step_ids": ["s5"],
    "values": {"s3.Username": "{{secret.username}}", "s3.Password": "{{secret.password}}"},
    "urls": {"s5": "{{input.practice_url}}"},
    "assertions": [
        {"after_step_id": "s4", "kind": "url_contains", "value": "/signin", "negate": True}
    ],
    "session_check": {"kind": "url_contains", "value": "/signin", "negate": True},
    "row_reset_url": "{{input.practice_url}}",
}


def test_the_plan_splits_setup_from_per_row_work(recording):
    use_case = build_usecase(PLAN, recording, source_run_id="run-1")
    assert [s.id for s in use_case.setup_steps] == ["s1", "s3", "s4", "s4_check1"]
    assert [s.id for s in use_case.row_steps] == ["s5"]
    assert use_case.row_reset.url == "{{input.practice_url}}"
    assert use_case.session_check.negate is True


def test_credentials_become_secret_templates(recording):
    use_case = build_usecase(PLAN, recording)
    form = next(s for s in use_case.setup_steps if s.action == "fill_form")
    assert [f.value for f in form.fields] == ["{{secret.username}}", "{{secret.password}}"]
    assert "nitin" not in use_case.model_dump_json()


def test_per_row_values_become_input_templates(recording):
    use_case = build_usecase(PLAN, recording)
    assert use_case.row_steps[0].url == "{{input.practice_url}}"


def test_assertions_are_woven_in_after_the_step_they_verify(recording):
    use_case = build_usecase(PLAN, recording)
    ids = [s.id for s in use_case.setup_steps]
    assert ids.index("s4_check1") == ids.index("s4") + 1


def test_locators_survive_into_the_use_case_unchanged(recording):
    use_case = build_usecase(PLAN, recording)
    click = next(s for s in use_case.setup_steps if s.action == "click")
    assert (click.locators[0].role, click.locators[0].name) == ("button", "Sign in")


def test_a_new_use_case_is_always_a_draft(recording):
    use_case = build_usecase(PLAN, recording)
    assert use_case.status == "draft"
    assert use_case.runnable is False, "review is mandatory before a batch can run it"


def test_allowed_domains_default_to_what_the_recording_visited(recording):
    assert build_usecase(PLAN, recording).allowed_domains == ["www.ixl.com"]


def test_steps_the_plan_ignored_are_reported(recording):
    plan = {**PLAN, "row_step_ids": []}
    use_case = build_usecase(plan, recording)
    assert any("dropped by the plan" in w and "s5" in w for w in use_case.warnings)


def test_an_unknown_step_id_is_warned_about_rather_than_crashing(recording):
    plan = {**PLAN, "row_step_ids": ["s5", "s999"]}
    use_case = build_usecase(plan, recording)
    assert any("s999" in w for w in use_case.warnings)
    assert [s.id for s in use_case.row_steps] == ["s5"]


def test_a_plan_with_no_assertions_is_warned_about(recording):
    plan = {k: v for k, v in PLAN.items() if k != "assertions"}
    use_case = build_usecase(plan, recording)
    assert any("no assertions" in w for w in use_case.warnings)


def test_script_steps_are_flagged_and_left_disabled():
    recording = pre_filter(
        build_events(("browser_run_code_unsafe", {"code": "await page.click('x')"}, True, "ok"))
    )
    use_case = build_usecase({"name": "x", "row_step_ids": ["s1"]}, recording)
    assert use_case.allow_scripts is False
    assert any("raw-JavaScript" in w for w in use_case.warnings)
    assert use_case.status == "draft"


def test_the_model_cannot_contribute_a_locator(recording):
    """Structural guarantee: the plan schema has no way to express one."""
    schema = BUILD_TOOL["input_schema"]["properties"]
    assert "locators" not in schema and "selector" not in schema
    text = str(schema)
    assert "css" not in text and "xpath" not in text


# --- the single LLM call ---------------------------------------------------


class OneCallLLM:
    """Returns a fixed plan and counts how many times it was asked."""

    model = "fake"

    def __init__(self, plan: dict | None = None, text: str = "") -> None:
        self.plan = plan
        self.text = text
        self.calls = 0
        self.tools_offered: list = []

    async def run_turn(self, *, system, messages, tools, on_text_delta=None, timeout=None):
        self.calls += 1
        self.tools_offered = tools
        if self.plan is None:
            return LLMTurn(text=self.text or "I could not do that")
        return LLMTurn(
            text="",
            tool_calls=[ToolCallRequest(id="t1", name="build_usecase", input=self.plan)],
            stop_reason="tool_use",
        )


async def test_distillation_spends_exactly_one_llm_call():
    llm = OneCallLLM(PLAN)
    events = build_events(
        ("browser_navigate", {"url": "https://www.ixl.com/signin"}, True, "ok"),
        ("browser_snapshot", {}, True, SNAPSHOT),
        (
            "browser_fill_form",
            {"fields": [{"target": "ref=e17", "name": "Username", "value": "nitin"},
                        {"target": "ref=e21", "name": "Password", "value": "x"}]},
            True, "filled",
        ),
        ("browser_click", {"target": "ref=e30", "element": "Sign in button"}, True, "ok"),
        ("browser_navigate", {"url": "https://www.ixl.com/math/g5"}, True, "ok"),
    )
    use_case = await distill(events, task="sign in", llm=llm, source_run_id="run-1")

    assert llm.calls == 1, "the entire cost of the feature is this one call"
    assert use_case.name == "IXL practice"
    assert use_case.source_run_id == "run-1"
    assert [t["name"] for t in llm.tools_offered] == ["build_usecase"]


async def test_the_reduction_is_recorded_on_the_use_case():
    llm = OneCallLLM(PLAN)
    events = build_events(
        ("browser_navigate", {"url": "https://www.ixl.com/signin"}, True, "ok"),
        ("browser_snapshot", {}, True, SNAPSHOT),
        ("browser_click", {"target": "#a"}, False, "failed"),
        (
            "browser_fill_form",
            {"fields": [{"target": "ref=e17", "name": "Username", "value": "n"},
                        {"target": "ref=e21", "name": "Password", "value": "x"}]},
            True, "filled",
        ),
        ("browser_click", {"target": "ref=e30", "element": "Sign in button"}, True, "ok"),
        ("browser_navigate", {"url": "https://www.ixl.com/math/g5"}, True, "ok"),
    )
    use_case = await distill(events, task="sign in", llm=llm)
    assert "tool calls ->" in use_case.warnings[0]


async def test_a_run_with_nothing_successful_is_refused_without_calling_the_model():
    llm = OneCallLLM(PLAN)
    events = build_events(("browser_click", {"target": "#a"}, False, "failed"))
    with pytest.raises(DistillationError, match="no successful actions"):
        await distill(events, task="t", llm=llm)
    assert llm.calls == 0, "a hopeless run must not cost a token"


async def test_a_model_that_answers_in_prose_is_an_error_not_a_broken_use_case():
    llm = OneCallLLM(None, text="I am not sure what you want.")
    events = build_events(("browser_navigate", {"url": "https://example.com"}, True, "ok"))
    with pytest.raises(DistillationError, match="not sure what you want"):
        await distill(events, task="t", llm=llm)


# --- a recorded keypress, which used to crash distillation -----------------


def test_a_target_less_keypress_becomes_a_step():
    """Regression: `browser_press_key` carries only a key, never a target.

    Requiring a locator for it made run 0fdb18ca -- which pressed Enter to
    submit a dialog -- impossible to distil at all, returning a 500.
    """
    recording = pre_filter(build_events(("browser_press_key", {"key": "Enter"}, True, "pressed")))
    use_case = build_usecase({"name": "x", "row_step_ids": ["s1"]}, recording)

    assert [s.action for s in use_case.row_steps] == ["press"]
    assert use_case.row_steps[0].value == "Enter"
    assert use_case.row_steps[0].locators == []


def test_a_step_the_schema_rejects_is_dropped_with_a_warning_not_a_crash():
    """One unbuildable step must not cost the whole recording."""
    recording = pre_filter(
        build_events(
            ("browser_navigate", {"url": "https://example.com"}, True, "ok"),
            # A click with no target at all cannot be replayed.
            ("browser_click", {}, True, "clicked"),
        )
    )
    use_case = build_usecase({"name": "x", "row_step_ids": ["s1", "s2"]}, recording)

    assert [s.action for s in use_case.row_steps] == ["navigate"], "the good step survives"
    assert any("could not be turned into a replayable step" in w for w in use_case.warnings)
    assert any("browser_click" in w for w in use_case.warnings)
