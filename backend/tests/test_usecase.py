"""Schema and validation for a recorded use case.

These validators are the last gate before something runs a thousand times
unattended, so each one here corresponds to a specific way a batch can go
silently wrong.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from usecase import (
    Assertion,
    FormField,
    InputSpec,
    Locator,
    MissingValue,
    SecretSpec,
    Step,
    UseCase,
    WaitFor,
    has_template,
    render_template,
    template_refs,
)


def role(name: str = "Username", kind: str = "textbox") -> Locator:
    return Locator(strategy="role", role=kind, name=name)


def simple(**overrides) -> UseCase:
    base = dict(
        name="Test use case",
        row_steps=[Step(id="s1", action="navigate", url="https://example.com")],
    )
    base.update(overrides)
    return UseCase(**base)


# --- locators --------------------------------------------------------------


def test_a_locator_requires_the_field_its_strategy_uses():
    with pytest.raises(ValidationError, match="requires 'selector'"):
        Locator(strategy="css")
    with pytest.raises(ValidationError, match="requires 'role'"):
        Locator(strategy="role")
    with pytest.raises(ValidationError, match="requires 'text'"):
        Locator(strategy="text")


def test_brittle_rungs_are_flagged():
    assert role().brittle is False
    assert Locator(strategy="css", selector="#a").brittle is False
    assert Locator(strategy="text", text="Sign in").brittle is True
    assert Locator(strategy="nth", nth=2).brittle is True


def test_locator_describes_itself_for_the_review_ui():
    assert role().describe() == 'role=textbox name="Username"'
    assert Locator(strategy="css", selector="#name").describe() == "css=#name"


# --- assertions ------------------------------------------------------------


def test_an_assertion_requires_a_subject():
    with pytest.raises(ValidationError, match="requires a value"):
        Assertion(kind="url_contains")
    with pytest.raises(ValidationError, match="requires a locator"):
        Assertion(kind="element_visible")
    with pytest.raises(ValidationError, match="requires a count"):
        Assertion(kind="element_count", locator=role())


def test_negation_reads_correctly():
    check = Assertion(kind="url_contains", value="/signin", negate=True)
    assert check.describe() == "NOT URL contains '/signin'"


# --- steps -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"action": "click"}, "requires at least one locator"),
        ({"action": "navigate"}, "requires a url"),
        ({"action": "assert"}, "requires an assertion"),
        ({"action": "script", "code": None}, "requires code"),
        ({"action": "fill_form"}, "requires at least one field"),
        ({"action": "wait"}, "requires wait_for"),
    ],
)
def test_a_step_must_carry_what_its_action_needs(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        Step(id="s1", **kwargs)


def test_extract_requires_an_output_name():
    with pytest.raises(ValidationError, match="requires an output name"):
        Step(id="s1", action="extract", locators=[role("Score", "status")])


def test_a_templated_selector_is_refused():
    """A templated selector is a selector-injection hole and hides drift."""
    with pytest.raises(ValidationError, match="not allowed inside a locator"):
        Step(
            id="s1",
            action="click",
            locators=[Locator(strategy="css", selector="#user-{{input.id}}")],
        )


def test_a_templated_form_field_locator_is_refused():
    with pytest.raises(ValidationError, match="not allowed inside a form field locator"):
        Step(
            id="s1",
            action="fill_form",
            fields=[
                FormField(
                    name="a",
                    value="x",
                    locators=[Locator(strategy="css", selector="#{{input.f}}")],
                )
            ],
        )


def test_a_step_reports_the_templates_it_uses():
    step = Step(id="s1", action="fill", locators=[role()], value="{{secret.password}}")
    assert step.references() == {("secret", "password")}


def test_form_field_values_are_included_in_references():
    step = Step(
        id="s1",
        action="fill_form",
        fields=[
            FormField(name="Username", value="{{secret.username}}", locators=[role()]),
            FormField(name="Search", value="{{input.term}}", locators=[role("Search")]),
        ],
    )
    assert step.references() == {("secret", "username"), ("input", "term")}


# --- the three validators that stop a silent batch failure -----------------


def test_publishing_a_script_step_is_refused_unless_opted_into():
    with pytest.raises(ValidationError, match="cannot publish"):
        simple(
            status="ready",
            row_steps=[Step(id="s1", action="script", code="await page.click('x')")],
        )


def test_a_draft_may_hold_a_script_step_so_a_person_can_read_it():
    """The review UI has to be able to show the code being approved."""
    use_case = simple(row_steps=[Step(id="s1", action="script", code="await page.click('x')")])
    assert use_case.status == "draft"
    assert [s.id for s in use_case.script_steps] == ["s1"]
    assert use_case.blocked_scripts == ["s1"], "it still refuses to execute"


def test_script_steps_publish_once_opted_in():
    use_case = simple(
        status="ready",
        allow_scripts=True,
        row_steps=[Step(id="s1", action="script", code="await page.click('x')")],
    )
    assert use_case.row_steps[0].action == "script"
    assert use_case.blocked_scripts == []


def test_setup_may_not_reference_a_per_row_input():
    """Setup runs once, so a row input there applies row 1's value to all rows."""
    with pytest.raises(ValidationError, match="cannot reference a per-row input"):
        simple(
            inputs=[InputSpec(name="term")],
            setup_steps=[Step(id="u1", action="fill", locators=[role()], value="{{input.term}}")],
        )


def test_setup_may_reference_a_secret():
    use_case = simple(
        secrets=[SecretSpec(name="password")],
        setup_steps=[
            Step(id="u1", action="fill", locators=[role()], value="{{secret.password}}")
        ],
    )
    assert use_case.setup_steps[0].value == "{{secret.password}}"


def test_a_template_must_reference_a_declared_name():
    with pytest.raises(ValidationError, match="not declared"):
        simple(row_steps=[Step(id="s1", action="fill", locators=[role()], value="{{input.nope}}")])


def test_declaring_the_name_makes_it_valid():
    use_case = simple(
        inputs=[InputSpec(name="term")],
        row_steps=[Step(id="s1", action="fill", locators=[role()], value="{{input.term}}")],
    )
    assert use_case.input_names == {"term"}


def test_duplicate_step_ids_are_refused():
    with pytest.raises(ValidationError, match="duplicate step id"):
        simple(
            row_steps=[
                Step(id="s1", action="navigate", url="https://example.com"),
                Step(id="s1", action="navigate", url="https://example.org"),
            ]
        )


def test_a_step_id_may_not_collide_across_phases():
    with pytest.raises(ValidationError, match="duplicate step id"):
        simple(
            setup_steps=[Step(id="s1", action="navigate", url="https://example.com")],
            row_steps=[Step(id="s1", action="navigate", url="https://example.org")],
        )


def test_declaring_an_output_nothing_extracts_is_refused():
    with pytest.raises(ValidationError, match="never extracted"):
        simple(outputs=["score"])


def test_a_declared_output_backed_by_an_extract_step_is_valid():
    use_case = simple(
        row_steps=[
            Step(id="s1", action="extract", locators=[role("Score", "status")], output="score")
        ],
        outputs=["score"],
    )
    assert use_case.outputs == ["score"]


# --- templating ------------------------------------------------------------


def test_template_refs_finds_every_kind():
    found = template_refs(
        {"a": "{{input.x}}", "b": ["{{secret.y}}", {"c": "{{ env.Z }}"}]}
    )
    assert found == {("input", "x"), ("secret", "y"), ("env", "Z")}


def test_has_template_is_false_for_plain_text():
    assert has_template("no braces here") is False
    assert has_template("{{ not.a.kind }}") is False


def test_rendering_substitutes_through_nested_structures():
    rendered = render_template(
        {"url": "{{input.base}}/p", "fields": ["{{secret.pw}}"]},
        inputs={"base": "https://example.com"},
        secrets={"pw": "hunter2"},
    )
    assert rendered == {"url": "https://example.com/p", "fields": ["hunter2"]}


def test_rendering_a_missing_value_raises_rather_than_typing_nothing():
    """Silently typing "" into a login form and reporting success is the worst
    failure mode a batch can have, so this must be loud."""
    with pytest.raises(MissingValue, match="secret.password"):
        render_template("{{secret.password}}", inputs={}, secrets={})


def test_rendering_leaves_non_strings_alone():
    assert render_template({"n": 5, "flag": True}, inputs={}, secrets={}) == {"n": 5, "flag": True}


def test_whitespace_inside_the_braces_is_tolerated():
    assert render_template("{{ input.x }}", inputs={"x": "ok"}, secrets={}) == "ok"


# --- helpers ---------------------------------------------------------------


def test_missing_inputs_are_reported_before_the_browser_opens():
    use_case = simple(
        inputs=[
            InputSpec(name="required_one"),
            InputSpec(name="optional_one", required=False),
            InputSpec(name="defaulted", default="d"),
        ],
        row_steps=[
            Step(
                id="s1",
                action="fill",
                locators=[role()],
                value="{{input.required_one}}{{input.optional_one}}{{input.defaulted}}",
            )
        ],
    )
    assert use_case.missing_inputs({}) == ["required_one"]
    assert use_case.missing_inputs({"required_one": "x"}) == []


def test_missing_secrets_are_reported():
    use_case = simple(
        secrets=[SecretSpec(name="password"), SecretSpec(name="pin", required=False)],
        setup_steps=[
            Step(id="u1", action="fill", locators=[role()], value="{{secret.password}}")
        ],
    )
    assert use_case.missing_secrets({}) == ["password"]


def test_defaults_are_filled_in():
    use_case = simple(
        inputs=[InputSpec(name="term", default="fallback")],
        row_steps=[Step(id="s1", action="fill", locators=[role()], value="{{input.term}}")],
    )
    assert use_case.with_defaults({}) == {"term": "fallback"}
    assert use_case.with_defaults({"term": "given"}) == {"term": "given"}


def test_brittle_steps_are_surfaced_for_review():
    use_case = simple(
        row_steps=[
            Step(id="s1", action="click", locators=[Locator(strategy="text", text="Go")]),
            Step(id="s2", action="click", locators=[role("Go", "button")]),
        ]
    )
    assert [s.id for s in use_case.brittle_steps()] == ["s1"]


def test_a_draft_is_not_runnable_until_published():
    use_case = simple()
    assert use_case.runnable is False
    use_case.status = "ready"
    assert use_case.runnable is True


def test_bump_produces_the_next_version_without_mutating_the_original():
    first = simple()
    second = first.bump()
    assert (first.version, second.version) == (1, 2)


def test_all_steps_covers_every_phase_including_row_reset():
    use_case = simple(
        setup_steps=[Step(id="u1", action="navigate", url="https://example.com")],
        row_reset=Step(id="reset", action="navigate", url="https://example.com/x"),
        teardown_steps=[Step(id="t1", action="navigate", url="https://example.com/out")],
    )
    assert [s.id for s in use_case.all_steps] == ["u1", "reset", "s1", "t1"]


# --- persistence -----------------------------------------------------------


def test_a_use_case_round_trips_through_json_with_its_assert_alias():
    original = simple(
        row_steps=[
            Step(
                id="s1",
                action="assert",
                assertion=Assertion(kind="url_contains", value="/done"),
            )
        ]
    )
    payload = json.loads(original.model_dump_json(by_alias=True))
    assert "assert" in payload["row_steps"][0], "stored JSON uses the readable alias"

    restored = UseCase.model_validate(payload)
    assert restored.row_steps[0].assertion.value == "/done"


def test_wait_for_round_trips():
    original = simple(
        row_steps=[Step(id="s1", action="wait", wait_for=WaitFor(kind="time", seconds=2.0))]
    )
    restored = UseCase.model_validate(json.loads(original.model_dump_json(by_alias=True)))
    assert restored.row_steps[0].wait_for.seconds == 2.0


def test_unknown_fields_are_rejected_rather_than_silently_dropped():
    with pytest.raises(ValidationError):
        UseCase(name="x", row_steps=[], nonsense=True)


# --- actions that legitimately have no target ------------------------------
#
# `browser_press_key` sends a key to whatever has focus and `browser_file_upload`
# answers an open file chooser. Neither is ever recorded with a target, and
# requiring one made a recording containing an Enter keypress impossible to
# distil at all.


def test_press_needs_no_locator():
    step = Step(id="s1", action="press", value="Enter")
    assert step.locators == []
    assert step.summary() == "press"


def test_press_still_requires_a_key():
    with pytest.raises(ValidationError, match="requires a key"):
        Step(id="s1", action="press")


def test_press_may_still_carry_a_locator_when_one_was_recorded():
    step = Step(id="s1", action="press", value="Enter", locators=[role("Search", "textbox")])
    assert step.locators[0].name == "Search"


def test_upload_needs_no_locator_but_needs_a_path():
    assert Step(id="s1", action="upload", value="/tmp/a.pdf").locators == []
    with pytest.raises(ValidationError, match="requires a file path"):
        Step(id="s1", action="upload")


@pytest.mark.parametrize("action", ["click", "fill", "select", "hover", "extract"])
def test_the_actions_that_do_need_a_target_still_require_one(action):
    kwargs = {"output": "x"} if action == "extract" else {}
    with pytest.raises(ValidationError, match="requires at least one locator"):
        Step(id="s1", action=action, **kwargs)


def test_publishing_with_an_input_nothing_reads_is_refused():
    """Asking for a value on every row and ignoring it is never right."""
    with pytest.raises(ValidationError, match="no step reads them"):
        simple(status="ready", inputs=[InputSpec(name="unused_thing")])


def test_a_draft_may_hold_an_unused_input_so_it_can_be_reviewed():
    use_case = simple(inputs=[InputSpec(name="unused_thing")])
    assert [i.name for i in use_case.inputs] == ["unused_thing"]


# --- assertions the allowlist makes impossible -----------------------------
#
# Regression: a distilled use case asserted `NOT url_contains "ixl.com"` while
# restricted to www.ixl.com. Every row failed, and the message blamed the page.


@pytest.mark.parametrize(
    ("value", "negate", "domains", "impossible"),
    [
        # The bug, exactly.
        ("ixl.com", True, ["www.ixl.com"], True),
        ("www.ixl.com", True, ["www.ixl.com"], True),
        # Negating a path is the correct pattern and must survive.
        ("/signin", True, ["www.ixl.com"], False),
        ("/dashboard", False, ["www.ixl.com"], False),
        # Asserting the domain positively is pointless but not impossible.
        ("ixl.com", False, ["www.ixl.com"], False),
        # A host that is not reachable at all.
        ("google.com", False, ["www.ixl.com"], True),
        # Reachable via a wildcard entry.
        ("shop.example.com", False, ["*.example.com"], False),
        # Several domains: negation only impossible if it holds for all of them.
        ("example.com", True, ["www.example.com", "cdn.example.com"], True),
        ("example.com", True, ["www.example.com", "other.org"], False),
        # No confinement means nothing is provable.
        ("ixl.com", True, ["*"], False),
        ("ixl.com", True, [], False),
    ],
)
def test_unsatisfiable_url_assertions_are_detected(value, negate, domains, impossible):
    check = Assertion(kind="url_contains", value=value, negate=negate)
    assert (check.unsatisfiable_reason(domains) is not None) is impossible


def test_only_url_assertions_are_judged():
    """Nothing is knowable in advance about page text or titles."""
    for kind in ("text_present", "title_contains"):
        check = Assertion(kind=kind, value="anything", negate=True)
        assert check.unsatisfiable_reason(["www.ixl.com"]) is None


def test_publishing_an_impossible_assertion_is_refused():
    with pytest.raises(ValidationError, match="can never pass"):
        simple(
            status="ready",
            allowed_domains=["www.ixl.com"],
            row_steps=[
                Step(
                    id="s1",
                    action="assert",
                    assertion=Assertion(kind="url_contains", value="ixl.com", negate=True),
                )
            ],
        )


def test_an_impossible_session_check_is_refused_too():
    """One that never passes makes the batch re-run sign-in after every row."""
    with pytest.raises(ValidationError, match="can never pass"):
        simple(
            status="ready",
            allowed_domains=["www.ixl.com"],
            session_check=Assertion(kind="url_contains", value="ixl.com", negate=True),
        )


def test_a_draft_may_hold_one_so_it_can_be_seen_and_repaired():
    use_case = simple(
        allowed_domains=["www.ixl.com"],
        row_steps=[
            Step(
                id="s1",
                action="assert",
                assertion=Assertion(kind="url_contains", value="ixl.com", negate=True),
            )
        ],
    )
    assert [where for where, _ in use_case.impossible_assertions()] == ["s1"]


def test_a_sound_use_case_reports_no_impossible_assertions():
    use_case = simple(
        allowed_domains=["www.ixl.com"],
        session_check=Assertion(kind="url_contains", value="/signin", negate=True),
        row_steps=[
            Step(id="s1", action="assert", assertion=Assertion(kind="text_present", value="Done"))
        ],
    )
    assert use_case.impossible_assertions() == []
