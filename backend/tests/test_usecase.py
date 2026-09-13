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
    """A templated selector is a selector-injection hole.

    The value is spliced into a query language, so a spreadsheet cell reading
    ``a, button`` addresses every link on the page.
    """
    with pytest.raises(ValidationError, match="not allowed in a locator's selector"):
        Step(
            id="s1",
            action="click",
            locators=[Locator(strategy="css", selector="#user-{{input.id}}")],
        )


def test_a_templated_frame_selector_is_refused():
    """`frames` is CSS too, and a scope is where one would most easily hide."""
    with pytest.raises(ValidationError, match="not allowed in a locator's frames"):
        Step(
            id="s1",
            action="click",
            locators=[
                Locator(strategy="role", role="button", frames=["#f-{{input.id}}"])
            ],
        )


def test_a_templated_selector_inside_a_scope_is_refused():
    """The rung a reviewer reads says `role=button`; the injection is the
    thing it is scoped inside."""
    with pytest.raises(ValidationError, match="not allowed in a locator's selector"):
        Step(
            id="s1",
            action="click",
            locators=[
                Locator(
                    strategy="role",
                    role="button",
                    within=Locator(strategy="css", selector=".row-{{input.id}}"),
                )
            ],
        )


def test_a_templated_form_field_selector_is_refused():
    with pytest.raises(ValidationError, match="not allowed in a locator's selector"):
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


def test_a_templated_accessible_name_is_allowed():
    """The case the blanket ban made unrepresentable.

    A search whose dropdown is filled from the row -- type a customer number,
    pick the customer name -- has no replayable locator otherwise: the name
    that was recorded belongs to the row it was recorded on, and no other rung
    can say "the name from this row's column". Playwright matches a name as a
    plain string, so there is no syntax for a value to escape into.
    """
    step = Step(
        id="s1",
        action="click",
        locators=[Locator(strategy="role", role="option", name="{{input.customer_name}}")],
    )

    assert step.locators[0].name == "{{input.customer_name}}"


def test_an_input_named_only_by_a_locator_is_still_a_declared_reference():
    """Otherwise it passes review undeclared and then looks for the literal
    text "{{input.customer_name}}" on every row."""
    step = Step(
        id="s1",
        action="click",
        locators=[Locator(strategy="role", role="option", name="{{input.customer_name}}")],
    )

    assert ("input", "customer_name") in step.references()


def test_a_templated_rung_renders_for_the_row_in_hand():
    locator = Locator(
        strategy="role",
        role="option",
        name="{{input.customer_name}}",
        within=Locator(strategy="role", role="listbox", has_text="{{input.customer_name}}"),
    )

    rendered = locator.render(lambda value: value.replace("{{input.customer_name}}", "Globex"))

    assert rendered.name == "Globex"
    assert rendered.within is not None and rendered.within.has_text == "Globex"
    assert locator.name == "{{input.customer_name}}", "the stored rung is untouched"


def test_a_rung_with_no_template_renders_to_itself():
    """Every rung of every recording made before this existed takes this path,
    so it must not allocate or change anything."""
    locator = Locator(strategy="role", role="button", name="Save")

    assert locator.render(lambda value: "SHOULD NOT BE CALLED") is locator


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


def test_the_env_namespace_answers_from_the_deployment():
    """What makes one document runnable in dev, UAT and production.

    A recording holds the URLs of the environment it was made against. An input
    cannot carry the difference -- inputs are per row, and a base URL in a
    spreadsheet column is how a UAT dataset ends up pointed at production -- so
    the document names the part that varies and the deployment answers it.
    """
    rendered = render_template(
        "{{env.base_url}}/orders?ref={{input.reference}}",
        inputs={"reference": "A-1024"},
        secrets={},
        env={"base_url": "https://uat.example.com"},
    )
    assert rendered == "https://uat.example.com/orders?ref=A-1024"


def test_an_unanswered_env_reference_stops_rather_than_rendering_nothing():
    """Same strictness as an input: a blank URL is not a navigation."""
    with pytest.raises(MissingValue):
        render_template("{{env.base_url}}/orders", inputs={}, secrets={}, env={})


# ---------------------------------------------------------------------------
# Saying where to look, not only what to look for
# ---------------------------------------------------------------------------


def test_a_scoped_locator_reads_as_where_it_is():
    from usecase import Locator

    locator = Locator(
        strategy="role",
        role="button",
        name="Invite",
        within=Locator(strategy="role", role="dialog", name="Invite a user"),
    )

    assert locator.describe() == 'role=button name="Invite" in role=dialog name="Invite a user"'
    assert locator.scoped


def test_a_scope_made_of_markup_makes_the_whole_rung_markup():
    """The ladder walks semantic rungs first because they survive a redesign.

    A role scoped inside a CSS selector does not: the selector breaks when the
    markup changes, however durable the rung hanging off it. Calling it
    semantic would order the more fragile of two rungs first, which is the one
    thing that ordering exists to prevent.
    """
    from usecase import Locator

    assert Locator(strategy="role", role="button", name="X").semantic
    assert not Locator(
        strategy="role",
        role="button",
        name="X",
        within=Locator(strategy="css", selector=".row"),
    ).semantic


def test_a_scope_chain_has_a_depth_limit():
    from usecase import Locator, MAX_SCOPE_DEPTH

    scope = Locator(strategy="role", role="main")
    for _ in range(MAX_SCOPE_DEPTH):
        scope = Locator(strategy="role", role="group", within=scope)

    with pytest.raises(ValidationError):
        Locator(strategy="role", role="button", name="X", within=scope)


def test_only_the_outermost_rung_carries_the_frames():
    """`_build` walks the scope chain and composes one call per level, all of
    them already inside the frame. A frame hop buried three levels in is a
    shape the executor would have to interpret rather than perform."""
    from usecase import Locator

    with pytest.raises(ValidationError):
        Locator(
            strategy="role",
            role="button",
            within=Locator(strategy="role", role="dialog", frames=["#pay"]),
        )


def test_templating_is_refused_inside_a_scope_too():
    """A templated selector is a selector-injection hole wherever it sits, and
    `within` added a second place for one to hide."""
    from usecase import Step, Locator

    with pytest.raises(ValidationError):
        Step(
            id="s1",
            action="click",
            locators=[
                Locator(
                    strategy="role",
                    role="button",
                    name="Go",
                    within=Locator(strategy="css", selector="#row-{{input.id}}"),
                )
            ],
        )


def test_a_definition_recorded_before_scopes_existed_still_validates():
    """Every new field defaults to "not said", so a stored use case written
    under the previous schema means exactly what it meant then."""
    from usecase import Locator

    locator = Locator.model_validate({"strategy": "role", "role": "button", "name": "Save"})

    assert locator.within is None and locator.frames == [] and not locator.scoped
    assert locator.describe() == 'role=button name="Save"'


# ---------------------------------------------------------------------------
# A locator made of the row's own data
# ---------------------------------------------------------------------------


def searched(name: str = "Acme Ltd", *, typed: str = "{{input.customer_number}}"):
    """Type a search value, then click a suggestion by the text it showed."""
    from usecase import Locator, Step

    return [
        Step(id="s1", action="navigate", url="https://crm.test/"),
        Step(
            id="s2",
            action="fill",
            value=typed,
            locators=[Locator(strategy="placeholder", text="Customer number")],
        ),
        Step(
            id="s3",
            action="click",
            locators=[
                Locator(strategy="role", role="option", name=name),
                Locator(strategy="text", text=name),
            ],
        ),
    ]


def test_a_suggestion_clicked_after_a_per_row_search_is_recognised():
    from usecase import data_derived_clicks

    assert data_derived_clicks(searched()) == [2]


def test_a_suggestion_clicked_after_a_fixed_search_is_left_alone():
    """A workflow that searches the same thing every row has a stable
    suggestion, and rewriting it would throw away a good locator and make the
    step ambiguous among its siblings for nothing."""
    from usecase import data_derived_clicks

    assert data_derived_clicks(searched(typed="OVERDUE")) == []


def test_a_static_menu_item_is_left_alone():
    """`menuitem` is in the suggestion roles, so the guard that saves this is
    the per-row search above it -- which a menu does not have."""
    from usecase import Locator, Step, data_derived_clicks

    steps = [
        Step(
            id="s1",
            action="click",
            locators=[Locator(strategy="role", role="button", name="Actions")],
        ),
        Step(
            id="s2",
            action="click",
            locators=[Locator(strategy="role", role="menuitem", name="Export as CSV")],
        ),
    ]

    assert data_derived_clicks(steps) == []


def test_an_ordinary_button_after_a_per_row_fill_is_left_alone():
    """Typing a reference and clicking Submit is the commonest shape there is.
    "Submit" is the page's word, not the row's."""
    from usecase import Locator, Step, data_derived_clicks

    steps = [
        Step(
            id="s1",
            action="fill",
            value="{{input.reference}}",
            locators=[Locator(strategy="label", text="Reference")],
        ),
        Step(
            id="s2",
            action="click",
            locators=[Locator(strategy="role", role="button", name="Submit")],
        ),
    ]

    assert data_derived_clicks(steps) == []


def test_waiting_and_pressing_enter_do_not_break_the_connection():
    """A real recording of a search has an Enter and a wait between the typing
    and the click. Neither changes whose data the suggestion is showing."""
    from usecase import Locator, Step, data_derived_clicks

    steps = searched()
    steps.insert(2, Step(id="w1", action="press", value="Enter"))
    steps.insert(3, Step(id="w2", action="wait", wait_for={"kind": "time", "seconds": 1}))

    assert data_derived_clicks(steps) == [4]


def test_the_recorded_name_is_removed_from_what_executes():
    from usecase import strip_data_locators

    steps = searched()
    notes = strip_data_locators(steps)

    assert [loc.describe() for loc in steps[2].locators] == ["role=option"]
    assert len(notes) == 1 and "Acme Ltd" in notes[0]


def test_the_recorded_name_is_kept_where_a_reviewer_can_see_it():
    """A person deciding whether this was the right call needs to see what the
    recording actually said. `rejected_locators` is shown and never executed."""
    from usecase import strip_data_locators

    steps = searched()
    strip_data_locators(steps)

    assert [loc.describe() for loc in steps[2].rejected_locators] == [
        'role=option name="Acme Ltd"',
        "text='Acme Ltd'",
    ]


def test_the_recorded_name_is_deleted_rather_than_demoted():
    """Keeping it as a fallback rung would be worse than keeping nothing.

    A ladder takes the first rung matching exactly one element, so a demoted
    "Acme Ltd" sits unused on every row that works -- and fires on exactly the
    rows where the search returned several and the unnamed rung was refused as
    ambiguous. It would act only when it is certainly wrong.
    """
    from usecase import strip_data_locators

    steps = searched()
    strip_data_locators(steps)

    assert not any("Acme Ltd" in loc.describe() for loc in steps[2].locators)


def test_a_dropdown_that_echoes_what_was_typed_is_caught_whatever_its_markup():
    """The second rule, and the certain one.

    When the suggestion repeats the search value -- "C-1001 Acme Ltd" -- the
    locator contains, verbatim, a value the person declared as per-row data.
    That is the same string twice, not an inference, so it holds for a dropdown
    built from plain divs with no ARIA role for the first rule to see.
    """
    from usecase import Locator, Step, data_derived_clicks

    steps = [
        Step(
            id="s1",
            action="fill",
            value="{{input.customer_number}}",
            locators=[Locator(strategy="placeholder", text="Search")],
        ),
        Step(
            id="s2",
            action="click",
            locators=[Locator(strategy="text", text="C-1001 Acme Ltd")],
        ),
    ]

    assert data_derived_clicks(steps, ["C-1001"]) == [1]
    assert data_derived_clicks(steps) == [], "without the recorded value there is no signal"


def test_a_short_recorded_value_is_not_treated_as_evidence():
    """"12" appears inside half the labels on a page, and matching on one would
    strip the name off a perfectly good locator."""
    from usecase import Locator, Step, data_derived_clicks

    steps = [
        Step(
            id="s1",
            action="fill",
            value="{{input.qty}}",
            locators=[Locator(strategy="label", text="Quantity")],
        ),
        Step(
            id="s2",
            action="click",
            locators=[Locator(strategy="role", role="button", name="Add 12 items")],
        ),
    ]

    assert data_derived_clicks(steps, ["12"]) == []


def test_a_text_only_suggestion_is_reported_rather_than_emptied():
    """Every rung named it by its text, so nothing is left once the text goes.

    Deleting the step would be inventing a locator for an element nobody here
    has seen. Saying so lets a person point it at something else, and a locator
    may now take a column as `{{input.name}}`.
    """
    from usecase import Locator, Step, strip_data_locators

    steps = [
        Step(
            id="s1",
            action="fill",
            value="{{input.customer_number}}",
            locators=[Locator(strategy="placeholder", text="Search")],
        ),
        Step(
            id="s2",
            action="click",
            locators=[Locator(strategy="text", text="C-1001 Acme Ltd")],
        ),
    ]

    notes = strip_data_locators(steps, ["C-1001"])

    assert len(notes) == 1
    assert "Edit the step's locator before publishing" in notes[0]
    assert [loc.describe() for loc in steps[1].locators] == ["text='C-1001 Acme Ltd'"], (
        "left alone, because there is no honest replacement"
    )


# ---------------------------------------------------------------------------
# A URL made of one sign-in's data
# ---------------------------------------------------------------------------


AUTHORIZE = (
    "https://login.example.com/oauth2/authorize"
    "?client_id=8b21&response_type=code"
    "&redirect_uri=https%3A%2F%2Fcrm.example.com%2Fcb&scope=openid"
    "&state=Ab9xQ2zKp&nonce=Nn41Kd"
)
CALLBACK = "https://crm.example.com/cb?code=0.AXkAr9&state=Ab9xQ2zKp&session_state=4f1c"


def test_single_use_sign_in_parameters_are_removed():
    from usecase import strip_volatile_params

    cleaned, removed = strip_volatile_params(AUTHORIZE)

    assert removed == ["state", "nonce"]
    assert "state=" not in cleaned and "nonce=" not in cleaned
    assert "client_id=8b21" in cleaned and "response_type=code" in cleaned


def test_everything_kept_is_kept_byte_for_byte():
    """This is only allowed to shorten a URL.

    Parsing the query and rebuilding it re-encodes a `redirect_uri` that
    arrived percent-encoded, and escapes the braces of a parameter already
    templated -- turning a working substitution into a literal that matches
    nothing.
    """
    from usecase import strip_volatile_params

    url = (
        "https://crm.example.com/orders"
        "?ref={{input.reference}}&back=https%3A%2F%2Fa.test%2Fx&state=Ab9"
    )

    cleaned, _ = strip_volatile_params(url)

    assert cleaned == (
        "https://crm.example.com/orders?ref={{input.reference}}&back=https%3A%2F%2Fa.test%2Fx"
    )


def test_a_url_with_nothing_volatile_is_returned_unchanged():
    from usecase import strip_volatile_params

    url = "https://crm.example.com/orders?ref=A-1024&sort=date"

    assert strip_volatile_params(url) == (url, [])


def test_a_sign_in_callback_is_recognised():
    from usecase import is_authorization_callback

    assert is_authorization_callback(CALLBACK)
    assert is_authorization_callback("https://a.test/acs?SAMLResponse=PHNhbWxw")


def test_a_product_code_is_not_a_sign_in_callback():
    """`code` on its own is an ordinary word a real application uses for a
    product code or a country code. `code` *with* `state` is OAuth."""
    from usecase import is_authorization_callback

    assert not is_authorization_callback("https://shop.test/items?code=SKU-11")
    assert not is_authorization_callback("https://shop.test/items?state=NY")


def test_the_application_is_the_origin_not_the_identity_provider():
    """Behind SSO the first recorded URL is the identity provider, so binding
    `{{env.base_url}}` to it meant promoting the use case repointed the
    identity provider at the UAT address."""
    from usecase import application_origin

    origin = application_origin([AUTHORIZE, CALLBACK, "https://crm.example.com/orders"])

    assert origin == "https://crm.example.com"


def test_a_recording_that_never_leaves_the_identity_provider_reads_the_redirect_uri():
    """An authorization request states where the application lives -- that is
    what the parameter is for."""
    from usecase import application_origin

    assert application_origin([AUTHORIZE]) == "https://crm.example.com"


def test_an_ordinary_recording_still_takes_its_first_url():
    from usecase import application_origin

    assert (
        application_origin(["https://crm.example.com/orders", "https://docs.test/help"])
        == "https://crm.example.com"
    )


def test_the_callback_step_is_dropped_and_the_rest_are_cleaned():
    from usecase import Step, clean_recorded_urls

    steps = [
        Step(id="s1", action="navigate", url=AUTHORIZE),
        Step(id="s2", action="navigate", url=CALLBACK),
        Step(id="s3", action="navigate", url="https://crm.example.com/orders?sessionDataKey=91ab"),
    ]

    notes = clean_recorded_urls(steps)

    assert [step.id for step in steps] == ["s1", "s3"], "the callback is gone"
    assert "state=" not in (steps[0].url or "")
    assert steps[1].url == "https://crm.example.com/orders"
    assert len(notes) == 3


def test_a_url_check_stops_asserting_on_a_value_that_changes_every_sign_in():
    from usecase import Step, clean_recorded_urls

    steps = [
        Step(
            id="s1",
            action="assert",
            **{"assert": {"kind": "url_contains", "value": "/orders?ref=A-1&state=Ab9"}},
        )
    ]

    notes = clean_recorded_urls(steps)

    assert steps[0].assertion.value == "/orders?ref=A-1"
    assert len(notes) == 1


def test_a_recording_with_no_sign_in_is_left_completely_alone():
    from usecase import Step, clean_recorded_urls

    steps = [
        Step(id="s1", action="navigate", url="https://crm.example.com/orders?ref=A-1024"),
        Step(id="s2", action="click", locators=[role("Submit")]),
    ]
    before = [step.model_dump(mode="json") for step in steps]

    assert clean_recorded_urls(steps) == []
    assert [step.model_dump(mode="json") for step in steps] == before


# ---------------------------------------------------------------------------
# Steps that were never the workflow
# ---------------------------------------------------------------------------


def test_tab_keystrokes_are_left_out_of_the_recording():
    """A person filling a form uses Tab to get between the fields, and codegen
    records a `press` aimed at whatever had focus. One real recording had
    `press Shift+Tab` on a "Forgot password?" link, twice, inside a login."""
    from usecase import Locator, Step, drop_focus_keystrokes

    steps = [
        Step(id="s1", action="fill", value="ada", locators=[Locator(strategy="label", text="Email")]),
        Step(id="s2", action="press", value="Tab", locators=[Locator(strategy="label", text="Email")]),
        Step(
            id="s3",
            action="press",
            value="Shift+Tab",
            locators=[Locator(strategy="role", role="link", name="Forgot password?")],
        ),
        Step(id="s4", action="fill", value="pw", locators=[Locator(strategy="label", text="Password")]),
    ]

    notes = drop_focus_keystrokes(steps)

    assert [step.id for step in steps] == ["s1", "s4"]
    assert len(notes) == 1 and "Tab" in notes[0]


def test_a_key_that_does_something_is_kept():
    """Enter submits, Escape dismisses, the arrows choose from a list. Only
    the two keys that purely move focus go."""
    from usecase import Locator, Step, drop_focus_keystrokes

    steps = [
        Step(id="s1", action="press", value="Enter", locators=[Locator(strategy="label", text="Password")]),
        Step(id="s2", action="press", value="Escape"),
        Step(id="s3", action="press", value="ArrowDown"),
    ]

    assert drop_focus_keystrokes(steps) == []
    assert [step.id for step in steps] == ["s1", "s2", "s3"]


# ---------------------------------------------------------------------------
# Shapes that cannot replay
# ---------------------------------------------------------------------------


def test_the_two_lists_of_meaningless_roles_agree():
    """`usecase` keeps its own copy so the schema does not depend on the
    parser. They have to mean the same thing."""
    from snapshot import STRUCTURAL_ROLES
    from usecase import STRUCTURAL_ROLES_WITHOUT_MEANING

    assert STRUCTURAL_ROLES_WITHOUT_MEANING == STRUCTURAL_ROLES


def signing_in(**overrides):
    """A use case whose setup signs in and whose row does the work."""
    from usecase import Locator, Step, UseCase

    body = {
        "name": "work",
        "allowed_domains": ["example.com"],
        "secrets": [{"name": "password"}],
        "setup_steps": [
            Step(id="s1", action="navigate", url="https://example.com/login"),
            Step(
                id="s2",
                action="fill",
                value="{{secret.password}}",
                locators=[Locator(strategy="label", text="Password")],
            ),
            Step(
                id="s3",
                action="click",
                locators=[Locator(strategy="role", role="button", name="Sign in")],
            ),
        ],
        "row_steps": [
            Step(
                id="s4",
                action="click",
                locators=[Locator(strategy="role", role="link", name="Reports")],
            )
        ],
    }
    body.update(overrides)
    return UseCase(**body)


def ready(draft):
    from usecase import UseCase

    return UseCase.model_validate(
        {**draft.model_dump(mode="json", by_alias=True), "status": "ready"}
    )


def test_a_step_that_can_only_count_anonymous_wrappers_cannot_be_published():
    """`role=generic [24]` -- the shape one real recording published and then
    failed on, twice, with the draft's own warning saying it would."""
    from usecase import Locator, Step

    draft = signing_in(
        row_steps=[
            Step(id="s4", action="click", locators=[Locator(strategy="role", role="generic", nth=24)])
        ]
    )

    with pytest.raises(ValidationError, match="anonymous page wrappers"):
        ready(draft)


def test_a_wrapper_rung_is_fine_when_another_rung_can_name_the_element():
    """The ladder only has to have one rung that can work. A repair that puts
    a real name in front of the wrapper has fixed the step."""
    from usecase import Locator, Step

    draft = signing_in(
        row_steps=[
            Step(
                id="s4",
                action="click",
                locators=[
                    Locator(strategy="role", role="radio", name="Nayra Asati"),
                    Locator(strategy="role", role="generic", nth=24),
                ],
            )
        ]
    )

    ready(draft)


def signs_out(**overrides):
    from usecase import Locator, Step

    return signing_in(
        row_steps=[
            Step(id="s4", action="click", locators=[Locator(strategy="role", role="link", name="Reports")]),
            Step(id="s5", action="click", locators=[Locator(strategy="role", role="link", name="Sign out")]),
        ],
        **overrides,
    )


def test_a_row_that_signs_out_can_still_be_published():
    """It runs one record perfectly, and somebody may only ever want one.

    Refusing this at publish left a person with a recording they had just spent
    ten minutes on and could only delete -- a worse outcome than the failing
    batch it was preventing. The refusal belongs where the failure is.
    """
    ready(signs_out())


def test_a_row_that_signs_out_cannot_be_batched_when_signing_in_is_setup():
    """The defect behind "record one worked and everything after failed".

    Signing in happens once for a whole batch, deliberately -- a thousand
    records must not sign in a thousand times. A record that signs out destroys
    that shared session, so record two starts with nothing signed in and every
    step fails. No amount of healing a locator fixes it.
    """
    from usecase import unbatchable_reasons

    reasons = unbatchable_reasons(signs_out())

    assert len(reasons) == 1
    assert "signs out at the end of every record" in reasons[0]
    assert "single record runs fine" in reasons[0], "says what does work"


def test_a_row_that_signs_itself_in_may_sign_itself_out():
    """Then each record is self-contained, and the sign-out is correct."""
    from usecase import Locator, Step, unbatchable_reasons

    draft = signing_in(
        setup_steps=[Step(id="s1", action="navigate", url="https://example.com/login")],
        row_steps=[
            Step(
                id="s3",
                action="fill",
                value="{{secret.password}}",
                locators=[Locator(strategy="label", text="Password")],
            ),
            Step(id="s4", action="click", locators=[Locator(strategy="role", role="link", name="Reports")]),
            Step(id="s5", action="click", locators=[Locator(strategy="role", role="link", name="Sign out")]),
        ],
    )

    assert unbatchable_reasons(draft) == []


def test_a_session_check_makes_the_sign_out_survivable():
    """The run notices it has been signed out and runs setup again. That is
    what the check is for, and having one is the third way to fix this."""
    from usecase import unbatchable_reasons

    draft = signs_out(
        session_check={"kind": "url_contains", "value": "/login", "negate": True}
    )

    assert unbatchable_reasons(draft) == []


def test_an_ordinary_use_case_can_be_batched():
    from usecase import unbatchable_reasons

    assert unbatchable_reasons(signing_in()) == []


def test_an_ordinary_use_case_publishes_exactly_as_before():
    ready(signing_in())


def test_a_draft_may_hold_anything_while_it_is_being_worked_on():
    """The gate is publishing, not saving. A person fixing a recording has to
    be able to store the broken version they are fixing."""
    from usecase import Locator, Step

    signing_in(
        row_steps=[
            Step(id="s4", action="click", locators=[Locator(strategy="role", role="generic", nth=24)]),
            Step(id="s5", action="click", locators=[Locator(strategy="role", role="link", name="Sign out")]),
        ]
    )


# --- asserting on an attribute ---------------------------------------------
#
# The gap this fills, found by reading what other recorders can check: the
# identifier a later step needs is often in a link rather than in the words a
# person sees, so "the row's link points at account A-1001" was a check
# nothing here could express. `extract` could already read an attribute; only
# asserting on one was missing.


def test_an_attribute_assertion_needs_a_locator_an_attribute_and_a_value():
    from usecase import Assertion

    for missing in (
        {"locator": None, "attribute": "href", "value": "/x"},
        {"locator": {"strategy": "role", "role": "link"}, "attribute": "", "value": "/x"},
        {"locator": {"strategy": "role", "role": "link"}, "attribute": "href", "value": ""},
    ):
        with pytest.raises(ValidationError):
            Assertion(kind="attribute_contains", **missing)


def test_an_attribute_assertion_reads_as_a_person_would_say_it():
    from usecase import Assertion

    check = Assertion(
        kind="attribute_contains",
        locator={"strategy": "role", "role": "link", "name": "Receipt"},
        attribute="href",
        value="/receipt/",
    )

    assert check.describe() == "role=link name=\"Receipt\" has href='/receipt/'"


# --- which rungs prove what an element says --------------------------------


def test_a_rung_that_matched_a_name_has_proved_the_wording():
    from usecase import Locator

    assert Locator(strategy="role", role="button", name="Save").matches_on_text
    assert Locator(strategy="text", text="Save").matches_on_text
    assert Locator(strategy="role", role="row", has_text="Acme").matches_on_text


def test_a_rung_that_says_only_where_to_look_has_not():
    """These are the rungs `Step.expect_text` is checked against, and the
    reason it exists: they match a position, not a control."""
    from usecase import Locator

    assert not Locator(strategy="css", selector="div > svg").matches_on_text
    assert not Locator(strategy="role", role="button").matches_on_text
    assert not Locator(strategy="nth", nth=3).matches_on_text
    # A test id survives the control behind it being replaced -- which is the
    # whole point of one, and why it is not proof of wording.
    assert not Locator(strategy="test_id", text="save-button").matches_on_text
