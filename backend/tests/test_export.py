"""A use case, rendered as a Playwright script somebody can read.

The inverse of `test_codegen.py`, and the property those two share is the one
worth testing hardest: a recording goes in as Python, becomes a document, and
comes back out as Python that means the same thing.

Two rules are load-bearing here and each has a test that fails loudly if it
stops holding. **The output must parse**, because a script nobody can run is
worse than no export -- it looks like a deliverable. And **nothing here may
execute**: this module returns a string, and the one thing the codebase has
always refused to do with generated Python is run it.
"""

from __future__ import annotations

import ast

import pytest

from export import (
    SECRET_PREFIX,
    as_playwright_python,
    condition_expr,
    locator_expr,
    suggested_filename,
    value_expr,
)
from usecase import Assertion, ExtractColumn, Locator, Step, UseCase


def use_case(**overrides) -> UseCase:
    base = dict(
        name="Update billing address",
        status="ready",
        base_url="https://vendor.test",
        allowed_domains=["vendor.test"],
        inputs=[{"name": "customer_number"}, {"name": "address"}],
        secrets=[{"name": "username"}],
        setup_steps=[
            Step(id="u1", action="navigate", url="{{env.base_url}}/signin"),
            Step(
                id="u2",
                action="fill",
                locators=[Locator(strategy="label", text="Username")],
                value="{{secret.username}}",
            ),
        ],
        row_steps=[
            Step(
                id="s1",
                action="fill",
                locators=[Locator(strategy="role", role="textbox", name="Search")],
                value="ACC-{{input.customer_number}}",
            ),
            Step(
                id="s2",
                action="click",
                locators=[
                    Locator(
                        strategy="role",
                        role="button",
                        name="Edit",
                        within=Locator(strategy="role", role="row", has_text="Billing"),
                    ),
                    Locator(strategy="css", selector="tr.billing button"),
                ],
                intent="opens the billing row for editing",
            ),
            Step(
                id="s3",
                action="fill",
                locators=[Locator(strategy="role", role="textbox", name="Address")],
                value="{{input.address}}",
            ),
        ],
    )
    base.update(overrides)
    return UseCase(**base)


# --- it has to be Python ---------------------------------------------------


def test_the_exported_script_parses():
    """The whole deliverable in one assertion. A script that does not parse is
    worse than no export, because it looks like something that works."""
    ast.parse(as_playwright_python(use_case()))


def test_every_action_and_check_parses():
    """One of each, so a new action or assertion kind that renders to broken
    syntax fails here rather than in somebody's pipeline."""
    case = use_case(
        inputs=[],
        row_steps=[
            Step(id="a1", action="navigate", url="https://vendor.test/x"),
            Step(id="a2", action="click", locators=[Locator(strategy="role", role="button", name="Go")]),
            Step(id="a3", action="fill", locators=[Locator(strategy="role", role="textbox", name="N")], value="x"),
            Step(id="a4", action="select", locators=[Locator(strategy="role", role="combobox", name="C")], value="one"),
            Step(id="a5", action="hover", locators=[Locator(strategy="text", text="Menu")]),
            Step(id="a6", action="press", value="Enter"),
            Step(id="a7", action="wait", wait_for={"kind": "text", "value": "Saved"}),
            Step(id="a8", action="wait", wait_for={"kind": "load_state", "state": "networkidle"}),
            Step(id="a9", action="extract", locators=[Locator(strategy="role", role="status")], output="note"),
            Step(
                id="a10",
                action="extract_rows",
                locators=[Locator(strategy="role", role="row")],
                columns=[ExtractColumn(name="account", selector="td.acct")],
                output="rows",
            ),
            Step(id="a11", action="assert", **{"assert": Assertion(kind="url_contains", value="/done")}),
            Step(
                id="a12",
                action="assert",
                **{
                    "assert": Assertion(
                        kind="attribute_contains",
                        locator=Locator(strategy="role", role="link", name="Receipt"),
                        attribute="href",
                        value="/receipt/",
                    )
                },
            ),
            Step(
                id="a13",
                action="assert",
                **{"assert": Assertion(kind="element_count", locator=Locator(strategy="role", role="row"), count=3)},
            ),
        ]
    )

    ast.parse(as_playwright_python(case))


def test_a_name_with_quotes_in_it_does_not_break_the_file():
    """Real page text contains apostrophes and quotation marks, and a
    generator that assumes otherwise produces a file that will not parse."""
    case = use_case(
        inputs=[],
        row_steps=[
            Step(
                id="q1",
                action="click",
                locators=[Locator(strategy="role", role="button", name="Don't \"save\" this")],
            )
        ]
    )

    ast.parse(as_playwright_python(case))


# --- values -----------------------------------------------------------------


def test_a_row_value_becomes_a_column_read():
    assert value_expr("{{input.account}}") == 'row["account"]'


def test_a_value_wrapped_in_text_becomes_an_f_string():
    """The case a naive renderer gets wrong by producing a script that runs and
    types the template instead of the value."""
    assert value_expr("ACC-{{input.account}}") == "f'ACC-{row[\"account\"]}'"


def test_a_secret_is_read_from_the_environment_and_never_written_down():
    expr = value_expr("{{secret.password}}")

    assert expr == f'os.environ["{SECRET_PREFIX}PASSWORD"]'
    assert "password" not in expr.replace("PASSWORD", ""), "the name, never a value"


def test_the_recorded_origin_becomes_one_constant():
    """So the exported script can be pointed at UAT the same way the use case
    can, rather than carrying dev's hostname into somebody's pipeline."""
    assert value_expr("{{env.base_url}}/signin") == "f'{BASE_URL}/signin'"


# --- locators ---------------------------------------------------------------


def test_a_role_rung_becomes_get_by_role():
    spec = Locator(strategy="role", role="button", name="Save", exact=True)

    assert locator_expr(spec) == 'page.get_by_role("button", name="Save", exact=True)'


def test_a_scoped_rung_becomes_the_chain_it_is():
    spec = Locator(
        strategy="role",
        role="button",
        name="Edit",
        within=Locator(strategy="role", role="row", has_text="Acme"),
    )

    assert locator_expr(spec) == (
        'page.get_by_role("row").filter(has_text="Acme").get_by_role("button", name="Edit")'
    )


def test_a_rung_inside_a_frame_descends_first():
    """An element inside a frame is not on the page as far as any other call is
    concerned, so the frame has to come first or the script finds nothing."""
    spec = Locator(strategy="role", role="button", name="Pay", frames=["iframe#pay"])

    assert locator_expr(spec) == 'page.frame_locator("iframe#pay").get_by_role("button", name="Pay")'


def test_a_position_is_carried_through():
    spec = Locator(strategy="role", role="button", name="Edit", nth=2)

    assert locator_expr(spec).endswith(".nth(2)")


# --- what the export keeps, and what it admits losing -----------------------


def test_the_other_rungs_of_the_ladder_are_written_beside_the_step():
    """Not dropped. The engine would have tried them, and somebody debugging
    the script needs to know what it is not doing."""
    script = as_playwright_python(use_case())

    assert "# also recorded: css=tr.billing button" in script


def test_the_header_says_the_document_is_the_source_of_truth():
    """The one thing a reader must take away, because the failure mode of this
    feature is somebody editing the file and expecting TRACE to notice."""
    script = as_playwright_python(use_case())

    assert "one-way export" in script
    assert "use case document" in script.lower()
    assert "only the leading" in script, "and that it runs one rung per step"


def test_a_step_intent_becomes_the_comment_above_it():
    assert "# opens the billing row for editing" in as_playwright_python(use_case())


def test_the_phases_stay_separate():
    """Signing in once is the structural fact that makes a batch work, and an
    export that inlined everything would hide it."""
    script = as_playwright_python(use_case())

    assert "def setup(page):" in script
    assert "def do_row(page, row):" in script
    assert script.index("def setup") < script.index("def do_row")


def test_a_condition_becomes_an_if():
    case = use_case(
        inputs=[],
        row_steps=[
            Step(
                id="c1",
                action="click",
                locators=[Locator(strategy="role", role="button", name="Accept")],
                when=Assertion(
                    kind="element_visible",
                    locator=Locator(strategy="role", role="button", name="Accept"),
                ),
            )
        ]
    )
    script = as_playwright_python(case)

    assert 'if page.get_by_role("button", name="Accept").first.is_visible():' in script
    ast.parse(script)


def test_an_optional_step_becomes_a_try():
    case = use_case(
        inputs=[],
        row_steps=[
            Step(
                id="o1",
                action="click",
                locators=[Locator(strategy="role", role="button", name="Dismiss")],
                optional=True,
            )
        ]
    )
    script = as_playwright_python(case)

    assert "try:  # recorded as optional" in script
    ast.parse(script)


def test_a_script_step_is_omitted_unless_the_use_case_allows_scripts():
    """An export that quietly ran the JavaScript a use case is not permitted to
    run would be a way around the gate rather than a script."""
    # A draft, because publishing a use case with a script step in it is
    # already refused -- which is the gate this test is about not routing
    # around.
    case = use_case(
        status="draft",
        inputs=[],
        row_steps=[Step(id="j1", action="script", code="document.title = 'x'")],
    )

    assert "page.evaluate" not in as_playwright_python(case)
    assert "omitted" in as_playwright_python(case)

    permitted = as_playwright_python(use_case(**{
        "allow_scripts": True,
        "inputs": [],
        "row_steps": [Step(id="j1", action="script", code="document.title = 'x'")],
    }))
    assert "page.evaluate" in permitted


def test_a_recorded_expectation_is_noted_where_it_applies():
    """Only for a rung that does not match on text -- which is the only place
    the engine checks it, and printing it elsewhere would misdescribe what
    runs."""
    positional = use_case(
        inputs=[],
        row_steps=[
            Step(
                id="p1",
                action="click",
                locators=[Locator(strategy="css", selector="div > svg")],
                expect_text="Export",
            )
        ]
    )
    named = use_case(
        inputs=[],
        row_steps=[
            Step(
                id="p2",
                action="click",
                locators=[Locator(strategy="role", role="button", name="Export")],
                expect_text="Export",
            )
        ]
    )

    assert "recorded against text: 'Export'" in as_playwright_python(positional)
    assert "recorded against text" not in as_playwright_python(named)


# --- conditions as expressions ---------------------------------------------


def test_a_condition_is_counted_rather_than_awaited():
    """Matching the engine. Waiting for a banner that is not there would cost
    the timeout on every row of a batch."""
    expr = condition_expr(Assertion(kind="text_present", value="Saved"))

    assert expr == 'page.get_by_text("Saved").count() > 0'
    assert "expect(" not in expr


def test_a_negated_condition_is_wrapped_rather_than_inverted_by_hand():
    expr = condition_expr(Assertion(kind="url_contains", value="/signin", negate=True))

    assert expr == 'not ("/signin" in page.url)'


# --- naming ------------------------------------------------------------------


def test_the_filename_comes_from_the_use_case_name():
    assert suggested_filename(use_case()) == "update_billing_address.py"


def test_a_name_made_only_of_punctuation_still_produces_a_filename():
    assert suggested_filename(use_case(name="***")) == "use_case.py"


# --- the invariant ----------------------------------------------------------


def test_this_module_never_runs_what_it_writes():
    """Codegen output is data, never code -- in both directions. The parser
    side asserts this about itself; this is the same assertion for the
    generator."""
    import inspect

    import export

    source = inspect.getsource(export)
    for forbidden in ("exec(", "eval(", "subprocess", "importlib", "__import__"):
        assert forbidden not in source, f"export.py must not {forbidden}"
    # `re.compile` is used to render assertions and is not a way to run
    # anything; the bare builtin would be.
    assert "compile(" not in source.replace("re.compile(", "")
