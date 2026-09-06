"""Parsing what ``playwright codegen`` writes.

Every script here is the shape codegen actually emits for
``--target=python-async``, because that -- not the Playwright API in general --
is the input this has to handle. The parser is a golden-file test in spirit: a
Playwright upgrade that changes the output should fail here loudly rather than
produce steps that quietly do the wrong thing.
"""

from __future__ import annotations

import pytest

from codegen import CodegenError, parse, summarise, urls_of


def script(body: str) -> str:
    """A codegen script with the launcher boilerplate codegen really writes."""
    indented = "\n".join(f"    {line}" for line in body.strip().splitlines())
    return f'''import asyncio
from playwright.async_api import Playwright, async_playwright, expect


async def run(playwright: Playwright) -> None:
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()
{indented}

    # ---------------------
    await context.close()
    await browser.close()


async def main() -> None:
    async with async_playwright() as playwright:
        await run(playwright)


asyncio.run(main())
'''


# --- the ordinary recording ------------------------------------------------


def test_a_sign_in_and_a_form_becomes_steps():
    recording = parse(
        script(
            """
await page.goto("https://example.com/login")
await page.get_by_label("Username").fill("nitin")
await page.get_by_label("Password").fill("s3cret")
await page.get_by_role("button", name="Sign in").click()
await page.get_by_placeholder("Order reference").fill("A-1024")
await page.get_by_role("button", name="Submit").click()
"""
        )
    )

    assert [step.action for step in recording.steps] == [
        "navigate",
        "fill",
        "fill",
        "click",
        "fill",
        "click",
    ]
    assert recording.steps[0].url == "https://example.com/login"
    assert recording.steps[1].locators[0].strategy == "label"
    assert recording.steps[1].locators[0].text == "Username"
    assert recording.steps[4].locators[0].strategy == "placeholder"


def test_the_launcher_boilerplate_is_not_a_step():
    """`launch`, `new_context`, `new_page`, `close` are the harness, not the
    workflow. Recording them would reopen a browser inside a browser."""
    recording = parse(script('await page.goto("https://example.com/")'))
    assert len(recording.steps) == 1


def test_every_typed_value_is_collected_for_parameterising():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await page.get_by_label("Email").fill("a@b.com")
await page.get_by_label("Quantity").fill("7")
"""
        )
    )
    assert recording.typed == ["a@b.com", "7"]


def test_visited_urls_become_the_allowlist():
    recording = parse(
        script(
            """
await page.goto("https://shop.example.com/one")
await page.goto("https://Shop.Example.com/two")
await page.goto("https://other.test/x")
"""
        )
    )
    assert urls_of(recording) == ["shop.example.com", "other.test"]
    assert recording.start_url == "https://shop.example.com/one"


# --- locators --------------------------------------------------------------


def test_a_role_locator_keeps_its_accessible_name():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await page.get_by_role("textbox", name="Search").click()
"""
        )
    )
    locator = recording.steps[1].locators[0]
    assert (locator.strategy, locator.role, locator.name) == ("role", "textbox", "Search")


def test_a_named_role_also_records_a_text_rung():
    """Free robustness: a redesign that turns a link into a button keeps the
    wording, and the second rung still finds it.

    Text rather than label, because each rung is executed as the Playwright
    call it names and `get_by_label` only matches form controls -- a `label`
    rung for a button could never match.
    """
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await page.get_by_role("link", name="Continue").click()
"""
        )
    )
    ladder = recording.steps[1].locators
    assert [rung.strategy for rung in ladder] == ["role", "text"]
    assert ladder[1].text == "Continue"


def test_nth_is_carried_onto_the_locator():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await page.get_by_role("button", name="Edit").nth(2).click()
"""
        )
    )
    assert recording.steps[1].locators[0].nth == 2


def test_a_css_locator_is_recorded_as_one():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await page.locator("#submit").click()
"""
        )
    )
    locator = recording.steps[1].locators[0]
    assert (locator.strategy, locator.selector) == ("css", "#submit")


def test_a_test_id_locator_is_recorded():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await page.get_by_test_id("checkout").click()
"""
        )
    )
    assert recording.steps[1].locators[0].strategy == "test_id"


# --- what it refuses to guess about ----------------------------------------


def test_a_chained_locator_is_reported_rather_than_approximated():
    """One locator scoped inside another has no rung in this schema, and an
    approximation clicks something else on row one."""
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await page.get_by_role("row", name="Ada").get_by_role("button", name="Edit").click()
"""
        )
    )
    assert len(recording.steps) == 1
    assert "cannot record" in recording.unsupported[0].reason


def test_last_is_recorded_as_a_position_never_as_a_fixed_index():
    """``.last`` used to be refused, and the reason it gave still holds.

    The objection was that "the last one" cannot be held as an index, and that
    guessing one from a recording is how a batch clicks the wrong row. That is
    right, and nothing here guesses: -1 means "whichever is last when this
    runs", and the engine performs it as Playwright's own ``.last``. What
    changed is that refusing the step threw away what the person recorded in
    order to avoid a mistake nobody was making.
    """
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await page.get_by_role("button", name="Remove").last.click()
"""
        )
    )

    assert not recording.unsupported, recording.unsupported
    click = recording.steps[-1]
    assert click.locators[0].nth == -1
    assert all(
        locator.nth == -1 for locator in click.locators
    ), "every rung of the ladder addresses the same one"


def test_a_frame_is_reported():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await page.frame_locator("#payment").get_by_label("Card number").fill("4111")
"""
        )
    )
    assert len(recording.steps) == 1
    assert recording.unsupported


def test_a_file_upload_says_why_it_cannot_be_replayed():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await page.get_by_label("Attachment").set_input_files("invoice.pdf")
"""
        )
    )
    assert len(recording.steps) == 1
    assert "on the machine that recorded it" in recording.unsupported[0].reason


def test_an_unknown_page_call_is_reported_not_ignored():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await page.emulate_media(media="print")
"""
        )
    )
    assert "emulate_media" in recording.unsupported[0].reason


def test_ignorable_page_calls_are_neither_steps_nor_complaints():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await page.wait_for_load_state("networkidle")
await page.wait_for_timeout(500)
"""
        )
    )
    assert len(recording.steps) == 1
    assert recording.unsupported == []


# --- assertions ------------------------------------------------------------


def test_a_visibility_assertion_is_recorded():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await expect(page.get_by_text("Thank you")).to_be_visible()
"""
        )
    )
    [check] = recording.assertions
    assert check.kind == "element_visible"
    assert check.negate is False


def test_a_hidden_assertion_is_the_same_check_negated():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await expect(page.get_by_text("Error")).to_be_hidden()
"""
        )
    )
    assert recording.assertions[0].negate is True


def test_a_url_assertion_is_recorded():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await expect(page).to_have_url("https://example.com/done")
"""
        )
    )
    [check] = recording.assertions
    assert (check.kind, check.value) == ("url_contains", "https://example.com/done")


def test_a_text_assertion_carries_the_expected_text():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await expect(page.get_by_role("heading")).to_have_text("Order complete")
"""
        )
    )
    [check] = recording.assertions
    assert (check.kind, check.value) == ("text_present", "Order complete")


def test_an_unsupported_expectation_is_reported():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await expect(page.get_by_role("checkbox")).to_be_checked()
"""
        )
    )
    assert "to_be_checked" in recording.unsupported[0].reason


# --- refusing to parse at all ----------------------------------------------


def test_a_script_that_is_not_python_is_refused():
    with pytest.raises(CodegenError, match="not valid Python"):
        parse("await page.goto(")


def test_a_script_with_no_run_function_is_refused():
    with pytest.raises(CodegenError, match="no run function"):
        parse("print('hello')\n")


def test_a_recording_with_no_actions_says_so():
    """Closing the window without doing anything is a mistake worth naming,
    rather than an empty use case that runs a thousand times doing nothing."""
    with pytest.raises(CodegenError, match="no actions were recorded"):
        parse(script("pass"))


def test_the_script_is_never_executed():
    """The file comes from a subprocess driving a page the user was shown.

    Executing it would turn anything that can influence codegen's output into
    code execution here, so the parser must be indifferent to what the script
    would do if run.
    """
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
import os; os.environ["PWNED"] = "1"
"""
        )
    )
    import os

    assert "PWNED" not in os.environ
    assert len(recording.steps) == 1


def test_summarise_says_what_was_found():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await expect(page.get_by_text("Hi")).to_be_visible()
await page.emulate_media(media="print")
"""
        )
    )
    assert summarise(recording) == "1 step(s), 1 check(s), 1 line(s) not understood"


def test_a_recorded_keypress_keeps_its_key():
    """The regression that lost a whole recording.

    ``Step`` checks an action has what it needs as it is constructed, so the
    parser has to read the key *before* building the step. It used to build
    first and assign after, which meant every recorded keypress raised
    "'press' requires a key" from a line that had the key right there.
    """
    recording = parse(
        script(
            """
await page.goto("https://example.com/form")
await page.get_by_role("textbox", name="Full name").press("Tab")
"""
        )
    )

    pressed = [step for step in recording.steps if step.action == "press"]
    assert len(pressed) == 1, recording.unsupported
    assert pressed[0].value == "Tab"
    assert pressed[0].locators


def test_a_line_the_schema_rejects_costs_only_that_line():
    """A recording is minutes of someone's time; one bad line must not void it."""
    recording = parse(
        script(
            """
await page.goto("https://example.com/form")
await page.get_by_role("textbox", name="Full name").fill("Ada")
await page.get_by_role("textbox", name="Full name").press("")
await page.get_by_role("button", name="Submit").click()
"""
        )
    )

    assert [step.action for step in recording.steps] == ["navigate", "fill", "click"]
    assert len(recording.unsupported) == 1
    assert "press" in recording.unsupported[0].reason


# --- how codegen addresses one of several matches --------------------------


def test_first_is_read_rather_than_dropping_the_step():
    """The bug that silently emptied real recordings.

    ``.first`` is a property, not a call, so it reaches the AST as an attribute
    wrapping the chain. The walker followed calls only, so every step written
    this way was refused -- and codegen writes ``.first`` whenever a locator
    matched more than one element, which on a real page is most of them. These
    four lines are taken verbatim from recordings that lost them.
    """
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await page.get_by_role("combobox").first.select_option("d8ec9327-b8e3")
await page.get_by_role("button", name="Chat").first.click()
await page.locator("tr:nth-child(14) > td:nth-child(3)").first.click()
"""
        )
    )

    assert not recording.unsupported, recording.unsupported
    assert [s.action for s in recording.steps] == ["navigate", "select", "click", "click"]

    chosen = next(s for s in recording.steps if s.action == "select")
    assert chosen.value == "d8ec9327-b8e3"
    assert chosen.locators[0].role == "combobox"
    assert chosen.locators[0].nth == 0, "'.first' is the zeroth match"
    assert "d8ec9327-b8e3" in recording.typed, "a chosen option is a value you can parameterise"


def test_one_locator_scoped_inside_another_is_still_refused():
    """The shape the schema genuinely has no rung for, kept honest."""
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await page.locator("#profileDropdown").get_by_text("Log out").click()
"""
        )
    )

    assert [s.action for s in recording.steps] == ["navigate"]
    assert len(recording.unsupported) == 1
    assert "chained" in recording.unsupported[0].reason


def test_selecting_several_options_at_once_says_what_it_could_not_take():
    recording = parse(
        script(
            """
await page.goto("https://example.com/")
await page.get_by_label("Tags").select_option(["a", "b"])
"""
        )
    )

    assert len(recording.unsupported) == 1
    assert "more than one option" in recording.unsupported[0].reason


# --- naming a value after the control it went into -------------------------


def test_a_value_carries_the_control_it_was_typed_into():
    """Naming a field from its value is guesswork; the control is what was seen.

    "test" and a UUID say nothing about what they are. The accessible name
    codegen wrote into the locator is what the person was looking at.
    """
    recording = parse(
        script(
            """
await page.goto("https://example.com/new")
await page.get_by_role("textbox", name="Brief description").fill("test")
await page.get_by_label("Project name").fill("test")
await page.get_by_placeholder("Order reference").fill("A-1024")
"""
        )
    )

    assert [(v.label, v.action) for v in recording.values] == [
        ("Brief description", "fill"),
        ("Project name", "fill"),
        ("Order reference", "fill"),
    ]
    assert recording.typed == ["test", "test", "A-1024"]


def test_a_control_with_no_accessible_name_offers_no_label():
    """The case that has nothing to give, admitted rather than guessed.

    A custom dropdown records as ``get_by_role("combobox")`` with nothing
    naming it, and the option it stores is the value the page uses -- a UUID on
    a real application. Neither is human-readable and neither can be made so
    from the recording alone.
    """
    recording = parse(
        script(
            """
await page.goto("https://example.com/new")
await page.get_by_role("combobox").first.select_option("e45fde3f-552b-4d8c")
"""
        )
    )

    chosen = recording.values[0]
    assert chosen.label == ""
    assert chosen.action == "select"
    assert chosen.value == "e45fde3f-552b-4d8c"


# --- a name that contains another name -------------------------------------


def test_exact_is_carried_off_the_recorded_locator():
    """`exact=True` is codegen telling two controls apart, and it was dropped.

    A page with a "+ Invite User" button and, in the dialog it opens, an
    "Invite" button. Playwright matches an accessible name as a
    case-insensitive substring, so the dialog's button can only be named
    unambiguously with ``exact=True`` -- which codegen writes, and which the
    parser discarded. The recorded locator then meant "either of these", and a
    replay took whichever came first.
    """
    recording = parse(
        script(
            """
await page.get_by_role("button", name="+ Invite User").click()
await page.get_by_role("button", name="Invite", exact=True).click()
"""
        )
    )

    opener, submit = recording.steps
    assert opener.locators[0].name == "+ Invite User"
    assert opener.locators[0].exact is False
    assert submit.locators[0].name == "Invite"
    assert submit.locators[0].exact is True, "the whole point of the recorded line"


def test_an_exact_rung_does_not_get_a_loose_text_fallback():
    """The fallback must not find the control the exact rung exists to avoid."""
    recording = parse(
        script('await page.get_by_role("button", name="Invite", exact=True).click()')
    )

    assert [(loc.strategy, loc.exact) for loc in recording.steps[0].locators] == [
        ("role", True),
        ("text", True),
    ]


def test_exact_is_read_off_the_other_get_by_calls_too():
    """Every ``get_by_*`` that matches by name takes it, and codegen writes it."""
    recording = parse(
        script(
            """
await page.get_by_label("Name", exact=True).fill("Ada")
await page.get_by_text("Total", exact=True).click()
"""
        )
    )

    assert [loc.exact for step in recording.steps for loc in step.locators] == [True, True]


def test_exact_survives_a_round_trip_through_the_document():
    """It is stored, or replay reads back the locator that was already wrong."""
    from usecase import Locator

    stored = Locator(strategy="role", role="button", name="Invite", exact=True)
    assert Locator.model_validate(stored.model_dump()).exact is True
    assert "exact" in stored.describe()
