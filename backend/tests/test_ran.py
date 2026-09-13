"""The Playwright expression the server ran, kept instead of discarded.

This is the fix for a recording that worked perfectly and then failed twice on
replay, in the one way nothing here was watching for.

A profile picker. The agent clicked a ref; the accessibility tree said

    role=radio name="Nayra Asati"

and the server reported having run

    page.locator('label').filter({ hasText: 'Nayra Asati' }).click()

because the site draws a styled radio whose input is not clickable and whose
label is. The recording kept the first and threw away the second. On replay the
radio resolved -- there is exactly one -- and `click` waited thirty seconds and
gave up. Nothing was wrong with finding the element. A repair then offered the
same locator with `exact` flipped, because the error said "timeout" and named a
locator, and the person was told to try again.

Two things come out of that, and both are tested here: the server's own
expression belongs in the ladder, and an action that cannot be performed is a
reason to try the next rung rather than to fail the step.
"""

from __future__ import annotations

import pytest

from agent.ran import locator_from, ran_code

NAYRA = """### Ran Playwright code
```js
await page.locator('label').filter({ hasText: 'Nayra Asati' }).click();
```
### Page
- Page URL: https://www.ixl.com/signin
### Snapshot
- [Snapshot](.playwright-mcp/page.yml)
"""


# --- reading the reply ------------------------------------------------------


def test_the_code_is_taken_out_of_the_servers_reply():
    assert ran_code(NAYRA) == (
        "await page.locator('label').filter({ hasText: 'Nayra Asati' }).click();"
    )


def test_a_reply_with_no_code_in_it_yields_nothing():
    assert ran_code("### Page\n- Page URL: https://x.test") == ""
    assert ran_code("") == ""


# --- the two expressions that broke a real recording -----------------------


def test_the_label_click_that_the_tree_could_not_describe():
    """The whole reason this module exists."""
    assert locator_from(ran_code(NAYRA)).describe() == "css=label has_text='Nayra Asati'"


def test_the_secret_word_field_the_tree_called_ambiguous():
    """Same session: the tree-derived rung matched three elements and the mark
    was refused, while the server had used a selector that matches one."""
    code = "await page.locator('input[name=\"secretWord\"]').fill('x');"

    assert locator_from(code).describe() == 'css=input[name="secretWord"]'


# --- the vocabulary ---------------------------------------------------------


@pytest.mark.parametrize(
    "code, described",
    [
        (
            "await page.getByRole('button', { name: 'Sign in', exact: true }).click();",
            'role=button name="Sign in" exact',
        ),
        ("await page.getByRole('button').nth(3).click();", "role=button [3]"),
        ("await page.getByText('Continue').first().click();", "text='Continue'"),
        ("await page.getByLabel('Username').fill('ada');", "label='Username'"),
        ("await page.getByPlaceholder('Search').fill('x');", "placeholder='Search'"),
        ("await page.getByTestId('save-btn').click();", "test_id='save-btn'"),
        (
            "await page.getByRole('figure', { name: 'Row 1' })"
            ".getByRole('textbox', { name: 'answer' }).fill('7');",
            'role=textbox name="answer" in role=figure name="Row 1"',
        ),
        (
            "await page.frameLocator('#pay').getByRole('button', { name: 'Pay' }).click();",
            'role=button name="Pay" in frame #pay',
        ),
        (
            "await page.locator('#pay').contentFrame()"
            ".getByRole('textbox', { name: 'Card' }).fill('1');",
            'role=textbox name="Card" in frame #pay',
        ),
    ],
)
def test_the_forms_the_server_actually_emits(code, described):
    assert locator_from(code).describe() == described


@pytest.mark.parametrize(
    "code",
    [
        "await page.goto('https://x.test');",
        "await page.keyboard.press('Enter');",
        "await page.waitForTimeout(500);",
        "",
        "// nothing here",
    ],
)
def test_a_statement_that_names_no_element_yields_nothing(code):
    assert locator_from(code) is None


def test_a_chain_with_something_unmodelled_in_it_yields_nothing():
    """Half a chain is worse than none: it would produce a rung that looks
    reviewed and finds something else."""
    code = "await page.getByRole('button').scrollIntoViewIfNeeded().click();"

    assert locator_from(code) is None


def test_a_value_with_brackets_and_quotes_in_it_survives():
    """Page text contains both, and a regex that stops at the first bracket
    would truncate the name it was reading."""
    code = "await page.getByRole('button', { name: 'Save (draft)' }).click();"

    assert locator_from(code).describe() == 'role=button name="Save (draft)"'


def test_first_records_no_position_the_way_codegen_does():
    """0 is this schema's "no position given". A resolver that acts on
    whichever element comes first is what the ladder exists to refuse, so
    `.first()` must not become a claim about ordering."""
    assert locator_from("await page.getByText('Edit').first().click();").nth == 0
