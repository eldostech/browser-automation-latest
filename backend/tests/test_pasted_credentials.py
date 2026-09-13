"""A password typed into a task, which no redactor could have caught.

The redactor protects values this system was *told* about. This is the other
half: the value nobody declared. A real task arrived as

    Login with below credentials
    User: nitin
    Password : HappyLearning@123

and that string became the run row, the `run_started` event, the audit entry,
the use case's own description, and the first message sent to a model. None of
those knew it was a password, so none of them could hide it.

So it is recognised at the boundary, before the first write. The useful part is
what it is replaced *with*: when a stored credential is bound, the value
becomes `{{secret.slot}}`, which is exactly the literal the recorder is already
instructed to type and the tool layer substitutes at the moment of typing. The
task keeps working, the value is never written down, and the same task runs for
a different account tomorrow.
"""

from __future__ import annotations

import pytest

from redaction import PLACEHOLDER, find_credentials, scrub_credentials

TASK = """Go to this website

Login with below credentials
User: Nitinasati
Password : HappyLearning@123

Select Nayra Asati and open the assignment."""


# --- finding it -------------------------------------------------------------


def test_the_real_task_that_prompted_this_is_recognised():
    found = {item.label.casefold(): item.value for item in find_credentials(TASK)}

    assert found["password"] == "HappyLearning@123"
    assert found["user"] == "Nitinasati"


def test_only_the_password_half_counts_as_a_secret():
    """An account name in a task is ordinary and often necessary. A password is
    not, and the two get different treatment: one is rewritten when there is
    somewhere to rewrite it to, the other refuses the request."""
    scrubbed = scrub_credentials(TASK)

    assert scrubbed.labels == ["password"]
    assert len(scrubbed.found) == 2


@pytest.mark.parametrize(
    "text, value",
    [
        ("api key = sk-live-abc123", "sk-live-abc123"),
        ("Token: eyJhbGciOi.Jz", "eyJhbGciOi.Jz"),
        ('password: "quoted-value"', "quoted-value"),
        ("PWD:shouty-Value1", "shouty-Value1"),
        ("otp = 483920", "483920"),
    ],
)
def test_the_shapes_people_actually_paste(text, value):
    assert [item.value for item in find_credentials(text)] == [value]


@pytest.mark.parametrize(
    "text",
    [
        "The password field is on the right",
        "Click the login button, then wait",
        "Enter your email address in the second box",
        "the account summary shows a balance",
    ],
)
def test_ordinary_prose_about_a_login_is_left_alone(text):
    """The separator is what keeps this from firing on half the tasks anybody
    writes. A sentence *about* a password is not a password."""
    assert find_credentials(text) == []
    assert scrub_credentials(text).text == text


# --- replacing it -----------------------------------------------------------


def test_a_bound_credential_turns_the_value_into_the_slot_that_holds_it():
    """The case that keeps the task working. `{{secret.password}}` is the
    literal the recorder is told to type, and the tool layer substitutes it at
    the moment of typing."""
    scrubbed = scrub_credentials(TASK, slots=["username", "password"])

    assert "{{secret.password}}" in scrubbed.text
    assert "{{secret.username}}" in scrubbed.text
    assert "HappyLearning@123" not in scrubbed.text
    assert "Nitinasati" not in scrubbed.text
    assert "Select Nayra Asati" in scrubbed.text, "the rest of the task is untouched"


def test_a_slot_is_matched_by_name_and_never_guessed():
    """A slot called `login_password` is plainly the one a "Password:" label
    means. A slot called `api_base` is plainly not."""
    with_related = scrub_credentials("Password: hunter2", slots=["login_password"])
    with_unrelated = scrub_credentials("Password: hunter2", slots=["api_base"])

    assert "{{secret.login_password}}" in with_related.text
    assert PLACEHOLDER in with_unrelated.text
    assert "hunter2" not in with_unrelated.text


def test_with_nothing_bound_the_value_is_still_removed():
    """Whether or not there is anywhere to point the task at, the value does
    not get written down."""
    scrubbed = scrub_credentials(TASK)

    assert "HappyLearning@123" not in scrubbed.text
    assert PLACEHOLDER in scrubbed.text


def test_the_same_value_is_removed_everywhere_it_appears():
    """People repeat a password in a task -- once in the instructions and once
    in a note to themselves -- and removing the first occurrence only would
    look like it worked."""
    text = "Password: hunter2. If hunter2 does not work, stop."

    assert "hunter2" not in scrub_credentials(text).text


def test_a_task_with_no_credentials_is_returned_unchanged():
    """The common path, and it must allocate no surprises: the text that comes
    back is the text that went in."""
    text = "Open the customer list and read the balance for each row."

    assert scrub_credentials(text).text == text
    assert scrub_credentials(text).found == []
