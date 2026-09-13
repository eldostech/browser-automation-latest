"""Declared fields: parameterisation without guessing, credentials without leaks.

Two guarantees are asserted here, and they are the point of the feature:

* a credential the user typed reaches the *browser* and nothing else -- not the
  prompt, not the model's message history, not an event, not the database;
* a value the user named becomes a use case input under that name, by lookup
  rather than by the model's judgement.
"""

from __future__ import annotations

import json

import pytest

from fields import FieldSet, secret_placeholder
from redaction import Redactor


PASSWORD = "s3cret-Example-Pw!"


def make_fields(**overrides):
    payload = [
        {"name": "full_name", "value": "Nitin Asati"},
        {"name": "work_email", "value": "nitin@example.com"},
        {"name": "password", "value": PASSWORD, "secret": True},
    ]
    payload = overrides.get("payload", payload)
    return FieldSet.from_payload(payload)


# --- the shape of a declaration --------------------------------------------


def test_inputs_and_secrets_are_separated():
    fields = make_fields()
    assert [f.name for f in fields.inputs] == ["full_name", "work_email"]
    assert [f.name for f in fields.secrets] == ["password"]
    assert fields.secret_values == {"password": PASSWORD}


def test_a_field_name_must_work_as_a_column_header():
    """It becomes a CSV header and a `{{input.x}}` template, so it is an identifier."""
    with pytest.raises(ValueError, match="not a usable field name"):
        FieldSet.from_payload([{"name": "full name", "value": "x"}])
    with pytest.raises(ValueError, match="not a usable field name"):
        FieldSet.from_payload([{"name": "2nd_choice", "value": "x"}])


def test_a_duplicate_name_is_refused():
    with pytest.raises(ValueError, match="declared twice"):
        FieldSet.from_payload(
            [{"name": "email", "value": "a@b.c"}, {"name": "email", "value": "d@e.f"}]
        )


# --- the credential guarantee ----------------------------------------------


def test_the_prompt_never_contains_a_credential():
    """The model is told the placeholder, never the value.

    This is the load-bearing one. Redaction can clean an event on its way to
    storage, but the message history is replayed to the model on every turn --
    a secret that enters the prompt is a secret the model keeps seeing, and no
    later pass can take it back out of a conversation that already happened.
    """
    block = make_fields().prompt_block()

    assert PASSWORD not in block
    assert secret_placeholder("password") in block
    # Non-secret values are supposed to be there: they are the data.
    assert "Nitin Asati" in block


def test_what_is_persisted_carries_no_secret_value():
    stored = make_fields().persistable()

    assert stored["inputs"] == {
        "full_name": "Nitin Asati",
        "work_email": "nitin@example.com",
    }
    assert stored["secret_slots"] == ["password"]
    assert PASSWORD not in json.dumps(stored)


def test_a_short_value_is_not_substituted_by_search():
    """"1" would rewrite every digit in the recording."""
    from fields import substitution_map

    mapping = substitution_map({"inputs": {"quantity": "1", "city": "Manchester"}})
    assert "Manchester" in mapping
    assert "1" not in mapping


def test_longer_values_are_substituted_first():
    """A value containing another must win, or the shorter one leaves a
    broken template embedded in the longer one's replacement."""
    from fields import parameterise, substitution_map

    mapping = substitution_map(
        {"inputs": {"short": "Acme", "long": "Acme Corporation"}}
    )
    assert parameterise("Acme Corporation", mapping) == "{{input.long}}"


# --- redaction is still the second line of defence -------------------------


def test_a_page_echoing_the_password_back_is_still_redacted():
    """The model never sees the credential, but the *page* might repeat it.

    A site that renders "signed in as s3cret-..." would put the value into a
    tool result. The revealer cannot help there -- the value is arriving, not
    leaving -- so the redactor is still registered with the real values.
    """
    redactor = Redactor(make_fields().secret_values.values())
    assert PASSWORD not in redactor.text(f"Welcome, your password {PASSWORD} was accepted")


# --- the declaration outranks the model ------------------------------------


def test_a_value_containing_a_template_is_not_re_expanded():
    """Substitution runs once. A value that looks like a template stays data."""
    from usecase import render_code

    rendered = render_code(
        "f('{{input.a}}')", inputs={"a": "{{input.b}}", "b": "SHOULD NOT APPEAR"}, secrets={}
    )
    assert "SHOULD NOT APPEAR" not in rendered
    assert "{{input.b}}" in rendered


def test_a_value_is_never_expanded_as_a_template_itself():
    """A row of batch input must not be able to read a secret.

    Substitution replaces a template with a value. If the *result* is then
    scanned again, a value whose text happens to be `{{secret.password}}` gets
    expanded -- and a spreadsheet cell becomes a way to render any credential
    bound to the run into a visible field.

    `re.sub` does not rescan what it inserted, so one pass is safe and two are
    not. An earlier render_code ran a pass for quoted templates and another for
    bare ones, and leaked exactly this way.
    """
    from usecase import render_code, render_template

    secrets = {"password": "REAL-PASSWORD"}

    # Script steps.
    rendered = render_code(
        "f('{{input.a}}')", inputs={"a": "{{secret.password}}"}, secrets=secrets
    )
    assert "REAL-PASSWORD" not in rendered
    assert "{{secret.password}}" in rendered

    # Ordinary form fields, which take a different path.
    value = render_template(
        "{{input.a}}", inputs={"a": "{{secret.password}}"}, secrets=secrets
    )
    assert value == "{{secret.password}}"


def test_both_template_spellings_are_handled_in_one_pass():
    """Quoted and bare forms, together, without a second scan."""
    from usecase import render_code

    rendered = render_code(
        "page.fill('#a', '{{input.x}}'); const n = {{input.y}};",
        inputs={"x": "one", "y": "two"},
        secrets={},
    )
    assert rendered == 'page.fill(\'#a\', "one"); const n = "two";'
