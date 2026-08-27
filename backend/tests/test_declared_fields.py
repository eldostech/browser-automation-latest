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


def test_the_revealer_swaps_placeholders_only_at_dispatch():
    from runner import _revealer

    reveal = _revealer({"password": PASSWORD})
    arguments = {
        "fields": [
            {"name": "Password", "value": secret_placeholder("password")},
            {"name": "Email", "value": "nitin@example.com"},
        ]
    }

    revealed = reveal(arguments)

    assert revealed["fields"][0]["value"] == PASSWORD
    assert revealed["fields"][1]["value"] == "nitin@example.com"
    # The input is not mutated: the event recorded for this call must keep the
    # placeholder, which it cannot do if we rewrote the dict in place.
    assert arguments["fields"][0]["value"] == secret_placeholder("password")


def test_no_revealer_is_built_when_nothing_is_secret():
    """The common case pays nothing: no walk over every tool argument."""
    from runner import _revealer

    assert _revealer({}) is None


def test_an_unknown_placeholder_is_left_alone():
    """A model inventing «secret:admin» gets no credential, and no crash."""
    from runner import _revealer

    reveal = _revealer({"password": PASSWORD})
    assert reveal({"v": secret_placeholder("admin")})["v"] == secret_placeholder("admin")


# --- deterministic parameterisation ----------------------------------------


def test_declared_values_become_templates_by_lookup():
    from distill import apply_declarations, PreFilterResult
    from distill import RecordedStep

    step = RecordedStep(
        call_id="c1",
        step=1,
        seq=1,
        tool="browser_fill_form",
        action="fill_form",
        arguments={
            "fields": [
                {"name": "Full name", "value": "Nitin Asati"},
                {"name": "Email", "value": "nitin@example.com"},
                {"name": "Password", "value": secret_placeholder("password")},
            ]
        },
        fields=[
            {"name": "Full name", "value": "Nitin Asati", "locators": []},
            {"name": "Password", "value": secret_placeholder("password"), "locators": []},
        ],
    )
    pre = PreFilterResult(
        steps=[step],
        warnings=[],
        literals=["Nitin Asati", "Submit"],
        start_url=None,
        domains=["example.com"],
        stats={},
    )

    found = apply_declarations(pre, make_fields().persistable())

    values = [f["value"] for f in step.arguments["fields"]]
    assert values == [
        "{{input.full_name}}",
        "{{input.work_email}}",
        "{{secret.password}}",
    ]
    assert "{{input.full_name}}" in found
    # A declared value is no longer offered to the model as a loose literal to
    # think about -- it is already settled.
    assert "Nitin Asati" not in pre.literals
    assert "Submit" in pre.literals


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
