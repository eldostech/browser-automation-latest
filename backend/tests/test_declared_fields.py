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


# --- the declaration outranks the model ------------------------------------


def test_a_declared_template_survives_the_models_renaming():
    """The bug this guards: declared fields vanished from the finished use case.

    The model is asked to parameterise the recording, and it does -- including
    values that were *already* parameterised from the declaration, which it
    renames to something of its own. The use case then asked for
    `{{input.name}}` while the declared input was `full_name`, so the declared
    inputs looked unused and were dropped. A user filled in fields at record
    time and the finished use case never asked for them.
    """
    from distill import _prefer_declared

    # The model's override is refused where a declaration already stands.
    assert _prefer_declared("{{input.full_name}}", "{{input.name}}") == "{{input.full_name}}"
    assert _prefer_declared("{{secret.password}}", "{{input.pw}}") == "{{secret.password}}"

    # Where nothing was declared, the model is still free to parameterise.
    assert _prefer_declared("Nitin Asati", "{{input.name}}") == "{{input.name}}"
    assert _prefer_declared(None, "{{input.name}}") == "{{input.name}}"


# --- nothing here is specific to any one use case ---------------------------
#
# The fix was found by investigating one broken recording, which is exactly the
# circumstance in which a general mechanism quietly acquires a special case.
# These use field names, values and shapes with nothing in common with that
# recording, including several chosen to break a naive implementation.


ARBITRARY_FIELDS = [
    # Ordinary.
    ("customer_reference", "ACME-99271"),
    # Value containing regex metacharacters: a substitution built on re.sub
    # with an unescaped pattern would raise or mangle this.
    ("search_query", "price (USD) [2024] *special* +tax?"),
    # Value that looks like a template. A second substitution pass would try to
    # resolve it and fail.
    ("literal_template", "{{input.not_a_real_field}}"),
    # Backslashes and quotes, which have to survive both JSON encoding and
    # whatever the page does with them.
    ("windows_path", r"C:\Users\o'brien\file.txt"),
    # Non-ASCII, including a character JavaScript treats as a line terminator.
    ("unicode_name", "Zoë Ödegård\u2028"),
    # A name at the length limit of what a column header can be.
    ("a_very_long_field_name_that_is_still_a_valid_identifier", "x" * 200),
]


@pytest.mark.parametrize("name,value", ARBITRARY_FIELDS, ids=[n for n, _ in ARBITRARY_FIELDS])
def test_any_field_name_and_value_round_trips(name: str, value: str):
    """Declare -> substitute -> render, for values chosen to be awkward."""
    from fields import parameterise, substitution_map
    from usecase import render_code

    fields = FieldSet.from_payload([{"name": name, "value": value}])
    stored = fields.persistable()
    mapping = substitution_map(stored)

    # The recorded argument, as the browser tool received it.
    recorded = {"fields": [{"name": "Some label", "value": value}]}
    parameterised = parameterise(recorded, mapping)
    assert parameterised["fields"][0]["value"] == f"{{{{input.{name}}}}}"

    # And back again at replay, with a *different* value than was recorded.
    code = f"await page.fill('#x', '{{{{input.{name}}}}}');"
    rendered = render_code(code, inputs={name: "REPLACED"}, secrets={})
    assert '"REPLACED"' in rendered
    assert "{{" not in rendered


def test_a_value_containing_a_template_is_not_re_expanded():
    """Substitution runs once. A value that looks like a template stays data."""
    from usecase import render_code

    rendered = render_code(
        "f('{{input.a}}')", inputs={"a": "{{input.b}}", "b": "SHOULD NOT APPEAR"}, secrets={}
    )
    assert "SHOULD NOT APPEAR" not in rendered
    assert "{{input.b}}" in rendered


def test_arbitrary_numbers_of_fields_are_all_declared():
    """Nothing assumes four fields, or any particular count."""
    from distill import _merge_input_specs

    for count in (0, 1, 7, 40):
        declared = {"inputs": {f"field_{i}": f"value {i}" for i in range(count)}}
        specs = _merge_input_specs([], declared)
        assert {s.name for s in specs} == set(declared["inputs"])


def test_secret_slots_are_equally_generic():
    from fields import secret_placeholder
    from runner import _revealer

    slots = {"api_token": "tok-1", "db_password": "p@ss", "otp_seed": "ABCDEF"}
    reveal = _revealer(slots)
    template = {k: secret_placeholder(k) for k in slots}

    assert reveal(template) == slots


def test_a_referenced_input_is_declared_whatever_it_is_called():
    """The auto-declaration walks what the steps reference, not a fixed list."""
    from distill import InputSpec, _merge_input_specs
    from usecase import Step

    step = Step(
        id="s1",
        action="script",
        code="page.fill('#a', '{{input.zzz_unusual_name}}'); page.fill('#b', '{{input.q}}');",
    )
    referenced = {name for kind, name in step.references() if kind == "input"}
    assert referenced == {"zzz_unusual_name", "q"}

    specs = _merge_input_specs([], None)
    known = {s.name for s in specs}
    for name in sorted(referenced - known):
        specs.append(InputSpec(name=name, required=True))
    assert {s.name for s in specs} == referenced


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
