"""Tests for the secret-redaction pass.

The bar these set: a password that reached an event by *any* route must not
survive into the store or onto the WebSocket. The paths tested here are the
real ones observed in ``data/runs.db`` -- a pasted task, a ``fill_form``
argument, a tool result echoing the value, and the model's own prose.
"""

from __future__ import annotations

import pytest

from events import ErrorEvent, RunStarted, Thinking, ToolCall, ToolResult, dump_event
from redaction import MIN_SECRET_LENGTH, NULL_REDACTOR, PLACEHOLDER, Redactor

PASSWORD = "s3cret-Example-Pw!"


@pytest.fixture
def redactor() -> Redactor:
    return Redactor([PASSWORD])


# --- registration ----------------------------------------------------------


def test_short_values_are_ignored():
    r = Redactor(["ab"])
    assert not r.active
    assert r.text("ab cd ab") == "ab cd ab"


def test_blank_and_non_string_values_are_ignored():
    r = Redactor()
    r.update([None, "", "   ".strip()])
    assert len(r) == 0


def test_duplicates_are_registered_once(redactor: Redactor):
    redactor.add(PASSWORD)
    assert len(redactor) == 1


def test_minimum_length_is_the_documented_boundary():
    assert Redactor(["x" * (MIN_SECRET_LENGTH - 1)]).active is False
    assert Redactor(["x" * MIN_SECRET_LENGTH]).active is True


# --- text ------------------------------------------------------------------


def test_a_secret_is_replaced_everywhere_it_appears(redactor: Redactor):
    text = f"typed {PASSWORD} then retyped {PASSWORD}"
    result = redactor.text(text)
    assert PASSWORD not in result
    assert result.count(PLACEHOLDER) == 2


def test_overlapping_secrets_do_not_leave_a_readable_tail():
    # "secret" is contained in "secretvalue123". Redacting the short one first
    # would leave "value123" exposed.
    r = Redactor(["secret", "secretvalue123"])
    assert r.text("here is secretvalue123 ok") == f"here is {PLACEHOLDER} ok"


def test_non_secret_text_is_untouched(redactor: Redactor):
    assert redactor.text("nothing to see") == "nothing to see"


def test_null_redactor_is_a_pass_through():
    assert NULL_REDACTOR.active is False
    assert NULL_REDACTOR.text(PASSWORD) == PASSWORD


# --- structures ------------------------------------------------------------


def test_nested_structures_are_redacted(redactor: Redactor):
    payload = {
        "fields": [
            {"name": "Username", "value": "Nitinasati"},
            {"name": "Password", "value": PASSWORD},
        ]
    }
    result = redactor.structure(payload)
    assert result["fields"][1]["value"] == PLACEHOLDER
    assert result["fields"][0]["value"] == "Nitinasati"


def test_dictionary_keys_are_redacted_too(redactor: Redactor):
    assert redactor.structure({PASSWORD: "x"}) == {PLACEHOLDER: "x"}


def test_the_input_structure_is_not_mutated(redactor: Redactor):
    payload = {"value": PASSWORD}
    redactor.structure(payload)
    assert payload["value"] == PASSWORD, "redaction must copy, not mutate the caller's dict"


def test_non_string_leaves_survive(redactor: Redactor):
    payload = {"n": 42, "flag": True, "none": None, "f": 1.5}
    assert redactor.structure(payload) == payload


# --- events: every route a credential actually takes -----------------------


def test_a_credential_pasted_into_the_task_is_redacted(redactor: Redactor):
    event = RunStarted(
        run_id="r", seq=1, task=f"Sign in with password {PASSWORD}", tools=["browser_click"]
    )
    assert PASSWORD not in dump_event(redactor.event(event))["task"]


def test_a_fill_form_argument_is_redacted(redactor: Redactor):
    event = ToolCall(
        run_id="r",
        seq=2,
        step=1,
        call_id="c1",
        name="browser_fill_form",
        arguments={"fields": [{"name": "Password", "value": PASSWORD}]},
    )
    result = redactor.event(event)
    assert result.arguments["fields"][0]["value"] == PLACEHOLDER


def test_a_tool_result_echoing_the_value_is_redacted(redactor: Redactor):
    event = ToolResult(
        run_id="r", seq=3, step=1, call_id="c1", name="browser_type", ok=True,
        duration_ms=5, text=f'filled input with "{PASSWORD}"',
    )
    assert PASSWORD not in redactor.event(event).text


def test_model_prose_repeating_the_value_is_redacted(redactor: Redactor):
    event = Thinking(run_id="r", seq=4, step=1, text=f"I will type {PASSWORD} now", done=True)
    assert PASSWORD not in redactor.event(event).text


def test_an_error_message_quoting_the_value_is_redacted(redactor: Redactor):
    event = ErrorEvent(
        run_id="r", seq=5, step=1, kind="tool_failed",
        message=f"could not type {PASSWORD}",
        detail={"arguments": {"value": PASSWORD}},
    )
    result = redactor.event(event)
    assert PASSWORD not in result.message
    assert result.detail["arguments"]["value"] == PLACEHOLDER


def test_redaction_preserves_the_event_type_and_identity(redactor: Redactor):
    event = ToolCall(
        run_id="r", seq=6, step=2, call_id="c9", name="browser_type",
        arguments={"text": PASSWORD}, sensitive=True,
    )
    result = redactor.event(event)
    assert result.type == "tool_call"
    assert (result.run_id, result.seq, result.call_id, result.sensitive) == ("r", 6, "c9", True)


def test_an_event_with_no_secrets_in_it_round_trips_unchanged(redactor: Redactor):
    event = ToolCall(run_id="r", seq=7, step=1, call_id="c1", name="browser_snapshot")
    assert dump_event(redactor.event(event)) == dump_event(event)


def test_a_redactor_with_no_secrets_returns_the_identical_object():
    event = ToolCall(run_id="r", seq=8, step=1, call_id="c1", name="browser_snapshot")
    assert Redactor().event(event) is event
