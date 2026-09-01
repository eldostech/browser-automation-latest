"""Matching columns to declared inputs, without a model where possible.

The assertions worth having are about the cases where a mapper is *wrong*
rather than merely unhelpful, because an unhelpful mapping is corrected in the
UI and a wrong one runs a thousand times.
"""

from __future__ import annotations

from ingest import read_csv_text
from mapping import (
    CONFIDENT,
    apply_choice,
    apply_mapping,
    as_mapping,
    normalise,
    score,
    suggest,
    unresolved,
)


def columns(csv_text: str):
    return read_csv_text(csv_text).columns


def by_field(suggestions):
    return {s.field: s.column for s in suggestions}


# --- the free cases --------------------------------------------------------


def test_an_exact_name_matches_whatever_the_punctuation():
    cols = columns("Customer Email\na@b.com\n")
    [suggestion] = suggest([("customer_email", "string")], cols)
    assert suggestion.column == "Customer Email"
    assert suggestion.confident
    assert suggestion.score == 1.0


def test_a_synonym_matches():
    cols = columns("E-Mail\na@b.com\n")
    [suggestion] = suggest([("email", "string")], cols)
    assert suggestion.column == "E-Mail"
    assert suggestion.confident


def test_noise_words_do_not_prevent_a_match():
    cols = columns("Customer Email Address\na@b.com\n")
    [suggestion] = suggest([("email", "string")], cols)
    assert suggestion.column == "Customer Email Address"


def test_normalise_reduces_both_vocabularies_to_one():
    assert normalise("Customer Email") == "customer_email"
    assert normalise("  ORDER-ID  ") == "order_id"
    assert normalise("Website") == "link"


# --- where the value shape beats the name ----------------------------------


def test_a_badly_named_column_of_addresses_still_matches_an_email_field():
    cols = columns("Contact,Notes\na@b.com,hello\nc@d.com,there\n")
    [suggestion] = suggest([("email", "string")], cols)
    assert suggestion.column == "Contact"
    assert "email" in suggestion.reason


def test_contents_that_contradict_the_name_lower_the_score():
    """The names agree and the values do not. Offering that confidently is how
    a thousand records get an order number typed into an email field."""
    cols = columns("email\nhttps://example.com/a\nhttps://example.com/b\n")
    [suggestion] = suggest([("email", "string")], cols)
    assert suggestion.score < CONFIDENT
    assert "not" in suggestion.reason


def test_a_url_field_prefers_the_column_holding_urls():
    cols = columns("reference,link\nA-1,https://example.com/1\nA-2,https://example.com/2\n")
    [suggestion] = suggest([("record_url", "url")], cols)
    assert suggestion.column == "link"


# --- where types rule a column out entirely --------------------------------


def test_a_date_column_is_not_offered_for_a_number_field():
    cols = columns("quantity\n2026-08-30\n2026-08-31\n")
    [suggestion] = suggest([("quantity", "number")], cols)
    assert suggestion.column is None
    assert "no column" in suggestion.reason


def test_a_text_column_is_not_offered_for_a_boolean_field():
    cols = columns("active\nmaybe\nperhaps\n")
    assert score("active", "boolean", cols[0]) is None


def test_a_number_column_fills_a_string_field():
    """A form field is a string in the end, so string accepts anything."""
    cols = columns("quantity\n42\n")
    assert score("quantity", "string", cols[0]) is not None


# --- being honest about not knowing ----------------------------------------


def test_an_empty_dataset_says_so_rather_than_guessing():
    cols = columns("a\n1\n")
    [suggestion] = suggest([("record_url", "url")], [c for c in cols if False])
    assert suggestion.column is None
    assert "no columns" in suggestion.reason


def test_an_empty_column_is_ranked_below_a_populated_one():
    cols = columns("email,contact\n,a@b.com\n,c@d.com\n")
    [suggestion] = suggest([("email", "string")], cols)
    assert suggestion.column == "contact"


def test_one_column_may_be_offered_to_two_fields():
    """Resolving that is the user's call.

    Assigning columns greedily -- first field wins, second gets whatever is
    left -- produced worse answers than showing the ranking honestly, because
    the order of the fields is an accident of the recording.
    """
    cols = columns("name\nAda\n")
    suggestions = suggest([("first_name", "string"), ("last_name", "string")], cols)
    assert all(s.column == "name" for s in suggestions)


def test_alternatives_are_offered_so_the_ui_can_show_a_dropdown():
    cols = columns("email,contact_email,backup_email\na@b.com,c@d.com,e@f.com\n")
    [suggestion] = suggest([("email", "string")], cols)
    assert suggestion.column == "email"
    assert [c.column for c in suggestion.alternatives]


# --- what would be worth asking a model ------------------------------------


def test_a_clean_file_needs_no_model_call():
    cols = columns("record_url,answer\nhttps://a,1\n")
    suggestions = suggest([("record_url", "url"), ("answer", "string")], cols)
    assert unresolved(suggestions) == []


def test_an_unmatched_field_is_worth_asking_about():
    cols = columns("wibble\n1\n")
    suggestions = suggest([("record_url", "url")], cols)
    assert [s.field for s in unresolved(suggestions)] == ["record_url"]


def test_a_close_run_thing_is_worth_asking_about():
    """Two candidates within a hair of each other is not really a decision,
    and it is exactly the case a person -- or a model -- should look at."""
    cols = columns("email_one,email_two\na@b.com,c@d.com\n")
    suggestions = suggest([("email", "string")], cols)
    assert [s.field for s in unresolved(suggestions)] == ["email"]


# --- committing a decision -------------------------------------------------


def test_a_choice_must_name_a_column_that_was_offered():
    """The same invariant healing keeps about page elements.

    A column name that came from a model rather than from the file has no route
    into something that will run a thousand times.
    """
    cols = columns("email,contact\na@b.com,c@d.com\n")
    suggestions = suggest([("email", "string")], cols)
    apply_choice(suggestions, "email", "invented_column", "the model said so")
    assert suggestions[0].column == "email"


def test_a_choice_naming_a_real_alternative_is_taken():
    cols = columns("email,contact\na@b.com,c@d.com\n")
    suggestions = suggest([("email", "string")], cols)
    apply_choice(suggestions, "email", "contact", "the user picked it")
    assert suggestions[0].column == "contact"
    assert suggestions[0].reason == "the user picked it"


def test_a_field_can_be_cleared():
    cols = columns("email\na@b.com\n")
    suggestions = suggest([("email", "string")], cols)
    apply_choice(suggestions, "email", None, "not in this file")
    assert as_mapping(suggestions) == {}


# --- renaming the rows -----------------------------------------------------


def test_applying_a_mapping_renames_columns_to_field_names():
    rows = [{"Customer Email": "a@b.com", "Notes": "hi"}]
    assert apply_mapping(rows, {"email": "Customer Email"}) == [{"email": "a@b.com"}]


def test_an_unmapped_column_is_dropped_rather_than_passed_through():
    """It would otherwise reach validate_rows as an unknown input and send the
    user looking for a problem they have already solved."""
    rows = [{"a": "1", "spare": "2"}]
    assert apply_mapping(rows, {"a": "a"}) == [{"a": "1"}]


def test_a_mapped_column_missing_from_a_row_becomes_blank():
    rows = [{"a": "1"}, {}]
    assert apply_mapping(rows, {"a": "a"}) == [{"a": "1"}, {"a": ""}]
