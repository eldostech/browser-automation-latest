"""Reading files, and the fidelity rules that matter more than the reading.

The tests that carry weight here are the ones about *not* transforming values.
A batch types what the file said into a form, so a reader that helpfully parses
``0071`` into ``71`` produces a thousand wrong records and no error at all.
"""

from __future__ import annotations

import io

import pytest

from ingest import (
    BatchInputError,
    Dataset,
    read_csv_text,
    read_table,
    rows_from_json,
)


def workbook(rows: list[list], sheet_name: str = "Sheet1") -> bytes:
    from openpyxl import Workbook

    book = Workbook()
    book.active.title = sheet_name
    for row in rows:
        book.active.append(row)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


# --- fidelity --------------------------------------------------------------


def test_leading_zeros_survive_a_csv():
    """The reason CSV is read with dtype=str.

    An account number, a postcode and a phone number all lose their meaning
    when read as a number, and all three are exactly what these files hold.
    """
    dataset = read_csv_text("account,postcode\n0071,01234\n0072,00999\n")
    assert [row["account"] for row in dataset.rows] == ["0071", "0072"]
    assert [row["postcode"] for row in dataset.rows] == ["01234", "00999"]


def test_a_spreadsheet_integer_does_not_arrive_as_a_float():
    """Excel stores every number as a float, so 1234 comes back as 1234.0 and
    would be typed into the page that way."""
    dataset = read_table(workbook([["record_url", "answer"], ["https://a", 1234]]), "x.xlsx")
    assert dataset.rows[0]["answer"] == "1234"


def test_a_spreadsheet_date_becomes_a_plain_date():
    from datetime import datetime

    dataset = read_table(
        workbook([["when"], [datetime(2026, 8, 30)]]),
        "x.xlsx",
    )
    assert dataset.rows[0]["when"] == "2026-08-30"


def test_blank_cells_never_become_the_word_nan():
    """pandas fills a gap with a float sentinel, and str() of it is a word a
    form would happily accept."""
    dataset = read_csv_text("a,b\n1,\n2,y\n")
    assert dataset.rows[0]["b"] == ""
    assert "nan" not in repr(dataset.rows).lower()


# --- shapes of file --------------------------------------------------------


def test_a_bom_does_not_end_up_in_the_first_column_name():
    """A Windows spreadsheet export carries one, and reading it as plain utf-8
    leaves a column nothing will ever match."""
    data = "id,name\n1,a\n".encode("utf-8-sig")
    dataset = read_table(data, "rows.csv")
    assert dataset.column_names == ["id", "name"]


def test_a_tab_delimited_txt_is_read_without_being_told():
    dataset = read_table(b"record_url\tanswer\nhttps://a\t1\n", "rows.txt")
    assert dataset.column_names == ["record_url", "answer"]
    assert dataset.rows[0]["answer"] == "1"


def test_a_semicolon_csv_is_read_and_the_guess_is_reported():
    dataset = read_table(b"a;b\n1;2\n", "rows.csv")
    assert dataset.column_names == ["a", "b"]
    assert any("delimited" in w for w in dataset.warnings)


def test_a_non_utf8_file_is_read_and_flagged():
    dataset = read_table("name\nJosé\n".encode("cp1252"), "rows.csv")
    assert dataset.rows[0]["name"] == "José"
    assert any("not UTF-8" in w for w in dataset.warnings)


def test_trailing_blank_spreadsheet_rows_are_dropped():
    """An artefact of editing the file, not data -- and a run against one fills
    an empty form and reports success."""
    data = workbook([["record_url"], ["https://a"], [None], [""]])
    assert len(read_table(data, "x.xlsx")) == 1


def test_a_named_sheet_can_be_chosen_and_a_missing_one_says_what_exists():
    data = workbook([["record_url"], ["https://a"]], sheet_name="Prospects")
    assert read_table(data, "x.xlsx", "Prospects").rows[0]["record_url"] == "https://a"
    with pytest.raises(BatchInputError, match="Prospects"):
        read_table(data, "x.xlsx", "NotThere")


@pytest.mark.parametrize(
    "text,message",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("record_url\n", "no data rows"),
    ],
)
def test_unusable_files_are_refused_before_a_browser_opens(text, message):
    with pytest.raises(BatchInputError, match=message):
        read_csv_text(text)


@pytest.mark.parametrize(
    "rows,message",
    [
        ([], "empty"),
        ([["record_url"]], "no data rows"),
        # A header row of blank cells reads as no header at all, which is the
        # honest description of it.
        ([[None, None]], "empty"),
    ],
)
def test_unusable_workbooks_are_refused(rows, message):
    with pytest.raises(BatchInputError, match=message):
        read_table(workbook(rows), "x.xlsx")


def test_a_file_that_is_not_a_workbook_is_refused():
    with pytest.raises(BatchInputError, match="could not be read as a spreadsheet"):
        read_table(b"record_url\nhttps://a\n", "x.xlsx")


def test_an_unsupported_extension_says_what_is_supported():
    with pytest.raises(BatchInputError, match="CSV"):
        read_table(b"%PDF-1.4", "rows.pdf")


def test_rows_supplied_as_json_are_unioned_into_columns():
    dataset = rows_from_json([{"a": 1}, {"a": 2, "b": 3}])
    assert dataset.column_names == ["a", "b"]
    assert dataset.rows[1] == {"a": "2", "b": "3"}


# --- profiling -------------------------------------------------------------


def test_a_column_of_addresses_is_recognised_as_email():
    dataset = read_csv_text("contact\na@b.com\nc@d.org\n")
    assert dataset.profile("contact").shape == "email"


def test_one_bad_value_stops_a_column_being_a_shape():
    """Every value, not most.

    A column where nine in ten look like an email is a text column with a
    pattern, and treating it as an email column is how a mapper becomes
    confidently wrong.
    """
    dataset = read_csv_text("contact\na@b.com\nnot-an-email\n")
    assert dataset.profile("contact").shape is None


def test_kinds_are_inferred_without_touching_the_values():
    dataset = read_csv_text("n,d,t,b\n42,2026-08-30,hello,true\n7,2026-08-31,world,false\n")
    kinds = {c.name: c.kind for c in dataset.columns}
    assert kinds == {"n": "integer", "d": "date", "t": "text", "b": "boolean"}
    # The point of the whole exercise: inference informs mapping and never
    # rewrites what will be typed.
    assert dataset.rows[0]["n"] == "42"


def test_a_profile_counts_blanks_and_distinct_values():
    dataset = read_csv_text("x\na\na\n\nb\n")
    profile = dataset.profile("x")
    assert profile.non_null == 3
    assert profile.distinct == 2
    assert profile.examples == ["a", "b"]


def test_an_all_blank_column_is_empty_rather_than_text():
    dataset = read_csv_text("a,b\n1,\n2,\n")
    assert dataset.profile("b").kind == "empty"
    assert dataset.profile("b").is_empty


def test_a_column_of_unique_values_is_flagged():
    dataset = read_csv_text("id\n1\n2\n3\n")
    assert dataset.profile("id").is_unique


def test_the_api_shape_samples_rows_rather_than_shipping_them_all():
    dataset = read_csv_text("x\n" + "".join(f"{i}\n" for i in range(50)))
    body = dataset.to_dict(sample=5)
    assert body["row_count"] == 50
    assert len(body["sample"]) == 5
    assert body["columns"][0]["name"] == "x"


def test_examples_are_truncated():
    from ingest import EXAMPLE_MAX_CHARS

    dataset = read_csv_text("note\n" + "x" * 500 + "\n")
    assert len(dataset.profile("note").examples[0]) == EXAMPLE_MAX_CHARS


def test_a_dataset_round_trips_through_its_dict_form():
    """The profiles are stored as JSON and read back by the mapper, so the two
    directions have to agree."""
    from ingest import ColumnProfile

    original = read_csv_text("email\na@b.com\n").columns[0]
    assert ColumnProfile.from_dict(original.to_dict()) == original
