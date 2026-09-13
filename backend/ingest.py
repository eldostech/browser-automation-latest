"""Reading a file of input rows, and describing what is in it.

One entry point, :func:`read_table`, for CSV, Excel and delimited text. It
replaces ``batch.parse_csv`` and ``batch.parse_workbook``, which between them
did the reading but nothing else.

Why pandas now, when the requirements file argued against it
------------------------------------------------------------
That argument -- "this needs cell values, not a dataframe, and pandas would
pull in numpy for nothing" -- was right for the job it was written about, which
was reading rows and handing them to a browser. It stops being right here,
because the mapper needs a *description* of each column as well as its
contents: what type it holds, how often it is empty, how many distinct values
it has, and what a few of them look like. Inferring that by hand is writing
pandas badly. Delimiter sniffing and encoding fallbacks are the same story.

What is deliberately **not** delegated to pandas is type coercion. Every value
comes back as the text a form would receive, because that is what a replay
types into a field. Letting pandas parse a column of account numbers turns
``0071`` into ``71`` and ``2026-01-02`` into a Timestamp, and the recorded step
then fills a different string than the one in the file. So CSV and text are
read with ``dtype=str`` and Excel cells are converted explicitly. The inferred
type lives in the *profile*, where it informs mapping, and never touches the
value.

Redaction
---------
:meth:`ColumnProfile.examples` are shown in the UI and may be sent to a model
when a mapping is ambiguous, so they are sampled from the data itself. They are
capped in length and count; a column holding something sensitive is the
uploader's decision to make, and the same values are about to be typed into a
web form regardless.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any

log = logging.getLogger(__name__)

#: How many example values a profile carries. Three is enough for a person to
#: recognise a column and for a model to tell an email from an order number,
#: and few enough that a wide file does not turn into a large payload.
EXAMPLE_COUNT = 3

#: Examples are truncated at this length. A free-text column can hold a page.
EXAMPLE_MAX_CHARS = 80

#: Encodings tried in order. utf-8-sig first because a Windows spreadsheet
#: export reliably carries a BOM, and reading it as plain utf-8 leaves the
#: first column named "﻿id" -- which then matches no declared input and
#: produces a baffling error a long way from its cause.
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL = re.compile(r"^https?://\S+$", re.IGNORECASE)
_INTEGER = re.compile(r"^[+-]?\d+$")
_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?$|^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")
_BOOLEAN = frozenset({"true", "false", "yes", "no", "y", "n", "0", "1"})
_PHONE = re.compile(r"^[+(]?[\d][\d\s().-]{6,}$")


class BatchInputError(ValueError):
    """The uploaded rows could not be used. Raised before any browser opens."""


@dataclass(slots=True)
class ColumnProfile:
    """What one column holds, as far as the file can say.

    This is what the mapper reasons over. ``kind`` is inferred from the values
    and never applied to them -- see the module docstring.
    """

    name: str
    #: text | integer | number | date | boolean | empty
    kind: str
    #: A recognised value shape, where there is one: email, url, phone, date.
    #: Narrower than ``kind`` and the more useful signal when matching a
    #: column named "Contact" to a field recorded with an email in it.
    shape: str | None
    non_null: int
    nulls: int
    distinct: int
    examples: list[str]

    @property
    def is_empty(self) -> bool:
        return self.non_null == 0

    @property
    def is_unique(self) -> bool:
        """Every populated value differs. Often an id, and rarely a good
        candidate for a field that repeats."""
        return self.non_null > 1 and self.distinct == self.non_null

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "shape": self.shape,
            "non_null": self.non_null,
            "nulls": self.nulls,
            "distinct": self.distinct,
            "examples": list(self.examples),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnProfile":
        return cls(
            name=str(data["name"]),
            kind=str(data.get("kind") or "text"),
            shape=data.get("shape"),
            non_null=int(data.get("non_null") or 0),
            nulls=int(data.get("nulls") or 0),
            distinct=int(data.get("distinct") or 0),
            examples=[str(v) for v in (data.get("examples") or [])],
        )


@dataclass(slots=True)
class Dataset:
    """Rows plus a description of each column.

    Replaces ``BatchRows``. The rows are the same shape they always were --
    ``{column: text}`` -- so everything downstream of this is unchanged; what
    is new is ``columns`` carrying profiles rather than bare names.
    """

    rows: list[dict[str, Any]]
    columns: list[ColumnProfile]
    warnings: list[str] = field(default_factory=list)
    #: csv | xlsx | text | json, for the UI and for error messages.
    source: str = "csv"

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]

    def profile(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def to_dict(self, *, sample: int = 5) -> dict[str, Any]:
        """What the API returns. The rows are sampled: a dataset listing that
        ships ten thousand rows to draw a preview table is a mistake that only
        shows up in production."""
        return {
            "source": self.source,
            "row_count": len(self.rows),
            "columns": [column.to_dict() for column in self.columns],
            "sample": self.rows[:sample],
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_table(data: bytes, filename: str = "", sheet: str | None = None) -> Dataset:
    """Read an uploaded file into rows and profiles.

    The extension chooses the reader, falling back to delimited text -- which
    covers the case that actually happens, a CSV saved as ``.txt``.
    """
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix in ("xlsx", "xlsm", "xltx", "xls"):
        return _read_excel(data, sheet)
    if suffix in ("csv", "txt", "tsv", "tab", ""):
        return _read_delimited(data, source="csv" if suffix == "csv" else "text")
    raise BatchInputError(
        f"{suffix or 'that'} files are not supported. Upload a CSV, a spreadsheet, "
        "or a delimited text file."
    )


def read_csv_text(text: str) -> Dataset:
    """Rows from CSV text, for the JSON API that posts a `csv` field."""
    return _read_delimited(text.encode("utf-8"), source="csv")


def rows_from_json(rows: list[dict[str, Any]] | None) -> Dataset:
    """Rows supplied directly, for callers that already parsed their own file."""
    if not rows:
        raise BatchInputError("no rows were supplied")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    normalised = [
        {column: _as_text(row.get(column)) for column in columns} for row in rows
    ]
    return Dataset(
        rows=normalised, columns=_profile(normalised, columns), source="json"
    )


def _decode(data: bytes) -> tuple[str, list[str]]:
    """Text, and a warning if it took more than the obvious encoding."""
    if not data:
        raise BatchInputError("the uploaded file is empty")
    for index, encoding in enumerate(ENCODINGS):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if index <= 1:
            return text, []
        return text, [
            f"this file is not UTF-8; it was read as {encoding}. Check any accented "
            "characters before running a thousand rows."
        ]
    raise BatchInputError("that file is not text this can read")


def _sniff(sample: str) -> str:
    """The delimiter, guessed, with the guess bounded to plausible ones.

    ``csv.Sniffer`` will happily decide a column of sentences is
    space-delimited, so the candidate set is narrowed to the four that a data
    file actually uses.
    """
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        # A single-column file has no delimiter to find, which is not an error.
        return "\t" if "\t" in sample.splitlines()[0] else ","


def _read_delimited(data: bytes, *, source: str) -> Dataset:
    import pandas as pd

    text, warnings = _decode(data)
    if not text.strip():
        raise BatchInputError("the uploaded file is empty")

    delimiter = _sniff(text[:8192])
    try:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            # Everything is text. See the module docstring: coercing here is
            # how leading zeros and date formats get silently rewritten.
            dtype=str,
            keep_default_na=False,
            na_values=[],
            skip_blank_lines=True,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the uploader verbatim
        raise BatchInputError(f"that file could not be read: {exc}") from exc

    if frame.columns.empty:
        raise BatchInputError("the first line must be a header naming each input column")

    unnamed = [str(c) for c in frame.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        warnings.append(
            f"{len(unnamed)} column(s) had no name in the header and were ignored"
        )
        frame = frame.drop(columns=unnamed)

    columns = [str(c).strip() for c in frame.columns if str(c).strip()]
    if not columns:
        raise BatchInputError("the header row has no usable column names")
    frame.columns = [str(c).strip() for c in frame.columns]

    rows = _rows_from_frame(frame, columns)
    if not rows:
        raise BatchInputError("the file has a header but no data rows")
    if delimiter != "," and source == "csv":
        warnings.append(f"read as {delimiter!r}-delimited rather than comma-delimited")
    return Dataset(rows=rows, columns=_profile(rows, columns), warnings=warnings, source=source)


def _read_excel(data: bytes, sheet: str | None) -> Dataset:
    import pandas as pd

    try:
        book = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    except Exception as exc:  # noqa: BLE001 - surfaced to the uploader verbatim
        raise BatchInputError(f"that file could not be read as a spreadsheet: {exc}") from exc

    if sheet and sheet not in book.sheet_names:
        raise BatchInputError(
            f"the workbook has no sheet named {sheet!r}. It has: "
            + ", ".join(book.sheet_names)
        )

    frame = book.parse(sheet or book.sheet_names[0])
    if frame.columns.empty:
        raise BatchInputError("that sheet is empty")

    warnings: list[str] = []
    unnamed = [str(c) for c in frame.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        warnings.append(
            f"{len(unnamed)} column(s) had no name in the header row and were ignored"
        )
        frame = frame.drop(columns=unnamed)

    frame.columns = [str(c).strip() for c in frame.columns]
    columns = [c for c in frame.columns if c]
    if not columns:
        raise BatchInputError("the first row must name each input column")

    rows = _rows_from_frame(frame, columns)
    if not rows:
        raise BatchInputError("that sheet has a header but no data rows")
    if len(book.sheet_names) > 1 and not sheet:
        warnings.append(
            f"the workbook has {len(book.sheet_names)} sheets; {book.sheet_names[0]!r} was used"
        )
    return Dataset(rows=rows, columns=_profile(rows, columns), warnings=warnings, source="xlsx")


def _rows_from_frame(frame, columns: list[str]) -> list[dict[str, Any]]:
    """Every cell as the text a form would receive, blank rows dropped.

    Trailing blank rows are an artefact of editing a spreadsheet rather than
    data, and a run against one fills an empty form and reports success.
    """
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        row = {column: _as_text(record.get(column)) for column in columns}
        if any(value != "" for value in row.values()):
            rows.append(row)
    return rows


def _as_text(value: Any) -> str:
    """One cell as the text a form would receive.

    Excel stores every number as a float, so an integer id arrives as
    ``1234.0`` and would be typed into the page that way. pandas adds its own
    sentinels -- ``NaN`` for a blank cell, ``NaT`` for a blank date -- which
    stringify into words that would be typed just as happily.
    """
    if value is None:
        return ""
    # NaN is the only value not equal to itself, which is how a float sentinel
    # is detected without importing numpy to ask.
    if isinstance(value, float) and value != value:
        return ""
    if str(type(value)) == "<class 'pandas._libs.tslibs.nattype.NaTType'>":
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == time.min else value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value).strip()


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def _profile(rows: list[dict[str, Any]], columns: list[str]) -> list[ColumnProfile]:
    return [_profile_column(name, [row.get(name, "") for row in rows]) for name in columns]


def _profile_column(name: str, values: list[str]) -> ColumnProfile:
    populated = [v for v in values if v not in ("", None)]
    distinct = len(set(populated))

    examples: list[str] = []
    for value in populated:
        if value in examples:
            continue
        examples.append(value[:EXAMPLE_MAX_CHARS])
        if len(examples) == EXAMPLE_COUNT:
            break

    return ColumnProfile(
        name=name,
        kind=_kind_of(populated),
        shape=_shape_of(populated),
        non_null=len(populated),
        nulls=len(values) - len(populated),
        distinct=distinct,
        examples=examples,
    )


def _all(values: list[str], predicate) -> bool:
    """True when every value matches. An empty column matches nothing, which
    is why ``all([])`` is not what is wanted here."""
    return bool(values) and all(predicate(v) for v in values)


def _kind_of(values: list[str]) -> str:
    if not values:
        return "empty"
    if _all(values, lambda v: v.lower() in _BOOLEAN) and len(set(v.lower() for v in values)) <= 2:
        return "boolean"
    if _all(values, lambda v: bool(_INTEGER.match(v))):
        return "integer"
    if _all(values, lambda v: bool(_NUMBER.match(v))):
        return "number"
    if _all(values, lambda v: bool(_DATE.match(v))):
        return "date"
    return "text"


def _shape_of(values: list[str]) -> str | None:
    """A recognised value shape, if every populated value has it.

    Every, not most: a column where nine values in ten look like an email is
    not an email column, it is a text column with a pattern, and treating it as
    the former is how a mapper becomes confidently wrong.
    """
    if not values:
        return None
    for shape, pattern in (("email", _EMAIL), ("url", _URL), ("date", _DATE)):
        if _all(values, lambda v, p=pattern: bool(p.match(v))):
            return shape
    if _all(values, lambda v: bool(_PHONE.match(v))) and not _all(
        values, lambda v: bool(_INTEGER.match(v))
    ):
        return "phone"
    return None


__all__ = [
    "BatchInputError",
    "ColumnProfile",
    "Dataset",
    "EXAMPLE_COUNT",
    "read_csv_text",
    "read_table",
    "rows_from_json",
]
