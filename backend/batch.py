"""Running a use case over a file of input rows.

One shared browser session for the whole file, rows in sequence. That decision
buys speed -- sign in once rather than a thousand times -- and costs coupling:
one bad row can leave a modal open, a form half-filled, or the session logged
out. Sequential execution is what makes the recovery rules simple enough to
state exactly, which is what :class:`BatchRunner` implements:

1. A failed row never aborts the batch by itself. Record it and move on.
2. ``row_reset`` runs before every row, failed or not.
3. ``session_check`` runs between rows. If it fails the session is presumed
   logged out: re-run ``setup_steps`` **once**, then re-check.
4. If re-login fails, stop. Remaining rows stay ``pending``, never ``failed`` --
   they were not attempted, and saying otherwise would corrupt the results file.
5. Abort after N consecutive row failures. Ten minutes of a broken selector
   failing 400 rows is worse than stopping and telling someone.

Resume re-runs only rows that are not ``succeeded``, which covers all three
ways a batch ends early: re-login failure, the circuit breaker, and a process
restart.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import uuid
from datetime import date, datetime, time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from usecase import UseCase

log = logging.getLogger(__name__)

#: Columns the results file adds after the caller's own input columns.
RESULT_COLUMNS: tuple[str, ...] = (
    "row_index",
    "status",
    "failed_step_id",
    "error",
    "duration_ms",
    "llm_calls",
    "llm_tokens",
)


class BatchInputError(ValueError):
    """The uploaded rows could not be used. Raised before any browser opens."""


@dataclass(slots=True)
class BatchRows:
    rows: list[dict[str, Any]]
    columns: list[str]
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)


def parse_workbook(data: bytes, sheet: str | None = None) -> BatchRows:
    """Read input rows from an .xlsx workbook.

    Spreadsheets are how people actually keep lists of records, so accepting
    one removes an export step that is easy to get wrong -- a re-saved CSV
    silently mangles leading zeros, dates and anything containing a comma.

    ``openpyxl`` rather than pandas: this needs cell values, not a dataframe,
    and pandas would pull in numpy for nothing.
    """
    from openpyxl import load_workbook

    try:
        book = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001 - surfaced to the uploader verbatim
        raise BatchInputError(f"that file could not be read as a spreadsheet: {exc}") from exc

    try:
        worksheet = book[sheet] if sheet else book.worksheets[0]
    except KeyError:
        raise BatchInputError(
            f"the workbook has no sheet named {sheet!r}. It has: "
            + ", ".join(book.sheetnames)
        ) from None

    rows_iter = worksheet.iter_rows(values_only=True)
    header = next(rows_iter, None)
    if header is None:
        raise BatchInputError("that sheet is empty")

    columns = [str(cell).strip() for cell in header if cell is not None and str(cell).strip()]
    if not columns:
        raise BatchInputError("the first row must name each input column")

    rows: list[dict[str, Any]] = []
    for cells in rows_iter:
        row = {
            column: _cell_text(value)
            for column, value in zip(columns, cells)
        }
        # A spreadsheet's trailing blank rows are an artefact of editing it,
        # not data.
        if any(value not in (None, "") for value in row.values()):
            rows.append(row)

    if not rows:
        raise BatchInputError("that sheet has a header but no data rows")
    return BatchRows(rows=rows, columns=columns)


def _cell_text(value: Any) -> str:
    """One cell as the text a form would receive.

    Excel stores every number as a float, so an integer id arrives as "1234.0"
    and would be typed into the page that way.
    """
    if value is None:
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


def parse_csv(text: str) -> BatchRows:
    """Parse an uploaded CSV into input rows.

    Blank lines are skipped and values are stripped, because a spreadsheet
    export reliably contains both and neither is worth failing a 1,000-row job
    over. A missing header is not recoverable, so it raises.
    """
    if not text or not text.strip():
        raise BatchInputError("the uploaded file is empty")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise BatchInputError("the first line must be a header naming each input column")

    columns = [name.strip() for name in reader.fieldnames if name and name.strip()]
    if not columns:
        raise BatchInputError("the header row has no usable column names")

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for number, raw in enumerate(reader, start=2):
        row = {
            key.strip(): (value.strip() if isinstance(value, str) else value)
            for key, value in raw.items()
            if key and key.strip()
        }
        if not any(v not in (None, "") for v in row.values()):
            continue
        if None in raw:  # csv puts surplus cells under the None key
            warnings.append(f"line {number} has more cells than the header; the extras were ignored")
        rows.append(row)

    if not rows:
        raise BatchInputError("the file has a header but no data rows")
    return BatchRows(rows=rows, columns=columns, warnings=warnings)


def validate_rows(usecase: UseCase, rows: BatchRows) -> list[str]:
    """Check every row against the input schema **before** a browser opens.

    A bad column should fail in a millisecond, not on record 700. Returns the
    problems found; an empty list means the file is good to run.
    """
    problems: list[str] = []

    declared = usecase.input_names
    unknown = [c for c in rows.columns if c not in declared]
    if unknown:
        problems.append(
            "these columns do not match any declared input: "
            + ", ".join(sorted(unknown))
            + ". Expected: "
            + (", ".join(sorted(declared)) or "(none)")
        )

    missing_everywhere: set[str] = set()
    for index, row in enumerate(rows.rows):
        missing = usecase.missing_inputs(usecase.with_defaults(row))
        if missing:
            missing_everywhere.update(missing)
            if len(problems) < 6:
                problems.append(f"row {index + 1} is missing: {', '.join(missing)}")

    if len(rows.rows) > 5 and missing_everywhere:
        problems.append(
            "required input(s) missing from one or more rows: "
            + ", ".join(sorted(missing_everywhere))
        )
    return problems


def results_csv(usecase: UseCase, executions: Iterable[dict[str, Any]]) -> str:
    """One row out per row in, with a stable column order so files diff cleanly."""
    input_columns = [spec.name for spec in usecase.inputs]
    output_columns = list(usecase.outputs)
    header = [*input_columns, *RESULT_COLUMNS, *output_columns]

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)

    for execution in executions:
        inputs = execution.get("inputs") or {}
        outputs = execution.get("outputs") or {}
        writer.writerow(
            [
                *[inputs.get(name, "") for name in input_columns],
                execution.get("row_index", ""),
                execution.get("status", ""),
                execution.get("failed_step_id") or "",
                execution.get("error") or "",
                execution.get("duration_ms") or "",
                execution.get("llm_calls", 0),
                execution.get("llm_tokens", 0),
                *[outputs.get(name, "") for name in output_columns],
            ]
        )
    return buffer.getvalue()


@dataclass(slots=True)
class BatchProgress:
    """Live counters, surfaced while a batch is running."""

    total: int = 0
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    pending: int = 0
    relogins: int = 0
    stopped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "pending": self.pending,
            "relogins": self.relogins,
            "stopped_reason": self.stopped_reason,
        }


class BatchRunner:
    """Drives one use case over many rows on a single session.

    Takes an already-constructed executor so the batch logic can be tested
    without a browser: the recovery contract is the interesting part and it is
    entirely about *when* to call setup, reset and check, not about MCP.
    """

    def __init__(
        self,
        executor: Any,
        rows: list[dict[str, Any]],
        *,
        failure_streak_limit: int = 5,
        row_delay: float = 0.0,
        on_row: Callable[[int, dict[str, Any], Any], Any] | None = None,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        self.executor = executor
        self.rows = rows
        self.failure_streak_limit = max(failure_streak_limit, 1)
        self.row_delay = row_delay
        self.on_row = on_row
        self._sleep = sleep
        self.progress = BatchProgress(total=len(rows), pending=len(rows))
        self.results: list[Any] = []

    async def run(self) -> BatchProgress:
        setup = await self.executor.run_setup()
        if not setup.ok:
            self.progress.stopped_reason = f"setup failed: {setup.error}"
            log.error("batch setup failed", extra={"error": setup.error})
            return self.progress

        streak = 0
        for index, row in enumerate(self.rows):
            result = await self.executor.run_row(row)
            self.results.append(result)

            self.progress.attempted += 1
            self.progress.pending -= 1
            if result.ok:
                self.progress.succeeded += 1
                streak = 0
            else:
                self.progress.failed += 1
                streak += 1

            if self.on_row is not None:
                await self.on_row(index, row, result)

            if streak >= self.failure_streak_limit:
                self.progress.stopped_reason = (
                    f"stopped after {streak} consecutive row failures. Something is broken "
                    "for every row, not just these; the remaining rows were not attempted."
                )
                log.error("batch circuit breaker tripped", extra={"streak": streak})
                break

            if index == len(self.rows) - 1:
                break

            if not await self._session_is_healthy():
                self.progress.stopped_reason = (
                    "the shared session could not be re-authenticated. The remaining rows "
                    "were not attempted; resume the batch to run them."
                )
                break

            if self.row_delay and self._sleep is not None:
                await self._sleep(self.row_delay)

        await self.executor.run_teardown()
        return self.progress

    async def _session_is_healthy(self) -> bool:
        """Check the shared session, signing in again once if it has dropped."""
        if await self.executor.check_session():
            return True

        log.warning("shared session looks signed out; re-running setup once")
        self.progress.relogins += 1
        again = await self.executor.run_setup()
        if not again.ok:
            log.error("re-login failed", extra={"error": again.error})
            return False
        return await self.executor.check_session()


def new_batch_id() -> str:
    return uuid.uuid4().hex


def rows_from_json(payload: Any) -> BatchRows:
    """Accept a JSON array of objects as an alternative to a CSV upload."""
    if not isinstance(payload, list) or not payload:
        raise BatchInputError("rows must be a non-empty JSON array of objects")
    if not all(isinstance(item, dict) for item in payload):
        raise BatchInputError("every row must be a JSON object")

    columns: list[str] = []
    for row in payload:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    return BatchRows(rows=[dict(row) for row in payload], columns=columns)


def summarise(progress: BatchProgress) -> str:
    parts = [
        f"{progress.succeeded} succeeded",
        f"{progress.failed} failed",
        f"{progress.pending} not attempted",
    ]
    if progress.relogins:
        parts.append(f"{progress.relogins} re-login(s)")
    return ", ".join(parts)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
