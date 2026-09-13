"""Running a use case over a set of input rows.

Reading the file and describing its columns is ``ingest.py``. What is left here
is the part that knows about *running*: checking rows against a use case before
a browser opens, driving them one at a time, and writing the results back out.

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
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ingest import Dataset
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


def validate_rows(usecase: UseCase, rows: Dataset) -> list[str]:
    """Check every row against the input schema **before** a browser opens.

    A bad column should fail in a millisecond, not on record 700. Returns the
    problems found; an empty list means the file is good to run.

    Reading and profiling the file is ``ingest.py``'s job; this is the part
    that knows what a *use case* needs, which is why it stayed here.
    """
    problems: list[str] = []

    declared = usecase.input_names
    unknown = [c for c in rows.column_names if c not in declared]
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
        indices: list[int] | None = None,
        run_row: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.executor = executor
        #: How one row is run. Defaults to the executor's own method, which is
        #: the Strict and Guided path; a caller can supply the operate graph
        #: instead, which replays first and asks an agent to clear the way only
        #: when a row fails. Injected rather than branched on here so this loop
        #: -- the failure streak, the pacing, the row indices -- has one shape
        #: whichever is running underneath it.
        self._run_row = run_row
        self.rows = rows
        #: The original row numbers, when this is a resume running a subset.
        #: Without them a resumed batch would label its rows 0..n again and the
        #: results file would disagree with the first attempt.
        self.indices = list(indices or [])
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
            # Stamped on every step row this produces, so a thousand-row batch
            # can be read back one record at a time.
            self.executor.row_index = self.indices[index] if self.indices else index
            result = await (
                self._run_row(row) if self._run_row else self.executor.run_row(row)
            )
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

            # A row is the unit somebody resumes, retries and reads results by,
            # so it is the point at which what happened has to be on disk
            # rather than in this process. Events are written in batches now
            # (see `eventbuffer.py`), which makes that a decision rather than a
            # side effect of writing each one as it happened -- and it bounds
            # what a hard kill can lose to the row in flight.
            # Both getattrs matter: a test injects its own `_run_row` with a
            # stand-in executor, and a sink built for a run with no database
            # behind it has nothing to flush.
            flush = getattr(getattr(self.executor, "sink", None), "flush", None)
            if flush is not None:
                await flush()

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



def summarise(progress: BatchProgress) -> str:
    parts = [
        f"{progress.succeeded} succeeded",
        f"{progress.failed} failed",
        f"{progress.pending} not attempted",
    ]
    if progress.relogins:
        parts.append(f"{progress.relogins} re-login(s)")
    return ", ".join(parts)
