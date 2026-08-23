"""Batch execution: one shared session, rows in sequence, and the recovery
contract that makes that safe.

The runner is tested against a fake executor rather than a browser, because
what matters here is *when* setup, reset and the session check are called --
not how a click reaches Playwright.
"""

from __future__ import annotations

import pytest

from batch import (
    BatchInputError,
    BatchRunner,
    parse_csv,
    results_csv,
    rows_from_json,
    summarise,
    validate_rows,
)
from replay import RowResult
from usecase import InputSpec, Step, UseCase


class FakeExecutor:
    """Scripted stand-in for :class:`replay.UseCaseExecutor`."""

    def __init__(
        self,
        row_results: list[bool] | None = None,
        *,
        session_health: list[bool] | None = None,
        setup_results: list[bool] | None = None,
        outputs: dict | None = None,
    ) -> None:
        self.row_results = list(row_results or [])
        self.session_health = list(session_health or [])
        self.setup_results = list(setup_results or [])
        self.outputs = outputs or {}
        self.calls: list[str] = []
        self.rows_seen: list[dict] = []

    async def run_setup(self) -> RowResult:
        self.calls.append("setup")
        ok = self.setup_results.pop(0) if self.setup_results else True
        return RowResult(ok=ok, error=None if ok else "could not sign in")

    async def run_row(self, inputs: dict) -> RowResult:
        self.calls.append("row")
        self.rows_seen.append(inputs)
        ok = self.row_results.pop(0) if self.row_results else True
        return RowResult(
            ok=ok,
            outputs=dict(self.outputs) if ok else {},
            failed_step_id=None if ok else "s1",
            error=None if ok else "element not found",
            duration_ms=5,
        )

    async def run_teardown(self) -> RowResult:
        self.calls.append("teardown")
        return RowResult(ok=True)

    async def check_session(self) -> bool:
        self.calls.append("check")
        return self.session_health.pop(0) if self.session_health else True


def rows(n: int) -> list[dict]:
    return [{"record_url": f"https://example.com/{i}"} for i in range(n)]


# --- the shared session --------------------------------------------------


async def test_setup_runs_once_for_the_whole_file():
    """The reason the schema splits setup from rows: sign in once, not 3 times."""
    executor = FakeExecutor([True, True, True])
    progress = await BatchRunner(executor, rows(3)).run()

    assert executor.calls.count("setup") == 1
    assert executor.calls.count("row") == 3
    assert (progress.succeeded, progress.failed, progress.pending) == (3, 0, 0)


async def test_every_row_gets_its_own_inputs():
    executor = FakeExecutor([True, True])
    await BatchRunner(executor, rows(2)).run()
    assert executor.rows_seen == [
        {"record_url": "https://example.com/0"},
        {"record_url": "https://example.com/1"},
    ]


async def test_a_failing_setup_stops_before_any_row_runs():
    executor = FakeExecutor([True], setup_results=[False])
    progress = await BatchRunner(executor, rows(3)).run()

    assert "row" not in executor.calls
    assert progress.attempted == 0
    assert progress.pending == 3
    assert "setup failed" in progress.stopped_reason


async def test_teardown_runs_at_the_end():
    executor = FakeExecutor([True])
    await BatchRunner(executor, rows(1)).run()
    assert executor.calls[-1] == "teardown"


# --- rule 1: a failed row does not abort the batch -------------------------


async def test_a_failed_row_does_not_stop_the_others():
    executor = FakeExecutor([True, False, True])
    progress = await BatchRunner(executor, rows(3)).run()

    assert executor.calls.count("row") == 3
    assert (progress.succeeded, progress.failed) == (2, 1)
    assert progress.stopped_reason is None


async def test_a_failed_row_records_its_failing_step():
    executor = FakeExecutor([False])
    runner = BatchRunner(executor, rows(1))
    await runner.run()
    assert runner.results[0].failed_step_id == "s1"


# --- rule 3: the session check, and one automatic re-login -----------------


async def test_the_session_is_checked_between_rows_but_not_after_the_last():
    executor = FakeExecutor([True, True, True])
    await BatchRunner(executor, rows(3)).run()
    assert executor.calls.count("check") == 2, "no point checking after the final row"


async def test_a_dropped_session_triggers_exactly_one_re_login():
    # Unhealthy after row 1; healthy again once setup has re-run.
    executor = FakeExecutor([True, True], session_health=[False, True])
    progress = await BatchRunner(executor, rows(2)).run()

    assert executor.calls.count("setup") == 2, "signed in again, once"
    assert progress.relogins == 1
    assert progress.succeeded == 2
    assert progress.stopped_reason is None


# --- rule 4: unattempted rows stay pending, never failed -------------------


async def test_a_failed_re_login_stops_and_leaves_the_rest_pending():
    """Marking unattempted rows as failed would corrupt the results file."""
    executor = FakeExecutor(
        [True, True, True, True, True],
        session_health=[False],
        setup_results=[True, False],
    )
    progress = await BatchRunner(executor, rows(5)).run()

    assert progress.attempted == 1
    assert progress.succeeded == 1
    assert progress.failed == 0, "the four unattempted rows are not failures"
    assert progress.pending == 4
    assert "could not be re-authenticated" in progress.stopped_reason


async def test_a_session_that_stays_unhealthy_after_re_login_stops_the_batch():
    executor = FakeExecutor([True, True, True], session_health=[False, False])
    progress = await BatchRunner(executor, rows(3)).run()

    assert progress.attempted == 1
    assert progress.pending == 2


# --- rule 5: the circuit breaker -------------------------------------------


async def test_consecutive_failures_trip_the_circuit_breaker():
    executor = FakeExecutor([False] * 10)
    progress = await BatchRunner(executor, rows(10), failure_streak_limit=3).run()

    assert progress.attempted == 3
    assert progress.failed == 3
    assert progress.pending == 7
    assert "3 consecutive row failures" in progress.stopped_reason


async def test_a_success_resets_the_failure_streak():
    executor = FakeExecutor([False, False, True, False, False])
    progress = await BatchRunner(executor, rows(5), failure_streak_limit=3).run()

    assert progress.attempted == 5, "the streak never reached three in a row"
    assert progress.stopped_reason is None


async def test_scattered_failures_do_not_trip_the_breaker():
    executor = FakeExecutor([True, False, True, False, True])
    progress = await BatchRunner(executor, rows(5), failure_streak_limit=3).run()
    assert (progress.succeeded, progress.failed, progress.pending) == (3, 2, 0)


# --- progress reporting ----------------------------------------------------


async def test_a_per_row_callback_fires_for_every_attempted_row():
    seen: list[tuple[int, bool]] = []

    async def record(index, row, result):
        seen.append((index, result.ok))

    await BatchRunner(FakeExecutor([True, False]), rows(2), on_row=record).run()
    assert seen == [(0, True), (1, False)]


async def test_the_delay_between_rows_is_applied():
    slept: list[float] = []

    async def sleep(seconds: float):
        slept.append(seconds)

    await BatchRunner(
        FakeExecutor([True, True, True]), rows(3), row_delay=0.25, sleep=sleep
    ).run()
    assert slept == [0.25, 0.25], "between rows, not after the last"


def test_summarise_reads_clearly():
    from batch import BatchProgress

    text = summarise(BatchProgress(total=5, succeeded=3, failed=1, pending=1, relogins=2))
    assert text == "3 succeeded, 1 failed, 1 not attempted, 2 re-login(s)"


# --- input parsing ---------------------------------------------------------


def test_a_csv_is_parsed_into_rows():
    parsed = parse_csv("record_url,answer\nhttps://a,1\nhttps://b,2\n")
    assert parsed.columns == ["record_url", "answer"]
    assert parsed.rows == [
        {"record_url": "https://a", "answer": "1"},
        {"record_url": "https://b", "answer": "2"},
    ]


def test_blank_lines_and_padding_are_tolerated():
    """A spreadsheet export reliably contains both."""
    parsed = parse_csv("record_url , answer\n  https://a , 1 \n\n\nhttps://b,2\n")
    assert len(parsed) == 2
    assert parsed.rows[0] == {"record_url": "https://a", "answer": "1"}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("record_url,answer\n", "no data rows"),
    ],
)
def test_unusable_files_are_refused_with_a_reason(text: str, message: str):
    with pytest.raises(BatchInputError, match=message):
        parse_csv(text)


def test_json_rows_are_accepted_too():
    parsed = rows_from_json([{"a": 1}, {"a": 2, "b": 3}])
    assert parsed.columns == ["a", "b"]
    assert len(parsed) == 2


@pytest.mark.parametrize("payload", [[], {}, "nope", [1, 2]])
def test_bad_json_rows_are_refused(payload):
    with pytest.raises(BatchInputError):
        rows_from_json(payload)


# --- validation before the browser opens -----------------------------------


def use_case() -> UseCase:
    return UseCase(
        name="x",
        inputs=[InputSpec(name="record_url", type="url"), InputSpec(name="answer")],
        row_steps=[
            Step(id="s1", action="navigate", url="{{input.record_url}}"),
            Step(
                id="s2",
                action="fill",
                locators=[{"strategy": "css", "selector": "#a"}],
                value="{{input.answer}}",
            ),
        ],
    )


def test_a_good_file_validates_clean():
    assert validate_rows(use_case(), parse_csv("record_url,answer\nhttps://a,1\n")) == []


def test_an_unknown_column_is_reported_with_what_was_expected():
    problems = validate_rows(use_case(), parse_csv("record_url,nonsense\nhttps://a,1\n"))
    assert any("nonsense" in p and "record_url" in p for p in problems)


def test_a_row_missing_a_required_input_is_reported_by_number():
    problems = validate_rows(use_case(), parse_csv("record_url,answer\nhttps://a,\n"))
    assert any("row 1" in p and "answer" in p for p in problems)


def test_only_the_first_few_bad_rows_are_listed_individually():
    text = "record_url,answer\n" + "".join("https://a,\n" for _ in range(50))
    problems = validate_rows(use_case(), parse_csv(text))
    assert len(problems) <= 8, "a 1,000-row file must not produce 1,000 error lines"
    assert any("missing from one or more rows" in p for p in problems)


# --- results ---------------------------------------------------------------


def test_the_results_csv_joins_inputs_to_outcomes():
    case = UseCase(
        name="x",
        inputs=[InputSpec(name="record_url", type="url")],
        row_steps=[
            Step(
                id="s1",
                action="extract",
                locators=[{"strategy": "css", "selector": "#s"}],
                output="score",
            )
        ],
        outputs=["score"],
    )
    csv_text = results_csv(
        case,
        [
            {"inputs": {"record_url": "https://a"}, "outputs": {"score": "92%"},
             "row_index": 0, "status": "succeeded", "duration_ms": 120,
             "llm_calls": 0, "llm_tokens": 0},
            {"inputs": {"record_url": "https://b"}, "outputs": None, "row_index": 1,
             "status": "failed", "failed_step_id": "s1", "error": "not found",
             "duration_ms": 90, "llm_calls": 0, "llm_tokens": 0},
        ],
    )
    lines = csv_text.strip().splitlines()

    assert lines[0] == (
        "record_url,row_index,status,failed_step_id,error,duration_ms,"
        "llm_calls,llm_tokens,score"
    )
    assert lines[1] == "https://a,0,succeeded,,,120,0,0,92%"
    assert lines[2].startswith("https://b,1,failed,s1,not found,90,0,0")


def test_the_results_header_is_stable_for_an_empty_batch():
    csv_text = results_csv(use_case(), [])
    assert csv_text.strip().splitlines() == [
        "record_url,answer,row_index,status,failed_step_id,error,duration_ms,llm_calls,llm_tokens"
    ]
