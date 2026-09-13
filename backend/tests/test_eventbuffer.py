"""Batched event writing: what it saves, and what it must not trade away.

The saving is round trips against a remote database. The thing not traded away
is the resume contract -- ``seq`` is the token a reconnecting client sends, so
a client must never see a ``seq`` that a later catch-up read cannot return.
"""

from __future__ import annotations

import asyncio

import pytest

from events import StepFinished, StepStarted, Thinking
from eventbuffer import EventBuffer, RowBuffer

pytestmark = pytest.mark.anyio


class Recorder:
    """Collects what was written and what was published, in order."""

    def __init__(self, fail: bool = False) -> None:
        self.writes: list[list] = []
        self.published: list[list] = []
        self.order: list[str] = []
        self.fail = fail

    async def write(self, batch: list) -> None:
        if self.fail:
            raise RuntimeError("the database is unreachable")
        self.writes.append(list(batch))
        self.order.append(f"write:{len(batch)}")

    def publish(self, batch: list) -> None:
        self.published.append(list(batch))
        self.order.append(f"publish:{len(batch)}")


def started(seq: int) -> StepStarted:
    return StepStarted(
        run_id="r1", seq=seq, step=seq, step_id=f"s{seq}", action="click",
        description="", phase="row",
    )


def thinking(seq: int, text: str) -> Thinking:
    return Thinking(run_id="r1", seq=seq, step=1, text=text)


# --- what it saves ---------------------------------------------------------


async def test_adding_an_event_does_no_work():
    """The whole point: a step hands over an event and carries on.

    Against a database in another rack this is the difference between a step
    that waits three round trips per event and one that waits none.
    """
    recorder = Recorder()
    buffer = EventBuffer(recorder.write, recorder.publish, interval=999)

    for seq in range(1, 6):
        buffer.add(started(seq))

    assert recorder.writes == [], "nothing written yet"
    assert buffer.pending == 5


async def test_a_flush_writes_the_whole_batch_in_one_call():
    recorder = Recorder()
    buffer = EventBuffer(recorder.write, recorder.publish, interval=999)
    for seq in range(1, 6):
        buffer.add(started(seq))

    await buffer.flush()

    assert len(recorder.writes) == 1, "one round trip, not five"
    assert [event.seq for event in recorder.writes[0]] == [1, 2, 3, 4, 5]


async def test_a_full_batch_flushes_without_waiting_for_the_interval():
    """Bounds memory, and keeps one chatty run from building a batch so large
    that writing it becomes its own latency problem."""
    recorder = Recorder()
    buffer = EventBuffer(recorder.write, recorder.publish, interval=999, max_batch=3)

    for seq in range(1, 5):
        buffer.add(started(seq))
    await asyncio.sleep(0)  # let the scheduled flush run
    await asyncio.sleep(0)

    assert recorder.writes, "the batch filled and went on its own"


async def test_the_timer_flushes_without_anybody_asking():
    recorder = Recorder()
    buffer = EventBuffer(recorder.write, recorder.publish, interval=0.01)
    buffer.add(started(1))

    await asyncio.sleep(0.05)

    assert [event.seq for event in recorder.writes[0]] == [1]
    await buffer.aclose()


# --- what it must not trade away -------------------------------------------


async def test_events_are_published_only_after_they_are_durable():
    """The resume contract. `seq` is what a reconnecting client sends, so it
    must never see one that a later catch-up read cannot return.

    Publishing first would be faster still and would open exactly that gap: a
    reconnect landing in the window would skip the events in flight, and skip
    them permanently.
    """
    recorder = Recorder()
    buffer = EventBuffer(recorder.write, recorder.publish, interval=999)
    buffer.add(started(1))
    buffer.add(started(2))

    await buffer.flush()

    assert recorder.order == ["write:2", "publish:2"]


async def test_closing_writes_what_is_left():
    """A run that ends, crashes or is cancelled flushes on the way out."""
    recorder = Recorder()
    buffer = EventBuffer(recorder.write, recorder.publish, interval=999)
    buffer.add(started(1))

    await buffer.aclose()

    assert [event.seq for event in recorder.writes[0]] == [1]


async def test_closing_twice_is_harmless():
    recorder = Recorder()
    buffer = EventBuffer(recorder.write, recorder.publish, interval=999)
    buffer.add(started(1))

    await buffer.aclose()
    await buffer.aclose()

    assert len(recorder.writes) == 1


async def test_a_failed_write_still_reaches_the_watchers():
    """A live view that silently stops is worse than one showing events whose
    write has already failed -- and the failure is logged either way."""
    recorder = Recorder(fail=True)
    buffer = EventBuffer(recorder.write, recorder.publish, interval=999)
    buffer.add(started(1))

    await buffer.flush()

    assert recorder.published and recorder.published[0][0].seq == 1


async def test_flushing_nothing_does_nothing():
    recorder = Recorder()
    buffer = EventBuffer(recorder.write, recorder.publish, interval=999)

    await buffer.flush()

    assert recorder.order == []


# --- step rows -------------------------------------------------------------


async def test_step_rows_are_written_once_per_row_not_once_per_step():
    """Nothing reads a step row while the row is still running, so paying a
    round trip per step bought nothing."""
    written: list[list] = []

    async def write(rows: list) -> None:
        written.append(list(rows))

    rows = RowBuffer(write)
    for step in range(4):
        rows.add({"step_id": f"s{step}"})

    assert written == [], "nothing written mid-row"
    await rows.flush()

    assert len(written) == 1 and len(written[0]) == 4


async def test_a_row_buffer_flushed_twice_writes_once():
    written: list[list] = []

    async def write(rows: list) -> None:
        written.append(list(rows))

    rows = RowBuffer(write)
    rows.add({"step_id": "s1"})
    await rows.flush()
    await rows.flush()

    assert len(written) == 1
