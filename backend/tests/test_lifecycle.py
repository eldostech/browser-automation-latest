"""`RunLifecycle` is the one place every run type announces its ending --
except the agent, which also announces its own through the graph's `finish`
node, because `run_agent_session` has to work standalone with no lifecycle
around it at all. `Terminal.announced` is the seam between those two facts:
set it and this class persists the terminal row without emitting a second,
possibly disagreeing, `run_finished` event. Leave it unset and every other
run type gets exactly the announcement this class exists to guarantee.
"""

from __future__ import annotations

import pytest

from lifecycle import RunLifecycle, Terminal

pytestmark = pytest.mark.anyio


class FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    def publish(self, run_id: str, event: object) -> None:
        self.published.append((run_id, event))


class FakeStore:
    def __init__(self) -> None:
        self.appended: list[object] = []
        self.finished: list[dict] = []

    async def append_event(self, event: object) -> None:
        self.appended.append(event)

    async def mark_started(self, run_id: str) -> None:
        return None

    async def finish_run(self, run_id: str, status: str, **fields) -> None:
        self.finished.append({"run_id": run_id, "status": status, **fields})


async def test_an_unannounced_terminal_emits_run_finished():
    store, bus = FakeStore(), FakeBus()
    async with RunLifecycle("r1", store, bus) as run:
        run.finish(Terminal(status="succeeded", steps=3))

    assert len(store.appended) == 1
    assert store.appended[0].status == "succeeded"
    assert len(store.finished) == 1


async def test_an_announced_terminal_persists_without_emitting_again():
    """The agent's own `finish` node already put `run_finished` on this same
    sink -- this is what stops the lifecycle putting a second one next to it
    with a status computed by a different rule."""
    store, bus = FakeStore(), FakeBus()
    async with RunLifecycle("r1", store, bus) as run:
        run.finish(Terminal(status="succeeded", steps=3, announced=True))

    assert store.appended == [], "a second run_finished was emitted"
    assert len(store.finished) == 1, "the terminal row must still be persisted"
    assert store.finished[0]["status"] == "succeeded"


async def test_a_crash_before_any_finish_call_still_announces_once():
    """An unhandled exception never reaches `run.finish(...)` at all -- the
    default, unannounced Terminal is the safety net that keeps a genuine crash
    from leaving a run the dashboard waits on forever."""
    store, bus = FakeStore(), FakeBus()
    with pytest.raises(RuntimeError):
        async with RunLifecycle("r1", store, bus):
            raise RuntimeError("boom")

    assert len(store.appended) == 1, "a crash must still announce exactly once"
    assert store.appended[0].error == "The run ended without producing an outcome."
    assert len(store.finished) == 1
    assert store.finished[0]["status"] == "failed"
