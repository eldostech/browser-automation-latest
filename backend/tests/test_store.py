"""Persistence: runs, events, replay-by-seq, artifacts, restart recovery."""

from __future__ import annotations

from events import RunFinished, Thinking, ToolCall


async def test_run_lifecycle_is_persisted(store):
    await store.create_run("r1", "do a thing", "https://example.com", {"max_steps": 5})

    run = await store.get_run("r1")
    assert run is not None and run.status == "pending"

    await store.mark_started("r1")
    assert (await store.get_run("r1")).status == "running"

    await store.finish_run(
        "r1", "succeeded", steps=4, duration_ms=1234, summary="ok", result={"answer": "ok", "data": [1]}
    )

    run = await store.get_run("r1")
    assert run.status == "succeeded"
    assert run.steps == 4
    assert run.result == {"answer": "ok", "data": [1]}
    assert run.finished_at is not None


async def test_events_replay_after_a_sequence_number(store):
    await store.create_run("r1", "t", None, {})
    for seq in range(1, 6):
        await store.append_event(ToolCall(run_id="r1", seq=seq, step=1, call_id=f"c{seq}", name="x"))

    assert len(await store.get_events("r1", after_seq=0)) == 5
    replayed = await store.get_events("r1", after_seq=3)
    assert [event.seq for event in replayed] == [4, 5]


async def test_thinking_events_upsert_on_the_same_seq(store):
    """Streaming rewrites one row rather than appending a bubble per delta."""
    await store.create_run("r1", "t", None, {})
    await store.append_event(Thinking(run_id="r1", seq=1, step=1, text="I am", done=False))
    await store.append_event(Thinking(run_id="r1", seq=1, step=1, text="I am thinking", done=False))
    await store.append_event(Thinking(run_id="r1", seq=1, step=1, text="I am thinking.", done=True))

    events = await store.get_events("r1")
    assert len(events) == 1
    assert events[0].text == "I am thinking."
    assert events[0].done is True


async def test_next_seq_continues_after_existing_events(store):
    await store.create_run("r1", "t", None, {})
    assert await store.next_seq("r1") == 1
    await store.append_event(ToolCall(run_id="r1", seq=7, step=1, call_id="c", name="x"))
    assert await store.next_seq("r1") == 8


async def test_events_are_isolated_per_run(store):
    await store.create_run("r1", "t", None, {})
    await store.create_run("r2", "t", None, {})
    await store.append_event(ToolCall(run_id="r1", seq=1, step=1, call_id="c", name="a"))
    await store.append_event(ToolCall(run_id="r2", seq=1, step=1, call_id="c", name="b"))

    assert [e.name for e in await store.get_events("r1")] == ["a"]
    assert [e.name for e in await store.get_events("r2")] == ["b"]


async def test_artifacts_are_written_to_disk_and_indexed(store):
    await store.create_run("r1", "t", None, {})
    record = await store.save_artifact("r1", b"\x89PNG-bytes", seq=3)

    from pathlib import Path

    assert Path(record.path).read_bytes() == b"\x89PNG-bytes"
    assert record.bytes == len(b"\x89PNG-bytes")

    fetched = await store.get_artifact(record.id)
    assert fetched is not None and fetched.run_id == "r1" and fetched.seq == 3
    assert [a.id for a in await store.list_artifacts("r1")] == [record.id]


async def test_filtering_and_counting_runs(store):
    await store.create_run("r1", "a", None, {})
    await store.create_run("r2", "b", None, {})
    await store.finish_run("r2", "failed", steps=1, duration_ms=10, error="nope")

    assert await store.count_runs() == 2
    assert await store.count_runs("failed") == 1
    assert [r.id for r in await store.list_runs(status="failed")] == ["r2"]


async def test_interrupted_runs_are_reaped_on_restart(store):
    """A backend crash must not leave a run 'running' forever."""
    await store.create_run("r1", "t", None, {})
    await store.mark_started("r1")
    await store.create_run("r2", "t", None, {})
    await store.finish_run("r2", "succeeded", steps=1, duration_ms=5)

    reaped = await store.reap_orphaned_runs()

    assert reaped == 1
    assert (await store.get_run("r1")).status == "failed"
    assert (await store.get_run("r2")).status == "succeeded"

    # The dashboard learns about it through the normal event stream.
    events = await store.get_events("r1")
    assert len(events) == 1
    assert isinstance(events[0], RunFinished)
    assert events[0].status == "failed"


# --- schema versioning -----------------------------------------------------


async def test_a_fresh_database_is_stamped_at_the_current_version(store):
    """SCHEMA already builds the final shape, so nothing should need migrating."""
    from store import SCHEMA_VERSION

    assert await store.schema_version() == SCHEMA_VERSION


async def test_migrate_is_idempotent(store):
    from store import SCHEMA_VERSION

    assert await store.migrate() == SCHEMA_VERSION
    assert await store.migrate() == SCHEMA_VERSION


async def test_an_unversioned_database_is_brought_forward(store):
    """A database written before user_version existed must still migrate."""
    from store import SCHEMA_VERSION

    await store.db.execute("PRAGMA user_version = 0")
    await store.db.commit()
    assert await store.schema_version() == 0

    assert await store.migrate() == SCHEMA_VERSION
    assert await store.schema_version() == SCHEMA_VERSION
