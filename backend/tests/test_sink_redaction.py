"""Redaction applied where it counts: the sink every event passes through.

The unit tests in ``test_redaction.py`` prove the pass works. These prove it is
actually wired in, and -- the part that is easy to get wrong -- that it covers
the WebSocket broadcast as well as the database. Redacting only on the way to
storage would still ship the password to every connected browser.
"""

from __future__ import annotations

from events import ToolCall, ToolResult
from redaction import PLACEHOLDER, Redactor
from runner import EventBus, RunEventSink

PASSWORD = "s3cret-Example-Pw!"


async def test_secrets_are_redacted_in_the_database(store):
    bus = EventBus()
    sink = RunEventSink("r1", store, bus, redactor=Redactor([PASSWORD]))

    await sink.emit(
        ToolCall(
            run_id="r1",
            seq=sink.reserve_seq(),
            step=1,
            call_id="c1",
            name="browser_fill_form",
            arguments={"fields": [{"name": "Password", "value": PASSWORD}]},
        )
    )

    stored = await store.get_events("r1")
    assert stored[0].arguments["fields"][0]["value"] == PLACEHOLDER


async def test_secrets_are_redacted_on_the_websocket_broadcast(store):
    bus = EventBus()
    queue = bus.subscribe("r1")
    sink = RunEventSink("r1", store, bus, redactor=Redactor([PASSWORD]))

    await sink.emit(
        ToolResult(
            run_id="r1",
            seq=sink.reserve_seq(),
            step=1,
            call_id="c1",
            name="browser_type",
            ok=True,
            duration_ms=3,
            text=f"typed {PASSWORD}",
        )
    )

    published = queue.get_nowait()
    assert PASSWORD not in published.text
    bus.unsubscribe("r1", queue)


async def test_the_raw_bytes_on_disk_never_contain_the_secret(store, tmp_path):
    """Defence in depth: grep the database file itself, not the parsed events."""
    sink = RunEventSink("r1", store, EventBus(), redactor=Redactor([PASSWORD]))
    await sink.emit(
        ToolCall(
            run_id="r1",
            seq=sink.reserve_seq(),
            step=1,
            call_id="c1",
            name="browser_type",
            arguments={"text": PASSWORD},
        )
    )
    # Checkpoint the WAL so everything is in the main database file.
    await store.db.execute("PRAGMA wal_checkpoint(FULL)")
    await store.db.commit()

    blob = store.db_path.read_bytes()
    assert PASSWORD.encode() not in blob


async def test_a_sink_without_secrets_stores_events_verbatim(store):
    sink = RunEventSink("r1", store, EventBus())
    await sink.emit(
        ToolCall(
            run_id="r1",
            seq=sink.reserve_seq(),
            step=1,
            call_id="c1",
            name="browser_type",
            arguments={"text": "ordinary text"},
        )
    )
    stored = await store.get_events("r1")
    assert stored[0].arguments["text"] == "ordinary text"
