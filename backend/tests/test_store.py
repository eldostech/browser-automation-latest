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


async def test_creating_many_executions_at_once_matches_one_at_a_time(store):
    """A batch used to insert its execution rows one at a time, each a
    separate session/round trip -- on a remote database that scaled the wait
    before row 0 even started with the row count. `create_executions` is the
    replacement: one session, one bulk insert, the identical rows."""
    usecase_id, _ = await store.save_usecase(
        {
            "id": "uc00000000000000000000000000ex01",
            "name": "Bulk insert target",
            "status": "ready",
            "allowed_domains": ["vendor.test"],
            "row_steps": [
                {"id": "s1", "action": "click",
                 "locators": [{"strategy": "role", "role": "button", "name": "Go"}]}
            ],
        }
    )
    # No batch_id: a real one is a foreign key to a row in `batches`, which
    # this test has no reason to create just to prove a bulk insert works --
    # `run_id` alone is enough to identify these rows as one group.
    rows = [
        {
            "id": f"ex{i:030d}",
            "usecase_id": usecase_id,
            "version": 1,
            "run_id": "r-bulk-insert-test",
            "row_index": i,
            "inputs": {"n": i},
            "owner_id": None,
            "owner_email": "person@example.com",
        }
        for i in range(5)
    ]

    await store.create_executions(rows)

    saved = await store.list_executions(usecase_id=usecase_id)
    assert len(saved) == 5
    assert [row["row_index"] for row in saved] == [0, 1, 2, 3, 4]
    assert saved[0]["status"] == "pending"
    assert saved[3]["inputs"] == {"n": 3}
    assert saved[0]["owner_email"] == "person@example.com"


async def test_creating_no_executions_is_a_quiet_no_op(store):
    await store.create_executions([])
    assert await store.list_executions(batch_id="nothing-here") == []


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


async def test_interrupted_runs_are_reaped_on_restart(store, root_store):
    """A backend crash must not leave a run 'running' forever."""
    await store.create_run("r1", "t", None, {})
    await store.mark_started("r1")
    await store.create_run("r2", "t", None, {})
    await store.finish_run("r2", "succeeded", steps=1, duration_ms=5)

    # Reaping is deliberately unscoped -- it runs at startup, before any
    # request has established who is calling -- so it lives on the root store.
    reaped = await root_store.reap_orphaned_runs()

    assert reaped == 1
    assert (await store.get_run("r1")).status == "failed"
    assert (await store.get_run("r2")).status == "succeeded"

    # The dashboard learns about it through the normal event stream.
    events = await store.get_events("r1")
    assert len(events) == 1
    assert isinstance(events[0], RunFinished)
    assert events[0].status == "failed"


# --- schema ----------------------------------------------------------------
#
# The old ``user_version`` scheme is gone. It was a version stamp with no
# migrations behind it, so "the database is at v3" told you only that a build
# which knew about v3 had opened the file -- never that its shape was right.
# Alembic owns this now, and the two tests below assert what that stamp only
# implied.


async def test_every_row_belongs_to_a_workspace(root_store):
    """Nothing is reachable without passing through a tenant check.

    Asserted structurally rather than through behaviour, because the failure
    this guards against is a table added later without tenancy -- an omission
    that stays invisible until one customer sees another's data.

    A table qualifies one of two ways: it carries ``workspace_id`` itself, or
    every row of it hangs off a table that does. ``events`` and
    ``usecase_versions`` are the second kind -- they are scoped through their
    parent run or use case, which is also why they cascade on delete. That
    indirection is fine; having *neither* is not.
    """
    from db.base import Base

    #: Workspaces are the tenant, and a session is reached through its user.
    roots = {"workspaces", "user_sessions", "alembic_version"}
    scoped = {t.name for t in Base.metadata.sorted_tables if "workspace_id" in t.columns}

    def reaches_a_workspace(table) -> bool:
        return any(
            key.column.table.name in scoped | roots for key in table.foreign_keys
        )

    orphans = [
        table.name
        for table in Base.metadata.sorted_tables
        if table.name not in roots
        and table.name not in scoped
        and not reaches_a_workspace(table)
    ]
    assert not orphans, (
        f"these tables have no workspace_id and no path to one, so their rows "
        f"belong to nobody: {orphans}"
    )


# --- agent tool servers -----------------------------------------------------


async def test_a_tool_server_is_only_visible_to_its_own_workspace(root_store):
    """Two workspaces may both register a server called "crm" -- the same
    property `save_credential` already guarantees, and for the same reason."""
    a = root_store.workspace(await root_store.ensure_workspace("A", "tenant-a"))
    b = root_store.workspace(await root_store.ensure_workspace("B", "tenant-b"))

    await a.save_tool_server("s1", "crm", "stdio", {"command": "crm-mcp"})
    await b.save_tool_server("s2", "crm", "stdio", {"command": "other-mcp"})

    a_rows = await a.list_tool_servers()
    b_rows = await b.list_tool_servers()

    assert [r["name"] for r in a_rows] == ["crm"]
    assert a_rows[0]["connection"]["command"] == "crm-mcp"
    assert [r["name"] for r in b_rows] == ["crm"]
    assert b_rows[0]["connection"]["command"] == "other-mcp"

    # Cross-tenant delete must fail rather than succeed on the wrong row.
    assert await a.delete_tool_server("s2") is False
    assert [r["name"] for r in (await b.list_tool_servers())] == ["crm"]


async def test_saving_the_same_name_twice_replaces_it(store):
    await store.save_tool_server("s1", "crm", "stdio", {"command": "old"})
    await store.save_tool_server("s1", "crm", "stdio", {"command": "new"}, enabled=False)

    rows = await store.list_tool_servers()
    assert len(rows) == 1
    assert rows[0]["connection"]["command"] == "new"
    assert rows[0]["enabled"] is False


async def test_disabled_servers_are_excluded_when_asked(store):
    await store.save_tool_server("s1", "on", "stdio", {"command": "a"}, enabled=True)
    await store.save_tool_server("s2", "off", "stdio", {"command": "b"}, enabled=False)

    assert {r["name"] for r in await store.list_tool_servers()} == {"on", "off"}
    assert {r["name"] for r in await store.list_tool_servers(enabled_only=True)} == {"on"}


async def test_the_migrations_reproduce_the_models(db_settings):
    """Applying every migration to an empty schema yields exactly the models.

    Two failures at once: a model changed without a migration, and a migration
    that does not actually build what the models describe. Both are otherwise
    discovered in production as a missing column.

    Run against a scratch schema, migrated from nothing, because the test
    schema is built by ``create_all`` and has no Alembic history to check.
    """
    import os
    import subprocess
    import sys
    import uuid
    from pathlib import Path

    import asyncpg

    scratch = f"alembic_check_{uuid.uuid4().hex[:8]}"
    backend = Path(__file__).resolve().parent.parent
    env = {
        **os.environ,
        "DB_SCHEMA": scratch,
        "DB_HOST": db_settings.db_host,
        "DB_PORT": str(db_settings.db_port),
        "DB_NAME": db_settings.db_name,
        "DB_USER": db_settings.db_user,
        "DB_PASSWORD": db_settings.db_password,
    }

    def alembic(*args):
        return subprocess.run(
            [sys.executable, "-m", "alembic", *args],
            cwd=backend,
            capture_output=True,
            text=True,
            env=env,
        )

    try:
        upgrade = alembic("upgrade", "head")
        assert upgrade.returncode == 0, f"alembic upgrade failed:\n{upgrade.stderr}"

        check = alembic("check")
        assert check.returncode == 0, (
            "alembic check failed -- the models and the migrations disagree:\n"
            f"{check.stdout}\n{check.stderr}"
        )
    finally:
        conn = await asyncpg.connect(
            host=db_settings.db_host,
            port=db_settings.db_port,
            user=db_settings.db_user,
            password=db_settings.db_password,
            database=db_settings.db_name,
        )
        try:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{scratch}" CASCADE')
        finally:
            await conn.close()
