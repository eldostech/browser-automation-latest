"""Retroactively remove secrets from a database that already has them in it.

``redaction.py`` protects what is written from now on. It does nothing about
what is already stored, and two paths put real credentials there:

* a value typed into a page before the redactor knew about it, which lands in a
  tool argument, a tool result echoing it back, or an error quoting the call;
* a password pasted into the **task text** -- "Login with below credentials /
  User: x / Password : y" -- which becomes the run row, the `run_started`
  event, the audit entry and the first message sent to a model. The router
  refuses that now; rows written before it did are still sitting there.

This rewrites those rows in place. JSON keeps its structure, so timelines still
render: only the secret substrings change.

**Postgres, the database this actually runs on.** The previous version of this
script spoke sqlite, which the backend stopped using, so it could not clean
anything -- which is how credentials survived in a live deployment long enough
to be noticed on a screen. Connection details come from the same settings the
application uses, and ``DB_SCHEMA`` selects the schema.

Usage::

    # What is in there? Finds pasted credentials by shape. Writes nothing.
    python backend/scripts/purge_secrets.py --scan

    # Remove everything --scan found, by value. Still writes nothing.
    python backend/scripts/purge_secrets.py --pasted

    # Apply it.
    python backend/scripts/purge_secrets.py --pasted --apply

    # Or name values yourself, one per line, keeping them out of your history.
    python backend/scripts/purge_secrets.py --secrets-file leaked.txt --apply

A dry run is the default, deliberately: this edits rows in place and there is
no undo. ``--secret`` exists for a one-off and is visible in the process list
while it runs, so prefer ``--secrets-file``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redaction import PLACEHOLDER, Redactor, find_credentials  # noqa: E402

#: Every column that can hold free text somebody typed, with how to read it.
#: Not every text column in the schema -- an id, a status or an email is not a
#: place a password ends up, and rewriting one would break a join.
TEXT_COLUMNS: dict[str, tuple[str, tuple[str, ...]]] = {
    "runs": ("id", ("task", "summary", "error")),
    "usecases": ("id", ("description",)),
    "run_steps": ("id", ("locator", "error", "page_url")),
    "executions": ("id", ("error",)),
    "healing_memory": ("id", ("dom_context", "explanation")),
    "targets": ("id", ("description",)),
    "batches": ("id", ("error",)),
    "jobs": ("id", ("error",)),
}

#: JSON columns, same idea. The whole document is rewritten string by string.
JSON_COLUMNS: dict[str, tuple[str, tuple[str, ...]]] = {
    "events": ("run_id, seq", ("payload",)),
    "runs": ("id", ("options", "result")),
    "audit_log": ("id", ("detail",)),
    "usecase_versions": ("usecase_id, version", ("definition",)),
    "executions": ("id", ("inputs", "outputs")),
    "datasets": ("id", ("rows",)),
    "batches": ("id", ("input_rows",)),
    "jobs": ("id", ("payload",)),
}


async def connect(schema: str):
    import asyncpg

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import Settings

    settings = Settings()
    connection = await asyncpg.connect(
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )
    await connection.execute(f'set search_path to "{schema}"')
    return connection, settings


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------


async def scan(connection, schema: str) -> list[str]:
    """Credential values discoverable by shape, most common first.

    Only the *task* columns are scanned for shape, because that is where a
    person types prose. A value found there is then purged from everywhere,
    which is the point: the same string is in the events and the audit log
    under keys no pattern would recognise.
    """
    seen: dict[str, int] = {}
    rows = await connection.fetch(
        f'select task from "{schema}".runs where task is not null'
    )
    for row in rows:
        for found in find_credentials(row["task"] or ""):
            if found.is_secret:
                seen[found.value] = seen.get(found.value, 0) + 1

    also = await connection.fetch(
        f'select description from "{schema}".usecases where description is not null'
    )
    for row in also:
        for found in find_credentials(row["description"] or ""):
            if found.is_secret:
                seen[found.value] = seen.get(found.value, 0) + 1

    return [value for value, _ in sorted(seen.items(), key=lambda kv: -kv[1])]


async def count_rows(connection, schema: str, redactor: Redactor) -> dict[str, int]:
    """How many rows hold at least one of these values, per table.column."""
    counts: dict[str, int] = {}
    for table, (_, columns) in TEXT_COLUMNS.items():
        for column in columns:
            rows = await connection.fetch(
                f'select "{column}" as value from "{schema}"."{table}" '
                f'where "{column}" is not null'
            )
            hits = sum(1 for row in rows if redactor.text(row["value"]) != row["value"])
            if hits:
                counts[f"{table}.{column}"] = hits
    for table, (_, columns) in JSON_COLUMNS.items():
        for column in columns:
            rows = await connection.fetch(
                f'select "{column}"::text as value from "{schema}"."{table}" '
                f'where "{column}" is not null'
            )
            hits = sum(1 for row in rows if redactor.text(row["value"]) != row["value"])
            if hits:
                counts[f"{table}.{column}"] = hits
    return counts


# ---------------------------------------------------------------------------
# Rewriting
# ---------------------------------------------------------------------------


async def purge(connection, schema: str, redactor: Redactor, *, apply: bool) -> int:
    """Rewrite every affected row. Returns how many rows changed."""
    changed = 0

    for table, (key, columns) in TEXT_COLUMNS.items():
        keys = [part.strip() for part in key.split(",")]
        for column in columns:
            rows = await connection.fetch(
                f'select {key}, "{column}" as value from "{schema}"."{table}" '
                f'where "{column}" is not null'
            )
            for row in rows:
                before = row["value"]
                after = redactor.text(before)
                if after == before:
                    continue
                changed += 1
                if not apply:
                    continue
                where = " and ".join(f'"{name}" = ${i + 2}' for i, name in enumerate(keys))
                await connection.execute(
                    f'update "{schema}"."{table}" set "{column}" = $1 where {where}',
                    after,
                    *[row[name] for name in keys],
                )

    for table, (key, columns) in JSON_COLUMNS.items():
        keys = [part.strip() for part in key.split(",")]
        for column in columns:
            rows = await connection.fetch(
                f'select {key}, "{column}"::text as value from "{schema}"."{table}" '
                f'where "{column}" is not null'
            )
            for row in rows:
                before = row["value"]
                after = redactor.text(before)
                if after == before:
                    continue
                # Parsed and re-dumped rather than written as raw text, so a
                # rewrite that somehow produced invalid JSON fails here instead
                # of leaving a column nothing can read.
                try:
                    document = json.dumps(json.loads(after))
                except json.JSONDecodeError:
                    print(f"  ! skipped {table}.{column}: rewriting broke the JSON")
                    continue
                changed += 1
                if not apply:
                    continue
                where = " and ".join(f'"{name}" = ${i + 2}' for i, name in enumerate(keys))
                await connection.execute(
                    f'update "{schema}"."{table}" set "{column}" = $1::jsonb where {where}',
                    document,
                    *[row[name] for name in keys],
                )

    return changed


# ---------------------------------------------------------------------------
# Driving it
# ---------------------------------------------------------------------------


def values_from(args: argparse.Namespace) -> list[str]:
    values = list(args.secret or [])
    if args.secrets_file:
        text = Path(args.secrets_file).read_text(encoding="utf-8")
        values += [line.strip() for line in text.splitlines() if line.strip()]
    return values


async def run(args: argparse.Namespace) -> int:
    schema = args.schema or settings_schema()
    connection, _ = await connect(schema)
    try:
        values = values_from(args)
        if args.scan or args.pasted:
            discovered = await scan(connection, schema)
            if args.scan:
                print(f"{len(discovered)} credential value(s) found by shape:")
                for value in discovered:
                    print(f"  {_masked(value)}")
                if discovered:
                    redactor = Redactor(discovered)
                    print()
                    for where, hits in (await count_rows(connection, schema, redactor)).items():
                        print(f"{hits:>6}  rows in {where}")
                    print("\nRe-run with --pasted --apply to remove them.")
                return 0
            values += discovered

        if not values:
            print("Nothing to purge: give --secret, --secrets-file or --pasted.")
            return 1

        redactor = Redactor(values)
        print(f"purging {len(values)} value(s) from schema {schema!r}")
        for where, hits in (await count_rows(connection, schema, redactor)).items():
            print(f"{hits:>6}  rows in {where}")

        changed = await purge(connection, schema, redactor, apply=args.apply)
        if args.apply:
            print(f"\nrewrote {changed} row(s). Values now read {PLACEHOLDER}.")
            print("Rotate the credential anyway: it was stored, and backups predate this.")
        else:
            print(f"\n{changed} row(s) would change. Nothing written -- add --apply.")
        return 0
    finally:
        await connection.close()


def settings_schema() -> str:
    from config import Settings

    return Settings().db_schema


def _masked(value: str) -> str:
    """Enough to recognise, not enough to reuse -- this goes to a terminal."""
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]} ({len(value)} chars)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secret", action="append", help="a value to remove; repeatable")
    parser.add_argument("--secrets-file", help="a file of values, one per line")
    parser.add_argument(
        "--pasted",
        action="store_true",
        help="find credentials by shape in task text, then remove them everywhere",
    )
    parser.add_argument("--scan", action="store_true", help="report only; write nothing")
    parser.add_argument("--apply", action="store_true", help="actually rewrite rows")
    parser.add_argument("--schema", help="override DB_SCHEMA")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
