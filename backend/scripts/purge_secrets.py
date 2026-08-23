"""Retroactively redact secrets from an existing run database.

The redaction pass in ``redaction.py`` protects events written from now on. It
does nothing for what is already on disk -- and real credentials were recorded
in plaintext before it existed, inside ``browser_fill_form`` arguments, tool
results, and the task text of runs where they were pasted in.

This rewrites those rows in place. Event payloads keep their structure, so the
timeline still renders; only the secret substrings change.

Usage::

    # See what would change. Nothing is written.
    python backend/scripts/purge_secrets.py --secret 's3cret-Example-Pw!' --dry-run

    # Find candidates first if you are not sure what leaked.
    python backend/scripts/purge_secrets.py --scan

    # Apply. Takes a backup next to the database unless --no-backup.
    python backend/scripts/purge_secrets.py --secret 's3cret-Example-Pw!'

Secrets can also be supplied one-per-line in a file via ``--secrets-file``,
which keeps them out of your shell history -- preferable, since a value passed
with ``--secret`` is visible in the process list while this runs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redaction import PLACEHOLDER, Redactor  # noqa: E402

#: Argument keys whose values are worth flagging in --scan. These are the field
#: names the model actually used when it typed credentials into a page.
_SUSPICIOUS_KEYS = re.compile(r"pass(word|wd)?|secret|token|api[-_]?key|credential", re.IGNORECASE)


def _walk_strings(node, path=""):
    """Yield ``(json_path, value)`` for every string anywhere in ``node``."""
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_strings(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _walk_strings(item, f"{path}[{index}]")


def scan(db_path: Path) -> int:
    """Report values that look like credentials, so an operator can confirm them.

    Deliberately reports rather than guesses: this prints what to pass to
    ``--secret`` instead of redacting on a heuristic, because a wrong guess
    here destroys legitimate event data.
    """
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    findings: dict[str, set[str]] = {}

    for row in connection.execute("SELECT run_id, seq, payload FROM events"):
        payload = json.loads(row["payload"])
        for path, value in _walk_strings(payload):
            if not value or len(value) < 4:
                continue
            # A value sitting under a password-ish key, or next to a "name"
            # sibling that says so -- the shape fill_form actually produces.
            if _SUSPICIOUS_KEYS.search(path):
                findings.setdefault(value, set()).add(f"{row['run_id'][:8]}#{row['seq']} {path}")

    # fill_form nests as {"fields": [{"name": "Password", "value": "..."}]},
    # where the key is "value" and only the sibling "name" identifies it.
    for row in connection.execute("SELECT run_id, seq, payload FROM events"):
        payload = json.loads(row["payload"])
        for field in _iter_form_fields(payload):
            name = str(field.get("name") or "")
            value = field.get("value")
            if isinstance(value, str) and value and _SUSPICIOUS_KEYS.search(name):
                findings.setdefault(value, set()).add(
                    f"{row['run_id'][:8]}#{row['seq']} field {name!r}"
                )

    connection.close()

    if not findings:
        print("No likely credentials found.")
        return 0

    print(f"{len(findings)} candidate secret(s) found:\n")
    for value, locations in sorted(findings.items(), key=lambda kv: -len(kv[1])):
        shown = sorted(locations)
        print(f"  {value!r}  ({len(locations)} occurrence(s))")
        for location in shown[:4]:
            print(f"      {location}")
        if len(shown) > 4:
            print(f"      ... and {len(shown) - 4} more")
    print("\nRe-run with --secret '<value>' (repeatable) to redact these.")
    return len(findings)


def _iter_form_fields(node):
    if isinstance(node, dict):
        fields = node.get("fields")
        if isinstance(fields, list):
            for field in fields:
                if isinstance(field, dict):
                    yield field
        for value in node.values():
            yield from _iter_form_fields(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_form_fields(item)


def purge(db_path: Path, secrets: list[str], *, dry_run: bool, backup: bool) -> int:
    redactor = Redactor(secrets)
    if not redactor.active:
        print("No usable secrets given (values shorter than 4 characters are ignored).")
        return 1

    if backup and not dry_run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = db_path.with_name(f"{db_path.name}.{stamp}.bak")
        shutil.copy2(db_path, destination)
        print(f"Backup written to {destination}")

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row

    changed_events = 0
    for row in connection.execute("SELECT run_id, seq, payload FROM events"):
        payload = row["payload"]
        cleaned = json.dumps(redactor.structure(json.loads(payload)))
        if cleaned == payload:
            continue
        changed_events += 1
        if not dry_run:
            connection.execute(
                "UPDATE events SET payload=? WHERE run_id=? AND seq=?",
                (cleaned, row["run_id"], row["seq"]),
            )

    # The runs table holds the task text and the final summary/result, which is
    # where a pasted credential also lands.
    changed_runs = 0
    for row in connection.execute("SELECT id, task, summary, result, error FROM runs"):
        updates = {
            column: redactor.text(row[column])
            for column in ("task", "summary", "result", "error")
            if isinstance(row[column], str) and redactor.text(row[column]) != row[column]
        }
        if not updates:
            continue
        changed_runs += 1
        if not dry_run:
            assignments = ", ".join(f"{column}=?" for column in updates)
            connection.execute(
                f"UPDATE runs SET {assignments} WHERE id=?", (*updates.values(), row["id"])
            )

    if dry_run:
        connection.close()
        print(
            f"[dry run] would redact {changed_events} event row(s) "
            f"and {changed_runs} run row(s). Nothing was written."
        )
        return 0

    connection.commit()
    # Order matters. The rewritten rows are sitting in the write-ahead log, and
    # the *old* pages are still in the main file. Checkpoint first to fold the
    # new pages in, then VACUUM to rewrite the file without the freed old ones,
    # then checkpoint again to drain what VACUUM itself logged.
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.execute("VACUUM")
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()

    print(
        f"Redacted {changed_events} event row(s) and {changed_runs} run row(s). "
        f"Replacement token: {PLACEHOLDER}"
    )
    return verify(db_path, secrets)


def verify(db_path: Path, secrets: list[str]) -> int:
    """Grep the raw database files. Returns 0 only if no secret survives.

    Reading rows back would prove nothing: a checkpoint that could not run
    leaves the old plaintext in freed pages, where it is invisible to SQL and
    perfectly visible to anyone with the file. This is the only check that
    actually answers the question, so the script's exit code depends on it and
    not on the number of rows updated.
    """
    survivors: dict[str, list[str]] = {}
    for suffix in ("", "-wal", "-shm"):
        path = db_path.with_name(db_path.name + suffix)
        if not path.is_file():
            continue
        blob = path.read_bytes()
        for secret in secrets:
            if secret.encode("utf-8") in blob:
                survivors.setdefault(secret, []).append(path.name)

    if not survivors:
        print("Verified: no secret bytes remain in the database files.")
        return 0

    print("\n*** VERIFICATION FAILED -- plaintext is still on disk ***", file=sys.stderr)
    for secret, files in survivors.items():
        preview = secret[:2] + "..." + secret[-2:] if len(secret) > 6 else "<short>"
        print(f"  {preview} still present in: {', '.join(files)}", file=sys.stderr)
    print(
        "\nAlmost always this means the backend is running and holding the\n"
        "write-ahead log open, so the checkpoint could not complete.\n"
        "Stop the backend, then re-run this command.",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "data" / "runs.db",
        help="path to runs.db (default: ./data/runs.db)",
    )
    parser.add_argument("--secret", action="append", default=[], help="a value to redact (repeatable)")
    parser.add_argument("--secrets-file", type=Path, help="file of secrets, one per line")
    parser.add_argument("--scan", action="store_true", help="report likely credentials and exit")
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    parser.add_argument("--no-backup", action="store_true", help="skip the .bak copy")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="grep the raw database files for the given secrets and exit",
    )
    args = parser.parse_args()

    if not args.db.is_file():
        print(f"No database at {args.db}", file=sys.stderr)
        return 2

    if args.scan:
        scan(args.db)
        return 0

    secrets = list(args.secret)
    if args.secrets_file:
        secrets += [
            line.strip()
            for line in args.secrets_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]

    if not secrets:
        parser.error("give at least one --secret / --secrets-file, or use --scan")

    if args.verify_only:
        return verify(args.db, secrets)

    return purge(args.db, secrets, dry_run=args.dry_run, backup=not args.no_backup)


if __name__ == "__main__":
    raise SystemExit(main())
