"""Find and repair stored use cases that can never succeed.

Distillation now rejects an assertion the domain allowlist makes impossible,
but use cases recorded before that check existed still carry them, and a use
case with an unsatisfiable assertion fails every single row while blaming the
page rather than itself.

This scans what is already stored and, with ``--apply``, saves a repaired
copy as a **new version**. Existing versions are never rewritten, so a batch
reading an older version is unaffected and the original stays inspectable.

Usage::

    python backend/scripts/repair_usecases.py               # report only
    python backend/scripts/repair_usecases.py --apply        # write new versions

Repairs made:

* an assertion whose ``unsatisfiable_reason`` fires is removed
* a session check in the same state is removed
* an input no step reads is removed

All three are cases where the thing removed could not have done anything
except cause a failure or demand a value that was then ignored.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from store import Store  # noqa: E402
from usecase import UseCase  # noqa: E402


def repair(definition: dict) -> tuple[dict, list[str]]:
    """Return a repaired copy plus the list of what changed."""
    use_case = UseCase.model_validate({**definition, "status": "draft"})
    notes: list[str] = []

    broken = {where for where, _ in use_case.impossible_assertions()}
    if broken:
        for where, why in use_case.impossible_assertions():
            notes.append(f"removed {where}: it {why}")

    patched = dict(definition)
    for phase in ("setup_steps", "row_steps", "teardown_steps"):
        patched[phase] = [s for s in (patched.get(phase) or []) if s.get("id") not in broken]
    if "session_check" in broken:
        patched["session_check"] = None

    # Re-derive which inputs anything still reads, after the removals above.
    rebuilt = UseCase.model_validate({**patched, "status": "draft"})
    referenced = {
        name for step in rebuilt.all_steps for kind, name in step.references() if kind == "input"
    }
    unused = [spec.name for spec in rebuilt.inputs if spec.name not in referenced]
    if unused:
        notes.append(f"removed input(s) no step reads: {', '.join(unused)}")
        patched["inputs"] = [i for i in (patched.get("inputs") or []) if i["name"] not in unused]

    if notes and not any(
        s.get("action") == "assert"
        for phase in ("setup_steps", "row_steps")
        for s in (patched.get(phase) or [])
    ):
        notes.append(
            "WARNING: nothing verifies a row any more. It will report success even when a row "
            "silently does nothing. Add an assertion before running a batch."
        )

    return patched, notes


async def main_async(db: Path, apply: bool) -> int:
    store = Store(db, db.parent / "artifacts")
    await store.connect()
    try:
        rows = await store.list_usecases(limit=500)
        if not rows:
            print("No use cases stored.")
            return 0

        repaired = 0
        for row in rows:
            definition = await store.get_usecase(row["id"])
            if definition is None:
                continue
            patched, notes = repair(definition)
            if not notes:
                print(f"[ok]     {row['id'][:8]}  {row['name']!r}")
                continue

            repaired += 1
            print(f"[repair] {row['id'][:8]}  {row['name']!r}")
            for note in notes:
                print(f"           - {note}")

            if apply:
                # Back to draft: the repair changed what it verifies, so a
                # person should look at it again before it runs unattended.
                patched["status"] = "draft"
                _, version = await store.save_usecase(patched, created_by="repair_usecases")
                await store.set_usecase_status(row["id"], "draft")
                print(f"           -> saved as v{version}, status draft")

        if repaired and not apply:
            print(f"\n{repaired} use case(s) need repair. Re-run with --apply to write them.")
        elif not repaired:
            print("\nNothing to repair.")
        return 0
    finally:
        await store.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "data" / "runs.db",
        help="path to runs.db (default: ./data/runs.db)",
    )
    parser.add_argument("--apply", action="store_true", help="write the repairs as new versions")
    args = parser.parse_args()

    if not args.db.is_file():
        print(f"No database at {args.db}", file=sys.stderr)
        return 2
    return asyncio.run(main_async(args.db, args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
