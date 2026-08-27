"""Account administration from the command line.

The bootstrap administrator's password is printed once, to the log, on the boot
that creates it -- and only bcrypt output is stored, so it cannot be recovered
afterwards. This is the way back in when that message is lost, and the way to
create the first accounts for other people before the admin UI is used.

Usage::

    python backend/scripts/manage.py users
    python backend/scripts/manage.py add-user alice@example.com --role author
    python backend/scripts/manage.py reset-password admin@localhost
    python backend/scripts/manage.py set-role alice@example.com admin
    python backend/scripts/manage.py workspaces

Passwords are never taken as arguments -- an argument lands in shell history
and in the process list, where anyone on the machine can read it. Omit
``--password`` and a strong one is generated and printed; supply ``--ask`` to
be prompted without echo.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from auth.passwords import PasswordError, generate_password, hash_password  # noqa: E402
from auth.rbac import Role, is_valid_role  # noqa: E402
from auth.service import AuthError, AuthService  # noqa: E402
from config import get_settings  # noqa: E402
from db.base import utcnow  # noqa: E402
from db.models import User  # noqa: E402
from store import Store  # noqa: E402


async def _open() -> Store:
    store = Store(get_settings())
    await store.connect()
    return store


def _resolve_password(args) -> str:
    """A password from a prompt, or a generated one. Never from argv."""
    if args.ask:
        first = getpass.getpass("New password: ")
        if first != getpass.getpass("Repeat: "):
            raise SystemExit("The two passwords did not match.")
        return first
    generated = generate_password()
    print(f"Generated password: {generated}")
    return generated


async def cmd_users(args) -> int:
    store = await _open()
    try:
        auth = AuthService(store.sessions, get_settings())
        for workspace in await store.list_workspaces():
            print(f"\n{workspace['name']}  ({workspace['slug']})")
            users = await auth.list_users(workspace["id"])
            if not users:
                print("  (no accounts)")
            for user in users:
                state = "" if user["is_active"] else "  [disabled]"
                seen = user["last_login_at"] or "never"
                print(f"  {user['email']:<40} {user['role']:<10} last login {seen}{state}")
    finally:
        await store.close()
    return 0


async def cmd_workspaces(args) -> int:
    store = await _open()
    try:
        for workspace in await store.list_workspaces():
            print(f"{workspace['id']}  {workspace['slug']:<20} {workspace['name']}")
    finally:
        await store.close()
    return 0


async def cmd_add_user(args) -> int:
    store = await _open()
    try:
        settings = get_settings()
        auth = AuthService(store.sessions, settings)

        workspace_id = args.workspace or await store.default_workspace_id()
        if workspace_id is None:
            workspace_id = await store.ensure_workspace(
                settings.bootstrap_workspace_name, "default"
            )
            print(f"Created the first workspace ({settings.bootstrap_workspace_name}).")

        password = args.password or _resolve_password(args)
        try:
            user = await auth.create_user(
                workspace_id=workspace_id,
                email=args.email,
                password=password,
                role=args.role,
            )
        except AuthError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Created {user['email']} as {user['role']}.")
    finally:
        await store.close()
    return 0


async def cmd_reset_password(args) -> int:
    """Set a password directly, by email, across any workspace.

    Deliberately not routed through ``AuthService.reset_password``: that one is
    workspace-scoped because it serves a workspace admin over HTTP. Someone
    with shell access to the database server is already past that boundary, and
    needs this to work when they do not know which workspace the account is in.
    """
    store = await _open()
    try:
        settings = get_settings()
        password = args.password or _resolve_password(args)
        email = args.email.strip().lower()

        async with store.sessions() as session:
            user = await session.scalar(select(User).where(User.email == email))
            if user is None:
                print(f"error: no account for {email}", file=sys.stderr)
                return 1
            try:
                user.password_hash = hash_password(
                    password, rounds=settings.auth_bcrypt_rounds
                )
            except PasswordError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            # Signs out every existing session for this account.
            user.credentials_changed_at = utcnow()
            if args.activate:
                user.is_active = True
            await session.commit()
        print(f"Password reset for {email}. All existing sessions are signed out.")
    finally:
        await store.close()
    return 0


async def cmd_set_role(args) -> int:
    if not is_valid_role(args.role):
        print(
            f"error: unknown role {args.role!r}; expected one of "
            + ", ".join(r.value for r in Role),
            file=sys.stderr,
        )
        return 1

    store = await _open()
    try:
        email = args.email.strip().lower()
        async with store.sessions() as session:
            user = await session.scalar(select(User).where(User.email == email))
            if user is None:
                print(f"error: no account for {email}", file=sys.stderr)
                return 1
            was, user.role = user.role, args.role
            await session.commit()
        print(f"{email}: {was} -> {args.role}")
    finally:
        await store.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("users", help="list accounts by workspace").set_defaults(fn=cmd_users)
    sub.add_parser("workspaces", help="list workspaces").set_defaults(fn=cmd_workspaces)

    add = sub.add_parser("add-user", help="create an account")
    add.add_argument("email")
    add.add_argument("--role", default=Role.OPERATOR.value, choices=[r.value for r in Role])
    add.add_argument("--workspace", help="workspace id (default: the first one)")
    add.add_argument("--password", help=argparse.SUPPRESS)  # discouraged; see module docstring
    add.add_argument("--ask", action="store_true", help="prompt for the password")
    add.set_defaults(fn=cmd_add_user)

    reset = sub.add_parser("reset-password", help="set a password and sign out its sessions")
    reset.add_argument("email")
    reset.add_argument("--password", help=argparse.SUPPRESS)
    reset.add_argument("--ask", action="store_true", help="prompt for the password")
    reset.add_argument("--activate", action="store_true", help="also re-enable the account")
    reset.set_defaults(fn=cmd_reset_password)

    role = sub.add_parser("set-role", help="change an account's role")
    role.add_argument("email")
    role.add_argument("role", choices=[r.value for r in Role])
    role.set_defaults(fn=cmd_set_role)

    args = parser.parse_args()
    return asyncio.run(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
