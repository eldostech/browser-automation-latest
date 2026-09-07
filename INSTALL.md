# Installing TRACE from scratch

A complete, ordered walkthrough for standing this up on a machine that has
never run it before — a new laptop, a fresh VM, a different workspace. Follow
the steps in order; each one assumes the previous ones are done. Where a
step's reasoning matters, it says so in one line and points at `README.md`
for the full explanation — this document is the checklist, not the case for
it.

Total time on a normal connection: 15–25 minutes, most of it downloads.

## What you end up with

Two things running on your own machine, talking to a PostgreSQL you already
have:

- a **Python API** (FastAPI) on port 8000 — the backend, the database access,
  the browser automation itself
- a **web dashboard** (React, via Vite) on port 5173 — what you actually look
  at

No Docker is required. There is a compose file for people who would rather
not install Postgres locally, but it is an alternative, not the path this
guide follows.

---

## 0. Prerequisites

Install these first, then verify each one with the command shown.

| Requirement | Version | Check |
|---|---|---|
| **Python** | 3.11, 3.12, or 3.13 | `python --version` |
| **Node.js** | 18 or newer | `node --version` |
| **PostgreSQL** | 14+ (17/18 tested) | `psql --version` |
| **Git** | any recent | `git --version` |

Node is needed to build the dashboard, full stop. It is **not** needed to
record a workflow by hand — Playwright's Python package brings its own
browser — but it **is** needed if you turn on the optional AI agent later,
since that drives the browser through `npx @playwright/mcp`.

If any of these are missing:

- **Python** — <https://www.python.org/downloads/> (check "Add to PATH" on
  the Windows installer)
- **Node.js** — <https://nodejs.org/> (the LTS build)
- **PostgreSQL** — <https://www.postgresql.org/download/>, or your OS package
  manager (`brew install postgresql`, `apt install postgresql`)

You will also want **AWS credentials** for Claude on Amazon Bedrock — there
is no other model provider, and no separate API key to manage. If you already
have `aws configure` or SSO set up, you have everything this needs. Verify:

```bash
aws sts get-caller-identity
```

If that fails, the app still installs and runs — recording and replaying a
workflow by hand need no model at all — but AI mapping suggestions, the
optional AI agent, and self-healing will not work until credentials resolve.

---

## 1. Clone the repository

```bash
git clone https://github.com/eldostech/browser-automation-latest.git trace
cd trace
```

Everything below assumes you are in this directory (the repository root)
unless a step says otherwise.

---

## 2. Set up Python

```bash
python -m venv .venv
```

**Windows** (PowerShell or Git Bash):

```bash
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -r backend/requirements.txt
```

**macOS / Linux**:

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r backend/requirements.txt
```

From here on this guide writes the Windows path (`.venv/Scripts/...`) in
every command. On macOS/Linux, read that as `.venv/bin/...` throughout — it
is the only thing that differs.

**Use the virtual environment. Do not install into your system Python.**
These pins (`langchain-core`, `pydantic`, `boto3`) will fight anything else
on your machine that needs a different version.

This installs deterministic recording and replay — the whole product with no
AI agent involved. Leave it here unless you specifically want the agent; see
[step 9](#9-optional-turn-on-the-ai-agent).

---

## 3. Install the browser

```bash
.venv/Scripts/python -m playwright install chromium
```

A few hundred MB, one time. This is a Chromium that Playwright registered
itself, at the exact revision its Python package expects — a browser you
already have on the machine is not a substitute, and running this command
again when you already have the right build is free (it no-ops). One browser
serves both recording and replay, deliberately: whatever `playwright codegen`
writes is exactly what the replay engine has to parse, so they cannot drift
apart.

---

## 4. Point it at a database

You need a running PostgreSQL server and one database inside it — this does
not create the database for you, only its own schema inside one.

**If you already have a PostgreSQL server reachable**, either use its default
`postgres` database or make a dedicated one:

```bash
psql -U postgres -c "CREATE DATABASE trace"
```

**If you have no PostgreSQL server yet**, install it (see [step 0](#0-prerequisites))
and start it, then run the command above.

> Whichever database you point at, pick a **schema name nothing else has
> ever migrated** in the next step. The app creates and owns that schema
> entirely — if you reuse one another project's Alembic already stamped, the
> next step fails with `Can't locate revision identified by ...`. A fresh
> name always works.

**pgvector** (optional, needed only for healing memory — recalling past
fixes to a broken locator): check the server has the extension available:

```bash
psql -c "SELECT * FROM pg_available_extensions WHERE name = 'vector'"
```

A row back means you are set; the migration in step 6 installs it into your
schema for you. Nothing back means it is not on the server — install it
(Windows: the EDB installer's StackBuilder; macOS: `brew install pgvector`;
Debian/Ubuntu: `apt install postgresql-17-pgvector`, matching your server's
major version), or just set `HEALING_MEMORY_ENABLED=false` in the next step
and skip it — everything else works unchanged.

---

## 5. Configure the environment

Copy the template:

```bash
cp .env.example .env
```

Open `.env` and set these four things — everything else has a working
default:

```ini
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres            # or the database you created in step 4
DB_USER=postgres
DB_PASSWORD=your-actual-password
DB_SCHEMA=trace              # any name nothing else has migrated
```

Then generate an encryption key for stored site credentials (the sign-in
details a use case reuses on every run — unrelated to your own login to this
app):

```bash
.venv/Scripts/python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Paste the output into `.env`:

```ini
CREDENTIALS_KEY=the-key-you-just-generated
```

**Do not skip this.** Leaving it blank does not fail loudly — it silently
*disables* credential storage, so any use case that needs a sign-in has
nowhere to keep it. And once real credentials are stored under a key,
**losing that key makes them permanently unreadable** — back it up with your
other secrets, not in this repository.

Optionally, set who the first administrator will be (otherwise it defaults
to `admin@localhost`, which still works — you can change the email later):

```ini
BOOTSTRAP_ADMIN_EMAIL=you@example.com
```

Leave `BOOTSTRAP_ADMIN_PASSWORD` blank. A strong one is generated for you and
printed once, in step 7 — that is the intended way to get it, not something
you choose in this file.

Everything else in `.env.example` is documented inline, right above each
setting. `README.md`'s [Environment variables](README.md#environment-variables)
section groups the ones worth knowing about on day one.

---

## 6. Create the database tables

Run this **from `backend/`** — that is where `alembic.ini` lives:

```bash
cd backend
../.venv/Scripts/python -m alembic upgrade head
cd ..
```

This creates the schema named by `DB_SCHEMA`, every table in it, and the
`vector` extension if pgvector is available. You should see seven (or more)
migrations apply. If it stops with an error, check
[Troubleshooting](#troubleshooting) below before continuing.

---

## 7. Install the frontend

```bash
cd frontend
npm install
cd ..
```

---

## 8. Run it

Two terminals, both at the repository root.

**Terminal 1 — the backend**:

```bash
cd backend
../.venv/Scripts/python serve.py
```

Do **not** run this with plain `uvicorn main:app --reload` — on Windows that
combination breaks browser automation outright. `serve.py` exists
specifically to avoid that; see the comment at the top of that file, or
`README.md`'s [Running it](README.md#running-it) section, if you want the
mechanical reason.

Wait for `Application startup complete`, then check it from anywhere:

```bash
curl http://127.0.0.1:8000/healthz
```

`"status": "ok"` means the database is reachable and your AWS credentials
resolved. `"degraded"` tells you which of the two is not — read the body, it
names the failing piece.

**This is also where your one-time administrator password is printed.** Look
in this terminal's output for a line naming the bootstrap admin and a
password. It is shown exactly once and never again — only its hash is kept.
Copy it now.

**Terminal 2 — the dashboard**, in a second terminal at the repository root:

```bash
cd frontend
npm run dev
```

Open **http://localhost:5173**. Sign in with the email from step 5
(`BOOTSTRAP_ADMIN_EMAIL`, or `admin@localhost` if you left it unset) and the
password from Terminal 1's startup log.

You're running. The in-app **Help** button (top right) opens a full guide —
one track for using it day to day, one for how it is built.

---

## 9. (Optional) turn on the AI agent

Everything above gives you the complete product: record a workflow by hand,
replay it over a spreadsheet, self-heal it later — none of that needs this
step. Turn this on only if you also want to **record** a workflow by
describing it in plain English and watching an agent work it out.

```bash
.venv/Scripts/python -m pip install -r backend/requirements-agent.txt
```

This is a separate install on purpose — it pulls in `langgraph`, `langchain`
and an MCP client, which a deployment that only replays should never need.
You'll also want Node 18+ on `PATH` (already required for the frontend, so
if you got this far you have it), which is what `npx @playwright/mcp` runs
on.

Then in `.env`:

```ini
AGENT_ENABLED=true
```

Restart the backend (Terminal 1). "Describe it" now appears alongside "Do it
myself" when recording a new workflow.

---

## 10. (Optional) run the worker as its own process

By default, the backend process also claims and runs queued batches
(`WORKER_ENABLED=true`) — the right setting for a single-machine install,
since nothing else needs to be started. Split it out only once you're running
more than one machine, or want the API to answer requests even while a large
batch is mid-run elsewhere:

```ini
# in .env
WORKER_ENABLED=false
```

```bash
cd backend
../.venv/Scripts/python -m worker
```

---

## 11. Verify the install

Two independent checks, either is enough to trust the install:

**Run the automated test suite** (needs the same PostgreSQL from step 4; it
uses its own schema, so it will not touch the tables from step 6):

```bash
cd backend
../.venv/Scripts/python -m pytest -q
cd ..
```

Everything should pass (a handful of tests are skipped by design — they need
`RUN_E2E=1` or a real model and are opt-in).

**Or, do the thing by hand**: in the dashboard, click **Record**, do a
trivial task against any site (search a term, read the result), close the
recording window, and confirm a use case is produced. That exercises the
browser, the database, and the API all at once.

---

## Troubleshooting

The full list, with more detail on each, is in `README.md`'s
[Troubleshooting](README.md#troubleshooting) section. The ones you are most
likely to hit on a first install:

| Symptom | Cause | Fix |
|---|---|---|
| `Configured db_schema is 'x' but the models were built for 'y'` | A stray `DB_SCHEMA` in your shell environment is overriding `.env` | Unset it, or make them agree |
| `Can't locate revision identified by '...'` | `DB_SCHEMA` names a schema another project already migrated | Point `DB_SCHEMA` at a name nothing else has used |
| `The 'vector' extension is not available on this PostgreSQL server` | pgvector isn't installed server-side | Install it, or set `HEALING_MEMORY_ENABLED=false` |
| `Executable doesn't exist at ...` when a run starts | The Chromium binary is missing or the wrong revision | Re-run `python -m playwright install chromium` |
| Recording answers `501` | `RECORDER_ENABLED=false` — correct on a machine with no display | Record from a machine that has one, or leave replay-only |
| A batch is queued and nothing runs it | Nothing is claiming work: `WORKER_ENABLED=false` with no separate `python -m worker` running | Either flip `WORKER_ENABLED=true`, or start the worker (step 10) |
| `AccessDeniedException` on the model | Your AWS account lacks access to that Bedrock model | Request access in the Bedrock console for the model named in `LLM_REPAIR_MODEL` |
| Backend won't start; nothing obviously wrong in `.env` | `DB_PASSWORD` (or another value) has a special character breaking something downstream | Settings reads it correctly — see `README.md`'s note on why DB config is split into parts rather than one URL |

If you're stuck beyond this list, `GET /healthz?deep=1` on the backend
reports the state of every dependency (database, Bedrock, the model) in one
call — paste its output when asking for help.

---

## Where to go next

- **`README.md`** — the full picture: architecture, the data model, the
  workflow end to end, the API, deployment.
- **The in-app Help page** (`#/help` once signed in) — day-to-day usage, and
  a technical-architecture track for anyone extending this.
- **`docs/design/`** — the design documents behind specific decisions, plus
  `architecture.drawio` for an importable system diagram.
