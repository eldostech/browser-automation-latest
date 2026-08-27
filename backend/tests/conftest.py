"""Shared fixtures and fakes.

Nothing in the default test run touches the network, spawns a browser, or calls
an LLM. The end-to-end test in ``test_e2e_static.py`` is the single exception
and is opt-in via ``RUN_E2E=1``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import pytest

# The schema is bound into SQLAlchemy's MetaData when db.base is imported, so
# it has to be chosen before any application module is imported -- hence
# before the sys.path line below, not in a fixture.
os.environ.setdefault("DB_SCHEMA", os.environ.get("TEST_DB_SCHEMA", "browser_test"))

# The backend is a flat module tree, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import AgentSpec, RunOptions  # noqa: E402
from config import Settings  # noqa: E402
from events import AgentEvent  # noqa: E402
from llm import LLMTurn, ToolCallRequest  # noqa: E402
from mcp_client import MCPConfig, ToolOutcome  # noqa: E402
from store import Store  # noqa: E402


# ---------------------------------------------------------------------------
# Fake MCP session
# ---------------------------------------------------------------------------


@dataclass
class FakeTool:
    name: str
    description: str = "a fake browser tool"
    schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    #: Called with the tool arguments; returns a ToolOutcome or raises.
    handler: Callable[[dict[str, Any]], ToolOutcome] | None = None


class FakeMCPSession:
    """Implements the slice of :class:`mcp_client.MCPBrowserSession` the agent uses."""

    def __init__(self, tools: list[FakeTool] | None = None) -> None:
        self.config = MCPConfig(tool_timeout=5.0)
        self._tools = tools or [
            FakeTool("browser_navigate"),
            FakeTool("browser_snapshot"),
            FakeTool("browser_click"),
            FakeTool("browser_type"),
            FakeTool("browser_take_screenshot"),
        ]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    # -- discovery ------------------------------------------------------
    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self._tools]

    def anthropic_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.schema}
            for t in self._tools
        ]

    def find_tool(self, *candidates: str, contains: tuple[str, ...] = ()) -> str | None:
        names = self.tool_names
        for candidate in candidates:
            if candidate in names:
                return candidate
        for fragment in contains:
            for name in names:
                if fragment in name.lower():
                    return name
        return None

    # -- invocation -----------------------------------------------------
    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> ToolOutcome:
        self.calls.append((name, dict(arguments or {})))
        tool = next((t for t in self._tools if t.name == name), None)
        if tool is None:
            return ToolOutcome(name=name, text=f"unknown tool {name}", is_error=True)
        if tool.handler is not None:
            return tool.handler(dict(arguments or {}))
        if "screenshot" in name:
            return ToolOutcome(name=name, images=[("image/png", b"\x89PNG-fake")], duration_ms=3)
        return ToolOutcome(name=name, text=f"- Page URL: https://example.com\n- ok: {name}", duration_ms=5)


# ---------------------------------------------------------------------------
# Scripted LLM
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Replays a fixed list of turns; the last turn repeats if the loop runs on."""

    model = "scripted"

    def __init__(self, turns: list[LLMTurn], repeat_last: bool = True) -> None:
        self.turns = list(turns)
        self.repeat_last = repeat_last
        #: Message history as seen by the model on each turn.
        self.calls: list[list[dict[str, Any]]] = []
        #: Tool schema handed to the model on each turn.
        self.tool_schemas: list[list[dict[str, Any]]] = []

    async def run_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_text_delta=None,
        timeout: float | None = None,
    ) -> LLMTurn:
        self.calls.append([dict(m) for m in messages])
        self.tool_schemas.append(list(tools))
        if self.turns:
            turn = self.turns.pop(0) if len(self.turns) > 1 or not self.repeat_last else self.turns[0]
        else:
            turn = LLMTurn(text="done", raw_content=[{"type": "text", "text": "done"}])
        if on_text_delta and turn.text:
            await on_text_delta(turn.text)
        return turn


def tool_turn(name: str, arguments: dict[str, Any], call_id: str = "call-1", text: str = "") -> LLMTurn:
    """An assistant turn that requests one tool call."""
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    content.append({"type": "tool_use", "id": call_id, "name": name, "input": arguments})
    return LLMTurn(
        text=text,
        tool_calls=[ToolCallRequest(id=call_id, name=name, input=arguments)],
        stop_reason="tool_use",
        raw_content=content,
    )


def final_turn(text: str) -> LLMTurn:
    return LLMTurn(text=text, stop_reason="end_turn", raw_content=[{"type": "text", "text": text}])


# ---------------------------------------------------------------------------
# Fake sink / approval gate
# ---------------------------------------------------------------------------


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self._seq = 0
        self.screenshots: list[bytes] = []

    def reserve_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def emit(self, event: AgentEvent) -> None:
        # Mirror the store's upsert semantics so streamed `thinking` blocks
        # collapse to one entry, exactly as a client would see after replay.
        for index, existing in enumerate(self.events):
            if existing.seq == event.seq:
                self.events[index] = event
                return
        self.events.append(event)

    async def save_screenshot(self, data: bytes, *, seq: int, mime: str = "image/png"):
        self.screenshots.append(data)
        return f"artifact-{seq}", f"/api/artifacts/artifact-{seq}"

    def of_type(self, event_type: str) -> list[AgentEvent]:
        return [e for e in self.events if e.type == event_type]


class AutoApprovalGate:
    """Answers every approval request with a fixed decision."""

    def __init__(self, decision: Literal["approved", "rejected", "timeout"] = "approved") -> None:
        self.decision = decision
        self.requests: list[str] = []
        self.paused = 0
        self.resumed = 0

    async def request(self, approval_id: str, timeout: float):
        self.requests.append(approval_id)
        return self.decision, None

    async def on_pause(self) -> None:
        self.paused += 1

    async def on_resume(self) -> None:
        self.resumed += 1


class NeverApprovalGate:
    """Blocks until cancelled -- used to test the approval timeout path."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    async def request(self, approval_id: str, timeout: float):
        self.requests.append(approval_id)
        try:
            await asyncio.wait_for(asyncio.Event().wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return "timeout", None
        return "approved", None

    async def on_pause(self) -> None:
        pass

    async def on_resume(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def options() -> RunOptions:
    return RunOptions(
        max_steps=6,
        timeout_seconds=30.0,
        allowed_domains=["example.com", "*.example.com"],
        require_approval=True,
        approval_timeout_seconds=2.0,
        screenshot_every_step=False,
    )


@pytest.fixture
def spec(options: RunOptions) -> AgentSpec:
    return AgentSpec(run_id="run-test", task="Find the pricing page", options=options)


@pytest.fixture
def sink() -> RecordingSink:
    return RecordingSink()


@pytest.fixture
def mcp() -> FakeMCPSession:
    return FakeMCPSession()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
#
# Tests run against a real Postgres, in a schema of their own. There is no
# SQLite fallback, deliberately: the parts most worth testing -- JSONB round
# trips, ``FOR UPDATE SKIP LOCKED``, ``ON CONFLICT`` upserts, cascade
# deletes -- either behave differently on SQLite or do not exist there, so a
# passing SQLite suite would be evidence about a database nobody runs.
#
# The schema is created once per session and truncated between tests, which is
# far faster than create/drop per test and gives the same isolation.

TEST_SCHEMA = os.environ.get("TEST_DB_SCHEMA", "browser_test")


def test_settings(**overrides) -> Settings:
    """Settings for a test, pointed at the test schema.

    ``_env_file=None`` is the important part: without it pydantic-settings
    reads the developer's real ``.env``, which is how this project shipped the
    same bug three times (an API key, then a default model, then a repair
    model, each leaking from a developer machine into the suite).
    """
    return Settings(_env_file=None, db_schema=TEST_SCHEMA, **overrides)


@pytest.fixture(scope="session")
def db_settings() -> Settings:
    env = os.environ
    return test_settings(
        db_host=env.get("TEST_DB_HOST", env.get("DB_HOST", "localhost")),
        db_port=int(env.get("TEST_DB_PORT", env.get("DB_PORT", "5432"))),
        db_name=env.get("TEST_DB_NAME", env.get("DB_NAME", "postgres")),
        db_user=env.get("TEST_DB_USER", env.get("DB_USER", "postgres")),
        db_password=env.get("TEST_DB_PASSWORD", env.get("DB_PASSWORD", "")),
    )


@pytest.fixture(scope="session")
async def db_engine(db_settings: Settings):
    """Create the test schema once, then let go of the connection.

    The engine is disposed immediately rather than yielded. pytest-asyncio
    gives each test its own event loop, and an asyncpg connection belongs to
    the loop that opened it -- so a session-scoped engine handed to a
    function-scoped test raises "attached to a different loop". Fixtures
    therefore share a *schema*, not a connection pool.
    """
    from db.engine import create_all, create_engine, drop_all, ensure_schema

    engine = create_engine(db_settings)
    try:
        await ensure_schema(engine, db_settings.db_schema)
        await drop_all(engine)  # a previous crashed run may have left tables
        await create_all(engine)
    finally:
        await engine.dispose()
    return db_settings


@pytest.fixture(autouse=True)
async def clean_tables(db_engine):
    """Empty every table before each test.

    One ``TRUNCATE ... CASCADE`` for all of them: truncating them separately
    would fail on the foreign keys between them, and disabling the keys to work
    around that would stop the tests exercising the cascade rules.

    This is autouse and runs *before* the ``client`` fixture, which matters:
    the app's lifespan recreates the bootstrap workspace and administrator, so
    each API test starts with exactly one workspace, one admin, and no data.
    Without it, use cases pile up across tests and any assertion about "the
    list" depends on which tests ran first.

    A raw asyncpg connection, opened and closed here, for the same
    loop-affinity reason ``db_engine`` disposes of its own: this runs on the
    test's event loop and must not borrow a connection made on another.
    """
    import asyncpg

    from db.base import Base

    settings = db_engine  # the fixture returns the settings it prepared
    tables = ", ".join(
        f'"{TEST_SCHEMA}"."{table.name}"' for table in Base.metadata.sorted_tables
    )
    conn = await asyncpg.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database=settings.db_name,
    )
    try:
        await conn.execute(f"TRUNCATE {tables} RESTART IDENTITY CASCADE")
    finally:
        await conn.close()


@pytest.fixture
async def root_store(db_settings: Settings, db_engine, tmp_path: Path):
    """The unscoped Store: connection lifecycle and cross-tenant operations.

    Only the handful of tests about startup recovery or workspace creation
    want this one. Everything else wants ``store`` below.
    """
    store = Store(db_settings, tmp_path / "artifacts")
    await store.connect()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture
async def store(root_store: Store):
    """A workspace-scoped store -- what almost every test actually wants.

    The fixture is named ``store`` because that is what the tests have always
    called the thing they read and write through. What changed underneath is
    that it is now confined to one tenant, which is the point: a test cannot
    accidentally assert on another workspace's rows, because it has no object
    that can reach them.
    """
    workspace_id = await root_store.ensure_workspace("Test", "test")
    return root_store.workspace(workspace_id)


@pytest.fixture
async def workspace(store):
    """Alias for ``store``, for tests that want the scoping to be explicit."""
    return store


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

#: The bootstrap administrator's password in tests. Fixed, so a test can sign
#: in; long enough to satisfy the same policy production uses.
TEST_ADMIN_PASSWORD = "test-admin-password"
TEST_ADMIN_EMAIL = "admin@test.invalid"


def api_settings(db_settings: Settings, tmp_path: Path, **overrides) -> Settings:
    """Settings for an app under test: the test schema, a temp artifacts
    directory, and deterministic agent guardrails."""
    return test_settings(
        db_host=db_settings.db_host,
        db_port=db_settings.db_port,
        db_name=db_settings.db_name,
        db_user=db_settings.db_user,
        db_password=db_settings.db_password,
        artifacts_dir=str(tmp_path / "artifacts"),
        agent_allowed_domains=["example.com"],
        agent_screenshot_every_step=False,
        agent_max_steps=6,
        bootstrap_admin_email=TEST_ADMIN_EMAIL,
        bootstrap_admin_password=TEST_ADMIN_PASSWORD,
        # bcrypt's cost is the point in production and pure waste in a suite
        # that logs in on every test. 4 is the library minimum.
        auth_bcrypt_rounds=4,
        # Several pools are alive at once during an API test (the app's, and
        # any the test opens on its own loop). Keep each one small so a few
        # hundred tests cannot exhaust Postgres's connection limit.
        db_pool_size=2,
        db_max_overflow=0,
        **overrides,
    )


def login_as(test_client, email: str, password: str) -> str:
    response = test_client.post(
        "/api/auth/login", json={"email": email, "password": password}
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def authenticate(test_client, email: str = TEST_ADMIN_EMAIL, password: str = TEST_ADMIN_PASSWORD):
    """Attach a bearer token to every subsequent request from this client."""
    test_client.headers["Authorization"] = f"Bearer {login_as(test_client, email, password)}"
    return test_client


async def make_user(app, email: str, role: str, password: str = TEST_ADMIN_PASSWORD) -> str:
    """Create an account in the app's own workspace and return its email."""
    workspace_id = await app.state.store.default_workspace_id()
    await app.state.auth.create_user(
        workspace_id=workspace_id, email=email, password=password, role=role
    )
    return email


#: Stores opened by ``app_workspace`` during a test, closed when it ends.
_TEST_LOOP_STORES: list[Store] = []


async def app_workspace(app):
    """A scoped store the calling test can use, on the *test's* event loop.

    Deliberately not ``app.state.store``. ``TestClient`` runs the application
    -- including its lifespan, and therefore its connection pool -- on a portal
    event loop of its own. An ``async def`` test awaiting a connection created
    on that loop raises "got Future attached to a different loop", because an
    asyncpg connection belongs to the loop that opened it.

    Opening a second pool from the test's own loop is not a workaround for that
    but the natural consequence of the database being real and shared: two
    engines, two event loops, one Postgres. Rows written here are visible to
    the request handlers immediately, which is exactly what a second worker
    process will also see in production.
    """
    store = Store(app.state.settings)
    await store.connect()
    _TEST_LOOP_STORES.append(store)
    workspace_id = await store.default_workspace_id()
    assert workspace_id is not None, "the app's lifespan should have made a workspace"
    return store.workspace(workspace_id)


@pytest.fixture(autouse=True)
async def _close_test_loop_stores():
    """Dispose of any pool ``app_workspace`` opened, so Postgres connections do
    not accumulate across a few hundred tests."""
    yield
    while _TEST_LOOP_STORES:
        await _TEST_LOOP_STORES.pop().close()


def build_app(db_settings, tmp_path, monkeypatch, *, session_cls=None, **overrides):
    """Construct an app for an API test.

    Every API test module used to carry its own copy of this, differing only in
    which fake browser session it installed -- four copies that had already
    drifted in which settings they overrode.
    """
    import main
    import runner as runner_module

    # Bedrock is the only provider, so /healthz is made deterministic with
    # fake AWS credentials rather than by pinning a different one.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATESTONLY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-not-used")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    async def _fake_probe(config, timeout: float = 20.0):
        return {"ok": True, "transport": "stdio", "tool_count": 5, "tools": ["browser_snapshot"]}

    monkeypatch.setattr("routers.health.probe", _fake_probe)
    monkeypatch.setattr(main, "probe", _fake_probe)
    if session_cls is not None:
        monkeypatch.setattr(runner_module, "MCPBrowserSession", session_cls)

    return main.create_app(api_settings(db_settings, tmp_path, **overrides))
