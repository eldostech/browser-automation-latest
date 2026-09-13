"""Turning an agent recovery's own give-up into something a person can act on.

`agent/operate.py` has no store -- the same reason `engine.py` cannot import
`llm` -- so a diagnosis it produces automatically (see `_propose_repair` there)
arrives here as a `RowResult.repair_proposal`, and `_persist_repair_proposal`
is where it either becomes a real draft version or is quietly dropped. These
tests are against that function directly, with a fake store: the interesting
behaviour is "does this call the store correctly and enrich the row's error",
not "does Postgres accept an insert", which `test_repair.py` already covers
for the pieces this reuses (`apply_fixes`, `is_unchanged`, `validate_patched`).
"""

from __future__ import annotations

import pytest

from engine import RowResult
from repair import PendingRepair, RepairProposal
from runner import _persist_repair_proposal
from snapshot import parse as parse_snapshot
from usecase import Locator, Step, UseCase

pytestmark = pytest.mark.anyio

PAGE = """### Page
- Page URL: https://example.com/contact
### Snapshot
```yaml
- generic "wrapper" [ref=e1]:
  - textbox "Your name" [ref=e2]
  - button "Request a demo" [ref=e3]
```"""


def use_case(**overrides) -> UseCase:
    base = dict(
        id="uc-1",
        name="Book a demo",
        status="ready",
        allowed_domains=["example.com"],
        row_steps=[
            Step(
                id="s10",
                action="fill",
                locators=[Locator(strategy="role", role="textbox", name="Full name")],
                value="Ada",
            ),
        ],
    )
    base.update(overrides)
    return UseCase(**base)


class FakeStore:
    """Records every call, and answers `get_usecase` with a fixed definition."""

    def __init__(self, definition: dict | None):
        self.definition = definition
        self.saved: list[dict] = []
        self.status_calls: list[tuple[str, str]] = []
        self.audits: list[dict] = []
        self.next_version = 5

    async def get_usecase(self, usecase_id: str) -> dict | None:
        return self.definition

    async def save_usecase(self, definition, *, created_by=None, created_by_id=None, owner_id=None):
        self.saved.append(definition)
        return definition["id"], self.next_version

    async def set_usecase_status(self, usecase_id: str, status: str) -> bool:
        self.status_calls.append((usecase_id, status))
        return True

    async def audit(self, action, **kwargs):
        self.audits.append({"action": action, **kwargs})


def a_proposal(**overrides) -> RepairProposal:
    base = dict(
        diagnosis="the field was renamed",
        fixes=[
            {
                "kind": "replace_locator",
                "step_id": "s10",
                "element_index": 0,
                "reason": "renamed",
            }
        ],
        confidence="high",
        tokens=900,
        usage={"input_tokens": 800, "output_tokens": 100},
    )
    base.update(overrides)
    return RepairProposal(**base)


def pending(**overrides) -> PendingRepair:
    return PendingRepair(proposal=a_proposal(**overrides), snapshot=parse_snapshot(PAGE))


async def test_no_pending_proposal_touches_nothing():
    store = FakeStore(definition=None)
    result = RowResult(ok=False, error="the recorded field could not be found")

    await _persist_repair_proposal(store, "uc-1", result, owner_id="u1")

    assert store.saved == []
    assert result.error == "the recorded field could not be found"


async def test_a_usable_proposal_becomes_a_draft_version():
    definition = use_case().model_dump(mode="json", by_alias=True)
    store = FakeStore(definition=definition)
    result = RowResult(ok=False, error="step 's10' could not find the field")
    result.repair_proposal = pending()

    await _persist_repair_proposal(store, "uc-1", result, owner_id="u1", owner_email="a@b.com")

    assert len(store.saved) == 1
    saved = store.saved[0]
    assert saved["status"] == "draft", "never applies unreviewed -- a draft, same as the button"
    assert saved["row_steps"][0]["locators"][0]["name"] == "Your name", "the fix leads"
    assert store.status_calls == [("uc-1", "draft")]
    assert store.audits[0]["action"] == "usecase.repair"
    assert store.audits[0]["detail"]["automatic"] is True

    assert "draft version 5" in result.error
    assert "the field was renamed" in result.error
    assert "step 's10' could not find the field" in result.error, "the original error is kept"


async def test_the_proposal_is_consumed_either_way():
    """Whether or not anything was saved, nothing downstream should see a
    pending proposal a second time."""
    definition = use_case().model_dump(mode="json", by_alias=True)
    store = FakeStore(definition=definition)
    result = RowResult(ok=False, error="x")
    result.repair_proposal = pending()

    await _persist_repair_proposal(store, "uc-1", result, owner_id="u1")

    assert result.repair_proposal is None


async def test_a_fix_that_changes_nothing_is_not_saved():
    definition = use_case().model_dump(mode="json", by_alias=True)
    store = FakeStore(definition=definition)
    result = RowResult(ok=False, error="x")
    # element_index 0 in PAGE is "Your name", the exact same locator the step
    # already carries -- prepending it changes nothing.
    result.repair_proposal = pending(
        fixes=[
            {
                "kind": "replace_locator",
                "step_id": "s10",
                "element_index": 0,
                "reason": "no-op",
            }
        ]
    )
    definition["row_steps"][0]["locators"] = [
        Locator(strategy="role", role="textbox", name="Your name").model_dump(exclude_none=True)
    ]
    store.definition = definition

    await _persist_repair_proposal(store, "uc-1", result, owner_id="u1")

    assert store.saved == []
    assert result.error == "x"


async def test_a_missing_use_case_is_a_quiet_no_op():
    """The use case could have been deleted between the row failing and this
    running; that is not this function's problem to raise about."""
    store = FakeStore(definition=None)
    result = RowResult(ok=False, error="x")
    result.repair_proposal = pending()

    await _persist_repair_proposal(store, "uc-1", result, owner_id="u1")

    assert store.saved == []
    assert result.error == "x"


# --- resolve_env: a base URL override must not pay for a query it never uses --


async def test_an_override_skips_the_target_lookup_entirely(root_store, db_settings):
    """`resolve_base_url` returns `override` outright and never looks at
    `targets` when one is given -- so querying `target_urls()` first, as
    every call used to, was a round trip spent on an answer already thrown
    away. A batch that pinned its base URL in `start_batch` paid for this a
    second time on every row-group; this proves it no longer does, by making
    the query raise if it is ever reached."""
    from bus import EventBus
    from runner import ReplayManager
    from usecase import UseCase

    manager = ReplayManager(root_store, db_settings, EventBus())
    workspace_id = await root_store.ensure_workspace("resolve-env test", "resolve-env-test")

    async def explode(*_a, **_k):
        raise AssertionError("target_urls() was called despite an override being given")

    manager.data(workspace_id).target_urls = explode  # type: ignore[method-assign]

    usecase = UseCase(id="u1", name="x", target="prod", allowed_domains=["example.com"])
    env = await manager.resolve_env(usecase, workspace_id, override="https://example.com/")

    assert env["base_url"] == "https://example.com"


async def test_with_no_override_the_lookup_still_runs(root_store, db_settings):
    """The other half of the same fix: when there is no override, targets are
    still read -- this is a short-circuit for one specific case, not a
    change to what a run with no override does."""
    from bus import EventBus
    from runner import ReplayManager
    from usecase import UseCase

    workspace_id = await root_store.ensure_workspace(
        "resolve-env test 2", "resolve-env-test-2"
    )
    await root_store.workspace(workspace_id).save_target(
        "prod", "https://targets.example.com"
    )
    manager = ReplayManager(root_store, db_settings, EventBus())
    usecase = UseCase(id="u1", name="x", target="prod", allowed_domains=["example.com"])

    env = await manager.resolve_env(usecase, workspace_id, override="")

    assert env["base_url"] == "https://targets.example.com"
