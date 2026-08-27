"""Event schema round-trip, plus a guard that the TypeScript mirror is in sync."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from events import (
    EVENT_TYPES,
    ApprovalRequired,
    ApprovalResolved,
    ErrorEvent,
    RunFinished,
    RunStarted,
    Screenshot,
    Thinking,
    StepFinished,
    StepStarted,
    ToolCall,
    ToolResult,
    dump_event,
    parse_event,
)

SAMPLES = [
    RunStarted(run_id="r1", seq=1, task="do a thing", start_url="https://example.com",
               options={"max_steps": 10}, tools=["browser_navigate"]),
    Thinking(run_id="r1", seq=2, step=1, text="I will start by...", done=False),
    ToolCall(run_id="r1", seq=3, step=1, call_id="c1", name="browser_click",
             arguments={"ref": "e12"}, sensitive=True),
    ToolResult(run_id="r1", seq=4, step=1, call_id="c1", name="browser_click", ok=True,
               duration_ms=120, text="clicked", truncated=False, attempts=2),
    Screenshot(run_id="r1", seq=5, step=1, artifact_id="a1", url="/api/artifacts/a1",
               caption="after click", page_url="https://example.com/x"),
    StepStarted(run_id="r1", seq=6, step=1, step_id="s1", action="click",
                description="Sign in button", phase="setup"),
    StepFinished(run_id="r1", seq=7, step=1, step_id="s1", ok=True, duration_ms=88,
                 matched_locator='role=button name="Sign in"', locator_rung=0),
    ApprovalRequired(run_id="r1", seq=6, step=2, approval_id="ap1", call_id="c2",
                     name="browser_click", arguments={"element": "Pay now"},
                     reason="payment", categories=["payment"], expires_at="2026-01-01T00:00:00+00:00"),
    ApprovalResolved(run_id="r1", seq=7, step=2, approval_id="ap1", decision="approved", note="ok"),
    ErrorEvent(run_id="r1", seq=8, step=2, kind="tool_failed", message="boom", recoverable=True),
    RunFinished(run_id="r1", seq=9, status="succeeded", steps=2, duration_ms=4200,
                summary="done", result={"answer": "done", "data": [1, 2]}),
]


@pytest.mark.parametrize("event", SAMPLES, ids=[e.type for e in SAMPLES])
def test_round_trip_through_json(event):
    """dump -> JSON -> parse must reproduce the identical model."""
    payload = dump_event(event)
    restored = parse_event(json.loads(json.dumps(payload)))
    assert restored == event
    assert restored.type == event.type
    assert restored.seq == event.seq


def test_every_declared_type_has_a_sample():
    assert {e.type for e in SAMPLES} == set(EVENT_TYPES)


def test_discriminator_selects_the_right_model():
    parsed = parse_event(
        {"type": "tool_result", "run_id": "r", "seq": 1, "ts": "2026-01-01T00:00:00+00:00",
         "step": 1, "call_id": "c", "name": "n", "ok": False, "duration_ms": 1}
    )
    assert isinstance(parsed, ToolResult)
    assert parsed.ok is False


def test_unknown_type_is_rejected():
    with pytest.raises(Exception):
        parse_event({"type": "not_a_real_event", "run_id": "r", "seq": 1})


def _typescript_mirror() -> str:
    path = Path(__file__).resolve().parents[2] / "frontend" / "src" / "lib" / "events.ts"
    return path.read_text(encoding="utf-8")


def test_typescript_mirror_declares_the_same_event_types():
    """`frontend/src/lib/events.ts` is hand-maintained; this catches drift."""
    source = _typescript_mirror()

    declared = set(re.findall(r"type:\s*'([a-z_]+)'", source))
    missing = set(EVENT_TYPES) - declared
    assert not missing, f"TypeScript event mirror is missing: {sorted(missing)}"

    listed = re.search(r"EVENT_TYPES:\s*AgentEventType\[\]\s*=\s*\[(.*?)\]", source, re.DOTALL)
    assert listed, "EVENT_TYPES array not found in events.ts"
    assert set(re.findall(r"'([a-z_]+)'", listed.group(1))) == set(EVENT_TYPES)


def test_typescript_mirror_declares_the_same_fields():
    """Every field the backend emits must exist in the mirrored interface.

    The test above compares type *names*, which is the weaker half of the
    guarantee: adding a field to an existing event passed it unnoticed, and the
    frontend then reads `undefined` at runtime with TypeScript perfectly happy,
    because the interface it was checked against never mentioned the field.

    The stronger check is per-event. It is one-directional on purpose -- the
    mirror may carry extra fields (a few are computed client-side) but may not
    be missing any the server sends.

    Generating this file from the models instead would delete it outright,
    which is the better answer; it needs a Node type-generator invoked from the
    Python build, and that cross-ecosystem step has to work on every developer
    machine and in CI. Until that is worth its weight, this holds the property
    that actually matters.
    """
    from events import EVENT_MODELS

    source = _typescript_mirror()

    def fields_of(body: str) -> set[str]:
        """Property names, with `?` optional markers and comments ignored."""
        return set(re.findall(r"^\s*(\w+)\??\s*:", body, re.MULTILINE))

    # `export interface X extends BaseEvent { ... }` -- the extends clause is
    # optional, and the shared fields (run_id, seq, ts) live on the parent.
    blocks = dict(
        re.findall(
            r"export interface (\w+)(?:\s+extends\s+\w+)?\s*\{(.*?)\n\}", source, re.DOTALL
        )
    )
    inherited = fields_of(blocks.get("BaseEvent", ""))
    assert inherited, "BaseEvent not found in events.ts; the mirror's shape changed"

    problems: list[str] = []
    for event_type, model in sorted(EVENT_MODELS.items()):
        body = next(
            (b for b in blocks.values() if re.search(rf"type:\s*'{event_type}'", b)),
            None,
        )
        if body is None:
            problems.append(f"{event_type}: no interface declares it")
            continue

        declared = fields_of(body) | inherited
        for field in sorted(set(model.model_fields) - declared):
            problems.append(f"{event_type}.{field} is emitted but not declared in events.ts")

    assert not problems, (
        "the TypeScript event mirror has drifted from the models:\n  "
        + "\n  ".join(problems)
    )
