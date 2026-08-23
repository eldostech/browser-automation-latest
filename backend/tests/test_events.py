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


def test_typescript_mirror_declares_the_same_event_types():
    """`frontend/src/lib/events.ts` is hand-maintained; this catches drift."""
    ts_path = Path(__file__).resolve().parents[2] / "frontend" / "src" / "lib" / "events.ts"
    source = ts_path.read_text(encoding="utf-8")

    declared = set(re.findall(r"type:\s*'([a-z_]+)'", source))
    missing = set(EVENT_TYPES) - declared
    assert not missing, f"TypeScript event mirror is missing: {sorted(missing)}"

    listed = re.search(r"EVENT_TYPES:\s*AgentEventType\[\]\s*=\s*\[(.*?)\]", source, re.DOTALL)
    assert listed, "EVENT_TYPES array not found in events.ts"
    assert set(re.findall(r"'([a-z_]+)'", listed.group(1))) == set(EVENT_TYPES)
