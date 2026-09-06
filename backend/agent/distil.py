"""Trajectory plus marks, into a draft use case.

This is the bridge the whole design turns on. Without it an authoring session
is a browser-using chatbot with an audit log; with it, one expensive session
becomes a document that replays four thousand times for nothing.

It is *bookkeeping*, and that is the achievement rather than a limitation. The
hard version of this problem -- read four hundred near-identical
sub-trajectories, work out where a record's work begins and which value varies
-- is one the agent was asked to answer while it still had the page in front of
it. `mark_setup_complete`, `begin_row` and `end_row` are that answer, so this
file cuts on declared boundaries instead of guessing at repeated shapes.

What each thing becomes:

===========================================  ===============================
everything before `mark_setup_complete`      ``setup_steps``, run once
the span between `begin_row` and `end_row`   ``row_steps``, run per row
second and later spans                       samples to check against
`mark_as_input(ref, "account")`              a declared input; the fill that
                                             typed it becomes a template
`mark_as_output(ref, "balance")`             an ``extract`` step, in place
`mark_as_secret(ref, "login")`               a credential slot, never a value
snapshots, waits, refusals, dead ends        dropped -- how it *found* the
                                             way, not the way
===========================================  ===============================

The one thing this refuses to do is guess. A mark that cannot be tied to a
call, a span that does not match the first, a value it cannot place: each is a
warning on the draft rather than a silently different use case.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urlsplit

from usecase import InputSpec, Locator, SecretSpec, Step, UseCase

from .marks import Mark, Marks
from .session import ToolCallRecord

log = logging.getLogger(__name__)

#: Arguments that carry the value an action types, per MCP tool.
VALUE_ARGUMENT: dict[str, str] = {
    "browser_type": "text",
    "browser_press_key": "key",
    "browser_select_option": "values",
}


@dataclass
class Draft:
    """A use case, and everything a reviewer needs to judge it.

    The warnings are not decoration. A draft distilled from a session that
    marked nothing, or whose second record took a different route, is still
    worth showing -- but it must not look like one that came out clean.
    """

    use_case: UseCase
    warnings: list[str] = field(default_factory=list)
    #: Values actually typed during recording, per input name. Used to verify
    #: the draft against the record it was recorded from, which is the only
    #: row we know the answer for.
    sample_inputs: dict[str, str] = field(default_factory=dict)
    #: How many complete records the session did. More than one is what turns
    #: "these steps worked once" into "these steps are the same every time".
    rows_recorded: int = 0


def distil(
    calls: Iterable[ToolCallRecord],
    marks: Marks,
    *,
    name: str,
    task: str = "",
    start_url: str = "",
    allowed_domains: Iterable[str] = (),
) -> Draft:
    """One session, as a draft use case."""
    calls = [call for call in calls if call.ok and not call.refused]
    warnings: list[str] = []

    problem = marks.unfinished()
    if problem:
        # Distilling anyway rather than refusing: a person can still read the
        # steps and fix the boundary by hand, and throwing the session away
        # loses the expensive part.
        warnings.append(problem)

    setup_end = marks.setup_ended_at
    if setup_end is None:
        setup_end = marks.rows[0][0] if marks.rows else 0
        warnings.append(
            "Setup was never marked complete, so nothing is treated as running "
            "once per batch. If this workflow signs in, the sign-in will run "
            "again for every row."
        )

    first_row = marks.rows[0] if marks.rows else (setup_end, 10**9, "")
    setup_calls = [c for c in calls if c.action and c.seq < setup_end]
    row_calls = [c for c in calls if c.action and first_row[0] < c.seq < first_row[1]]

    inputs, secrets, bindings, typed = _bindings(marks, calls, warnings)

    setup_steps = _steps(setup_calls, bindings, warnings)
    row_steps = _steps(row_calls, bindings, warnings)
    setup_steps, row_steps = _move_per_row_work(setup_steps, row_steps, warnings)
    outputs = _insert_reads(row_steps, row_calls, marks, first_row, warnings)

    if len(marks.rows) > 1:
        warnings.extend(_compare_spans(calls, marks, row_calls))
    elif marks.rows:
        warnings.append(
            "Only one record was recorded. The steps are what happened once; "
            "doing a second record would show that they are the same every "
            "time and that only the marked values differ."
        )

    if not row_steps:
        warnings.append("No steps fall inside the recorded row, so a batch would do nothing.")

    fields: dict[str, Any] = {
        "name": name,
        "description": task,
        "status": "draft",
        "authored_by": "agent",
        "base_url": _origin(start_url),
        "allowed_domains": sorted({_host(d) for d in allowed_domains if d}),
        "inputs": [InputSpec(name=n) for n in inputs],
        "secrets": [SecretSpec(name=n) for n in secrets],
        "setup_steps": setup_steps,
        "row_steps": row_steps,
        "outputs": outputs,
        "warnings": warnings,
    }
    use_case = _build(fields, warnings)
    return Draft(
        use_case=use_case,
        warnings=warnings,
        # What was actually typed, not the template it became. This is the row
        # the draft gets verified against, and a template would verify that
        # the site accepts the literal text "{{input.account}}".
        sample_inputs={name: typed[name] for name in inputs if name in typed},
        rows_recorded=len(marks.rows),
    )


def _build(fields: dict[str, Any], warnings: list[str]) -> UseCase:
    """The draft, or the most of it that can be built.

    Distillation must not raise. A session is the expensive part -- a model
    drove a browser for two minutes and a person watched it -- and losing all
    of that to a schema error at the last step is the worst possible way to
    spend it. A real session was lost exactly this way: the model named a
    column "Account number", which is the right answer to the question asked
    and not an identifier, and `InputSpec` refused it mid-construction.

    So a document that will not validate is rebuilt without the parts that
    would not, and the reviewer is told which. A draft missing an input is
    something a person can fix in a minute; a session that vanished is not.
    """
    try:
        return UseCase(**fields)
    except Exception as exc:  # noqa: BLE001 - the whole point is not to raise
        # Bound outside the handler: `except ... as` unbinds the name on the
        # way out, and the message below is the only account a reviewer gets.
        refusal = str(exc)
        log.warning("draft did not validate; degrading", extra={"error": refusal})

    warnings.append(
        "Some of what was recorded could not be turned into a valid use case "
        f"and was left out: {refusal}. Everything else is here, and the steps "
        "are worth reading before you decide whether to record it again."
    )
    # Dropped in the order that loses least. The steps are the recording; the
    # declared inputs and outputs are labels on it, and a reviewer can retype
    # a column name far more easily than a browser can redo the session.
    for give_up in ("inputs", "secrets", "outputs"):
        fields[give_up] = []
        try:
            return UseCase(**fields)
        except Exception:  # noqa: BLE001
            continue
    return UseCase(
        name=str(fields.get("name") or "Recorded by the agent"),
        status="draft",
        authored_by="agent",
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _steps(
    calls: list[ToolCallRecord],
    bindings: dict[int, tuple[str, str]],
    warnings: list[str],
) -> list[Step]:
    """The calls that became steps. One that cannot is named, not dropped
    silently -- a use case quietly missing a step is worse than one a reviewer
    is told about."""
    steps: list[Step] = []
    for call in calls:
        step = _step(call, bindings)
        if step is None:
            warnings.append(
                f"{call.name} at step {call.seq} could not be recorded: nothing "
                "says where it went, so a replay would have nowhere to go."
            )
            continue
        steps.append(step)
    return steps


def _move_per_row_work(
    setup: list[Step], row: list[Step], warnings: list[str]
) -> tuple[list[Step], list[Step]]:
    """A step that types a per-row value belongs in the row, wherever it fell.

    Models work in the order a person would: do the task, then say what the
    parts were. So the typing happens before `mark_setup_complete` and lands in
    setup -- and a setup step referencing `{{input.x}}` is refused by the
    schema, correctly, because setup runs once per batch and there is no row to
    take the value from.

    Rejecting the session over it would be pedantry. The mark is a *statement
    of fact* -- this value changes per record -- so the step that types it is
    row work by definition, and moving it is acting on what the agent said
    rather than guessing at what it meant. The alternative, seen once for real,
    is losing an otherwise perfect recording to an ordering convention.
    """
    stays, moves = [], []
    for step in setup:
        if any(kind == "input" for kind, _ in step.references()):
            moves.append(step)
        else:
            stays.append(step)
    if moves:
        warnings.append(
            f"{len(moves)} step(s) that type a per-row value were recorded "
            "before the row began, and have been moved into the row. Check "
            "they are in the right order -- everything left in setup runs once "
            "per batch."
        )
    return stays, [*moves, *row]


def _step(call: ToolCallRecord, bindings: dict[int, tuple[str, str]]) -> Step | None:
    """One recorded call as the step a replay will perform, or None.

    Every field is worked out *before* the Step is constructed. The schema
    validates on construction -- a navigate with no url is refused there -- so
    building one and then assigning to it raises on the way in, with a message
    about a field the caller has in its hand.
    """
    fields: dict[str, Any] = {
        "id": f"a{call.seq}",
        "action": call.action,
        "description": call.element,
        "locators": [Locator.model_validate(loc) for loc in call.locators],
    }

    if call.action == "navigate":
        # `browser_navigate` says where it is going; `browser_navigate_back`
        # does not, and the page it landed on is the only thing that does.
        # Without this a "go back to the list" step is unreplayable.
        url = str(call.arguments.get("url") or "") or call.page_url
        if not url:
            return None
        fields["url"] = url

    value = _value_of(call)
    if value is not None:
        # A bound value becomes a template: the point of the whole exercise is
        # that the next four thousand rows type something else.
        bound = bindings.get(call.seq)
        fields["value"] = bound[1] if bound else value

    return Step(**fields)


def _value_of(call: ToolCallRecord) -> str | None:
    key = VALUE_ARGUMENT.get(call.name)
    if key is None:
        return None
    raw = call.arguments.get(key)
    if isinstance(raw, list):
        # `browser_select_option` takes a list; a step records one choice.
        return str(raw[0]) if raw else None
    return None if raw is None else str(raw)


def _insert_reads(
    steps: list[Step],
    row_calls: list[ToolCallRecord],
    marks: Marks,
    span: tuple[int, int | None, str],
    warnings: list[str],
) -> list[str]:
    """Put an ``extract`` step where each value was pointed at.

    Position is correctness, not tidiness. Appending every reading to the end
    would read the first page's field after the browser had already navigated
    to the third -- the same mistake the codegen path had to fix, arrived at
    from the other direction.
    """
    reads = [
        mark
        for mark in marks.entries
        if mark.kind == "mark_as_output" and span[0] < mark.after_call < (span[1] or 10**9)
    ]
    outputs: list[str] = []
    for mark in sorted(reads, key=lambda m: -m.after_call):
        if mark.described is None or not mark.described.ladder:
            warnings.append(f"Could not record how to read {mark.name!r}; it was skipped.")
            continue
        # Counted over the calls that became steps, not over every call: a
        # dropped one would shift every reading after it by a page.
        before = sum(
            1
            for step, call in zip(steps, [c for c in row_calls], strict=False)
            if call.seq < mark.after_call
        )
        steps.insert(
            before,
            Step(
                id=f"x{mark.after_call}",
                action="extract",
                output=mark.name,
                description=mark.described.describe_first(),
                locators=list(mark.described.ladder),
            ),
        )
        outputs.append(mark.name)
    return list(reversed(outputs))


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _bindings(
    marks: Marks, calls: list[ToolCallRecord], warnings: list[str]
) -> tuple[list[str], list[str], dict[int, tuple[str, str]], dict[str, str]]:
    """Which calls typed a per-row value or a credential, and what replaces it.

    A mark names an element; the call that typed into it is the most recent one
    before the mark with the same ref. Searching backwards rather than matching
    on the typed value, because two fields can hold the same text and the ref
    is unambiguous.
    """
    inputs: list[str] = []
    secrets: list[str] = []
    bindings: dict[int, tuple[str, str]] = {}
    #: The literal text that was typed, per input name.
    typed: dict[str, str] = {}

    for mark in marks.entries:
        if mark.kind not in {"mark_as_input", "mark_as_secret"}:
            continue
        call = _typed_into(calls, mark)
        if call is None:
            warnings.append(
                f"{mark.name!r} was marked on {mark.ref}, but nothing recorded "
                "typing into that element -- so no step reads it. Mark the "
                "value on the field the moment after filling it."
            )
            continue
        if mark.kind == "mark_as_input":
            inputs.append(mark.name)
            bindings[call.seq] = (mark.name, "{{input.%s}}" % mark.name)
            value = _value_of(call)
            if value is not None:
                typed[mark.name] = value
        else:
            secrets.append(mark.name)
            bindings[call.seq] = (mark.name, "{{secret.%s}}" % mark.name)
            # Deliberately not collected. A credential's value belongs in the
            # vault, and the one place it must never end up is a sample kept
            # beside a draft use case.
    return inputs, secrets, bindings, typed


def _typed_into(calls: list[ToolCallRecord], mark: Mark) -> ToolCallRecord | None:
    for call in reversed([c for c in calls if c.seq < mark.after_call]):
        if call.action and str(call.arguments.get("target") or "") == mark.ref:
            return call
    return None


# ---------------------------------------------------------------------------
# More than one record
# ---------------------------------------------------------------------------


def _compare_spans(
    calls: list[ToolCallRecord], marks: Marks, first: list[ToolCallRecord]
) -> list[str]:
    """Check the later records took the same route as the first.

    This is the gift of asking the agent to do two or three. A second span that
    matches is evidence the steps really are the same every time; one that does
    not is a warning on the review screen, which is far better than a silent
    guess at which of the two shapes was meant.
    """
    shape = [(call.action, call.element) for call in first]
    problems: list[str] = []
    for index, (start, end, key) in enumerate(marks.rows[1:], start=2):
        later = [
            (c.action, c.element)
            for c in calls
            if c.action and start < c.seq < (end or 10**9)
        ]
        if later != shape:
            problems.append(
                f"Record {index} ({key!r}) did not take the same route as the "
                f"first: {len(shape)} steps then, {len(later)} now. The steps "
                "recorded are the first record's; check them against this one."
            )
    return problems


# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------


def _origin(url: str) -> str:
    parts = urlsplit(url or "")
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else ""


def _host(value: str) -> str:
    parts = urlsplit(value if "//" in value else f"//{value}")
    return (parts.netloc or value).split("@")[-1].split(":")[0] or value


__all__ = ["Draft", "distil"]
