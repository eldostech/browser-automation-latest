"""Turning *doing* a task into *recording* one.

Playwright MCP can drive a browser; it cannot produce a use case. These tests
cover the two things that close that gap, and both exist because of failures
this codebase has already had.

`describe_element` resolves an ephemeral ref into the durable locator ladder a
step carries, and -- the part that is not bookkeeping -- decides `exact` from
what else is on the page and reports how many elements the result matches. A
real run failed on a page holding "+ Invite User" and, in the dialog it opens,
"Invite": Playwright reads a name as a substring, so the dialog's locator found
both and acted on the one the dialog was covering.

The marking tools replace inference with declaration. The design sized loop
detection as the hard part of distillation; an agent that says where a row
begins turns it into bookkeeping.
"""

from __future__ import annotations

import pytest

from agent import Marks, describe_element
from agent.tools import TOOLS
from snapshot import parse as parse_snapshot
from test_agent_tools import FakeMCP, session

pytestmark = pytest.mark.anyio


#: The page that broke a real run, as Playwright MCP renders it.
INVITE = """### Page
- Page URL: https://vendor.test/users
### Snapshot
```yaml
- generic [active] [ref=e1]:
  - heading "Users" [level=1] [ref=e2]
  - button "+ Invite User" [ref=e3]
  - dialog "Invite" [ref=e8]:
    - textbox "Email" [ref=e9]
    - button "Invite" [ref=e10]
```
"""

#: Three rows that look alike, which is the ordinary shape of a list page and
#: the ordinary source of an ambiguous locator.
LIST = """### Page
- Page URL: https://vendor.test/accounts
### Snapshot
```yaml
- generic [ref=e1]:
  - link "Open" [ref=e2]
  - link "Open" [ref=e3]
  - link "Open" [ref=e4]
```
"""

#: The shape of the real failure: a project card is a `generic` div with no
#: accessible name at all, one per project, and nothing else distinguishes
#: them in the tree. Unlike LIST's named "Open" links, there is no name for
#: position to even be a fallback from -- and a real replay found this exact
#: page rendering *zero* generic elements minutes later, proving the count
#: itself was never a stable property of the page.
GENERIC_WRAPPERS = """### Page
- Page URL: https://vendor.test/projects
### Snapshot
```yaml
- list [ref=e1]:
  - generic [ref=e2]
  - generic [ref=e3]
  - generic [ref=e4]
```
"""


def snap(text: str):
    return parse_snapshot(text)


# --- ref -> ladder ---------------------------------------------------------


def test_a_ref_becomes_the_ladder_a_recorded_step_would_carry():
    described = describe_element(snap(INVITE), "e3")

    assert described.role == "button"
    assert described.name == "+ Invite User"
    assert [(loc.strategy, loc.role, loc.name, loc.text) for loc in described.ladder] == [
        ("role", "button", "+ Invite User", None),
        ("text", None, None, "+ Invite User"),
    ]


def test_a_name_another_element_contains_is_recorded_as_exact():
    """The failure this is written from.

    "Invite" is a substring of "+ Invite User", so without `exact` the dialog's
    button and the page's are the same locator -- and the one behind the dialog
    is the one that gets found first.
    """
    described = describe_element(snap(INVITE), "e10")

    assert described.shadowed_by == ("+ Invite User",)
    assert described.ladder[0].exact is True
    assert described.ladder[1].exact is True, "the fallback must not undo it"
    assert not described.ambiguous, "exact is what makes it unambiguous"


def test_a_name_nothing_contains_stays_loose():
    """`exact` is not free: it also stops matching when a site adds a word."""
    assert describe_element(snap(INVITE), "e3").ladder[0].exact is False


def test_three_identical_rows_are_reported_as_ambiguous_now():
    """Four thousand rows later is the wrong time to find this out."""
    described = describe_element(snap(LIST), "e2")

    assert described.matches == 3
    assert described.ambiguous
    assert "matches 3 elements" in described.as_text()


def test_the_first_of_several_identical_matches_stays_honestly_unresolved():
    """`nth=0` is how the schema spells "no position given" -- deliberately,
    so an ambiguous rung with no explicit nth keeps refusing rather than
    silently acting on whichever element happens to load first. That means
    the ref that IS the first match cannot record its own position; this
    checks the message says so rather than claiming a fix it cannot make.
    """
    described = describe_element(snap(LIST), "e2")

    assert described.ladder[0].nth == 0, "e2 is the first of the three"
    text = described.as_text()
    assert "cannot tell apart" in text
    assert "first" in text


def test_the_second_and_third_of_identical_rows_get_a_usable_position():
    """These *do* get fixed: `nth` can address the 2nd match onward, so a ref
    that is not the first of its group now resolves instead of refusing."""
    second = describe_element(snap(LIST), "e3")
    assert second.ladder[0].nth == 1
    text = second.as_text()
    assert "2nd of them" in text
    assert "Acting on it is fine now" in text

    third = describe_element(snap(LIST), "e4")
    assert third.ladder[0].nth == 2
    assert "3rd of them" in third.as_text()


def test_an_unnamed_generic_wrapper_among_identical_siblings_is_unreliable():
    """The bug `nth` cannot fix: position among named duplicates ("Open" links,
    "Chat" buttons) is at least a real, if fragile, proxy for which one was
    meant. Position among anonymous `generic`/`group`/`none`/`presentation`
    wrappers is not -- there is no name for it to even be a fallback from, and
    a real replay found the exact same page exposing a different count of
    these elements minutes later. This must be flagged as categorically worse
    than an ordinary ambiguous match, not smoothed over by `nth` resolving.
    """
    described = describe_element(snap(GENERIC_WRAPPERS), "e3")

    assert described.matches == 3
    assert described.ambiguous
    assert described.unreliable
    assert described.ladder[0].nth == 1, "nth is still computed -- only the messaging changes"

    text = described.as_text()
    assert "not a real control" in text
    assert "is not something to rely on" in text
    assert "2nd of them" not in text, "the milder by-position note must not also fire"


def test_a_named_control_among_identical_siblings_is_not_flagged_unreliable():
    """The control case: `unreliable` must not fire just because a role happens
    to repeat -- only for the structural-role-and-no-name combination."""
    described = describe_element(snap(LIST), "e3")

    assert described.ambiguous
    assert not described.unreliable


def test_the_second_and_third_of_identical_rows_get_their_own_position():
    assert describe_element(snap(LIST), "e3").ladder[0].nth == 1
    assert describe_element(snap(LIST), "e4").ladder[0].nth == 2


def test_a_ref_the_page_is_not_showing_describes_nothing():
    described = describe_element(snap(INVITE), "e99")

    assert described.matches == 0
    assert described.ladder == []
    assert "not on the page" in described.as_text()


# --- the boundaries --------------------------------------------------------


def test_setup_can_only_end_once():
    marks = Marks()

    assert marks.setup_complete(3) == ""
    assert "already" in marks.setup_complete(9)
    assert marks.setup_ended_at == 3


def test_a_row_must_be_closed_before_another_opens():
    marks = Marks()
    marks.begin_row(4, "A-1001")

    assert "still open" in marks.begin_row(9, "A-1002")

    marks.end_row(8)
    assert marks.begin_row(9, "A-1002") == ""


def test_a_session_that_never_marked_a_row_cannot_be_distilled():
    """Enforced in code at the point `finish` is called, not asked for in a
    prompt. Without a row boundary there is no way to tell the sign-in that
    must run once from the work that must run four thousand times."""
    marks = Marks()
    assert "No row was recorded" in marks.unfinished()

    marks.begin_row(2, "A-1001")
    assert "never ended" in marks.unfinished()

    marks.end_row(6)
    assert marks.unfinished() == ""


def test_the_row_boundary_is_what_replaces_loop_detection():
    """Three records give three independent samples of the same shape, which
    is what distillation checks the steps against rather than inferring them."""
    marks = Marks()
    marks.setup_complete(5)
    for index, key in enumerate(("A-1001", "A-1002", "A-1003")):
        marks.begin_row(6 + index * 4, key)
        marks.end_row(9 + index * 4)

    assert [key for _, _, key in marks.rows] == ["A-1001", "A-1002", "A-1003"]
    assert marks.setup_ended_at == 5


# --- how far it got, for a stop that was not the model's own choice --------


def test_a_stop_before_any_row_says_so():
    """A budget exhausted during setup is a stop with nothing yet to show for
    it -- distinct from one that banked real rows, even though both currently
    map to the same "partial" status a run's own record shows."""
    marks = Marks()
    assert marks.progress_summary() == "Stopped during setup, before any row began."


def test_a_stop_mid_row_names_the_row_left_open():
    marks = Marks()
    marks.setup_complete(2)
    marks.begin_row(3, "A-1001")

    assert marks.progress_summary() == "0 row(s) completed; 'A-1001' was left open, unfinished."


def test_a_stop_after_completed_rows_counts_them():
    marks = Marks()
    marks.setup_complete(1)
    marks.begin_row(2, "A-1001")
    marks.end_row(3)
    marks.begin_row(4, "A-1002")
    marks.end_row(5)

    assert marks.progress_summary() == "2 row(s) completed."


# --- marking through a session --------------------------------------------


async def test_a_mark_never_reaches_the_browser():
    """They are ours. The model cannot tell, and does not need to."""
    async with await session(replies={"browser_snapshot": INVITE}) as tools:
        await tools.call("browser_snapshot")
        await tools.call("begin_row", {"key": "A-1"})
        before = len(tools.provider.calls)

        await tools.call("mark_as_output", {"ref": "e10", "column": "status"})

        assert len(tools.provider.calls) == before


async def test_marking_an_element_records_its_durable_locator():
    """The mark is made while the page it came from is still on screen, which
    is the whole reason to do this during recording rather than after."""
    async with await session(replies={"browser_snapshot": INVITE}) as tools:
        await tools.call("browser_snapshot")
        await tools.call("begin_row", {"key": "A-1"})
        result = await tools.call("mark_as_output", {"ref": "e10", "column": "status"})

    assert not result.is_error, result.text
    mark = tools.marks.entries[-1]
    assert mark.kind == "mark_as_output"
    assert mark.name == "status"
    assert mark.described is not None
    assert mark.described.ladder[0].exact is True


async def test_marking_an_ambiguous_element_is_refused_with_the_count():
    """Recording it would produce a step that can act on the wrong row."""
    async with await session(replies={"browser_snapshot": LIST}) as tools:
        await tools.call("browser_snapshot")
        await tools.call("begin_row", {"key": "A-1"})
        result = await tools.call("mark_as_output", {"ref": "e2", "column": "link"})

    assert result.is_error
    assert "3 elements" in result.text
    assert [m.kind for m in tools.marks.entries] == ["begin_row"]


async def test_marking_before_looking_says_so():
    """No snapshot means no page, and a ref that resolves against nothing."""
    async with await session() as tools:
        await tools.call("begin_row", {"key": "A-1"})
        result = await tools.call("mark_as_input", {"ref": "e4", "name": "account"})

    assert result.is_error


async def test_a_mark_is_tied_to_the_call_it_followed():
    """Position is correctness. A value has to be read on the page it was
    pointed at -- appending every reading to the end would read the first
    page's field after the browser had moved to the third."""
    async with await session(replies={"browser_snapshot": INVITE}) as tools:
        await tools.call("browser_snapshot")          # 1
        await tools.call("mark_setup_complete")       # 2
        await tools.call("begin_row", {"key": "A-1"})  # 3
        await tools.call("mark_as_output", {"ref": "e10", "column": "status"})  # 4

    assert [(m.kind, m.after_call) for m in tools.marks.entries] == [
        ("setup_complete", 2),
        ("begin_row", 3),
        ("mark_as_output", 4),
    ]


async def test_the_same_column_cannot_be_marked_twice():
    async with await session(replies={"browser_snapshot": INVITE}) as tools:
        await tools.call("browser_snapshot")
        await tools.call("begin_row", {"key": "A-1"})
        await tools.call("mark_as_output", {"ref": "e10", "column": "status"})
        again = await tools.call("mark_as_output", {"ref": "e3", "column": "status"})

    assert again.is_error
    assert "already been marked" in again.text


async def test_describe_element_answers_without_marking_anything():
    """Offered so the agent can look before it commits, which is the advice the
    tool's own description gives it."""
    async with await session(replies={"browser_snapshot": INVITE}) as tools:
        await tools.call("browser_snapshot")
        result = await tools.call("describe_element", {"ref": "e10"})

    assert not result.is_error
    assert 'role=button name="Invite" exact' in result.text
    assert tools.marks.entries == [], "looking is not marking"


async def test_the_marking_tools_are_offered_beside_the_browser_ones():
    """From the model's side there is no difference: it calls a tool and gets
    an answer. That ours never reach the browser is the session's business."""
    async with await session() as tools:
        names = {spec.name for spec in tools.tools}

    mark_tool_names = {name for name, t in TOOLS.items() if t.handler is not None}
    assert mark_tool_names <= names
    assert "browser_click" in names
    assert "browser_evaluate" not in names, "refusals are still removals"


async def test_a_mark_is_audited_like_any_other_call():
    """It is part of what happened, and a trajectory that omits the marks
    cannot be distilled by anything reading it afterwards."""
    async with await session(replies={"browser_snapshot": INVITE}) as tools:
        await tools.call("browser_snapshot")
        await tools.call("mark_setup_complete")

    assert [c.name for c in tools.calls] == ["browser_snapshot", "mark_setup_complete"]
    assert tools.calls[-1].action == "", "a mark is not a step"
    assert tools.calls[-1].ok


async def test_a_per_row_value_marked_outside_a_row_is_refused():
    """Found by a real session.

    A model marked an input and an output without ever opening a row.
    Distillation then had marks it could not place and a session that could not
    be saved. Being told at the moment of the mistake costs one turn; finding
    out at the end costs the whole session.
    """
    async with await session(replies={"browser_snapshot": INVITE}) as tools:
        await tools.call("browser_snapshot")
        result = await tools.call("mark_as_output", {"ref": "e10", "column": "status"})

    assert result.is_error
    assert "begin_row" in result.text
    assert tools.marks.entries == []


async def test_a_credential_may_be_marked_during_setup():
    """The exception, and it is not an inconsistency: a credential is typed
    once per batch, which is precisely outside a row."""
    async with await session(replies={"browser_snapshot": INVITE}) as tools:
        await tools.call("browser_snapshot")
        result = await tools.call("mark_as_secret", {"ref": "e9", "slot": "vendor"})

    assert not result.is_error, result.text
