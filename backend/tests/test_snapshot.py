"""Tests for the accessibility-snapshot parser.

The fixture is real server output captured from ``data/runs.db``, not
hand-written -- the parser's whole value is that it handles what Playwright MCP
actually emits, including the messy parts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from snapshot import Snapshot, extract_ref, is_ref, parse

FIXTURE = Path(__file__).parent / "fixtures" / "snapshot_signin.txt"


@pytest.fixture(scope="module")
def real() -> Snapshot:
    return parse(FIXTURE.read_text(encoding="utf-8"))


# --- page metadata ---------------------------------------------------------


def test_page_url_and_title_are_extracted(real: Snapshot):
    assert real.page_url == "https://www.ixl.com/math/grade-5/multiply-decimals-using-area-models"
    assert real.page_title == "IXL | Multiply decimals using area models | 5th grade math"


# --- node parsing ----------------------------------------------------------


def test_parses_a_real_snapshot(real: Snapshot):
    assert len(real) > 50
    assert all(node.ref for node in real), "every indexed node must carry a ref"


def test_role_and_name_are_separated(real: Snapshot):
    node = real.get("e15")
    assert node is not None
    assert node.role == "button"
    assert node.name == "Open navigation menu"


def test_attributes_on_both_sides_of_the_ref_are_captured(real: Snapshot):
    # - button "Open navigation menu" [expanded] [active] [ref=e15] [cursor=pointer]
    node = real.get("e15")
    assert node is not None
    assert "expanded" in node.attrs
    assert "active" in node.attrs
    assert node.attrs.get("cursor") == "pointer"
    assert "ref" not in node.attrs, "ref is promoted to its own field"


def test_nodes_without_a_ref_are_skipped(real: Snapshot):
    # The fixture contains `- button "copy link"` with no ref -- nothing that
    # can be acted on, so it must not appear.
    assert all(n.name != "copy link" for n in real)


def test_property_lines_are_not_mistaken_for_nodes(real: Snapshot):
    assert all(n.role not in {"text", "url"} for n in real)


def test_trailing_text_content_is_captured(real: Snapshot):
    # - generic [ref=e102]: "0"
    node = real.get("e102")
    assert node is not None
    assert node.text == '"0"'


def test_depth_tracks_indentation(real: Snapshot):
    root = real.get("e1")
    nested = real.get("e12")
    assert root is not None and nested is not None
    assert nested.depth > root.depth


# --- the distillation direction: ref -> role + name ------------------------


def test_by_ref_resolves_an_ephemeral_ref_to_a_durable_description(real: Snapshot):
    node = real.by_ref["e107"]
    assert (node.role, node.name) == ("textbox", "answer")
    assert node.describe() == 'textbox "answer"'


def test_get_returns_none_for_an_unknown_ref(real: Snapshot):
    assert real.get("e999999") is None


# --- the replay direction: role + name -> live ref -------------------------


def test_locate_finds_a_control_by_role_and_name(real: Snapshot):
    node = real.locate("button", "Open navigation menu")
    assert node is not None and node.ref == "e15"


def test_locate_is_case_and_whitespace_insensitive(real: Snapshot):
    node = real.locate("button", "  open   NAVIGATION menu ")
    assert node is not None and node.ref == "e15"


def test_locate_falls_back_to_substring(real: Snapshot):
    node = real.locate("button", "navigation menu")
    assert node is not None and node.ref == "e15"


def test_locate_returns_none_when_nothing_matches(real: Snapshot):
    assert real.locate("button", "Definitely Not On This Page") is None


def test_locate_honours_nth(real: Snapshot):
    buttons = real.find("button")
    assert len(buttons) > 1
    assert real.locate("button", nth=1) is buttons[1]
    assert real.locate("button", nth=len(buttons) + 50) is None


def test_exact_match_wins_over_substring():
    snap = parse(
        """```yaml
- button "Submit the whole form" [ref=e1]
- button "Submit" [ref=e2]
```"""
    )
    node = snap.locate("button", "Submit")
    assert node is not None and node.ref == "e2"


def test_structural_wrappers_do_not_shadow_real_controls():
    snap = parse(
        """```yaml
- generic "Sign in" [ref=e1]:
  - button "Sign in" [ref=e2]
```"""
    )
    node = snap.locate("generic", "Sign in")
    assert node is not None and node.ref == "e1", "an explicit generic role still resolves"

    node = snap.locate("button", "Sign in")
    assert node is not None and node.ref == "e2"


# --- robustness ------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "not a snapshot at all", "### Page\n- Page URL: x"])
def test_unparseable_input_yields_an_empty_snapshot_rather_than_raising(text: str):
    snap = parse(text)
    assert len(snap) == 0


def test_a_truncated_snapshot_still_yields_the_nodes_it_contains():
    # Tool results are truncated at a character budget, so an unterminated
    # fence is a normal occurrence, not a corruption.
    snap = parse(
        """### Snapshot
```yaml
- textbox "Username" [ref=e17]
- textbox "Password" [ref=e21]
- button "Sign i"""
    )
    assert [n.ref for n in snap] == ["e17", "e21"]


def test_escaped_quotes_in_a_name_survive():
    snap = parse('```yaml\n- button "Say \\"hello\\"" [ref=e1]\n```')
    node = snap.get("e1")
    assert node is not None and node.name == 'Say "hello"'


def test_snapshot_without_a_fence_is_still_parsed():
    snap = parse('- textbox "Username" [ref=e17]\n- button "Sign in" [ref=e18]')
    assert len(snap) == 2
    assert snap.locate("button", "Sign in").ref == "e18"


def test_roles_histogram_reports_what_was_on_the_page(real: Snapshot):
    counts = real.roles()
    assert counts["button"] >= 3
    assert sum(counts.values()) == len(real)


# --- ref helpers -----------------------------------------------------------


@pytest.mark.parametrize(
    "value", ["ref=e15", "[ref=e15]", " ref=f1e42 ", "e15", " e407 "]
)
def test_is_ref_recognises_every_spelling(value: str):
    assert is_ref(value) is True
    assert extract_ref(value) in {"e15", "f1e42", "e407"}


def test_a_bare_ref_is_a_ref_not_a_selector():
    """This assertion used to say the opposite, and that was the bug.

    A bare ``e15`` was classed as a real selector, so it never reached the
    snapshot lookup -- it fell through to the text fallback and became
    ``text=e15``, a locator that cannot match anything. A recorded sign-in
    produced one of those for every step and failed on replay at the first
    field, which looked like the credentials were being lost.

    Playwright MCP's own tools take the ref bare, in a ``ref`` argument, so the
    model writes it that way into ``target`` too. All three spellings are the
    same thing and are now treated as such; whether it resolves is then decided
    by the snapshot, not by how it was written.
    """
    assert is_ref("e49") is True
    assert extract_ref("e49") == "e49"


@pytest.mark.parametrize(
    "value", ["button:has-text('Sign in')", "input[type='password']", "#name", "", "e2e", "email"]
)
def test_is_ref_rejects_real_selectors(value: str):
    assert is_ref(value) is False
