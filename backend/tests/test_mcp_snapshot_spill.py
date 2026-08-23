"""Playwright MCP spills a large snapshot to a file and returns a link.

Left unhandled this is catastrophic and silent: the snapshot parser reads no
nodes from a link, so every ``role`` locator fails to resolve, every
snapshot-based assertion sees an empty page, and the model is shown a page with
nothing on it. Nothing errors -- it just cannot find anything.

Fixed in the MCP client so the agent loop and the replay executor both get the
tree inline, and treated as untrusted input because the path comes from tool
output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_client import MAX_SNAPSHOT_FILE_BYTES, MCPBrowserSession, MCPConfig
from snapshot import parse

TREE = """- generic [ref=e1]:
  - textbox "Full name" [ref=e2]
  - textbox "Work email" [ref=e3]
  - button "Request a demo" [ref=e4]
"""


def result_with_link(path: str) -> str:
    return (
        "### Ran Playwright code\n"
        "```js\nawait page.goto('https://example.com');\n```\n"
        "### Page\n"
        "- Page URL: https://example.com/contact\n"
        "### Snapshot\n"
        f"- [Snapshot]({path})\n"
    )


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spill = tmp_path / ".playwright-mcp"
    spill.mkdir()
    (spill / "page-1.yml").write_text(TREE, encoding="utf-8")
    return MCPBrowserSession(MCPConfig())


# --- the bug ---------------------------------------------------------------


def test_a_linked_snapshot_parses_to_nothing_without_the_fix():
    """The failure this guards against: silent, not an error."""
    assert len(parse(result_with_link(".playwright-mcp/page-1.yml"))) == 0


def test_the_linked_tree_is_spliced_back_in(session):
    inlined = session._inline_snapshot_links(result_with_link(".playwright-mcp/page-1.yml"))

    snapshot = parse(inlined)
    assert len(snapshot) == 4
    assert snapshot.locate("textbox", "Full name") is not None
    assert snapshot.page_url == "https://example.com/contact"


def test_the_windows_spelling_of_the_path_works(session):
    """The server writes a backslash path on Windows."""
    inlined = session._inline_snapshot_links(result_with_link(r".playwright-mcp\page-1.yml"))
    assert len(parse(inlined)) == 4


def test_text_without_a_link_is_returned_untouched(session):
    text = "### Page\n- Page URL: https://example.com\n### Snapshot\n```yaml\n- button \"Go\" [ref=e1]\n```"
    assert session._inline_snapshot_links(text) == text


def test_an_output_dir_is_searched_too(tmp_path, monkeypatch):
    """A server started with --output-dir writes somewhere else entirely."""
    monkeypatch.chdir(tmp_path)
    elsewhere = tmp_path / "artifacts"
    elsewhere.mkdir()
    (elsewhere / "page-9.yml").write_text(TREE, encoding="utf-8")

    session = MCPBrowserSession(MCPConfig(output_dir=str(elsewhere)))
    assert len(parse(session._inline_snapshot_links(result_with_link("page-9.yml")))) == 4


# --- the path comes from tool output, so it is untrusted -------------------


def test_a_path_escaping_the_working_directory_is_refused(session, tmp_path):
    secret = tmp_path.parent / "secret.yml"
    secret.write_text("- generic [ref=e1]\n", encoding="utf-8")

    link = result_with_link("../secret.yml")
    assert session._inline_snapshot_links(link) == link, "left as a link, not read"


def test_an_absolute_path_outside_the_directory_is_refused(session, tmp_path):
    outside = tmp_path.parent / "outside.yml"
    outside.write_text(TREE, encoding="utf-8")

    link = result_with_link(str(outside))
    assert session._inline_snapshot_links(link) == link


@pytest.mark.parametrize("name", ["page-1.txt", "page-1.exe", "passwd", "page-1.yml.exe"])
def test_only_yaml_files_are_read(session, tmp_path, name):
    (tmp_path / ".playwright-mcp" / name).write_text(TREE, encoding="utf-8")
    link = result_with_link(f".playwright-mcp/{name}")
    assert session._inline_snapshot_links(link) == link


def test_a_missing_file_leaves_the_link_alone(session):
    link = result_with_link(".playwright-mcp/does-not-exist.yml")
    assert session._inline_snapshot_links(link) == link


def test_an_oversized_file_is_not_read(session, tmp_path):
    big = tmp_path / ".playwright-mcp" / "huge.yml"
    big.write_bytes(b"- generic [ref=e1]\n" * (MAX_SNAPSHOT_FILE_BYTES // 10))
    link = result_with_link(".playwright-mcp/huge.yml")
    assert session._inline_snapshot_links(link) == link


# --- the shape a real result arrives in ------------------------------------


def test_several_links_in_one_result_are_all_inlined(session, tmp_path):
    (tmp_path / ".playwright-mcp" / "page-2.yml").write_text(
        '- button "Second" [ref=e9]\n', encoding="utf-8"
    )
    text = (
        result_with_link(".playwright-mcp/page-1.yml")
        + "\n### Another\n"
        + "- [Snapshot](.playwright-mcp/page-2.yml)\n"
    )
    snapshot = parse(session._inline_snapshot_links(text))
    assert snapshot.locate("button", "Second") is not None
    assert snapshot.locate("textbox", "Full name") is not None
