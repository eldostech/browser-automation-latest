"""The prompt files in ``backend/prompts/``.

Prompts are the part of this system most likely to be edited casually, so
these tests hold the load-bearing parts in place: that every prompt the code
asks for exists, that placeholders on disk match what the code actually
supplies, and that nothing renders with a leftover ``$name`` in it.

The agent's prompts went with the agent. What is left is what a model is still
asked to do -- repair a broken locator -- plus the message the allowlist shows
when it refuses a navigation.
"""

from __future__ import annotations

import pytest

import prompt_loader
from prompt_loader import (
    HEAL,
    HEAL_REQUEST,
    NAVIGATION_BLOCKED,
    PROMPTS_DIR,
    REQUIRED_PROMPTS,
    PromptNotFound,
    available,
    load,
    placeholders,
    render,
)


# --- the files exist and nothing is orphaned -------------------------------


def test_every_required_prompt_exists():
    missing = [name for name in REQUIRED_PROMPTS if not (PROMPTS_DIR / f"{name}.md").is_file()]
    assert not missing, f"missing prompt files: {missing}"


def test_no_orphan_prompt_files():
    """A file nothing loads is dead weight; a rename that misses one is a bug."""
    assert set(available()) == set(REQUIRED_PROMPTS)


def test_no_prompt_is_empty():
    for name in REQUIRED_PROMPTS:
        assert load(name).strip(), f"{name}.md is empty"


def test_missing_prompt_raises_a_helpful_error():
    with pytest.raises(PromptNotFound) as excinfo:
        load("definitely_not_a_prompt")
    message = str(excinfo.value)
    assert "definitely_not_a_prompt" in message
    assert HEAL in message  # lists what *is* available


# --- placeholders match what the code supplies -----------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        (NAVIGATION_BLOCKED, {"url", "allowlist"}),
    ],
)
def test_placeholders_match_the_call_sites(name, expected):
    """If someone adds a $placeholder to a file, render() would raise at
    runtime. This fails in CI instead."""
    assert placeholders(name) == expected


# --- rendering -------------------------------------------------------------


def test_missing_placeholder_raises_rather_than_leaking_a_dollar_sign():
    """A raw '$task' reaching the model is far worse than a loud failure."""
    with pytest.raises(KeyError) as excinfo:
        render(HEAL_REQUEST)
    assert "step" in str(excinfo.value)


def test_the_navigation_message_renders_without_leftovers():
    """A raw `$url` reaching a user is a bug they cannot act on."""
    rendered = render(NAVIGATION_BLOCKED, url="https://x.test", allowlist="example.com")
    assert "$" not in rendered
    assert "https://x.test" in rendered


def test_reload_picks_up_an_edited_file(tmp_path, monkeypatch):
    scratch = tmp_path / "prompts"
    scratch.mkdir()
    (scratch / "heal.md").write_text("first version", encoding="utf-8")

    monkeypatch.setattr(prompt_loader, "PROMPTS_DIR", scratch)
    prompt_loader.reload()
    assert load(HEAL) == "first version"

    (scratch / "heal.md").write_text("second version", encoding="utf-8")
    assert load(HEAL) == "first version", "should still be cached"

    prompt_loader.reload()
    assert load(HEAL) == "second version"

    # Leave the shared cache clean for the rest of the session.
    monkeypatch.undo()
    prompt_loader.reload()


def test_a_prompt_may_use_a_placeholder_called_name():
    """`render(name, /, ...)` — without positional-only, `$name` is unrenderable."""
    import prompt_loader

    assert prompt_loader.render("repair_request", **{
        "name": "Book a demo",
        "error": "e",
        "failed_step": "s",
        "failed_step_id": "s1",
        "failed_action": "fill",
        "wanted": "w",
        "page_url": "u",
        "allowed_domains": "d",
        "steps": "x",
        "candidates": "c",
        "past_fixes": "p",
    }).startswith("Use case: Book a demo")
