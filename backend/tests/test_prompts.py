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
        "purpose": "p",
        "call_log": "c",
        "page_url": "u",
        "allowed_domains": "d",
        "steps": "x",
        "candidates": "c",
        "was_working": "b",
        "past_fixes": "p",
    }).startswith("Use case: Book a demo")


# --- one policy, in the prompts rather than in the code -------------------
#
# Asked for explicitly, and asked for *as a prompt*: a consent banner is
# rejected, never accepted. Hard-coding it would mean matching button text in
# Python, which is a worse place for a judgement about a page than the prompt
# of the thing looking at the page.
#
# It is also the failure from a real run: a banner still up when the next click
# happened, so the click was intercepted and the step timed out on an element
# it had found.


def test_every_prompt_that_drives_a_browser_says_reject_never_accept():
    from prompt_loader import AUTHOR, EXPLORE, RECOVER, load

    for name in (AUTHOR, RECOVER, EXPLORE):
        text = load(name)
        assert "Reject, never accept" in text, name
        assert "Reject all" in text, name
        assert "cannot be undone" in text or "cannot be taken back" in text, name


def test_the_recorder_is_told_to_clear_it_before_anything_else():
    """Both halves matter: refusing is the consent decision, clearing it first
    is what stops the recording being made against a page nobody can act on."""
    from prompt_loader import AUTHOR, load

    text = load(AUTHOR)
    assert "before anything else" in text
    assert "intercepted" in text


def test_the_healer_treats_a_banner_as_the_problem_not_a_candidate():
    from prompt_loader import HEAL, load

    text = load(HEAL)
    assert "not the answer, it is the problem" in text


def test_no_consent_policy_is_hard_coded_in_the_engine_or_the_recorder():
    """The rule lives in the prompts. A list of banner button labels in Python
    would be this judgement made twice, in the place with less context."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for module in ("engine.py", "agent/session.py", "agent/distil.py", "agent/marks.py"):
        source = (root / module).read_text(encoding="utf-8")
        assert "Reject all" not in source, module
        assert "Accept all" not in source, module
