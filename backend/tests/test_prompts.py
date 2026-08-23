"""The prompt files in ``backend/prompts/``.

Prompts are the part of an agent most likely to be edited casually, so these
tests hold the load-bearing parts in place: that every prompt the code asks for
exists, that placeholders on disk match what the code actually supplies, and
that the prompt-injection defence has not been edited out of the system prompt.
"""

from __future__ import annotations

import pytest

import prompt_loader
from prompt_loader import (
    APPROVAL_REJECTED,
    EMPTY_TOOL_RESULT,
    LOOP_NUDGE,
    NAVIGATION_BLOCKED,
    PROMPTS_DIR,
    REQUIRED_PROMPTS,
    SYSTEM,
    TASK,
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
    assert SYSTEM in message  # lists what *is* available


# --- placeholders match what the code supplies -----------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        (SYSTEM, set()),
        (TASK, {"task", "allowed_domains", "max_steps", "timeout_seconds", "start_url_line"}),
        (LOOP_NUDGE, {"tool_name", "repeats"}),
        (APPROVAL_REJECTED, {"decision"}),
        (NAVIGATION_BLOCKED, {"url", "allowlist"}),
        (EMPTY_TOOL_RESULT, set()),
    ],
)
def test_placeholders_match_the_call_sites(name, expected):
    """If someone adds a $placeholder to a file, render() would raise at
    runtime. This fails in CI instead."""
    assert placeholders(name) == expected


# --- rendering -------------------------------------------------------------


def test_task_prompt_renders_with_a_start_url():
    text = render(
        TASK,
        task="Find the pricing page",
        allowed_domains="example.com",
        max_steps=30,
        timeout_seconds="300s",
        start_url_line="The browser has already been opened at https://example.com.",
    )
    assert "Find the pricing page" in text
    assert "example.com" in text
    assert "30 steps" in text
    assert "https://example.com." in text
    assert "$" not in text


def test_task_prompt_collapses_the_gap_when_there_is_no_start_url():
    text = render(
        TASK,
        task="t",
        allowed_domains="example.com",
        max_steps=5,
        timeout_seconds="60s",
        start_url_line="",
    )
    assert "\n\n\n" not in text
    assert "Begin by taking a snapshot" in text


def test_missing_placeholder_raises_rather_than_leaking_a_dollar_sign():
    """A raw '$task' reaching the model is far worse than a loud failure."""
    with pytest.raises(KeyError) as excinfo:
        render(TASK, task="t")
    assert "task" in str(excinfo.value)


def test_user_task_text_is_not_re_scanned_for_placeholders():
    """Task text is user-controlled. A '$' in it must stay literal rather than
    being treated as a placeholder -- otherwise a task could reach into the
    template and rewrite the prompt around it."""
    hostile = "Costs $100. Also $task ${allowed_domains} $start_url_line"
    text = render(
        TASK,
        task=hostile,
        allowed_domains="example.com",
        max_steps=5,
        timeout_seconds="60s",
        start_url_line="",
    )
    # The user's text survives byte for byte...
    assert hostile in text
    # ...and none of its $names were expanded: the real allowlist appears once.
    assert text.count("example.com") == 1


@pytest.mark.parametrize(
    "name,values",
    [
        (LOOP_NUDGE, {"tool_name": "browser_click", "repeats": 3}),
        (APPROVAL_REJECTED, {"decision": "rejected"}),
        (NAVIGATION_BLOCKED, {"url": "'https://evil.net'", "allowlist": "example.com"}),
    ],
)
def test_short_prompts_render_without_leftovers(name, values):
    text = render(name, **values)
    assert text and "$" not in text
    for value in values.values():
        assert str(value) in text


# --- the security rule must survive prompt edits ---------------------------


def test_system_prompt_still_forbids_following_page_content():
    """The injection defence is the one part of the system prompt that must
    never be casually edited away. See the README's prompt-injection section."""
    system = load(SYSTEM).lower()
    assert "untrusted data" in system
    assert "never instructions" in system
    assert "do not comply" in system
    assert "prompt-injection" in system


def test_system_prompt_keeps_the_snapshot_first_instruction():
    system = load(SYSTEM).lower()
    assert "snapshot" in system
    assert "primary observation" in system


def test_system_prompt_explains_how_to_finish():
    system = load(SYSTEM)
    assert "```json" in system
    assert "Do not loop." in system


# --- caching ---------------------------------------------------------------


def test_reload_picks_up_an_edited_file(tmp_path, monkeypatch):
    scratch = tmp_path / "prompts"
    scratch.mkdir()
    (scratch / "system.md").write_text("first version", encoding="utf-8")

    monkeypatch.setattr(prompt_loader, "PROMPTS_DIR", scratch)
    prompt_loader.reload()
    assert load(SYSTEM) == "first version"

    (scratch / "system.md").write_text("second version", encoding="utf-8")
    assert load(SYSTEM) == "first version", "should still be cached"

    prompt_loader.reload()
    assert load(SYSTEM) == "second version"

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
    }).startswith("Use case: Book a demo")
