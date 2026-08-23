"""Loads the model-facing prompt templates in ``backend/prompts/``.

Every string this application sends to the LLM lives in that directory as a
Markdown file, not inline in Python. Prompt wording is the part of an agent
that gets tuned most often and by the widest range of people, so it is kept
where it can be read, diffed and edited without touching the loop.

Two decisions worth stating:

* **Files, not Python constants.** A prompt change shows up in review as a
  prose diff rather than a diff inside a code file, and someone tuning wording
  never has to worry about escaping or breaking an import.
* **``string.Template`` (``$name``), not f-strings or ``str.format``.** Prompts
  routinely contain ``{`` and ``}`` -- JSON examples, code snippets -- which
  ``str.format`` would try to interpret and choke on. ``$`` placeholders avoid
  that entirely. A literal ``$`` in a prompt must be written ``$$``.

Substitution is strict: a placeholder with no supplied value raises rather than
silently sending a raw ``$task`` to the model.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

#: Prompt names, so call sites are not stringly typed.
SYSTEM = "system"
TASK = "task"
LOOP_NUDGE = "loop_nudge"
APPROVAL_REJECTED = "approval_rejected"
NAVIGATION_BLOCKED = "navigation_blocked"
EMPTY_TOOL_RESULT = "empty_tool_result"
DISTILL = "distill"

#: Every prompt the application expects to find on disk. ``test_prompts.py``
#: asserts this matches the directory, so a deleted or renamed file fails the
#: suite instead of an agent run.
REQUIRED_PROMPTS: tuple[str, ...] = (
    SYSTEM,
    TASK,
    LOOP_NUDGE,
    APPROVAL_REJECTED,
    NAVIGATION_BLOCKED,
    EMPTY_TOOL_RESULT,
    DISTILL,
)

_BLANK_RUN = re.compile(r"\n{3,}")


class PromptNotFound(FileNotFoundError):
    """A prompt file is missing -- almost always a typo or a bad deploy."""


def available() -> list[str]:
    """Names of every prompt file present on disk."""
    if not PROMPTS_DIR.is_dir():
        return []
    return sorted(path.stem for path in PROMPTS_DIR.glob("*.md"))


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Read a prompt template verbatim. Cached -- call :func:`reload` in tests."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        raise PromptNotFound(
            f"No prompt named {name!r} in {PROMPTS_DIR}. Available: {', '.join(available()) or 'none'}"
        )
    return path.read_text(encoding="utf-8").strip()


def render(name: str, **values: object) -> str:
    """Fill a prompt's ``$placeholders``.

    Runs of blank lines are collapsed to one, so an optional line that renders
    empty (``$start_url_line`` when there is no start URL) does not leave a
    gap in the middle of the prompt.
    """
    template = Template(load(name))
    try:
        text = template.substitute(**values)
    except KeyError as exc:
        raise KeyError(
            f"Prompt {name!r} needs a value for ${exc.args[0]}; got: {sorted(values)}"
        ) from exc
    return _BLANK_RUN.sub("\n\n", text).strip()


def placeholders(name: str) -> set[str]:
    """The ``$names`` a prompt expects. Used by the tests to catch drift."""
    pattern = Template.pattern
    found = set()
    for match in pattern.finditer(load(name)):
        identifier = match.group("named") or match.group("braced")
        if identifier:
            found.add(identifier)
    return found


def reload() -> None:
    """Drop the cache so edited files are picked up (tests, and dev reloads)."""
    load.cache_clear()
