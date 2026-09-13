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
#:
#: For a while this list was only what a model is asked during a *replay*:
#: repair a broken locator, and explain a navigation the allowlist refused.
#: The authoring prompts are back, for an agent with a different job from the
#: one that was deleted -- it records a workflow the engine then repeats for
#: nothing, rather than being the only way to run anything.
NAVIGATION_BLOCKED = "navigation_blocked"
HEAL = "heal"
HEAL_REQUEST = "heal_request"
REPAIR = "repair"
REPAIR_REQUEST = "repair_request"
AUTHOR = "author"
AUTHOR_TASK = "author_task"
BRIEF = "brief"
BRIEF_REQUEST = "brief_request"
WALKTHROUGH = "walkthrough"
WALKTHROUGH_REQUEST = "walkthrough_request"
RECOVER = "recover"
EXPLORE = "explore"

#: Every prompt the application expects to find on disk. ``test_prompts.py``
#: asserts this matches the directory, so a deleted or renamed file fails the
#: suite rather than a run.
REQUIRED_PROMPTS: tuple[str, ...] = (
    NAVIGATION_BLOCKED,
    HEAL,
    HEAL_REQUEST,
    REPAIR,
    REPAIR_REQUEST,
    AUTHOR,
    AUTHOR_TASK,
    BRIEF,
    BRIEF_REQUEST,
    WALKTHROUGH,
    WALKTHROUGH_REQUEST,
    RECOVER,
    EXPLORE,
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


def render(name: str, /, **values: object) -> str:
    """Fill a prompt's ``$placeholders``.

    ``name`` is positional-only on purpose: without it, a prompt containing a
    ``$name`` placeholder could not be rendered at all, because the keyword
    would collide with this parameter. That is a trap worth closing once here
    rather than renaming placeholders around it forever.

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
