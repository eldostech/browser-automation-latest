"""A published use case, rendered as a Playwright script somebody can read.

The inverse of ``codegen.py``. That module parses ``page.get_by_role(...)``
into a ``Locator``; this one writes a ``Locator`` back out as the call it came
from, so a recording can leave here as a file an engineer runs in their own
pipeline.

**Export, never execute.** This module returns a string. Nothing in this
codebase imports, ``exec``s or runs what it produces, and the use case
document stays the source of truth -- the same rule that has always applied to
codegen output in the other direction. That is the whole reason this exists in
this shape: people keep asking for "a Python script I can see and edit", and
the honest way to give them one is a one-way export, not a source file the
platform starts depending on.

What the export deliberately loses, and why saying so matters more than fixing
it: the ladder. A step carries ranked rungs and the engine walks them, taking
the first that matches exactly one visible element. A script has one call per
action, so the export writes the leading rung and lists the rest as comments.
A script that fell through a ladder would be a reimplementation of the engine
in generated code, which is the thing nobody should maintain twice. So an
exported script is a good starting point and a worse runner than the engine,
and the header says so where somebody will read it.
"""

from __future__ import annotations

import re
from typing import Any

from usecase import Assertion, Locator, Step, UseCase

#: Where a secret comes from in an exported script. Never the value: the
#: export is a file that gets committed, mailed and pasted into tickets.
SECRET_PREFIX = "TRACE_SECRET_"

#: The template forms the executor understands, as they appear in a value.
_TEMPLATE = re.compile(r"\{\{\s*(input|secret|env)\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

_BY_STRATEGY = {
    "text": "get_by_text",
    "label": "get_by_label",
    "placeholder": "get_by_placeholder",
    "alt_text": "get_by_alt_text",
    "test_id": "get_by_test_id",
}


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _py(value: str) -> str:
    """One string as a Python literal, quotes chosen to avoid escaping."""
    if '"' not in value:
        return '"' + value.replace("\\", "\\\\") + '"'
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _source_expr(kind: str, name: str) -> str:
    if kind == "input":
        return f"row[{_py(name)}]"
    if kind == "secret":
        return f"os.environ[{_py(SECRET_PREFIX + name.upper())}]"
    return "BASE_URL" if name == "base_url" else f"os.environ[{_py(name.upper())}]"


def value_expr(value: str | None) -> str:
    """A recorded value as a Python expression.

    Three cases, because the middle one is where a naive renderer produces a
    script that runs and types the wrong thing:

    * no template: a plain literal.
    * exactly one template and nothing else: the source expression itself, so
      a number stays whatever the CSV reader made it.
    * a template inside other text: an f-string, which is the only form that
      keeps ``ACC-{{input.account}}`` meaning what it meant.
    """
    if value is None:
        return '""'
    found = list(_TEMPLATE.finditer(value))
    if not found:
        return _py(value)
    if len(found) == 1 and found[0].group(0) == value.strip():
        return _source_expr(found[0].group(1), found[0].group(2))

    out = value
    for match in reversed(found):
        expr = _source_expr(match.group(1), match.group(2))
        out = out[: match.start()] + "{" + expr + "}" + out[match.end() :]
    # An f-string cannot hold the quote it is delimited by, and the
    # expressions above are already quoted with `"`.
    return "f'" + out.replace("\\", "\\\\").replace("'", "\\'") + "'"


# ---------------------------------------------------------------------------
# Locators
# ---------------------------------------------------------------------------


def locator_expr(spec: Locator, *, root: str = "page") -> str:
    """One rung as the Playwright expression it was recorded as.

    Scope and frames are rendered as the chain they are: ``within`` becomes
    the expression it is scoped inside, and ``frames`` becomes a
    ``frame_locator`` chain, because an element inside a frame is not on the
    page as far as any other call is concerned.
    """
    base = root
    for selector in spec.frames:
        base = f"{base}.frame_locator({_py(selector)})"
    if spec.within is not None:
        base = locator_expr(spec.within, root=base)

    if spec.strategy == "role":
        args = [_py(spec.role or "")]
        if spec.name:
            args.append(f"name={_py(spec.name)}")
        if spec.exact and spec.name:
            args.append("exact=True")
        expr = f"{base}.get_by_role({', '.join(args)})"
    elif spec.strategy == "css":
        expr = f"{base}.locator({_py(spec.selector or '')})"
    elif spec.strategy == "nth":
        # A position on its own has nothing to be a position *of*; the engine
        # refuses such a rung too.
        expr = f"{base}.locator({_py('*')})"
    else:
        method = _BY_STRATEGY.get(spec.strategy, "get_by_text")
        args = [_py(spec.text or "")]
        if spec.exact and spec.strategy != "test_id":
            args.append("exact=True")
        expr = f"{base}.{method}({', '.join(args)})"

    if spec.has_text:
        expr += f".filter(has_text={_py(spec.has_text)})"
    if spec.nth:
        expr += f".nth({spec.nth})"
    return expr


def _target(step: Step) -> str:
    return locator_expr(step.locators[0]) if step.locators else "page"


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def assertion_lines(check: Assertion, indent: str) -> list[str]:
    """One assertion as ``expect(...)`` calls.

    Playwright's own assertions rather than bare ``assert``, because they wait.
    A recorded check that passes in the engine and fails in the export because
    the page had not finished rendering would make the export look wrong when
    it is the script that is.
    """
    timeout = f"timeout={check.timeout_ms}"
    negate = ".not_" if check.negate else ""
    value = check.value or ""
    if check.kind == "url_contains":
        return [f"{indent}expect(page){negate}.to_have_url(re.compile({_py(re.escape(value))}), {timeout})"]
    if check.kind == "title_contains":
        return [f"{indent}expect(page){negate}.to_have_title(re.compile({_py(re.escape(value))}), {timeout})"]
    if check.kind == "text_present":
        return [
            f"{indent}expect(page.get_by_text({_py(value)}).first){negate}"
            f".to_be_visible({timeout})"
        ]
    if check.kind == "element_visible":
        target = locator_expr(check.locator) if check.locator else "page"
        return [f"{indent}expect({target}.first){negate}.to_be_visible({timeout})"]
    if check.kind == "element_count":
        target = locator_expr(check.locator) if check.locator else "page"
        return [f"{indent}expect({target}){negate}.to_have_count({check.count or 0}, {timeout})"]
    if check.kind == "attribute_contains":
        target = locator_expr(check.locator) if check.locator else "page"
        return [
            f"{indent}expect({target}.first){negate}.to_have_attribute("
            f"{_py(check.attribute)}, re.compile({_py(re.escape(value))}), {timeout})"
        ]
    return [f"{indent}# unsupported check: {check.describe()}"]


def condition_expr(check: Assertion) -> str:
    """One assertion as a boolean expression, for a step's ``when``.

    Counted rather than awaited, deliberately, matching what the engine does:
    a condition asks what is on the page now. Waiting for a cookie banner that
    is not there would cost the timeout on every row.
    """
    value = check.value or ""
    if check.kind == "url_contains":
        expr = f"{_py(value)} in page.url"
    elif check.kind == "title_contains":
        expr = f"{_py(value)} in page.title()"
    elif check.kind == "text_present":
        expr = f"page.get_by_text({_py(value)}).count() > 0"
    elif check.kind == "element_visible":
        target = locator_expr(check.locator) if check.locator else "page"
        expr = f"{target}.first.is_visible()"
    elif check.kind == "element_count":
        target = locator_expr(check.locator) if check.locator else "page"
        expr = f"{target}.count() == {check.count or 0}"
    elif check.kind == "attribute_contains":
        target = locator_expr(check.locator) if check.locator else "page"
        expr = (
            f"{_py(value)} in ({target}.first.get_attribute({_py(check.attribute)}) or \"\")"
        )
    else:  # pragma: no cover - the schema constrains `kind`
        expr = "True"
    return f"not ({expr})" if check.negate else expr


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_lines(step: Step, indent: str, *, allow_scripts: bool) -> list[str]:
    """One step, as the lines of Python it becomes, comments included."""
    lines: list[str] = []
    if step.intent:
        lines.append(f"{indent}# {step.intent}")
    elif step.description:
        lines.append(f"{indent}# {step.description}")

    # The rungs this export cannot walk. Written down rather than dropped: the
    # engine takes the first that matches one visible element, and somebody
    # debugging the script needs to know what it would have tried next.
    for extra in step.locators[1:]:
        lines.append(f"{indent}# also recorded: {extra.describe()}")
    if step.expect_text and step.locators and not step.locators[0].matches_on_text:
        lines.append(f"{indent}# recorded against text: {step.expect_text!r}")

    # A condition wraps the action, so the action is rendered at whatever
    # depth it ends up at rather than rendered once and re-indented.
    if step.when is not None:
        lines.append(f"{indent}if {condition_expr(step.when)}:")
        lines.extend(_action_lines(step, indent + "    ", allow_scripts=allow_scripts))
        return lines

    if step.optional or step.on_failure == "continue":
        # The engine records the failure and carries on. A bare `except` is
        # what that looks like in a script, and the comment is what stops a
        # reader treating it as sloppiness.
        lines.append(f"{indent}try:  # recorded as optional")
        lines.extend(_action_lines(step, indent + "    ", allow_scripts=allow_scripts))
        lines.append(f"{indent}except Exception as exc:")
        lines.append(f"{indent}    print({_py('skipped ' + step.id + ':')}, exc)")
        return lines

    lines.extend(_action_lines(step, indent, allow_scripts=allow_scripts))
    return lines


def _action_lines(step: Step, indent: str, *, allow_scripts: bool) -> list[str]:
    timeout = f"timeout={step.timeout_ms}"
    target = _target(step)

    if step.action == "navigate":
        return [f"{indent}page.goto({value_expr(step.url)}, {timeout})"]
    if step.action == "click":
        return [f"{indent}{target}.click({timeout})"]
    if step.action == "fill":
        return [f"{indent}{target}.fill({value_expr(step.value)}, {timeout})"]
    if step.action == "select":
        return [f"{indent}{target}.select_option({value_expr(step.value)}, {timeout})"]
    if step.action == "hover":
        return [f"{indent}{target}.hover({timeout})"]
    if step.action == "press":
        if step.locators:
            return [f"{indent}{target}.press({value_expr(step.value)}, {timeout})"]
        return [f"{indent}page.keyboard.press({value_expr(step.value)})"]
    if step.action == "upload":
        return [f"{indent}{target}.set_input_files({value_expr(step.value)}, {timeout})"]
    if step.action == "assert" and step.assertion is not None:
        return assertion_lines(step.assertion, indent)
    if step.action == "wait":
        return _wait_lines(step, indent)
    if step.action == "extract":
        read = (
            f"{target}.first.get_attribute({_py(step.attribute)})"
            if step.attribute
            else f"{target}.first.inner_text()"
        )
        return [f"{indent}outputs[{_py(step.output or step.id)}] = {read}"]
    if step.action == "extract_rows":
        return _extract_rows_lines(step, indent)
    if step.action == "download":
        return [
            f"{indent}with page.expect_download() as download:",
            f"{indent}    {target}.click({timeout})",
            f"{indent}outputs[{_py(step.output or step.id)}] = download.value.suggested_filename",
        ]
    if step.action == "script":
        if not allow_scripts:
            # The document refuses to run this without a person turning it on,
            # and an export that quietly turned it on would be a way around a
            # gate rather than a script.
            return [
                f"{indent}# omitted: this step runs raw JavaScript and this use case",
                f"{indent}# has not been granted permission to run scripts.",
            ]
        return [f"{indent}page.evaluate({_py(step.code or '')})"]
    return [f"{indent}# unsupported action: {step.action}"]


def _wait_lines(step: Step, indent: str) -> list[str]:
    wait = step.wait_for
    if wait is None:
        return [f"{indent}page.wait_for_timeout(1000)"]
    if wait.kind == "text":
        return [
            f"{indent}expect(page.get_by_text({value_expr(wait.value)}).first)"
            f".to_be_visible(timeout={wait.timeout_ms})"
        ]
    if wait.kind == "text_gone":
        return [
            f"{indent}expect(page.get_by_text({value_expr(wait.value)}).first)"
            f".not_.to_be_visible(timeout={wait.timeout_ms})"
        ]
    if wait.kind == "load_state":
        return [f"{indent}page.wait_for_load_state({_py(wait.state or 'load')})"]
    return [f"{indent}page.wait_for_timeout({wait.timeout_ms})"]


def _extract_rows_lines(step: Step, indent: str) -> list[str]:
    lines = [
        f"{indent}rows_read = []",
        f"{indent}for match in {_target(step)}.all():",
        f"{indent}    rows_read.append({{",
    ]
    for column in step.columns:
        read = (
            f"match.locator({_py(column.selector)}).first.get_attribute({_py(column.attribute)})"
            if column.attribute
            else f"match.locator({_py(column.selector)}).first.inner_text()"
        )
        lines.append(f"{indent}        {_py(column.name)}: {read},")
    lines.append(f"{indent}    }})")
    lines.append(f"{indent}outputs[{_py(step.output or step.id)}] = rows_read")
    return lines


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------


def _phase(name: str, steps: list[Step], *, allow_scripts: bool) -> list[str]:
    if not steps:
        return [f"    # no {name} steps were recorded", "    return"]
    out: list[str] = []
    for step in steps:
        out.extend(step_lines(step, "    ", allow_scripts=allow_scripts))
        out.append("")
    return out


def as_playwright_python(use_case: UseCase, *, version: int | None = None) -> str:
    """The whole use case as one runnable script.

    Shaped the way the document is shaped, because that is the thing worth
    teaching a reader: setup runs once per session, the row function runs once
    per spreadsheet row, and the reset runs between rows. A script that
    inlined all of it would hide the one structural fact that makes a batch
    work -- that signing in happens once.
    """
    inputs = [spec.name for spec in use_case.inputs]
    secrets = [spec.name for spec in use_case.secrets]
    stamp = f"v{version}" if version else f"v{use_case.version}"

    head = [
        '"""' + use_case.name + " -- exported from TRACE.",
        "",
        f"Use case {use_case.id} {stamp}, status {use_case.status}.",
        "",
        "This is a one-way export. The use case document, not this file, is what",
        "TRACE runs, and it is what healing and repair edit. Change a step here and",
        "the platform knows nothing about it; change it there and re-export.",
        "",
        "It is also a weaker runner than the engine, in one specific way worth",
        "knowing before you rely on it: each recorded step carries a ranked ladder of",
        "locators and the engine walks it, taking the first rung that matches exactly",
        "one visible element. A script has one call per action, so only the leading",
        "rung is executed here; the others are written beside it as comments.",
        "",
    ]
    if use_case.instructions:
        head += ["What it does:", "", use_case.instructions, ""]
    if inputs:
        head += [
            "Per-row columns expected in the CSV: " + ", ".join(inputs),
            "",
        ]
    if secrets:
        head += [
            "Credentials, read from the environment:",
            *[f"  {SECRET_PREFIX}{name.upper()}" for name in secrets],
            "",
        ]
    head += [
        "Run it:",
        "",
        "  pip install playwright && playwright install chromium",
        "  python this_file.py rows.csv",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import csv",
        "import os",
        "import re",
        "import sys",
        "",
        "from playwright.sync_api import expect, sync_playwright",
        "",
        f"BASE_URL = os.environ.get({_py('TRACE_BASE_URL')}, {_py(use_case.base_url)})",
        "",
        "outputs: dict[str, object] = {}",
        "",
        "",
    ]

    body = [
        "def setup(page):",
        '    """Runs once per session. The sign-in lives here."""',
        *_phase("setup", use_case.setup_steps, allow_scripts=use_case.allow_scripts),
        "",
        "def do_row(page, row):",
        '    """Runs once per row of the CSV."""',
        *_phase("row", use_case.row_steps, allow_scripts=use_case.allow_scripts),
        "",
    ]

    if use_case.row_reset is not None:
        body += [
            "def reset(page):",
            '    """Returns the browser to a known state between rows."""',
            *step_lines(use_case.row_reset, "    ", allow_scripts=use_case.allow_scripts),
            "",
        ]
    else:
        body += ["def reset(page):", "    pass", "", ""]

    if use_case.teardown_steps:
        body += [
            "def teardown(page):",
            *_phase("teardown", use_case.teardown_steps, allow_scripts=use_case.allow_scripts),
            "",
        ]
    else:
        body += ["def teardown(page):", "    pass", "", ""]

    tail = [
        "def main() -> int:",
        "    path = sys.argv[1] if len(sys.argv) > 1 else " + _py("rows.csv"),
        "    with open(path, newline=" + _py("") + ", encoding=" + _py("utf-8") + ") as handle:",
        "        rows = list(csv.DictReader(handle))",
        "    if not rows:",
        "        print(" + _py("no rows to run") + ")",
        "        return 1",
        "",
        "    failures = 0",
        "    with sync_playwright() as playwright:",
        "        browser = playwright.chromium.launch(headless=True)",
        "        page = browser.new_page()",
        "        setup(page)",
        "        for number, row in enumerate(rows, start=1):",
        "            try:",
        "                do_row(page, row)",
        "                print(f" + _py("row {number} ok") + ")",
        "            except Exception as exc:",
        "                failures += 1",
        "                print(f" + _py("row {number} failed: {exc}") + ")",
        "            reset(page)",
        "        teardown(page)",
        "        browser.close()",
        "    if outputs:",
        "        print(outputs)",
        "    return 1 if failures else 0",
        "",
        "",
        'if __name__ == "__main__":',
        "    raise SystemExit(main())",
        "",
    ]

    return "\n".join([*head, *body, *tail])


def suggested_filename(use_case: UseCase) -> str:
    """A file name from the use case's own name, safe on every platform."""
    stem = re.sub(r"[^A-Za-z0-9]+", "_", use_case.name).strip("_").lower() or "use_case"
    return f"{stem}.py"


__all__ = [
    "SECRET_PREFIX",
    "as_playwright_python",
    "assertion_lines",
    "condition_expr",
    "locator_expr",
    "step_lines",
    "suggested_filename",
    "value_expr",
]
