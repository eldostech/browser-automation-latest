"""Which Claude models can this account actually use?

Bedrock's 403 for a model you lack access to is easy to mistake for a wrong
model ID, because the message names a model ID you never typed -- the
cross-region inference profile's region prefix is stripped, so
``us.anthropic.claude-sonnet-5`` comes back as ``anthropic.claude-sonnet-5``.

The two cases are distinguishable, and this tells them apart:

* **400 "The provided model identifier is invalid"** -- the ID *shape* is
  wrong. Fix the string.
* **403 "not available for this account"** -- the ID was understood. The
  account does not have that model; request access in the Bedrock console.

Usage::

    python backend/scripts/check_models.py              # the configured three
    python backend/scripts/check_models.py --survey     # plus common candidates
    python backend/scripts/check_models.py --model us.anthropic.claude-opus-5

Each check is one request with ``max_tokens=1``, so a survey costs a handful
of tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from llm import build_llm  # noqa: E402

#: Worth probing when the configured model turns out to be unavailable. Not
#: exhaustive -- a starting point that covers the current families.
SURVEY: tuple[str, ...] = (
    "us.anthropic.claude-opus-5",
    "us.anthropic.claude-sonnet-5",
    "us.anthropic.claude-opus-4-8",
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
)


async def check(model: str) -> tuple[str, bool, str]:
    client = build_llm(settings, model)
    result = await client.check_access()
    if result["ok"]:
        return model, True, "available"
    error = result["error"]
    if "identifier is invalid" in error or "400" in error[:40]:
        return model, False, "INVALID ID -- fix the string"
    if result.get("access_problem"):
        return model, False, "no access -- request it in the Bedrock console"
    return model, False, error[:90]


async def main_async(models: list[str]) -> int:
    print(f"provider={settings.llm_provider} api={settings.bedrock_api}\n")
    results = await asyncio.gather(*(check(m) for m in models))

    usable: list[str] = []
    for model, ok, note in results:
        print(f"  {'OK ' if ok else '   '} {model:<48} {note}")
        if ok:
            usable.append(model)

    configured = settings.models_in_use
    print("\nconfigured:")
    for role, model in configured.items():
        state = "ok" if model in usable else "NOT USABLE"
        print(f"  {role:<10} {model:<48} {state}")

    broken = [r for r, m in configured.items() if m not in usable]
    if broken:
        print(
            f"\n{len(broken)} role(s) point at a model this account cannot use: "
            + ", ".join(broken)
        )
        if usable:
            print("Usable right now: " + ", ".join(usable))
        return 1

    print("\nEvery configured model is usable.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", action="append", default=[], help="probe this ID (repeatable)")
    parser.add_argument("--survey", action="store_true", help="also probe common candidates")
    args = parser.parse_args()

    models = list(dict.fromkeys(args.model or list(settings.models_in_use.values())))
    if args.survey:
        models = list(dict.fromkeys([*models, *SURVEY]))
    return asyncio.run(main_async(models))


if __name__ == "__main__":
    raise SystemExit(main())
