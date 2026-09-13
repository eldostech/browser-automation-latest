"""A `BaseChatModel` that plays back a fixed script.

`create_agent` needs an actual LangChain chat model -- this codebase's own
`ScriptedLLM` (test_agent_author.py) implements the older `LLMClient`/
`run_turn` protocol instead, which the graph no longer calls. This is that
same idea -- a test drives the graph without a network or a real model --
rewritten against the interface `create_agent` actually requires.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def turn_calling(tool: str, *, thinking: str = "", **arguments: Any) -> AIMessage:
    """One scripted turn: the model calls exactly one tool.

    Carries a fake but nonzero `usage_metadata` -- a real provider always
    reports usage, and `BudgetMiddleware` reads it to feed `Spend.turn()`, so
    a script with no usage at all would under-test the very thing it is
    meant to stand in for.

    `thinking` stands in for whatever prose a real turn would carry alongside
    its tool call -- ordinary commentary, or Claude's extended-thinking
    content -- for tests that check a turn's reasoning reaches the transcript
    rather than only its tool call. Empty by default, matching every scripted
    turn before this parameter existed. Named apart from `arguments` on
    purpose: `text` is a real argument of `browser_type` and others, and would
    collide with it here.
    """
    return AIMessage(
        content=thinking,
        tool_calls=[{"name": tool, "args": dict(arguments), "id": f"c{tool}-{id(arguments)}", "type": "tool_call"}],
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )


class ScriptedChatModel(BaseChatModel):
    """Plays back `responses` in order, one per model call.

    `bind_tools` is a no-op returning `self`: what tools are offered is the
    graph's business, not this fake's, and a script does not need to inspect
    the schema to know what it was told to call.
    """

    responses: List[AIMessage] = []
    calls: List[List[BaseMessage]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(messages)
        idx = len(self.calls) - 1
        # A script that runs out answers with an empty, tool-call-free turn --
        # matching the older `ScriptedLLM`'s own fallback -- rather than
        # raising. A turn with no tool call is nudged and the loop goes
        # round again, so a test that scripts fewer turns than the loop
        # actually takes still terminates on its budget instead of crashing.
        message = self.responses[idx] if idx < len(self.responses) else AIMessage(content="")
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # A real model call always has some latency. Without this, a scripted
        # session can finish between one `await` and the next -- fast enough
        # that a test asserting a task is still in flight loses the race
        # against a graph that, unlike a real one, has nothing to wait on.
        await asyncio.sleep(0.02)
        return self._generate(messages, stop, run_manager, **kwargs)


__all__ = ["ScriptedChatModel", "turn_calling"]
