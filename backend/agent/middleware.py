"""The authoring loop's cross-cutting concerns, as `create_agent` middleware.

Everything here used to be hand-written inside `author.py`'s `decide`/`act`
nodes. Moving it to middleware is not a stylistic choice: `create_agent`
provides the model-calling and tool-dispatch nodes now, so a budget check or a
"the model called finish" check has nowhere else to attach.

Two things are deliberately *not* here. Approval uses LangChain's own
`HumanInTheLoopMiddleware` directly (see `graph.py`) rather than a hand-rolled
fourth middleware -- its `after_model` hook recomputes the interrupt payload
from already-checkpointed state with no id it mints and no side effect,
which is the safe form of the pattern this codebase's own hand-rolled
`approve` node got wrong before it was rewritten. And tool dispatch itself
-- the guard, the ref discipline, secret substitution, marks, redaction,
audit -- is not middleware either; it already lives in
`AgentToolSession.call()`, reached through `tool_adapter.py`, and none of
these three middlewares touch it.
"""

from __future__ import annotations

import logging
import time
from typing import Any, NotRequired

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState, ModelRequest, ModelResponse, hook_config
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from chat import text_of, usage_of
from events import Thinking
from llm import translate_access_error

from .author import Emit
from .budget import Spend
from .marks import Marks
from .session import AgentToolSession
from .tools.finish import NAME as FINISH

log = logging.getLogger(__name__)

#: How many of the most recent browser snapshots stay in full. Ported
#: unchanged from `author.py`'s `for_model` -- see there for the measured
#: session this fixed: N snapshots in history costs O(N^2) tokens over a
#: session, and a ref from any but the most recent one is stale anyway.
SNAPSHOTS_KEPT = 2


class AuthorGraphState(AgentState):
    """`AgentState` plus the outcome fields the old `AuthorState` carried.

    `messages` is everything `create_agent` needs; these three are everything
    `run.py` needs afterward to build an `AuthorResult` the same shape it
    always has.
    """

    status: NotRequired[str]
    summary: NotRequired[str]
    stopped_by: NotRequired[str]


class BudgetMiddleware(AgentMiddleware[AuthorGraphState, Any, Any]):
    """The budget check, the prompt-cache hint, and the snapshot-pruning
    that used to live inside `decide()` and `LangChainLLM.run_turn`.

    One instance per session, closed over that session's `Spend` -- the same
    lifetime `Wiring` gave it before.
    """

    state_schema = AuthorGraphState

    def __init__(
        self,
        spend: Spend,
        *,
        model_name: str,
        marks: Marks,
        provider: str = "bedrock",
        rates: "tuple[float, float] | None" = None,
    ) -> None:
        super().__init__()
        self.spend = spend
        self.model_name = model_name
        #: Which provider is answering. Two things need it: the prompt-cache
        #: hint below, which only Bedrock understands, and the access-error
        #: message, which named Bedrock whatever had actually refused the call.
        self.provider = provider
        #: The provider's own price for this model, when it publishes one.
        #: Without it an OpenRouter session is costed from a default that
        #: happens to be Claude Sonnet's -- and the USD ceiling is enforced
        #: against that number, so a cheap model would stop early and an
        #: expensive one would not stop at all.
        self.rates = rates
        #: For the stop message only -- see `abefore_model`. Read, never
        #: written: this middleware does not track progress, it reports it.
        self.marks = marks

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AuthorGraphState, runtime: Any) -> dict[str, Any] | None:
        """A limit is a stop, not a crash: end cleanly and keep everything
        the session already did, exactly as `stop_for_budget` used to.

        "Kept everything it did" is doing a lot of work in that sentence when
        what it did is nothing yet -- a session that runs out of budget before
        its first row completes looks, from `runs.status` alone, identical to
        one that banked three of them (`agent/manager.py` maps this status,
        "partial", to "succeeded" for both, deliberately: there is a draft
        either way, worth a person's look rather than a bare failure). The
        `stopped_by` message is where that difference still has to show up,
        since the status code no longer carries it.
        """
        hit = self.spend.exceeded()
        if hit is None:
            self.spend.step()
            return None
        limit, message = hit
        stopped_by = f"{message} {self.marks.progress_summary()}"
        log.info("agent session stopped by budget", extra={"limit": limit})
        return {
            "jump_to": "end",
            "status": "partial",
            "stopped_by": stopped_by,
            "messages": [AIMessage(content=f"Stopped: {stopped_by}")],
        }

    async def awrap_model_call(
        self, request: ModelRequest[Any], handler: Any
    ) -> ModelResponse[Any]:
        request.messages = _for_model(request.messages)
        # Bedrock's prompt cache: the system prompt and tool schemas are
        # identical on every turn of a loop that resends its whole history
        # every time. See the measured session in `llm.py`'s docstring for
        # why this is not an optimisation somebody can skip.
        #
        # Bedrock's, and only Bedrock's. `ChatOpenAI` hands its keyword
        # arguments to the OpenAI SDK, which refuses one it does not know --
        # so sending this to OpenRouter did not merely miss the saving, it
        # failed the call with `AsyncCompletions.create() got an unexpected
        # keyword argument 'cache_control'` on the first turn of every
        # session. `LangChainLLM.run_turn` guards its own copy of this; the
        # authoring loop reaches the model through `create_agent` instead and
        # needed the same guard here.
        if self.provider == "bedrock":
            request.model_settings = {
                **request.model_settings,
                "cache_control": {"ttl": "5m"},
            }

        started = time.monotonic()
        try:
            response = await handler(request)
        except Exception as exc:  # noqa: BLE001 - narrowed by translate_access_error
            raise translate_access_error(
                exc, model=self.model_name, provider=self.provider
            ) from exc
        finally:
            log.info(
                "model turn",
                extra={
                    "model": self.model_name,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "tool_count": len(request.tools),
                },
            )

        for message in response.result:
            usage = usage_of(message)
            if usage.get("input_tokens") or usage.get("output_tokens"):
                self.spend.turn(usage, self.model_name, rates=self.rates)
        return response


class ThinkingMiddleware(AgentMiddleware[AuthorGraphState, Any, Any]):
    """Surfaces what the model said before it acted -- if it said anything.

    `create_agent`'s tool-dispatch node reads a turn's tool calls and nothing
    else; ordinary commentary and Claude's extended-thinking blocks alike
    reached nobody -- not the live transcript, not a person watching, not
    this codebase's own audit trail. `agent/operate.py`'s older hand-rolled
    loop always emitted this for its own recover/explore turns; this graph,
    built later on `create_agent`, never gained the equivalent. A real
    session went 62 events end to end without one word of reasoning visible
    anywhere, which is indistinguishable from a model that never reasoned at
    all even on a turn where, given room to think, it may well have.

    Deliberately its own middleware rather than folded into `BudgetMiddleware`
    above: that one's docstring already names what it owns, and emitting an
    event is not a budget concern.
    """

    state_schema = AuthorGraphState

    def __init__(self, emit: Emit, *, run_id: str, spend: Spend) -> None:
        super().__init__()
        self.emit = emit
        self.run_id = run_id
        #: For the step number an event is stamped with, only -- the same
        #: counter `BudgetMiddleware.abefore_model` advances each turn.
        self.spend = spend

    async def awrap_model_call(
        self, request: ModelRequest[Any], handler: Any
    ) -> ModelResponse[Any]:
        response = await handler(request)
        for message in response.result:
            text = text_of(message).strip()
            if text:
                await self.emit(
                    Thinking(
                        run_id=self.run_id, seq=0, step=self.spend.steps,
                        text=text, done=True,
                    )
                )
        return response


class FinishMiddleware(AgentMiddleware[AuthorGraphState, Any, Any]):
    """Answers the `finish` tool call -- the one call `AgentToolSession`
    never sees, because ending the session is not something a browser or an
    MCP server does.

    `after_model` rather than `wrap_tool_call`: `finish` needs to end the
    *graph*, and the state-update-with-`jump_to` shape only that hook (and
    `before_model`) supports.
    """

    state_schema = AuthorGraphState

    def __init__(self, session: AgentToolSession) -> None:
        super().__init__()
        # Not named `tools`: `AgentMiddleware.tools` means "extra BaseTools
        # this middleware contributes to the graph", and `create_agent`
        # iterates it expecting exactly that -- shadowing it with the
        # `AgentToolSession` broke construction outright rather than quietly.
        self.session = session

    @hook_config(can_jump_to=["end"])
    async def aafter_model(self, state: AuthorGraphState, runtime: Any) -> dict[str, Any] | None:
        messages = state["messages"]
        last = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if not last or not last.tool_calls:
            return None

        call = next((c for c in last.tool_calls if c["name"] == FINISH), None)
        if call is None:
            return None

        problem = self.session.marks.unfinished()
        if problem:
            # Refused, and the session continues -- a recording with no row
            # boundary cannot be distilled, so accepting it here would mean
            # discovering that on the review screen with nothing to do about
            # it. Only the `finish` call is answered; any other tool call the
            # model made in the same turn still runs normally.
            revised = [c for c in last.tool_calls if c is not call]
            refusal = ToolMessage(content=problem, name=FINISH, tool_call_id=call["id"], status="error")
            if revised:
                last.tool_calls = revised
                return {"messages": [last, refusal]}
            return {"messages": [refusal]}

        summary = str(call["args"].get("summary") or "")
        status = "succeeded" if call["args"].get("complete", True) else "partial"
        done_message = ToolMessage(content=summary or "Done.", name=FINISH, tool_call_id=call["id"])
        return {"jump_to": "end", "status": status, "summary": summary, "messages": [done_message]}


def _for_model(messages: list[BaseMessage]) -> list[BaseMessage]:
    """The conversation, with stale page snapshots taken out.

    The `AuthorState`-era version of this worked on this codebase's own
    provider-native message dicts; `create_agent` hands middleware real
    `BaseMessage` objects instead, so this is that same rule -- keep the most
    recent `SNAPSHOTS_KEPT` browser snapshots, replace older ones with a
    placeholder that says why -- rewritten against `ToolMessage`.
    """
    kept = 0
    out: list[BaseMessage] = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and isinstance(message.content, str) and "### Page" in message.content:
            kept += 1
            if kept > SNAPSHOTS_KEPT:
                out.append(
                    message.model_copy(
                        update={
                            "content": (
                                "(page omitted: this is no longer the current "
                                "page and its refs are stale. Take a snapshot "
                                "if you need to look again.)"
                            )
                        }
                    )
                )
                continue
        out.append(message)
    out.reverse()
    return out


__all__ = ["AuthorGraphState", "BudgetMiddleware", "FinishMiddleware", "ThinkingMiddleware"]
