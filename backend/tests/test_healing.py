"""Healing: one broken step repaired, at a cost you can see.

The properties that matter are all about *not* spending money by accident:
healing is off unless injected, offered only to steps that ask for it, capped
by a budget, and unable to invent a locator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import RecordingSink
from healing import (
    CHOOSE_TOOL,
    HealingBudget,
    Repair,
    StepHealer,
    apply_repairs,
)
from llm import LLMTurn, ToolCallRequest
from engine import UseCaseExecutor
from snapshot import parse as parse_snapshot
from fake_browser import SIGNED_IN, SIGNED_OUT, role, session_serving
from usecase import Locator, Step, UseCase

RENAMED = """### Page
- Page URL: https://example.com/signin
### Snapshot
```yaml
- textbox "Username" [ref=e1]
- textbox "Password" [ref=e2]
- button "Log in" [ref=e3]
```"""


class ChoosingLLM:
    """Answers with a fixed index and reports token usage."""

    model = "fake"

    def __init__(self, index: int | None = 0, tokens: int = 900) -> None:
        self.index = index
        self.tokens = tokens
        self.calls = 0
        self.last_user_message = ""

    async def run_turn(self, *, system, messages, tools, on_text_delta=None, timeout=None):
        self.calls += 1
        self.last_user_message = messages[-1]["content"]
        usage = {"input_tokens": self.tokens, "output_tokens": 0}
        if self.index is None:
            return LLMTurn(text="I am not sure", usage=usage)
        return LLMTurn(
            tool_calls=[
                ToolCallRequest(
                    id="t1",
                    name="choose_element",
                    input={"index": self.index, "confidence": "high", "reason": "renamed"},
                )
            ],
            stop_reason="tool_use",
            usage=usage,
        )


def broken_step() -> Step:
    return Step(id="s1", action="click", locators=[role("Sign in")], on_failure="heal")


# --- the budget ------------------------------------------------------------


def test_a_fresh_budget_is_not_exhausted():
    assert HealingBudget().exhausted is False


def test_the_attempt_cap_exhausts_the_budget():
    budget = HealingBudget(max_attempts=2)
    budget.record(10)
    assert budget.exhausted is False
    budget.record(10)
    assert budget.exhausted is True


def test_the_token_cap_exhausts_the_budget():
    budget = HealingBudget(max_attempts=99, max_tokens=1000)
    budget.record(1200)
    assert budget.exhausted is True


async def test_an_exhausted_budget_does_not_call_the_model():
    llm = ChoosingLLM()
    healer = StepHealer(llm, HealingBudget(max_attempts=0))
    assert await healer.repair(broken_step(), parse_snapshot(RENAMED)) is None
    assert llm.calls == 0, "a broken selector must not quietly become expensive"


# --- choosing --------------------------------------------------------------


async def test_a_renamed_control_is_found_again():
    llm = ChoosingLLM(index=2)  # the "Log in" button
    healer = StepHealer(llm)

    repair = await healer.repair(broken_step(), parse_snapshot(RENAMED))

    assert repair is not None
    assert repair.locator.strategy == "role"
    assert (repair.locator.role, repair.locator.name) == ("button", "Log in")
    assert repair.tokens == 900


async def test_only_the_controls_on_the_page_are_offered():
    llm = ChoosingLLM(index=0)
    healer = StepHealer(llm)
    await healer.repair(broken_step(), parse_snapshot(RENAMED))

    assert 'textbox "Username"' in llm.last_user_message
    assert 'button "Log in"' in llm.last_user_message


async def test_the_healer_cannot_invent_a_locator():
    """Structural: the tool schema only accepts an index into the real page.

    Everything else it may return is prose about *why* -- a confidence, a
    reason, an explanation a person reads later. None of it can name an
    element, which is what stops a hallucinated selector having any route into
    a use case.
    """
    properties = CHOOSE_TOOL["input_schema"]["properties"]
    assert set(properties) == {"index", "confidence", "reason", "explanation"}
    assert CHOOSE_TOOL["input_schema"]["required"] == ["index"]
    assert "selector" not in str(properties)
    assert "locator" not in str(properties)


@pytest.mark.parametrize("index", [-1, 99, None])
async def test_a_declined_or_out_of_range_answer_is_no_repair(index):
    healer = StepHealer(ChoosingLLM(index=index))
    assert await healer.repair(broken_step(), parse_snapshot(RENAMED)) is None


async def test_a_model_error_is_a_failed_row_not_a_crash():
    class Boom:
        model = "boom"

        async def run_turn(self, **kwargs):
            raise RuntimeError("provider is down")

    assert await StepHealer(Boom()).repair(broken_step(), parse_snapshot(RENAMED)) is None


async def test_an_empty_page_yields_no_repair():
    healer = StepHealer(ChoosingLLM())
    assert await healer.repair(broken_step(), parse_snapshot("")) is None
    assert await healer.repair(broken_step(), None) is None


def test_structural_roles_are_not_offered_as_candidates():
    healer = StepHealer(ChoosingLLM())
    snapshot = parse_snapshot(
        '```yaml\n- generic "wrapper" [ref=e1]:\n  - button "Go" [ref=e2]\n```'
    )
    assert [n.role for n in healer.candidates(snapshot)] == ["button"]


def test_candidates_are_capped_so_a_huge_page_cannot_blow_the_budget():
    from healing import MAX_CANDIDATES

    lines = "\n".join(f'- button "Button {i}" [ref=e{i}]' for i in range(200))
    snapshot = parse_snapshot(f"```yaml\n{lines}\n```")
    assert len(StepHealer(ChoosingLLM()).candidates(snapshot)) == MAX_CANDIDATES


# --- integration with the executor -----------------------------------------


async def test_no_healer_means_no_repair_and_no_model():
    """The default. There is no code path from the executor to an LLM."""
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[broken_step()],
    )
    result = await UseCaseExecutor(
        use_case, session_serving([RENAMED])(None), RecordingSink(), run_id="r1"
    ).run_row({})

    assert not result.ok
    assert "no element matched" in result.error


async def test_a_heal_step_is_repaired_and_the_row_then_succeeds():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"],
        row_steps=[broken_step()],
    )
    mcp = session_serving([RENAMED])(None)
    sink = RecordingSink()
    llm = ChoosingLLM(index=2)
    runner = UseCaseExecutor(
        use_case, mcp, sink, run_id="r1", healer=StepHealer(llm)
    )

    result = await runner.run_row({})

    assert result.ok, result.error
    assert llm.calls == 1
    # The repaired step clicked the control the healer chose. It is described
    # by role and name now rather than by an MCP ref, because the engine
    # addresses elements with locators.
    # The page renamed "Sign in" to "Log in"; the healer found the control
    # under its new name and the retry clicked that. The target is described by
    # role and name now rather than by an MCP ref, because the engine addresses
    # elements with locators.
    clicked = [args["target"] for name, args in mcp.calls if name == "click"]
    assert clicked and "Log in" in clicked[0]


async def test_a_repair_is_recorded_and_labelled_in_the_timeline():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"], row_steps=[broken_step()]
    )
    sink = RecordingSink()
    runner = UseCaseExecutor(
        use_case, session_serving([RENAMED])(None), sink, run_id="r1",
        healer=StepHealer(ChoosingLLM(index=2)),
    )
    await runner.run_row({})

    assert [r.locator.name for r in runner.healed] == ["Log in"]
    healed_events = [e for e in sink.of_type("error") if e.kind == "healed"]
    assert len(healed_events) == 1
    assert "Log in" in healed_events[0].message
    assert healed_events[0].recoverable is True


async def test_a_step_that_did_not_ask_for_healing_is_not_offered_it():
    use_case = UseCase(
        name="x",
        allowed_domains=["example.com"],
        row_steps=[Step(id="s1", action="click", locators=[role("Sign in")])],
    )
    llm = ChoosingLLM(index=2)
    result = await UseCaseExecutor(
        use_case, session_serving([RENAMED])(None), RecordingSink(), run_id="r1",
        healer=StepHealer(llm),
    ).run_row({})

    assert not result.ok
    assert llm.calls == 0, "on_failure defaults to abort, which never heals"


async def test_a_repair_that_still_fails_leaves_the_row_failed():
    use_case = UseCase(
        name="x", allowed_domains=["example.com"], row_steps=[broken_step()]
    )
    # Index 0 is a textbox; clicking it is fine, but pick a page where the
    # chosen element does not exist by making the healer point at nothing.
    llm = ChoosingLLM(index=None)
    result = await UseCaseExecutor(
        use_case, session_serving([RENAMED])(None), RecordingSink(), run_id="r1",
        healer=StepHealer(llm),
    ).run_row({})

    assert not result.ok


# --- writing the repair back -----------------------------------------------


def test_apply_repairs_prepends_rather_than_replacing():
    """A wrong repair should degrade to what the recording knew, not lose it."""
    definition = {
        "row_steps": [
            {
                "id": "s1",
                "action": "click",
                "locators": [{"strategy": "role", "role": "button", "name": "Sign in"}],
            }
        ]
    }
    patched = apply_repairs(
        definition,
        [Repair(step_id="s1", locator=Locator(strategy="role", role="button", name="Log in"))],
    )
    locators = patched["row_steps"][0]["locators"]

    assert locators[0]["name"] == "Log in", "the repair leads"
    assert locators[1]["name"] == "Sign in", "the original survives as a fallback"


def test_apply_repairs_leaves_untouched_steps_alone():
    definition = {"row_steps": [{"id": "s1", "action": "click", "locators": []}]}
    assert apply_repairs(definition, []) == definition


def test_apply_repairs_covers_every_phase():
    definition = {
        "setup_steps": [{"id": "u1", "action": "click", "locators": []}],
        "row_steps": [],
        "teardown_steps": [],
    }
    patched = apply_repairs(
        definition,
        [Repair(step_id="u1", locator=Locator(strategy="role", role="button", name="Go"))],
    )
    assert patched["setup_steps"][0]["locators"][0]["name"] == "Go"


def test_a_duplicate_repair_does_not_double_the_ladder():
    healed = {"strategy": "role", "role": "button", "name": "Log in"}
    definition = {"row_steps": [{"id": "s1", "action": "click", "locators": [healed]}]}
    patched = apply_repairs(
        definition,
        [Repair(step_id="s1", locator=Locator(strategy="role", role="button", name="Log in"))],
    )
    assert len(patched["row_steps"][0]["locators"]) == 1


# --- the structural guarantee still holds ----------------------------------


def test_the_engine_still_never_imports_an_llm():
    """Healing lives in healing.py precisely so this stays true."""
    source = (Path(__file__).parent.parent / "engine.py").read_text(encoding="utf-8")
    assert "import llm" not in source
    assert "from llm" not in source


# --- the switch ------------------------------------------------------------


def test_healing_off_never_even_constructs_a_model(tmp_path):
    """The default configuration has no route from a replay to an LLM."""
    from config import Settings
    from runner import EventBus, ReplayManager
    from store import Store

    def explode():
        raise AssertionError("the LLM factory was called with healing disabled")

    settings = Settings(_env_file=None)
    assert settings.replay_healing_enabled is False, "healing must be off by default"

    manager = ReplayManager(
        Store(tmp_path / "x.db", tmp_path / "a"), settings, EventBus(), llm_factory=explode
    )
    assert manager.make_healer() is None


def test_healing_on_builds_a_budgeted_healer(tmp_path):
    from config import Settings
    from runner import EventBus, ReplayManager
    from store import Store

    settings = Settings(
        _env_file=None,
        replay_healing_enabled=True,
        replay_heal_max_attempts=7,
        replay_heal_max_tokens=1234,
    )
    manager = ReplayManager(
        Store(tmp_path / "x.db", tmp_path / "a"),
        settings,
        EventBus(),
        llm_factory=lambda: ChoosingLLM(),
    )
    healer = manager.make_healer()

    assert healer is not None
    assert healer.budget.max_attempts == 7
    assert healer.budget.max_tokens == 1234


def test_no_factory_means_no_healer_even_when_enabled(tmp_path):
    from config import Settings
    from runner import EventBus, ReplayManager
    from store import Store

    manager = ReplayManager(
        Store(tmp_path / "x.db", tmp_path / "a"),
        Settings(_env_file=None, replay_healing_enabled=True),
        EventBus(),
    )
    assert manager.make_healer() is None


# --- what gets remembered, and when ----------------------------------------


class RecordingMemory:
    """A healing memory that only records what it was asked to write."""

    def __init__(self) -> None:
        self.written: list[dict] = []

    @property
    def available(self) -> bool:
        return True

    async def recall(self, **_: object) -> list:
        return []

    async def remember(self, **fields: object) -> None:
        self.written.append(dict(fields))


async def test_proposing_a_repair_does_not_remember_it_yet():
    """The write used to happen here, which recorded what the model *believed*
    about a page rather than what turned out to be true of it."""
    memory = RecordingMemory()
    healer = StepHealer(ChoosingLLM(index=2), memory=memory)

    repair = await healer.repair(broken_step(), parse_snapshot(RENAMED))

    assert repair is not None
    assert memory.written == [], "nothing is known yet about whether this worked"


async def test_a_repair_that_worked_is_remembered():
    memory = RecordingMemory()
    healer = StepHealer(ChoosingLLM(index=2), memory=memory)
    repair = await healer.repair(broken_step(), parse_snapshot(RENAMED))

    await healer.confirm(repair, True)

    assert len(memory.written) == 1
    assert memory.written[0]["new_locator"]["name"] == "Log in"


async def test_a_repair_that_failed_is_not_remembered():
    """The case the old write could not see, and the one that matters.

    Recall puts past fixes in front of the model as context, so a confident
    wrong answer does not merely fail to help -- it argues for repeating
    itself, every time that site breaks again.
    """
    memory = RecordingMemory()
    healer = StepHealer(ChoosingLLM(index=2), memory=memory)
    repair = await healer.repair(broken_step(), parse_snapshot(RENAMED))

    await healer.confirm(repair, False)

    assert memory.written == []


async def test_a_repair_the_model_was_unsure_of_is_not_remembered_even_when_it_worked():
    """A guess that happened to land is still a guess, and it is exactly the
    answer not to offer as evidence next time."""
    memory = RecordingMemory()
    healer = StepHealer(ChoosingLLM(index=2), memory=memory)
    repair = await healer.repair(broken_step(), parse_snapshot(RENAMED))
    repair.confidence = "low"

    await healer.confirm(repair, True)

    assert memory.written == []


async def test_confirming_without_a_memory_is_harmless():
    """A healer built without an embedder has no memory behind it, and the
    executor calls `confirm` on every repair regardless."""
    healer = StepHealer(ChoosingLLM(index=2))
    repair = await healer.repair(broken_step(), parse_snapshot(RENAMED))

    await healer.confirm(repair, True)
