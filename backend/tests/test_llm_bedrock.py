"""The LLM seam: provider construction, credentials, model roles, access errors.

Every test here is offline. Credential resolution is a local operation
(environment variables and ``~/.aws``), so none of this calls AWS, and the
developer's own environment is masked out so results are deterministic.

Provider wiring now lives in ``langchain-aws``. What remains ours -- and
therefore what is tested here -- is the seam: which model each role gets, how
credentials are reported, and turning a provider access failure into something
an operator can act on.
"""

from __future__ import annotations

import pytest

from chat import chat_model, to_langchain
from config import Settings
from llm import (
    BEDROCK_BEARER_TOKEN_ENV,
    LangChainLLM,
    LLMAccessError,
    bedrock_auth_status,
    build_llm,
    llm_health,
)


@pytest.fixture(autouse=True)
def isolated_aws_env(monkeypatch):
    """Hide any real AWS configuration the developer happens to have set."""
    for var in (
        BEDROCK_BEARER_TOKEN_ENV,
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
    ):
        monkeypatch.delenv(var, raising=False)
    # A region must resolve or no Bedrock endpoint can be built.
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def settings(**overrides) -> Settings:
    """Settings isolated from the operator's .env.

    ``_env_file=None`` matters: without it, every field this helper does not
    name is still read from .env, so these tests assert on whoever's machine is
    running them. That has bitten three times -- the API key, the default
    model, and the repair model.
    """
    base = {
        "_env_file": None,
        "aws_region": "us-east-1",
        "aws_profile": None,
        "llm_max_tokens": 4096,
        "llm_temperature": 0.0,
    }
    base.update(overrides)
    return Settings(**base)


# --- defaults --------------------------------------------------------------


@pytest.mark.parametrize("field", ["llm_repair_model"])
def test_default_models_are_inference_profiles_not_bare_model_ids(field):
    """Current Claude models on Bedrock are cross-region-profile only.

    A bare foundation-model ID is rejected with "on-demand throughput isn't
    supported", so every default must carry a region prefix.
    """
    model = getattr(Settings(_env_file=None), field)
    assert model.startswith(("us.", "eu.", "apac.", "global.")), model


def test_bedrock_needs_no_api_key_setting():
    """Bedrock authenticates with AWS credentials, so there is nothing to set.

    An unused ANTHROPIC_API_KEY in .env is ignored rather than quietly looking
    like configuration that matters. `OPENROUTER_API_KEY` is a real setting
    now, and only the second provider reads it.
    """
    assert not hasattr(Settings(_env_file=None), "anthropic_api_key")
    assert build_llm(settings()).provider == "bedrock"


def test_bedrock_is_still_what_a_deployment_gets_by_default():
    """Adding a second provider must not move an existing installation onto it."""
    assert Settings(_env_file=None).llm_provider == "bedrock"


# --- construction ----------------------------------------------------------


def test_a_bedrock_model_is_built_from_langchain_aws():
    model = chat_model(settings(), "us.anthropic.claude-sonnet-4-6")
    assert type(model).__name__ == "ChatBedrockConverse"
    assert model.region_name == "us-east-1"


def test_a_named_profile_is_passed_through():
    """langchain-aws resolves the profile eagerly, so a bogus one proves it
    reached the client -- which is the part we are responsible for."""
    with pytest.raises(Exception) as excinfo:
        chat_model(settings(aws_profile="no-such-profile"), "us.anthropic.claude-sonnet-4-6")
    assert "no-such-profile" in str(excinfo.value)


# --- extended thinking -------------------------------------------------------


def test_thinking_is_off_by_default_setting_value():
    model = chat_model(settings(llm_thinking_budget_tokens=0))
    assert not (model.additional_model_request_fields or {}).get("thinking")
    assert model.temperature == 0.0, "untouched when thinking never gets involved"


def test_thinking_is_enabled_with_the_configured_budget():
    model = chat_model(settings(llm_thinking_budget_tokens=2048, llm_max_tokens=8192))
    assert model.additional_model_request_fields["thinking"] == {
        "type": "enabled",
        "budget_tokens": 2048,
    }


def test_temperature_is_dropped_while_thinking_is_enabled():
    """Confirmed against the real model, not assumed from documentation: a
    non-default temperature alongside `thinking` is rejected outright --
    "`temperature` may only be set to 1 when thinking is enabled" -- so this
    is a request that must never be sent, not one worth handling by retrying."""
    model = chat_model(settings(llm_thinking_budget_tokens=2048, llm_temperature=0.0))
    assert model.temperature is None


def test_a_budget_that_would_leave_no_room_for_a_reply_is_clamped():
    model = chat_model(
        settings(llm_thinking_budget_tokens=8192, llm_max_tokens=4096)
    )
    thinking = model.additional_model_request_fields["thinking"]
    assert thinking["budget_tokens"] < 4096
    assert thinking["budget_tokens"] >= 1024


# --- the bearer-token / profile conflict -----------------------------------


def test_bearer_token_plus_profile_fails_with_an_actionable_message(monkeypatch):
    """Bedrock accepts one or the other; its own error names neither."""
    monkeypatch.setenv(BEDROCK_BEARER_TOKEN_ENV, "some-bedrock-api-key")

    with pytest.raises(ValueError) as excinfo:
        build_llm(settings(aws_profile="default"))

    message = str(excinfo.value)
    assert BEDROCK_BEARER_TOKEN_ENV in message
    assert "AWS_PROFILE" in message


def test_bearer_token_alone_is_fine(monkeypatch):
    monkeypatch.setenv(BEDROCK_BEARER_TOKEN_ENV, "some-bedrock-api-key")
    assert build_llm(settings(aws_profile=None)).describe()["auth"]["method"] == "bearer_token"


# --- credential resolution -------------------------------------------------


def test_bearer_token_is_detected_and_reported(monkeypatch):
    monkeypatch.setenv(BEDROCK_BEARER_TOKEN_ENV, "some-bedrock-api-key")
    status = bedrock_auth_status()

    assert status["ok"] is True
    assert status["method"] == "bearer_token"
    # The value itself must never be echoed back -- only its variable name.
    assert "some-bedrock-api-key" not in str(status)
    assert status["source"] == BEDROCK_BEARER_TOKEN_ENV
    assert "precedence" in status["note"]


def test_sigv4_credentials_are_reported_with_their_source(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    status = bedrock_auth_status()

    assert status["ok"] is True
    assert status["method"] == "sigv4"
    assert status["source"]
    assert "secret" not in str(status)


def test_missing_credentials_are_reported_not_raised(monkeypatch):
    import botocore.session

    monkeypatch.setattr(botocore.session.Session, "get_credentials", lambda self: None)
    assert bedrock_auth_status()["ok"] is False


def test_auth_status_never_raises(monkeypatch):
    import botocore.session

    def boom(self):
        raise RuntimeError("profile is corrupt")

    monkeypatch.setattr(botocore.session.Session, "get_credentials", boom)
    status = bedrock_auth_status()
    assert status["ok"] is False
    assert "profile is corrupt" in status["error"]


# --- health ----------------------------------------------------------------


def test_health_reports_bedrock_shape(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    health = llm_health(settings())

    assert health["provider"] == "bedrock"
    assert health["configured"] is True
    assert health["region"] == "us-east-1"
    assert health["auth"]["method"] == "sigv4"


def test_health_reports_missing_aws_credentials(monkeypatch):
    import botocore.session

    monkeypatch.setattr(botocore.session.Session, "get_credentials", lambda self: None)
    assert llm_health(settings())["configured"] is False


def test_health_always_reports_bedrock():
    assert llm_health(settings())["provider"] == "bedrock"


# --- one process, three models ---------------------------------------------


def test_build_llm_takes_a_model_override():
    assert build_llm(settings()).model == settings().llm_repair_model
    assert build_llm(settings(), "us.anthropic.claude-opus-5").model == (
        "us.anthropic.claude-opus-5"
    )


def test_one_model_is_built_and_cached():
    """There used to be three roles. Two of them went with the agent.

    A workflow is recorded by watching someone do it, and a codegen script is
    parsed rather than interpreted -- so neither the driver nor the distiller
    has anything left to do. What remains is the model that looks at a page
    when a step breaks.
    """
    from llm import RepairModel

    model = RepairModel(settings())
    assert model.client.model == settings().llm_repair_model
    assert model.client is model.client, "built once and cached"


def test_an_injected_client_is_used_as_is():
    """So a scripted model in a test is the one that gets called."""
    from llm import RepairModel

    scripted = object()
    model = RepairModel(settings(), client=scripted)
    assert model.client is scripted
    assert model() is scripted, "and it works as a zero-argument factory"


# --- a model the account cannot use ----------------------------------------
#
# Regression: Bedrock returned 403 "anthropic.claude-sonnet-5 is not available
# for this account" and the run died as "agent run crashed" with a traceback.
# The message names an ID the operator never typed -- the inference profile's
# region prefix is stripped -- so it reads like a wrong model ID.


class _Boom(Exception):
    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


def _client(model: str = "us.anthropic.claude-sonnet-5") -> LangChainLLM:
    return build_llm(settings(llm_repair_model=model))


@pytest.mark.parametrize(
    ("status", "expected", "names_model"),
    [
        # 403/404 are about the model, so the message names it. 401 is about
        # the credentials, where naming the model would only mislead.
        (403, "cannot use", True),
        (404, "has no model", True),
        (401, "credentials were rejected", False),
    ],
)
def test_access_failures_become_an_actionable_error(status, expected, names_model):
    translated = _client()._translate(_Boom(status, "provider detail"))  # noqa: SLF001
    message = str(translated)

    assert isinstance(translated, LLMAccessError)
    assert expected in message
    assert ("us.anthropic.claude-sonnet-5" in message) is names_model
    assert "provider detail" in message, "keeps what the provider said"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("AccessDeniedException: you lack access", "cannot use"),
        ("Model is not available for this account", "cannot use"),
        ("ResourceNotFoundException: could not be found", "has no model"),
    ],
)
def test_botocore_errors_without_a_status_code_are_still_recognised(text, expected):
    """botocore raises named exceptions carrying no HTTP status at all, so the
    access check would otherwise miss the very failure it exists to catch."""
    translated = _client()._translate(_Boom(None, text))  # noqa: SLF001
    assert isinstance(translated, LLMAccessError)
    assert expected in str(translated)


@pytest.mark.parametrize("status", [429, 500])
def test_other_failures_are_passed_through_untouched(status):
    original = _Boom(status, "something else")
    assert _client()._translate(original) is original  # noqa: SLF001


async def test_check_access_reports_a_missing_model_without_raising(monkeypatch):
    client = _client()

    async def deny(*args, **kwargs):
        raise _Boom(403, "not available for this account")

    # A LangChain model is a pydantic object, so patch the class, not the
    # instance -- setattr on the instance is rejected as an unknown field.
    monkeypatch.setattr(type(client._model), "ainvoke", deny)  # noqa: SLF001
    result = await client.check_access()

    assert result["ok"] is False
    assert result["access_problem"] is True
    assert result["model"] == "us.anthropic.claude-sonnet-5"


async def test_check_access_reports_success(monkeypatch):
    client = _client()

    async def allow(*args, **kwargs):
        return object()

    monkeypatch.setattr(type(client._model), "ainvoke", allow)  # noqa: SLF001
    assert (await client.check_access())["ok"] is True


def test_history_converts_to_langchain_messages():
    """The shapes that actually occur: prose, an assistant turn, tool results."""
    converted = to_langchain(
        "you drive a browser",
        [
            {"role": "user", "content": "open example.com"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "on it"},
                    {"type": "tool_use", "id": "c1", "name": "browser_navigate",
                     "input": {"url": "https://example.com"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "c1", "content": "ok",
                     "is_error": False}
                ],
            },
        ],
    )

    assert [type(m).__name__ for m in converted] == [
        "SystemMessage", "HumanMessage", "AIMessage", "ToolMessage",
    ]
    assert converted[2].tool_calls[0]["name"] == "browser_navigate"
    assert converted[3].tool_call_id == "c1"
    assert converted[3].status == "success"


def test_a_failed_tool_result_is_marked_as_an_error():
    converted = to_langchain(
        "",
        [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "nope", "is_error": True}
        ]}],
    )
    assert converted[0].status == "error"


# --- reading a turn's reasoning back out -------------------------------------


def test_extended_thinking_content_reaches_text_of():
    """The block shape here -- `reasoning_content` holding `text`/`signature`
    -- is exactly what the real model returned with thinking enabled, not a
    guess: confirmed by calling it. Without this, a turn could reason at
    length and `text_of` would still report nothing, because nothing else in
    this codebase reads any block type but `text`."""
    from langchain_core.messages import AIMessage

    from chat import text_of

    message = AIMessage(
        content=[
            {
                "type": "reasoning_content",
                "reasoning_content": {"text": "The button is the second match.", "signature": "abc"},
            },
            {"type": "text", "text": "Clicking it now."},
        ]
    )
    assert text_of(message) == "The button is the second match.Clicking it now."


def test_a_turn_with_only_a_tool_call_has_no_text():
    from langchain_core.messages import AIMessage

    from chat import text_of

    message = AIMessage(content=[{"type": "tool_use", "id": "c1", "name": "browser_click", "input": {}}])
    assert text_of(message) == ""
