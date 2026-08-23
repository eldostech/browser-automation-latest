"""Bedrock provider wiring: credential resolution, auth conflicts, model IDs.

Every test here is offline. Credential resolution is a local operation
(environment variables and ``~/.aws``), so none of this calls AWS, and the
developer's own environment is masked out so results are deterministic.
"""

from __future__ import annotations

import pytest

from config import Settings
from llm import (
    BEDROCK_BEARER_TOKEN_ENV,
    AnthropicLLM,
    BedrockLLM,
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
    # A region must resolve or the SDK cannot build a Bedrock endpoint URL.
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


class _StubSettings:
    """Bypasses Settings' own validation to exercise defensive branches."""

    def __init__(self, **values):
        defaults = {
            "llm_provider": "bedrock",
            "llm_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "bedrock_api": "invoke",
            "aws_region": "us-east-1",
            "aws_profile": None,
            "anthropic_api_key": "",
            "llm_max_tokens": 4096,
            "llm_temperature": 0.0,
        }
        defaults.update(values)
        self.__dict__.update(defaults)


def settings(**overrides) -> Settings:
    base = {
        "llm_provider": "bedrock",
        "llm_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock_api": "invoke",
        "aws_region": "us-east-1",
        "aws_profile": None,
        "anthropic_api_key": "",
        "llm_max_tokens": 4096,
        "llm_temperature": 0.0,
    }
    base.update(overrides)
    return Settings(**base)


# --- defaults --------------------------------------------------------------


def test_bedrock_is_the_default_provider():
    assert Settings().llm_provider == "bedrock"


def test_default_model_is_an_inference_profile_not_a_bare_model_id():
    """Current Claude models on Bedrock are inference-profile only.

    The bare foundation-model ID is rejected at invoke time with
    "on-demand throughput isn't supported", so the default must carry a
    region prefix.
    """
    model = Settings().llm_model
    assert "haiku" in model
    assert model.startswith(("us.", "eu.", "apac.", "global.")), model
    assert model.endswith("-v1:0"), model


def test_no_api_key_is_required_for_bedrock(monkeypatch):
    """The whole point: Bedrock authenticates with AWS credentials.

    Isolated from both credential sources -- ``_env_file=None`` ignores the
    operator's ``.env`` and ``delenv`` clears the process environment. Without
    that this asserts on whoever's laptop is running the suite rather than on
    the code, and fails for anyone who has a real key configured.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert Settings(_env_file=None).anthropic_api_key == ""
    llm = build_llm(settings())
    assert isinstance(llm, BedrockLLM)


# --- credential resolution -------------------------------------------------


def test_bearer_token_is_detected_and_reported(monkeypatch):
    monkeypatch.setenv(BEDROCK_BEARER_TOKEN_ENV, "some-bedrock-api-key")
    status = bedrock_auth_status()
    assert status["ok"] is True
    assert status["method"] == "bearer_token"
    # The token value itself must never be echoed back -- only its variable name.
    assert "some-bedrock-api-key" not in str(status)
    assert status["source"] == BEDROCK_BEARER_TOKEN_ENV
    # And the operator is told it overrides IAM, which is easy to miss.
    assert "precedence" in status["note"]


def test_sigv4_credentials_are_reported_with_their_source(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    status = bedrock_auth_status()
    assert status["ok"] is True
    assert status["method"] == "sigv4"
    assert status["source"]  # botocore's label, e.g. "env"
    assert "secret" not in str(status)


def test_missing_credentials_are_reported_not_raised(monkeypatch):
    import botocore.session

    monkeypatch.setattr(botocore.session.Session, "get_credentials", lambda self: None)
    status = bedrock_auth_status()
    assert status["ok"] is False
    assert "credentials" in status["error"].lower()


def test_auth_status_never_raises(monkeypatch):
    import botocore.session

    def boom(self):
        raise RuntimeError("profile is corrupt")

    monkeypatch.setattr(botocore.session.Session, "get_credentials", boom)
    status = bedrock_auth_status()
    assert status["ok"] is False
    assert "profile is corrupt" in status["error"]


# --- the bearer-token / profile conflict -----------------------------------


def test_bearer_token_plus_profile_fails_with_an_actionable_message(monkeypatch):
    """The SDK's own error names neither variable; ours names both."""
    monkeypatch.setenv(BEDROCK_BEARER_TOKEN_ENV, "some-bedrock-api-key")

    with pytest.raises(ValueError) as excinfo:
        build_llm(settings(aws_profile="default"))

    message = str(excinfo.value)
    assert BEDROCK_BEARER_TOKEN_ENV in message
    assert "AWS_PROFILE" in message


def test_bearer_token_alone_is_fine(monkeypatch):
    monkeypatch.setenv(BEDROCK_BEARER_TOKEN_ENV, "some-bedrock-api-key")
    llm = build_llm(settings(aws_profile=None))
    assert llm.describe()["auth"]["method"] == "bearer_token"


# --- endpoint selection ----------------------------------------------------


def test_invoke_api_targets_bedrock_runtime():
    llm = build_llm(settings(bedrock_api="invoke"))
    assert "bedrock-runtime" in str(llm._client.base_url)  # noqa: SLF001 - test seam


def test_mantle_api_targets_a_different_endpoint():
    llm = build_llm(settings(bedrock_api="mantle"))
    assert "bedrock-runtime" not in str(llm._client.base_url)  # noqa: SLF001


def test_region_override_is_applied_to_the_endpoint():
    llm = build_llm(settings(aws_region="eu-west-1"))
    assert "eu-west-1" in str(llm._client.base_url)  # noqa: SLF001


# --- health reporting ------------------------------------------------------


def test_health_reports_bedrock_shape(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    health = llm_health(settings())
    assert health["provider"] == "bedrock"
    assert health["configured"] is True
    assert health["api"] == "invoke"
    assert health["region"] == "us-east-1"
    assert health["auth"]["method"] == "sigv4"


def test_health_reports_missing_aws_credentials(monkeypatch):
    import botocore.session

    monkeypatch.setattr(botocore.session.Session, "get_credentials", lambda self: None)
    health = llm_health(settings())
    assert health["configured"] is False


def test_health_reports_anthropic_provider():
    health = llm_health(settings(llm_provider="anthropic", anthropic_api_key="sk-ant-x"))
    assert health["provider"] == "anthropic"
    assert health["configured"] is True
    assert "sk-ant-x" not in str(health)


def test_health_flags_an_unknown_provider():
    """Defensive branch: Settings rejects this, but a programmatic caller can
    still hand build_llm/llm_health an arbitrary object."""
    health = llm_health(_StubSettings(llm_provider="nope"))
    assert health["configured"] is False


# --- provider selection ----------------------------------------------------


def test_anthropic_provider_requires_a_key():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_llm(settings(llm_provider="anthropic", anthropic_api_key=""))


def test_anthropic_provider_builds_with_a_key():
    llm = build_llm(settings(llm_provider="anthropic", anthropic_api_key="sk-ant-x"))
    assert isinstance(llm, AnthropicLLM)


def test_settings_rejects_an_unknown_provider_at_load_time():
    """The typed setting is the first line of defence -- a typo in .env fails
    at startup rather than at the first LLM call."""
    with pytest.raises(Exception, match="bedrock"):
        Settings(llm_provider="nope")


def test_build_llm_rejects_an_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        build_llm(_StubSettings(llm_provider="nope"))


def test_both_providers_satisfy_the_llm_client_protocol():
    """The agent loop only depends on run_turn/describe/model."""
    for llm in (
        build_llm(settings()),
        build_llm(settings(llm_provider="anthropic", anthropic_api_key="sk-ant-x")),
    ):
        assert isinstance(llm.model, str) and llm.model
        assert callable(llm.run_turn)
        assert isinstance(llm.describe(), dict)
