"""Two providers, and choosing between them per request.

The point of a second provider is comparing models for accuracy, so the thing
under test is not "OpenRouter works" -- it is that a *choice* reaches the model
that actually gets called. A picker whose selection is quietly ignored is worse
than no picker, because the comparison it produces is a lie.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from catalog import ModelCatalogue
from config import Settings
from llm import ModelChoice, ModelPool

pytestmark = pytest.mark.anyio


def settings(**overrides) -> Settings:
    # Discovery off unless a test asks: it is an AWS call, and the suite does
    # not make those. See `conftest.test_settings`.
    overrides.setdefault("bedrock_discover", False)
    return Settings(db_schema="browser_test", _env_file=None, **overrides)


# --- resolving a choice ----------------------------------------------------


def test_no_choice_means_the_configured_one():
    """A request that did not ask carries nothing, so the deployment decides."""
    resolved = ModelChoice.resolve(settings(), None, None)

    assert resolved == ModelChoice("bedrock", settings().llm_repair_model)


def test_naming_a_model_switches_provider_and_model():
    resolved = ModelChoice.resolve(
        settings(), "openrouter", "anthropic/claude-sonnet-4.5"
    )

    assert resolved.provider == "openrouter"
    assert resolved.model == "anthropic/claude-sonnet-4.5"
    assert resolved.describe() == "openrouter:anthropic/claude-sonnet-4.5"


def test_naming_only_bedrock_takes_its_configured_model():
    """Useful there, because Bedrock has one configured default."""
    resolved = ModelChoice.resolve(settings(), "bedrock", None)

    assert resolved.model == settings().llm_repair_model


def test_naming_only_openrouter_leaves_the_model_empty():
    """A provider fronting hundreds of models has no sensible default to pick
    on your behalf, and guessing one is how a comparison silently runs on
    something nobody chose."""
    resolved = ModelChoice.resolve(settings(openrouter_model=""), "openrouter", None)

    assert resolved.provider == "openrouter"
    assert resolved.model == ""


def test_an_unknown_provider_is_refused_by_name():
    with pytest.raises(ValueError, match="not a provider this build can reach"):
        ModelChoice.resolve(settings(), "definitely-not-a-provider", "x")


# --- the pool --------------------------------------------------------------


def test_the_same_choice_reuses_one_client():
    pool = ModelPool(settings(openrouter_api_key="k"))
    choice = ModelChoice("openrouter", "a/b")

    assert pool.for_choice(choice) is pool.for_choice(choice)


def test_two_choices_are_two_clients():
    """A person comparing two models has both in flight by definition, so a
    single cached slot would hand the second request the first one's model."""
    pool = ModelPool(settings(openrouter_api_key="k"))

    first = pool.for_choice(ModelChoice("openrouter", "a/b"))
    second = pool.for_choice(ModelChoice("openrouter", "c/d"))

    assert first is not second
    assert first.model == "a/b" and second.model == "c/d"


def test_an_injected_client_answers_for_every_choice():
    """The test seam. Honouring it only for the default would make a real
    provider call the moment a test named another model."""

    class Scripted:
        model = "scripted"

    scripted = Scripted()
    pool = ModelPool(settings(), client=scripted)

    assert pool.client is scripted
    assert pool.for_choice(ModelChoice("openrouter", "a/b")) is scripted


def test_openrouter_without_a_key_says_so_rather_than_failing_at_the_call():
    pool = ModelPool(settings(openrouter_api_key=""))

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not set"):
        pool.for_choice(ModelChoice("openrouter", "a/b"))


def test_the_prompt_cache_is_only_sent_to_bedrock():
    """`ChatOpenAI` rejects an unknown keyword outright, so sending Bedrock's
    cache flag to OpenRouter would fail every call rather than miss a saving."""
    from llm import build_llm

    bedrock = build_llm(settings(), "us.anthropic.claude-opus-5", "bedrock")
    router = build_llm(settings(openrouter_api_key="k"), "a/b", "openrouter")

    assert bedrock.provider == "bedrock"
    assert router.provider == "openrouter"
    assert "auth" in bedrock.describe(), "Bedrock reports how it authenticated"
    assert "auth" not in router.describe(), "OpenRouter authenticates with a key"


# --- the catalogue ---------------------------------------------------------


OPENROUTER_REPLY = {
    "data": [
        {
            "id": "anthropic/claude-sonnet-4.5",
            "name": "Anthropic: Claude Sonnet 4.5",
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
            "context_length": 200000,
        },
        {
            "id": "meta-llama/llama-3.3-70b-instruct",
            "name": "Meta: Llama 3.3 70B",
            "pricing": {"prompt": "0.00000012", "completion": "0.0000003"},
            "context_length": 131072,
        },
        {"id": "", "name": "nameless, and skipped"},
    ]
}


async def test_the_catalogue_offers_both_providers():
    async def fetch():
        return OPENROUTER_REPLY

    catalogue = await ModelCatalogue(
        settings(openrouter_api_key="k"), fetch=fetch
    ).read()

    providers = {model.provider for model in catalogue.models}
    assert providers == {"bedrock", "openrouter"}
    assert catalogue.problems == {}


async def test_prices_arrive_per_million_not_per_token():
    """OpenRouter quotes per token, as a string, because the number is too
    small to survive being read as anything else. Every price in this codebase
    and on every provider's pricing page is per million."""

    async def fetch():
        return OPENROUTER_REPLY

    catalogue = await ModelCatalogue(
        settings(openrouter_api_key="k"), fetch=fetch
    ).read()
    sonnet = next(m for m in catalogue.models if m.id == "anthropic/claude-sonnet-4.5")

    assert sonnet.prices == (3.0, 15.0)
    assert sonnet.context == 200000


async def test_a_model_with_no_id_is_skipped():
    async def fetch():
        return OPENROUTER_REPLY

    catalogue = await ModelCatalogue(
        settings(openrouter_api_key="k"), fetch=fetch
    ).read()

    assert all(model.id for model in catalogue.models)


async def test_a_missing_key_is_reported_as_the_reason_not_an_absence():
    """"OpenRouter is missing" with no explanation is a gap a person fills in
    with a guess."""
    catalogue = await ModelCatalogue(settings(openrouter_api_key="")).read()

    assert "OPENROUTER_API_KEY is not set" in catalogue.problems["openrouter"]
    assert all(model.provider == "bedrock" for model in catalogue.models)


async def test_a_provider_that_will_not_answer_still_leaves_a_usable_picker():
    async def fetch():
        raise RuntimeError("connection refused")

    catalogue = await ModelCatalogue(
        settings(openrouter_api_key="k"), fetch=fetch
    ).read()

    assert "connection refused" in catalogue.problems["openrouter"]
    assert [model.provider for model in catalogue.models] == ["bedrock"] * 3


async def test_the_catalogue_is_fetched_once_and_reused():
    """Reading it means asking OpenRouter for several hundred models, which
    belongs behind a cache rather than in front of a page load."""
    calls = []

    async def fetch():
        calls.append(1)
        return OPENROUTER_REPLY

    catalogue = ModelCatalogue(settings(openrouter_api_key="k"), fetch=fetch)
    await catalogue.read()
    await catalogue.read()
    await catalogue.read()

    assert len(calls) == 1


async def test_bedrock_models_carry_a_price_only_when_the_table_lists_one():
    """Rendering the table's default as though it were this model's price is
    how somebody budgets a batch against a figure nobody checked."""
    catalogue = await ModelCatalogue(
        settings(bedrock_models=["us.anthropic.claude-opus-5", "us.meta.llama-99"])
    ).read()

    by_id = {model.id: model for model in catalogue.models}
    assert by_id["us.anthropic.claude-opus-5"].prices == (15.0, 75.0)
    assert by_id["us.meta.llama-99"].prices is None


# --- pricing ---------------------------------------------------------------


def test_a_provider_price_beats_the_hand_maintained_table():
    """The table lists five Claude models. OpenRouter fronts hundreds, so
    without this every session on anything else would be costed from a default
    that happens to be Sonnet's."""
    from pricing import price_of

    usage = {"input_tokens": 1_000_000, "output_tokens": 0}

    assert price_of("meta-llama/llama-3.3-70b", usage) == pytest.approx(3.0)
    assert price_of(
        "meta-llama/llama-3.3-70b", usage, rates=(0.12, 0.30)
    ) == pytest.approx(0.12)


# --- the endpoint ----------------------------------------------------------


@pytest.fixture
def client(db_settings, db_engine, tmp_path, monkeypatch):
    from conftest import authenticate, build_app

    app = build_app(db_settings, tmp_path, monkeypatch)
    with TestClient(app) as test_client:
        yield authenticate(test_client)


def test_the_endpoint_lists_what_can_be_reached(client: TestClient):
    body = client.get("/api/models").json()

    assert "bedrock" in body["providers"] and "openrouter" in body["providers"]
    assert body["default"]["provider"] in body["providers"]
    assert any(model["provider"] == "bedrock" for model in body["models"])


def test_naming_a_provider_with_no_model_is_refused_with_the_reason(
    client: TestClient,
):
    response = client.post("/api/models/check", json={"provider": "openrouter"})

    assert response.status_code == 422
    assert "Name a model" in response.json()["detail"]


def test_an_unknown_provider_is_refused_by_the_endpoint(client: TestClient):
    response = client.post(
        "/api/models/check", json={"provider": "nope", "model": "x"}
    )

    assert response.status_code == 422
    assert "not a provider this build can reach" in response.json()["detail"]


# --- a provider that can be switched off entirely -------------------------
#
# Asked for by an operator taking this into a company: "when I disable
# OpenRouter, it shouldn't make any call from my code". So the test is not
# that the provider is hidden -- it is that every path which could produce a
# request refuses, including the one nobody thinks of, which is the model
# catalogue: reading it is itself an HTTP request to openrouter.ai.


def switched_off():
    return settings(openrouter_enabled=False, openrouter_api_key="or-key-that-is-set")


def test_naming_the_provider_on_a_request_is_refused():
    from llm import ModelChoice

    with pytest.raises(ValueError, match="OPENROUTER_ENABLED"):
        ModelChoice.resolve(switched_off(), "openrouter", "anthropic/claude-sonnet-4.5")


def test_building_a_client_for_it_is_refused():
    from llm import build_llm

    with pytest.raises(ValueError, match="OPENROUTER_ENABLED"):
        build_llm(switched_off(), "anthropic/claude-sonnet-4.5", "openrouter")


def test_the_layer_that_would_construct_the_http_client_refuses_too():
    """Defence in depth, and the one that decides whether a packet leaves: a
    caller that forgot to ask still gets nothing."""
    from chat import chat_model

    with pytest.raises(ValueError, match="OPENROUTER_ENABLED"):
        chat_model(switched_off(), "anthropic/claude-sonnet-4.5", "openrouter")


async def test_the_catalogue_is_never_fetched():
    """The path most easily forgotten. Listing models is a request to
    openrouter.ai, made in front of a dashboard page load, and a key being
    present is not permission to use it."""
    from catalog import ModelCatalogue

    def must_not_be_called():
        raise AssertionError("the OpenRouter catalogue was fetched")

    catalogue = ModelCatalogue(switched_off(), fetch=must_not_be_called)
    listing = await catalogue.read()

    assert [m for m in listing.models if m.provider == "openrouter"] == []
    assert "OPENROUTER_ENABLED" in listing.problems["openrouter"]


async def test_a_price_lookup_is_not_a_way_round_it():
    """`prices_for` reads the same catalogue, from the costing layer, which is
    a long way from anything that looks like choosing a provider."""
    from catalog import ModelCatalogue

    def must_not_be_called():
        raise AssertionError("the OpenRouter catalogue was fetched")

    catalogue = ModelCatalogue(switched_off(), fetch=must_not_be_called)

    assert await catalogue.prices_for("anthropic/claude-sonnet-4.5") is None


def test_why_it_is_missing_is_said_rather_than_left_to_be_guessed():
    """A picker with no OpenRouter in it and no reason given looks broken."""
    reason = settings(openrouter_enabled=False).openrouter_available

    assert reason is False


def test_defaulting_to_a_forbidden_provider_is_refused_at_startup():
    """Rather than failing one call at a time with a message about a flag."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="LLM_PROVIDER"):
        settings(llm_provider="openrouter", openrouter_enabled=False)


# --- Bedrock, discovered rather than hand-listed --------------------------


def _summary(model_id: str, *, kinds: list[str], name: str = "", vendor: str = "Meta",
             status: str = "ACTIVE") -> dict:
    return {
        "modelId": model_id,
        "modelName": name or model_id,
        "providerName": vendor,
        "inferenceTypesSupported": kinds,
        "modelLifecycle": {"status": status},
    }


def test_an_on_demand_model_is_offered_by_its_own_id():
    from catalog import _bedrock_info

    info = _bedrock_info(_summary("meta.llama3-8b", kinds=["ON_DEMAND"], name="Llama 3 8B"), {})

    assert info.id == "meta.llama3-8b"
    assert info.name == "Meta Llama 3 8B"


def test_a_profile_only_model_is_offered_by_the_id_that_can_run_it():
    """Its own id cannot be invoked, so offering it would put an entry in the
    dropdown that fails the moment somebody picks it."""
    from catalog import _bedrock_info

    info = _bedrock_info(
        _summary("meta.llama4-scout", kinds=["INFERENCE_PROFILE"]),
        {"meta.llama4-scout": "us.meta.llama4-scout"},
    )

    assert info.id == "us.meta.llama4-scout"


def test_a_model_with_no_way_to_invoke_it_on_demand_is_left_out():
    """Available only against a provisioned throughput commitment. Choosing it
    fails with a billing error nobody reading a dropdown could predict."""
    from catalog import _bedrock_info

    assert _bedrock_info(_summary("meta.llama-provisioned", kinds=["PROVISIONED"]), {}) is None
    assert _bedrock_info(_summary("meta.llama4", kinds=["INFERENCE_PROFILE"]), {}) is None


def test_a_legacy_model_is_offered_and_says_so():
    from catalog import _bedrock_info

    info = _bedrock_info(
        _summary("ai21.jamba", kinds=["ON_DEMAND"], name="Jamba", vendor="AI21", status="LEGACY"),
        {},
    )

    assert "(legacy)" in info.name


def test_a_refusal_names_the_permission_and_keeps_the_configured_list():
    """The objection that kept this as configuration for so long. It is a
    degradation now, not a reason to show nothing."""
    from catalog import _bedrock_refusal

    said = _bedrock_refusal(Exception("AccessDeniedException: not authorized to perform"))

    assert "bedrock:ListFoundationModels" in said
    assert "BEDROCK_MODELS are still" in said
    assert "BEDROCK_DISCOVER=false" in said


async def test_discovery_off_offers_exactly_what_was_configured():
    from catalog import ModelCatalogue

    listing = await ModelCatalogue(
        settings(
            bedrock_discover=False,
            openrouter_enabled=False,
            bedrock_models=["us.anthropic.claude-sonnet-5"],
        )
    ).read()

    assert [m.id for m in listing.models] == ["us.anthropic.claude-sonnet-5"]
