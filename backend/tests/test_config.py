"""Settings loaded the way they are loaded in production: from a .env file.

This file exists because of a shipped bug. Every other test built ``Settings``
in Python with real list objects, so nothing exercised the dotenv path -- and
pydantic-settings runs ``json.loads()`` on ``list[str]`` fields *before* any
validator, which made a perfectly ordinary comma-separated
``AGENT_ALLOWED_DOMAINS`` crash the backend at import time with
``SettingsError``. The fields are annotated ``NoDecode`` to prevent that; these
tests hold that behaviour in place.
"""

from __future__ import annotations

import pytest

from config import Settings, _split_csv


def write_env(tmp_path, body: str):
    path = tmp_path / ".env"
    path.write_text(body, encoding="utf-8")
    return str(path)


# --- the regression --------------------------------------------------------


def test_comma_separated_domains_load_from_a_dotenv_file(tmp_path):
    """The exact shape of .env.example must not raise."""
    env = write_env(
        tmp_path,
        "AGENT_ALLOWED_DOMAINS=example.com,*.example.com\n"
        "CORS_ORIGINS=http://localhost:5173\n",
    )
    settings = Settings(_env_file=env)
    assert settings.agent_allowed_domains == ["example.com", "*.example.com"]
    assert settings.cors_origins == ["http://localhost:5173"]


def test_the_shipped_env_example_parses(tmp_path):
    """Parse the real .env.example, so the template can never drift into a
    state that crashes a fresh clone on first boot."""
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / ".env.example"
    body = example.read_text(encoding="utf-8")
    # Strip comments the way a user does when copying it to .env.
    active = "\n".join(
        line for line in body.splitlines() if line.strip() and not line.strip().startswith("#")
    )
    settings = Settings(_env_file=write_env(tmp_path, active))
    assert settings.agent_allowed_domains
    assert settings.cors_origins
    assert settings.llm_model.startswith(("us.", "eu.", "apac.", "global."))


def test_whitespace_around_entries_is_trimmed(tmp_path):
    env = write_env(tmp_path, "AGENT_ALLOWED_DOMAINS= example.com , shop.test ,\n")
    assert Settings(_env_file=env).agent_allowed_domains == ["example.com", "shop.test"]


def test_json_array_form_also_works(tmp_path):
    """NoDecode disables pydantic's JSON handling, so we accept it ourselves --
    a value written either way behaves the same."""
    env = write_env(tmp_path, 'AGENT_ALLOWED_DOMAINS=["example.com", "shop.test"]\n')
    assert Settings(_env_file=env).agent_allowed_domains == ["example.com", "shop.test"]


def test_a_single_domain_is_still_a_list(tmp_path):
    env = write_env(tmp_path, "AGENT_ALLOWED_DOMAINS=example.com\n")
    assert Settings(_env_file=env).agent_allowed_domains == ["example.com"]


# --- other .env round-trips ------------------------------------------------


def test_scalars_and_bools_load_from_dotenv(tmp_path):
    env = write_env(
        tmp_path,
        "AGENT_MAX_STEPS=7\n"
        "AGENT_TIMEOUT_SECONDS=45.5\n"
        "AGENT_REQUIRE_APPROVAL=false\n"
        "MCP_HEADLESS=true\n",
    )
    settings = Settings(_env_file=env)
    assert settings.agent_max_steps == 7
    assert settings.agent_timeout_seconds == 45.5
    assert settings.agent_require_approval is False
    assert settings.mcp_headless is True


def test_blank_optional_values_become_none(tmp_path):
    env = write_env(tmp_path, "AWS_PROFILE=\nAWS_REGION=\nMCP_STORAGE_STATE=\n")
    settings = Settings(_env_file=env)
    assert settings.aws_profile is None
    assert settings.aws_region is None
    assert settings.mcp_storage_state is None


def test_bedrock_settings_load_from_dotenv(tmp_path):
    env = write_env(
        tmp_path,
        "LLM_PROVIDER=bedrock\n"
        "LLM_MODEL=eu.anthropic.claude-haiku-4-5-20251001-v1:0\n"
        "BEDROCK_API=mantle\n"
        "AWS_REGION=eu-west-1\n"
        "AWS_PROFILE=work\n",
    )
    settings = Settings(_env_file=env)
    assert settings.aws_region == "eu-west-1"
    assert settings.aws_profile == "work"
    # A colon in the value must survive -- dotenv splits on '=', not ':'.
    assert settings.llm_model.endswith("-v1:0")


def test_a_typo_in_a_typed_setting_fails_at_load(tmp_path):
    env = write_env(tmp_path, "AGENT_MAX_STEPS=not-a-number\n")
    with pytest.raises(Exception, match="agent_max_steps"):
        Settings(_env_file=env)


def test_a_setting_that_no_longer_exists_is_ignored_not_fatal(tmp_path):
    """An old .env keeps working.

    LLM_PROVIDER, ANTHROPIC_API_KEY and BEDROCK_API are gone; a stale line for
    any of them must not stop the backend starting.
    """
    env = write_env(
        tmp_path,
        "LLM_PROVIDER=anthropic\nANTHROPIC_API_KEY=sk-ant-stale\nBEDROCK_API=mantle\n",
    )
    settings = Settings(_env_file=env)

    assert not hasattr(settings, "llm_provider")
    assert not hasattr(settings, "anthropic_api_key")
    assert not hasattr(settings, "bedrock_api")


def test_defaults_apply_with_no_dotenv_file(tmp_path):
    settings = Settings(_env_file=str(tmp_path / "does-not-exist.env"))
    assert settings.agent_allowed_domains == ["example.com", "*.example.com"]
    assert settings.llm_model.startswith(("us.", "eu.", "apac.", "global."))


# --- the parser itself -----------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("a.com,b.com", ["a.com", "b.com"]),
        ("a.com, b.com", ["a.com", "b.com"]),
        ('["a.com", "b.com"]', ["a.com", "b.com"]),
        (["a.com", "b.com"], ["a.com", "b.com"]),
        ("a.com,,b.com,", ["a.com", "b.com"]),
        ("", []),
        (None, []),
        ("*", ["*"]),
        # Malformed JSON falls back to CSV rather than raising.
        ('["a.com"', ['["a.com"']),
    ],
)
def test_split_csv(raw, expected):
    assert _split_csv(raw) == expected


# --- the example file is the documentation ---------------------------------


def test_every_setting_appears_in_the_env_example():
    """A setting nobody can discover may as well not be configurable.

    `.env.example` is where an operator finds out what they can change, so a
    field added without a line here is invisible. Eight had already drifted out
    of it by the time this was written, including the two AWS variables the
    header explicitly promised were documented "below" -- in a section that no
    longer existed.

    A commented-out line counts: some settings are best left unset, and showing
    the name with its default is the documentation.
    """
    import re
    from pathlib import Path

    from config import Settings

    example = (
        Path(__file__).resolve().parents[2] / ".env.example"
    ).read_text(encoding="utf-8")
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", example, re.MULTILINE))

    missing = sorted({name.upper() for name in Settings.model_fields} - documented)
    assert not missing, (
        "these settings exist but are not in .env.example, so nobody will find "
        f"them: {missing}"
    )


def test_the_env_example_mentions_nothing_that_is_not_a_setting():
    """The reverse drift: a line for a setting that has since been removed.

    ``VITE_API_BASE`` is the one deliberate exception -- it is read by Vite at
    build time, not by the backend, and it belongs in the same file because it
    is the same thing an operator has to fill in.
    """
    import re
    from pathlib import Path

    from config import Settings

    example = (
        Path(__file__).resolve().parents[2] / ".env.example"
    ).read_text(encoding="utf-8")
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", example, re.MULTILINE))

    known = {name.upper() for name in Settings.model_fields} | {"VITE_API_BASE"}
    stale = sorted(documented - known)
    assert not stale, (
        f"these appear in .env.example but are not settings any more: {stale}"
    )
