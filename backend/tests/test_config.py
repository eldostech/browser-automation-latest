"""Settings loaded the way they are loaded in production: from a .env file.

This file exists because of a shipped bug. Every other test built ``Settings``
in Python with real list objects, so nothing exercised the dotenv path -- and
pydantic-settings runs ``json.loads()`` on ``list[str]`` fields *before* any
validator, which made a perfectly ordinary comma-separated
``CORS_ORIGINS`` crash the backend at import time with
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


def test_comma_separated_lists_load_from_a_dotenv_file(tmp_path):
    """A list setting is written comma-separated, not as JSON.

    Nobody writes a JSON array in a .env file, and pydantic's default list
    parsing demands one -- which crashed the backend at import time with a
    message about JSON naming a variable the operator had written perfectly
    reasonably.
    """
    env = write_env(
        tmp_path,
        "CORS_ORIGINS=http://localhost:5173,https://app.example.com\n",
    )
    settings = Settings(_env_file=env)
    assert settings.cors_origins == ["http://localhost:5173", "https://app.example.com"]


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
    assert settings.cors_origins
    assert settings.cors_origins
    assert settings.llm_repair_model.startswith(("us.", "eu.", "apac.", "global."))


def test_whitespace_around_entries_is_trimmed(tmp_path):
    env = write_env(tmp_path, "CORS_ORIGINS= example.com , shop.test ,\n")
    assert Settings(_env_file=env).cors_origins == ["example.com", "shop.test"]


def test_json_array_form_also_works(tmp_path):
    """NoDecode disables pydantic's JSON handling, so we accept it ourselves --
    a value written either way behaves the same."""
    env = write_env(tmp_path, 'CORS_ORIGINS=["example.com", "shop.test"]\n')
    assert Settings(_env_file=env).cors_origins == ["example.com", "shop.test"]


def test_a_single_domain_is_still_a_list(tmp_path):
    env = write_env(tmp_path, "CORS_ORIGINS=example.com\n")
    assert Settings(_env_file=env).cors_origins == ["example.com"]


# --- other .env round-trips ------------------------------------------------


def test_scalars_and_bools_load_from_dotenv(tmp_path):
    env = write_env(
        tmp_path,
        "REPLAY_STEP_TIMEOUT=45.5\n"
        "REPLAY_FAILURE_STREAK_LIMIT=7\n"
        "REPLAY_HEALING_ENABLED=false\n"
        "BROWSER_HEADLESS=true\n",
    )
    settings = Settings(_env_file=env)
    assert settings.replay_step_timeout == 45.5
    assert settings.replay_failure_streak_limit == 7
    assert settings.replay_healing_enabled is False
    assert settings.browser_headless is True


def test_blank_optional_values_become_none(tmp_path):
    env = write_env(tmp_path, "AWS_PROFILE=\nAWS_REGION=\nMCP_STORAGE_STATE=\n")
    settings = Settings(_env_file=env)
    assert settings.aws_profile is None
    assert settings.aws_region is None


def test_bedrock_settings_load_from_dotenv(tmp_path):
    env = write_env(
        tmp_path,
        "LLM_PROVIDER=bedrock\n"
        "LLM_REPAIR_MODEL=eu.anthropic.claude-haiku-4-5-20251001-v1:0\n"
        "BEDROCK_API=mantle\n"
        "AWS_REGION=eu-west-1\n"
        "AWS_PROFILE=work\n",
    )
    settings = Settings(_env_file=env)
    assert settings.aws_region == "eu-west-1"
    assert settings.aws_profile == "work"
    # A colon in the value must survive -- dotenv splits on '=', not ':'.
    assert settings.llm_repair_model.endswith("-v1:0")


def test_a_typo_in_a_typed_setting_fails_at_load(tmp_path):
    env = write_env(tmp_path, "REPLAY_FAILURE_STREAK_LIMIT=not-a-number\n")
    with pytest.raises(Exception, match="replay_failure_streak_limit"):
        Settings(_env_file=env)


def test_a_setting_that_no_longer_exists_is_ignored_not_fatal(tmp_path):
    """An old .env keeps working.

    ANTHROPIC_API_KEY and BEDROCK_API are gone; a stale line for either must
    not stop the backend starting.
    """
    env = write_env(tmp_path, "ANTHROPIC_API_KEY=sk-ant-stale\nBEDROCK_API=mantle\n")
    settings = Settings(_env_file=env)

    assert not hasattr(settings, "anthropic_api_key")
    assert not hasattr(settings, "bedrock_api")


def test_a_provider_value_from_an_older_version_starts_on_the_default(tmp_path):
    """`LLM_PROVIDER` existed, was removed when this went Bedrock-only, and is
    back now that there are two providers again.

    An operator upgrading across all of that may still have
    `LLM_PROVIDER=anthropic` in a file nobody has opened in months, and a
    backend that will not start because of one dead line is worse than one that
    starts on its default and says so in the log.
    """
    env = write_env(tmp_path, "LLM_PROVIDER=anthropic\n")

    assert Settings(_env_file=env).llm_provider == "bedrock"


def test_a_provider_typo_still_fails_at_load(tmp_path):
    """Forgiving one historical value is not the same as accepting anything."""
    env = write_env(tmp_path, "LLM_PROVIDER=bedrok\n")

    with pytest.raises(Exception, match="llm_provider"):
        Settings(_env_file=env)


def test_defaults_apply_with_no_dotenv_file(tmp_path):
    settings = Settings(_env_file=str(tmp_path / "does-not-exist.env"))
    assert settings.cors_origins == ["http://localhost:5173"]
    assert settings.llm_repair_model.startswith(("us.", "eu.", "apac.", "global."))


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

    Three deliberate exceptions, all read by Vite rather than the backend, and
    all belonging in this file because they are the same thing an operator has
    to fill in: ``VITE_API_BASE``, ``FRONTEND_PORT`` and ``BACKEND_ORIGIN``.
    """
    import re
    from pathlib import Path

    from config import Settings

    example = (
        Path(__file__).resolve().parents[2] / ".env.example"
    ).read_text(encoding="utf-8")
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", example, re.MULTILINE))

    known = {name.upper() for name in Settings.model_fields} | {
        # Read by Vite from this same file; see vite.config.ts.
        "VITE_API_BASE",
        "FRONTEND_PORT",
        "BACKEND_ORIGIN",
    }
    stale = sorted(documented - known)
    assert not stale, (
        f"these appear in .env.example but are not settings any more: {stale}"
    )


def test_vite_reads_the_same_env_file_the_backend_does():
    """`VITE_API_BASE` lives in the repository's .env, so Vite must look there.

    Vite's env directory defaults to its own root -- `frontend/` -- so the
    variable documented in this project's `.env` was read by the backend,
    ignored by Vite, and appeared not to work. `envDir` in vite.config.ts is
    what makes the documented location the real one.

    Asserted here rather than in the frontend because this is a statement about
    two config files agreeing, and only one test suite runs in CI on every
    push.
    """
    from pathlib import Path

    config = (
        Path(__file__).resolve().parents[2] / "frontend" / "vite.config.ts"
    ).read_text(encoding="utf-8")

    assert "envDir" in config, (
        "vite.config.ts must set envDir to the repository root, or VITE_API_BASE "
        "in .env is silently ignored"
    )
    # The proxy target should follow the backend's own PORT rather than
    # hard-coding a second copy of it.
    assert "env.PORT" in config, (
        "the dev proxy should derive its target from PORT in the same .env, so "
        "moving the backend does not leave the proxy behind"
    )
