"""Configuration validation — every error must name the variable that caused it."""

from __future__ import annotations

import pytest

from cleave import config


def test_defaults_are_sane(monkeypatch):
    cfg = config.settings()
    assert cfg.enrich_batch == 6
    assert cfg.enrich_doc_chars == 24000
    assert cfg.llm_timeout_s == 90.0
    assert cfg.gemini_model == "gemini-2.5-flash"
    assert cfg.ollama_url.startswith("http")


def test_a_non_numeric_batch_names_the_variable(monkeypatch):
    monkeypatch.setenv("CLEAVE_ENRICH_BATCH", "six")
    config.reload()
    with pytest.raises(config.ConfigError, match="CLEAVE_ENRICH_BATCH"):
        config.settings()


def test_a_zero_batch_is_rejected(monkeypatch):
    """The regression: this used to surface as ``range() arg 3 must not be zero``
    raised from the middle of enrichment."""
    monkeypatch.setenv("CLEAVE_ENRICH_BATCH", "0")
    config.reload()
    with pytest.raises(config.ConfigError, match="CLEAVE_ENRICH_BATCH"):
        config.settings()


def test_an_unknown_provider_lists_the_valid_ones(monkeypatch):
    monkeypatch.setenv("CLEAVE_LLM", "banana")
    config.reload()
    with pytest.raises(config.ConfigError) as exc:
        config.settings()
    message = str(exc.value)
    assert "CLEAVE_LLM" in message
    for valid in ("none", "ollama", "gemini"):
        assert valid in message


def test_a_url_without_a_scheme_is_rejected(monkeypatch):
    monkeypatch.setenv("CLEAVE_STT_URL", "127.0.0.1:8000")
    config.reload()
    with pytest.raises(config.ConfigError, match="CLEAVE_STT_URL"):
        config.settings()


def test_a_bad_timeout_names_the_variable(monkeypatch):
    monkeypatch.setenv("CLEAVE_LLM_TIMEOUT", "-5")
    config.reload()
    with pytest.raises(config.ConfigError, match="CLEAVE_LLM_TIMEOUT"):
        config.settings()


def test_settings_are_cached_until_reloaded(monkeypatch):
    first = config.settings()
    assert config.settings() is first
    monkeypatch.setenv("CLEAVE_ENRICH_BATCH", "3")
    assert config.settings() is first  # still cached
    config.reload()
    assert config.settings().enrich_batch == 3


def test_a_real_environment_variable_beats_the_dotenv_file(monkeypatch, tmp_path):
    """``CLEAVE_LLM=none uv run pytest`` must win over a .env that says otherwise.

    This is why ``load_dotenv`` is called with ``override=False``.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("CLEAVE_LLM=gemini\nGEMINI_MODEL=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setattr(config, "_DOTENV_LOADED", False)
    monkeypatch.setenv("CLEAVE_LLM", "none")
    config.reload()

    cfg = config.settings()

    assert cfg.llm == "none"  # the real variable wins
    assert cfg.gemini_model == "from-dotenv"  # the file still supplies what is unset


def test_the_dotenv_file_is_read_when_the_variable_is_unset(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("CLEAVE_ENRICH_DOC_CHARS=1234\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setattr(config, "_DOTENV_LOADED", False)
    monkeypatch.delenv("CLEAVE_ENRICH_DOC_CHARS", raising=False)
    config.reload()

    assert config.settings().enrich_doc_chars == 1234
