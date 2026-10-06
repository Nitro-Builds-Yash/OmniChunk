"""Shared fixtures and test isolation.

Two things this file exists to guarantee.

Paths resolve from ``__file__`` rather than the working directory. The suite
hard-coded ``"tests/fixtures"``, so running ``pytest tests/test_cleave.py`` from
anywhere but the repo root failed on a missing file rather than on a real
assertion.

And no test may reach a paid API. ``config.settings()`` now loads the repo's
``.env``, which on a developer machine can hold a live ``GEMINI_API_KEY``; every
test therefore starts from a scrubbed environment with the provider forced off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cleave import config, http
from cleave.web import jobs as jobs_mod
from cleave.web import search as search_mod

FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: Everything ``config.Settings`` reads. Cleared so a developer's shell or
#: ``.env`` cannot change what the suite tests.
_ENV_VARS = (
    "CLEAVE_LLM",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "CLEAVE_OLLAMA_URL",
    "CLEAVE_OLLAMA_MODEL",
    "CLEAVE_LLM_TIMEOUT",
    "CLEAVE_ENRICH_BATCH",
    "CLEAVE_ENRICH_DOC_CHARS",
    "CLEAVE_STT_URL",
    "CLEAVE_LOG_LEVEL",
    "CLEAVE_OFFLINE_FALLBACK",
)


@pytest.fixture(scope="session")
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch):
    """Every test starts from a known, offline configuration."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CLEAVE_LLM", "none")
    monkeypatch.setenv("CLEAVE_OFFLINE_FALLBACK", "0")
    # Do not read the repo's real .env: it may hold a billable key.
    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    config.reload()
    yield
    config.reload()
    http.reset_client()


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the job store at a temp dir so web tests never touch data/jobs/."""
    data = tmp_path / "jobs"
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(jobs_mod, "DATA", data)
    saved = dict(jobs_mod.JOBS)
    jobs_mod.JOBS.clear()
    search_mod.clear_cache()
    yield data
    jobs_mod.JOBS.clear()
    jobs_mod.JOBS.update(saved)
    search_mod.clear_cache()


@pytest.fixture
def client(tmp_data_dir: Path):
    """A TestClient with lifespan run, so startup behaviour is exercised too."""
    from fastapi.testclient import TestClient

    from cleave.app import app

    with TestClient(app) as c:
        yield c
