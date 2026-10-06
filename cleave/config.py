"""Every environment variable Cleave reads, validated once, in one place.

Two problems motivated this module.

The first is that ``.env`` was documented in the user guide and shipped as
``.env.example``, but nothing ever loaded it — the values were read straight
from ``os.environ``, so a user who filled the file in got silence rather than a
configured provider. ``settings()`` loads it now.

The second is that the numeric settings were parsed at import time with a bare
``int(os.environ.get(...))``. ``CLEAVE_ENRICH_BATCH=0`` did not fail there; it
failed much later and much worse, as ``range() arg 3 must not be zero`` raised
from the middle of enrichment, naming nothing the user had set. Validation here
is eager and every message names the variable that caused it.

Settings are cached rather than frozen into module constants. ``CLEAVE_LLM`` has
always been read per call so a test could switch providers without reimporting,
and the rest now behave the same way: change the environment, call ``reload()``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

VALID_LLM = ("", "none", "ollama", "gemini")

#: Set once ``.env`` has been consulted. Tests flip it to keep a developer's
#: real ``.env`` — which may hold a live, billable API key — out of the suite.
_DOTENV_LOADED = False


class ConfigError(ValueError):
    """A setting is present but unusable. The message always names the variable."""


@dataclass(frozen=True, slots=True)
class Settings:
    """The resolved configuration for one run."""

    llm: str  # "" (auto) | none | ollama | gemini
    gemini_api_key: str
    gemini_model: str
    ollama_url: str
    ollama_model: str
    llm_timeout_s: float
    enrich_batch: int
    enrich_doc_chars: int
    stt_url: str
    video_url: str = "http://127.0.0.1:8001"
    log_level: str = "INFO"
    offline_fallback: bool = True
    # Boundary scoring weights:
    weight_semantic: float = 1.0
    weight_structure: float = 1.2
    weight_temporal: float = 1.0
    weight_visual: float = 0.9
    weight_ocr: float = 0.8
    weight_graph: float = 0.8
    weight_consensus: float = 1.1
    weight_relationship_loss: float = 1.5
    weight_fragmentation: float = 0.6
    # Chunking token budgets:
    min_chunk_tokens: int = 80
    target_chunk_tokens: int = 500
    max_chunk_tokens: int = 700
    tabular_target_tokens: int = 900
    tabular_max_tokens: int = 1200
    completeness_threshold: float = 0.7
    # Video pipeline settings:
    #: When True, synthetic fallback is allowed even without CLEAVE_OFFLINE_FALLBACK.
    #: Set CLEAVE_EVALUATION_MODE=1 in test/evaluation environments only.
    evaluation_mode: bool = False
    #: When False, synthetic fallback raises VideoWorkerUnavailable in production.
    #: Set CLEAVE_ALLOW_SYNTHETIC_FALLBACK=0 to enforce real worker requirement.
    allow_synthetic_fallback: bool = True
    #: Boundary signal weight for a shot-only camera cut (vs 0.85 for a scene change).
    weight_shot_change: float = 0.35
    #: Minimum boundary score for speaker change to force a multimodal event split.
    speaker_boundary_threshold: float = 0.55


def _load_dotenv_once() -> None:
    """Read ``.env`` into the environment, without overriding a real variable.

    ``override=False`` is the load-bearing part: ``CLEAVE_LLM=none uv run pytest``
    is the documented way to run the suite offline, and a ``.env`` that happened
    to set ``CLEAVE_LLM=gemini`` must not win over it.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    try:
        from dotenv import load_dotenv  # noqa: PLC0415 — optional at import time

        if ENV_FILE.exists():
            load_dotenv(ENV_FILE, override=False)
            log.debug("loaded settings from %s", ENV_FILE)
    except ImportError:  # pragma: no cover - python-dotenv is a declared dep
        log.debug("python-dotenv not installed; reading os.environ only")


def _str_env(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not an integer") from None
    if value < minimum:
        raise ConfigError(f"{name}={value} must be >= {minimum}")
    return value


def _float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not a number") from None
    if value < minimum:
        raise ConfigError(f"{name}={value} must be >= {minimum}")
    return value


def _url_env(name: str, default: str) -> str:
    value = _str_env(name, default).rstrip("/")
    if not value.startswith(("http://", "https://")):
        raise ConfigError(f"{name}={value!r} must start with http:// or https://")
    return value


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Read, validate and cache the environment. Call ``reload()`` after changing it."""
    _load_dotenv_once()

    llm = _str_env("CLEAVE_LLM", "").lower()
    if llm not in VALID_LLM:
        valid = ", ".join(v for v in VALID_LLM if v)
        raise ConfigError(f"CLEAVE_LLM={llm!r} is not one of: {valid}")

    return Settings(
        llm=llm,
        gemini_api_key=_str_env("GEMINI_API_KEY", ""),
        gemini_model=_str_env("GEMINI_MODEL", "gemini-2.5-flash"),
        ollama_url=_url_env("CLEAVE_OLLAMA_URL", "http://127.0.0.1:11434"),
        ollama_model=_str_env("CLEAVE_OLLAMA_MODEL", ""),
        llm_timeout_s=_float_env("CLEAVE_LLM_TIMEOUT", 90.0, minimum=1.0),
        enrich_batch=_int_env("CLEAVE_ENRICH_BATCH", 6, minimum=1),
        enrich_doc_chars=_int_env("CLEAVE_ENRICH_DOC_CHARS", 24000, minimum=500),
        stt_url=_url_env("CLEAVE_STT_URL", "http://127.0.0.1:8000"),
        video_url=_url_env("CLEAVE_VIDEO_URL", "http://127.0.0.1:8001"),
        log_level=_str_env("CLEAVE_LOG_LEVEL", "INFO").upper(),
        offline_fallback=_str_env("CLEAVE_OFFLINE_FALLBACK", "1").lower()
        not in ("0", "false", "no"),
        weight_semantic=_float_env("CLEAVE_WEIGHT_SEMANTIC", 1.0),
        weight_structure=_float_env("CLEAVE_WEIGHT_STRUCTURE", 1.2),
        weight_temporal=_float_env("CLEAVE_WEIGHT_TEMPORAL", 1.0),
        weight_visual=_float_env("CLEAVE_WEIGHT_VISUAL", 0.9),
        weight_ocr=_float_env("CLEAVE_WEIGHT_OCR", 0.8),
        weight_graph=_float_env("CLEAVE_WEIGHT_GRAPH", 0.8),
        weight_consensus=_float_env("CLEAVE_WEIGHT_CONSENSUS", 1.1),
        weight_relationship_loss=_float_env("CLEAVE_WEIGHT_RELATIONSHIP_LOSS", 1.5),
        weight_fragmentation=_float_env("CLEAVE_WEIGHT_FRAGMENTATION", 0.6),
        min_chunk_tokens=_int_env("CLEAVE_MIN_CHUNK_TOKENS", 80, minimum=10),
        target_chunk_tokens=_int_env("CLEAVE_TARGET_CHUNK_TOKENS", 500, minimum=50),
        max_chunk_tokens=_int_env("CLEAVE_MAX_CHUNK_TOKENS", 700, minimum=100),
        tabular_target_tokens=_int_env("CLEAVE_TABULAR_TARGET_TOKENS", 900, minimum=100),
        tabular_max_tokens=_int_env("CLEAVE_TABULAR_MAX_TOKENS", 1200, minimum=150),
        completeness_threshold=_float_env("CLEAVE_COMPLETENESS_THRESHOLD", 0.7, minimum=0.0),
        evaluation_mode=_str_env("CLEAVE_EVALUATION_MODE", "0").lower() in ("1", "true", "yes"),
        allow_synthetic_fallback=_str_env("CLEAVE_ALLOW_SYNTHETIC_FALLBACK", "1").lower()
        not in ("0", "false", "no"),
        weight_shot_change=_float_env("CLEAVE_WEIGHT_SHOT_CHANGE", 0.35),
        speaker_boundary_threshold=_float_env("CLEAVE_SPEAKER_BOUNDARY_THRESHOLD", 0.55),
    )


def reload() -> None:
    """Drop the cached settings, so the next call re-reads the environment."""
    settings.cache_clear()
