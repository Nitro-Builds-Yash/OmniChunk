"""LLM providers: a local one and a paid one behind the same interface.

Two rules hold for every provider here:

* ``complete_json`` returns ``("", {})`` on ANY failure, so callers fall back to
  the deterministic path instead of handling exceptions. A rate limit degrades
  the output; it never fails the job.
* every call reports its token usage, so `usage.py` can price it. A provider
  that cannot report usage is not cheaper — it is unaccountable.

Selection order is deliberate: a local model, if one is actually serving, costs
nothing and keeps the document on the machine, so it wins by default. The API
is the fallback, not the assumption.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

from .config import settings
from .http import client, request_with_retry

log = logging.getLogger(__name__)

#: Small instruct models that follow a JSON schema reliably, best first. The
#: first one already pulled locally is used.
OLLAMA_PREFERRED = (
    "granite4.2:8b",
    "qwen3.5:4b",
    "llama3.2:3b",
    "qwen2.5:7b",
    "gemma3:4b",
    "phi4-mini",
    "mistral:7b",
)


#: How long a tag probe is trusted. A success is stable — a model list rarely
#: changes mid-session. A failure is cached for only a few seconds, because the
#: previous ``lru_cache(maxsize=1)`` remembered one transient connection refusal
#: for the entire process lifetime: start Ollama a moment after the server and
#: it would never be found, with no log line to say so.
_TAGS_TTL_OK = 60.0
_TAGS_TTL_FAIL = 5.0

_tags_state: tuple[float, tuple[str, ...]] | None = None
_tags_lock = threading.Lock()


def ollama_tags() -> tuple[str, ...]:
    """Models currently served by Ollama, cached briefly.

    Locked because ``describe_providers`` builds two providers and the homepage
    can be requested concurrently.
    """
    global _tags_state
    with _tags_lock:
        now = time.monotonic()
        if _tags_state is not None and now < _tags_state[0]:
            return _tags_state[1]
        url = settings().ollama_url
        try:
            # A liveness probe, so no retries: a retry would treble the latency
            # of every homepage render when Ollama simply is not running.
            r = client().get(f"{url}/api/tags", timeout=2.0)
            r.raise_for_status()
            tags = tuple(m["name"] for m in r.json().get("models", []))
            _tags_state = (now + _TAGS_TTL_OK, tags)
            return tags
        except Exception as exc:
            log.info(
                "ollama not reachable at %s (%s) — re-probing in %.0fs", url, exc, _TAGS_TTL_FAIL
            )
            _tags_state = (now + _TAGS_TTL_FAIL, ())
            return ()


def reset_tags_cache() -> None:
    """Forget the probe result. For tests, and after a config reload."""
    global _tags_state
    with _tags_lock:
        _tags_state = None


class LLMProvider(Protocol):
    name: str

    @property
    def model(self) -> str:
        """The billing identity of this provider, e.g. ``ollama/qwen3:4b``.

        A read-only property rather than an attribute: the implementations
        derive it (Ollama prefixes its tag) and nothing should assign to it.
        """
        ...  # pragma: no cover

    def is_configured(self) -> bool: ...

    def complete_json(
        self, prompt: str, *, system: str | None = None, schema: dict | None = None
    ) -> tuple[str, dict]:
        """→ (text, usage). ``text == ""`` means failure.

        usage: {model, in_tokens, out_tokens, cached_tokens}.
        """


class OllamaProvider:
    """Local models over HTTP.

    A separate process rather than an in-process library, for the same reason
    audio is: this venv is pinned by Docling and cannot also satisfy an
    inference stack. HTTP turns that constraint into an interface.
    """

    name = "ollama"

    def __init__(self, model: str | None = None) -> None:
        self._model = model or settings().ollama_model

    @property
    def model(self) -> str:
        return f"ollama/{self._model}" if self._model else "ollama/none"

    @property
    def local_model(self) -> str:
        """The bare model tag, without the ``ollama/`` prefix — for the UI."""
        return self._model

    def is_configured(self) -> bool:
        tags = self._tags()
        if not tags:
            return False
        if self._model:
            return any(t == self._model or t.startswith(self._model.split(":")[0]) for t in tags)
        for want in OLLAMA_PREFERRED:  # pick the best model already pulled
            for have in tags:
                if have == want or have.startswith(want.split(":")[0] + ":"):
                    self._model = have
                    return True
        self._model = tags[0]
        return True

    def _tags(self) -> tuple[str, ...]:
        return ollama_tags()

    def complete_json(
        self, prompt: str, *, system: str | None = None, schema: dict | None = None
    ) -> tuple[str, dict]:
        if not self._model:
            return "", {}
        body: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            # Reasoning models (qwen3, deepseek-r1) otherwise emit the
            # schema-constrained JSON inside their thinking block and leave
            # `response` empty. We want the answer, not the deliberation.
            "think": False,
            "options": {"temperature": 0.2},
        }
        if system:
            body["system"] = system
        if schema:
            # Ollama constrains decoding to the schema via GBNF, so the reply is
            # valid JSON by construction rather than by asking nicely.
            body["format"] = schema
        try:
            cfg = settings()
            r = request_with_retry(
                "POST",
                f"{cfg.ollama_url}/api/generate",
                json=body,
                timeout=cfg.llm_timeout_s,
                # A local model that fails twice is down, not busy; a third
                # attempt just costs another full timeout.
                attempts=2,
            )
            r.raise_for_status()
            data = r.json()
            text = (data.get("response") or "").strip()
            if not text:
                # Some builds ignore think=False; the JSON is still in there.
                text = (data.get("thinking") or "").strip()
            return text, {
                "model": self.model,
                "in_tokens": int(data.get("prompt_eval_count", 0)),
                "out_tokens": int(data.get("eval_count", 0)),
                "cached_tokens": 0,
            }
        except Exception as exc:
            log.warning("ollama call failed: %s", exc)
            return "", {"model": self.model}


class GeminiProvider:
    name = "gemini"

    def __init__(self) -> None:
        cfg = settings()
        self.key = cfg.gemini_api_key
        self._model = cfg.gemini_model

    @property
    def model(self) -> str:
        return self._model

    def is_configured(self) -> bool:
        return bool(self.key)

    def complete_json(
        self, prompt: str, *, system: str | None = None, schema: dict | None = None
    ) -> tuple[str, dict]:
        if not self.is_configured():
            return "", {}
        gen: dict = {"temperature": 0.2}
        if schema:
            gen["responseMimeType"] = "application/json"
            gen["responseSchema"] = schema
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen,
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        try:
            # The key goes in a header, not a query parameter: query strings end
            # up in proxy logs, browser history and exception reprs.
            r = request_with_retry(
                "POST",
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self._model}:generateContent",
                headers={"x-goog-api-key": self.key},
                json=body,
                timeout=settings().llm_timeout_s,
                attempts=3,
            )
            r.raise_for_status()
            data = r.json()
            text = "".join(
                p.get("text", "") for p in data["candidates"][0]["content"]["parts"]
            ).strip()
            um = data.get("usageMetadata", {})
            return text, {
                "model": self._model,
                "in_tokens": int(um.get("promptTokenCount", 0)),
                "out_tokens": int(um.get("candidatesTokenCount", 0)),
                # Gemini caches repeated prefixes implicitly and bills them lower.
                "cached_tokens": int(um.get("cachedContentTokenCount", 0)),
            }
        except Exception as exc:
            log.warning("gemini call failed: %s", exc)
            return "", {"model": self._model}


class NoneProvider:
    name = "none"
    model = "none"

    def is_configured(self) -> bool:
        return True

    def complete_json(
        self, prompt: str, *, system: str | None = None, schema: dict | None = None
    ) -> tuple[str, dict]:
        return "", {}


def get_provider() -> LLMProvider:
    """Local first, then the API, then nothing.

    ``CLEAVE_LLM`` forces the choice: ``none`` | ``ollama`` | ``gemini``.
    Without it, a running local model is preferred because it is free and keeps
    the document on this machine; the paid API is the fallback.
    """
    choice = settings().llm
    if choice == "none":
        return NoneProvider()
    if choice == "ollama":
        o = OllamaProvider()
        return o if o.is_configured() else NoneProvider()
    if choice == "gemini":
        g = GeminiProvider()
        return g if g.is_configured() else NoneProvider()

    o = OllamaProvider()
    if o.is_configured():
        log.info("using local model %s (no API cost)", o.model)
        return o
    g = GeminiProvider()
    if g.is_configured():
        return g
    return NoneProvider()


def describe_providers() -> list[dict]:
    """What is available right now — drives the UI's provider strip."""
    o, g = OllamaProvider(), GeminiProvider()
    o_ok, g_ok = o.is_configured(), g.is_configured()
    active = get_provider()
    return [
        {
            "name": "ollama",
            "label": "Local (Ollama)",
            "model": o.local_model or "—",
            "available": o_ok,
            "cost": "free",
            "active": active.name == "ollama",
        },
        {
            "name": "gemini",
            "label": "Gemini API",
            "model": g.model if g_ok else "—",
            "available": g_ok,
            "cost": "paid",
            "active": active.name == "gemini",
        },
    ]
