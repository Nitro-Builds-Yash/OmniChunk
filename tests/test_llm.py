"""Provider selection and the contract every provider must keep.

The module docstring in ``cleave/llm.py`` states two rules: ``complete_json``
never raises, and every call reports usage. Neither was covered by a test.
"""

from __future__ import annotations

import httpx
import pytest

from cleave import config, http, llm


@pytest.fixture(autouse=True)
def _fresh_probe():
    llm.reset_tags_cache()
    yield
    llm.reset_tags_cache()


def _install(monkeypatch, handler):
    """Point both the pool and llm.py's import-time binding at a mock."""
    mock = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(http, "client", lambda: mock)
    monkeypatch.setattr(llm, "client", lambda: mock)
    monkeypatch.setattr(llm, "request_with_retry", http.request_with_retry)
    return mock


# ───────── selection ─────────


def test_cleave_llm_none_selects_the_none_provider():
    assert llm.get_provider().name == "none"


def test_gemini_without_a_key_falls_back_to_none(monkeypatch):
    monkeypatch.setenv("CLEAVE_LLM", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config.reload()
    assert llm.get_provider().name == "none"


def test_gemini_with_a_key_is_selected(monkeypatch):
    monkeypatch.setenv("CLEAVE_LLM", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    config.reload()
    provider = llm.get_provider()
    assert provider.name == "gemini"
    assert provider.model == "gemini-2.5-flash"


def test_no_key_is_scavenged_from_other_projects(monkeypatch):
    """The removed behaviour: llm.py used to read a key out of two hard-coded
    sibling repositories under ~/PycharmProjects."""
    assert not hasattr(llm, "_find_gemini_key")
    monkeypatch.setenv("CLEAVE_LLM", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config.reload()
    assert llm.GeminiProvider().key == ""


# ───────── the never-raises contract ─────────


@pytest.mark.parametrize(
    "handler",
    [
        pytest.param(lambda r: httpx.Response(500), id="server-error"),
        pytest.param(lambda r: httpx.Response(429), id="rate-limited"),
        pytest.param(lambda r: httpx.Response(200, text="not json"), id="malformed-body"),
        pytest.param(lambda r: httpx.Response(200, json={}), id="no-candidates"),
    ],
)
def test_gemini_complete_json_never_raises(monkeypatch, handler):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(http.time, "sleep", lambda _s: None)
    config.reload()
    _install(monkeypatch, handler)

    text, usage = llm.GeminiProvider().complete_json("prompt")

    assert text == ""
    assert "model" in usage


def test_ollama_complete_json_never_raises(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda _s: None)
    _install(monkeypatch, lambda r: httpx.Response(500))

    text, usage = llm.OllamaProvider(model="qwen3:4b").complete_json("prompt")

    assert text == ""
    assert usage.get("model", "").startswith("ollama/")


# ───────── credentials travel in a header ─────────


def test_gemini_sends_the_key_as_a_header_not_a_query_parameter(monkeypatch):
    """The regression: the key used to ride in ``?key=`` and land in logs."""
    seen: dict = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["header"] = request.headers.get("x-goog-api-key")
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": '{"results":[]}'}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2},
            },
        )

    monkeypatch.setenv("GEMINI_API_KEY", "secret-key-value")
    config.reload()
    _install(monkeypatch, handler)

    text, usage = llm.GeminiProvider().complete_json("prompt")

    assert text == '{"results":[]}'
    assert seen["header"] == "secret-key-value"
    assert "secret-key-value" not in seen["url"]
    assert "key=" not in seen["url"]
    assert usage["in_tokens"] == 10 and usage["out_tokens"] == 2


# ───────── the tag probe must not poison itself ─────────


def test_a_failed_tag_probe_is_not_cached_for_the_process_lifetime(monkeypatch):
    """The regression: one transient failure disabled Ollama until restart."""
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"models": [{"name": "qwen3:4b"}]})

    _install(monkeypatch, handler)
    # A failure is cached only briefly; expire it immediately so the second
    # call re-probes. The old lru_cache would have kept () for good.
    monkeypatch.setattr(llm, "_TAGS_TTL_FAIL", -1.0)

    assert llm.ollama_tags() == ()  # first probe fails
    assert llm.ollama_tags() == ("qwen3:4b",)  # re-probes rather than staying dead
    assert len(calls) == 2


def test_a_successful_probe_is_cached(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"models": [{"name": "qwen3:4b"}]})

    _install(monkeypatch, handler)
    assert llm.ollama_tags() == ("qwen3:4b",)
    assert llm.ollama_tags() == ("qwen3:4b",)
    assert len(calls) == 1


# ───────── UI-facing description ─────────


def test_describe_providers_reports_both_without_private_attributes(monkeypatch):
    _install(monkeypatch, lambda r: httpx.Response(200, json={"models": []}))
    rows = llm.describe_providers()
    assert {r["name"] for r in rows} == {"ollama", "gemini"}
    for row in rows:
        assert {"label", "model", "available", "cost", "active"} <= set(row)


def test_none_provider_reports_no_usage():
    text, usage = llm.NoneProvider().complete_json("prompt")
    assert text == "" and usage == {}
