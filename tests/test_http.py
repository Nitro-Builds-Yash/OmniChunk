"""Retry policy tests. No network: every call goes through a MockTransport."""

from __future__ import annotations

import httpx
import pytest

from cleave import http


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Retries are real; waiting for them is not."""
    slept: list[float] = []
    monkeypatch.setattr(http.time, "sleep", slept.append)
    return slept


def _install(monkeypatch, handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(http, "client", lambda: client)
    return client


def test_a_transient_429_is_retried_and_then_succeeds(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(429)
        return httpx.Response(200, json={"ok": True})

    _install(monkeypatch, handler)
    resp = http.request_with_retry("POST", "https://example.test/v1", attempts=3)

    assert resp.status_code == 200
    assert len(calls) == 3


def test_a_400_is_not_retried(monkeypatch):
    """A bad request is the server's final answer; retrying only wastes time."""
    calls = []
    _install(monkeypatch, lambda r: (calls.append(r), httpx.Response(400))[1])

    resp = http.request_with_retry("POST", "https://example.test/v1", attempts=3)

    assert resp.status_code == 400
    assert len(calls) == 1


def test_the_last_retryable_response_is_returned_not_raised(monkeypatch):
    calls = []
    _install(monkeypatch, lambda r: (calls.append(r), httpx.Response(503))[1])

    resp = http.request_with_retry("GET", "https://example.test/v1", attempts=2)

    assert resp.status_code == 503  # caller keeps its own raise_for_status
    assert len(calls) == 2


def test_transport_errors_are_retried_then_raised(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("refused", request=request)

    _install(monkeypatch, handler)
    with pytest.raises(httpx.ConnectError):
        http.request_with_retry("GET", "https://example.test/v1", attempts=3)
    assert len(calls) == 3


def test_retry_after_is_honoured_and_capped(monkeypatch, _no_sleep):
    """A server that says when to come back is trusted — but only so far."""
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "86400"})
        return httpx.Response(200)

    _install(monkeypatch, handler)
    http.request_with_retry("GET", "https://example.test/v1", attempts=2)

    assert _no_sleep, "should have waited once"
    assert all(w <= http.MAX_RETRY_DELAY for w in _no_sleep)


def test_jitter_never_exceeds_the_computed_delay(monkeypatch, _no_sleep):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500) if len(calls) < 3 else httpx.Response(200)

    _install(monkeypatch, handler)
    http.request_with_retry(
        "GET", "https://example.test/v1", attempts=3, base_delay=1.0, max_delay=4.0
    )

    assert all(0 <= w <= 4.0 for w in _no_sleep)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("2", 2.0),
        ("0", 0.0),
        ("not-a-date", None),
        (None, None),
    ],
)
def test_retry_after_parsing(header, expected):
    headers = {"retry-after": header} if header is not None else {}
    resp = httpx.Response(429, headers=headers)
    assert http._retry_after_seconds(resp) == expected


def test_the_client_is_pooled_and_resettable():
    first = http.client()
    assert http.client() is first
    http.reset_client()
    assert http.client() is not first
    http.reset_client()
