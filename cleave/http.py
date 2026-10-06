"""One HTTP client and one retry policy for every outbound call.

Three callers share this: Ollama, Gemini and the STT worker. Each of them used
to build a fresh ``httpx`` request per call — a new connection pool and a fresh
TLS handshake every time — and none of them retried anything. Enrichment fires
up to three batches concurrently, so a Gemini 429 was not a hypothetical: it was
caught by a blanket ``except``, turned into an empty completion, and the job
quietly produced unenriched chunks with nothing in the UI to say why.

The retry policy is deliberately narrow. Only transport errors and the statuses
that actually mean "try again" (408, 429, 5xx) are retried; a 400 or a 401 is a
bad request or a bad key, and retrying it just makes the user wait longer for
the same answer. ``Retry-After`` is honoured when the server sends one, because
a server that tells you when to come back knows better than our backoff curve —
but it is capped, so a hostile or absurd value cannot pin a job for a day.

Backoff uses full jitter (``uniform(0, delay)``) rather than a fixed curve. With
three concurrent enrichment batches, fixed backoff would retry all three at the
same instant and reproduce exactly the burst that earned the 429.
"""

from __future__ import annotations

import logging
import random
import time
from email.utils import parsedate_to_datetime
from functools import lru_cache

import httpx

log = logging.getLogger(__name__)

#: Statuses worth a second attempt. Everything else is the server's final answer.
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})

#: Never wait longer than this between attempts, whatever ``Retry-After`` says.
MAX_RETRY_DELAY = 20.0


@lru_cache(maxsize=1)
def client() -> httpx.Client:
    """The process-wide connection pool.

    Reused across jobs and threads: ``httpx.Client`` is thread-safe, and
    enrichment calls it from a ``ThreadPoolExecutor``.
    """
    return httpx.Client(
        timeout=httpx.Timeout(30.0, connect=5.0),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        headers={"user-agent": "cleave/0.1"},
    )


def reset_client() -> None:
    """Close and forget the pooled client. For tests, and for a config reload."""
    if not hasattr(client, "cache_info"):
        return  # a test has swapped the factory out; nothing pooled to close
    if client.cache_info().currsize:
        try:
            client().close()
        except Exception:  # pragma: no cover - closing must never be fatal
            log.debug("closing the pooled http client failed", exc_info=True)
    client.cache_clear()


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """``Retry-After`` in seconds, accepting both the delta and HTTP-date forms."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    raw = raw.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    import datetime as _dt  # noqa: PLC0415 — only needed on the HTTP-date path

    now = _dt.datetime.now(_dt.UTC) if when.tzinfo else _dt.datetime.now()
    return max(0.0, (when - now).total_seconds())


def request_with_retry(
    method: str,
    url: str,
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = MAX_RETRY_DELAY,
    **kwargs,
) -> httpx.Response:
    """One HTTP call with bounded retries. Raises on final transport failure.

    Returns the response as-is — including a retryable status on the last
    attempt — so the caller keeps its own ``raise_for_status`` semantics and
    sees the real status code.
    """
    last_error: Exception | None = None
    response: httpx.Response | None = None

    for attempt in range(attempts):
        try:
            response = client().request(method, url, **kwargs)
        except httpx.TransportError as exc:
            last_error = exc
            response = None
            if attempt == attempts - 1:
                break
            delay = min(max_delay, base_delay * (2**attempt))
        else:
            if response.status_code not in RETRY_STATUS or attempt == attempts - 1:
                return response
            hinted = _retry_after_seconds(response)
            delay = min(max_delay, hinted if hinted is not None else base_delay * (2**attempt))

        wait = random.uniform(0, delay)  # noqa: S311 — jitter, not cryptography
        status = response.status_code if response is not None else last_error
        log.warning(
            "%s %s → %s; retrying in %.1fs (attempt %d/%d)",
            method,
            httpx.URL(url).host,
            status,
            wait,
            attempt + 1,
            attempts,
        )
        time.sleep(wait)

    if response is not None:
        return response
    if last_error is None:  # pragma: no cover - unreachable: attempts >= 1
        raise RuntimeError(f"{method} {url} made no attempt")
    raise last_error
