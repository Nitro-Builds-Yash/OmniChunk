"""Access control and abuse limits for the HTTP surface.

Two problems, one module. The app used to be reachable through an ngrok tunnel
with no authentication at all, so anyone holding the URL could upload files,
read every job's artifacts and spend the install's LLM quota. And nothing
bounded how fast one client could submit jobs.

* ``CLEAVE_API_KEY`` — when set, every route except ``/health`` and
  ``/static`` needs it, as an ``X-API-Key`` header, an ``Authorization: Bearer``
  token, or the password of HTTP Basic auth (so a browser gets a native login
  prompt and HTMX requests carry the credentials automatically). Unset keeps
  the zero-config local experience.
* ``CLEAVE_RATE_LIMIT_PER_MIN`` — a per-client sliding window on job creation,
  the one route that costs real CPU and possibly money.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import threading
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..config import settings

#: Reachable without a key: liveness probes and the static assets the login
#: page itself needs.
PUBLIC_PREFIXES = ("/health", "/static/", "/favicon.svg")

#: Routes counted against the submission rate limit.
RATE_LIMITED = (("POST", "/api/jobs"), ("POST", "/api/jobs/"))


def _presented_key(request: Request) -> str | None:
    if key := request.headers.get("x-api-key"):
        return key
    auth = request.headers.get("authorization", "")
    scheme, _, value = auth.partition(" ")
    if scheme.lower() == "bearer" and value:
        return value.strip()
    if scheme.lower() == "basic" and value:
        try:
            decoded = base64.b64decode(value.strip()).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            return None
        return decoded.partition(":")[2]  # any username; the password is the key
    return None


def _client_id(request: Request) -> str:
    return request.client.host if request.client else "unknown"


class _SlidingWindow:
    """At most ``limit`` events per ``window`` seconds, per key."""

    def __init__(self, window: float = 60.0) -> None:
        self.window = window
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int) -> tuple[bool, int]:
        """(allowed, seconds until a slot frees)."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= limit:
                return False, max(1, int(self.window - (now - q[0])) + 1)
            q.append(now)
            return True, 0

    def clear(self) -> None:
        with self._lock:
            self._hits.clear()


submissions = _SlidingWindow()


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        cfg = settings()
        path = request.url.path

        if cfg.api_key and not path.startswith(PUBLIC_PREFIXES):
            presented = _presented_key(request)
            if presented is None or not hmac.compare_digest(
                presented.encode("utf-8"), cfg.api_key.encode("utf-8")
            ):
                return JSONResponse(
                    {"detail": "authentication required"},
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="Cleave"'},
                )

        if cfg.rate_limit_per_min and (request.method, path) in RATE_LIMITED:
            ok, retry = submissions.allow(_client_id(request), cfg.rate_limit_per_min)
            if not ok:
                return JSONResponse(
                    {"detail": f"too many jobs; retry in {retry}s"},
                    status_code=429,
                    headers={"Retry-After": str(retry)},
                )

        return await call_next(request)
