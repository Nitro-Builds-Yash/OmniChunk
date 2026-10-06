"""Token and cost accounting, per model.

Every model call in Cleave is recorded here: which model, how many tokens went
in and out, how many were served from cache, and what it cost. The ledger is
kept per job and accumulated across the whole install, because the interesting
question is rarely "what did this file cost" — it is "where is the spend going,
and is it going anywhere it does not have to."

Local models are recorded exactly like paid ones, at zero cost. That way the
comparison between "run it on the API" and "run it here" is visible in the same
table rather than being an argument.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

LEDGER_PATH = Path(__file__).resolve().parent.parent / "data" / "usage.json"

#: Guards the read-modify-write of the cumulative ledger file. ``Ledger._lock``
#: cannot do this job: it is per instance, and every call below builds a fresh
#: ``Ledger``, so two overlapping jobs shared no lock at all.
_LEDGER_FILE_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class Price:
    """USD per million tokens."""

    inp: float
    out: float
    cached_inp: float = 0.0  # cache hits bill at a discount; 0.0 = same as inp

    def cost(self, in_tokens: int, out_tokens: int, cached_tokens: int = 0) -> float:
        fresh = max(0, in_tokens - cached_tokens)
        cached_rate = self.cached_inp or self.inp
        return (fresh * self.inp + cached_tokens * cached_rate + out_tokens * self.out) / 1e6


#: Published rates, USD per million tokens (input, output, cached input).
#: Anything not listed is billed at the DEFAULT rate and flagged in the ledger,
#: so an unpriced model shows up as an estimate rather than silently as $0.
PRICING: dict[str, Price] = {
    "gemini-2.5-flash": Price(0.30, 2.50, 0.075),
    "gemini-2.5-flash-lite": Price(0.10, 0.40, 0.025),
    "gemini-3.7-flash": Price(0.375, 1.875, 0.09),
    "gemini-3.6-flash": Price(0.75, 3.75, 0.1875),
    "gemini-2.0-flash": Price(0.10, 0.40, 0.025),
}
DEFAULT_PRICE = Price(0.30, 2.50, 0.075)


def is_local(model: str) -> bool:
    return model.startswith(("local/", "ollama/", "mlx/"))


def price_for(model: str) -> tuple[Price, bool]:
    """→ (price, is_known). Local models are free and always 'known'."""
    if is_local(model):
        return Price(0.0, 0.0), True
    for name, p in PRICING.items():
        if model.startswith(name):
            return p, True
    return DEFAULT_PRICE, False


@dataclass
class ModelUsage:
    model: str
    calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    failures: int = 0
    estimated: bool = False  # priced with the fallback rate, not a published one

    @property
    def local(self) -> bool:
        return is_local(self.model)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cost_usd"] = round(self.cost_usd, 6)
        d["local"] = self.local
        d["cache_hit_pct"] = (
            round(100 * self.cached_tokens / self.in_tokens, 1) if self.in_tokens else 0.0
        )
        return d


@dataclass
class Ledger:
    """Thread-safe: enrichment runs calls concurrently."""

    models: dict[str, ModelUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, model: str, in_tokens: int, out_tokens: int, cached_tokens: int = 0) -> float:
        price, known = price_for(model)
        cost = price.cost(in_tokens, out_tokens, cached_tokens)
        with self._lock:
            u = self.models.setdefault(model, ModelUsage(model=model, estimated=not known))
            u.calls += 1
            u.in_tokens += in_tokens
            u.out_tokens += out_tokens
            u.cached_tokens += cached_tokens
            u.cost_usd += cost
        return cost

    def record_failure(self, model: str) -> None:
        with self._lock:
            self.models.setdefault(model, ModelUsage(model=model)).failures += 1

    # ── views ──

    @property
    def total_cost(self) -> float:
        return sum(u.cost_usd for u in self.models.values())

    @property
    def total_calls(self) -> int:
        return sum(u.calls for u in self.models.values())

    @property
    def total_in(self) -> int:
        return sum(u.in_tokens for u in self.models.values())

    @property
    def total_out(self) -> int:
        return sum(u.out_tokens for u in self.models.values())

    def to_dict(self) -> dict:
        rows = sorted(self.models.values(), key=lambda u: -u.cost_usd)
        return {
            "by_model": [u.to_dict() for u in rows],
            "totals": {
                "calls": self.total_calls,
                "in_tokens": self.total_in,
                "out_tokens": self.total_out,
                "cached_tokens": sum(u.cached_tokens for u in self.models.values()),
                "cost_usd": round(self.total_cost, 6),
                "local_calls": sum(u.calls for u in self.models.values() if u.local),
                "paid_calls": sum(u.calls for u in self.models.values() if not u.local),
            },
        }

    def merge(self, other: dict) -> None:
        """Fold a persisted snapshot back in, for the cumulative ledger."""
        for row in other.get("by_model", []):
            with self._lock:
                u = self.models.setdefault(
                    row["model"],
                    ModelUsage(model=row["model"], estimated=row.get("estimated", False)),
                )
                u.calls += row.get("calls", 0)
                u.in_tokens += row.get("in_tokens", 0)
                u.out_tokens += row.get("out_tokens", 0)
                u.cached_tokens += row.get("cached_tokens", 0)
                u.cost_usd += row.get("cost_usd", 0.0)
                u.failures += row.get("failures", 0)


def append_to_cumulative(job_ledger: Ledger, job_id: str) -> dict:
    """Fold one job's usage into the install-wide ledger and return it.

    Read-modify-write under a module lock. ``BackgroundTasks`` are *not*
    serialised across requests — two uploads seconds apart genuinely overlap —
    so the unguarded version here silently dropped one job's spend. The lock is
    process-wide rather than a file lock: two server processes sharing one data
    directory would still race, which is out of scope for a single-process app.

    The write is atomic (temp file, then replace) so a crash mid-write cannot
    leave a truncated ``usage.json`` that the next read silently discards.
    """
    with _LEDGER_FILE_LOCK:
        cumulative = Ledger()
        if LEDGER_PATH.exists():
            try:
                cumulative.merge(json.loads(LEDGER_PATH.read_text()))
            except (OSError, ValueError) as exc:
                log.warning("could not read usage ledger (%s); starting a fresh one", exc)
        cumulative.merge(job_ledger.to_dict())

        out = cumulative.to_dict()
        out["last_job"] = job_id
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = LEDGER_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=1))
        tmp.replace(LEDGER_PATH)
        return out


def read_cumulative() -> dict | None:
    if not LEDGER_PATH.exists():
        return None
    try:
        return json.loads(LEDGER_PATH.read_text())
    except (OSError, ValueError):
        return None
