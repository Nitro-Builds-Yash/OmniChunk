"""Text hot paths, run in Rust when ``cleave_core`` is installed.

``cleave_core`` (``rust/cleave_core``) is an optional PyO3 extension. Every
function here has a pure-Python implementation that produces **identical**
output — the same tokens, the same BM25 scores in the same summation order, the
same MinHash signatures — so installing the extension changes speed, never
results. ``tests/test_rust_core.py`` checks that parity on randomized input.

Build it with::

    uv run maturin develop -r -m rust/cleave_core/Cargo.toml

Set ``CLEAVE_NO_RUST=1`` to force the Python paths (for debugging or parity
checks).
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from typing import Any, Protocol

log = logging.getLogger(__name__)

_core: Any = None
if os.environ.get("CLEAVE_NO_RUST", "").lower() not in ("1", "true", "yes"):
    try:
        import cleave_core as _core  # type: ignore[no-redef]
    except ImportError:
        _core = None

#: Which implementation is live: "rust" or "python".
BACKEND = "rust" if _core is not None else "python"


def backend_info() -> dict[str, str]:
    return {
        "backend": BACKEND,
        "version": getattr(_core, "__version__", "") if _core is not None else "",
    }


# ───────── tokenization ─────────

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_SEP = re.compile(r"[-_.]")


def py_tokenize(text: str) -> list[str]:
    """Lower-cased word tokens; keeps ``SO-10042`` and ``3.2.1`` whole, plus their parts."""
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        out.append(tok)
        if any(c in tok for c in "-_."):
            out.extend(p for p in _SEP.split(tok) if p)
    return out


def tokenize(text: str) -> list[str]:
    return _core.tokenize(text) if _core is not None else py_tokenize(text)


def tokenize_many(texts: list[str]) -> list[list[str]]:
    """Tokenize a batch (parallel across cores in Rust, GIL released)."""
    if _core is not None:
        return _core.tokenize_many(texts)
    return [py_tokenize(t) for t in texts]


# ───────── approximate token pieces ─────────

_APPROX = re.compile(r"\s*\S+|\s+$")


def py_approx_encode(text: str) -> list[str]:
    return _APPROX.findall(text)


def approx_encode(text: str) -> list[str]:
    return _core.approx_encode(text) if _core is not None else py_approx_encode(text)


def approx_count(text: str) -> int:
    return _core.approx_count(text) if _core is not None else len(py_approx_encode(text))


def fixed_windows(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """Fixed-size/overlap windows over approximate token pieces."""
    if _core is not None:
        return _core.fixed_windows(text, size, overlap)
    pieces = py_approx_encode(text)
    step = max(1, size - overlap)
    return ["".join(pieces[i : i + size]) for i in range(0, len(pieces), step)]


# ───────── BM25 ─────────


class Scorer(Protocol):
    def scores(self, query: list[str]) -> list[float]: ...


class PyBM25:
    """Okapi BM25 (k1=1.5, b=0.75) over pre-tokenised documents."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = [0.0] * len(self.tf)
        avg = self.avg or 1.0
        # Per query token, then per document — the same summation order as the
        # Rust inverted index, so the two agree to the last bit.
        for t in query:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, (tf, dl) in enumerate(zip(self.tf, self.len)):
                f = tf.get(t)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * dl / avg)
                    out[i] += idf * f * (self.k1 + 1) / (f + norm)
        return out


def bm25_from_texts(texts: list[str]) -> Scorer:
    """A BM25 scorer over raw texts (tokenized with :func:`tokenize`)."""
    if _core is not None:
        return _core.BM25(texts)
    return PyBM25([py_tokenize(t) for t in texts])


def bm25_from_tokens(docs: list[list[str]]) -> Scorer:
    if _core is not None:
        return _core.BM25.from_tokens(docs)
    return PyBM25(docs)


# ───────── MinHash near-duplicates ─────────

NUM_PERM = 128
BANDS = 16
_M64 = (1 << 64) - 1


def _fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & _M64
    return h


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


_SEEDS = [_splitmix64(i + 1) for i in range(NUM_PERM)]


def py_minhash_signature(text: str) -> list[int] | None:
    toks = py_tokenize(text)
    if not toks:
        return None
    w = min(5, len(toks))
    shingles = {_fnv1a64(" ".join(toks[i : i + w]).encode()) for i in range(len(toks) - w + 1)}
    sig = [_M64] * NUM_PERM
    for h in shingles:
        for i, seed in enumerate(_SEEDS):
            v = _splitmix64(h ^ seed)
            sig[i] = min(sig[i], v)
    return sig


def minhash_signature(text: str) -> list[int] | None:
    if _core is not None:
        return _core.minhash_signature(text)
    return py_minhash_signature(text)


def py_near_duplicate_groups(texts: list[str], threshold: float = 0.8) -> list[list[int]]:
    sigs = [py_minhash_signature(t) for t in texts]
    rows = NUM_PERM // BANDS
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, sig in enumerate(sigs):
        if sig is None:
            continue
        for band in range(BANDS):
            h = band
            for v in sig[band * rows : (band + 1) * rows]:
                h = _splitmix64(h ^ v)
            buckets.setdefault((band, h), []).append(i)
    parent = list(range(len(texts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        for ai, a in enumerate(members):
            for b in members[ai + 1 :]:
                if (a, b) in checked:
                    continue
                checked.add((a, b))
                sa, sb = sigs[a], sigs[b]
                assert sa is not None and sb is not None  # noqa: S101 — bucketed ⇒ signed
                same = sum(x == y for x, y in zip(sa, sb))
                if same / NUM_PERM >= threshold:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for i in range(len(texts)):
        groups.setdefault(find(i), []).append(i)
    return sorted((g for g in groups.values() if len(g) > 1), key=lambda g: g[0])


def near_duplicate_groups(texts: list[str], threshold: float = 0.8) -> list[list[int]]:
    """Groups of indices whose texts are near-duplicates (estimated Jaccard of
    5-word shingles ≥ ``threshold``), each sorted, ordered by first member."""
    if _core is not None:
        return [list(g) for g in _core.near_duplicate_groups(texts, threshold)]
    return py_near_duplicate_groups(texts, threshold)
