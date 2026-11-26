"""Hybrid retrieval over knowledge units: BM25 + vectors, fused, graph-expanded.

Vector-only search had three weaknesses. It missed exact tokens embeddings
blur together (order ids like ``SO-10042``, section numbers, product codes); it
returned nothing at all when the embedding model could not load; and it
ignored the context graph Cleave spends most of its effort building.

The ranking here is:

1. **BM25** over ``embed_text`` (heading path + content) — always available,
   pure Python, no model.
2. **Dense cosine** with the shared embedder, when it loads.
3. **Reciprocal rank fusion** of the two lists (k=60), which needs no score
   calibration between them.
4. **Optional cross-encoder re-rank** of the fused head (``CLEAVE_RERANKER_MODEL``).
5. **Graph expansion**: each hit carries the units it is linked to by
   caption, schema, parent/child and reference edges, so a matched row group
   arrives with its schema card and a matched figure with its caption.

Per-job indexes are cached keyed by ``units.json`` mtime, so reprocessing a job
invalidates them, and the cache is bounded.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .. import _accel
from .jobs import job_artifact

log = logging.getLogger(__name__)

TOP_K = 5
RRF_K = 60
_CACHE_MAX = 16

#: Edges worth following at query time: the ones that carry meaning a hit
#: needs (its caption, its schema, its section), not mere reading order.
EXPAND_TYPES = {
    "captions",
    "captioned_by",
    "references",
    "has_schema",
    "schema_of",
    "child_of",
    "parent_of",
    "explains",
    "illustrated_by",
    "answered_by",
}


#: Re-exported: the implementations live in ``cleave._accel`` (Rust when built).
tokenize = _accel.tokenize
BM25 = _accel.PyBM25


@dataclass(slots=True)
class _Index:
    units: list[dict]
    bm25: _accel.Scorer
    vectors: Any = None  # numpy matrix, lazily built
    by_id: dict[str, dict] = field(default_factory=dict)


@dataclass(slots=True)
class Hit:
    score: float
    job_id: str
    unit: dict
    signals: dict[str, float] = field(default_factory=dict)
    related: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "job_id": self.job_id,
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "unit": self.unit,
            "related": [
                {"id": r["id"], "relation": r["_relation"], "content": r.get("content", "")[:500]}
                for r in self.related
            ],
        }


_CACHE: OrderedDict[str, tuple[float, _Index]] = OrderedDict()
_LOCK = threading.Lock()


def clear_cache() -> None:
    """Forget every cached index. For tests."""
    with _LOCK:
        _CACHE.clear()


def _index(job_id: str) -> _Index | None:
    path = job_artifact(job_id, "units.json")
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    with _LOCK:
        hit = _CACHE.get(job_id)
        if hit and hit[0] == mtime:
            _CACHE.move_to_end(job_id)
            return hit[1]
    units = json.loads(path.read_text())
    idx = _Index(
        units=units,
        bm25=_accel.bm25_from_texts([u.get("embed_text") or u.get("content", "") for u in units]),
        by_id={u["id"]: u for u in units},
    )
    with _LOCK:
        _CACHE[job_id] = (mtime, idx)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return idx


def _dense_scores(idx: _Index, qv: Any) -> list[float] | None:
    if qv is None:
        return None
    if idx.vectors is None:
        from ..semantic import embed  # noqa: PLC0415 — torch import stays lazy

        idx.vectors = embed([u.get("embed_text") or u.get("content", "") for u in idx.units])
        if idx.vectors is None:
            return None
    return [float(x) for x in idx.vectors @ qv]


def _ranks(scores: list[float], positive_only: bool) -> dict[int, int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if positive_only:
        order = [i for i in order if scores[i] > 0]
    return {i: r for r, i in enumerate(order, start=1)}


def search_jobs(job_ids: list[str], query: str, k: int = TOP_K) -> list[Hit]:
    """Top-``k`` hybrid hits across ``job_ids``. Blocking (may run inference)."""
    q_tokens = tokenize(query)
    qv = None
    try:
        from ..semantic import embed  # noqa: PLC0415

        q_emb = embed([query])
        qv = q_emb[0] if q_emb is not None else None
    except Exception as exc:  # torch missing, model download blocked…
        log.debug("dense retrieval unavailable: %s", exc)

    # Rank over the union of every requested job's units. Fusing per job and
    # then merging would make scores incomparable: each job's best match gets
    # rank 1, so a weak hit in one job tied a strong hit in another.
    pool: list[tuple[str, dict]] = []
    lex: list[float] = []
    dense: list[float] | None = [] if qv is not None else None
    indexes: list[tuple[str, _Index]] = []
    for j in job_ids:
        ix = _index(j)
        if ix is not None and ix.units:
            indexes.append((j, ix))
    if len(indexes) == 1:
        lex = indexes[0][1].bm25.scores(q_tokens)  # cached per-job index
    elif indexes:
        lex = _accel.bm25_from_texts(
            [u.get("embed_text") or u.get("content", "") for _, ix in indexes for u in ix.units]
        ).scores(q_tokens)
    for job_id, idx in indexes:
        pool.extend((job_id, u) for u in idx.units)
        if dense is not None:
            d = _dense_scores(idx, qv)
            dense = None if d is None else dense + d  # cosine is comparable across jobs

    lex_r = _ranks(lex, positive_only=True)
    dense_r = _ranks(dense, positive_only=False) if dense is not None else {}
    candidates: list[Hit] = []
    for i, (job_id, u) in enumerate(pool):
        rrf = 0.0
        signals: dict[str, float] = {}
        if i in lex_r:
            rrf += 1 / (RRF_K + lex_r[i])
            signals["bm25"] = lex[i]
        if dense is not None and i in dense_r:
            rrf += 1 / (RRF_K + dense_r[i])
            signals["cosine"] = dense[i]
        if rrf:
            candidates.append(Hit(rrf, job_id, u, signals))

    candidates.sort(key=lambda h: h.score, reverse=True)
    candidates = _collapse_duplicates(candidates)
    head = candidates[: max(k * 4, 20)]

    from ..semantic import rerank  # noqa: PLC0415

    rr = rerank(query, [h.unit.get("embed_text") or h.unit.get("content", "") for h in head])
    if rr is not None:
        for h, s in zip(head, rr):
            h.signals["rerank"] = s
            h.score = s
        head.sort(key=lambda h: h.score, reverse=True)

    hits = head[:k]
    for h in hits:
        h.related = _expand(h)
    return hits


def _collapse_duplicates(ranked: list[Hit]) -> list[Hit]:
    """Keep the best-ranked member of each near-duplicate group (see
    ``pipeline.mark_near_duplicates``), so one paragraph pasted into five
    documents does not fill all five result slots."""
    seen: set[tuple[str, str]] = set()
    out = []
    for h in ranked:
        key = (h.job_id, h.unit.get("metadata", {}).get("duplicate_of") or h.unit["id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _expand(hit: Hit, limit: int = 3) -> list[dict]:
    idx = _index(hit.job_id)
    if idx is None:
        return []
    out, seen = [], {hit.unit["id"]}
    for rel in hit.unit.get("relationships", []):
        target = rel.get("target_id")
        if rel.get("type") not in EXPAND_TYPES or target in seen or target not in idx.by_id:
            continue
        seen.add(target)
        out.append({**idx.by_id[target], "_relation": rel["type"]})
        if len(out) >= limit:
            break
    return out


def search_units(job_id: str, query: str) -> list[tuple[float, dict]]:
    """Single-job ``(score, unit)`` pairs, the shape the results template renders.

    Scores are normalised to the best hit (1.0) so the UI's "% match" reads
    sensibly whichever signal ranked them.
    """
    hits = search_jobs([job_id], query)
    if not hits:
        return []
    top = hits[0].score or 1.0
    return [(h.score / top, h.unit) for h in hits]


def embedding_available() -> bool:
    """Whether the dense model can be loaded at all (search works without it)."""
    from ..semantic import available  # noqa: PLC0415

    return bool(available())
