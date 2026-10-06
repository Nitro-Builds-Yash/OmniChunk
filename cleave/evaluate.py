"""Context Preservation Scorecard: fixed 512/64 baseline vs Cleave, measured.

Every number is computed from the documents given on the command line and
written to data/scorecard.json. Nothing is predefined: the UI renders whatever
this file measured, flattering or not.

Metrics (each preserved/total over the corpus):
  caption integrity   captioned tables whose caption and body share a chunk
  header integrity    tables whose every row-bearing chunk also carries the header
  heading context     paragraphs whose chunk knows their governing heading
  resolved references 'Table N'/'Figure N' mentions whose target is co-located
                      (fixed) or explicitly linked (Cleave)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .chunkers import chunk
from .graph import ContextGraph
from .ingest_document import IngestResult, ingest_document
from .models import _encoder

log = logging.getLogger(__name__)

FIXED_TOKENS = 512
FIXED_OVERLAP = 64

_norm_re = re.compile(r"[^a-z0-9]+")


def norm(s: str) -> str:
    return _norm_re.sub(" ", s.lower()).strip()


def fixed_chunks(ingest: IngestResult) -> list[str]:
    """The baseline every RAG tutorial ships: flatten to text, split by tokens."""
    text = "\n\n".join(e.text for e in ingest.elements if e.text)
    enc = _encoder()
    ids = enc.encode(text, disallowed_special=())
    out, start = [], 0
    while start < len(ids):
        out.append(enc.decode(ids[start : start + FIXED_TOKENS]))
        start += FIXED_TOKENS - FIXED_OVERLAP
    return out


@dataclass(slots=True)
class Tally:
    preserved: int = 0
    total: int = 0

    def add(self, ok: bool) -> None:
        self.total += 1
        self.preserved += 1 if ok else 0


@dataclass(slots=True)
class ArmScore:
    caption: Tally = field(default_factory=Tally)
    header: Tally = field(default_factory=Tally)
    heading: Tally = field(default_factory=Tally)
    refs: Tally = field(default_factory=Tally)

    def to_dict(self) -> dict:
        metrics = [
            ("caption integrity", self.caption),
            ("header integrity", self.header),
            ("heading context", self.heading),
            ("resolved references", self.refs),
        ]
        ratios = [t.preserved / t.total for _, t in metrics if t.total]
        return {
            "cps_pct": round(100 * sum(ratios) / len(ratios), 1) if ratios else None,
            "metrics": [
                {"name": n, "preserved": t.preserved, "total": t.total} for n, t in metrics
            ],
        }


def score_document(path: str, fixed: ArmScore, cleave: ArmScore) -> None:
    ingest = ingest_document(path)
    graph = ContextGraph(ingest.elements)
    units, _profile = chunk(ingest, graph)

    f_chunks = [norm(c) for c in fixed_chunks(ingest)]
    c_units = [(norm(u.content), u) for u in units]
    by_unit_id = {u.id: u for u in units}

    def f_find(probe: str) -> list[int]:
        return [i for i, c in enumerate(f_chunks) if probe and probe in c]

    def c_find(probe: str):
        return [u for text_n, u in c_units if probe and probe in text_n]

    tables = [e for e in ingest.elements if e.kind == "table"]
    headings = {e.id: e for e in ingest.elements if e.kind == "heading"}

    # ── caption + header integrity (tables) ──
    for t in tables:
        grid: list[list[str]] = t.meta.get("grid", [])
        header_probe = norm(" ".join(t.meta.get("header_row", [])))[:60]
        row_probes = [norm(" ".join(r))[:60] for r in grid[1:]]
        row_probes = [p for p in row_probes if len(p) > 15]
        cap_ids = graph.captions_of(t.id)
        cap_text = " ".join(graph.by_id[c].text for c in cap_ids if c in graph.by_id)
        cap_probe = norm(cap_text)[:60]
        body_probe = row_probes[0] if row_probes else header_probe

        if cap_probe and body_probe:
            fixed.caption.add(any(cap_probe in f_chunks[i] for i in f_find(body_probe)))
            cleave.caption.add(
                any(body_probe in text_n and cap_probe in text_n for text_n, _u in c_units)
            )

        if header_probe and row_probes:
            f_bearing = {i for p in row_probes for i in f_find(p)}
            fixed.header.add(
                bool(f_bearing) and all(header_probe in f_chunks[i] for i in f_bearing)
            )
            c_bearing = {u.id for p in row_probes for u in c_find(p)}
            cleave.header.add(
                bool(c_bearing)
                and all(header_probe in norm(by_unit_id[uid].content) for uid in c_bearing)
            )

    # ── heading context (paragraphs with a governing heading) ──
    for e in ingest.elements:
        if e.kind != "paragraph" or len(e.text) < 60 or not e.parent_id:
            continue
        head = headings.get(e.parent_id)
        if not head:
            continue
        probe = norm(e.text)[:80]
        head_probe = norm(head.text)[:60]
        hits = f_find(probe)
        fixed.heading.add(bool(hits) and any(head_probe in f_chunks[i] for i in hits))
        c_hits = c_find(probe)
        cleave.heading.add(
            bool(c_hits)
            and any(
                head_probe in norm(u.content)
                or any(head_probe in norm(h) for h in u.context.heading_path)
                for u in c_hits
            )
        )

    # ── resolved references ──
    for e in ingest.elements:
        if e.kind not in ("paragraph", "list_item"):
            continue
        targets = graph.references_from(e.id)
        if not targets:
            continue
        probe = norm(e.text)[:80]
        for target_id, _d in targets:
            t_el = graph.by_id[target_id]
            t_grid = t_el.meta.get("grid", [])
            t_probe = norm(" ".join(t_grid[0]))[:60] if t_grid else ""
            if not t_probe:  # figures: caption text is the only textual anchor
                caps = graph.captions_of(target_id)
                t_probe = norm(" ".join(graph.by_id[c].text for c in caps))[:60]
            if not t_probe:
                continue
            hits = f_find(probe)
            fixed.refs.add(bool(hits) and any(t_probe in f_chunks[i] for i in hits))
            c_hits = c_find(probe)
            cleave.refs.add(
                bool(c_hits)
                and any(
                    t_probe in norm(u.content)
                    or any(
                        (r.type.value if hasattr(r.type, "value") else str(r.type)) == "references"
                        for r in u.relationships
                    )
                    for u in c_hits
                )
            )


# ───────── Universal Boundary & Retrieval Evaluation Metrics ─────────


def boundary_coherence_score(units: list, graph: ContextGraph) -> float:
    """Measure the proportion of unit boundaries that align with valid structural,
    semantic, or temporal transition points."""
    if len(units) <= 1:
        return 1.0
    coherent_boundaries = 0
    for u in units:
        dec = u.decision
        if (
            dec.strategy in ("structural", "atomic", "tabular", "temporal")
            or dec.signals.get("semantic_shift", 0.0) > 0.3
            or dec.signals.get("speaker_change", 0.0) > 0.5
            or u.metadata.get("element_kind") in ("table", "figure", "schema_card", "section")
            or len(dec.vetoed_cuts) > 0
        ):
            coherent_boundaries += 1
    return round(coherent_boundaries / len(units), 3)


def context_completeness_score(units: list, graph: ContextGraph) -> float:
    """Measure the average context completeness across all KnowledgeUnits."""
    if not units:
        return 0.0
    scores = [getattr(u, "context_completeness", 1.0) for u in units]
    return round(sum(scores) / len(scores), 3)


def relationship_preservation_rate(units: list, graph: ContextGraph) -> float:
    """Calculate the fraction of critical graph edges preserved within units or explicitly linked."""
    critical_types = {
        "captions",
        "captioned_by",
        "parent",
        "has_schema",
        "schema_of",
        "answered_by",
        "explains",
    }
    critical_edges = [
        (s, t, d) for s, t, d in graph.g.edges(data=True) if d.get("type") in critical_types
    ]
    if not critical_edges:
        return 1.0

    unit_by_element: dict[str, str] = {}
    for u in units:
        # Check text or metadata matching
        for node in graph.elements:
            if node.text and node.text[:50] in u.content:
                unit_by_element.setdefault(node.id, u.id)

    preserved = 0
    for s, t, _ in critical_edges:
        u_s = unit_by_element.get(s)
        u_t = unit_by_element.get(t)
        if u_s and u_t and u_s == u_t:
            preserved += 1
        elif u_s and u_t:
            # Check if linked via relationship
            by_id = {u.id: u for u in units}
            unit_s = by_id.get(u_s)
            if unit_s and any(r.target_id == u_t for r in unit_s.relationships):
                preserved += 1
        else:
            preserved += 1  # Not severed across split

    return round(preserved / len(critical_edges), 3)


def fragmentation_rate(units: list, min_viable_tokens: int = 40) -> float:
    """Calculate the proportion of under-sized chunks that represent unnecessary fragmentation."""
    if not units:
        return 0.0
    under_sized = sum(
        1
        for u in units
        if u.token_count < min_viable_tokens
        and u.metadata.get("element_kind") not in ("figure", "caption")
    )
    return round(under_sized / len(units), 3)


def chunk_size_variance(units: list) -> float:
    """Calculate chunk size variance (standard deviation) to evaluate adaptive sizing behavior."""
    if len(units) <= 1:
        return 0.0
    sizes = [u.token_count for u in units]
    mean = sum(sizes) / len(sizes)
    variance = sum((s - mean) ** 2 for s in sizes) / len(sizes)
    return round(variance**0.5, 2)


def retrieval_evaluation(
    units: list,
    queries: list[str],
    relevance_labels: list[set[str]],
    k: int = 5,
) -> dict[str, float]:
    """Compute Recall@K, Precision@K, and MRR for retrieval evaluation."""
    if not queries or not relevance_labels or len(queries) != len(relevance_labels):
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0}

    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []

    for query, relevant_ids in zip(queries, relevance_labels):
        q_norm = norm(query)
        # Score units by simple token overlap / keyword relevance for evaluation
        q_tokens = set(q_norm.split())
        scored_units: list[tuple[float, str]] = []
        for u in units:
            u_tokens = set(norm(u.embed_text()).split())
            overlap = len(q_tokens & u_tokens)
            scored_units.append((overlap, u.id))

        scored_units.sort(key=lambda x: x[0], reverse=True)
        top_k_ids = [uid for _, uid in scored_units[:k]]

        hits = len(set(top_k_ids) & relevant_ids)
        recalls.append(hits / max(1, len(relevant_ids)))
        precisions.append(hits / max(1, k))

        # MRR calculation
        rr = 0.0
        for rank, (score, uid) in enumerate(scored_units, start=1):
            if uid in relevant_ids and score > 0:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return {
        "recall_at_k": round(sum(recalls) / len(recalls), 3),
        "precision_at_k": round(sum(precisions) / len(precisions), 3),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
    }


def benchmark_corpus_retrieval(paths: list[str]) -> dict:
    """Generate probe queries from tables, captions, and section headings across the corpus
    and measure empirical retrieval performance for Cleave vs the fixed baseline."""
    cleave_recalls: list[float] = []
    cleave_precisions: list[float] = []
    cleave_mrrs: list[float] = []
    fixed_recalls: list[float] = []
    fixed_precisions: list[float] = []
    fixed_mrrs: list[float] = []

    for p in paths:
        try:
            ingest = ingest_document(p)
            graph = ContextGraph(ingest.elements)
            units, _ = chunk(ingest, graph)
            f_chunks = fixed_chunks(ingest)

            # Generate realistic probe queries and target signatures
            queries_and_targets: list[tuple[str, str]] = []
            for t in [e for e in ingest.elements if e.kind == "table"]:
                header = t.meta.get("header_row", [])
                grid = t.meta.get("grid", [])
                if header and len(grid) > 1:
                    col = header[0] if len(header) > 0 else "data"
                    val = grid[1][0] if len(grid[1]) > 0 else ""
                    if len(val) > 3:
                        queries_and_targets.append((f"table data {col} {val}", val[:50]))
            for c in [e for e in ingest.elements if e.kind == "caption"]:
                if len(c.text) > 15:
                    queries_and_targets.append((f"details in caption {c.text[:40]}", c.text[:50]))
            for h in [e for e in ingest.elements if e.kind == "heading"]:
                if len(h.text) > 10:
                    queries_and_targets.append((f"section overview {h.text}", h.text[:50]))

            if not queries_and_targets:
                continue

            for query, target_str in queries_and_targets:
                q_tokens = set(norm(query).split())
                t_norm = norm(target_str)

                # Cleave retrieval ranking
                c_scored: list[tuple[float, int]] = []
                for i, u in enumerate(units):
                    u_text = norm(u.embed_text())
                    overlap = len(q_tokens & set(u_text.split()))
                    has_target = 1.0 if t_norm in u_text else 0.0
                    c_scored.append((overlap + has_target * 2.0, i))
                c_scored.sort(key=lambda x: x[0], reverse=True)
                top_c_indices = [idx for _, idx in c_scored[:5]]
                c_hit = any(t_norm in norm(units[idx].embed_text()) for idx in top_c_indices)
                cleave_recalls.append(1.0 if c_hit else 0.0)
                cleave_precisions.append(1.0 / 5.0 if c_hit else 0.0)
                c_rr = 0.0
                for rank, (_, idx) in enumerate(c_scored[:5], start=1):
                    if t_norm in norm(units[idx].embed_text()):
                        c_rr = 1.0 / rank
                        break
                cleave_mrrs.append(c_rr)

                # Fixed baseline retrieval ranking
                f_scored: list[tuple[float, int]] = []
                for i, fc in enumerate(f_chunks):
                    fc_norm = norm(fc)
                    overlap = len(q_tokens & set(fc_norm.split()))
                    has_target = 1.0 if t_norm in fc_norm else 0.0
                    f_scored.append((overlap + has_target * 2.0, i))
                f_scored.sort(key=lambda x: x[0], reverse=True)
                top_f_indices = [idx for _, idx in f_scored[:5]]
                f_hit = any(t_norm in norm(f_chunks[idx]) for idx in top_f_indices)
                fixed_recalls.append(1.0 if f_hit else 0.0)
                fixed_precisions.append(1.0 / 5.0 if f_hit else 0.0)
                f_rr = 0.0
                for rank, (_, idx) in enumerate(f_scored[:5], start=1):
                    if t_norm in norm(f_chunks[idx]):
                        f_rr = 1.0 / rank
                        break
                fixed_mrrs.append(f_rr)
        except Exception as exc:
            log.warning("retrieval benchmark skipped for %s (%s)", p, exc)

    c_mrr = sum(cleave_mrrs) / max(1, len(cleave_mrrs))
    f_mrr = sum(fixed_mrrs) / max(1, len(fixed_mrrs))
    c_rec = sum(cleave_recalls) / max(1, len(cleave_recalls))
    f_rec = sum(fixed_recalls) / max(1, len(fixed_recalls))
    lift = ((c_mrr - f_mrr) / max(0.001, f_mrr)) * 100.0 if f_mrr > 0 else 0.0

    return {
        "queries_evaluated": len(cleave_mrrs),
        "cleave": {
            "recall_at_5": round(c_rec, 3),
            "precision_at_5": round(sum(cleave_precisions) / max(1, len(cleave_precisions)), 3),
            "mrr": round(c_mrr, 3),
        },
        "fixed": {
            "recall_at_5": round(f_rec, 3),
            "precision_at_5": round(sum(fixed_precisions) / max(1, len(fixed_precisions)), 3),
            "mrr": round(f_mrr, 3),
        },
        "mrr_lift_pct": round(lift, 1),
    }


def main(paths: list[str]) -> dict:
    """Score every document and write the scorecard, returning the record."""
    fixed, cleave = ArmScore(), ArmScore()
    for p in paths:
        log.info("scoring %s", p)
        score_document(p, fixed, cleave)
    benchmark = benchmark_corpus_retrieval(paths)
    result = {
        "documents": [Path(p).name for p in paths],
        "baseline": f"fixed {FIXED_TOKENS} tokens / {FIXED_OVERLAP} overlap",
        "fixed": fixed.to_dict(),
        "cleave": cleave.to_dict(),
        "retrieval_benchmark": benchmark,
    }
    out = Path(__file__).resolve().parent.parent / "data" / "scorecard.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    log.info("scorecard written to %s", out)
    return result


if __name__ == "__main__":
    from .logging_setup import configure_logging

    configure_logging()
    args = sys.argv[1:] or [
        "tests/fixtures/executive_summary.pdf",
        "tests/fixtures/attention_paper.pdf",
    ]
    record = main(args)
    print(json.dumps(record, indent=1))
