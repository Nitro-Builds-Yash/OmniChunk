"""Context graph: element-level relationships as a logical in-memory graph.

The graph is an active chunking intelligence — it constrains chunk boundaries,
evaluates relationship loss, assembles heading context, resolves cross-references,
and computes topological separation scores.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import networkx as nx

from .models import ContentElement

log = logging.getLogger(__name__)

_REF_RE = re.compile(r"\b(Table|Figure|Fig\.?)\s+(\d+)", re.IGNORECASE)

#: max vertical gap (PDF points) for treating a caption as adjacent to a float
_CAPTION_GAP_PT = 60.0

#: Edge importance weights for calculating relationship loss when boundaries cross edges
EDGE_IMPORTANCE: dict[str, float] = {
    "captions": 1.0,
    "captioned_by": 1.0,
    "parent": 0.95,
    "has_schema": 0.90,
    "schema_of": 0.90,
    "answered_by": 0.90,
    "question_for": 0.90,
    "explains": 0.85,
    "illustrated_by": 0.85,
    "occurs_during": 0.80,
    "occurs_in_scene": 0.80,
    "slide_content_of": 0.80,
    "references": 0.75,
    "next": 0.05,
}

_QA_PATTERNS = [
    re.compile(
        r"^(who|what|where|when|why|how|can|could|would|should|is|are|do|does|did)\b.*\?", re.I
    ),
]


class ContextGraph:
    def __init__(self, elements: list[ContentElement]):
        self.elements = elements
        self.by_id = {e.id: e for e in elements}
        self.g = nx.DiGraph()
        for e in elements:
            self.g.add_node(e.id, kind=e.kind)
        self._build()

    @property
    def node_count(self) -> int:
        return self.g.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.g.number_of_edges()

    # ───────── construction ─────────

    def _build(self) -> None:
        self._hierarchy()
        self._captions()
        self._references()
        self._explains_and_illustrates()
        self._conversational_edges()
        self._multimodal_edges()
        self._reading_order()

    def _add(
        self,
        src: str,
        dst: str,
        type_: str,
        confidence: float,
        evidence: str,
        importance: float | None = None,
    ) -> None:
        imp = importance if importance is not None else EDGE_IMPORTANCE.get(type_, 0.5)
        self.g.add_edge(
            src, dst, type=type_, confidence=confidence, evidence=evidence, importance=imp
        )

    def _hierarchy(self) -> None:
        for e in self.elements:
            if e.parent_id and e.parent_id in self.by_id:
                self._add(e.parent_id, e.id, "parent", 1.0, "heading ancestry", importance=0.95)

    def _captions(self) -> None:
        captions = [e for e in self.elements if e.kind == "caption"]
        claimed: set[str] = set()
        # pass 1: Docling's own caption references — deterministic
        for e in self.elements:
            if e.kind not in ("table", "figure"):
                continue
            for cid in e.meta.get("caption_ids", []):
                if cid in self.by_id:
                    self._add(
                        cid, e.id, "captions", 1.0, "Docling caption reference", importance=1.0
                    )
                    self._add(
                        e.id, cid, "captioned_by", 1.0, "Docling caption reference", importance=1.0
                    )
                    claimed.add(cid)
        # pass 2: bbox adjacency for floats Docling left uncaptioned
        for e in self.elements:
            if e.kind not in ("table", "figure") or self.captions_of(e.id):
                continue
            best: tuple[float, ContentElement] | None = None
            for c in captions:
                if c.id in claimed or c.page != e.page or not (c.bbox and e.bbox):
                    continue
                gap = _vertical_gap(e.bbox, c.bbox)
                if (
                    gap <= _CAPTION_GAP_PT
                    and _h_overlap(e.bbox, c.bbox) > 0
                    and (best is None or gap < best[0])
                ):
                    best = (gap, c)
            if best:
                gap, c = best
                ev = f"bbox adjacency: caption {gap:.0f}pt from {e.kind} on page {e.page}"
                self._add(c.id, e.id, "captions", 0.8, ev, importance=1.0)
                self._add(e.id, c.id, "captioned_by", 0.8, ev, importance=1.0)
                claimed.add(c.id)
                e.meta.setdefault("caption_ids", []).append(c.id)

    def _references(self) -> None:
        """'Table 3' / 'Figure 2' mentions → edges to the actual float.
        Resolution: caption text containing the label wins; ordinal fallback."""
        tables = [e for e in self.elements if e.kind == "table"]
        figures = [e for e in self.elements if e.kind == "figure"]

        # Pre-build caption-based index: ("table", N) → float element.
        # This replaces per-match regex compilation with a single upfront scan.
        _caption_index: dict[tuple[str, int], ContentElement] = {}
        _TABLE_LABEL = re.compile(r"\bTable\s+(\d+)\b", re.I)
        _FIG_LABEL = re.compile(r"\bFig(?:ure)?\.?\s+(\d+)\b", re.I)
        for pool, label_re, key_prefix in (
            (tables, _TABLE_LABEL, "table"),
            (figures, _FIG_LABEL, "figure"),
        ):
            for f in pool:
                for cid in f.meta.get("caption_ids", []):
                    cap = self.by_id.get(cid)
                    if cap:
                        for lm in label_re.finditer(cap.text):
                            _caption_index.setdefault((key_prefix, int(lm.group(1))), f)

        def resolve(word: str, num: int) -> ContentElement | None:
            key_prefix = "table" if word.lower().startswith("t") else "figure"
            hit = _caption_index.get((key_prefix, num))
            if hit:
                return hit
            # ordinal fallback
            pool = tables if key_prefix == "table" else figures
            return pool[num - 1] if 0 < num <= len(pool) else None

        for e in self.elements:
            if e.kind not in ("paragraph", "list_item", "caption"):
                continue
            for m in _REF_RE.finditer(e.text):
                target = resolve(m.group(1), int(m.group(2)))
                if target and target.id != e.id and not self.g.has_edge(e.id, target.id):
                    self._add(
                        e.id,
                        target.id,
                        "references",
                        0.9,
                        f"text mentions {m.group(0)!r}",
                        importance=0.75,
                    )

    def _explains_and_illustrates(self) -> None:
        """Detect prose directly preceding or explaining a table/figure."""
        for i, e in enumerate(self.elements):
            if e.kind in ("table", "figure") and i > 0:
                prev = self.elements[i - 1]
                if (
                    prev.kind in ("paragraph", "list_item")
                    and len(prev.text) > 20
                    and not self.g.has_edge(prev.id, e.id)
                ):
                    self._add(
                        prev.id,
                        e.id,
                        "explains",
                        0.85,
                        f"prose introduces adjacent {e.kind}",
                        importance=0.85,
                    )
                    self._add(
                        e.id,
                        prev.id,
                        "illustrated_by",
                        0.85,
                        f"{e.kind} illustrated by preceding prose",
                        importance=0.85,
                    )

    def _conversational_edges(self) -> None:
        """Detect question-answer and dialogue dependencies in speech or prose."""
        for i in range(len(self.elements) - 1):
            a, b = self.elements[i], self.elements[i + 1]
            if (
                a.text.strip().endswith("?") or any(rx.search(a.text) for rx in _QA_PATTERNS)
            ) and b.kind in ("speech_segment", "paragraph"):
                self._add(
                    a.id,
                    b.id,
                    "answered_by",
                    0.90,
                    "question answered by following element",
                    importance=0.90,
                )
                self._add(
                    b.id,
                    a.id,
                    "question_for",
                    0.90,
                    "answer directly addresses question",
                    importance=0.90,
                )

    def _multimodal_edges(self) -> None:
        """Detect temporal overlap and visual co-occurrence in video/audio."""
        visual_events = [e for e in self.elements if e.kind == "visual_event"]
        speech_segments = [e for e in self.elements if e.kind == "speech_segment"]

        for s in speech_segments:
            if s.t0 is None or s.t1 is None:
                continue
            for v in visual_events:
                if v.t0 is None or v.t1 is None:
                    continue
                # Check for temporal overlap
                overlap = min(s.t1, v.t1) - max(s.t0, v.t0)
                if overlap > 0:
                    self._add(
                        s.id,
                        v.id,
                        "occurs_during",
                        0.85,
                        f"speech span ({s.t0:.1f}s–{s.t1:.1f}s) overlaps visual event ({v.t0:.1f}s–{v.t1:.1f}s)",
                        importance=0.80,
                    )

    def _reading_order(self) -> None:
        for a, b in zip(self.elements, self.elements[1:]):
            if not self.g.has_edge(a.id, b.id):
                self._add(a.id, b.id, "next", 1.0, "reading order", importance=0.05)

    # ───────── queries & graph intelligence ─────────

    def heading_path(self, el_id: str) -> list[str]:
        path: list[str] = []
        seen: set[str] = set()
        cur = self.by_id.get(el_id)
        while cur and cur.parent_id and cur.parent_id not in seen:
            seen.add(cur.parent_id)
            parent = self.by_id.get(cur.parent_id)
            if parent is None:
                break
            path.append(parent.text)
            cur = parent
        return list(reversed(path))

    def captions_of(self, float_id: str) -> list[str]:
        return [
            src for src, _, d in self.g.in_edges(float_id, data=True) if d["type"] == "captions"
        ]

    def surrounding_text(self, el_id: str, max_chars: int = 320) -> tuple[str | None, str | None]:
        """Nearest prose before and after an element."""
        try:
            idx = next(i for i, e in enumerate(self.elements) if e.id == el_id)
        except StopIteration:
            return None, None
        prose = ("paragraph", "list_item")

        def scan(rng) -> str | None:
            for i in rng:
                e = self.elements[i]
                if e.kind in prose and len(e.text) > 40:
                    return e.text[:max_chars]
                if e.kind == "heading":
                    break  # a heading is a hard context boundary
            return None

        return scan(range(idx - 1, -1, -1)), scan(range(idx + 1, len(self.elements)))

    def references_to(self, el_id: str) -> list[tuple[str, dict[str, Any]]]:
        return [
            (src, d) for src, _, d in self.g.in_edges(el_id, data=True) if d["type"] == "references"
        ]

    def references_from(self, el_id: str) -> list[tuple[str, dict[str, Any]]]:
        return [
            (dst, d)
            for _, dst, d in self.g.out_edges(el_id, data=True)
            if d["type"] == "references"
        ]

    def relationship_loss(self, left_ids: set[str], right_ids: set[str]) -> tuple[float, list[str]]:
        """Calculate the penalty and severed reasons when cutting between left_ids and right_ids.

        Higher loss means a boundary would sever strong relationships (e.g. caption ↔ float,
        question ↔ answer, explanatory prose ↔ table).
        """
        loss = 0.0
        severed: list[str] = []
        for src, dst, d in self.g.edges(data=True):
            edge_type = d.get("type", "")
            if edge_type in ("next", "previous"):
                continue  # linear sequence cuts are normal
            if (src in left_ids and dst in right_ids) or (src in right_ids and dst in left_ids):
                imp = d.get("importance", EDGE_IMPORTANCE.get(edge_type, 0.5))
                loss += imp
                severed.append(f"severs {edge_type} ({src} ↔ {dst}): {d.get('evidence', '')}")
        return loss, severed

    def graph_separation_score(self, el_a_id: str, el_b_id: str) -> float:
        """Measure topological separation in the graph between adjacent elements (0.0 to 1.0).

        Returns 1.0 if unconnected or in different branches, 0.0 if strongly interconnected.
        """
        if el_a_id not in self.g or el_b_id not in self.g:
            return 1.0
        # If directly connected by non-next edge, separation is very low
        if any(self.g.has_edge(el_a_id, el_b_id) or self.g.has_edge(el_b_id, el_a_id) for _ in [1]):
            direct_d = self.g.get_edge_data(el_a_id, el_b_id) or self.g.get_edge_data(
                el_b_id, el_a_id
            )
            if direct_d and direct_d.get("type") not in ("next", "previous"):
                return 0.1
        # Check heading ancestry separation
        path_a = self.heading_path(el_a_id)
        path_b = self.heading_path(el_b_id)
        if path_a != path_b:
            return 0.95
        return 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": e.id,
                    "kind": e.kind,
                    "page": e.page,
                    "text": e.text[:120] + ("…" if len(e.text) > 120 else ""),
                }
                for e in self.elements
            ],
            "edges": [
                {
                    "source": s,
                    "target": t,
                    "type": d["type"],
                    "confidence": round(d["confidence"], 3),
                    "evidence": d["evidence"],
                    "importance": round(d.get("importance", 0.5), 2),
                }
                for s, t, d in self.g.edges(data=True)
            ],
        }


def _vertical_gap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Min vertical distance between two boxes, robust to bbox origin
    (Docling PDF boxes are bottom-left origin, so t > b)."""
    a_lo, a_hi = min(a[1], a[3]), max(a[1], a[3])
    b_lo, b_hi = min(b[1], b[3]), max(b[1], b[3])
    if a_hi < b_lo:
        return b_lo - a_hi
    if b_hi < a_lo:
        return a_lo - b_hi
    return 0.0


def _h_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
