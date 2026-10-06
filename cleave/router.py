"""Adaptive routing: profile the content, choose a strategy, choose safe cuts.

Three routing checks in priority order (temporal → structural → flat fallback),
atomic elements carved out first regardless. Escalation is orthogonal: cheap
triggers mark which chunks *deserve* an LLM, whether or not one is ever called.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .boundary_engine import choose_universal_cut
from .graph import ContextGraph
from .models import ContentElement, Profile, count_tokens

log = logging.getLogger(__name__)

TARGET_TOKENS = 500
MAX_TOKENS = 700

_TEXT_KINDS = ("paragraph", "list_item", "code", "caption")

# Sentences that lean on context that may live outside the chunk.
# Combined into a single alternation for a single regex match per sentence
# instead of 4 separate calls.
_ANAPHORA_RE = re.compile(
    r"\bas (shown|described|noted|mentioned|discussed) (above|earlier|previously|below)\b"
    r"|\bthis (table|figure|section|chart|diagram|approach|result)\b"
    r"|\bthe (former|latter)\b"
    r"|^(This|These|It|They)\b",
    re.I | re.M,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def is_tabular(elements: list[ContentElement]) -> bool:
    """True when the input IS a dataset rather than prose that contains tables.

    The discriminator is the share of content carried by tables: a report with
    a couple of tables still reads as prose, a spreadsheet is nothing else.
    """
    tables = [e for e in elements if e.kind == "table"]
    if not tables or not any(e.meta.get("grid") for e in tables):
        return False
    prose = [e for e in elements if e.kind in ("paragraph", "list_item")]
    table_tokens = sum(count_tokens(e.text) for e in tables)
    prose_tokens = sum(count_tokens(e.text) for e in prose)
    return table_tokens > 0 and table_tokens >= 4 * prose_tokens


def build_profile(elements: list[ContentElement]) -> Profile:
    text_els = [e for e in elements if e.kind in _TEXT_KINDS]
    headings = [e for e in elements if e.kind == "heading"]
    p = Profile(
        element_count=len(elements),
        text_element_count=len(text_els),
        heading_count=len(headings),
        heading_density=len(headings) / max(1, len(text_els)),
        table_count=sum(1 for e in elements if e.kind == "table"),
        figure_count=sum(1 for e in elements if e.kind == "figure"),
        caption_count=sum(1 for e in elements if e.kind == "caption"),
        has_timestamps=any(e.t0 is not None for e in elements),
        total_tokens=sum(count_tokens(e.text) for e in elements),
    )
    p.is_tabular = is_tabular(elements)
    if p.is_tabular:
        grids = [e.meta.get("grid") or [] for e in elements if e.kind == "table"]
        p.row_count = sum(max(0, len(g) - 1) for g in grids)
        p.column_count = max((len(g[0]) for g in grids if g), default=0)
    p.route, p.route_reason = _route(p)
    return p


def _route(p: Profile) -> tuple[str, str]:
    if p.has_timestamps:
        return "temporal", "content carries timestamps — chunking follows speakers and time"
    if p.is_tabular:
        sheets = f"{p.table_count} sheet{'s' if p.table_count != 1 else ''}"
        return "tabular", (
            f"{sheets}, {p.row_count:,} rows × {p.column_count} columns and almost no prose — "
            "this is a dataset: chunk by row groups, repeat the header, profile the schema"
        )
    if p.heading_count >= 3 and p.heading_density >= 0.03:
        return "structural", (
            f"{p.heading_count} headings over {p.text_element_count} text elements "
            f"(density {p.heading_density:.2f}) — document structure is trustworthy, "
            "sections define chunks"
        )
    base_reason = (
        f"only {p.heading_count} headings for {p.text_element_count} text elements — "
        "no usable hierarchy"
    )
    try:
        from .semantic import available  # noqa: PLC0415

        if available():
            return "semantic", base_reason + ", grouping by embedding topic drift"
    except Exception as exc:
        log.warning("semantic availability probe failed (%s) — routing to paragraph_fallback", exc)
    return "paragraph_fallback", base_reason + ", packing at paragraph boundaries"


# ───────── cut selection with universal boundary engine and hard vetoes ─────────


@dataclass(slots=True)
class CutResult:
    index: int | None  # boundary BEFORE elements[index]; None = keep whole
    vetoes: list[str] = field(default_factory=list)
    overflow: bool = False
    trace: dict[str, Any] = field(default_factory=dict)


def choose_cut(
    region: list[ContentElement], graph: ContextGraph, target_tokens: int = TARGET_TOKENS
) -> CutResult:
    """Pick the optimal boundary that maximizes multi-modal cohesion and severs
    no hard relationship. Overflow beats severance: if every candidate is vetoed,
    the region stays whole and says so."""
    res = choose_universal_cut(region, graph, target_tokens=target_tokens)
    return CutResult(index=res.index, vetoes=res.vetoes, overflow=res.overflow, trace=res.trace)


def _caption_pair(a: ContentElement, b: ContentElement, graph: ContextGraph) -> str | None:
    for x, y in ((a, b), (b, a)):
        if (
            x.kind == "caption"
            and y.kind in ("table", "figure")
            and x.id in graph.captions_of(y.id)
        ):
            return f"CAPTIONS {x.id} ↔ {y.id}"
    return None


# ───────── escalation (flags only — executing LLM calls is a stretch stage) ─────────


def anaphora_rate(text: str) -> float:
    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        return 0.0
    hits = sum(1 for s in sentences if _ANAPHORA_RE.search(s))
    return hits / len(sentences)


def escalation_flags(
    content: str,
    heading_path: list[str],
    strategy: str,
    kind: str = "text",
    has_caption: bool = True,
) -> list[str]:
    flags: list[str] = []
    rate = anaphora_rate(content)
    if rate > 0.10:
        flags.append(f"anaphora rate {rate:.2f} — meaning leans on context outside the chunk")
    if not heading_path and strategy not in ("temporal", "atomic"):
        flags.append("orphan — no heading ancestry to situate it")
    if kind in ("table", "figure") and not has_caption:
        flags.append(f"uncaptioned {kind} — nothing states what it shows")
    return flags
