"""Cleave data contract.

Plain dataclasses (not pydantic) so the pipeline stays importable without the
web stack; the server serializes via ``to_dict()``. This module is the frozen
contract shared with external modality workers (see CONTRACT.md) — change it
deliberately or not at all.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

# ───────── enums ─────────


class Modality(str, Enum):
    TEXT = "text"
    DOCUMENT = "document"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    #: Reserved for contract consumers; Cleave itself never emits it.
    MULTIMODAL = "multimodal"


class KnowledgeUnitType(str, Enum):
    """Semantic classification of a knowledge unit."""

    SECTION = "section"
    TABLE = "table"
    FIGURE = "figure"
    SCHEMA_CARD = "schema_card"
    ROW_GROUP = "row_group"
    PROCEDURE = "procedure"
    QUESTION_ANSWER = "question_answer"
    DISCUSSION = "discussion"
    DECISION = "decision"
    ACTION_ITEM = "action_item"
    SPEAKER_TURN = "speaker_turn"
    VISUAL_SCENE = "visual_scene"
    MULTIMODAL_EVENT = "multimodal_event"
    VIDEO_EVENT = "video_event"  # event-level multimodal chunk from universal boundary engine
    NARRATIVE = "narrative"
    GENERIC = "generic"


class RelationType(str, Enum):
    """Universal graph relationship types."""

    CAPTIONS = "captions"
    CAPTIONED_BY = "captioned_by"
    REFERENCES = "references"
    NEXT = "next"
    PREVIOUS = "previous"
    SPOKEN_BY = "spoken_by"  # reserved for contract consumers; not emitted here
    SCHEMA_OF = "schema_of"  # schema card → the row groups it describes
    HAS_SCHEMA = "has_schema"  # row group → its schema card
    # Universal graph relation types:
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"
    EXPLAINS = "explains"
    ILLUSTRATED_BY = "illustrated_by"
    ANSWERED_BY = "answered_by"
    QUESTION_FOR = "question_for"
    PRODUCES_DECISION = "produces_decision"
    OCCURS_DURING = "occurs_during"
    PERFORMS_ACTION = "performs_action"
    OCCURS_IN_SCENE = "occurs_in_scene"
    SLIDE_CONTENT_OF = "slide_content_of"


# ───────── intermediate representation ─────────

#: Element kinds every modality normalizes into. Documents use the first row;
#: audio/video workers use the second.
ELEMENT_KINDS = (
    "heading",
    "paragraph",
    "list_item",
    "table",
    "figure",
    "caption",
    "code",
    "speech_segment",
    "visual_event",
    "other",
)


@dataclass(slots=True)
class ContentElement:
    """One normalized piece of source content — the universal intermediate
    representation. Kept intentionally thin: confidence/evidence live on
    relationships (where they earn their keep), not on every element."""

    id: str
    kind: str
    text: str
    level: int | None = None  # heading depth (1-based)
    parent_id: str | None = None  # hierarchy pointer (heading ancestry)
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    t0: float | None = None  # temporal span, seconds
    t1: float | None = None
    speaker: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)  # e.g. table grid rows


# ───────── universal boundary candidate ─────────


@dataclass(slots=True)
class BoundaryCandidate:
    """Evidence about where an information boundary exists across any modality.

    Extractors produce candidates; the universal boundary engine scores them,
    validates constraints, and decides optimal cuts.
    """

    index: int | None = None  # boundary index in element stream (cut before region[index])
    position: int | None = None  # token or char offset
    timestamp: float | None = None  # time in seconds
    modality: Modality | str = Modality.DOCUMENT
    left_element_id: str | None = None
    right_element_id: str | None = None
    signals: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "general"
    reason: str = ""
    is_hard: bool = False
    is_soft: bool = False
    veto_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "position": self.position,
            "timestamp": round(self.timestamp, 3) if self.timestamp is not None else None,
            "modality": self.modality.value
            if isinstance(self.modality, Modality)
            else str(self.modality),
            "left_element_id": self.left_element_id,
            "right_element_id": self.right_element_id,
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "reason": self.reason,
            "is_hard": self.is_hard,
            "is_soft": self.is_soft,
            "veto_reasons": list(self.veto_reasons),
        }


# ───────── knowledge unit parts ─────────


@dataclass(slots=True)
class Relationship:
    type: RelationType | str
    target_id: str
    confidence: float = 1.0  # 1.0 deterministic · 0.7-0.9 heuristic · ≤0.6 inferred
    evidence: str | None = None  # human-readable mechanism, e.g. bbox adjacency


@dataclass(slots=True)
class Provenance:
    source_uri: str
    source_sha256: str | None = None
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class Temporal:
    start_s: float
    end_s: float
    speaker: str | None = None


@dataclass(slots=True)
class Context:
    """What a reader needs to understand ``content`` standing alone."""

    document_title: str | None = None
    heading_path: list[str] = field(default_factory=list)
    situating_summary: str | None = None  # LLM-written; None until enrichment runs
    leading: str | None = None  # surrounding-text windows (optional)
    trailing: str | None = None
    tier: int = 0  # 0 deterministic · 1 local model · 2 LLM


@dataclass(slots=True)
class ChunkingDecision:
    """Why this chunk exists — the receipt shown in the demo. Every unit is
    explainable: strategy, reason, the cuts we refused to make, and cost."""

    strategy: str  # structural | paragraph_fallback | semantic |
    # temporal | atomic | universal
    reason: str
    signals: dict[str, float] = field(default_factory=dict)
    vetoed_cuts: list[str] = field(default_factory=list)
    severed_refs: int = 0  # reserved: the vetoes mean this stays 0
    escalation_flags: list[str] = field(default_factory=list)  # why it WOULD deserve an LLM
    llm_calls: int = 0
    cost_usd: float = 0.0


@dataclass(slots=True)
class KnowledgeUnit:
    id: str
    content: str
    modality: Modality
    context: Context
    provenance: Provenance
    decision: ChunkingDecision
    relationships: list[Relationship] = field(default_factory=list)
    temporal: Temporal | None = None
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    knowledge_unit_type: str = "generic"
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    level: int | None = None
    context_completeness: float = 1.0
    missing_context: list[str] = field(default_factory=list)
    boundary_trace: dict[str, Any] = field(default_factory=dict)

    def embed_text(self) -> str:
        """Exactly what a downstream system should embed — context first, so a
        chunk retrieved alone still knows where it lives.

        Surrounding prose is included because it is often the only thing that
        says what a table or figure is *for*; it stays out of `content` so a
        consumer can still display the chunk clean.
        """
        parts: list[str] = []
        if self.context.heading_path:
            parts.append(" > ".join(self.context.heading_path))
        if self.context.situating_summary:
            parts.append(self.context.situating_summary)
        if self.context.leading:
            parts.append(f"[before] {self.context.leading}")
        parts.append(self.content)
        if self.context.trailing:
            parts.append(f"[after] {self.context.trailing}")
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["modality"] = (
            self.modality.value if isinstance(self.modality, Modality) else str(self.modality)
        )
        for rel in d["relationships"]:
            rel["type"] = (
                rel["type"].value if isinstance(rel["type"], RelationType) else str(rel["type"])
            )
            rel["confidence"] = round(rel["confidence"], 3)
        d["decision"]["cost_usd"] = round(d["decision"]["cost_usd"], 6)
        d["decision"]["signals"] = {k: round(v, 4) for k, v in d["decision"]["signals"].items()}
        if d["temporal"]:
            d["temporal"]["start_s"] = round(d["temporal"]["start_s"], 3)
            d["temporal"]["end_s"] = round(d["temporal"]["end_s"], 3)
        d["embed_text"] = self.embed_text()
        d["context_completeness"] = round(self.context_completeness, 3)
        return d


# ───────── document profile ─────────


@dataclass(slots=True)
class Profile:
    """Cheap deterministic signals the router reads. Milliseconds, no models."""

    element_count: int = 0
    text_element_count: int = 0
    heading_count: int = 0
    heading_density: float = 0.0
    table_count: int = 0
    figure_count: int = 0
    caption_count: int = 0
    has_timestamps: bool = False
    is_tabular: bool = False  # the input IS a dataset, not prose containing tables
    row_count: int = 0
    column_count: int = 0
    total_tokens: int = 0
    route: str = ""  # strategy chosen for the non-atomic remainder
    route_reason: str = ""
    #: What the chunker actually ran. Usually identical to ``route``; they differ
    #: when the semantic route was chosen but produced no usable topic groups,
    #: which the profile previously reported as a successful semantic run.
    route_actual: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["heading_density"] = round(self.heading_density, 4)
        return d


# ───────── shared helpers ─────────


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken  # noqa: PLC0415 — lazy: keep import time sane

    return tiktoken.get_encoding("cl100k_base")


@lru_cache(maxsize=4096)
def _count_cached(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


def count_tokens(text: str) -> int:
    """Token count, memoised.

    The same element text is counted several times per chunk — once when a
    region is measured, again when the unit records its signals, again for the
    joined content — and tiktoken is not free. Long strings skip the cache: a
    whole document's joined content is rarely asked for twice and would hold
    megabytes resident.
    """
    if not text:
        return 0
    if len(text) > 20_000:
        return len(_encoder().encode(text, disallowed_special=()))
    return _count_cached(text)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
