"""Context completeness scoring and dependency resolution.

Evaluates whether a generated chunk can be understood independently and attaches or
links any missing context.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .graph import ContextGraph
from .models import KnowledgeUnit, Modality, RelationType

log = logging.getLogger(__name__)

_ANAPHORA_PATTERNS = [
    re.compile(
        r"\bas (shown|described|noted|mentioned|discussed) (above|earlier|previously|below)\b", re.I
    ),
    re.compile(r"\bthis (table|figure|section|chart|diagram|approach|result|data)\b", re.I),
    re.compile(r"\bthe (former|latter)\b", re.I),
    re.compile(r"^(This|These|It|They)\b"),
    re.compile(r"\bsee (table|figure|fig\.?|section)\s+\d+\b", re.I),
]

_VISUAL_DEICTIC = re.compile(
    r"\b(as you can see (here|on (this|the) slide)|on the (left|right|top|bottom)|in this diagram)\b",
    re.I,
)


@dataclass(slots=True)
class ContextCompleteness:
    score: float
    missing_dependencies: list[str] = field(default_factory=list)
    attached_context: list[str] = field(default_factory=list)


def evaluate_context_completeness(
    unit: KnowledgeUnit,
    graph: ContextGraph,
) -> ContextCompleteness:
    """Assess whether a KnowledgeUnit has all necessary context to stand alone.

    Completeness Score:
        1.0 = fully self-contained or fully anchored by context
        0.0 = dangling references with no heading or situating context
    """
    missing: list[str] = []
    attached: list[str] = []
    total_checks = 0
    passed_checks = 0

    content = unit.content

    # 1. Document / Heading hierarchy context
    if unit.modality in (Modality.DOCUMENT, Modality.TEXT):
        total_checks += 1
        if unit.context.heading_path:
            passed_checks += 1
            attached.append(f"heading path: {' > '.join(unit.context.heading_path)}")
        elif unit.metadata.get("element_kind") not in ("schema_card", "table"):
            missing.append("missing heading ancestry or document section context")
        else:
            passed_checks += 1

    # 2. Tabular schema context
    if unit.metadata.get("element_kind") == "row_group":
        total_checks += 1
        has_schema_rel = any(
            r.type in (RelationType.HAS_SCHEMA, "has_schema") for r in unit.relationships
        )
        has_columns = bool(unit.metadata.get("columns"))
        if has_schema_rel and has_columns:
            passed_checks += 1
            attached.append("schema card reference and column headers attached")
        else:
            missing.append("missing dataset schema card or column headers")

    # 3. Float & Caption integrity
    if unit.metadata.get("element_kind") in ("table", "figure"):
        total_checks += 1
        caption_signal = unit.decision.signals.get("caption_confidence", 0.0)
        if caption_signal > 0.5 or "caption" in unit.metadata:
            passed_checks += 1
            attached.append("caption attached to float")
        else:
            missing.append(f"uncaptioned {unit.metadata.get('element_kind')}")

    # 4. Anaphora & dangling cross-references
    total_checks += 1
    anaphora_found: list[str] = []
    for rx in _ANAPHORA_PATTERNS:
        match = rx.search(content)
        if match:
            anaphora_found.append(match.group(0))

    if anaphora_found:
        if unit.context.leading or unit.context.situating_summary or unit.relationships:
            passed_checks += 1
            attached.append(
                f"anaphora situated via surrounding context ({len(anaphora_found)} references)"
            )
        else:
            missing.append(
                f"dangling anaphoric references: {', '.join(repr(a) for a in anaphora_found[:3])}"
            )
    else:
        passed_checks += 1

    # 5. Audio / Speaker context
    if unit.modality == Modality.AUDIO or unit.temporal:
        total_checks += 1
        if unit.temporal and unit.temporal.speaker:
            passed_checks += 1
            attached.append(f"speaker attribution: {unit.temporal.speaker}")
        else:
            missing.append("unattributed speaker in speech segment")

    # 6. Video multimodal visual context
    if unit.modality == Modality.VIDEO or (unit.temporal and "visual_summary" in unit.metadata):
        total_checks += 1
        has_visual = bool(unit.metadata.get("visual_summary") or unit.metadata.get("scene"))
        refers_to_visual = bool(_VISUAL_DEICTIC.search(content))
        if has_visual:
            passed_checks += 1
            attached.append(f"visual metadata: {unit.metadata.get('visual_summary')}")
        elif refers_to_visual:
            missing.append(
                "speech refers to on-screen visual content, but visual metadata is missing"
            )
        else:
            passed_checks += 1

    score = passed_checks / max(1, total_checks)
    return ContextCompleteness(
        score=score,
        missing_dependencies=missing,
        attached_context=attached,
    )


def enrich_context_completeness(unit: KnowledgeUnit, graph: ContextGraph) -> KnowledgeUnit:
    """Evaluate completeness and update unit's completeness metadata."""
    completeness = evaluate_context_completeness(unit, graph)
    unit.context_completeness = round(completeness.score, 3)
    unit.missing_context = completeness.missing_dependencies
    if completeness.attached_context:
        unit.metadata["attached_context"] = completeness.attached_context
    return unit
