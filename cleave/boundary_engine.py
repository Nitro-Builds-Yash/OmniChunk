"""Universal Boundary Decision Engine.

Extractors produce evidence about boundaries across modalities. The boundary engine
combines multi-modal signals, evaluates graph relationship loss, applies hard and soft
constraints, and optimizes chunk boundaries to determine what can safely be separated
without destroying meaning, context, relationships, or attribution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .config import settings
from .graph import ContextGraph
from .models import (
    BoundaryCandidate,
    ContentElement,
    Modality,
    count_tokens,
)

log = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_QA_RE = re.compile(
    r"^(who|what|where|when|why|how|can|could|would|should|is|are|do|does|did)\b", re.I
)


@dataclass(slots=True)
class UniversalCutResult:
    index: int | None  # boundary before region[index]; None = keep whole
    vetoes: list[str] = field(default_factory=list)
    overflow: bool = False
    chosen_candidate: BoundaryCandidate | None = None
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


# ───────── Universal Boundary Candidate Generation ─────────


def generate_candidates_for_region(
    region: list[ContentElement],
    graph: ContextGraph,
    modality: Modality | str = Modality.DOCUMENT,
) -> list[BoundaryCandidate]:
    """Generate boundary candidate split points for an element sequence."""
    if len(region) < 2:
        return []

    candidates: list[BoundaryCandidate] = []
    tokens = [count_tokens(e.text) for e in region]
    accumulated_tokens = 0

    for i in range(1, len(region)):
        before, after = region[i - 1], region[i]
        accumulated_tokens += tokens[i - 1]
        signals: dict[str, float] = {}
        vetoes: list[str] = []
        is_hard = False
        is_soft = False
        reasons: list[str] = []

        # 1. Structural signals (headings, section depths)
        if after.kind == "heading":
            level = after.level or 1
            signals["structural_strength"] = max(0.6, 1.0 - (level - 1) * 0.15)
            reasons.append(f"heading transition to {after.text[:40]!r} (level {level})")
            if level <= 2:
                is_hard = True
            else:
                is_soft = True
        elif before.kind == "heading":
            # HARD VETO: cutting after heading would strand the heading at the end of the previous chunk
            vetoes.append(f"would strand heading {before.text[:60]!r} at chunk boundary")

        # 2. Float & Caption integrity (hard constraint)
        cap_pair = _check_caption_pair(before, after, graph)
        if cap_pair:
            vetoes.append(f"would sever caption pair: {cap_pair}")

        # 3. List item transitions
        if before.kind == "list_item" and after.kind == "list_item":
            signals["list_continuation"] = 1.0
            is_soft = True

        # 4. Temporal signals (audio / speech / video)
        if before.t1 is not None and after.t0 is not None:
            gap = max(0.0, float(after.t0 - before.t1))
            signals["temporal_gap"] = min(1.0, gap / 3.0)
            if (
                before.speaker != after.speaker
                and before.speaker is not None
                and after.speaker is not None
            ):
                signals["speaker_change"] = 1.0
                reasons.append(f"speaker change ({before.speaker} → {after.speaker})")
                # Speaker change is a STRONG soft boundary — not a hard one.
                # The boundary engine decides based on combined evidence; a single
                # speaker change does not automatically split a conversational event.
                is_soft = True
            elif gap >= 2.0:
                signals["pause_strength"] = min(1.0, gap / 5.0)
                reasons.append(f"pause duration {gap:.1f}s")
                is_soft = True

        # 5. Visual signals (video keyframes, visual events, scene transitions)
        # Shot-level camera cuts are weaker than semantic scene/context transitions.
        # Use meta["visual_change_type"] when the video worker provides it:
        #   "shot"  — camera angle/cut only (weak boundary)
        #   "scene" — location/activity/context change (strong boundary)
        #   "slide" — slide/screen change (strong boundary)
        #   "event" — meaningful occurrence (strong boundary)
        cfg_ref = settings()
        vc_type_before = before.meta.get("visual_change_type")
        vc_type_after = after.meta.get("visual_change_type")
        vc_type = vc_type_after or vc_type_before  # prefer the arriving element's type

        before_vis = before.meta.get("visual_summary") or before.meta.get("scene")
        after_vis = after.meta.get("visual_summary") or after.meta.get("scene")

        if vc_type == "shot":
            # Camera-only cut — contributes weak evidence; should not anchor a chunk alone
            signals["visual_change"] = cfg_ref.weight_shot_change
            reasons.append(f"shot change (camera cut only, weight={cfg_ref.weight_shot_change})")
        elif vc_type in ("scene", "slide", "event"):
            # Meaningful visual context change
            signals["visual_change"] = 0.85
            reasons.append(f"visual {vc_type} transition")
        elif before_vis and after_vis and before_vis != after_vis:
            # No explicit type from worker — infer from summary change (scene-level)
            signals["visual_change"] = 0.85
            reasons.append(f"visual scene transition ({before_vis[:30]!r} → {after_vis[:30]!r})")

        if after.kind == "visual_event":
            signals["scene_change"] = 0.95
            reasons.append("visual event boundary")
            is_soft = True

        # 6. OCR signals (slide title or content change)
        before_ocr = before.meta.get("ocr_text")
        after_ocr = after.meta.get("ocr_text")
        if before_ocr and after_ocr and before_ocr != after_ocr:
            signals["ocr_change"] = 0.80
            reasons.append("OCR slide text change")

        # 7. Discourse / Question-Answer transitions
        if before.text.strip().endswith("?") or _QA_RE.search(before.text.strip()):
            signals["qa_transition"] = 0.90
            reasons.append("question-answer transition")

        # 8. Graph separation signal
        signals["graph_separation"] = graph.graph_separation_score(before.id, after.id)

        # 9. Cross-modal agreement: count independent signals above a meaningful
        # threshold and produce a consensus bonus. More signals = higher bonus.
        # pause_strength and semantic_shift are counted separately from their
        # parent categories to reward genuinely independent evidence.
        independent_signals = sum(
            1
            for s in (
                signals.get("speaker_change"),
                signals.get("visual_change") or signals.get("scene_change"),
                signals.get("temporal_gap"),
                signals.get("ocr_change"),
                signals.get("structural_strength"),
                signals.get("semantic_shift"),  # pre-computed by video_boundary
                signals.get("pause_strength"),  # counted separately from temporal_gap
            )
            if s and s > 0.5
        )
        if independent_signals >= 2:
            signals["multimodal_consensus"] = min(1.0, 0.4 + independent_signals * 0.2)
            reasons.append(f"multimodal agreement ({independent_signals} independent signals)")

        cand = BoundaryCandidate(
            index=i,
            position=accumulated_tokens,
            timestamp=after.t0,
            modality=modality,
            left_element_id=before.id,
            right_element_id=after.id,
            signals=signals,
            confidence=1.0 if not vetoes else 0.0,
            source="universal_extractor",
            reason="; ".join(reasons) if reasons else "paragraph boundary",
            is_hard=is_hard,
            is_soft=is_soft,
            veto_reasons=vetoes,
        )
        candidates.append(cand)

    return candidates


# ───────── Boundary Scoring Engine ─────────


def score_candidate(
    candidate: BoundaryCandidate,
    graph: ContextGraph,
    left_ids: set[str],
    right_ids: set[str],
    token_distance: int,
    target_tokens: int,
) -> tuple[float, float, list[str]]:
    """Compute universal score for a boundary candidate.

    Returns (total_score, relationship_loss, severed_reasons).
    """
    cfg = settings()
    signals = candidate.signals

    # 1. Graph relationship loss penalty
    rel_loss, severed_reasons = graph.relationship_loss(left_ids, right_ids)

    # 2. Token target proximity / fragmentation penalty
    diff = abs(token_distance - target_tokens)
    fragmentation_cost = diff / max(1, target_tokens)

    # 3. Weighted multi-modal score
    score = (
        cfg.weight_structure * signals.get("structural_strength", 0.0)
        + cfg.weight_semantic * signals.get("semantic_shift", 0.0)
        + cfg.weight_temporal
        * (
            signals.get("speaker_change", 0.0)
            + signals.get("temporal_gap", 0.0)
            + signals.get(
                "pause_strength", 0.0
            )  # counted separately for consensus but scored together
        )
        + cfg.weight_visual * (signals.get("visual_change", 0.0) + signals.get("scene_change", 0.0))
        + cfg.weight_ocr * signals.get("ocr_change", 0.0)
        + cfg.weight_graph * signals.get("graph_separation", 0.5)
        + cfg.weight_consensus * signals.get("multimodal_consensus", 0.0)
        - cfg.weight_relationship_loss * rel_loss
        - cfg.weight_fragmentation * fragmentation_cost
    )

    # Penalty for splitting inside consecutive list items
    if signals.get("list_continuation"):
        score -= 0.5

    return score, rel_loss, severed_reasons


def _check_caption_pair(a: ContentElement, b: ContentElement, graph: ContextGraph) -> str | None:
    for x, y in ((a, b), (b, a)):
        if (
            x.kind == "caption"
            and y.kind in ("table", "figure")
            and x.id in graph.captions_of(y.id)
        ):
            return f"CAPTIONS {x.id} ↔ {y.id}"
    return None


# ───────── Universal Boundary Optimizer ─────────


def choose_universal_cut(
    region: list[ContentElement],
    graph: ContextGraph,
    target_tokens: int | None = None,
    max_tokens: int | None = None,
) -> UniversalCutResult:
    """Universal boundary decision engine: scores all candidates, enforces hard
    constraints, evaluates relationship loss, and selects the optimal cut.

    Backward-compatible drop-in replacement for choose_cut.
    """
    cfg = settings()
    target = target_tokens or cfg.target_chunk_tokens

    if len(region) < 2:
        return UniversalCutResult(index=None, overflow=True)

    tokens = [count_tokens(e.text) for e in region]
    candidates = generate_candidates_for_region(region, graph)

    if not candidates:
        return UniversalCutResult(index=None, overflow=True)

    valid_scored: list[tuple[float, BoundaryCandidate, list[str]]] = []
    all_vetoes: list[str] = []
    candidate_trace: list[dict[str, Any]] = []

    for cand in candidates:
        idx = cand.index
        if idx is None:
            continue
        left_ids = {e.id for e in region[:idx]}
        right_ids = {e.id for e in region[idx:]}
        toks_before = sum(tokens[:idx])

        # Check hard constraint vetoes
        if cand.veto_reasons:
            all_vetoes.extend(cand.veto_reasons)
            candidate_trace.append(
                {
                    "index": idx,
                    "status": "vetoed",
                    "reasons": cand.veto_reasons,
                    "signals": cand.signals,
                }
            )
            continue

        score, rel_loss, severed = score_candidate(
            cand, graph, left_ids, right_ids, toks_before, target
        )

        # If relationship loss is too severe (e.g. cutting caption from float), veto
        if rel_loss >= 1.0:
            veto_msg = (
                f"cut before {region[idx].id} rejected: high relationship loss ({rel_loss:.2f})"
            )
            all_vetoes.append(veto_msg)
            candidate_trace.append(
                {
                    "index": idx,
                    "status": "vetoed_by_graph_loss",
                    "rel_loss": rel_loss,
                    "severed": severed,
                }
            )
            continue

        valid_scored.append((score, cand, severed))
        candidate_trace.append(
            {
                "index": idx,
                "score": round(score, 4),
                "status": "valid",
                "rel_loss": round(rel_loss, 4),
                "signals": cand.signals,
                "tokens_before": toks_before,
            }
        )

    if not valid_scored:
        return UniversalCutResult(
            index=None,
            vetoes=all_vetoes,
            overflow=True,
            candidate_scores=candidate_trace,
            trace={"total_candidates": len(candidates), "valid": 0},
        )

    # Pick the candidate with the highest universal score
    valid_scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_cand, severed = valid_scored[0]

    return UniversalCutResult(
        index=best_cand.index,
        vetoes=all_vetoes,
        overflow=False,
        chosen_candidate=best_cand,
        candidate_scores=candidate_trace,
        trace={
            "chosen_index": best_cand.index,
            "best_score": round(best_score, 4),
            "signals": best_cand.signals,
            "reason": best_cand.reason,
            "severed_low_priority": severed,
        },
    )
