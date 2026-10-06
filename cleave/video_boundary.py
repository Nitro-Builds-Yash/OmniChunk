"""Video-specific boundary logic for the Universal Boundary Decision Engine.

This module encapsulates video boundary scoring so that boundary_engine.py
stays modality-agnostic and chunkers_multimodal.py stays thin.

Key concepts
____________
score_video_candidates()
    Wraps generate_candidates_for_region() and layers in video-specific signals
    (semantic shift from adjacent speech embeddings, visual change type resolution).

select_event_windows()
    Converts scored BoundaryCandidates into (t0, t1) temporal event windows that
    chunk_multimodal_stream() can iterate over.

fusion_confidence()
    Heuristic measure of how strongly a set of speech segments and visual elements
    belong together: temporal overlap x entity intersection. No LLM required.

propagate_context()
    Carries the last speaker and last named entities forward into the next chunk
    Context.leading so that pronouns like "he" and "it" remain resolvable without
    overwriting the raw content.

_compute_semantic_shifts()
    Lightweight semantic shift detection using the existing MiniLM sentence-transformer
    model from cleave.semantic (cached -- not loaded per call).
"""

from __future__ import annotations

import logging
from typing import Any

from .boundary_engine import generate_candidates_for_region, score_candidate
from .graph import ContextGraph
from .models import BoundaryCandidate, ContentElement, KnowledgeUnit, Modality, count_tokens

log = logging.getLogger(__name__)


# _________________ Semantic shift detection _________________


def _compute_semantic_shifts(elements: list[ContentElement]) -> dict[int, float]:
    """Return a mapping {boundary_index: shift_score} for adjacent speech pairs.

    boundary_index i means the boundary *before* elements[i] (consistent with
    BoundaryCandidate.index convention).  Only speech_segment elements are
    compared; visual events are skipped.  Uses the MiniLM model from
    cleave.semantic -- returns an empty dict gracefully when unavailable.
    """
    if len(elements) < 2:
        return {}
    try:
        from .semantic import embed  # noqa: PLC0415 -- lazy, model is cached there
    except ImportError:
        return {}

    speech_indices = [
        i for i, e in enumerate(elements) if e.kind == "speech_segment" and e.text.strip()
    ]
    if len(speech_indices) < 2:
        return {}

    texts = [elements[i].text for i in speech_indices]
    vecs = embed(texts)
    if vecs is None:
        return {}

    import numpy as np  # noqa: PLC0415

    vecs = np.asarray(vecs)
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)  # cosine similarity (normalized)

    mean = float(sims.mean())
    std = float(sims.std()) if len(sims) > 1 else 0.0
    threshold = mean - std  # same criterion as semantic.py for consistency

    shifts: dict[int, float] = {}
    for pair_idx, sim in enumerate(sims):
        if sim < threshold:
            # boundary between speech_indices[pair_idx] and speech_indices[pair_idx+1]
            right_elem_idx = speech_indices[pair_idx + 1]
            shift_score = min(1.0, (threshold - sim) / max(std, 0.05))
            shifts[right_elem_idx] = shift_score

    return shifts


# _________________ Video candidate scoring _________________


def score_video_candidates(
    elements: list[ContentElement],
    graph: ContextGraph,
) -> list[BoundaryCandidate]:
    """Generate and enrich BoundaryCandidates for a temporal video element stream.

    Calls the universal generate_candidates_for_region() and then enriches each
    candidate with:
    - semantic_shift: pre-computed MiniLM shift score for speech pairs
    - Shot vs scene disambiguation is handled in boundary_engine via meta["visual_change_type"]

    The enriched candidates are ready to pass directly to score_candidate().
    """
    if len(elements) < 2:
        return []

    candidates = generate_candidates_for_region(elements, graph, modality=Modality.VIDEO)

    # Pre-compute semantic shifts once for the whole stream
    semantic_shifts = _compute_semantic_shifts(elements)

    for cand in candidates:
        idx = cand.index
        if idx is None:
            continue

        # Inject pre-computed semantic shift -- boundary_engine score_candidate()
        # already reads signals["semantic_shift"] in the weighted sum.
        if idx in semantic_shifts:
            cand.signals["semantic_shift"] = semantic_shifts[idx]
            if cand.reason:
                cand.reason = cand.reason + "; semantic topic shift detected"
            else:
                cand.reason = "semantic topic shift detected"

        # Recalculate multimodal_consensus after semantic_shift injection
        independent_signals = sum(
            1
            for s in (
                cand.signals.get("speaker_change"),
                cand.signals.get("visual_change") or cand.signals.get("scene_change"),
                cand.signals.get("temporal_gap"),
                cand.signals.get("ocr_change"),
                cand.signals.get("structural_strength"),
                cand.signals.get("semantic_shift"),
                cand.signals.get("pause_strength"),
            )
            if s and s > 0.5
        )
        if independent_signals >= 2:
            cand.signals["multimodal_consensus"] = min(1.0, 0.4 + independent_signals * 0.2)

    return candidates


# _________________ Event window selection _________________


def select_event_windows(
    elements: list[ContentElement],
    graph: ContextGraph,
    target_tokens: int = 500,
) -> list[dict[str, Any]]:
    """Convert scored BoundaryCandidates into event windows with boundary metadata.

    Returns a list of dicts with keys:
        elements            -- list[ContentElement] belonging to this window
        t0, t1              -- float temporal span (seconds)
        start_candidate     -- BoundaryCandidate that opened this window (or None)
        end_candidate       -- BoundaryCandidate that closed this window (or None)
        boundary_metadata   -- dict ready for KnowledgeUnit.boundary_trace

    If no candidates score above threshold the whole stream is returned as one window.
    """
    from .config import settings  # noqa: PLC0415

    cfg = settings()
    threshold = cfg.speaker_boundary_threshold

    if not elements:
        return []

    timed = sorted(
        [e for e in elements if e.t0 is not None],
        key=lambda e: (e.t0 or 0.0, e.t1 or 0.0),
    )
    if not timed:
        return []

    if len(timed) < 2:
        e0 = timed[0]
        return [_window(timed, e0.t0 or 0.0, e0.t1 or 0.0, None, None)]

    candidates = score_video_candidates(timed, graph)

    # Score each candidate and collect cuts above threshold
    tokens = [count_tokens(e.text) for e in timed]
    scored_cuts: list[tuple[int, float, BoundaryCandidate]] = []

    for cand in candidates:
        idx = cand.index
        if idx is None or cand.veto_reasons:
            continue
        left_ids = {e.id for e in timed[:idx]}
        right_ids = {e.id for e in timed[idx:]}
        toks_before = sum(tokens[:idx])
        score, rel_loss, _ = score_candidate(
            cand, graph, left_ids, right_ids, toks_before, target_tokens
        )

        # Normalise score to [0, 1]: positive scores map above 0.5
        normalised = max(0.0, min(1.0, 0.5 + score * 0.15))
        cand.confidence = normalised

        # A boundary is meaningful when:
        # - normalised confidence exceeds threshold, OR
        # - relationship loss is low AND multiple signals agree
        multi_agree = cand.signals.get("multimodal_consensus", 0) > 0.6
        if normalised >= threshold or (rel_loss < 0.5 and multi_agree):
            scored_cuts.append((idx, normalised, cand))

    if not scored_cuts:
        t0 = timed[0].t0 or 0.0
        t1 = max(e.t1 or e.t0 or 0.0 for e in timed)
        return [_window(timed, t0, t1, None, None)]

    scored_cuts.sort(key=lambda x: x[0])

    windows: list[dict[str, Any]] = []
    prev_idx = 0
    prev_cand: BoundaryCandidate | None = None

    for cut_idx, _confidence, cand in scored_cuts:
        segment = timed[prev_idx:cut_idx]
        if not segment:
            prev_cand = cand
            continue
        t0 = segment[0].t0 or 0.0
        t1 = max(e.t1 or e.t0 or 0.0 for e in segment)
        windows.append(_window(segment, t0, t1, prev_cand, cand))
        prev_idx = cut_idx
        prev_cand = cand

    tail = timed[prev_idx:]
    if tail:
        t0 = tail[0].t0 or 0.0
        t1 = max(e.t1 or e.t0 or 0.0 for e in tail)
        windows.append(_window(tail, t0, t1, prev_cand, None))

    return windows


def _window(
    elements: list[ContentElement],
    t0: float,
    t1: float,
    start_cand: BoundaryCandidate | None,
    end_cand: BoundaryCandidate | None,
) -> dict[str, Any]:
    """Build a window dict with boundary metadata for a contiguous element group."""

    def _signals_list(cand: BoundaryCandidate | None) -> list[str]:
        if not cand:
            return []
        return [k for k, v in cand.signals.items() if v and v > 0.3]

    start_conf = round(start_cand.confidence, 4) if start_cand else 0.0
    end_conf = round(end_cand.confidence, 4) if end_cand else 0.0
    all_signals = list(dict.fromkeys(_signals_list(start_cand) + _signals_list(end_cand)))

    return {
        "elements": elements,
        "t0": t0,
        "t1": t1,
        "start_candidate": start_cand,
        "end_candidate": end_cand,
        "boundary_metadata": {
            "start_confidence": start_conf,
            "end_confidence": end_conf,
            "contributing_signals": all_signals,
        },
    }


# _________________ Fusion confidence _________________


def fusion_confidence(
    speech_segs: list[ContentElement],
    visual_els: list[ContentElement],
) -> tuple[float, str]:
    """Heuristic fusion confidence between speech and visual elements.

    Returns (score: float, label: str) where label is one of
    "strong", "medium", or "weak".

    Uses temporal overlap (60%) and entity intersection (40%) -- no LLM.
    """
    if not speech_segs and not visual_els:
        return 0.0, "weak"
    if not speech_segs or not visual_els:
        return 0.3, "weak"

    # Temporal overlap
    s_t0 = min((s.t0 or 0.0) for s in speech_segs)
    s_t1 = max((s.t1 or s.t0 or 0.0) for s in speech_segs)
    v_t0 = min((v.t0 or 0.0) for v in visual_els)
    v_t1 = max((v.t1 or v.t0 or 0.0) for v in visual_els)

    overlap = max(0.0, min(s_t1, v_t1) - max(s_t0, v_t0))
    total_span = max(s_t1, v_t1) - min(s_t0, v_t0)
    temporal_score = overlap / max(total_span, 1e-6)

    # Entity overlap (from meta["entities"])
    speech_ents: set[str] = {
        ent.lower() for s in speech_segs for ent in (s.meta.get("entities") or [])
    }
    visual_ents: set[str] = {
        ent.lower() for v in visual_els for ent in (v.meta.get("entities") or [])
    }
    union = speech_ents | visual_ents
    entity_score = len(speech_ents & visual_ents) / max(len(union), 1) if union else 0.5

    if speech_ents and visual_ents and not (speech_ents & visual_ents):
        score = round(0.4 * temporal_score, 4)
    else:
        score = round(0.5 * temporal_score + 0.5 * entity_score, 4)
    if score > 0.7:
        label = "strong"
    elif score > 0.4:
        label = "medium"
    else:
        label = "weak"
    return score, label


# _________________ Context propagation _________________


def propagate_context(prev_unit: KnowledgeUnit | None) -> str | None:
    """Produce a conservative context hint for the next chunk.

    Carries the last speaker and last few entities forward so that pronouns
    (he, she, it, they) and bare references remain resolvable.

    Rules:
    - Only populated when there IS a previous unit (not for the first chunk).
    - Never modifies content -- result goes into Context.leading.
    - Returns None when there is nothing meaningful to propagate.
    """
    if prev_unit is None:
        return None

    hint_parts: list[str] = []

    # Last speaker
    if prev_unit.temporal and prev_unit.temporal.speaker:
        hint_parts.append(f"Last speaker: {prev_unit.temporal.speaker}")

    # Last named entities (up to 4, most recent)
    ents = (prev_unit.entities or [])[-4:]
    if ents:
        hint_parts.append(f"Entities in prior context: {', '.join(ents)}")

    # Last sentence for continuity
    if prev_unit.content:
        sentences = [s.strip() for s in prev_unit.content.rstrip().split(".") if s.strip()]
        if sentences and len(sentences[-1]) > 10:
            hint_parts.append(f'Preceding: "{sentences[-1]}"')

    return "; ".join(hint_parts) if hint_parts else None
