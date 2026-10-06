"""Multimodal Late-Fusion Chunker: Universal Boundary Decision approach.

Architecture (new)
──────────────────
Instead of anchoring chunks on visual events, every modality proposes
BoundaryCandidates. The Universal Boundary Decision Engine scores them,
cross-modal agreement boosts confidence, and the winning boundaries define
event windows.

Flow:
    timed_elements (speech + visual, sorted by t0)
        ↓
    video_boundary.select_event_windows()
        ↓  scores all candidates via generate_candidates_for_region()
        ↓  adds MiniLM semantic shift for speech pairs
        ↓  distinguishes shot vs scene via meta["visual_change_type"]
        ↓  selects cuts above speaker_boundary_threshold
        ↓
    event windows (each: list[ContentElement], t0, t1, boundary_metadata)
        ↓
    for each window:
        fusion_confidence()  (temporal overlap + entity intersection, no LLM)
        propagate_context()  (carry last speaker/entities to resolve pronouns)
        ↓
    KnowledgeUnit(type=VIDEO_EVENT, boundary_trace={start_conf, end_conf, signals})

Backward compatibility
──────────────────────
- chunk_multimodal_stream() signature is unchanged.
- Audio-only fallback (no visual events) still routes to _temporal_units().
- KnowledgeUnitType.MULTIMODAL_EVENT is preserved; new units use VIDEO_EVENT.
- All existing metadata keys are preserved or extended.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .conversational import classify_conversational_elements
from .graph import ContextGraph
from .models import (
    ChunkingDecision,
    ContentElement,
    Context,
    KnowledgeUnit,
    KnowledgeUnitType,
    Modality,
    Provenance,
    Temporal,
    count_tokens,
)
from .video_boundary import fusion_confidence, propagate_context, select_event_windows

log = logging.getLogger(__name__)


def chunk_multimodal_stream(
    stream: list[ContentElement],
    graph: ContextGraph,
    new_unit_id: Callable[[], str],
    base_provenance: Callable[[ContentElement], Provenance],
    title: str | None,
) -> list[tuple[KnowledgeUnit, list[str]]]:
    """Universal-boundary multimodal chunker.

    Aligns visual events and speech segments on a shared timeline using the
    Universal Boundary Decision Engine. Produces event-level KnowledgeUnits
    with boundary metadata explaining which signals contributed to each cut.

    Falls back to _temporal_units() when no visual elements are present.
    """
    timed_elements = [e for e in stream if e.t0 is not None]
    if not timed_elements:
        return []

    # Separate visual from speech for fusion analysis; route audio-only streams
    timed_elements.sort(key=lambda e: (e.t0 or 0.0, e.t1 or 0.0))
    has_visuals = any(e.kind in ("visual_event", "figure", "slide") for e in timed_elements)

    if not has_visuals:
        # Audio-only: delegate to the temporal chunker (speaker turn boundaries)
        from .chunkers import _temporal_units  # noqa: PLC0415

        return _temporal_units(stream, graph, new_unit_id, base_provenance, title)

    # ── Universal Boundary Decision ──────────────────────────────────────────
    # select_event_windows() calls generate_candidates_for_region() internally,
    # layers semantic shift, shot/scene discrimination, and cross-modal consensus,
    # then returns partitioned temporal windows with boundary metadata.
    event_windows = select_event_windows(timed_elements, graph)

    units: list[tuple[KnowledgeUnit, list[str]]] = []
    prev_unit: KnowledgeUnit | None = None

    for win in event_windows:
        win_elements: list[ContentElement] = win["elements"]
        t0: float = win["t0"]
        t1: float = win["t1"]
        boundary_meta: dict = win["boundary_metadata"]

        if not win_elements:
            continue

        # Partition window into speech vs visual for fusion analysis
        speech_segs = [e for e in win_elements if e.kind == "speech_segment"]
        visual_els = [e for e in win_elements if e.kind in ("visual_event", "figure", "slide")]

        # Fusion confidence: how strongly does speech belong with this visual?
        fuse_score, fuse_label = fusion_confidence(speech_segs, visual_els)

        # Context propagation: carry last speaker + entities for pronoun resolution
        context_hint = propagate_context(prev_unit)

        # Conversational classification (action item, decision, Q&A, etc.)
        speech_for_conv = speech_segs or win_elements
        ku_type, conv_meta = classify_conversational_elements(speech_for_conv)
        # For multimodal windows with visual content, prefer VIDEO_EVENT unless
        # conversational classification found a more specific type
        if visual_els and ku_type in (
            KnowledgeUnitType.NARRATIVE.value,
            KnowledgeUnitType.SPEAKER_TURN.value,
            KnowledgeUnitType.DISCUSSION.value,
            KnowledgeUnitType.GENERIC.value,
        ):
            ku_type = KnowledgeUnitType.VIDEO_EVENT.value

        # Build the unified content string
        content = _build_content(t0, t1, visual_els, speech_segs, fuse_label)

        # All speakers present in the window (attribution preserved on elements)
        speakers = list(dict.fromkeys(s.speaker for s in speech_segs if s.speaker))
        speaker_str = ", ".join(speakers) if speakers else None

        # Entities from visual meta
        visual_entities: list[str] = [
            ent for vis in visual_els for ent in (vis.meta.get("entities") or [])
        ]

        # Check if any elements are synthetic (evaluation fallback data)
        is_synthetic = any(e.meta.get("synthetic") for e in win_elements)
        extraction_modes = list(
            dict.fromkeys(
                e.meta.get("extraction_mode", "unknown")
                for e in win_elements
                if e.meta.get("extraction_mode")
            )
        )

        # Assemble metadata — preserve all existing keys, extend with new ones
        metadata: dict = {
            "has_visual": bool(visual_els),
            "has_audio": bool(speech_segs),
            "granularity": "adaptive",
            "entities": visual_entities,
            "fusion_confidence": fuse_score,
            "fusion_label": fuse_label,
            "speaker_count": len(speakers),
            "size_reason": ["universal_boundary_engine", "cross_modal_agreement"],
            **conv_meta,
        }
        if is_synthetic:
            metadata["synthetic"] = True
            metadata["extraction_mode"] = "synthetic_fallback"
            metadata["data_confidence"] = 0.0
        elif extraction_modes:
            metadata["extraction_mode"] = (
                extraction_modes[0] if len(extraction_modes) == 1 else extraction_modes
            )

        # Decide reason string from boundary signals
        signals_str = (
            ", ".join(boundary_meta.get("contributing_signals", [])) or "temporal continuity"
        )
        reason = (
            f"Universal boundary event [{t0:.1f}s\u2013{t1:.1f}s]: "
            f"{len(speech_segs)} speech + {len(visual_els)} visual elements fused "
            f"({fuse_label} confidence, signals: {signals_str})"
        )

        anchor = win_elements[0]
        unit = KnowledgeUnit(
            id=new_unit_id(),
            content=content,
            modality=Modality.VIDEO,
            context=Context(
                document_title=title,
                leading=context_hint,  # context propagation for pronoun resolution
            ),
            provenance=base_provenance(anchor),
            decision=ChunkingDecision(
                strategy="universal_boundary",
                reason=reason,
                signals={
                    "fusion_confidence": fuse_score,
                    "start_boundary_confidence": boundary_meta.get("start_confidence", 0.0),
                    "end_boundary_confidence": boundary_meta.get("end_confidence", 0.0),
                    "speaker_count": float(len(speakers)),
                    "visual_element_count": float(len(visual_els)),
                    "speech_element_count": float(len(speech_segs)),
                },
                escalation_flags=[],
            ),
            temporal=Temporal(start_s=t0, end_s=t1, speaker=speaker_str),
            entities=visual_entities,
            metadata=metadata,
            token_count=count_tokens(content),
            knowledge_unit_type=ku_type,
            # boundary_trace stores the full evidence record for debugging & RAG
            boundary_trace={
                "start_confidence": boundary_meta.get("start_confidence", 0.0),
                "end_confidence": boundary_meta.get("end_confidence", 0.0),
                "contributing_signals": boundary_meta.get("contributing_signals", []),
                "fusion_score": fuse_score,
                "fusion_label": fuse_label,
                "speaker_attribution": {s.id: s.speaker for s in speech_segs if s.speaker},
            },
        )

        units.append((unit, [e.id for e in win_elements]))
        prev_unit = unit

    return units


# ─────────────────────────────────────────────────────────────────────────────


def _build_content(
    t0: float,
    t1: float,
    visual_els: list[ContentElement],
    speech_segs: list[ContentElement],
    fuse_label: str,
) -> str:
    """Build the unified content string for a multimodal event window.

    Format preserves speaker attribution on each speech segment so that
    the content is self-explanatory without requiring the temporal metadata.
    Visual and speech are clearly labelled to enable clean RAG display.
    """
    parts: list[str] = []

    # Visual section
    if visual_els:
        vis_descriptions = " | ".join(v.text for v in visual_els if v.text)
        if vis_descriptions:
            parts.append(f"Visual [{t0:.1f}s\u2013{t1:.1f}s]: {vis_descriptions}")

    # Speech section with per-speaker attribution
    if speech_segs:
        dialogue_lines: list[str] = []
        for s in speech_segs:
            if s.speaker:
                dialogue_lines.append(f"[{s.speaker}] {s.text}")
            else:
                dialogue_lines.append(s.text)
        spoken = " ".join(dialogue_lines)
        parts.append(f"Spoken: {spoken}")

    # When fusion is weak, add an explicit note so RAG consumers can weight accordingly
    if fuse_label == "weak" and visual_els and speech_segs:
        parts.append("[Note: speech and visual have low semantic overlap in this window]")

    return "\n\n".join(parts) if parts else f"[empty window {t0:.1f}s\u2013{t1:.1f}s]"
