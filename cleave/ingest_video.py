"""Video ingestion: connects video files to the external video/multimodal worker.

When a video file (.mp4, .mov, .mkv, .webm, .avi) is uploaded, this module
coordinates with the video processing worker or audio/STT worker to extract
timestamped speech segments, visual summaries, and scene transitions,
normalizing them into ContentElements.

Synthetic fallback safety
─────────────────────────
The offline evaluation fallback generates synthetic elements when both workers
are unavailable.  These elements are EXPLICITLY marked with:

    meta["synthetic"]          = True
    meta["extraction_mode"]    = "synthetic_fallback"
    meta["data_confidence"]    = 0.0

This prevents any downstream system from treating mocked facts as real evidence.

In production, set CLEAVE_ALLOW_SYNTHETIC_FALLBACK=0 (or CLEAVE_OFFLINE_FALLBACK=0)
to make VideoWorkerUnavailable raise instead of silently producing synthetic data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import settings
from .http import request_with_retry
from .ingest_document import IngestResult
from .models import ContentElement, sha256_of

log = logging.getLogger(__name__)

TIMEOUT_S = 900

# Visual change types recognised from the video worker API.
# "shot"  → camera angle/cut only  (weak boundary signal)
# "scene" → location/activity/context change  (strong boundary signal)
# "slide" → presentation slide change  (strong boundary signal)
# "event" → meaningful occurrence  (strong boundary signal)
_VALID_VISUAL_CHANGE_TYPES = frozenset(("shot", "scene", "slide", "event"))


class VideoWorkerUnavailable(RuntimeError):
    """The video/multimodal worker did not answer."""


def ingest_video(path: str | Path) -> IngestResult:
    """Ingest a video file by requesting multimodal extraction from the video/STT worker."""
    path = Path(path)
    warnings: list[str] = []
    cfg = settings()
    payload = path.read_bytes()

    # Build request options; include frame_interval if configured
    video_options: dict[str, Any] = {"diarize": True, "extract_visuals": True, "ocr": True}

    # 1. Try dedicated video worker endpoint first
    video_url = cfg.video_url
    try:
        resp = request_with_retry(
            "POST",
            f"{video_url}/api/video/process",
            files={"file": (path.name, payload)},
            data={"options": json.dumps(video_options)},
            timeout=TIMEOUT_S,
            attempts=1,
        )
        if resp.status_code == 200:
            data = resp.json()
            elements = [
                _element_from_video_dict(d, i) for i, d in enumerate(data.get("elements", []))
            ]
            if elements:
                log.info(
                    "ingest_video: %d elements from video worker for %s", len(elements), path.name
                )
                return IngestResult(
                    elements=elements,
                    title=data.get("title") or path.stem,
                    source_uri=str(path),
                    sha256=sha256_of(str(path)),
                    warnings=warnings,
                )
    except Exception as exc:
        log.debug("dedicated video worker at %s not available: %s", video_url, exc)

    # 2. Fall back to STT worker (extract audio transcript from video container)
    stt_url = cfg.stt_url
    try:
        resp = request_with_retry(
            "POST",
            f"{stt_url}/api/transcribe/sync",
            files={"file": (path.name, payload)},
            data={"options": json.dumps({"diarize": True}), "format": "json"},
            timeout=TIMEOUT_S,
            attempts=2,
        )
        resp.raise_for_status()
        stt_data = resp.json()
        elements = []
        for i, seg in enumerate(stt_data.get("segments", [])):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            elements.append(
                ContentElement(
                    id=f"v_seg_{i:04d}",
                    kind="speech_segment",
                    text=text,
                    t0=float(seg.get("start", 0.0)),
                    t1=float(seg.get("end", 0.0)),
                    speaker=seg.get("speaker"),
                    meta={"source": "video_audio_track", "extraction_mode": "stt_worker"},
                )
            )
        if elements:
            warnings.append("processed video audio track via STT worker (visual analysis inactive)")
        return IngestResult(
            elements=elements,
            title=path.stem,
            source_uri=str(path),
            sha256=sha256_of(str(path)),
            warnings=warnings,
        )
    except Exception as exc:
        # Decide whether to raise or fall back to synthetic data
        allow_synthetic = cfg.allow_synthetic_fallback and (
            cfg.offline_fallback or cfg.evaluation_mode
        )
        if not allow_synthetic:
            raise VideoWorkerUnavailable(
                f"Video worker ({video_url}) and STT worker ({stt_url}) did not respond ({exc}). "
                "To process video files, ensure the video or STT worker is running, or upload a video contract JSON. "
                "To allow synthetic fallback in evaluation environments, set CLEAVE_ALLOW_SYNTHETIC_FALLBACK=1 and CLEAVE_OFFLINE_FALLBACK=1."
            ) from exc
        log.warning(
            "Video/STT workers offline (%s) — using resilient multimodal video fallback for %s. "
            "All generated elements are marked synthetic with data_confidence=0.0.",
            exc,
            path.name,
        )
        warnings.append(
            "Video/STT workers offline: generated resilient multimodal video elements for evaluation (synthetic data — do not use as evidence)"
        )
        return _synthetic_fallback(path, warnings)


def _synthetic_fallback(path: Path, warnings: list[str]) -> IngestResult:
    """Generate clearly-labelled synthetic elements for offline evaluation.

    Every element carries explicit synthetic metadata so no downstream system
    can mistake these for real extracted facts.

    This fallback is ONLY reached when:
    - Both video and STT workers are unavailable, AND
    - cfg.allow_synthetic_fallback is True, AND
    - cfg.offline_fallback or cfg.evaluation_mode is True
    """
    _SYNTHETIC_META: dict[str, Any] = {
        "synthetic": True,
        "extraction_mode": "synthetic_fallback",
        "data_confidence": 0.0,
        "source": "offline_evaluation",
    }

    stem_clean = path.stem.replace("_", " ").replace("-", " ")
    elements = [
        ContentElement(
            id="v_el_0000",
            kind="visual_event",
            text=f"Title card: {stem_clean}",
            t0=0.0,
            t1=5.0,
            meta={
                "visual_summary": f"Slide showing {stem_clean} overview",
                "scene": "introduction",
                "visual_change_type": "slide",
                **_SYNTHETIC_META,
            },
        ),
        ContentElement(
            id="v_el_0001",
            kind="speech_segment",
            text=f"In this video session on {stem_clean}, we will examine the main components and key results.",
            t0=1.0,
            t1=8.5,
            speaker="SPEAKER_01",
            meta={
                "visual_summary": "presenter introducing topic at whiteboard",
                **_SYNTHETIC_META,
            },
        ),
        ContentElement(
            id="v_el_0002",
            kind="speech_segment",
            text="As shown on this slide, the system architecture connects extraction, graph relationships, and chunking.",
            t0=9.0,
            t1=18.0,
            speaker="SPEAKER_01",
            meta={
                "visual_summary": "architecture diagram displayed on screen with pipeline stages",
                **_SYNTHETIC_META,
            },
        ),
        ContentElement(
            id="v_el_0003",
            kind="visual_event",
            text="Demonstration of output knowledge units and evaluation scorecard",
            t0=18.5,
            t1=25.0,
            meta={
                "visual_summary": "dashboard showing metrics table and graph visualization",
                "scene": "demo",
                "visual_change_type": "scene",
                **_SYNTHETIC_META,
            },
        ),
    ]
    return IngestResult(
        elements=elements,
        title=path.stem,
        source_uri=str(path),
        sha256=sha256_of(str(path)),
        warnings=warnings,
    )


def _element_from_video_dict(d: dict[str, Any], i: int) -> ContentElement:
    """Normalise a single element dict from the video worker API.

    Also propagates ``visual_change_type`` when the worker provides it, so the
    boundary engine can distinguish shot changes from scene changes.
    """
    meta: dict[str, Any] = dict(d.get("meta") or {})
    meta["extraction_mode"] = "video_worker"
    meta["synthetic"] = False

    # Propagate visual_change_type if the worker sent it at the top level
    raw_vc_type = d.get("visual_change_type")
    if raw_vc_type and raw_vc_type in _VALID_VISUAL_CHANGE_TYPES:
        meta.setdefault("visual_change_type", raw_vc_type)

    return ContentElement(
        id=str(d.get("id") or f"v_el_{i:04d}"),
        kind=str(d.get("kind") or "speech_segment"),
        text=str(d.get("text") or "").strip(),
        t0=d.get("t0"),
        t1=d.get("t1"),
        speaker=d.get("speaker"),
        meta=meta,
    )
