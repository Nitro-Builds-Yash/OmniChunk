"""Vision Worker: Multimodal Image and Video understanding using Gemini 1.5/2.0
extracts scene transitions, OCR, objects, speaker events, and temporal alignment
into universal ContentElements.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
from pathlib import Path

from ..config import settings
from ..http import request_with_retry
from ..ingest_document import IngestResult
from ..models import ContentElement, sha256_of

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".svg", ".bmp", ".gif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

IMAGE_ANALYSIS_PROMPT = """Analyze this image in detail for knowledge extraction and universal chunking.
Extract:
1. "title": Short descriptive title.
2. "ocr_text": Any visible text, headlines, labels, or captions.
3. "visual_description": Detailed description of objects, diagrams, charts, spatial relationships, or scene content.
4. "entities": List of key entities, products, people, or concepts present.
5. "elements": List of granular elements with:
   - "kind": ("figure" | "caption" | "paragraph" | "heading" | "table")
   - "text": element content
   - "bbox": [ymin, xmin, ymax, xmax] normalized (0 to 1) or empty
"""

VIDEO_ANALYSIS_PROMPT = """Analyze this video for multimodal chunking and knowledge extraction.
Provide a chronological timeline of scenes, speech, visual events, and on-screen text.
Return a JSON object with:
1. "title": Overall video title or topic.
2. "scenes": Array of timeline events, each containing:
   - "t0": Start time in seconds (float)
   - "t1": End time in seconds (float)
   - "speaker": Speaker name or ID if spoken, else null
   - "speech_text": Transcribed speech if any
   - "visual_event": What is occurring visually on screen (actions, slides, demonstrations)
   - "ocr_text": On-screen text, titles, or slide bullet points
   - "entities": Extracted concepts or entities
"""


def _get_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".mp4":
        return "video/mp4"
    return "application/octet-stream"


def process_image_file(path: str | Path) -> IngestResult:
    """Analyze an image using Gemini Vision to extract text, layout, objects, and relationships."""
    path = Path(path)
    cfg = settings()
    warnings: list[str] = []

    if not cfg.gemini_api_key:
        # Graceful fallback when Gemini key is not present
        warnings.append("Gemini API key missing; image processed with basic metadata placeholder.")
        elem = ContentElement(
            id="img_0000",
            kind="figure",
            text=f"Image file: {path.name}",
            meta={"filename": path.name, "size_bytes": path.stat().st_size},
        )
        return IngestResult(
            elements=[elem],
            title=path.stem,
            source_uri=str(path),
            sha256=sha256_of(str(path)),
            warnings=warnings,
        )

    try:
        raw_bytes = path.read_bytes()
        b64_data = base64.b64encode(raw_bytes).decode("utf-8")
        mime = _get_mime_type(path)

        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": IMAGE_ANALYSIS_PROMPT},
                        {"inlineData": {"mimeType": mime, "data": b64_data}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }

        r = request_with_retry(
            "POST",
            f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.gemini_model}:generateContent",
            headers={"x-goog-api-key": cfg.gemini_api_key},
            json=body,
            timeout=cfg.llm_timeout_s,
            attempts=2,
        )
        r.raise_for_status()
        data = r.json()
        raw_text = "".join(
            p.get("text", "") for p in data["candidates"][0]["content"]["parts"]
        ).strip()
        parsed = json.loads(raw_text)

        elements: list[ContentElement] = []
        raw_elements = parsed.get("elements", [])

        if raw_elements:
            for idx, el in enumerate(raw_elements):
                elements.append(
                    ContentElement(
                        id=f"img_el_{idx:04d}",
                        kind=el.get("kind", "figure"),
                        text=el.get("text", ""),
                        bbox=tuple(el["bbox"]) if el.get("bbox") and len(el["bbox"]) == 4 else None,
                        meta={"source": "gemini_vision", "entities": parsed.get("entities", [])},
                    )
                )
        else:
            # Fallback to general description + OCR
            desc = parsed.get("visual_description", "")
            ocr = parsed.get("ocr_text", "")
            if desc:
                elements.append(
                    ContentElement(
                        id="img_el_0000",
                        kind="figure",
                        text=desc,
                        meta={"source": "gemini_vision_description"},
                    )
                )
            if ocr:
                elements.append(
                    ContentElement(
                        id="img_el_0001",
                        kind="caption",
                        text=ocr,
                        meta={"source": "gemini_vision_ocr"},
                    )
                )

        return IngestResult(
            elements=elements or [ContentElement(id="img_0000", kind="figure", text=path.stem)],
            title=parsed.get("title") or path.stem,
            source_uri=str(path),
            sha256=sha256_of(str(path)),
            warnings=warnings,
        )
    except Exception as exc:
        log.warning("Image vision processing failed (%s); returning fallback placeholder", exc)
        warnings.append(f"Image analysis error: {exc}")
        return IngestResult(
            elements=[ContentElement(id="img_0000", kind="figure", text=f"Image: {path.name}")],
            title=path.stem,
            source_uri=str(path),
            sha256=sha256_of(str(path)),
            warnings=warnings,
        )


def process_video_file(path: str | Path) -> IngestResult:
    """Analyze a video using Gemini Multimodal to extract temporal timeline,
    speech, visual events, and on-screen slide text."""
    path = Path(path)
    cfg = settings()
    warnings: list[str] = []

    # First check if external video service or Gemini is reachable
    if not cfg.gemini_api_key:
        warnings.append("Gemini API key missing; falling back to standard video ingestion.")
        from ..ingest_video import ingest_video

        return ingest_video(path)

    try:
        raw_bytes = path.read_bytes()
        # Cap direct payload size to 20MB for inline; larger files can use standard audio/video worker
        if len(raw_bytes) > 20 * 1024 * 1024:
            log.info("Video file %s > 20MB; routing to dedicated video pipeline", path.name)
            from ..ingest_video import ingest_video

            return ingest_video(path)

        b64_data = base64.b64encode(raw_bytes).decode("utf-8")
        mime = _get_mime_type(path)

        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": VIDEO_ANALYSIS_PROMPT},
                        {"inlineData": {"mimeType": mime, "data": b64_data}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }

        r = request_with_retry(
            "POST",
            f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.gemini_model}:generateContent",
            headers={"x-goog-api-key": cfg.gemini_api_key},
            json=body,
            timeout=90.0,
            attempts=2,
        )
        r.raise_for_status()
        data = r.json()
        raw_text = "".join(
            p.get("text", "") for p in data["candidates"][0]["content"]["parts"]
        ).strip()
        parsed = json.loads(raw_text)

        elements: list[ContentElement] = []
        scenes = parsed.get("scenes", [])

        for idx, sc in enumerate(scenes):
            t0 = float(sc.get("t0", 0.0))
            t1 = float(sc.get("t1", t0 + 5.0))
            speaker = sc.get("speaker")
            speech = sc.get("speech_text", "").strip()
            visual = sc.get("visual_event", "").strip()
            ocr = sc.get("ocr_text", "").strip()

            # 1. Speech element if spoken
            if speech:
                elements.append(
                    ContentElement(
                        id=f"v_speech_{idx:04d}",
                        kind="speech_segment",
                        text=speech,
                        t0=t0,
                        t1=t1,
                        speaker=speaker,
                        meta={"scene_index": idx, "ocr": ocr},
                    )
                )

            # 2. Visual scene element
            if visual or ocr:
                full_visual_text = f"[Scene {idx + 1}] {visual}"
                if ocr:
                    full_visual_text += f" | On-screen Text: {ocr}"
                elements.append(
                    ContentElement(
                        id=f"v_visual_{idx:04d}",
                        kind="visual_event",
                        text=full_visual_text,
                        t0=t0,
                        t1=t1,
                        meta={"scene_index": idx, "entities": sc.get("entities", [])},
                    )
                )

        if elements:
            log.info(
                "Gemini multimodal video extracted %d elements from %s", len(elements), path.name
            )
            return IngestResult(
                elements=elements,
                title=parsed.get("title") or path.stem,
                source_uri=str(path),
                sha256=sha256_of(str(path)),
                warnings=warnings,
            )
    except Exception as exc:
        log.warning(
            "Gemini multimodal video processing failed (%s); falling back to STT worker", exc
        )
        warnings.append(f"Gemini video analysis failed: {exc}")

    # Fallback to standard video ingestion
    from ..ingest_video import ingest_video

    res = ingest_video(path)
    res.warnings.extend(warnings)
    return res
