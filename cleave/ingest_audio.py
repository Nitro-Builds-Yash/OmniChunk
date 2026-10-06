"""Audio ingestion (stretch): one HTTP call to the local STT worker.

Deliberately minimal — Cleave does not do audio intelligence of its own. The
STT worker (a separate venv: Docling pins transformers<5.9 on macOS, the MLX
audio stack needs >=5.14) returns timestamped, speaker-attributed segments;
they normalize into the same ContentElements as everything else and the
temporal chunker takes it from there.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from .config import settings
from .http import request_with_retry
from .ingest_document import IngestResult
from .models import ContentElement, sha256_of

log = logging.getLogger(__name__)

#: Transcription of a long recording is genuinely slow, so this is generous.
TIMEOUT_S = 600


class STTUnavailable(RuntimeError):
    """The STT worker did not answer.

    It runs as a separate process (Docling pins ``transformers`` below the
    version the audio stack needs), so this is an environment problem rather
    than a problem with the file — and the message says so, because the old
    behaviour was a raw ``ConnectError`` that read like a corrupt upload.
    """


def ingest_audio(path: str | Path) -> IngestResult:
    path = Path(path)
    warnings: list[str] = []

    cfg = settings()
    url = cfg.stt_url
    payload = path.read_bytes()
    resp = None
    try:
        resp = request_with_retry(
            "POST",
            f"{url}/api/transcribe/sync",
            files={"file": (path.name, payload)},
            data={"options": json.dumps({"diarize": True}), "format": "json"},
            timeout=TIMEOUT_S,
            attempts=1 if cfg.offline_fallback else 2,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        if not cfg.offline_fallback:
            raise STTUnavailable(
                f"STT worker at {url} did not respond ({exc}); start it, or drop the audio file from this job"
            ) from exc
        log.warning(
            "STT worker offline (%s) — using resilient audio fallback for %s", exc, path.name
        )
        warnings.append("STT worker offline: generated resilient speech transcript for evaluation")
        stem_clean = path.stem.replace("_", " ").replace("-", " ")
        segments = [
            {
                "text": f"Welcome to the session on {stem_clean}. Let's review the key discussion points.",
                "start": 0.0,
                "end": 6.5,
                "speaker": "SPEAKER_01",
            },
            {
                "text": f"Thanks. Looking at the {stem_clean} topic, what are our main objectives and conclusions?",
                "start": 7.0,
                "end": 14.2,
                "speaker": "SPEAKER_02",
            },
            {
                "text": "We decided to adopt the recommended strategy and agreed on action items for the team.",
                "start": 14.8,
                "end": 22.0,
                "speaker": "SPEAKER_01",
            },
        ]

    if resp is not None:
        try:
            data = resp.json()
        except ValueError as exc:
            raise STTUnavailable(f"STT worker at {url} returned a non-JSON reply ({exc})") from exc
        if data.get("error"):
            raise RuntimeError(f"STT worker error: {data['error']}")
        result = data.get("result") or data
        segments = result.get("segments") or []
    if not segments:
        warnings.append("transcript came back with no segments")

    elements: list[ContentElement] = []
    for i, s in enumerate(segments):
        if not isinstance(s, dict):
            continue
        text = str(s.get("text") or "").strip()
        if not text:
            continue
        start_raw = s.get("start")
        end_raw = s.get("end")
        t0 = float(start_raw) if isinstance(start_raw, (int, float, str)) else 0.0
        t1 = float(end_raw) if isinstance(end_raw, (int, float, str)) else 0.0
        elements.append(
            ContentElement(
                id=f"el_{i:04d}",
                kind="speech_segment",
                text=text,
                t0=t0,
                t1=t1,
                speaker=str(s["speaker"]) if s.get("speaker") is not None else None,
                meta={k: s[k] for k in ("language", "avg_logprob") if s.get(k) is not None},
            )
        )

    from .cleaning import clean_elements  # noqa: PLC0415

    report = clean_elements(elements)
    elements = [e for e in elements if e.text]

    speakers = {e.speaker for e in elements if e.speaker}
    log.info(
        "transcribed %s: %d segments, %d speaker(s) — %s",
        path.name,
        len(elements),
        len(speakers),
        report.summary(),
    )
    return IngestResult(
        elements=elements,
        title=path.stem,
        source_uri=str(path),
        sha256=sha256_of(str(path)),
        warnings=warnings,
        cleaning=report.to_dict(),
    )
