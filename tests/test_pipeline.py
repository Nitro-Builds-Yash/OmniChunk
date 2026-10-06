"""Audio ingestion errors, and the job-level resilience they motivated.

The behaviour under test: a stopped STT worker used to raise straight out of the
per-file loop and fail the entire batch, so an untranscribable audio file threw
away a PDF that had parsed perfectly well beside it.
"""

from __future__ import annotations

import json

import httpx
import pytest

from cleave import http, pipeline
from cleave.ingest_audio import STTUnavailable, ingest_audio
from cleave.web import jobs as jobs_mod


def _install(monkeypatch, handler):
    mock = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(http, "client", lambda: mock)
    monkeypatch.setattr(http.time, "sleep", lambda _s: None)
    return mock


# ───────── the STT worker is a separate process ─────────


def test_a_stopped_stt_worker_gives_a_readable_error(monkeypatch, tmp_path):
    audio = tmp_path / "clip.m4a"
    audio.write_bytes(b"not really audio")

    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    _install(monkeypatch, handler)

    with pytest.raises(STTUnavailable) as exc:
        ingest_audio(audio)

    message = str(exc.value)
    assert "127.0.0.1:8000" in message  # says where it looked
    assert "start it" in message  # and what to do about it


def test_a_non_json_reply_is_also_reported_as_unavailable(monkeypatch, tmp_path):
    audio = tmp_path / "clip.m4a"
    audio.write_bytes(b"x")
    _install(monkeypatch, lambda r: httpx.Response(200, text="<html>gateway</html>"))

    with pytest.raises(STTUnavailable, match="non-JSON"):
        ingest_audio(audio)


def test_a_worker_error_field_is_surfaced(monkeypatch, tmp_path):
    audio = tmp_path / "clip.m4a"
    audio.write_bytes(b"x")
    _install(monkeypatch, lambda r: httpx.Response(200, json={"error": "model missing"}))

    with pytest.raises(RuntimeError, match="model missing"):
        ingest_audio(audio)


def test_a_successful_transcript_becomes_elements(monkeypatch, tmp_path):
    audio = tmp_path / "clip.m4a"
    audio.write_bytes(b"x")
    _install(
        monkeypatch,
        lambda r: httpx.Response(
            200,
            json={
                "result": {
                    "segments": [
                        {"text": "Hello there.", "start": 0.0, "end": 2.0, "speaker": "A"},
                        {"text": "Hi back.", "start": 2.0, "end": 4.0, "speaker": "B"},
                    ]
                }
            },
        ),
    )

    result = ingest_audio(audio)

    assert [e.speaker for e in result.elements] == ["A", "B"]
    assert all(e.kind == "speech_segment" for e in result.elements)


# ───────── one bad file must not discard the rest ─────────


def test_one_failing_file_does_not_discard_the_others(monkeypatch, tmp_data_dir):
    """The regression: a stopped STT worker used to fail the whole job."""
    job = jobs_mod.Job(id="mix1234567", filename="two files", use_llm=False)
    jobs_mod.JOBS[job.id] = job
    good, bad = tmp_data_dir / "good.md", tmp_data_dir / "bad.m4a"
    good.write_text("# Title\n\nSome prose.\n")
    bad.write_bytes(b"x")

    real = pipeline._process_file

    def flaky(job_, path, **kw):
        if path.suffix == ".m4a":
            raise STTUnavailable("STT worker at http://127.0.0.1:8000 did not respond")
        return real(job_, path, **kw)

    monkeypatch.setattr(pipeline, "_process_file", flaky)
    pipeline.run_job(job.id, [good, bad])

    assert job.status == "done", f"job failed instead of degrading: {job.error}"
    record = json.loads((job.dir / "profile.json").read_text())
    assert record["totals"]["units"] > 0  # the markdown survived
    warnings = " ".join(record["totals"]["warnings"])
    assert "bad.m4a" in warnings and "STT worker" in warnings  # and the failure is reported


def test_a_job_where_everything_fails_is_still_an_error(monkeypatch, tmp_data_dir):
    job = jobs_mod.Job(id="bad1234567", filename="one file", use_llm=False)
    jobs_mod.JOBS[job.id] = job
    only = tmp_data_dir / "only.m4a"
    only.write_bytes(b"x")

    monkeypatch.setattr(
        pipeline, "_process_file", lambda *a, **k: (_ for _ in ()).throw(STTUnavailable("down"))
    )
    pipeline.run_job(job.id, [only])

    assert job.status == "error"
    assert "every input failed" in (job.error or "")


# ───────── offline fallback mode ─────────


def test_offline_fallback_allows_audio_ingestion_when_stt_is_down(monkeypatch, tmp_path):
    monkeypatch.setenv("CLEAVE_OFFLINE_FALLBACK", "1")
    from cleave import config

    config.reload()
    audio = tmp_path / "meeting_notes.m4a"
    audio.write_bytes(b"audio-bytes")
    _install(monkeypatch, lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused", request=r)))

    res = ingest_audio(audio)
    assert len(res.elements) > 0
    assert any("SPEAKER" in str(e.speaker) for e in res.elements)


def test_offline_fallback_allows_video_ingestion_when_workers_are_down(monkeypatch, tmp_path):
    monkeypatch.setenv("CLEAVE_OFFLINE_FALLBACK", "1")
    from cleave import config

    config.reload()
    from cleave.ingest_video import ingest_video

    video = tmp_path / "IMG_8966 - Trim.mp4"
    video.write_bytes(b"video-bytes")
    _install(monkeypatch, lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused", request=r)))

    res = ingest_video(video)
    assert len(res.elements) > 0
    assert any(e.kind == "visual_event" for e in res.elements)
    assert any(e.kind == "speech_segment" for e in res.elements)
    assert "IMG_8966 - Trim" in res.title
