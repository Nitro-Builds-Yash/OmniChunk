"""Web-layer tests, most of them regressions for real defects.

The upload and artifact routes are where user-controlled strings met the
filesystem, and neither path was covered. Two of these tests fail outright
against the previous implementation: a crafted filename wrote outside the job
directory, and a crafted job id read outside the data directory.
"""

from __future__ import annotations

import json

import pytest

from cleave.web import jobs as jobs_mod
from cleave.web.uploads import MAX_FILES, safe_upload_name

# ───────── filename sanitisation ─────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../../evil.txt", "evil.txt"),
        ("..\\..\\evil.txt", "evil.txt"),
        ("/etc/passwd.txt", "passwd.txt"),
        ("C:\\Windows\\system32\\cmd.txt", "cmd.txt"),
        ("a/b/c.pdf", "c.pdf"),
        ("...", "upload0"),
        ("..", "upload0"),
        ("", "upload0"),
        (None, "upload0"),
        (".env", "env"),
    ],
)
def test_upload_names_are_reduced_to_a_leaf(raw, expected):
    assert safe_upload_name(raw, 0) == expected


def test_upload_name_is_length_bounded():
    assert len(safe_upload_name("x" * 500 + ".pdf", 0)) <= 120


# ───────── upload route ─────────


def _upload(client, filename: str, content: bytes = b"# hello\n\nsome text\n"):
    return client.post(
        "/api/jobs",
        files=[("files", (filename, content, "text/markdown"))],
        data={"use_llm": "false"},
        follow_redirects=False,
    )


def test_traversal_filename_cannot_escape_the_job_directory(client, tmp_data_dir, monkeypatch):
    """The regression: ``../../../pwned.txt`` used to be written verbatim."""
    monkeypatch.setattr("cleave.pipeline.run_job", lambda *a, **k: None)
    outside = tmp_data_dir.parent.parent / "pwned.txt"

    resp = _upload(client, "../../../pwned.txt")

    assert resp.status_code == 303
    assert not outside.exists()
    written = [p.name for p in tmp_data_dir.rglob("*") if p.is_file()]
    assert written == ["pwned.txt"]  # kept, but as a leaf inside the job dir


def test_windows_separators_are_also_stripped(client, tmp_data_dir, monkeypatch):
    monkeypatch.setattr("cleave.pipeline.run_job", lambda *a, **k: None)
    resp = _upload(client, "..\\..\\pwned.txt")
    assert resp.status_code == 303
    for path in tmp_data_dir.rglob("*"):
        assert ".." not in path.name


def test_unsupported_extension_is_rejected(client, monkeypatch):
    monkeypatch.setattr("cleave.pipeline.run_job", lambda *a, **k: None)
    resp = _upload(client, "payload.exe", b"MZ")
    assert resp.status_code == 415


def test_too_many_files_is_rejected_before_anything_is_written(client, tmp_data_dir, monkeypatch):
    monkeypatch.setattr("cleave.pipeline.run_job", lambda *a, **k: None)
    files = [("files", (f"f{i}.txt", b"x", "text/plain")) for i in range(MAX_FILES + 1)]

    resp = client.post("/api/jobs", files=files, data={"use_llm": "false"}, follow_redirects=False)

    assert resp.status_code == 413
    assert not list(tmp_data_dir.rglob("*.txt"))


# ───────── artifact routes ─────────


@pytest.mark.parametrize(
    "job_id",
    [
        "nonexistent",
        "../..",
        "..%2f..",
        "../../../etc",
    ],
)
def test_artifact_routes_reject_unknown_or_traversing_ids(client, job_id):
    """These three routes used to build a path straight from the URL."""
    for artifact in ("units", "graph", "profile"):
        resp = client.get(f"/api/jobs/{job_id}/{artifact}")
        assert resp.status_code == 404, f"{job_id}/{artifact} returned {resp.status_code}"


def test_query_route_rejects_an_unknown_job(client):
    resp = client.post("/jobs/../../x/query", data={"q": "anything"})
    assert resp.status_code in (400, 404)


def test_artifact_route_serves_a_real_job(client, tmp_data_dir):
    job = jobs_mod.Job(id="abc1234567", filename="x.md", status="done", progress=100)
    jobs_mod.JOBS[job.id] = job
    job.dir.mkdir(parents=True, exist_ok=True)
    (job.dir / "units.json").write_text(json.dumps([{"id": "ku_0000"}]))

    resp = client.get(f"/api/jobs/{job.id}/units")

    assert resp.status_code == 200
    assert resp.json() == [{"id": "ku_0000"}]


# ───────── resilience of the read paths ─────────


def test_homepage_survives_a_corrupt_scorecard(client, tmp_path, monkeypatch):
    """A malformed scorecard used to take the whole homepage down."""
    root = tmp_path / "fakeroot"
    (root / "data").mkdir(parents=True)
    (root / "data" / "scorecard.json").write_text("{ this is not json")
    monkeypatch.setattr("cleave.web.routes_pages.ROOT", root)

    assert client.get("/").status_code == 200


def test_results_404_rather_than_500_on_an_unreadable_profile(client, tmp_data_dir):
    job = jobs_mod.Job(id="def1234567", filename="x.md", status="done", progress=100)
    jobs_mod.JOBS[job.id] = job
    job.dir.mkdir(parents=True, exist_ok=True)
    (job.dir / "units.json").write_text("[]")
    (job.dir / "profile.json").write_text("{ broken")

    assert client.get(f"/jobs/{job.id}/results").status_code == 404


def test_results_tolerate_a_profile_without_totals(client, tmp_data_dir):
    """An artifact from an older schema should still render."""
    job = jobs_mod.Job(id="aaa1234567", filename="x.md", status="done", progress=100)
    jobs_mod.JOBS[job.id] = job
    job.dir.mkdir(parents=True, exist_ok=True)
    (job.dir / "units.json").write_text("[]")
    (job.dir / "profile.json").write_text(json.dumps({"title": "old build"}))

    assert client.get(f"/jobs/{job.id}/results").status_code == 200


# ───────── the rest of the surface ─────────


def test_health_reports_the_selected_provider(client):
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["llm"] == "none"  # CLEAVE_LLM=none, set by conftest


def test_usage_endpoint_lists_providers(client):
    body = client.get("/api/usage").json()
    assert {p["name"] for p in body["providers"]} == {"ollama", "gemini"}


def test_favicon_is_served(client):
    resp = client.get("/favicon.svg")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/svg+xml")


def test_unknown_job_page_is_404(client):
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_status_fragment_stops_polling_once_terminal(client, tmp_data_dir):
    """The self-terminating poll: a finished job must not keep asking."""
    running = jobs_mod.Job(id="run1234567", filename="x.md", status="running", progress=40)
    done = jobs_mod.Job(id="don1234567", filename="x.md", status="done", progress=100)
    jobs_mod.JOBS[running.id] = running
    jobs_mod.JOBS[done.id] = done

    # A running job re-fetches itself on a timer; a finished one must not.
    assert "every 700ms" in client.get(f"/jobs/{running.id}/status").text
    assert "every 700ms" not in client.get(f"/jobs/{done.id}/status").text


def test_rehydrate_skips_a_corrupt_profile_without_crashing(tmp_data_dir):
    (tmp_data_dir / "bad0000000").mkdir(parents=True)
    (tmp_data_dir / "bad0000000" / "profile.json").write_text("{ not json")
    (tmp_data_dir / "good000000").mkdir(parents=True)
    (tmp_data_dir / "good000000" / "profile.json").write_text(
        json.dumps({"totals": {"wall_clock_s": 1.5}, "title": "fine"})
    )

    jobs_mod.rehydrate_jobs()

    assert "good000000" in jobs_mod.JOBS
    assert "bad0000000" not in jobs_mod.JOBS


# ───────── image uploads & export formats ─────────


def test_image_uploads_are_accepted(client, tmp_data_dir, monkeypatch):
    monkeypatch.setattr("cleave.pipeline.run_job", lambda *a, **k: None)
    resp = _upload(client, "diagram.png", b"\x89PNG\r\n\x1a\nfake-image-bytes")
    assert resp.status_code == 303
    written = [p.name for p in tmp_data_dir.rglob("*") if p.is_file()]
    assert "diagram.png" in written


def test_export_formats_return_structured_payloads(client, tmp_data_dir):
    job = jobs_mod.Job(id="exp1234567", filename="doc.pdf", status="done", progress=100)
    jobs_mod.JOBS[job.id] = job
    job.dir.mkdir(parents=True, exist_ok=True)
    sample_units = [
        {
            "id": "ku_0001",
            "content": "Sample chunk content",
            "embed_text": "Overview > Section 1\n\nSample chunk content",
            "modality": "document",
            "context": {"heading_path": ["Overview", "Section 1"], "document_title": "Test Doc"},
            "provenance": {"source_uri": "doc.pdf"},
            "decision": {"strategy": "structural", "reason": "section under budget"},
            "knowledge_unit_type": "section",
            "context_completeness": 1.0,
            "relationships": [{"type": "references", "target_id": "ku_0002"}],
            "token_count": 25,
        }
    ]
    (job.dir / "units.json").write_text(json.dumps(sample_units))

    # 1. LangChain export
    lc_resp = client.get(f"/api/jobs/{job.id}/export?format=langchain")
    assert lc_resp.status_code == 200
    lc_data = lc_resp.json()
    assert len(lc_data) == 1
    assert "page_content" in lc_data[0]
    assert lc_data[0]["metadata"]["id"] == "ku_0001"

    # 2. LlamaIndex export
    llama_resp = client.get(f"/api/jobs/{job.id}/export?format=llamaindex")
    assert llama_resp.status_code == 200
    llama_data = llama_resp.json()
    assert len(llama_data) == 1
    assert llama_data[0]["id_"] == "ku_0001"
    assert llama_data[0]["relationships"]["references"] == "ku_0002"

    # 3. Qdrant export
    qdrant_resp = client.get(f"/api/jobs/{job.id}/export?format=qdrant")
    assert qdrant_resp.status_code == 200
    qdrant_data = qdrant_resp.json()
    assert qdrant_data[0]["payload"]["document"] == sample_units[0]["embed_text"]

    # 4. Chroma export
    chroma_resp = client.get(f"/api/jobs/{job.id}/export?format=chroma")
    assert chroma_resp.status_code == 200
    chroma_data = chroma_resp.json()
    assert chroma_data["ids"] == ["ku_0001"]
