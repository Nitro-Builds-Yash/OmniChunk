"""Security, persistence, offline operation, hybrid retrieval, library and CLI."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from cleave import config
from cleave import semantic as semantic_mod
from cleave.web import jobs as jobs_mod
from cleave.web import search as search_mod
from cleave.web import security as security_mod


@pytest.fixture(autouse=True)
def _no_models(monkeypatch):
    """Keep these tests offline and fast: no embedding or re-ranking model."""
    monkeypatch.setattr(semantic_mod, "embed", lambda texts: None)
    monkeypatch.setattr(semantic_mod, "rerank", lambda q, texts: None)


# ───────── tokenizer fallback ─────────


def test_approx_encoder_is_lossless_and_counts():
    from cleave.models import _ApproxEncoder

    enc = _ApproxEncoder()
    text = "  Revenue grew 12%\nin Q3 ,  say  analysts. "
    toks = enc.encode(text)
    assert enc.decode(toks) == text
    assert len(toks) >= 6


def test_encoder_falls_back_when_tiktoken_cannot_load(monkeypatch):
    import sys
    import types

    from cleave import models

    broken = types.ModuleType("tiktoken")

    def boom(_name):
        raise OSError("network blocked")

    broken.get_encoding = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tiktoken", broken)
    models._encoder.cache_clear()
    models._count_cached.cache_clear()
    try:
        assert models.count_tokens("one two three") == 3
    finally:
        models._encoder.cache_clear()
        models._count_cached.cache_clear()


# ───────── auth & rate limiting ─────────


def test_no_key_configured_keeps_local_access_open(client):
    assert client.get("/api/usage").status_code == 200


def test_api_key_is_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("CLEAVE_API_KEY", "s3cret")
    config.reload()
    assert client.get("/api/usage").status_code == 401
    assert client.get("/api/usage", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/usage", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/api/usage", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    basic = base64.b64encode(b"anyone:s3cret").decode()
    assert client.get("/", headers={"Authorization": f"Basic {basic}"}).status_code == 200
    r = client.get("/")
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Basic")


def test_health_and_static_stay_public(client, monkeypatch):
    monkeypatch.setenv("CLEAVE_API_KEY", "s3cret")
    config.reload()
    assert client.get("/health/live").status_code == 200
    assert client.get("/static/favicon.svg").status_code == 200


def test_job_submissions_are_rate_limited(client, monkeypatch, fixtures):
    monkeypatch.setenv("CLEAVE_RATE_LIMIT_PER_MIN", "2")
    config.reload()
    security_mod.submissions.clear()
    monkeypatch.setattr(jobs_mod, "submit", lambda *a, **k: None)
    data = (fixtures / "flat_essay.md").read_bytes()
    codes = [
        client.post(
            "/api/jobs",
            files={"files": ("a.md", data, "text/markdown")},
            follow_redirects=False,
        ).status_code
        for _ in range(3)
    ]
    assert codes == [303, 303, 429]


# ───────── persistence, concurrency, retry ─────────


def test_job_state_survives_a_restart(tmp_data_dir):
    job = jobs_mod.Job(id="abc123", filename="a.md", filenames=["a.md"], status="running")
    jobs_mod.JOBS[job.id] = job
    jobs_mod.save(job)

    jobs_mod.JOBS.clear()  # the process "restarts"
    jobs_mod.rehydrate_jobs()

    back = jobs_mod.JOBS["abc123"]
    assert back.status == "error"
    assert back.message == "interrupted"
    assert "retry" in (back.error or "")


def test_retry_reruns_a_failed_job_from_stored_inputs(client, tmp_data_dir, monkeypatch):
    job = jobs_mod.Job(id="deadbeef00", filename="a.md", filenames=["a.md"], status="error")
    (job.dir / "input").mkdir(parents=True)
    (job.dir / "input" / "a.md").write_text("# Title\n\nbody")
    jobs_mod.JOBS[job.id] = job

    ran: list[tuple[str, list[Path]]] = []
    monkeypatch.setattr("cleave.pipeline.run_job", lambda jid, paths: ran.append((jid, paths)))
    r = client.post(f"/api/jobs/{job.id}/retry")
    assert r.status_code == 200
    assert ran
    assert ran[0][0] == job.id
    assert [p.name for p in ran[0][1]] == ["a.md"]


def test_retry_refuses_a_job_that_did_not_fail(client, tmp_data_dir):
    job = jobs_mod.Job(id="donejob000", filename="a.md", status="done")
    jobs_mod.JOBS[job.id] = job
    assert client.post(f"/api/jobs/{job.id}/retry").status_code == 409


def test_submit_respects_the_concurrency_limit(tmp_data_dir, monkeypatch):
    import threading
    import time

    monkeypatch.setenv("CLEAVE_MAX_CONCURRENT_JOBS", "1")
    config.reload()
    running, peak, lock = 0, 0, threading.Lock()

    def fake_run(_jid, _paths):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1

    threads = []
    for i in range(3):
        job = jobs_mod.Job(id=f"j{i}", filename="x")
        jobs_mod.JOBS[job.id] = job
        t = threading.Thread(target=jobs_mod.submit, args=(job, fake_run, []))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    assert peak == 1


# ───────── hybrid search ─────────


def _write_units(job_id: str, units: list[dict]) -> None:
    job = jobs_mod.Job(id=job_id, filename="t", status="done")
    job.dir.mkdir(parents=True, exist_ok=True)
    (job.dir / "units.json").write_text(json.dumps(units))
    jobs_mod.JOBS[job_id] = job


UNITS = [
    {
        "id": "schema",
        "content": "Schema: order_id, revenue_usd, region",
        "embed_text": "sales.csv schema card: order_id text, revenue_usd number",
        "relationships": [{"type": "schema_of", "target_id": "rows1"}],
        "context": {"heading_path": ["sales.csv"]},
    },
    {
        "id": "rows1",
        "content": "| order_id | revenue_usd |\n| SO-10042 | 991.5 |",
        "embed_text": "| order_id | revenue_usd |\n| SO-10042 | 991.5 |",
        "relationships": [
            {"type": "has_schema", "target_id": "schema"},
            {"type": "next", "target_id": "rows2"},
        ],
        "context": {"heading_path": ["sales.csv"]},
    },
    {
        "id": "rows2",
        "content": "| order_id | revenue_usd |\n| SO-10077 | 12.0 |",
        "embed_text": "| order_id | revenue_usd |\n| SO-10077 | 12.0 |",
        "relationships": [],
        "context": {"heading_path": ["sales.csv"]},
    },
]


def test_bm25_finds_exact_identifiers_without_any_embedding_model(tmp_data_dir):
    _write_units("j1", UNITS)
    hits = search_mod.search_jobs(["j1"], "revenue for SO-10042")
    assert hits[0].unit["id"] == "rows1"
    assert "bm25" in hits[0].signals


def test_hits_carry_graph_context_but_not_reading_order(tmp_data_dir):
    _write_units("j1", UNITS)
    top = search_mod.search_jobs(["j1"], "SO-10042")[0]
    related = {r["id"]: r["_relation"] for r in top.related}
    assert related == {"schema": "has_schema"}  # `next` is not followed


def test_search_spans_jobs_and_api_returns_json(client, tmp_data_dir):
    _write_units("j1", UNITS[:2])
    _write_units("j2", [{**UNITS[2], "relationships": []}])
    r = client.get("/api/search", params={"q": "SO-10077"})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"][0]["job_id"] == "j2"
    assert client.get("/api/search", params={"q": "x", "jobs": "nope"}).status_code == 404


def test_query_page_no_longer_503s_without_embeddings(client, tmp_data_dir):
    _write_units("j1", UNITS)
    r = client.post("/jobs/j1/query", data={"q": "SO-10042"})
    assert r.status_code == 200
    assert "rows1" in r.text


def test_tokenize_keeps_codes_whole_and_split():
    toks = search_mod.tokenize("Order SO-10042 in §3.2.1")
    assert {"so-10042", "so", "10042", "3.2.1"} <= set(toks)


# ───────── library, CLI, benchmark, storage ─────────


def test_library_chunk_file(fixtures):
    from cleave import chunk_file_detailed

    res = chunk_file_detailed(fixtures / "sales_q3.csv")
    assert res.units
    assert res.profile.get("route") == "tabular"
    assert all("order_id" in u.content for u in res.units if "SO-" in u.content)


def test_cli_chunk_writes_jsonl(fixtures, tmp_path):
    from cleave.cli import main

    out = tmp_path / "u.jsonl"
    assert main(["chunk", str(fixtures / "flat_essay.md"), "-o", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows
    assert rows[0]["source_file"].endswith("flat_essay.md")


def test_cli_serve_refuses_public_bind_without_key():
    from cleave.cli import main

    assert main(["serve", "--host", "0.0.0.0"]) == 2  # noqa: S104


def test_retrieval_benchmark_cleave_beats_fixed(fixtures, tmp_path):
    from cleave.eval_retrieval import run

    qa = Path(__file__).resolve().parent.parent / "benchmarks" / "qa_fixtures.jsonl"
    report = run(qa, fixtures, dense=False)
    o = report["overall"]
    assert o["cleave"]["recall@3"] >= o["fixed"]["recall@3"]
    assert o["cleave"]["recall@3"] >= 0.9


def test_vector_point_ids_are_stable_and_job_unique():
    import uuid

    a = uuid.uuid5(uuid.NAMESPACE_URL, "/data/jobs/a/input/x.pdf#f0_ku_0000")
    b = uuid.uuid5(uuid.NAMESPACE_URL, "/data/jobs/b/input/x.pdf#f0_ku_0000")
    assert a != b
    assert a == uuid.uuid5(uuid.NAMESPACE_URL, "/data/jobs/a/input/x.pdf#f0_ku_0000")
