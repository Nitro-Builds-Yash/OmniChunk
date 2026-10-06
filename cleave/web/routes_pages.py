"""The HTML surface: home, job shell, status poll, results, search."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from . import search
from .jobs import JOBS, job_artifact, require_job
from .templating import ROOT, templates

log = logging.getLogger(__name__)

router = APIRouter()

#: What ``_results.html`` needs to render at all. An artifact written by an
#: older build can be missing any of these, and the template formats several of
#: them numerically ("%.4f" on cost), so a bare ``{}`` still raises.
_TOTALS_DEFAULTS = {
    "units": 0,
    "tier0_pct": 0,
    "flagged_for_enrichment": 0,
    "enriched": 0,
    "llm_calls": 0,
    "cost_usd": 0.0,
    "in_tokens": 0,
    "out_tokens": 0,
    "wall_clock_s": 0.0,
    "warnings": [],
    "use_llm": False,
}


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    from ..llm import describe_providers  # noqa: PLC0415
    from ..usage import read_cumulative  # noqa: PLC0415

    scorecard = None
    sc_path = ROOT / "data" / "scorecard.json"
    if sc_path.exists():
        try:
            scorecard = json.loads(sc_path.read_text())
        except (OSError, ValueError) as exc:
            # A malformed scorecard used to take the homepage down with it.
            log.warning("scorecard unreadable (%s) — rendering without it", exc)
    jobs = sorted(JOBS.values(), key=lambda j: j.created, reverse=True)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "jobs": jobs,
            "scorecard": scorecard,
            "usage": read_cumulative(),
            "providers": describe_providers(),
        },
    )


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    return templates.TemplateResponse(request, "job.html", {"job": require_job(job_id)})


@router.get("/jobs/{job_id}/status", response_class=HTMLResponse)
def job_status(request: Request, job_id: str):
    return templates.TemplateResponse(request, "_status.html", {"job": require_job(job_id)})


@router.get("/jobs/{job_id}/results", response_class=HTMLResponse)
def job_results(request: Request, job_id: str):
    job = require_job(job_id)
    try:
        units = json.loads((job.dir / "units.json").read_text())
        meta = json.loads((job.dir / "profile.json").read_text())
    except (OSError, ValueError) as exc:
        # An artifact from an older build should 404, not 500 the page.
        log.warning("job %s: results unreadable (%s)", job_id, exc)
        raise HTTPException(404, "results are unreadable") from exc
    graph_path = job.dir / "graph.json"
    graph = json.loads(graph_path.read_text()) if graph_path.exists() else None
    return templates.TemplateResponse(
        request,
        "_results.html",
        {
            "job": job,
            "units": units,
            "profile": meta.get("profile"),
            "totals": {**_TOTALS_DEFAULTS, **meta.get("totals", {})},
            "title": meta.get("title"),
            "graph": graph,
            "usage": meta.get("usage"),
            "enrichment": meta.get("enrichment"),
            "cleaning": meta.get("cleaning"),
            "files": meta.get("files"),
        },
    )


@router.post("/jobs/{job_id}/query", response_class=HTMLResponse)
async def query_job(request: Request, job_id: str):
    form = await request.form()
    q = str(form.get("q", "")).strip()
    if not q:
        raise HTTPException(400, "nothing to search")
    if not job_artifact(job_id, "units.json").exists():
        raise HTTPException(404, "no results to search")
    # Embedding is model inference: off the event loop, or every other
    # connection — including the status poller — stalls for its duration.
    hits = await run_in_threadpool(search.search_units, job_id, q)
    if not hits and not search.embedding_available():
        raise HTTPException(503, "embedding model unavailable — search needs MiniLM")
    return templates.TemplateResponse(request, "_query.html", {"q": q, "hits": hits})
