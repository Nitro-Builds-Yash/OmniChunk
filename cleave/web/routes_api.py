"""The JSON surface: job creation, raw artifacts, usage, health."""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse

from . import jobs, uploads

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/jobs")
async def create_job(
    background: BackgroundTasks, files: list[UploadFile] = File(...), use_llm: str = Form("true")
):
    from ..pipeline import run_job  # noqa: PLC0415 — avoids an import cycle

    job_id = uuid.uuid4().hex[:10]
    dest_dir = jobs.DATA / job_id / "input"
    try:
        names, dest_paths = await uploads.save_uploads(files, dest_dir)
    except Exception:
        uploads.discard(dest_dir)
        raise
    job = jobs.Job(
        id=job_id,
        filename=jobs.display_name(names),
        filenames=names,
        use_llm=use_llm.lower() not in ("false", "0", "off", "no"),
    )
    jobs.JOBS[job_id] = job
    jobs.save(job)
    background.add_task(jobs.submit, job, run_job, dest_paths)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str, background: BackgroundTasks):
    """Re-run a failed (or restart-interrupted) job from its stored inputs."""
    from ..pipeline import run_job  # noqa: PLC0415

    job = jobs.require_job(job_id)
    if job.status != "error":
        raise HTTPException(409, f"only failed jobs can be retried (this one is {job.status})")
    inputs = jobs.reset_for_retry(job)
    background.add_task(jobs.submit, job, run_job, inputs)
    return {"id": job.id, "status": job.status, "files": [p.name for p in inputs]}


@router.get("/api/search")
def api_search(q: str, jobs_: str | None = Query(None, alias="jobs"), k: int = 5):
    """Hybrid search (BM25 + vectors, graph-expanded) over one or more jobs.

    ``jobs`` is a comma-separated list of job ids; omitted, every finished job
    is searched.
    """
    from . import search  # noqa: PLC0415

    if not q.strip():
        raise HTTPException(400, "nothing to search")
    ids = (
        [j for j in jobs_.split(",") if j]
        if jobs_
        else [j.id for j in jobs.JOBS.values() if j.status == "done"]
    )
    for j in ids:
        jobs.require_job(j)
    hits = search.search_jobs(ids, q, k=max(1, min(k, 50)))
    return {"query": q, "jobs": ids, "hits": [h.to_dict() for h in hits]}


@router.get("/api/jobs/{job_id}/units")
def api_units(job_id: str):
    return _artifact(job_id, "units.json")


@router.get("/api/jobs/{job_id}/graph")
def api_graph(job_id: str):
    return _artifact(job_id, "graph.json")


@router.get("/api/jobs/{job_id}/profile")
def api_profile(job_id: str):
    return _artifact(job_id, "profile.json")


@router.get("/api/jobs/{job_id}/export")
def api_export(job_id: str, format: str = "json"):
    """Export KnowledgeUnits formatted for downstream RAG and AI agent frameworks."""
    path = jobs.job_artifact(job_id, "units.json")
    if not path.exists():
        raise HTTPException(404, "units.json not available")
    try:
        units_data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise HTTPException(404, "units.json is unreadable") from exc

    fmt = format.lower()
    if fmt == "langchain":
        exported = [
            {
                "page_content": u.get("embed_text") or u.get("content", ""),
                "metadata": {
                    "id": u.get("id"),
                    "modality": u.get("modality"),
                    "heading_path": u.get("context", {}).get("heading_path", []),
                    "source_uri": u.get("provenance", {}).get("source_uri"),
                    "strategy": u.get("decision", {}).get("strategy"),
                    "knowledge_unit_type": u.get("knowledge_unit_type", "generic"),
                    "context_completeness": u.get("context_completeness", 1.0),
                },
            }
            for u in units_data
        ]
        return JSONResponse(exported)

    if fmt in ("llamaindex", "llama_index"):
        exported = [
            {
                "id_": u.get("id"),
                "text": u.get("embed_text") or u.get("content", ""),
                "extra_info": {
                    "modality": u.get("modality"),
                    "title": u.get("context", {}).get("document_title"),
                    "heading_path": u.get("context", {}).get("heading_path", []),
                    "source_uri": u.get("provenance", {}).get("source_uri"),
                    "knowledge_unit_type": u.get("knowledge_unit_type", "generic"),
                    "token_count": u.get("token_count", 0),
                },
                "relationships": {
                    r.get("type"): r.get("target_id") for r in u.get("relationships", [])
                },
            }
            for u in units_data
        ]
        return JSONResponse(exported)

    if fmt == "qdrant":
        exported = [
            {
                "id": u.get("id"),
                "payload": {
                    "document": u.get("embed_text") or u.get("content", ""),
                    "raw_content": u.get("content", ""),
                    "modality": u.get("modality"),
                    "heading_path": u.get("context", {}).get("heading_path", []),
                    "knowledge_unit_type": u.get("knowledge_unit_type", "generic"),
                    "decision": u.get("decision", {}),
                    "relationships": u.get("relationships", []),
                },
            }
            for u in units_data
        ]
        return JSONResponse(exported)

    if fmt == "chroma":
        chroma_data = {
            "ids": [u.get("id") for u in units_data],
            "documents": [u.get("embed_text") or u.get("content", "") for u in units_data],
            "metadatas": [
                {
                    "modality": u.get("modality"),
                    "source": u.get("provenance", {}).get("source_uri") or "",
                    "type": u.get("knowledge_unit_type", "generic"),
                    "strategy": u.get("decision", {}).get("strategy") or "",
                }
                for u in units_data
            ],
        }
        return JSONResponse(chroma_data)

    return JSONResponse(units_data)


def _artifact(job_id: str, name: str) -> JSONResponse:
    """One job artifact, looked up through the registry."""
    path = jobs.job_artifact(job_id, name)
    if not path.exists():
        raise HTTPException(404, f"{name} not available")
    try:
        return JSONResponse(json.loads(path.read_text()))
    except (OSError, ValueError) as exc:
        log.warning("job %s: %s unreadable (%s)", job_id, name, exc)
        raise HTTPException(404, f"{name} is unreadable") from exc


@router.get("/api/usage")
def api_usage():
    """Install-wide token and cost ledger, broken down by model."""
    from ..llm import describe_providers  # noqa: PLC0415
    from ..usage import read_cumulative  # noqa: PLC0415

    return {"cumulative": read_cumulative(), "providers": describe_providers()}


@router.get("/health")
def health():
    from .._accel import backend_info  # noqa: PLC0415
    from ..llm import get_provider  # noqa: PLC0415

    p = get_provider()
    return {
        "ok": True,
        "jobs": len(jobs.JOBS),
        "llm": p.name,
        "model": p.model,
        "text_engine": backend_info(),
    }


@router.get("/health/live", include_in_schema=False)
def live():
    """Liveness without touching providers — for container health checks."""
    return {"ok": True}
