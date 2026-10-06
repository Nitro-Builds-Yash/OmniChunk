"""The JSON surface: job creation, raw artifacts, usage, health."""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
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
    background.add_task(run_job, job_id, dest_paths)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


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
    from ..llm import get_provider  # noqa: PLC0415

    p = get_provider()
    return {"ok": True, "jobs": len(jobs.JOBS), "llm": p.name, "model": p.model}
