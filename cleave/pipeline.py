"""Job orchestration: ingest → graph → chunk → enrich → artifacts.

Lives outside ``cleave.web`` and imports nothing from FastAPI, so the pipeline
is usable from the evaluator, from a future CLI, and from tests without standing
up an HTTP app.

One file's failure no longer discards the rest of the job. A stopped STT worker
used to raise straight out of the per-file loop and fail the whole batch, so an
audio file nobody could transcribe threw away a 200-page PDF that had parsed
perfectly well beside it. Each file is now attempted independently and its
failure is reported as a warning; only a job where *everything* failed is an
error.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from .logging_setup import current_job_id
from .models import Relationship, RelationType
from .web.jobs import JOBS, Job, set_progress

log = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


class FileOutcome(NamedTuple):
    """One input file's contribution to a job.

    A NamedTuple rather than a bare tuple: the caller unpacked five positional
    values, and a sixth would have been a silent reordering bug.
    """

    units: list
    meta: dict
    graph_nodes: list
    graph_edges: list
    enrichment: dict | None


def run_job(job_id: str, input_paths: list[Path]) -> None:
    """Process every input file through its own ingest → graph → chunk →
    enrich pipeline, then merge the results into one job.

    Each file keeps the routing decision that fits it — a spreadsheet and a
    PDF in the same batch still get tabular and structural routes respectively
    — they are simply reported together and searched together. Element and
    unit ids are namespaced per file (see ``_prefix_*``) so nothing collides
    when the results are combined.
    """
    job = JOBS[job_id]
    job.status = "running"
    token = current_job_id.set(job_id)
    t0 = time.time()
    n = len(input_paths)
    try:
        from .usage import Ledger  # noqa: PLC0415

        ledger = Ledger()
        all_units: list = []
        files_meta: list[dict] = []
        graph_nodes: list = []
        graph_edges: list = []
        enrichments: list[dict] = []
        failures: list[tuple[str, str]] = []

        for i, input_path in enumerate(input_paths):
            lo, hi = 5 + int(85 * i / n), 5 + int(85 * (i + 1) / n)

            def progress(frac: float, msg: str, lo=lo, hi=hi) -> None:
                set_progress(job, lo + int((hi - lo) * min(1.0, max(0.0, frac))), msg)

            try:
                outcome = _process_file(
                    job, input_path, prefix=f"f{i}_", ledger=ledger, progress=progress
                )
            except Exception as exc:
                log.exception("job %s: %s failed", job_id, input_path.name)
                failures.append((input_path.name, f"{type(exc).__name__}: {exc}"))
                continue
            all_units.extend(outcome.units)
            files_meta.append(outcome.meta)
            graph_nodes.extend(outcome.graph_nodes)
            graph_edges.extend(outcome.graph_edges)
            if outcome.enrichment:
                enrichments.append(outcome.enrichment)

        if failures and not files_meta:
            detail = "; ".join(f"{name}: {err}" for name, err in failures)
            raise RuntimeError(f"every input failed — {detail}")

        if len(files_meta) > 1:
            _link_cross_document_relationships(all_units, files_meta, graph_edges)

        # Cross-chunk Entity and Topic Enrichment via Gemini
        if job.use_llm and all_units:
            try:
                from .enrich_entities import enrich_entities_batch  # noqa: PLC0415

                enrich_entities_batch(all_units, max_enrich=15)
            except Exception as exc:
                log.debug("Entity enrichment step skipped: %s", exc)

        set_progress(job, 90, f"{len(all_units)} knowledge units — writing artifacts & syncing…")
        graph = (
            {"nodes": graph_nodes, "edges": graph_edges} if (graph_nodes or graph_edges) else None
        )
        _write_artifacts(
            job,
            all_units,
            files_meta,
            graph,
            t0,
            ledger=ledger,
            enrichments=enrichments,
            failures=failures,
        )

        # Sync to external databases if available
        try:
            from .storage import get_vector_db  # noqa: PLC0415

            vdb = get_vector_db()
            if vdb.is_available():
                vdb.insert_units(all_units)
        except Exception as exc:
            log.debug("VectorDB sync skipped: %s", exc)

        job.elapsed_s = round(time.time() - t0, 1)
        job.progress, job.message, job.status = 100, "done", "done"
    except Exception as exc:  # surface honestly; never a silent dead job
        log.exception("job %s failed", job_id)
        job.error = f"{type(exc).__name__}: {exc}"
        job.message = "failed"
        job.status = "error"
    finally:
        current_job_id.reset(token)


def _prefix_element(e, prefix: str) -> None:
    """Namespace one element's id/parent_id/caption_ids so a second file's
    ``el_0000`` never collides with the first's."""
    e.id = prefix + e.id
    if e.parent_id:
        e.parent_id = prefix + e.parent_id
    if e.meta.get("caption_ids"):
        e.meta["caption_ids"] = [prefix + c for c in e.meta["caption_ids"]]


def _prefix_unit(u, prefix: str) -> None:
    """Namespace one unit's id and every relationship target it carries —
    all relationship targets in this pipeline are other units from the same
    file, so a uniform prefix keeps them internally consistent."""
    u.id = prefix + u.id
    for r in u.relationships:
        r.target_id = prefix + r.target_id


def _process_file(job: Job, input_path: Path, *, prefix: str, ledger, progress) -> FileOutcome:
    """Run the single-file pipeline for one upload in a (possibly multi-file) job."""
    suffix = input_path.suffix.lower()
    filename = input_path.name
    progress(0.0, f"understanding {filename}…")

    if suffix == ".json":
        from .ingest_contract import load_contract  # noqa: PLC0415

        progress(0.1, f"importing contract payload… ({filename})")
        imported, ready_units = load_contract(input_path)
        file_meta: dict
        if ready_units:
            for u in ready_units:
                _prefix_unit(u, prefix)
            file_meta = {
                "filename": filename,
                "title": ready_units[0].context.document_title,
                "source": str(input_path),
                "warnings": [],
                "profile": {
                    "route": "imported",
                    "route_reason": "knowledge units produced by an external modality "
                    "worker and imported through the contract",
                },
                "cleaning": None,
            }
            progress(1.0, f"done ({filename})")
            return FileOutcome(ready_units, file_meta, [], [], None)
        if imported is None:  # pragma: no cover - load_contract raises first
            raise ValueError(f"{filename} contained neither units nor elements")
        ingest = imported
    elif suffix in AUDIO_EXTS:
        from .ingest_audio import ingest_audio  # noqa: PLC0415

        progress(0.1, f"transcribing (STT worker)… ({filename})")
        ingest = ingest_audio(input_path)
    elif suffix in VIDEO_EXTS:
        from .workers.vision_worker import process_video_file  # noqa: PLC0415

        progress(0.1, f"processing video (Gemini multimodal worker)… ({filename})")
        ingest = process_video_file(input_path)
    elif suffix in IMAGE_EXTS:
        from .workers.vision_worker import process_image_file  # noqa: PLC0415

        progress(0.1, f"processing image (Gemini vision worker)… ({filename})")
        ingest = process_image_file(input_path)
    else:
        from .ingest_document import ingest_document  # noqa: PLC0415

        progress(0.1, f"parsing structure (Docling)… ({filename})")
        ingest = ingest_document(input_path)

    for e in ingest.elements:
        _prefix_element(e, prefix)

    progress(0.5, f"{len(ingest.elements)} elements — building context graph… ({filename})")
    from .chunkers import chunk  # noqa: PLC0415
    from .graph import ContextGraph  # noqa: PLC0415

    graph = ContextGraph(ingest.elements)
    progress(0.6, f"routing and chunking… ({filename})")
    units, profile = chunk(ingest, graph)
    for u in units:
        _prefix_unit(u, prefix)

    enrichment: dict | None = None
    flagged_n = sum(1 for u in units if u.decision.escalation_flags)
    if flagged_n:
        from .enrich import enrich  # noqa: PLC0415

        progress(
            0.7,
            (
                f"enriching {flagged_n} context-poor chunks… ({filename})"
                if job.use_llm
                else f"skipping LLM enrichment ({filename})…"
            ),
        )
        doc_text = "\n\n".join(e.text for e in ingest.elements if e.text)
        enrichment = enrich(
            units,
            doc_text,
            ledger=ledger,
            use_llm=job.use_llm,
            progress=lambda i, n2: progress(
                0.7 + 0.25 * i / max(1, n2), f"enriching… {i}/{n2} ({filename})"
            ),
        )

    file_meta = {
        "filename": filename,
        "title": ingest.title,
        "source": ingest.source_uri,
        "warnings": ingest.warnings,
        "profile": profile.to_dict(),
        "cleaning": ingest.cleaning,
    }
    progress(1.0, f"done ({filename})")
    gd = graph.to_dict()
    return FileOutcome(units, file_meta, gd["nodes"], gd["edges"], enrichment)


def _merge_cleaning(cleanings: list[dict | None]) -> dict | None:
    total_fixes = elements_changed = chars_removed = 0
    by_rule: Counter = Counter()
    for c in cleanings:
        if not c:
            continue
        total_fixes += c.get("total_fixes", 0)
        elements_changed += c.get("elements_changed", 0)
        chars_removed += c.get("chars_removed", 0)
        by_rule.update(c.get("by_rule", {}))
    if not total_fixes:
        return None
    return {
        "total_fixes": total_fixes,
        "elements_changed": elements_changed,
        "chars_removed": chars_removed,
        "by_rule": dict(by_rule.most_common()),
    }


def _merge_enrichment(parts: list[dict]) -> dict | None:
    if not parts:
        return None
    merged = dict(parts[-1])
    for k in ("flagged", "enriched", "api_calls", "calls_saved_by_batching", "failed_calls"):
        merged[k] = sum(p.get(k, 0) for p in parts)
    return merged


def _write_artifacts(
    job: Job,
    units,
    files_meta: list[dict],
    graph: dict | None,
    t0: float,
    *,
    ledger=None,
    enrichments: list[dict] | None = None,
    failures: list[tuple[str, str]] | None = None,
) -> None:
    """units.json + graph.json + profile.json for one job.

    A single-file job writes the same flat schema this always has (``profile``,
    ``cleaning``, ``title``, ``source`` at the top level). A multi-file job adds
    a ``files`` array with one entry per input, alongside combined totals so
    existing single-file consumers of profile.json see nothing new.
    """
    from .usage import append_to_cumulative  # noqa: PLC0415

    job.dir.mkdir(parents=True, exist_ok=True)
    (job.dir / "units.json").write_text(json.dumps([u.to_dict() for u in units], indent=1))
    if graph is not None:
        (job.dir / "graph.json").write_text(json.dumps(graph, indent=1))

    usage = ledger.to_dict() if ledger is not None else None
    warnings = [
        w if len(files_meta) == 1 else f"{f['filename']}: {w}"
        for f in files_meta
        for w in f["warnings"]
    ]
    # A file that failed outright is a warning on the job, not a silent absence.
    warnings += [f"{name}: {err}" for name, err in (failures or [])]
    merged_enrichment = _merge_enrichment(enrichments or [])
    if merged_enrichment and merged_enrichment.get("warning"):
        warnings.append(merged_enrichment["warning"])

    totals = {
        "units": len(units),
        "tier0_pct": round(100 * sum(1 for u in units if u.context.tier == 0) / max(1, len(units))),
        "flagged_for_enrichment": sum(1 for u in units if u.decision.escalation_flags),
        "enriched": sum(1 for u in units if u.context.situating_summary),
        # Model calls are counted from the ledger, not per unit: batching means
        # one call can serve several units, and the bill follows the calls.
        "llm_calls": (usage or {}).get("totals", {}).get("calls", 0),
        "cost_usd": round((usage or {}).get("totals", {}).get("cost_usd", 0.0), 6),
        "in_tokens": (usage or {}).get("totals", {}).get("in_tokens", 0),
        "out_tokens": (usage or {}).get("totals", {}).get("out_tokens", 0),
        "wall_clock_s": round(time.time() - t0, 1),
        "warnings": warnings,
        "use_llm": job.use_llm,
    }

    record: dict = {"totals": totals}
    if len(files_meta) == 1:
        fm = files_meta[0]
        record["profile"] = fm["profile"]
        record["title"] = fm["title"]
        record["source"] = fm["source"]
        if fm["cleaning"]:
            record["cleaning"] = fm["cleaning"]
    else:
        record["files"] = files_meta
        record["title"] = job.filename
        combined_cleaning = _merge_cleaning([f["cleaning"] for f in files_meta])
        if combined_cleaning:
            record["cleaning"] = combined_cleaning
    if usage:
        record["usage"] = usage
    if merged_enrichment:
        record["enrichment"] = merged_enrichment
    (job.dir / "profile.json").write_text(json.dumps(record, indent=1))

    if ledger is not None and ledger.total_calls:
        append_to_cumulative(ledger, job.id)


def _link_cross_document_relationships(
    units: list, files_meta: list[dict], graph_edges: list
) -> None:
    """When a job contains multiple files, detect cross-document references and establish
    typed relationships between units across different uploaded files."""
    file_anchors: dict[str, str] = {}
    for u in units:
        # Find primary anchors for each file (schema cards, section heads, or first unit)
        fname = (
            Path(u.provenance.source_uri).name if u.provenance and u.provenance.source_uri else ""
        )
        if not fname:
            continue
        if fname not in file_anchors or u.knowledge_unit_type in ("schema_card", "section"):
            file_anchors[fname] = u.id

    for u in units:
        src_fname = (
            Path(u.provenance.source_uri).name if u.provenance and u.provenance.source_uri else ""
        )
        content_lower = u.content.lower()
        for target_fname, target_uid in file_anchors.items():
            if target_fname == src_fname or target_uid == u.id:
                continue
            stem = Path(target_fname).stem.lower()
            if (
                len(stem) > 3
                and (target_fname.lower() in content_lower or stem in content_lower)
                and not any(r.target_id == target_uid for r in u.relationships)
            ):
                # Found cross-document reference
                u.relationships.append(
                    Relationship(
                        type=RelationType.REFERENCES,
                        target_id=target_uid,
                        confidence=0.85,
                        evidence=f"cross-document mention of {target_fname}",
                    )
                )
                graph_edges.append(
                    {
                        "source": u.id,
                        "target": target_uid,
                        "type": "references",
                        "confidence": 0.85,
                        "evidence": f"cross-document mention of {target_fname}",
                        "importance": 0.80,
                    }
                )
