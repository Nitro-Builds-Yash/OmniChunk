"""Contract import: knowledge from an external modality worker.

This is the seam described in CONTRACT.md. A separate repository (the video
pipeline) parses something Cleave cannot, and hands back either normalized
ContentElements — in which case Cleave still does the graph, routing and
chunking — or finished KnowledgeUnits, which are rendered as-is.

Keeping this in the core means the video work never has to touch Cleave, and
Cleave never has to grow a dependency on it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .ingest_document import IngestResult
from .models import (
    ELEMENT_KINDS,
    ChunkingDecision,
    ContentElement,
    Context,
    KnowledgeUnit,
    Modality,
    Provenance,
    Relationship,
    RelationType,
    Temporal,
    count_tokens,
    sha256_of,
)

log = logging.getLogger(__name__)

SUPPORTED_CONTRACT = 1


def load_contract(path: str | Path) -> tuple[IngestResult | None, list[KnowledgeUnit]]:
    """Read a contract file.

    Returns ``(ingest, [])`` for element payloads — the caller runs the normal
    pipeline over them — or ``(None, units)`` for finished units.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        # A clean job error beats a raw JSONDecodeError from an uploaded file.
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc
    version = payload.get("contract")
    if version != SUPPORTED_CONTRACT:
        raise ValueError(
            f"unsupported contract version {version!r}; this build speaks {SUPPORTED_CONTRACT}"
        )
    source_uri = payload.get("source_uri") or str(path)
    prefix = path.stem[:12]

    if payload.get("units"):
        units = [_unit_from(d, source_uri, prefix, i) for i, d in enumerate(payload["units"])]
        log.info("contract import: %d finished units from %s", len(units), path.name)
        return None, units

    elements = [_element_from(d, prefix, i) for i, d in enumerate(payload.get("elements", []))]
    if not elements:
        raise ValueError("contract file contains neither 'units' nor 'elements'")

    from .cleaning import clean_elements  # noqa: PLC0415

    report = clean_elements(elements)
    log.info(
        "contract import: %d elements from %s — %s", len(elements), path.name, report.summary()
    )
    return IngestResult(
        elements=elements,
        title=payload.get("title") or path.stem,
        source_uri=source_uri,
        sha256=sha256_of(str(path)),
        warnings=[f"imported via contract v{version} from {path.name}"],
        cleaning=report.to_dict(),
    ), []


def _valid_kind(kind: Any) -> str:
    """Coerce an unrecognised element kind to ``other``.

    An external worker sending ``paragrahp`` used to produce an element that no
    router branch matched and no strategy ever emitted — silently absent from
    the output rather than visibly wrong.
    """
    if kind in ELEMENT_KINDS:
        return str(kind)
    if kind:
        log.warning("unknown element kind %r — treating it as 'other'", kind)
    return "other"


def _element_from(d: dict[str, Any], prefix: str, i: int) -> ContentElement:
    bbox = d.get("bbox")
    return ContentElement(
        id=f"{prefix}_{d.get('id', i)}",
        kind=_valid_kind(d.get("kind")),
        text=(d.get("text") or "").strip(),
        level=d.get("level"),
        parent_id=(f"{prefix}_{d['parent_id']}" if d.get("parent_id") else None),
        page=d.get("page"),
        bbox=tuple(bbox) if bbox else None,
        t0=d.get("t0"),
        t1=d.get("t1"),
        speaker=d.get("speaker"),
        meta=d.get("meta") or {},
    )


def _unit_from(d: dict[str, Any], source_uri: str, prefix: str, i: int) -> KnowledgeUnit:
    ctx = d.get("context") or {}
    prov = d.get("provenance") or {}
    dec = d.get("decision") or {}
    temp = d.get("temporal")
    content = d.get("content", "")
    return KnowledgeUnit(
        id=f"{prefix}_{d.get('id', i)}",
        content=content,
        modality=Modality(d.get("modality", "video")),
        context=Context(
            document_title=ctx.get("document_title"),
            heading_path=list(ctx.get("heading_path") or []),
            situating_summary=ctx.get("situating_summary"),
            leading=ctx.get("leading"),
            trailing=ctx.get("trailing"),
            tier=int(ctx.get("tier", 0)),
        ),
        provenance=Provenance(
            source_uri=prov.get("source_uri") or source_uri,
            source_sha256=prov.get("source_sha256"),
            page=prov.get("page"),
        ),
        decision=ChunkingDecision(
            strategy=dec.get("strategy", "imported"),
            reason=dec.get("reason", "produced by an external modality worker"),
            signals={k: float(v) for k, v in (dec.get("signals") or {}).items()},
            vetoed_cuts=list(dec.get("vetoed_cuts") or []),
            escalation_flags=list(dec.get("escalation_flags") or []),
            llm_calls=int(dec.get("llm_calls", 0)),
            cost_usd=float(dec.get("cost_usd", 0.0)),
        ),
        relationships=[
            Relationship(
                type=RelationType(r["type"]),
                target_id=f"{prefix}_{r['target_id']}",
                confidence=float(r.get("confidence", 1.0)),
                evidence=r.get("evidence"),
            )
            for r in (d.get("relationships") or [])
            if r.get("type") in {t.value for t in RelationType}
        ],
        temporal=(
            Temporal(
                start_s=float(temp["start_s"]),
                end_s=float(temp["end_s"]),
                speaker=temp.get("speaker"),
            )
            if temp
            else None
        ),
        entities=list(d.get("entities") or []),
        metadata=d.get("metadata") or {},
        token_count=int(d.get("token_count") or count_tokens(content)),
        knowledge_unit_type=str(d.get("knowledge_unit_type") or "generic"),
        parent_id=f"{prefix}_{d['parent_id']}" if d.get("parent_id") else None,
        child_ids=[f"{prefix}_{cid}" for cid in d.get("child_ids", [])],
        level=d.get("level"),
        context_completeness=float(d.get("context_completeness", 1.0)),
        missing_context=list(d.get("missing_context") or []),
        boundary_trace=d.get("boundary_trace") or {},
    )
