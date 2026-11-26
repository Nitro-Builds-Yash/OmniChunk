"""The library surface: chunk files from Python without the web app.

    >>> from cleave import chunk_file
    >>> units = chunk_file("report.pdf")
    >>> units[0].embed_text()

The same per-file pipeline the server runs (ingest → clean → graph → route →
chunk → optional enrichment), with no job registry, no artifacts on disk and,
by default, no LLM calls.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .models import KnowledgeUnit

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkResult:
    """Units plus the routing decision and warnings for one input file."""

    path: Path
    units: list[KnowledgeUnit]
    profile: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _LibraryJob:
    """The only fields ``pipeline._process_file`` reads from a job."""

    use_llm: bool = False


def chunk_file_detailed(path: str | Path, *, use_llm: bool = False) -> ChunkResult:
    """Chunk one file and return its units with the routing profile."""
    from .pipeline import _process_file  # noqa: PLC0415 — keep `import cleave` light
    from .usage import Ledger  # noqa: PLC0415

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    outcome = _process_file(
        _LibraryJob(use_llm=use_llm),  # type: ignore[arg-type]
        p,
        prefix="",
        ledger=Ledger(),
        progress=lambda _frac, msg: log.debug("%s", msg),
    )
    return ChunkResult(
        path=p,
        units=outcome.units,
        profile=outcome.meta.get("profile") or {},
        warnings=list(outcome.meta.get("warnings") or []),
    )


def chunk_file(path: str | Path, *, use_llm: bool = False) -> list[KnowledgeUnit]:
    """Chunk one file (PDF, DOCX, PPTX, MD, HTML, CSV, XLSX, audio, contract JSON)."""
    return chunk_file_detailed(path, use_llm=use_llm).units


def chunk_files(
    paths: Iterable[str | Path], *, use_llm: bool = False
) -> dict[Path, list[KnowledgeUnit]]:
    """Chunk several files independently; one failure does not stop the rest."""
    out: dict[Path, list[KnowledgeUnit]] = {}
    for path in paths:
        try:
            out[Path(path)] = chunk_file(path, use_llm=use_llm)
        except Exception:
            log.exception("failed to chunk %s", path)
    return out
