"""The job registry: what a job is, where it lives on disk, and how it comes back.

Deliberately small — a module-level dict and a directory per job. Jobs run in
seconds and their artifacts land on disk as JSON; nothing here needs a queue or
a database.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import HTTPException

from .templating import ROOT

log = logging.getLogger(__name__)

DATA = ROOT / "data" / "jobs"


@dataclass(slots=True)
class Job:
    id: str
    filename: str  # display summary, e.g. "3 files (a.pdf, b.mp3, …)"
    filenames: list[str] = field(default_factory=list)
    status: str = "queued"  # queued | running | done | error
    progress: int = 0
    message: str = "queued"
    error: str | None = None
    created: float = field(default_factory=time.time)
    elapsed_s: float = 0.0
    use_llm: bool = True  # per-job override for LLM enrichment

    @property
    def is_terminal(self) -> bool:
        return self.status in ("done", "error")

    @property
    def dir(self) -> Path:
        return DATA / self.id


JOBS: dict[str, Job] = {}


def require_job(job_id: str) -> Job:
    """The job, or a 404 — never a path built from unvalidated input.

    Every route that touches a job directory goes through here. Two API routes
    used to build ``DATA / job_id / name`` straight from the URL, so a job id of
    ``../..`` read files from outside the data directory entirely. Because a
    ``Job.id`` is either a ``uuid4`` hex or a directory name this module itself
    read from disk, going through the registry removes the traversal rather than
    trying to sanitise it.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job


def job_artifact(job_id: str, name: str) -> Path:
    """Path to one of a job's artifacts, guaranteed to sit inside its directory."""
    job = require_job(job_id)
    path = (job.dir / name).resolve()
    if not path.is_relative_to(DATA.resolve()):  # pragma: no cover - belt and braces
        raise HTTPException(404, "unknown job")
    return path


def display_name(names: list[str]) -> str:
    """One-line summary for a job's input files, for job lists and titles."""
    if not names:
        return "upload"
    if len(names) == 1:
        return names[0]
    shown = ", ".join(names[:3])
    more = f", +{len(names) - 3} more" if len(names) > 3 else ""
    return f"{len(names)} files ({shown}{more})"


def rehydrate_jobs() -> None:
    """Re-register finished jobs from disk on startup.

    The artifacts outlive the process, so a restart should not turn a completed
    job into a 404 — the results page reads from these files anyway. Only
    terminal jobs come back: anything that was mid-flight died with the old
    process and would be a lie to show as running.
    """
    if not DATA.exists():
        return
    for profile in DATA.glob("*/profile.json"):
        job_id = profile.parent.name
        if job_id in JOBS:
            continue
        try:
            meta = json.loads(profile.read_text())
        except (OSError, ValueError) as exc:
            # Silently skipping made a corrupt job simply vanish from the list.
            log.warning("skipping job %s: unreadable profile.json (%s)", job_id, exc)
            continue
        inputs = sorted((profile.parent / "input").glob("*"))
        names = [p.name for p in inputs]
        JOBS[job_id] = Job(
            id=job_id,
            filename=display_name(names) if names else (meta.get("title") or job_id),
            filenames=names,
            status="done",
            progress=100,
            message="done",
            created=profile.stat().st_mtime,
            elapsed_s=meta.get("totals", {}).get("wall_clock_s", 0.0),
        )
    if JOBS:
        log.info("restored %d completed job(s) from disk", len(JOBS))


def set_progress(job: Job, progress: int, message: str) -> None:
    # progress never goes backwards; terminal states are set only by the worker
    job.progress = max(job.progress, min(99, progress))
    job.message = message
