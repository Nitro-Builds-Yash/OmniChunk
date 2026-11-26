"""The job registry: what a job is, where it lives on disk, and how it comes back.

A module-level dict is the hot cache; a small SQLite file beside the job
directories is the record. Before it existed, a restart silently lost every
queued or running job, and nothing bounded how many ran at once — ten uploads
meant ten Docling parses fighting for the same cores. Now job state survives a
restart (interrupted jobs come back as retryable errors, never as fake
"running" ones), and ``submit`` admits at most ``CLEAVE_MAX_CONCURRENT_JOBS``
into the pipeline while the rest wait as ``queued``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
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


# ───────── persistence ─────────

_DB_LOCK = threading.Lock()
_COLUMNS = (
    "id",
    "filename",
    "filenames",
    "status",
    "progress",
    "message",
    "error",
    "created",
    "elapsed_s",
    "use_llm",
)


def _db_path() -> Path:
    # Resolved per call: tests repoint DATA at a temp dir.
    return DATA.parent / "jobs.sqlite3"


def _connect() -> sqlite3.Connection:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_db_path(), timeout=5.0)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, filename TEXT, filenames TEXT,"
        " status TEXT, progress INTEGER, message TEXT, error TEXT, created REAL,"
        " elapsed_s REAL, use_llm INTEGER)"
    )
    return conn


def save(job: Job) -> None:
    """Upsert one job's state. Best effort: a read-only disk must not kill a job."""
    row = asdict(job)
    row["filenames"] = json.dumps(row["filenames"])
    row["use_llm"] = int(row["use_llm"])
    try:
        with _DB_LOCK, _connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO jobs ({', '.join(_COLUMNS)}) "  # noqa: S608 — fixed column list
                f"VALUES ({', '.join('?' * len(_COLUMNS))})",
                [row[c] for c in _COLUMNS],
            )
    except sqlite3.Error as exc:
        log.warning("could not persist job %s: %s", job.id, exc)


def _load_saved() -> list[Job]:
    if not _db_path().exists():
        return []
    try:
        with _DB_LOCK, _connect() as conn:
            rows = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM jobs").fetchall()  # noqa: S608
    except sqlite3.Error as exc:
        log.warning("job registry unreadable (%s); falling back to artifacts", exc)
        return []
    out = []
    for r in rows:
        d = dict(zip(_COLUMNS, r))
        d["filenames"] = json.loads(d["filenames"] or "[]")
        d["use_llm"] = bool(d["use_llm"])
        out.append(Job(**d))
    return out


# ───────── bounded execution ─────────

_slots_lock = threading.Lock()
_slots: threading.BoundedSemaphore | None = None
_slots_size = 0


def _job_slots() -> threading.BoundedSemaphore:
    global _slots, _slots_size
    from ..config import settings  # noqa: PLC0415

    n = settings().max_concurrent_jobs
    with _slots_lock:
        if _slots is None or _slots_size != n:
            _slots, _slots_size = threading.BoundedSemaphore(n), n
        return _slots


def submit(job: Job, run: Callable[[str, list[Path]], None], paths: list[Path]) -> None:
    """Run ``run(job.id, paths)`` once a concurrency slot is free.

    Blocking — called from FastAPI's background threadpool. The job stays
    ``queued`` (and says so) until admitted, and its final state is persisted
    whatever the pipeline did.
    """
    save(job)
    slots = _job_slots()
    if not slots.acquire(blocking=False):
        job.message = "waiting for a free worker…"
        save(job)
        slots.acquire()
    try:
        run(job.id, paths)
    finally:
        slots.release()
        save(job)


def reset_for_retry(job: Job) -> list[Path]:
    """Put a failed job back to ``queued`` and return its stored inputs."""
    inputs = sorted(p for p in (job.dir / "input").glob("*") if p.is_file())
    if not inputs:
        raise HTTPException(409, "this job's input files are gone; upload again")
    job.status, job.progress, job.message, job.error = "queued", 0, "queued (retry)", None
    job.created = time.time()
    save(job)
    return inputs


def rehydrate_jobs() -> None:
    """Restore the registry on startup: SQLite first, then legacy artifacts.

    A job that was queued or running when the process died is restored as an
    error that says so and can be retried — showing it as running would be a
    lie, and dropping it (the old behaviour) lost the upload's existence.
    """
    for job in _load_saved():
        if job.id in JOBS:
            continue
        if not job.is_terminal:
            job.status, job.message = "error", "interrupted"
            job.error = "the server restarted while this job was in progress — retry it"
            save(job)
        JOBS[job.id] = job
    _rehydrate_from_artifacts()


def _rehydrate_from_artifacts() -> None:
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
        save(JOBS[job_id])
    if JOBS:
        log.info("restored %d job(s) from disk", len(JOBS))


def set_progress(job: Job, progress: int, message: str) -> None:
    # progress never goes backwards; terminal states are set only by the worker
    job.progress = max(job.progress, min(99, progress))
    job.message = message
    save(job)
