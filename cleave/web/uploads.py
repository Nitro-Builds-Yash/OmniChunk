"""Taking files from the browser and putting them somewhere safe.

``UploadFile.filename`` is whatever the client chose to send. A browser sends a
leaf name; anything else can send ``../../../evil.txt``, and the destination
used to be built as ``dest_dir / filename`` with no reduction to a basename —
so a crafted upload wrote outside the job directory. The extension allowlist did
not help, because ``.txt`` is on it.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath

from fastapi import HTTPException, UploadFile

log = logging.getLogger(__name__)

MAX_UPLOAD = 50 * 1024 * 1024  # per file
MAX_FILES = 20  # per job
MAX_TOTAL_UPLOAD = 200 * 1024 * 1024  # per job, across all files

DOC_EXTS = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".html", ".htm", ".md", ".txt"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
CONTRACT_EXTS = {".json"}  # payloads from external modality workers (CONTRACT.md)
ALLOWED_EXTS = DOC_EXTS | IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS | CONTRACT_EXTS


def safe_upload_name(raw: str | None, index: int) -> str:
    """Reduce a client-supplied filename to a leaf name that cannot escape.

    Both separators are stripped regardless of platform: a POSIX server still
    receives Windows-shaped names, and ``Path("..\\evil.txt").name`` on Linux is
    the whole string. Leading dots go too, so neither ``..`` nor a dotfile can
    be constructed.
    """
    leaf = PurePosixPath(PureWindowsPath(raw or "").as_posix()).name
    leaf = leaf.strip().lstrip(".")
    if not leaf or leaf in {".", ".."}:
        return f"upload{index}"
    return leaf[:120]


async def save_uploads(files: list[UploadFile], dest_dir: Path) -> tuple[list[str], list[Path]]:
    """Stream uploads to ``dest_dir``. Returns (names, paths).

    Raises ``HTTPException`` on anything unacceptable; the caller removes the
    directory, so a rejected batch leaves nothing behind.
    """
    if not files:
        raise HTTPException(400, "no files uploaded")
    if len(files) > MAX_FILES:
        # Checked before writing anything: the per-file cap alone allowed
        # a hundred files of 50MB each.
        raise HTTPException(413, f"too many files ({len(files)}); the limit is {MAX_FILES}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    resolved_dir = dest_dir.resolve()
    names: list[str] = []
    paths: list[Path] = []
    total = 0

    for i, f in enumerate(files):
        fname = safe_upload_name(f.filename, i)
        suffix = Path(fname).suffix.lower()
        if suffix not in ALLOWED_EXTS:
            raise HTTPException(415, f"unsupported type {suffix!r} ({fname})")
        dest = dest_dir / fname
        if dest.exists():  # two files with the same name in one batch
            dest = dest_dir / f"{dest.stem}_{i}{dest.suffix}"
        if not dest.resolve().is_relative_to(resolved_dir):  # pragma: no cover
            # Unreachable given safe_upload_name, and kept so it stays that way.
            raise HTTPException(400, f"invalid filename {fname!r}")

        written = 0
        with dest.open("wb") as out:
            while chunk_bytes := await f.read(1 << 20):
                written += len(chunk_bytes)
                total += len(chunk_bytes)
                if written > MAX_UPLOAD:
                    raise HTTPException(413, f"{fname} exceeds 50MB")
                if total > MAX_TOTAL_UPLOAD:
                    raise HTTPException(
                        413, f"upload exceeds {MAX_TOTAL_UPLOAD // (1024 * 1024)}MB in total"
                    )
                out.write(chunk_bytes)
        names.append(dest.name)
        paths.append(dest)

    return names, paths


def discard(dest_dir: Path) -> None:
    """Remove a partially written job directory."""
    shutil.rmtree(dest_dir, ignore_errors=True)
