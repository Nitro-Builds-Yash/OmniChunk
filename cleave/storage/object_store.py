"""Object storage layer: uploads and retrieves raw media/documents to MinIO/S3
with transparent fallback to local disk storage.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import BinaryIO

from ..http import client

log = logging.getLogger(__name__)


class ObjectStore:
    """S3/MinIO compatible object store client with local filesystem fallback."""

    def __init__(self, endpoint_url: str | None = None, bucket: str = "cleave-raw") -> None:
        self.endpoint_url = endpoint_url or os.environ.get(
            "MINIO_ENDPOINT", "http://127.0.0.1:9000"
        )
        self.bucket = bucket
        self.local_root = Path("data/objects")
        self.local_root.mkdir(parents=True, exist_ok=True)
        self._minio_available: bool | None = None

    def is_minio_available(self) -> bool:
        if self._minio_available is not None:
            return self._minio_available
        try:
            r = client().get(f"{self.endpoint_url}/minio/health/live", timeout=1.0)
            self._minio_available = r.status_code == 200
        except Exception:
            self._minio_available = False
        return self._minio_available

    def put_object(self, key: str, data: bytes | BinaryIO) -> str:
        """Store an object and return its canonical URI."""
        key = key.lstrip("/")
        payload = data if isinstance(data, bytes) else data.read()

        if self.is_minio_available():
            try:
                # Basic S3/MinIO PUT endpoint via HTTP
                url = f"{self.endpoint_url}/{self.bucket}/{key}"
                resp = client().put(
                    url,
                    content=payload,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=10.0,
                )
                if resp.status_code in (200, 201):
                    log.debug("Stored %s in MinIO at %s", key, url)
                    return f"s3://{self.bucket}/{key}"
            except Exception as exc:
                log.warning("Failed MinIO upload for %s (%s); falling back to disk", key, exc)

        # Fallback: Local filesystem
        dest = self.local_root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        return f"file://{dest.resolve().as_posix()}"

    def get_object(self, uri: str) -> bytes | None:
        """Retrieve an object's bytes from its URI."""
        if uri.startswith("s3://") and self.is_minio_available():
            parts = uri.replace("s3://", "").split("/", 1)
            if len(parts) == 2:
                bucket, key = parts
                try:
                    r = client().get(f"{self.endpoint_url}/{bucket}/{key}", timeout=10.0)
                    if r.status_code == 200:
                        return r.content
                except Exception as exc:
                    log.warning("Failed MinIO download for %s: %s", uri, exc)

        # Local fallback
        path_str = uri.replace("file://", "")
        p = Path(path_str)
        if not p.is_absolute():
            p = self.local_root / path_str.lstrip("/")
        if p.exists() and p.is_file():
            return p.read_bytes()
        return None


_default_store: ObjectStore | None = None


def get_object_store() -> ObjectStore:
    global _default_store
    if _default_store is None:
        _default_store = ObjectStore()
    return _default_store
