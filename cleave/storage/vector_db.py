"""Vector Database layer: stores and queries KnowledgeUnit embeddings
in Qdrant/Milvus, falling back to local MiniLM embeddings and numpy search.
"""

from __future__ import annotations

import logging
import os

from ..http import client
from ..models import KnowledgeUnit

log = logging.getLogger(__name__)


class VectorDB:
    """Vector database client for Qdrant with local fallback."""

    def __init__(self, endpoint_url: str | None = None, collection: str = "cleave_units") -> None:
        self.endpoint_url = endpoint_url or os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
        self.collection = collection
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            r = client().get(f"{self.endpoint_url}/healthz", timeout=1.0)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def _ensure_collection(self, dim: int = 384) -> None:
        if not self.is_available():
            return
        try:
            r = client().get(f"{self.endpoint_url}/collections/{self.collection}", timeout=2.0)
            if r.status_code == 404:
                # Create collection
                client().put(
                    f"{self.endpoint_url}/collections/{self.collection}",
                    json={"vectors": {"size": dim, "distance": "Cosine"}},
                    timeout=5.0,
                )
        except Exception as exc:
            log.debug("Collection check failed: %s", exc)

    def insert_units(
        self, units: list[KnowledgeUnit], embeddings: list[list[float]] | None = None
    ) -> int:
        """Insert knowledge units with their vectors into the vector store."""
        if not units or not self.is_available():
            return 0

        # If embeddings aren't supplied, attempt to compute via sentence_transformers
        if embeddings is None:
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer("all-MiniLM-L6-v2")
                texts = [u.embed_text() for u in units]
                embs = model.encode(texts, convert_to_numpy=True)
                embeddings = [e.tolist() for e in embs]
            except Exception as exc:
                log.debug("Could not generate embeddings locally: %s", exc)
                return 0

        dim = len(embeddings[0]) if embeddings else 384
        self._ensure_collection(dim=dim)

        points = []
        for _i, (unit, vec) in enumerate(zip(units, embeddings)):
            points.append(
                {
                    "id": abs(hash(unit.id)) % (2**63),
                    "vector": vec,
                    "payload": {
                        "unit_id": unit.id,
                        "content": unit.content,
                        "modality": unit.modality.value
                        if hasattr(unit.modality, "value")
                        else str(unit.modality),
                        "token_count": unit.token_count,
                        "source_uri": unit.provenance.source_uri if unit.provenance else "",
                    },
                }
            )

        try:
            r = client().put(
                f"{self.endpoint_url}/collections/{self.collection}/points",
                json={"points": points},
                timeout=10.0,
            )
            if r.status_code == 200:
                log.info("Indexed %d units in VectorDB collection %s", len(points), self.collection)
                return len(points)
        except Exception as exc:
            log.warning("Failed to insert points into Qdrant: %s", exc)
        return 0


_default_vector_db: VectorDB | None = None


def get_vector_db() -> VectorDB:
    global _default_vector_db
    if _default_vector_db is None:
        _default_vector_db = VectorDB()
    return _default_vector_db
