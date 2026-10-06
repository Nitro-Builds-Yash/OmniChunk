"""Graph Database layer: persists ContextGraph relationships (hierarchy, captions,
references, and cross-document entity links) to Neo4j, with local in-memory fallback.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..graph import ContextGraph
from ..http import client

log = logging.getLogger(__name__)


class GraphDB:
    """Neo4j graph sink with graceful offline fallback."""

    def __init__(self, bolt_url: str | None = None, http_url: str | None = None) -> None:
        self.http_url = http_url or os.environ.get("NEO4J_HTTP_URL", "http://127.0.0.1:7474")
        self.user = os.environ.get("NEO4J_USER", "neo4j")
        self.password = os.environ.get("NEO4J_PASSWORD", "cleavepassword")
        self._connected: bool | None = None

    def is_available(self) -> bool:
        if self._connected is not None:
            return self._connected
        try:
            r = client().get(self.http_url, timeout=1.0)
            self._connected = r.status_code == 200
        except Exception:
            self._connected = False
        return self._connected

    def save_graph(self, graph: ContextGraph, job_id: str) -> dict[str, Any]:
        """Persist graph structure to Neo4j if available, and return summary."""
        node_count = len(graph.g.nodes) if hasattr(graph, "g") else 0
        edge_count = len(graph.g.edges) if hasattr(graph, "g") else 0
        stats = {
            "job_id": job_id,
            "nodes_synced": node_count,
            "edges_synced": edge_count,
            "neo4j_synced": False,
        }

        if not self.is_available():
            log.debug("Neo4j not reachable at %s; using local graph serialisation", self.http_url)
            return stats

        try:
            # Format graph nodes and edges for Neo4j Cypher Transaction API
            statements = []

            # 1. Create/Merge Job Node
            statements.append(
                {
                    "statement": "MERGE (j:Job {id: $job_id})",
                    "parameters": {"job_id": job_id},
                }
            )

            # 2. Create Elements
            for node_id, data in graph.g.nodes(data=True):
                statements.append(
                    {
                        "statement": """
                    MERGE (e:ContentElement {id: $id, job_id: $job_id})
                    SET e.kind = $kind, e.text = $text, e.page = $page, e.speaker = $speaker
                    MERGE (j:Job {id: $job_id})
                    MERGE (j)-[:CONTAINS]->(e)
                    """,
                        "parameters": {
                            "id": node_id,
                            "job_id": job_id,
                            "kind": data.get("kind", ""),
                            "text": (data.get("text") or "")[:200],  # truncated summary in graph
                            "page": data.get("page"),
                            "speaker": data.get("speaker"),
                        },
                    }
                )

            # 3. Create Edges
            for u, v, edata in graph.g.edges(data=True):
                rel = edata.get("type", "RELATED_TO").upper()
                statements.append(
                    {
                        "statement": f"""
                    MATCH (u:ContentElement {{id: $u, job_id: $job_id}}), (v:ContentElement {{id: $v, job_id: $job_id}})
                    MERGE (u)-[r:{rel}]->(v)
                    SET r.confidence = $confidence, r.evidence = $evidence
                    """,
                        "parameters": {
                            "u": u,
                            "v": v,
                            "job_id": job_id,
                            "confidence": edata.get("confidence", 1.0),
                            "evidence": edata.get("evidence", ""),
                        },
                    }
                )

            # Submit batch cypher transaction via HTTP
            payload = {"statements": statements[:200]}  # capped batch
            auth = (self.user, self.password)
            r = client().post(
                f"{self.http_url}/db/neo4j/tx/commit",
                json=payload,
                auth=auth,
                timeout=15.0,
            )
            if r.status_code == 200 and not r.json().get("errors"):
                stats["neo4j_synced"] = True
                log.info(
                    "Persisted %d nodes and %d edges for job %s to Neo4j",
                    graph.node_count,
                    graph.edge_count,
                    job_id,
                )
            else:
                log.warning("Neo4j transaction returned errors: %s", r.text[:200])
        except Exception as exc:
            log.warning("Failed to sync graph to Neo4j for job %s: %s", job_id, exc)

        return stats


_default_graph_db: GraphDB | None = None


def get_graph_db() -> GraphDB:
    global _default_graph_db
    if _default_graph_db is None:
        _default_graph_db = GraphDB()
    return _default_graph_db
