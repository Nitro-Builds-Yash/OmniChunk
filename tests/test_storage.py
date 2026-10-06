"""Tests for storage layer connectors and fallbacks."""

from cleave.graph import ContextGraph
from cleave.models import (
    ChunkingDecision,
    ContentElement,
    Context,
    KnowledgeUnit,
    Modality,
    Provenance,
)
from cleave.storage.graph_db import GraphDB
from cleave.storage.object_store import ObjectStore
from cleave.storage.vector_db import VectorDB


def test_object_store_fallback(tmp_path):
    store = ObjectStore(endpoint_url="http://127.0.0.1:9999")
    store.local_root = tmp_path

    key = "test/sample.txt"
    payload = b"Hello, Cleave Storage!"

    uri = store.put_object(key, payload)
    assert uri.startswith("file://")

    retrieved = store.get_object(uri)
    assert retrieved == payload


def test_graph_db_offline_fallback():
    db = GraphDB(http_url="http://127.0.0.1:9999")
    assert db.is_available() is False

    elements = [
        ContentElement(id="e1", kind="heading", text="Chapter 1"),
        ContentElement(id="e2", kind="paragraph", text="Intro", parent_id="e1"),
    ]
    graph = ContextGraph(elements)
    stats = db.save_graph(graph, job_id="job_test_01")
    assert stats["nodes_synced"] == 2
    assert stats["neo4j_synced"] is False


def test_vector_db_offline_fallback():
    vdb = VectorDB(endpoint_url="http://127.0.0.1:9999")
    assert vdb.is_available() is False

    unit = KnowledgeUnit(
        id="ku_0001",
        content="Test vector chunk",
        modality=Modality.DOCUMENT,
        context=Context(document_title="Title"),
        provenance=Provenance(source_uri="test.pdf"),
        decision=ChunkingDecision(strategy="structural", reason="test chunk"),
    )
    inserted = vdb.insert_units([unit])
    assert inserted == 0
