"""Comprehensive tests for Universal Boundary Decision Engine capabilities.

Tests cover:
1. Universal Boundary Candidate generation & multi-modal signals
2. Universal Boundary Scoring & multimodal consensus
3. Hard vs Soft constraint enforcement
4. ContextGraph active intelligence & relationship loss penalty
5. Video multimodal alignment (speech + visual + OCR)
6. Conversational knowledge unit classification
7. Adaptive chunk granularity
8. Hierarchical knowledge units (parent-child links)
9. Context completeness scoring
10. Universal evaluation metrics (coherence, completeness, relationship preservation, retrieval)
"""

from __future__ import annotations

from cleave.boundary_engine import generate_candidates_for_region
from cleave.chunkers import chunk
from cleave.completeness import enrich_context_completeness, evaluate_context_completeness
from cleave.conversational import classify_conversational_elements
from cleave.evaluate import (
    boundary_coherence_score,
    chunk_size_variance,
    context_completeness_score,
    fragmentation_rate,
    relationship_preservation_rate,
    retrieval_evaluation,
)
from cleave.graph import ContextGraph
from cleave.ingest_document import IngestResult
from cleave.models import (
    ContentElement,
    Context,
    KnowledgeUnit,
    KnowledgeUnitType,
    Modality,
    RelationType,
)


def _make_ingest(elements: list[ContentElement], title: str = "Test Doc") -> IngestResult:
    return IngestResult(
        elements=elements,
        title=title,
        source_uri="test_doc.md",
        sha256="testsha256",
    )


# ───────── 1. Boundary Candidate Generation & Signals ─────────


def test_document_heading_generates_strong_boundary_candidate():
    elements = [
        ContentElement(
            id="p1", kind="paragraph", text="Introduction paragraph with some background."
        ),
        ContentElement(id="h1", kind="heading", text="2. Method Overview", level=2),
        ContentElement(id="p2", kind="paragraph", text="Details of the method."),
    ]
    graph = ContextGraph(elements)
    candidates = generate_candidates_for_region(elements, graph)

    assert len(candidates) == 2
    # Candidate 0 is between p1 and h1 (cut before h1)
    cand_h1 = candidates[0]
    assert cand_h1.signals.get("structural_strength", 0.0) >= 0.8
    assert "heading transition" in cand_h1.reason


def test_heading_stranding_is_vetoed():
    elements = [
        ContentElement(id="h1", kind="heading", text="1. Introduction", level=1),
        ContentElement(id="p1", kind="paragraph", text="Paragraph under introduction."),
    ]
    graph = ContextGraph(elements)
    candidates = generate_candidates_for_region(elements, graph)

    assert len(candidates) == 1
    cand = candidates[0]
    assert any("would strand heading" in v for v in cand.veto_reasons)


# ───────── 2. Hard vs Soft Constraint Enforcement ─────────


def test_caption_float_severance_is_hard_veto():
    elements = [
        ContentElement(id="p1", kind="paragraph", text="Lead paragraph."),
        ContentElement(id="cap1", kind="caption", text="Figure 1: Architecture diagram"),
        ContentElement(id="fig1", kind="figure", text="", meta={"caption_ids": ["cap1"]}),
        ContentElement(id="p2", kind="paragraph", text="Follow up paragraph."),
    ]
    graph = ContextGraph(elements)
    candidates = generate_candidates_for_region(elements, graph)

    # Candidate between cap1 and fig1 (index 2 in region)
    cand_cap_fig = next(
        c for c in candidates if c.left_element_id == "cap1" and c.right_element_id == "fig1"
    )
    assert any("sever" in v for v in cand_cap_fig.veto_reasons)


def test_list_items_treated_as_soft_boundaries():
    elements = [
        ContentElement(id="p1", kind="paragraph", text="The steps are:"),
        ContentElement(id="li1", kind="list_item", text="First step"),
        ContentElement(id="li2", kind="list_item", text="Second step"),
    ]
    graph = ContextGraph(elements)
    candidates = generate_candidates_for_region(elements, graph)

    cand_list = next(
        c for c in candidates if c.left_element_id == "li1" and c.right_element_id == "li2"
    )
    assert cand_list.is_soft is True
    assert cand_list.signals.get("list_continuation") == 1.0


# ───────── 3. ContextGraph Active Intelligence & Relationship Loss ─────────


def test_graph_relationship_loss_calculated():
    elements = [
        ContentElement(id="p1", kind="paragraph", text="We see the details in Table 1 below."),
        ContentElement(
            id="t1", kind="table", text="Table body", meta={"grid": [["A", "B"], ["1", "2"]]}
        ),
        ContentElement(id="p2", kind="paragraph", text="Continuing discussion."),
    ]
    graph = ContextGraph(elements)
    # Prose explains table
    assert graph.g.has_edge("p1", "t1")

    loss, severed = graph.relationship_loss(left_ids={"p1"}, right_ids={"t1", "p2"})
    assert loss > 0.0
    assert any("explains" in s or "p1" in s for s in severed)


def test_graph_separation_score():
    elements = [
        ContentElement(id="h1", kind="heading", text="Section 1", level=1),
        ContentElement(id="p1", kind="paragraph", text="Text in section 1", parent_id="h1"),
        ContentElement(id="h2", kind="heading", text="Section 2", level=1),
        ContentElement(id="p2", kind="paragraph", text="Text in section 2", parent_id="h2"),
    ]
    graph = ContextGraph(elements)
    # p1 and p2 have different heading paths
    sep = graph.graph_separation_score("p1", "p2")
    assert sep >= 0.9


# ───────── 4. Video Multimodal Boundary Alignment & Consensus ─────────


def test_video_multimodal_consensus_boosts_confidence():
    elements = [
        ContentElement(
            id="v1",
            kind="speech_segment",
            text="This concludes our discussion on revenue.",
            t0=0.0,
            t1=10.0,
            speaker="Alice",
            meta={"visual_summary": "revenue slide", "ocr_text": "Q3 Revenue"},
        ),
        ContentElement(
            id="v2",
            kind="speech_segment",
            text="Now let us look at the new product roadmap.",
            t0=12.5,
            t1=22.0,
            speaker="Bob",
            meta={"visual_summary": "roadmap timeline", "ocr_text": "2027 Roadmap"},
        ),
    ]
    graph = ContextGraph(elements)
    candidates = generate_candidates_for_region(elements, graph, modality=Modality.VIDEO)

    assert len(candidates) == 1
    cand = candidates[0]
    # Check all independent signals
    assert cand.signals.get("speaker_change") == 1.0
    assert cand.signals.get("visual_change") == 0.85
    assert cand.signals.get("ocr_change") == 0.80
    assert cand.signals.get("temporal_gap") is not None
    # Consensus boost
    assert cand.signals.get("multimodal_consensus", 0.0) >= 0.8
    assert "multimodal agreement" in cand.reason


def test_partial_modality_video_elements_work_gracefully():
    # Video with only speech and visual summary, but no OCR
    elements = [
        ContentElement(
            id="s1",
            kind="speech_segment",
            text="Introductory speech.",
            t0=0.0,
            t1=5.0,
            speaker="A",
            meta={"visual_summary": "intro scene"},
        ),
        ContentElement(
            id="s2",
            kind="speech_segment",
            text="Action explanation.",
            t0=5.5,
            t1=12.0,
            speaker="A",
            meta={"visual_summary": "demo screen"},
        ),
    ]
    graph = ContextGraph(elements)
    candidates = generate_candidates_for_region(elements, graph, modality=Modality.VIDEO)

    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.signals.get("visual_change") == 0.85
    assert "ocr_change" not in cand.signals


# ───────── 5. Conversational Knowledge Units ─────────


def test_classify_conversational_question_answer():
    elements = [
        ContentElement(
            id="q1",
            kind="speech_segment",
            text="What is the expected release date?",
            speaker="Alice",
        ),
        ContentElement(
            id="a1",
            kind="speech_segment",
            text="We are aiming for end of next month.",
            speaker="Bob",
        ),
    ]
    ku_type, meta = classify_conversational_elements(elements)
    assert ku_type == KnowledgeUnitType.QUESTION_ANSWER.value
    assert meta.get("conversational_shape") == "Q&A"


def test_classify_conversational_decision():
    elements = [
        ContentElement(
            id="d1",
            kind="speech_segment",
            text="After reviewing the proposal, we agreed to go with Option B.",
            speaker="Alice",
        ),
    ]
    ku_type, meta = classify_conversational_elements(elements)
    assert ku_type == KnowledgeUnitType.DECISION.value
    assert meta.get("has_decision") is True


def test_classify_conversational_action_item():
    elements = [
        ContentElement(
            id="ai1",
            kind="speech_segment",
            text="Action item: Bob will follow up with the infra team.",
            speaker="Alice",
        ),
    ]
    ku_type, meta = classify_conversational_elements(elements)
    assert ku_type == KnowledgeUnitType.ACTION_ITEM.value
    assert meta.get("has_action_item") is True


# ───────── 6. Adaptive Granularity & Hierarchical Units ─────────


def test_chunking_attaches_adaptive_granularity_metadata():
    elements = [
        ContentElement(id="h1", kind="heading", text="1. System Architecture", level=1),
        ContentElement(
            id="p1",
            kind="paragraph",
            text="The system employs a modular microservice pipeline.",
            parent_id="h1",
        ),
        ContentElement(
            id="p2", kind="paragraph", text="Each component scales independently.", parent_id="h1"
        ),
    ]
    ingest = _make_ingest(elements)
    graph = ContextGraph(elements)
    units, _ = chunk(ingest, graph)

    for u in units:
        assert u.metadata.get("granularity") == "adaptive"
        assert u.metadata.get("size_reason")
        assert u.knowledge_unit_type in {t.value for t in KnowledgeUnitType}


def test_hierarchical_knowledge_unit_linking():
    elements = [
        ContentElement(id="h1", kind="heading", text="1. Overview", level=1),
        ContentElement(id="p1", kind="paragraph", text="Long paragraph 1 " * 40, parent_id="h1"),
        ContentElement(id="p2", kind="paragraph", text="Long paragraph 2 " * 40, parent_id="h1"),
    ]
    ingest = _make_ingest(elements)
    graph = ContextGraph(elements)
    units, _ = chunk(ingest, graph)

    # When section splits into multiple units, parent-child links exist
    if len(units) > 1:
        parent = units[0]
        assert parent.child_ids
        for child_id in parent.child_ids:
            child = next(u for u in units if u.id == child_id)
            assert child.parent_id == parent.id
            assert any(
                r.type == RelationType.CHILD_OF and r.target_id == parent.id
                for r in child.relationships
            )


# ───────── 7. Context Completeness Scoring ─────────


def test_context_completeness_detects_missing_context():
    # An orphan unit with bare anaphora and no heading context
    orphan_unit = KnowledgeUnit(
        id="ku_test_orphan",
        content="This table clearly shows the former result as discussed above.",
        modality=Modality.DOCUMENT,
        context=Context(heading_path=[]),
        provenance=None,
        decision=None,
    )
    graph = ContextGraph([])
    completeness = evaluate_context_completeness(orphan_unit, graph)

    assert completeness.score < 1.0
    assert any("heading" in m for m in completeness.missing_dependencies)
    assert any("anaphoric" in m for m in completeness.missing_dependencies)


def test_context_completeness_enriches_self_contained_unit():
    situated_unit = KnowledgeUnit(
        id="ku_test_situated",
        content="Revenue increased by 15% across European regions in Q3.",
        modality=Modality.DOCUMENT,
        context=Context(heading_path=["Financial Performance", "Regional Results"]),
        provenance=None,
        decision=None,
    )
    graph = ContextGraph([])
    enrich_context_completeness(situated_unit, graph)

    assert situated_unit.context_completeness == 1.0
    assert not situated_unit.missing_context


# ───────── 8. Evaluation Framework Metrics ─────────


def test_evaluation_metrics_framework():
    elements = [
        ContentElement(id="h1", kind="heading", text="1. Executive Summary", level=1),
        ContentElement(
            id="p1", kind="paragraph", text="Company revenue reached record levels.", parent_id="h1"
        ),
    ]
    ingest = _make_ingest(elements)
    graph = ContextGraph(elements)
    units, _ = chunk(ingest, graph)

    coherence = boundary_coherence_score(units, graph)
    assert 0.0 <= coherence <= 1.0

    completeness = context_completeness_score(units, graph)
    assert 0.0 <= completeness <= 1.0

    preservation = relationship_preservation_rate(units, graph)
    assert 0.0 <= preservation <= 1.0

    frag = fragmentation_rate(units)
    assert 0.0 <= frag <= 1.0

    variance = chunk_size_variance(units)
    assert variance >= 0.0

    retrieval_res = retrieval_evaluation(
        units=units,
        queries=["What is the company revenue record?"],
        relevance_labels=[{units[0].id}],
        k=1,
    )
    assert retrieval_res["recall_at_k"] == 1.0
    assert retrieval_res["precision_at_k"] == 1.0
    assert retrieval_res["mrr"] == 1.0
