"""Tests for late-fusion multimodal chunking."""

from cleave.chunkers_multimodal import chunk_multimodal_stream
from cleave.graph import ContextGraph
from cleave.models import (
    ContentElement,
    KnowledgeUnitType,
    Modality,
    Provenance,
)


def test_chunk_multimodal_stream_aligns_speech_and_visuals():
    elements = [
        ContentElement(
            id="v_vis_1",
            kind="visual_event",
            text="Slide 1: Executive Overview with system architecture diagram.",
            t0=0.0,
            t1=15.0,
            meta={"entities": ["Architecture", "System"]},
        ),
        ContentElement(
            id="v_speech_1",
            kind="speech_segment",
            text="Welcome everyone, today we'll review the architecture.",
            t0=1.0,
            t1=7.0,
            speaker="Alice",
        ),
        ContentElement(
            id="v_speech_2",
            kind="speech_segment",
            text="As you can see, the distributed bus handles incoming streams.",
            t0=7.5,
            t1=14.0,
            speaker="Alice",
        ),
        ContentElement(
            id="v_vis_2",
            kind="visual_event",
            text="Slide 2: Benchmarks and Evaluation Metrics.",
            t0=15.0,
            t1=30.0,
            meta={"entities": ["Benchmarks", "Metrics"]},
        ),
        ContentElement(
            id="v_speech_3",
            kind="speech_segment",
            text="Moving on to the benchmarks, latency improved by 40%.",
            t0=16.0,
            t1=25.0,
            speaker="Bob",
        ),
    ]

    graph = ContextGraph(elements)
    counter = 0

    def new_unit_id():
        nonlocal counter
        uid = f"ku_{counter:04d}"
        counter += 1
        return uid

    def base_provenance(el):
        return Provenance(source_uri="meeting.mp4")

    units = chunk_multimodal_stream(
        elements, graph, new_unit_id, base_provenance, title="Meeting Recording"
    )

    assert len(units) == 2

    # First unit covers Slide 1 and Alice's dialogue
    u1, members1 = units[0]
    # New universal boundary chunker emits VIDEO_EVENT for multimodal windows
    assert u1.knowledge_unit_type in (
        KnowledgeUnitType.MULTIMODAL_EVENT.value,
        KnowledgeUnitType.VIDEO_EVENT.value,
    )
    assert u1.modality == Modality.VIDEO
    assert u1.temporal.start_s == 0.0
    assert u1.temporal.end_s == 15.0
    assert "Alice" in (u1.temporal.speaker or "")
    assert "Slide 1" in u1.content
    assert "Welcome everyone" in u1.content
    assert len(members1) == 3

    # Second unit covers Slide 2 and Bob's dialogue
    u2, _members2 = units[1]
    assert u2.temporal.start_s == 15.0
    assert u2.temporal.end_s == 30.0
    assert "Bob" in (u2.temporal.speaker or "")
    assert "Slide 2" in u2.content
    assert "Moving on to the benchmarks" in u2.content
