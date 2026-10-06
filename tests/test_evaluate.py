"""The scorecard's own arithmetic, tested without parsing a document.

These guard the headline number. If ``ArmScore`` averaged unmeasured metrics as
zero, or the baseline splitter dropped text, the comparison the whole project
rests on would be wrong in a way no chunking test would notice.
"""

from __future__ import annotations

from cleave.evaluate import FIXED_OVERLAP, FIXED_TOKENS, ArmScore, Tally, fixed_chunks, norm
from cleave.ingest_document import IngestResult
from cleave.models import ContentElement, _encoder


def _ingest(text: str) -> IngestResult:
    return IngestResult(
        elements=[ContentElement(id="el_0000", kind="paragraph", text=text)],
        title="t",
        source_uri="t",
        sha256=None,
        warnings=[],
        cleaning=None,
    )


def test_the_baseline_splitter_overlaps_and_drops_nothing():
    enc = _encoder()
    words = " ".join(f"word{i}" for i in range(2000))
    chunks = fixed_chunks(_ingest(words))

    assert len(chunks) > 1
    ids = enc.encode(words, disallowed_special=())
    # Every token appears somewhere, and consecutive chunks share the overlap.
    assert enc.decode(ids[:FIXED_TOKENS]) == chunks[0]
    second_start = FIXED_TOKENS - FIXED_OVERLAP
    assert enc.decode(ids[second_start : second_start + FIXED_TOKENS]) == chunks[1]


def test_a_short_document_is_one_baseline_chunk():
    assert len(fixed_chunks(_ingest("just a little text"))) == 1


def test_an_empty_document_produces_no_chunks():
    assert fixed_chunks(_ingest("")) == []


def test_unmeasured_metrics_do_not_drag_the_score_down():
    """A metric with nothing to measure must be excluded, not counted as zero."""
    score = ArmScore()
    score.caption.add(True)
    score.caption.add(True)  # 2/2 measured; the other three are empty

    out = score.to_dict()

    assert out["cps_pct"] == 100.0
    assert {m["name"] for m in out["metrics"]} == {
        "caption integrity",
        "header integrity",
        "heading context",
        "resolved references",
    }


def test_a_score_with_nothing_measured_is_none():
    assert ArmScore().to_dict()["cps_pct"] is None


def test_metrics_average_as_ratios_not_as_raw_counts():
    """1/1 and 1/100 must average to ~50%, not to 2/101."""
    score = ArmScore()
    score.caption.add(True)
    score.header.add(True)
    for _ in range(99):
        score.header.add(False)

    assert score.to_dict()["cps_pct"] == 50.5


def test_tally_counts_both_outcomes():
    t = Tally()
    t.add(True)
    t.add(False)
    assert (t.preserved, t.total) == (1, 2)


def test_norm_collapses_punctuation_and_case():
    assert norm("Table 3: Results!") == norm("table  3   results")
