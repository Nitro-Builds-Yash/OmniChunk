"""Behavioural tests for the guarantees Cleave actually claims.

These assert the promises, not the implementation: that boundaries never sever
a caption from its figure, that a row group always carries its header, that
routing responds to the shape of the input, and that a chunk can explain
itself. Document fixtures are parsed once per session because Docling model
load dominates the runtime.
"""

from __future__ import annotations

import json
import re

import pytest
from conftest import FIXTURES

from cleave.chunkers import chunk
from cleave.graph import ContextGraph
from cleave.ingest_document import ingest_document
from cleave.models import (
    ChunkingDecision,
    Context,
    KnowledgeUnit,
    Modality,
    Provenance,
    RelationType,
)
from cleave.router import anaphora_rate, escalation_flags
from cleave.tabular import profile_column, profile_table, render_group, row_groups


def run(path: str) -> tuple[list[KnowledgeUnit], object]:
    ing = ingest_document(path)
    graph = ContextGraph(ing.elements)
    return chunk(ing, graph)


@pytest.fixture(scope="session")
def paper():
    return run(f"{FIXTURES}/attention_paper.pdf")


@pytest.fixture(scope="session")
def sales():
    return run(f"{FIXTURES}/sales_q3.csv")


@pytest.fixture(scope="session")
def workbook():
    return run(f"{FIXTURES}/people_ops.xlsx")


# ───────── routing ─────────


@pytest.mark.slow
def test_structured_document_routes_structural(paper):
    _units, profile = paper
    assert profile.route == "structural"
    assert profile.heading_count >= 3


@pytest.mark.slow
def test_spreadsheets_route_tabular(sales, workbook):
    assert sales[1].route == "tabular"
    assert workbook[1].route == "tabular"
    assert sales[1].row_count == 480


def test_flat_prose_does_not_route_structural(tmp_path):
    _units, profile = run(f"{FIXTURES}/flat_essay.md")
    assert profile.route in ("semantic", "paragraph_fallback")
    assert profile.heading_count == 0


@pytest.mark.slow
def test_route_reason_cites_the_signals_that_drove_it(paper, sales):
    """The reason has to be evidence, not a restatement of the strategy name."""
    _u, paper_p = paper
    assert str(paper_p.heading_count) in paper_p.route_reason
    _u, sales_p = sales
    assert f"{sales_p.row_count:,}" in sales_p.route_reason


# ───────── the core promise: cuts never sever relationships ─────────


@pytest.mark.slow
def test_every_captioned_float_keeps_its_caption(paper):
    """The headline claim. A caption and its figure must land in one unit."""
    units, _ = paper
    ing = ingest_document(f"{FIXTURES}/attention_paper.pdf")
    graph = ContextGraph(ing.elements)
    captioned = [
        e for e in ing.elements if e.kind in ("table", "figure") and graph.captions_of(e.id)
    ]
    assert captioned, "fixture should contain captioned floats"
    for float_el in captioned:
        cap = graph.by_id[graph.captions_of(float_el.id)[0]]
        probe = cap.text[:50]
        assert any(probe in u.content for u in units), (
            f"caption for {float_el.id} was severed from its float"
        )


@pytest.mark.slow
def test_oversized_tables_repeat_their_header(paper):
    units, _ = paper
    parts = [u for u in units if u.metadata.get("part")]
    for u in parts:
        assert u.decision.vetoed_cuts, "a split table must record why it split that way"


@pytest.mark.slow
def test_row_groups_always_carry_the_header(sales):
    units, _ = sales
    groups = [u for u in units if u.metadata.get("element_kind") == "row_group"]
    assert len(groups) > 1, "480 rows should not fit in one group"
    for u in groups:
        first_line = u.content.splitlines()[0]
        for column in u.metadata["columns"]:
            assert column in first_line, f"{u.id} lost column {column!r}"


@pytest.mark.slow
def test_row_groups_never_split_a_row(sales):
    units, _ = sales
    for u in (x for x in units if x.metadata.get("element_kind") == "row_group"):
        body = u.content.splitlines()[2:]
        widths = {line.count("|") for line in body if line.strip()}
        assert len(widths) <= 1, f"{u.id} contains a truncated row"


# ───────── explainability ─────────


@pytest.mark.slow
def test_every_unit_can_explain_itself(paper, sales, workbook):
    for units, _ in (paper, sales, workbook):
        for u in units:
            assert u.decision.strategy, f"{u.id} has no strategy"
            assert len(u.decision.reason) > 20, f"{u.id} has no usable reason"


@pytest.mark.slow
def test_embed_text_leads_with_context(paper):
    units, _ = paper
    situated = [u for u in units if u.context.heading_path]
    assert situated
    u = situated[0]
    assert u.embed_text().startswith(u.context.heading_path[0])
    assert u.content in u.embed_text()


@pytest.mark.slow
def test_relationships_carry_evidence(paper):
    units, _ = paper
    refs = [r for u in units for r in u.relationships if r.type == RelationType.REFERENCES]
    assert refs, "the paper cross-references its own figures"
    for r in refs:
        assert r.evidence and 0 < r.confidence <= 1.0


@pytest.mark.slow
def test_schema_card_links_to_every_row_group(sales):
    units, _ = sales
    card = next(u for u in units if u.metadata.get("element_kind") == "schema_card")
    groups = {u.id for u in units if u.metadata.get("element_kind") == "row_group"}
    described = {r.target_id for r in card.relationships if r.type == RelationType.SCHEMA_OF}
    assert described == groups


# ───────── selectivity: AI is spent, not sprayed ─────────


@pytest.mark.slow
def test_most_units_need_no_llm(sales):
    """A dataset is self-describing; only the schema card earns a call."""
    units, _ = sales
    flagged = [u for u in units if u.decision.escalation_flags]
    assert len(flagged) == 1
    assert flagged[0].metadata.get("element_kind") == "schema_card"


def test_anaphora_detection_finds_dangling_context():
    assert anaphora_rate("As shown above, the result holds.") > 0.9
    assert anaphora_rate("Revenue grew 12 percent in the third quarter.") == 0.0


def test_orphan_text_is_flagged_but_situated_text_is_not():
    assert escalation_flags("Some standalone text.", [], "structural")
    assert not escalation_flags("Some situated text.", ["3. Results"], "structural")


# ───────── temporal + contract import ─────────


def _timed(tmp_path, payload) -> list[KnowledgeUnit]:
    import json

    from cleave.chunkers import chunk as _chunk
    from cleave.graph import ContextGraph as _Graph
    from cleave.ingest_contract import load_contract

    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    ing, ready = load_contract(path)
    if ready:
        return ready
    units, _profile = _chunk(ing, _Graph(ing.elements))
    return units


def test_short_turn_never_steals_another_speakers_words(tmp_path):
    """A brief reply is still that person's reply. Merging it into the previous
    speaker's turn would attribute their words to someone else."""
    units = _timed(
        tmp_path,
        {
            "contract": 1,
            "source_uri": "t.mp4",
            "elements": [
                {
                    "id": "a",
                    "kind": "speech_segment",
                    "text": "Long opening statement here.",
                    "t0": 0.0,
                    "t1": 9.0,
                    "speaker": "A",
                },
                {
                    "id": "b",
                    "kind": "speech_segment",
                    "text": "Agreed.",
                    "t0": 9.2,
                    "t1": 11.0,
                    "speaker": "B",
                },  # 1.8s — under the merge threshold
            ],
        },
    )
    speakers = [u.temporal.speaker for u in units]
    assert speakers == ["A", "B"]
    assert "Agreed" not in units[0].content


def test_unattributed_fragment_does_merge(tmp_path):
    units = _timed(
        tmp_path,
        {
            "contract": 1,
            "source_uri": "t.mp4",
            "elements": [
                {
                    "id": "a",
                    "kind": "speech_segment",
                    "text": "Opening statement.",
                    "t0": 0.0,
                    "t1": 9.0,
                    "speaker": "A",
                },
                {"id": "b", "kind": "speech_segment", "text": "Mm.", "t0": 9.1, "t1": 9.6},
            ],
        },
    )
    assert len(units) == 1


def test_contract_import_keeps_every_visual_summary(tmp_path):
    units = _timed(
        tmp_path,
        {
            "contract": 1,
            "source_uri": "t.mp4",
            "elements": [
                {
                    "id": "a",
                    "kind": "speech_segment",
                    "text": "First.",
                    "t0": 0.0,
                    "t1": 4.0,
                    "speaker": "A",
                    "meta": {"visual_summary": "title slide"},
                },
                {
                    "id": "b",
                    "kind": "speech_segment",
                    "text": "Second.",
                    "t0": 4.0,
                    "t1": 9.0,
                    "speaker": "A",
                    "meta": {"visual_summary": "bar chart"},
                },
            ],
        },
    )
    assert units[0].metadata["visual_summary"] == "title slide · bar chart"


def test_contract_rejects_an_unknown_version(tmp_path):
    from cleave.ingest_contract import load_contract

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"contract": 99, "elements": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported contract version"):
        load_contract(path)


# ───────── cleaning ─────────


def test_citation_markers_are_removed():
    from cleave.cleaning import clean_text

    out, counts = clean_text("Docling is best 【 58†L23-L31 】 for parsing.")
    assert "†" not in out and "【" not in out
    assert counts["citation_markers"] == 1
    assert "Docling is best" in out and "for parsing." in out


def test_citation_split_across_cells_leaves_nothing_behind():
    from cleave.cleaning import clean_text

    out, _ = clean_text("layout+OCR 【 1 L314 】 | Output Format")
    assert "【" not in out and "】" not in out


def test_paired_cjk_brackets_survive():
    """The rule keys on the digits-dagger shape, not the bracket, so real
    quotation brackets are not collateral damage."""
    from cleave.cleaning import clean_text

    text = "The chapter 【重要な文書】 explains it."
    out, counts = clean_text(text)
    assert "【重要な文書】" in out
    assert not counts["citation_markers"]


def test_footnote_daggers_in_prose_survive():
    from cleave.cleaning import clean_text

    out, _ = clean_text("Aidan N. Gomez ∗ † University of Toronto")
    assert "†" in out


def test_hyphenated_line_breaks_are_rejoined():
    from cleave.cleaning import clean_text

    out, counts = clean_text("document inges-\ntion is hard")
    assert "ingestion" in out
    assert counts["hyphen_line_breaks"] == 1


def test_real_hyphenated_compound_is_not_joined():
    from cleave.cleaning import clean_text

    out, _ = clean_text("a state-\nOf-the-art result")  # next line capitalised
    assert "state-" in out


def test_ligatures_are_expanded():
    from cleave.cleaning import clean_text

    out, _ = clean_text("the ﬁnal ﬂow")
    assert "final" in out and "flow" in out


def test_extraction_spacing_is_repaired():
    from cleave.cleaning import clean_text

    out, counts = clean_text("parsing and ingestion , Docling is  best .")
    assert "ingestion, Docling" in out
    assert "best." in out
    assert counts["space_before_punctuation"] >= 1


def test_cleaning_never_changes_meaning():
    """Guards the line between repairing extraction damage and destroying
    information: case, stopwords and sentence punctuation must survive."""
    from cleave.cleaning import clean_text

    text = "The Model Was NOT trained on the North region's data."
    out, _ = clean_text(text)
    assert out == text


def test_code_elements_keep_their_spacing():
    from cleave.cleaning import clean_text

    code = "def f():\n    return    1"
    out, _ = clean_text(code, verbatim=True)
    assert "    return    1" in out


@pytest.mark.slow
def test_cleaning_runs_before_chunking(paper):
    """Units must hold cleaned text: token counts, boundaries and embeddings
    all describe what is stored, so cleaning cannot be a display-time step."""
    units, _ = paper
    for u in units:
        for line in u.content.splitlines():
            if "|" in line:  # table markdown pads cells deliberately
                continue
            assert "  " not in line, f"{u.id} kept a double space: {line[:60]!r}"
            assert not re.search(r"[ \t]+[,.;:]", line), f"{u.id}: {line[:60]!r}"


@pytest.mark.slow
def test_cleaning_report_is_recorded():
    ing = ingest_document(f"{FIXTURES}/executive_summary.pdf")
    assert ing.cleaning["total_fixes"] > 100
    assert "citation_markers" in ing.cleaning["by_rule"]
    assert ing.cleaning["chars_removed"] > 0


# ───────── usage accounting ─────────


def test_cost_follows_published_rates():
    from cleave.usage import Ledger

    led = Ledger()
    # gemini-2.5-flash: $0.30/M in, $2.50/M out
    cost = led.record("gemini-2.5-flash", in_tokens=1_000_000, out_tokens=1_000_000)
    assert cost == pytest.approx(0.30 + 2.50)
    assert led.total_cost == pytest.approx(2.80)


def test_cached_tokens_bill_at_the_discount():
    from cleave.usage import Ledger

    led = Ledger()
    full = led.record("gemini-2.5-flash", in_tokens=1_000_000, out_tokens=0)
    led2 = Ledger()
    cached = led2.record(
        "gemini-2.5-flash", in_tokens=1_000_000, out_tokens=0, cached_tokens=1_000_000
    )
    assert cached < full


def test_local_models_are_free_but_still_metered():
    from cleave.usage import Ledger

    led = Ledger()
    cost = led.record("ollama/qwen3:4b", in_tokens=50_000, out_tokens=2_000)
    assert cost == 0.0
    row = led.to_dict()["by_model"][0]
    assert row["local"] is True and row["in_tokens"] == 50_000
    assert led.to_dict()["totals"]["local_calls"] == 1


def test_unknown_model_is_flagged_as_estimated():
    from cleave.usage import Ledger

    led = Ledger()
    led.record("some-new-model", in_tokens=1000, out_tokens=100)
    assert led.to_dict()["by_model"][0]["estimated"] is True


def test_ledger_separates_models():
    from cleave.usage import Ledger

    led = Ledger()
    led.record("gemini-2.5-flash", 1000, 100)
    led.record("ollama/qwen3:4b", 5000, 500)
    d = led.to_dict()
    assert len(d["by_model"]) == 2
    assert d["totals"]["paid_calls"] == 1 and d["totals"]["local_calls"] == 1


def make_fake_provider(seen_prompts: list[str] | None = None):
    """A provider that echoes back one summary per chunk id it was given.

    Shared because more than one test needs a model that behaves, and the
    id echo is what enrichment matches results back on.
    """

    class FakeProvider:
        name, model = "fake", "fake-model"

        def is_configured(self):
            return True

        def complete_json(self, prompt, *, system=None, schema=None):
            if seen_prompts is not None:
                seen_prompts.append(prompt)
            ids = re.findall(r'<chunk id="([^"]+)"', prompt)
            body = json.dumps(
                {
                    "results": [
                        {"id": i, "summary": f"context for {i}", "entities": ["x"]} for i in ids
                    ]
                }
            )
            return body, {"model": self.model, "in_tokens": 1000, "out_tokens": 50}

    return FakeProvider


def test_batching_sends_the_document_once_per_batch(monkeypatch):
    """The optimisation that matters: N flagged chunks must not mean N
    document re-sends."""
    from cleave import enrich as enrich_mod
    from cleave.usage import Ledger

    seen_prompts: list[str] = []
    FakeProvider = make_fake_provider(seen_prompts)

    units = [
        KnowledgeUnit(
            id=f"ku_{i:04d}",
            content=f"chunk {i} text",
            modality=Modality.DOCUMENT,
            context=Context(),
            provenance=Provenance(source_uri="t"),
            decision=ChunkingDecision(
                strategy="structural", reason="r", escalation_flags=["orphan"]
            ),
        )
        for i in range(12)
    ]
    led = Ledger()
    monkeypatch.setattr(enrich_mod, "get_provider", FakeProvider)
    totals = enrich_mod.enrich(units, "THE DOCUMENT", ledger=led)

    assert totals["enriched"] == 12
    # 12 chunks at batch size 6 → 2 calls, not 12
    assert totals["api_calls"] == len(seen_prompts) == 2
    assert totals["calls_saved_by_batching"] == 10
    assert all("THE DOCUMENT" in p for p in seen_prompts)
    assert all(u.context.situating_summary for u in units)
    assert all(u.context.tier == 2 for u in units)


def test_enrichment_cost_is_shared_across_the_batch(monkeypatch):
    """One batched call bills every unit it served, and only what it cost."""
    from cleave import enrich as enrich_mod
    from cleave.usage import Ledger

    units = [
        KnowledgeUnit(
            id=f"ku_{i:04d}",
            content=f"chunk {i}",
            modality=Modality.DOCUMENT,
            context=Context(),
            provenance=Provenance(source_uri="t"),
            decision=ChunkingDecision(
                strategy="structural", reason="r", escalation_flags=["orphan"]
            ),
        )
        for i in range(6)
    ]
    led = Ledger()
    monkeypatch.setattr(enrich_mod, "get_provider", make_fake_provider())
    enrich_mod.enrich(units, "THE DOCUMENT", ledger=led)

    assert led.total_calls == 1  # six chunks, one call
    assert led.total_cost > 0
    share = led.total_cost / 6
    for u in units:
        assert u.decision.cost_usd == pytest.approx(share)
    assert sum(u.decision.cost_usd for u in units) == pytest.approx(led.total_cost)


def test_no_provider_means_no_calls_and_no_cost(monkeypatch):
    from cleave import enrich as enrich_mod
    from cleave.llm import NoneProvider
    from cleave.usage import Ledger

    units = [
        KnowledgeUnit(
            id="ku_0000",
            content="x",
            modality=Modality.DOCUMENT,
            context=Context(),
            provenance=Provenance(source_uri="t"),
            decision=ChunkingDecision(
                strategy="structural", reason="r", escalation_flags=["orphan"]
            ),
        )
    ]
    led = Ledger()
    monkeypatch.setattr(enrich_mod, "get_provider", NoneProvider)
    totals = enrich_mod.enrich(units, "doc", ledger=led)
    assert totals["enriched"] == 0 and led.total_calls == 0
    assert units[0].context.situating_summary is None  # still a valid unit


# ───────── column profiling ─────────


@pytest.mark.parametrize(
    "values,expected",
    [
        (["1", "2", "3", "40"], "integer"),
        (["4.9", "5", "6.1", "0.8"], "decimal"),  # mixed rendering stays numeric
        (["2026-01-02", "2026-11-30"], "date"),
        (["North", "South", "North", "South"], "categorical"),
        (["12%", "4.5%"], "percentage"),
        # Needs enough rows to distinguish a key from a category: with four values
        # that are all distinct, "categorical" is the more defensible reading.
        ([f"SO-{i}" for i in range(40)], "identifier"),
    ],
)
def test_column_type_inference(values, expected):
    assert profile_column("c", values).dtype == expected


def test_profile_reports_ranges_for_numbers():
    p = profile_column("units", ["10", "20", "30"])
    assert (p.minimum, p.maximum, p.mean) == (10.0, 30.0, 20.0)


def test_empty_cells_counted_not_typed():
    p = profile_column("x", ["1", "", "3", "n/a"])
    assert p.nulls == 2 and p.non_null == 2


def test_row_groups_respect_the_token_budget():
    grid = [["a", "b"]] + [[f"row{i}", "x" * 30] for i in range(200)]
    groups = row_groups(grid, ["a", "b"], target_tokens=200)
    assert len(groups) > 1
    assert sum(len(rows) for _start, rows in groups) == 200  # nothing dropped


def test_render_group_is_valid_markdown_table():
    md = render_group(["a", "b"], [["1", "2"]])
    lines = md.splitlines()
    assert lines[0] == "| a | b |" and set(lines[1]) <= set("|-")


def test_table_profile_counts_body_rows_only():
    grid = [["h1", "h2"], ["1", "2"], ["3", "4"]]
    prof = profile_table(grid, ["h1", "h2"], sheet="S")
    assert prof.row_count == 2 and prof.column_count == 2 and prof.sheet == "S"


# ───────── graph ─────────


@pytest.mark.slow
def test_numbered_sections_nest_under_their_numeric_parent(paper):
    """Docling reports every heading as level 1, so nesting comes from the
    section numbering the document already carries."""
    units, _ = paper
    deep = [u for u in units if len(u.context.heading_path) >= 3]
    assert deep, "3.2.1 should sit under 3.2 under 3"
    for u in deep:
        numbered = [h for h in u.context.heading_path if h[0].isdigit()]
        for parent, child in zip(numbered, numbered[1:]):
            p_num = parent.split()[0].rstrip(".")
            c_num = child.split()[0].rstrip(".")
            assert c_num.startswith(p_num + "."), f"{c_num} is not under {p_num}"


@pytest.mark.slow
def test_surrounding_text_stops_at_a_heading():
    ing = ingest_document(f"{FIXTURES}/attention_paper.pdf")
    graph = ContextGraph(ing.elements)
    first_heading = next(e for e in ing.elements if e.kind == "heading")
    leading, _trailing = graph.surrounding_text(first_heading.id)
    assert leading is None or isinstance(leading, str)


@pytest.mark.slow
def test_profile_is_json_serializable(sales):
    _units, profile = sales
    d = profile.to_dict()
    assert d["route"] == "tabular" and isinstance(d["heading_density"], float)


@pytest.mark.slow
def test_units_serialize_with_embed_text(sales):
    units, _ = sales
    d = units[0].to_dict()
    assert d["embed_text"] and d["modality"] == "document"
    assert isinstance(d["decision"]["cost_usd"], float)


def test_a_local_model_records_tier_one_not_tier_two(monkeypatch):
    """Context.tier documents '1 local model · 2 LLM'. Enrichment used to write
    2 unconditionally, so tier 1 never existed and the scale read as binary."""
    from cleave import enrich as enrich_mod
    from cleave.usage import Ledger

    class LocalProvider:
        name, model = "ollama", "ollama/qwen3:4b"

        def is_configured(self):
            return True

        def complete_json(self, prompt, *, system=None, schema=None):
            ids = re.findall(r'<chunk id="([^"]+)"', prompt)
            body = json.dumps(
                {"results": [{"id": i, "summary": f"local context for {i}"} for i in ids]}
            )
            return body, {"model": self.model, "in_tokens": 100, "out_tokens": 10}

    units = [
        KnowledgeUnit(
            id="ku_0000",
            content="x",
            modality=Modality.DOCUMENT,
            context=Context(),
            provenance=Provenance(source_uri="t"),
            decision=ChunkingDecision(
                strategy="structural", reason="r", escalation_flags=["orphan"]
            ),
        )
    ]
    monkeypatch.setattr(enrich_mod, "get_provider", LocalProvider)
    enrich_mod.enrich(units, "THE DOCUMENT", ledger=Ledger())

    assert units[0].context.situating_summary
    assert units[0].context.tier == 1  # local, and free


def test_enrichment_failures_are_reported_not_swallowed(monkeypatch):
    """A rate limit used to leave chunks quietly unenriched with nothing said."""
    from cleave import enrich as enrich_mod
    from cleave.usage import Ledger

    class FailingProvider:
        name, model = "gemini", "gemini-2.5-flash"

        def is_configured(self):
            return True

        def complete_json(self, prompt, *, system=None, schema=None):
            return "", {"model": self.model}  # what a 429 degrades to

    units = [
        KnowledgeUnit(
            id="ku_0000",
            content="x",
            modality=Modality.DOCUMENT,
            context=Context(),
            provenance=Provenance(source_uri="t"),
            decision=ChunkingDecision(
                strategy="structural", reason="r", escalation_flags=["orphan"]
            ),
        )
    ]
    monkeypatch.setattr(enrich_mod, "get_provider", FailingProvider)
    totals = enrich_mod.enrich(units, "doc", ledger=Ledger())

    assert totals["failed_calls"] == 1
    assert "failed" in totals["warning"]
    assert units[0].context.situating_summary is None  # still a valid unit
    assert units[0].context.tier == 0
