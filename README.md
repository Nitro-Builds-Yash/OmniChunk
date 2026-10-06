# Cleave / OmniChunk

**Understand information before you cut it.**

Cleave (OmniChunk) is a universal chunking and information-extraction engine. Most RAG pipelines start
by destroying information: convert the file to text, split every N tokens, hope for the
best. Tables lose their headers, captions lose their figures, and answers lose the
questions they were answering. Cleave profiles each input first, routes each region to the
right strategy, refuses to cut through relationships, and marks the chunks that genuinely
need AI enrichment — so intelligence is spent only where it changes the answer.

*The word "cleave" means both to split apart and to cling together. That is the product.*

📖 **[User Guide](USER_GUIDE.md)** — run it, and read what it shows you ·
🏗 **[Architecture](ARCHITECTURE.md)** — how it works inside ·
🔌 **[Contract](CONTRACT.md)** — plug in another modality

---

## 🎨 New UI & Typography Upgrades

The web interface now features an **Aurora Glass Design System** and dynamic customization directly in the header:

- **Typography System & Live Switcher:**
  - **Plus Jakarta Sans** (Modern tech sans-serif, default)
  - **Outfit** (Futuristic geometric display font for high-impact titles)
  - **Space Grotesk** (Cybernetic clean display style)
  - **JetBrains Mono** (Precision token metering and provenance hashes)
  - *Includes an in-header Font Switcher (`Jakarta` · `Outfit` · `Grotesk`) persisted in localStorage!*
- **Brand Identity Switcher:** Allows live switching between **OmniChunk**, **NexusChunk**, **SynapseSplit**, **AetherCleave**, or classic **Cleave**.
- **Aurora Glass Aesthetic:** Obsidian backdrop (`#070A12`) with ambient radial light leaks in electric cyan, indigo, and violet.
- **Glassmorphic Components:** Frosted glass cards (`backdrop-blur-2xl`), luminous 1px borders, and glowing button hover states.
- **Interactive Staging & Pipeline:** Drag & drop zone with modality badges (`PDF`, `DOCX`, `XLSX`, `CSV`, `Audio`, `Video`), multi-file queue, and animated multi-color progress pipeline.

---

## How it works

```
input ──▶ modality extraction ──▶ content elements ──▶ context graph
                                                            │
      knowledge units ◀── selective enrichment ◀── adaptive chunking ◀── profile & route
```

1. **Understand** — Docling parses documents into a typed structure (headings, paragraphs,
   tables with grids, figures, captions, reading order). Audio arrives as timestamped,
   speaker-attributed segments from a separate STT worker.
2. **Relate** — a context graph links captions to their floats (Docling refs, then bbox
   adjacency), resolves "Table 3"-style mentions to the actual table, and rebuilds heading
   ancestry from section numbering (Docling reports every heading as level 1, so *3.2.1*
   would otherwise look like a sibling of *3*). Every edge carries confidence and
   human-readable evidence.
3. **Route** — a deterministic profiler picks the strategy: timestamps → temporal; almost
   all content in tables → tabular; trustworthy hierarchy → structural; otherwise topic
   drift or paragraph packing. Tables and figures inside prose are carved out as atomic
   units with their captions, always.
4. **Cut without severing** — boundary candidates that would strand a heading or separate
   a caption from its figure are vetoed, and the veto is recorded. A table too big for one
   chunk splits by rows with its header repeated. Overflow beats severance.
5. **Escalate selectively** — cheap triggers (anaphora density, orphaned context,
   uncaptioned visuals) flag the minority of chunks whose meaning leans on absent context.
   Only those are candidates for LLM-written situating summaries.
   *Spreadsheets get their own treatment because a row without its header is data with no
   schema.* Each sheet is profiled into a **schema card** — column types, ranges, null
   counts, cardinality, inferred from the values — and the rows become groups that always
   repeat the header and never split mid-row. A 480-row CSV becomes 27 self-describing
   units and costs exactly one LLM call, because only the schema card needs one.
6. **Explain everything** — every knowledge unit carries content, heading path,
   relationships (with evidence), provenance, and a **decision receipt**: which strategy,
   why this boundary, which cuts were refused, what AI was used and what it cost.

## Run it

```bash
uv sync
uv run python -m cleave.app        # → http://127.0.0.1:8321
```

Upload a PDF/DOCX/MD, a CSV/XLSX, or an audio file (the latter needs the STT worker, see
below). Inspect the routing decision, the units, and each unit's receipt.

```bash
cd ~/PycharmProjects/STT && .venv/bin/python -m uvicorn --factory stt.server.app:app_factory --port 8000
```

## Measure it

```bash
uv run python -m cleave.evaluate tests/fixtures/executive_summary.pdf tests/fixtures/attention_paper.pdf
```

Writes `data/scorecard.json` — the Context Preservation Scorecard comparing a fixed
512-token/64-overlap baseline against Cleave on caption integrity, header integrity,
heading context, and reference resolution. The homepage renders whatever was measured;
no number in the UI is hand-written.

## Design positions (deliberate, evidence-based)

- **Fixed-size splitting is a strong baseline** (NAACL 2025, arXiv:2410.13070) and
  embedding-based "semantic chunking" often fails to beat it — so Cleave's flat-prose path
  is honest paragraph packing, and intelligence is spent on structure, relationships, and
  selective context instead.
- **No LLM in the core pipeline.** The document path is fully deterministic (tier 0);
  enrichment is opt-in and per-chunk, guided by measured signals — not applied uniformly
  the way contextual-retrieval implementations usually are.
- **One contract, many modalities.** `cleave/models.py` + [CONTRACT.md](CONTRACT.md)
  define the knowledge unit and element formats; the video pipeline (separate repository)
  plugs in by emitting the same shapes.

## Repository map

| Path | What lives there |
|---|---|
| `cleave/models.py` | The frozen data contract (elements, units, receipts) |
| `cleave/ingest_document.py` | Docling → content elements |
| `cleave/graph.py` | Context graph: captions, references, hierarchy, evidence |
| `cleave/router.py` | Profiler, routing cascade, veto-aware cut selection, escalation flags |
| `cleave/chunkers.py` | Structural / atomic / tabular / semantic / temporal strategies + unit assembly |
| `cleave/tabular.py` | Column type inference, schema cards, header-repeating row groups |
| `cleave/evaluate.py` | Fixed-size baseline + Context Preservation Scorecard |
| `cleave/app.py` | Composition root: builds the FastAPI app, mounts static, registers routers |
| `cleave/web/` | HTTP layer: job registry, upload safety, page and API routes, search |
| `cleave/pipeline.py` | Job orchestration (ingest → graph → chunk → enrich → artifacts) |
| `cleave/config.py` | Every environment variable, loaded from `.env` and validated once |
| `cleave/http.py` | One pooled HTTP client and one retry policy for every outbound call |
| `cleave/markdown.py` | Markdown table rendering shared by ingest, cleaning and chunking |
| `cleave/logging_setup.py` | Log format and per-job correlation, installed at startup |
| `cleave/ingest_audio.py` | Audio via the local STT worker |
| `cleave/ingest_contract.py` | Import from an external modality worker |
| `cleave/enrich.py`, `cleave/llm.py` | Selective LLM enrichment |
| `cleave/semantic.py` | Embedding topic-drift boundaries for flat prose |
| `tests/test_cleave.py` | Asserts the promises: captions never severed, headers always repeated |
| `tests/test_web.py` | Upload/artifact route safety, including path-traversal regressions |
| `tests/test_http.py`, `tests/test_config.py`, `tests/test_llm.py` | Retry policy, settings validation, provider selection |
