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
uv run cleave serve                # → http://127.0.0.1:8321
```

Or with Docker (set `CLEAVE_API_KEY` in `.env` first — the container listens on
0.0.0.0, and `cleave serve` refuses to do that without a key):

```bash
docker compose up                    # Cleave alone
docker compose --profile infra up    # + Qdrant, MinIO, Redpanda, Neo4j
```

## Make it fast (optional Rust core)

`rust/cleave_core` is a PyO3 extension with the text hot paths: tokenization,
an inverted-index BM25 with parallel batch queries, token counting, the
fixed-size baseline splitter, and MinHash/LSH near-duplicate detection.

```bash
uv run maturin develop -r -m rust/cleave_core/Cargo.toml   # needs a Rust toolchain
curl -s localhost:8321/health | jq .text_engine              # {"backend": "rust", ...}
```

It is optional and **result-identical**: every function has a pure-Python twin
in `cleave/_accel.py`, and parity tests fuzz both with Unicode-heavy input to
check the same tokens, bit-identical BM25 scores and the same MinHash groups.
`CLEAVE_NO_RUST=1` forces the Python paths. (A plain `uv sync` removes the
extension because it is not in the lockfile; re-run `maturin develop` after.)

| 20,000 docs × ~400 words, 2 cores | Python | Rust | |
|---|---|---|---|
| BM25 score, 1 query | 0.128 s | 0.001 s | **137×** |
| BM25, 200 queries top-10 | 18.7 s | 0.39 s | **48×** |
| near-duplicate groups (2,400 docs) | 64.4 s | 0.59 s | **109×** |
| approx token count | 2.37 s | 0.32 s | **7×** |
| BM25 index build | 7.72 s | 1.54 s | **5×** |
| tokenize corpus | 6.18 s | 1.90 s | **3×** |

`uv run python benchmarks/bench_text_engine.py` reproduces it; the parallel
paths scale with cores.

**Near-duplicate collapsing** is the feature this makes practical. After
chunking, units whose 5-word-shingle Jaccard similarity is ≥ 0.85 are tagged
(`metadata.duplicate_of` / `metadata.duplicates`): repeated boilerplate, a page
duplicated across PDFs, the same attachment uploaded twice. Nothing is deleted,
since every copy keeps its provenance, but search returns each text once
instead of filling the result list with copies. The job profile reports
`totals.near_duplicates`.

## Use it as a library or CLI

```python
from cleave import chunk_file

units = chunk_file("report.pdf")  # deterministic, no LLM by default
print(units[0].embed_text())
```

```bash
uv run cleave chunk report.pdf sales.csv -o units.jsonl
```

## Search it

Every job is searchable with **hybrid retrieval**: BM25 (exact ids like
`SO-10042`, section numbers) fused with dense vectors by reciprocal rank fusion,
an optional cross-encoder re-ranker (`CLEAVE_RERANKER_MODEL`), and **graph
expansion** — a matched row group comes back with its schema card, a figure
with its caption. Search keeps working (BM25 only) when no embedding model can
load.

```bash
curl "http://127.0.0.1:8321/api/search?q=budget+Engineering+Bengaluru"   # all jobs
curl "http://127.0.0.1:8321/api/search?q=SO-10042&jobs=<id1>,<id2>&k=10"
```

## Secure it

Unset, `CLEAVE_API_KEY` keeps the zero-config local experience. Set, every route
except `/health*` and `/static` requires it (`X-API-Key`, `Authorization:
Bearer`, or as the HTTP Basic password, so browsers get a login prompt). Job
submissions are rate-limited per client (`CLEAVE_RATE_LIMIT_PER_MIN`), at most
`CLEAVE_MAX_CONCURRENT_JOBS` run at once, and job state lives in
`data/jobs.sqlite3` — jobs interrupted by a restart come back as retryable
errors (`POST /api/jobs/{id}/retry`). `run_app_ngrok.py` refuses to open a
tunnel unless both `NGROK_AUTHTOKEN` and `CLEAVE_API_KEY` are set.

Upload a PDF/DOCX/MD, a CSV/XLSX, or an audio file (the latter needs the STT worker, see
below). Inspect the routing decision, the units, and each unit's receipt.

```bash
cd ~/PycharmProjects/STT && .venv/bin/python -m uvicorn --factory stt.server.app:app_factory --port 8000
```

## Measure it

```bash
uv run python -m cleave.evaluate tests/fixtures/executive_summary.pdf tests/fixtures/attention_paper.pdf
```

### Retrieval benchmark

Structure is a means; findable answers are the end. `cleave eval-retrieval` asks
questions whose answer must appear *together* in one retrieved chunk — for a
spreadsheet that includes the column name beside the value — and compares
Cleave against a fixed 512/64 splitter with the same retriever:

```bash
uv run cleave eval-retrieval benchmarks/qa_fixtures.jsonl          # BM25
uv run cleave eval-retrieval benchmarks/qa_fixtures.jsonl --dense  # + vectors
```

| 12 questions · CSV, XLSX, MD · BM25 | fixed 512/64 | Cleave |
|---|---|---|
| recall@1 | 0.667 | **0.917** |
| recall@3 | 0.667 | **1.000** |
| MRR | 0.667 | **0.958** |
| tokens in top-3 context | 912 | **823** |

CI runs it on every push and uploads the scorecard. Add your own QA lines to
grow it.

### Context preservation scorecard

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
| `cleave/api.py`, `cleave/cli.py` | Library entry points (`chunk_file`) and the `cleave` command |
| `cleave/eval_retrieval.py` | Retrieval recall/MRR benchmark vs the fixed-size baseline |
| `cleave/web/search.py` | Hybrid BM25 + dense retrieval, RRF, re-rank, graph expansion |
| `cleave/web/security.py` | API-key auth and job-submission rate limiting |
| `cleave/_accel.py` | Text hot paths: Rust (`cleave_core`) when built, identical pure-Python fallback |
| `rust/cleave_core/` | PyO3 extension: tokenizer, BM25, token counting, MinHash near-duplicates |
| `benchmarks/` | Retrieval QA set and the Python-vs-Rust speed benchmark |
| `cleave/storage/`, `cleave/events/` | Adapters: Qdrant is synced after each job; MinIO, Neo4j and Redpanda are **experimental** and not called by the pipeline yet |
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
