# Cleave — Architecture

How the system is put together, what happens to a file as it moves through it, and why each
mechanism works the way it does. For running it, see [USER_GUIDE.md](USER_GUIDE.md).

---

## 1. The problem this shape solves

A retrieval pipeline usually begins by destroying information. It converts a file to a
single string, splits every N tokens, and embeds the pieces. Everything that made the
document navigable — that this paragraph sits under that heading, that this caption belongs
to that figure, that this row is described by that header — is discarded before the first
embedding is computed.

Cleave inverts the order. **Understand first, then cut.** Structure is extracted, the
relationships between elements are made explicit, and only then are boundaries chosen — by
a policy that knows what it would be severing.

The design commits to three positions:

1. **The cut is a decision, and decisions should be auditable.** Every chunk records why it
   exists and which cuts were refused.
2. **Different inputs deserve different policies.** A spreadsheet and a research paper share
   no useful chunking strategy. Routing is per input, not global.
3. **AI is expensive and should be aimed.** Deterministic analysis finds where context is
   missing; a model is called only there.

A note on the third point, because it is easy to over-claim: published work is honest that
embedding-based "semantic chunking" often fails to beat fixed-size splitting
([arXiv:2410.13070](https://arxiv.org/abs/2410.13070)), and that LLM-driven boundary
selection is enormously expensive for marginal gain
([arXiv:2606.00881](https://arxiv.org/abs/2606.00881)). Cleave's value is not "cleverer
splitting." It is *structure preservation* plus *targeted enrichment*.

---

## 2. Pipeline

```
                    ┌──────────────────────────────────────────────┐
   INPUT            │  PDF · DOCX · PPTX · MD · HTML               │
                    │  CSV · XLSX                                  │
                    │  MP3 · M4A · WAV        JSON (contract)      │
                    └───────────────────┬──────────────────────────┘
                                        │  dispatch on extension
      ┌─────────────────────────────────┼──────────────────────────────┐
      ▼                                 ▼                              ▼
 ingest_document.py              ingest_audio.py              ingest_contract.py
 Docling → typed tree            HTTP → STT worker            external worker payload
      └─────────────────────────────────┼──────────────────────────────┘
                                        ▼
                          list[ContentElement]        ← universal representation
                                        ▼
                    cleaning.py ── normalise BEFORE anything measures or splits
                    citations · ligatures · hyphen breaks · spacing
                                        ▼
                    graph.py  ──  ContextGraph (networkx)
                    hierarchy · captions · references · reading order
                                        ▼
                    router.py ──  build_profile()  → cheap deterministic signals
                                  _route()         → strategy for this input
                                        ▼
                    chunkers.py ── strategy dispatch + choose_cut() vetoes
                    atomic │ structural │ tabular │ semantic │ temporal
                                        ▼
                    enrich.py ──  escalation flags → LLM, only where flagged
                                        ▼
                          list[KnowledgeUnit]
                                        ▼
              units.json · graph.json · profile.json  →  API · UI · retrieval
```

---

## 3. Layers

### 3.1 Ingestion — `ingest_document.py`, `ingest_audio.py`, `ingest_contract.py`

Each modality has one job: produce `ContentElement`s. Nothing downstream knows or cares
where they came from, which is what lets a separately-built video pipeline join later
without touching the core.

**Documents** go through Docling, which supplies a typed tree rather than a text blob:
headings with levels, paragraphs, tables with cell grids, figures, captions with references
back to their float, and page/bbox provenance for every item. Cleave flattens that tree
while preserving heading ancestry, using a stack that pops when a heading of equal or
shallower level appears.

Two Docling behaviours are worked around explicitly:

- **Sheet names are dropped.** An XLSX sheet becomes a table with a page number and nothing
  else. `_label_sheets()` reads the workbook's sheet names with openpyxl and maps page *N*
  to sheet *N−1*, because a row group that cannot say which sheet it came from is close to
  useless.
- **TableFormer is CPU-only on Apple Silicon.** Docling rewrites MPS to CPU for table
  structure (the upstream fix was rejected, not merged). The layout model does use MPS.
  `AcceleratorOptions(num_threads=10)` is set because the default of 4 leaves most of a
  14-core machine idle.

**Audio** is a single HTTP call to a separate STT service. This is a deliberate process
boundary, not indecision: Docling pins `transformers<5.9` on macOS while the MLX audio
stack requires `>=5.14`, so they cannot share a virtualenv. Splitting them across a request
turns a dependency conflict into an interface — and makes the modality workers independently
scalable, which is a better answer than a monolith would have given anyway.

**Contract** payloads come from external workers per [CONTRACT.md](CONTRACT.md), at either
of two levels: elements (Cleave then does graph, routing and chunking) or finished knowledge
units (rendered as-is). IDs are prefixed on import so two workers can never collide.

### 3.2 Cleaning — `cleaning.py`

Extraction leaves debris, and it is not free. The sample report arrives carrying 203
reference markers like `【32†L355-L364】` — roughly 1,200 tokens that mean nothing outside
the tool that wrote them, would be paid for on every enrichment call, and would sit in every
vector. Alongside those: ligatures a PDF encodes as single glyphs (`ﬁnance` is not
`finance` to a matcher), words hyphenated across line breaks, non-breaking spaces, and
spacing damage like `ingestion , Docling`.

Normalisation runs **inside ingestion, before anything measures or splits**. That ordering
is the point: token counts, routing signals, boundary choices and embeddings must all
describe the text that will actually be stored.

The rules are deliberately conservative, and the line is explicit: **repair what the
extractor damaged, remove what was never in the document, change nothing that carries
meaning.** No lowercasing, no stopword removal, no stemming, no punctuation stripping —
those improve a metric by destroying information, which is the opposite of this pipeline's
purpose. A test asserts it (`test_cleaning_never_changes_meaning`).

Two rules needed care. Citation removal keys on the *digits-dagger* shape rather than the
bracket, so genuine CJK quotation brackets survive and a footnote dagger in an author list
(`Aidan N. Gomez ∗ †`) is left alone — but markers split across a table-cell boundary arrive
as dangling halves, so unbalanced brackets are cleared separately. De-hyphenation only joins
when the next line starts lowercase, so `state-\nOf-the-art` stays intact.

Table cells are cleaned through the grid rather than the rendered markdown, which is
regenerated afterwards — cleaning the string alone would desynchronise the two.

Every rule reports how many times it fired. The counts land in `profile.json` and appear in
the UI, so cleaning is as auditable as chunking is, rather than happening invisibly.

### 3.3 Universal representation — `models.py`

```python
@dataclass(slots=True)
class ContentElement:
    id: str
    kind: str  # heading | paragraph | table | figure | caption | speech_segment | …
    text: str
    level: int | None  # heading depth
    parent_id: str | None  # heading ancestry
    page: int | None
    bbox: tuple | None  # spatial, inline
    t0: float | None  # temporal, inline
    t1: float | None
    speaker: str | None
    meta: dict  # table grid, header row, sheet name, worker extras
```

Spatial and temporal facts live *on* the element rather than in parallel arrays, so no
joins are needed to answer "where and when was this." Confidence and evidence are
deliberately **absent** here — they belong to relationships and decisions, which is where
they can actually be acted on. Adding them per element would be bookkeeping nothing reads.

### 3.4 Context graph — `graph.py`

A `networkx.DiGraph` built once per job, held in memory, serialised to `graph.json`. It is
a *logical* graph: the goal is to constrain boundaries and assemble context, not to run a
database. A graph store would add an install and a service for no gain at this scale, and
the JSON serialisation is the migration path if that ever changes.

Edges, each carrying `confidence` and a human-readable `evidence` string:

| Edge | How it is found | Confidence |
|---|---|---|
| `parent` | heading ancestry from the ingest stack | 1.0 |
| `captions` / `captioned_by` | Docling's own caption reference | 1.0 |
| `captions` / `captioned_by` | bbox adjacency — nearest caption within 60pt, overlapping horizontally, same page | 0.8 |
| `references` | regex `Table N` / `Figure N`, resolved by matching caption text, falling back to ordinal | 0.9 |
| `next` | reading order | 1.0 |

The evidence string is not decoration. `"bbox adjacency: caption 14pt below figure on page 4"`
is what lets a person confirm the system is right rather than take its word.

Queries the rest of the system needs: `heading_path()` (ancestry, ordered), `captions_of()`,
`references_from()`, and `surrounding_text()` — the nearest prose before and after an
element, stopping at a heading, which is often the only sentence that says what a table is
*for*.

### 3.5 Profiling and routing — `router.py`

Signals are deterministic and take milliseconds. No model runs before the routing decision.

| Signal | Meaning |
|---|---|
| `has_timestamps` | any element carries `t0` |
| `is_tabular` | table tokens ≥ 4× prose tokens — the input *is* a dataset |
| `heading_count`, `heading_density` | is there structure worth trusting |
| `table_count`, `figure_count`, `caption_count` | how much of the content is a float |
| `row_count`, `column_count` | dataset shape |

The cascade, in priority order:

```python
if has_timestamps:                                    → temporal
elif is_tabular:                                      → tabular
elif heading_count >= 3 and heading_density >= 0.03:  → structural
elif MiniLM available:                                → semantic
else:                                                 → paragraph_fallback
```

Order matters. Timestamps are unambiguous. `is_tabular` must precede `structural`, because
sheet-name headings would otherwise make a spreadsheet look structured. `structural` before
semantic because real hierarchy beats inferred topic drift, and it is free.

Note what is *not* a routing signal: topic drift. It lives inside the semantic chunker as a
boundary-finder. Using it to choose a strategy would mean paying for embeddings before
knowing whether they are needed.

### 3.6 Chunking — `chunkers.py`, `tabular.py`, `semantic.py`

**Atomic** (floats inside prose). A table or figure plus its caption is one unit, always.
Over budget, a table splits by rows with its header repeated on every part.

**Structural.** One unit per innermost section. Oversized sections split at veto-checked
paragraph boundaries; children inherit the full heading path, so a fragment still knows it
belongs to *3. Results › 3.2 Revenue*.

**Tabular.** Two kinds of unit per sheet:

- A **schema card** — every column profiled by inferred type, range, null count and
  cardinality. This is real extraction with no model involved, and it makes a dataset
  searchable by *shape* ("which file has a revenue column?") rather than only by cell
  contents. Type inference classifies numbers as one family before asking about decimals,
  because a column of `4.9, 5, 6.1` is decimal, not "mostly-decimal-therefore-text."
- **Row groups**, budgeted by tokens, that always repeat the header and never split a row,
  each linked back to the card by `has_schema`.

**Semantic** (flat prose). Boundaries where cosine similarity between adjacent elements
drops below mean − 1σ. Falls back to paragraph packing when MiniLM is unavailable, and says
so in the receipt (`strategy: paragraph_fallback`).

**Temporal** (timed content). Speaker change is a boundary; turns under 3s merge into their
neighbour; same-speaker runs over 120s split at the largest pause.

#### The veto — `choose_cut()`

This is the mechanism the product is named for.

```python
candidates = paragraph boundaries, ordered by |tokens_before − target|
for c in candidates:
    if c would strand a heading:            veto, record, continue
    if c would sever a CAPTIONS pair:       veto, record, continue
    if c splits a consecutive list run:     defer as least-bad
    return c
return None   # nothing safe → keep the region whole, overflow=True
```

**Overflow beats severance.** An oversized chunk is inconvenient; a chunk whose meaning
lives in another chunk is broken. Every refusal is recorded in `decision.vetoed_cuts` and
surfaced in the UI, which is what turns a policy into something a person can check.

### 3.7 Cost accounting and model routing — `usage.py`, `llm.py`

Every model call is metered: model, tokens in, tokens out, tokens served from
cache, and cost, priced from a published-rates table. A model with no listed rate is
billed at a fallback and marked `estimated`, so an unpriced model shows up as an
approximation rather than silently as free. Local models are recorded identically at zero
cost, which is what lets the two paths sit in one table and be compared rather than argued
about. Per-job usage lands in `profile.json`; an install-wide ledger accumulates in
`data/usage.json` and is served at `/api/usage`.

**Provider order is local-first.** `get_provider()` prefers a running Ollama instance
because it costs nothing and keeps the document on the machine, falls back to the Gemini
API, then to `NoneProvider`. `CLEAVE_LLM=none|ollama|gemini` forces the choice.

Ollama runs as a separate process for the same reason the STT worker does: this venv is
pinned by Docling and cannot also host an inference stack. HTTP turns that constraint into
an interface. Structured output uses Ollama's `format` parameter, which constrains decoding
to the JSON schema via GBNF — valid by construction rather than by asking politely. One
trap worth recording: reasoning models such as qwen3 emit the schema-constrained JSON
inside their *thinking* block and leave `response` empty, so the provider sends
`think: false` and falls back to reading `thinking` for builds that ignore it.

### 3.8 Selective enrichment — `enrich.py`

Flags are computed deterministically after chunking. A unit is a candidate if **any** hold:

- `anaphora_rate > 0.10` — sentences leaning on absent context ("as shown above", "this
  table", "the former", a bare sentence-initial "This")
- it is an orphan with no heading ancestry
- it is an uncaptioned table or figure

Everything else stays tier 0 and never touches a model. Measured on the fixtures: 72% of the
research paper and 96% of the 480-row CSV are tier 0.

**Batching is where the cost actually went.** Situating a chunk means showing the model the
document it came from, and the obvious shape sends that document again for every chunk. On
the 6k-token research paper with twelve flagged chunks that was 71,226 input tokens against
1,385 output — **86% of the spend was re-transmitting the same text.** Sending the document
once per *batch* of six chunks and asking for six summaries back cut it to 14,259 input
tokens and two calls: **80% fewer input tokens, 63% cheaper, with all twelve units still
enriched.** Cost is divided across the batch so each unit's receipt reflects what it
actually consumed.

Two smaller levers matter alongside it: Gemini bills cache-hit prefix tokens at roughly a
quarter rate and reports them in `cachedContentTokenCount`, which the ledger tracks
separately; and the document window is bounded (`CLEAVE_ENRICH_DOC_CHARS`) so a 200-page
report does not price itself out of being enriched at all.

Every provider returns `""` on any failure, so callers fall back to the deterministic path
rather than handling exceptions — a rate limit degrades the output instead of failing the
job. `CLEAVE_LLM=none` makes that the explicit mode: the pipeline runs identically, just
without summaries.

### 3.9 Serving — `app.py`, `cleave/web/`, `pipeline.py`

`app.py` is the composition root and nothing else: it builds the FastAPI app, mounts
the static files, registers the routers and keeps the `__main__` block. The layer it
used to hold is split by responsibility — `web/jobs.py` (the registry and its safe
path accessors), `web/uploads.py` (filename safety and size caps), `web/routes_pages.py`
and `web/routes_api.py` (the two surfaces), `web/search.py` (the retrieval demo) — and
the orchestration moved to `pipeline.py`, which imports nothing from FastAPI so it is
usable from the evaluator, a future CLI, or a test without standing up an HTTP app.
`cleave.app:app`, `cleave.app.JOBS` and `cleave.app.run_job` are all still importable
at their old names.

FastAPI with a module-level `JOBS` dict and `BackgroundTasks`. Jobs take seconds and
artifacts land on disk, so there is no queue, no database, and no cancellation. Progress
never moves backwards and only the worker sets a terminal state — a lesson borrowed from
the STT project, where publishing "done" from the progress channel let clients see finished
jobs with no results attached.

The UI polls a partial that **stops polling itself**: while the job runs the fragment
carries `hx-trigger`, and the terminal render simply omits it. No timer to clear, no
lifecycle to leak.

**A job can hold several input files.** The upload form accepts multiple files as one
submission; `run_job()` runs each through its own ingest → graph → route → chunk → enrich
pipeline (`_process_file()`), then merges the results. Every element and unit id is
namespaced with a per-file prefix (`_prefix_element`, `_prefix_unit`) before merging, so two
files' `el_0000` never collide and relationship targets stay internally consistent. Cleaning
stats and enrichment totals are summed across files (`_merge_cleaning`, `_merge_enrichment`);
a single-file job still writes the same flat `profile.json` shape it always has, and a
multi-file job adds a `files` array alongside the combined totals, so existing single-file
consumers see nothing new. Search and the context graph view operate over the merged set —
one job, searched as a whole, even though each file kept the routing decision that fit it.

**One file's failure does not discard the others.** Each input is attempted
independently; a file that raises is recorded as a warning on the job and the rest
still produce units. Only a job where *every* input failed is an error. Before this,
a stopped STT worker took a successfully parsed 200-page PDF down with the audio file
beside it.

**Per-job LLM override.** `Job.use_llm` (from the upload form's toggle) is threaded into
`enrich()`; when false it forces `NoneProvider` regardless of what `CLEAVE_LLM`/Ollama/Gemini
would otherwise select, so a user can force a deterministic, zero-cost run without touching
server config. The result page reports flagged-but-unenriched chunks as "toggle was off"
rather than "no provider found."

Per job on disk: `units.json`, `graph.json`, `profile.json` (profile + totals).

### 3.10 Configuration and outbound HTTP — `config.py`, `http.py`

Every environment variable is read, defaulted and validated in one place, and `.env` is
actually loaded — it was documented in the user guide and shipped as `.env.example` but
nothing ever read it, so filling it in did nothing. A real environment variable still
wins over the file (`override=False`), which is what keeps the documented offline test
invocation working. Validation is eager and every message names its variable:
`CLEAVE_ENRICH_BATCH=0` used to surface as `range() arg 3 must not be zero` raised from
the middle of enrichment. Settings are cached and re-readable rather than frozen into
module constants at import.

All three outbound callers — Ollama, Gemini, the STT worker — share one pooled client
and one retry policy. Each previously built a fresh request per call and none retried,
so a Gemini 429 became an empty completion and the job quietly produced unenriched
chunks with nothing in the UI to say why. Only transport errors and 408/429/5xx are
retried; `Retry-After` is honoured but capped, and backoff uses full jitter because
three concurrent enrichment batches retrying in lockstep would reproduce the burst that
earned the 429. The retries sit *below* `complete_json`'s never-raises contract, so a
rate limit still degrades the output rather than failing the job — but it is now
reported: failed calls are counted and surfaced as a job warning.

---

## 4. The Knowledge Unit

The output contract, and the interface external workers implement.

```python
@dataclass(slots=True)
class KnowledgeUnit:
    id: str
    content: str  # the retrievable text
    modality: Modality
    context: Context  # title, heading_path, situating_summary,
    #   leading/trailing prose, tier
    provenance: Provenance  # source_uri, sha256, page, bbox
    decision: ChunkingDecision  # strategy, reason, signals, vetoed_cuts,
    #   escalation_flags, llm_calls, cost_usd
    relationships: list[Relationship]  # type, target, confidence, evidence
    temporal: Temporal | None  # start_s, end_s, speaker
    entities: list[str]
    metadata: dict
    token_count: int

    def embed_text(self) -> str: ...
```

`embed_text()` is the contract with downstream retrieval, and it is deliberately not just
`content`: heading path first, then situating summary, then surrounding prose, then the
content itself. A chunk retrieved alone still knows where it came from.

Hierarchy is **denormalised** into `heading_path` rather than represented as edges on the
unit. The full element-level graph lives in `graph.json`; a unit carries only the
relationships where it is an endpoint, typically two to five. Copying the whole graph into
every chunk would make the payload quadratic for no benefit.

`ChunkingDecision` being part of the schema — not a log line — is the deliberate choice
that makes the system inspectable. Every chunk can answer "why do you exist, and what did
you refuse to do."

---

## 5. Evaluation — `evaluate.py`

The **Context Preservation Scorecard** compares a fixed 512-token/64-overlap baseline
against Cleave on four deterministic measures, each `preserved / total`:

| Metric | What it checks |
|---|---|
| caption integrity | captioned floats whose caption and body share a chunk |
| header integrity | every row-bearing chunk of a table also carries its header |
| heading context | paragraphs whose chunk knows their governing heading |
| resolved references | "Table N" mentions whose target is co-located or explicitly linked |

No LLM, no embeddings, no judgement calls — the same probe strings are searched in both
arms, so the comparison is symmetric. Results are written to `data/scorecard.json` and the
homepage renders that file. **No percentage is ever hardcoded**; if a metric comes out
unflattering it is shown as measured.

---

## 6. Deliberate exclusions

| Not used | Why |
|---|---|
| MinerU | ~32 s/page on Apple Silicon — unusable interactively |
| faster-whisper | CTranslate2 has no Metal backend; CPU-only on this hardware |
| Ollama as an MLX backend | requires >32 GB unified memory; a 24 GB Mac silently falls back to GGUF |
| Graph database | the graph is a means, not the product |
| Vector database | numpy cosine is sufficient at this scale |
| LangChain / LlamaIndex | nothing needed from them that isn't a few lines here |
| Weighted boundary scoring | hard vetoes are simpler *and* more legible in the receipt |
| Proposition/agentic chunking | published cost-benefit does not justify it |

---

## 7. Extending it

**A new modality** implements one function returning `IngestResult` and gets routing,
chunking, enrichment, the UI and retrieval for free. Emit `t0`/`t1` for temporal routing,
or `meta.grid` + `meta.header_row` for tabular.

**A new strategy** is a function returning `list[tuple[KnowledgeUnit, list[element_id]]]`,
plus a branch in the routing cascade and a colour in the template's `chip` map.

**A new relationship** is a `RelationType` member and an edge builder in `ContextGraph._build`;
add it to the hard-veto set in `choose_cut()` if severing it should be forbidden.

**A different LLM** is a class with `is_configured()` and `complete_json()` in `llm.py`.
