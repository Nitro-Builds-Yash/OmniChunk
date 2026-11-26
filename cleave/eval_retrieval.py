"""Retrieval benchmark: does Cleave's chunking make answers easier to find?

The Context Preservation Scorecard (``cleave.evaluate``) checks structure —
captions kept, headers repeated. This module checks the outcome that structure
is supposed to buy: given a question, does the top-k retrieved context contain
everything needed to answer it?

Each QA item names a file, a question, and ``must_contain`` — every string the
retrieved chunk must hold *together*. For a spreadsheet that deliberately
includes the column name (``revenue_usd``) next to the value: a row without
its header is exactly the failure Cleave exists to prevent, and a benchmark
that only looked for the number would not see it.

Two arms are compared over the same ingested text:

* ``fixed`` — the 512-token / 64-overlap splitter every RAG tutorial ships.
* ``cleave`` — the full routing pipeline (no LLM, so results are reproducible).

Both are retrieved with the same hybrid retriever (BM25, plus dense vectors
when the model loads), so the only variable is the chunking.

    uv run python -m cleave.eval_retrieval benchmarks/qa_fixtures.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .models import _encoder, count_tokens
from .web.search import BM25, RRF_K, tokenize

log = logging.getLogger(__name__)

FIXED_TOKENS, FIXED_OVERLAP = 512, 64
KS = (1, 3, 5)


@dataclass(slots=True)
class QA:
    file: str
    question: str
    must_contain: list[str]


def load_qa(path: Path) -> list[QA]:
    items = []
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            items.append(QA(d["file"], d["question"], list(d["must_contain"])))
    return items


def fixed_chunks_from_text(text: str) -> list[str]:
    enc = _encoder()
    ids = enc.encode(text, disallowed_special=())
    out, start = [], 0
    while start < len(ids):
        out.append(enc.decode(ids[start : start + FIXED_TOKENS]))
        start += FIXED_TOKENS - FIXED_OVERLAP
    return out


def _flatten(path: Path) -> str:
    """The same ingested, cleaned text Cleave sees — so only chunking differs."""
    from .ingest_document import ingest_document  # noqa: PLC0415

    ingest = ingest_document(path)
    return "\n\n".join(e.text for e in ingest.elements if e.text)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def answered(chunk: str, needles: list[str]) -> bool:
    c = _norm(chunk)
    return all(_norm(n) in c for n in needles)


def rank(chunks: list[str], question: str, dense: bool) -> list[int]:
    """Indices of ``chunks`` best-first, by BM25 fused with dense cosine (RRF)."""
    bm = BM25([tokenize(c) for c in chunks]).scores(tokenize(question))
    fused = [0.0] * len(chunks)
    for r, i in enumerate(sorted(range(len(chunks)), key=lambda i: bm[i], reverse=True), 1):
        if bm[i] > 0:
            fused[i] += 1 / (RRF_K + r)
    if dense:
        from .semantic import embed  # noqa: PLC0415

        vecs = embed(chunks)
        qv = embed([question])
        if vecs is not None and qv is not None:
            cos = vecs @ qv[0]
            order = sorted(range(len(chunks)), key=lambda i: cos[i], reverse=True)
            for r, i in enumerate(order, 1):
                fused[i] += 1 / (RRF_K + r)
    return sorted(range(len(chunks)), key=lambda i: fused[i], reverse=True)


def score_arm(chunks: list[str], items: list[QA], dense: bool) -> dict:
    hits = dict.fromkeys(KS, 0)
    rr_sum, ctx_tokens = 0.0, 0
    for qa in items:
        order = rank(chunks, qa.question, dense)
        first = next(
            (r for r, i in enumerate(order, 1) if answered(chunks[i], qa.must_contain)), None
        )
        for k in KS:
            hits[k] += int(first is not None and first <= k)
        rr_sum += 1 / first if first else 0.0
        ctx_tokens += sum(count_tokens(chunks[i]) for i in order[:3])
    n = max(1, len(items))
    return {
        **{f"recall@{k}": round(hits[k] / n, 3) for k in KS},
        "mrr": round(rr_sum / n, 3),
        "chunks": len(chunks),
        "avg_top3_context_tokens": round(ctx_tokens / n),
    }


def run(qa_path: Path, fixtures_dir: Path, dense: bool) -> dict:
    from .api import chunk_file  # noqa: PLC0415

    items = load_qa(qa_path)
    by_file: dict[str, list[QA]] = defaultdict(list)
    for qa in items:
        by_file[qa.file].append(qa)

    per_file: dict[str, dict] = {}
    totals: dict[str, list[tuple[int, dict]]] = {"fixed": [], "cleave": []}
    for name, qas in by_file.items():
        path = fixtures_dir / name
        cleave_chunks = [u.embed_text() for u in chunk_file(path)]
        fixed = fixed_chunks_from_text(_flatten(path))
        per_file[name] = {
            "questions": len(qas),
            "fixed": score_arm(fixed, qas, dense),
            "cleave": score_arm(cleave_chunks, qas, dense),
        }
        totals["fixed"].append((len(qas), per_file[name]["fixed"]))
        totals["cleave"].append((len(qas), per_file[name]["cleave"]))

    def weighted(rows: list[tuple[int, dict]]) -> dict:
        n = sum(w for w, _ in rows) or 1
        keys = [f"recall@{k}" for k in KS] + ["mrr", "avg_top3_context_tokens"]
        return {k: round(sum(w * r[k] for w, r in rows) / n, 3) for k in keys}

    return {
        "qa_file": str(qa_path),
        "questions": len(items),
        "retriever": "bm25+dense (RRF)" if dense else "bm25",
        "tokenizer": getattr(_encoder(), "name", "cl100k_base"),
        "overall": {arm: weighted(rows) for arm, rows in totals.items()},
        "per_file": per_file,
    }


def _print(report: dict) -> None:
    o = report["overall"]
    out = sys.stdout
    out.write(f"\n{report['questions']} questions · retriever: {report['retriever']}\n\n")
    out.write(f"{'metric':<26}{'fixed 512/64':>14}{'cleave':>10}\n")
    for k in [f"recall@{k}" for k in KS] + ["mrr", "avg_top3_context_tokens"]:
        out.write(f"{k:<26}{o['fixed'][k]:>14}{o['cleave'][k]:>10}\n")
    out.write("\n")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(prog="cleave eval-retrieval", description=__doc__.split("\n")[0])
    ap.add_argument("qa", type=Path, help="JSONL of {file, question, must_contain}")
    ap.add_argument("--fixtures", type=Path, default=Path("tests/fixtures"))
    ap.add_argument("--dense", action="store_true", help="fuse dense vectors (downloads a model)")
    ap.add_argument("-o", "--out", type=Path, default=Path("data/retrieval_scorecard.json"))
    args = ap.parse_args(argv)
    report = run(args.qa, args.fixtures, args.dense)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    _print(report)
    return report


if __name__ == "__main__":
    main()
