"""Python vs Rust (cleave_core) on Cleave's text hot paths.

uv run maturin develop -r -m rust/cleave_core/Cargo.toml
uv run python benchmarks/bench_text_engine.py [n_docs]
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

from cleave import _accel

if _accel._core is None:
    sys.exit("build cleave_core first (see docstring)")
core = _accel._core

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
rng = random.Random(0)
words = (Path(__file__).parent.parent / "tests/fixtures/flat_essay.md").read_text().split()
docs = [" ".join(rng.choices(words, k=400)) + f" SO-{10000 + i}" for i in range(N)]
dupes = docs[: N // 10] + [d + " (copy)" for d in docs[: N // 50]]
queries = [" ".join(rng.choices(words, k=6)) for _ in range(200)]


def timed(fn, repeat=1):
    t = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t) / repeat


rows = []


def bench(name, py, rs, repeat=1):
    tp, tr = timed(py, repeat), timed(rs, repeat)
    rows.append((name, tp, tr))


py_toks = [_accel.py_tokenize(d) for d in docs]
py_idx = _accel.PyBM25(py_toks)
rs_idx = core.BM25(docs)
big = " ".join(docs)

bench(
    "tokenize corpus",
    lambda: [_accel.py_tokenize(d) for d in docs],
    lambda: core.tokenize_many(docs),
)
bench(
    "BM25 index build",
    lambda: _accel.PyBM25([_accel.py_tokenize(d) for d in docs]),
    lambda: core.BM25(docs),
)
q = _accel.py_tokenize(queries[0])
bench("BM25 score 1 query", lambda: py_idx.scores(q), lambda: rs_idx.scores(q), repeat=5)
bench(
    "BM25 200 queries top-10",
    lambda: [
        sorted(enumerate(py_idx.scores(_accel.py_tokenize(x))), key=lambda p: -p[1])[:10]
        for x in queries
    ],
    lambda: rs_idx.batch_top_k(queries, 10),
)
bench(
    "approx token count", lambda: len(_accel.py_approx_encode(big)), lambda: core.approx_count(big)
)
bench(
    f"near-dup groups ({len(dupes)} docs)",
    lambda: _accel.py_near_duplicate_groups(dupes),
    lambda: core.near_duplicate_groups(dupes),
)

print(f"\n{N} docs × ~400 words · {os.cpu_count()} cores\n")
print(f"{'operation':<30}{'python':>10}{'rust':>10}{'speedup':>10}")
for name, tp, tr in rows:
    print(f"{name:<30}{tp:>9.3f}s{tr:>9.3f}s{tp / tr:>9.1f}×")
