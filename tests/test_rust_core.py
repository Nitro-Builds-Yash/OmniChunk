"""The Rust extension must agree with the pure-Python paths exactly.

Skipped when ``cleave_core`` is not built; the Python paths are always tested
by the rest of the suite (and by the duplicate tests below).
"""

from __future__ import annotations

import random

import pytest

from cleave import _accel

core = pytest.importorskip("cleave_core")

ALPHABET = [
    *"abcXYZ019 -_.\t\n\u00a7\u00c9\u00fcK\u212a\u00a0\u2003\x1c,;:!()",
    "SO-",
    "3.2.",
    "  ",
]


def _random_text(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(n))


@pytest.fixture(scope="module")
def corpus() -> list[str]:
    rng = random.Random(1234)
    return [_random_text(rng, rng.randint(0, 300)) for _ in range(400)]


def test_tokenize_parity(corpus):
    for text in corpus:
        assert core.tokenize(text) == _accel.py_tokenize(text), repr(text)
    assert core.tokenize_many(corpus) == [_accel.py_tokenize(t) for t in corpus]


def test_approx_encode_parity_and_losslessness(corpus):
    for text in [*corpus, "", "   ", "a", "end  "]:
        pieces = core.approx_encode(text)
        assert pieces == _accel.py_approx_encode(text), repr(text)
        assert "".join(pieces) == text
        assert core.approx_count(text) == len(pieces)


def test_fixed_windows_parity(corpus):
    text = " ".join(corpus)
    pieces = _accel.py_approx_encode(text)
    expected = ["".join(pieces[i : i + 50]) for i in range(0, len(pieces), 40)]
    assert core.fixed_windows(text, 50, 10) == expected


def test_bm25_scores_are_bit_identical(corpus):
    rust = core.BM25(corpus)
    py = _accel.PyBM25([_accel.py_tokenize(t) for t in corpus])
    rng = random.Random(7)
    for _ in range(40):
        q = _accel.py_tokenize(_random_text(rng, 40))
        assert rust.scores(q) == py.scores(q)
    assert len(rust) == len(corpus)


def test_bm25_batch_top_k_matches_single_queries(corpus):
    rust = core.BM25(corpus)
    queries = ["abc so-019", "xyz 3.2.1", "nothing-matches-qqq"]
    batch = rust.batch_top_k(queries, 5)
    for q, got in zip(queries, batch):
        assert got == rust.top_k(_accel.py_tokenize(q), 5)


def test_minhash_and_groups_parity():
    rng = random.Random(99)
    words = [
        "the",
        "a",
        "revenue",
        "growth",
        "north",
        "team",
        "report",
        "quarter",
        "region",
        "account",
        "summer",
        "field",
    ]
    base = [" ".join(rng.choice(words) for _ in range(60)) for _ in range(30)]
    texts = base + [b + " extra words here" for b in base[:10]] + ["", "short"]
    for t in texts:
        assert core.minhash_signature(t) == _accel.py_minhash_signature(t)
    assert [list(g) for g in core.near_duplicate_groups(texts, 0.8)] == (
        _accel.py_near_duplicate_groups(texts, 0.8)
    )


@pytest.mark.skipif(_accel.BACKEND != "rust", reason="CLEAVE_NO_RUST forces Python")
def test_backend_reports_rust():
    assert _accel.backend_info()["backend"] == "rust"
