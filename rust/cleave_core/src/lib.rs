//! Rust hot paths for Cleave.
//!
//! Everything here has a pure-Python twin in `cleave/_accel.py` and must stay
//! **bit-for-bit compatible** with it: the same tokens, the same BM25 scores in
//! the same floating-point summation order, the same MinHash signatures. The
//! parity tests in `tests/test_rust_core.py` enforce that, so installing or
//! removing this extension changes speed, never results.
//!
//! * [`tokenize`] — search tokens (`SO-10042` kept whole *and* split)
//! * [`approx_encode`] — lossless whitespace-anchored token pieces
//! * [`Bm25`] — Okapi BM25 over an inverted index, parallel batch queries
//! * [`near_duplicate_groups`] — MinHash + LSH near-duplicate detection
//! * [`fixed_windows`] — the fixed-size/overlap baseline splitter

use rayon::prelude::*;
use std::collections::HashMap;

// ───────── tokenization ─────────

#[inline]
fn is_tok(c: char) -> bool {
    c.is_ascii_lowercase() || c.is_ascii_digit()
}

#[inline]
fn is_sep(c: char) -> bool {
    matches!(c, '-' | '_' | '.')
}

/// Python's `str.lower()` then the regex `[a-z0-9]+(?:[-_.][a-z0-9]+)*`,
/// emitting each compound token followed by its `[-_.]`-separated parts.
pub fn tokenize(text: &str) -> Vec<String> {
    let lowered: Vec<char> = text.chars().flat_map(char::to_lowercase).collect();
    let n = lowered.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if !is_tok(lowered[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && is_tok(lowered[i]) {
            i += 1;
        }
        let mut compound = false;
        while i + 1 < n && is_sep(lowered[i]) && is_tok(lowered[i + 1]) {
            compound = true;
            i += 1;
            while i < n && is_tok(lowered[i]) {
                i += 1;
            }
        }
        let tok: String = lowered[start..i].iter().collect();
        if compound {
            let parts: Vec<String> = tok
                .split(is_sep)
                .filter(|p| !p.is_empty())
                .map(str::to_owned)
                .collect();
            out.push(tok);
            out.extend(parts);
        } else {
            out.push(tok);
        }
    }
    out
}

// ───────── approximate token pieces ─────────

/// Python's `\s` for `str` patterns: Unicode White_Space plus U+001C..U+001F.
#[inline]
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `re.findall(r"\s*\S+|\s+$", text)` — joining the pieces reproduces `text`.
pub fn approx_encode(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut iter = text.char_indices().peekable();
    let mut start = 0usize;
    let mut seen_non_space = false;
    while let Some((idx, c)) = iter.next() {
        if !is_py_space(c) {
            seen_non_space = true;
            let next_is_space = iter.peek().is_none_or(|&(_, d)| is_py_space(d));
            if next_is_space {
                let end = idx + c.len_utf8();
                out.push(&text[start..end]);
                start = end;
                seen_non_space = false;
            }
        }
    }
    if start < text.len() && !seen_non_space {
        out.push(&text[start..]); // trailing whitespace run
    }
    out
}

pub fn approx_count(text: &str) -> usize {
    approx_encode(text).len()
}

/// Baseline splitter over approximate token pieces: windows of `size` with
/// `overlap`, exactly like `eval_retrieval.fixed_chunks_from_text`.
pub fn fixed_windows(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let pieces = approx_encode(text);
    let step = size.saturating_sub(overlap).max(1);
    let mut out = Vec::new();
    let mut start = 0;
    while start < pieces.len() {
        let end = (start + size).min(pieces.len());
        out.push(pieces[start..end].concat());
        start += step;
    }
    out
}

// ───────── BM25 ─────────

pub struct Bm25 {
    k1: f64,
    b: f64,
    avg: f64,
    lens: Vec<f64>,
    idf: HashMap<String, f64>,
    postings: HashMap<String, Vec<(u32, u32)>>, // term -> [(doc, tf)] ascending doc
}

impl Bm25 {
    pub fn from_tokens(docs: Vec<Vec<String>>, k1: f64, b: f64) -> Self {
        let n = docs.len();
        let lens: Vec<f64> = docs.iter().map(|d| d.len() as f64).collect();
        let avg = if n > 0 {
            lens.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };
        let mut postings: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        for (di, doc) in docs.into_iter().enumerate() {
            let mut tf: HashMap<String, u32> = HashMap::new();
            for t in doc {
                *tf.entry(t).or_insert(0) += 1;
            }
            for (t, f) in tf {
                postings.entry(t).or_default().push((di as u32, f));
            }
        }
        let idf = postings
            .iter()
            .map(|(t, p)| {
                let df = p.len() as f64;
                (t.clone(), (1.0 + (n as f64 - df + 0.5) / (df + 0.5)).ln())
            })
            .collect();
        Bm25 {
            k1,
            b,
            avg,
            lens,
            idf,
            postings,
        }
    }

    pub fn from_texts(texts: &[String], k1: f64, b: f64) -> Self {
        let docs: Vec<Vec<String>> = texts.par_iter().map(|t| tokenize(t)).collect();
        Self::from_tokens(docs, k1, b)
    }

    pub fn len(&self) -> usize {
        self.lens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lens.is_empty()
    }

    /// Scores for every document. Contributions are added per query token in
    /// query order, matching the Python loop's summation order exactly.
    pub fn scores(&self, query: &[String]) -> Vec<f64> {
        let mut acc = vec![0.0f64; self.lens.len()];
        let avg = if self.avg == 0.0 { 1.0 } else { self.avg };
        for t in query {
            let (Some(post), Some(&idf)) = (self.postings.get(t), self.idf.get(t)) else {
                continue;
            };
            for &(d, f) in post {
                let d = d as usize;
                let f = f as f64;
                let norm = self.k1 * (1.0 - self.b + self.b * self.lens[d] / avg);
                acc[d] += idf * f * (self.k1 + 1.0) / (f + norm);
            }
        }
        acc
    }

    pub fn top_k(&self, query: &[String], k: usize) -> Vec<(usize, f64)> {
        let s = self.scores(query);
        let mut idx: Vec<usize> = (0..s.len()).filter(|&i| s[i] > 0.0).collect();
        // Stable on ties (lower index first), like Python's sorted(..., reverse=True)
        // over a key: sort descending by score, ascending by index.
        idx.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap().then(a.cmp(&b)));
        idx.truncate(k);
        idx.into_iter().map(|i| (i, s[i])).collect()
    }
}

// ───────── MinHash near-duplicates ─────────

pub const NUM_PERM: usize = 128;
pub const BANDS: usize = 16; // 16 bands × 8 rows: candidate threshold ≈ 0.71

#[inline]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[inline]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Word-shingle hashes (size 5, or the whole token list when shorter).
pub fn shingles(tokens: &[String], size: usize) -> Vec<u64> {
    if tokens.is_empty() {
        return vec![];
    }
    let w = size.min(tokens.len());
    let mut out: Vec<u64> = tokens
        .windows(w)
        .map(|win| fnv1a64(win.join(" ").as_bytes()))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

pub fn minhash(shingle_hashes: &[u64]) -> Option<[u64; NUM_PERM]> {
    if shingle_hashes.is_empty() {
        return None;
    }
    let mut sig = [u64::MAX; NUM_PERM];
    for &h in shingle_hashes {
        for (i, slot) in sig.iter_mut().enumerate() {
            let v = splitmix64(h ^ splitmix64(i as u64 + 1));
            if v < *slot {
                *slot = v;
            }
        }
    }
    Some(sig)
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Groups (each sorted, size ≥ 2, ordered by first member) of texts whose
/// estimated Jaccard similarity over 5-word shingles is ≥ `threshold`.
pub fn near_duplicate_groups(texts: &[String], threshold: f64) -> Vec<Vec<usize>> {
    let sigs: Vec<Option<[u64; NUM_PERM]>> = texts
        .par_iter()
        .map(|t| minhash(&shingles(&tokenize(t), 5)))
        .collect();
    let rows = NUM_PERM / BANDS;
    let mut buckets: HashMap<(usize, u64), Vec<usize>> = HashMap::new();
    for (i, sig) in sigs.iter().enumerate() {
        let Some(sig) = sig else { continue };
        for band in 0..BANDS {
            let mut h: u64 = band as u64;
            for v in &sig[band * rows..(band + 1) * rows] {
                h = splitmix64(h ^ v);
            }
            buckets.entry((band, h)).or_default().push(i);
        }
    }
    let mut parent: Vec<usize> = (0..texts.len()).collect();
    let mut checked = std::collections::HashSet::new();
    for members in buckets.values() {
        for (ai, &a) in members.iter().enumerate() {
            for &b in &members[ai + 1..] {
                if !checked.insert((a, b)) {
                    continue;
                }
                let (sa, sb) = (sigs[a].as_ref().unwrap(), sigs[b].as_ref().unwrap());
                let same = sa.iter().zip(sb.iter()).filter(|(x, y)| x == y).count();
                if same as f64 / NUM_PERM as f64 >= threshold {
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    if ra != rb {
                        parent[ra.max(rb)] = ra.min(rb);
                    }
                }
            }
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..texts.len() {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    let mut out: Vec<Vec<usize>> = groups.into_values().filter(|g| g.len() > 1).collect();
    for g in &mut out {
        g.sort_unstable();
    }
    out.sort_by_key(|g| g[0]);
    out
}

// ───────── Python bindings ─────────

#[cfg(feature = "python")]
mod py {
    use super::*;
    use pyo3::prelude::*;

    #[pyfunction]
    #[pyo3(name = "tokenize")]
    fn py_tokenize(text: &str) -> Vec<String> {
        tokenize(text)
    }

    #[pyfunction]
    #[pyo3(name = "tokenize_many")]
    fn py_tokenize_many(py: Python<'_>, texts: Vec<String>) -> Vec<Vec<String>> {
        py.allow_threads(|| texts.par_iter().map(|t| tokenize(t)).collect())
    }

    #[pyfunction]
    #[pyo3(name = "approx_encode")]
    fn py_approx_encode(text: &str) -> Vec<String> {
        approx_encode(text).into_iter().map(str::to_owned).collect()
    }

    #[pyfunction]
    #[pyo3(name = "approx_count")]
    fn py_approx_count(py: Python<'_>, text: String) -> usize {
        py.allow_threads(|| approx_count(&text))
    }

    #[pyfunction]
    #[pyo3(name = "fixed_windows", signature = (text, size=512, overlap=64))]
    fn py_fixed_windows(py: Python<'_>, text: String, size: usize, overlap: usize) -> Vec<String> {
        py.allow_threads(|| fixed_windows(&text, size, overlap))
    }

    #[pyfunction]
    #[pyo3(name = "near_duplicate_groups", signature = (texts, threshold=0.8))]
    fn py_near_duplicate_groups(
        py: Python<'_>,
        texts: Vec<String>,
        threshold: f64,
    ) -> Vec<Vec<usize>> {
        py.allow_threads(|| near_duplicate_groups(&texts, threshold))
    }

    #[pyfunction]
    #[pyo3(name = "minhash_signature")]
    fn py_minhash_signature(text: &str) -> Option<Vec<u64>> {
        minhash(&shingles(&tokenize(text), 5)).map(|s| s.to_vec())
    }

    /// BM25 index built from raw texts (tokenized in parallel, GIL released).
    #[pyclass(name = "BM25", frozen)]
    struct PyBm25 {
        inner: Bm25,
    }

    #[pymethods]
    impl PyBm25 {
        #[new]
        #[pyo3(signature = (texts, k1=1.5, b=0.75))]
        fn new(py: Python<'_>, texts: Vec<String>, k1: f64, b: f64) -> Self {
            PyBm25 {
                inner: py.allow_threads(|| Bm25::from_texts(&texts, k1, b)),
            }
        }

        #[staticmethod]
        #[pyo3(signature = (docs, k1=1.5, b=0.75))]
        fn from_tokens(py: Python<'_>, docs: Vec<Vec<String>>, k1: f64, b: f64) -> Self {
            PyBm25 {
                inner: py.allow_threads(|| Bm25::from_tokens(docs, k1, b)),
            }
        }

        fn __len__(&self) -> usize {
            self.inner.len()
        }

        /// Scores for every document; `query` is a list of tokens.
        fn scores(&self, py: Python<'_>, query: Vec<String>) -> Vec<f64> {
            py.allow_threads(|| self.inner.scores(&query))
        }

        fn top_k(&self, py: Python<'_>, query: Vec<String>, k: usize) -> Vec<(usize, f64)> {
            py.allow_threads(|| self.inner.top_k(&query, k))
        }

        /// Many queries (raw strings) in parallel across cores.
        fn batch_top_k(
            &self,
            py: Python<'_>,
            queries: Vec<String>,
            k: usize,
        ) -> Vec<Vec<(usize, f64)>> {
            py.allow_threads(|| {
                queries
                    .par_iter()
                    .map(|q| self.inner.top_k(&tokenize(q), k))
                    .collect()
            })
        }
    }

    #[pymodule]
    fn cleave_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        m.add("NUM_PERM", NUM_PERM)?;
        m.add("BANDS", BANDS)?;
        m.add_function(wrap_pyfunction!(py_tokenize, m)?)?;
        m.add_function(wrap_pyfunction!(py_tokenize_many, m)?)?;
        m.add_function(wrap_pyfunction!(py_approx_encode, m)?)?;
        m.add_function(wrap_pyfunction!(py_approx_count, m)?)?;
        m.add_function(wrap_pyfunction!(py_fixed_windows, m)?)?;
        m.add_function(wrap_pyfunction!(py_near_duplicate_groups, m)?)?;
        m.add_function(wrap_pyfunction!(py_minhash_signature, m)?)?;
        m.add_class::<PyBm25>()?;
        Ok(())
    }
}

// ───────── tests ─────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_keeps_compounds_and_parts() {
        assert_eq!(
            tokenize("Order SO-10042 in §3.2.1, a.-b"),
            vec!["order", "so-10042", "so", "10042", "in", "3.2.1", "3", "2", "1", "a", "b"]
        );
    }

    #[test]
    fn approx_encode_is_lossless() {
        for s in ["", "   ", "a", " a b  ", "x\n\ny  \t", "ünï cødé  ok "] {
            assert_eq!(approx_encode(s).concat(), s, "{s:?}");
        }
        assert_eq!(approx_encode(" a b  "), vec![" a", " b", "  "]);
        assert_eq!(approx_encode("   "), vec!["   "]);
    }

    #[test]
    fn bm25_ranks_the_exact_match_first() {
        let texts: Vec<String> = ["alpha beta", "gamma SO-10042 delta", "beta beta beta"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let idx = Bm25::from_texts(&texts, 1.5, 0.75);
        assert_eq!(idx.top_k(&tokenize("so-10042"), 3)[0].0, 1);
        assert_eq!(idx.top_k(&tokenize("beta"), 3)[0].0, 2);
    }

    #[test]
    fn near_duplicates_group_and_distinct_texts_do_not() {
        let base = "the quarterly revenue report shows strong growth in the north region with \
                    several new accounts opened by the field team this summer";
        let texts = vec![
            base.to_string(),
            format!("{base} indeed"),
            "a completely different paragraph about remote work and documentation habits \
             across distributed product teams"
                .to_string(),
        ];
        assert_eq!(near_duplicate_groups(&texts, 0.8), vec![vec![0, 1]]);
    }

    #[test]
    fn fixed_windows_overlap() {
        let text = (0..10)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let w = fixed_windows(&text, 4, 1);
        assert_eq!(w[0], "w0 w1 w2 w3");
        assert_eq!(w[1], " w3 w4 w5 w6");
        assert!(w.concat().len() > text.len());
    }
}
