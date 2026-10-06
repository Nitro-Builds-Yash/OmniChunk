"""Text normalisation, applied before anything measures or splits the content.

Extraction leaves debris. A PDF yields ligatures, words hyphenated across line
breaks, non-breaking spaces, and — depending on where the document came from —
reference markers like ``【32†L355-L364】`` that mean nothing outside their
original tool. None of it is content, all of it is embedded, and some of it is
expensive: the sample report carries 137 citation markers, roughly 1,200 tokens
of pure noise that would be paid for on every enrichment call and would sit in
every vector.

The rules here are deliberately conservative. Cleaning that changes meaning —
lowercasing, stripping stopwords, stemming, removing punctuation — is destroying
information to make a downstream metric look better, and this pipeline exists to
do the opposite. Everything below either removes something that was never in the
document, or repairs damage the extractor did.

Every rule reports how many times it fired, so cleaning is auditable in the same
way chunking decisions are: it appears in the job record and in the UI rather
than happening invisibly.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .markdown import table_markdown as _table_markdown

#: Reference markers left by tools that annotate sources, e.g. 【32†L355-L364】.
#: Brackets are optional because a marker that straddled a table-cell boundary
#: during extraction arrives as a dangling half. What identifies it either way
#: is the digits-dagger-line-numbers core, which no prose produces — so ordinary
#: CJK text using 【】 as quotation brackets is untouched.
_CITATION = re.compile(r"【?\s*\d+\s*†\s*L?[\d\s\-–L]{0,24}】?")

#: A bracket holding nothing but line numbers — the surviving half of a marker
#: whose dagger ended up in another table cell. Requires the content to be
#: entirely digits/L/dashes, which prose never is.
_CITATION_REMNANT = re.compile(r"【[\s\d\-–L]{1,24}】")

#: Brackets whose partner was removed with the other half of a split marker.
_LONE_OPEN = re.compile(r"【\s*")
_LONE_CLOSE = re.compile(r"\s*】")

#: Ligatures a PDF encodes as single glyphs; they break substring search and
#: tokenise badly ("ﬁnance" is not "finance" to a matcher).
_LIGATURES = str.maketrans(
    {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
    }
)

_INVISIBLE = re.compile(r"[­​‌‍⁠﻿]")  # soft hyphen, ZW*, BOM
_NBSP = re.compile(r"[   ]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: "inges-\ntion" → "ingestion". Only joins when the next line starts lowercase,
#: so a genuine hyphenated compound at a line end ("state-\nOf-the-art") and
#: enumerations survive.
_HYPHEN_BREAK = re.compile(r"(\w)[-‐‑]\s*\n\s*([a-z])")

_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?%)\]])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[])[ \t]+")
_DOT_LEADERS = re.compile(r"\.{4,}")  # table-of-contents leaders
_TRAILING_BULLET = re.compile(r"[ \t]*[·•▪●○◦]+[ \t]*$", re.M)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK = re.compile(r"\n{3,}")

#: Kinds whose text is data rather than prose. Code keeps its exact spacing;
#: table markdown is rebuilt from the grid, so only the cell pass touches it.
_VERBATIM_KINDS = frozenset({"code"})


@dataclass
class CleaningReport:
    """What was changed, by rule. Empty means the text arrived clean."""

    rules: Counter = field(default_factory=Counter)
    elements_changed: int = 0
    chars_removed: int = 0

    def merge(self, other: Counter) -> None:
        self.rules.update(other)

    @property
    def total(self) -> int:
        return sum(self.rules.values())

    def to_dict(self) -> dict:
        return {
            "total_fixes": self.total,
            "elements_changed": self.elements_changed,
            "chars_removed": self.chars_removed,
            "by_rule": dict(self.rules.most_common()),
        }

    def summary(self) -> str:
        if not self.total:
            return "no extraction artifacts found"
        top = ", ".join(f"{n} {rule.replace('_', ' ')}" for rule, n in self.rules.most_common(3))
        return f"{self.total} fixes across {self.elements_changed} elements — {top}"


def clean_text(text: str, *, verbatim: bool = False) -> tuple[str, Counter]:
    """Normalise one piece of text. → (cleaned, counts by rule)."""
    if not text:
        return text, Counter()
    counts: Counter = Counter()

    def sub(pattern: re.Pattern, repl: str, rule: str, s: str) -> str:
        s2, n = pattern.subn(repl, s)
        if n:
            counts[rule] += n
        return s2

    out = text

    # Unicode NFKC folds compatibility forms (ﬁ, full-width punctuation, odd
    # spaces) into their canonical equivalents. Applied first so later rules see
    # normal characters.
    normalised = unicodedata.normalize("NFKC", out)
    if normalised != out:
        counts["unicode_normalised"] += 1
        out = normalised

    if lig := sum(out.count(c) for c in "ﬁﬂﬀﬃﬄﬅﬆ"):
        out = out.translate(_LIGATURES)
        counts["ligatures"] += lig

    out = sub(_CONTROL, "", "control_chars", out)
    out = sub(_INVISIBLE, "", "invisible_chars", out)
    out = sub(_NBSP, " ", "non_breaking_spaces", out)
    out = sub(_CITATION, "", "citation_markers", out)
    out = sub(_CITATION_REMNANT, "", "citation_markers", out)

    # A marker split across a cell boundary leaves one bracket behind. Only
    # strip when they no longer balance, so paired 【】 used as real quotation
    # marks survive.
    if out.count("【") != out.count("】"):
        out = sub(_LONE_OPEN, "", "orphaned_citation_brackets", out)
        out = sub(_LONE_CLOSE, "", "orphaned_citation_brackets", out)

    if not verbatim:
        out = sub(_HYPHEN_BREAK, r"\1\2", "hyphen_line_breaks", out)
        out = sub(_DOT_LEADERS, " ", "dot_leaders", out)
        out = sub(_TRAILING_BULLET, "", "trailing_bullets", out)
        out = sub(_SPACE_BEFORE_PUNCT, r"\1", "space_before_punctuation", out)
        out = sub(_SPACE_AFTER_OPEN, r"\1", "space_after_bracket", out)
        out = sub(_MULTI_SPACE, " ", "repeated_spaces", out)
        out = sub(_MULTI_BLANK, "\n\n", "repeated_blank_lines", out)

    stripped = out.strip()
    if stripped != out:
        counts["surrounding_whitespace"] += 1
    return stripped, counts


# Pre-compiled quick-reject: if a cell contains none of these characters,
# none of the cleaning rules can fire. Avoids 12+ regex passes on short
# numeric or label cells that dominate large spreadsheets.
_CELL_NEEDS_CLEANING = re.compile(
    r"[【】†ﬁﬂﬀﬃﬄﬅﬆ­​‌‍⁠﻿\x00-\x08\x0b\x0c\x0e-\x1f\x7f  ]"
    r"|[ \t]{2}"
    r"|\.{4}"
)


def clean_cell(text: str) -> tuple[str, Counter]:
    """Fast-path cleaner for individual table cells.

    Most cells are short numbers or labels that need no cleaning at all.
    This avoids the full 12+ regex pass cost of clean_text for the common
    case, falling through to the full cleaner only when debris is detected.
    """
    if not text or len(text) < 3:
        return text, Counter()
    if not _CELL_NEEDS_CLEANING.search(text):
        # No markers, ligatures, invisible chars, or spacing debris detected.
        stripped = text.strip()
        if stripped == text:
            return text, Counter()
        return stripped, Counter({"surrounding_whitespace": 1})
    return clean_text(text)


def clean_elements(elements) -> CleaningReport:
    """Normalise every element in place, before profiling or chunking.

    Order matters: token counts, routing signals and boundary choices are all
    computed downstream, and they should describe the text that will actually be
    stored and embedded — not the raw extraction.
    """
    report = CleaningReport()
    for el in elements:
        verbatim = el.kind in _VERBATIM_KINDS
        before = el.text
        cleaned, counts = clean_text(before, verbatim=verbatim)
        if counts:
            el.text = cleaned
            report.merge(counts)
            report.elements_changed += 1
            report.chars_removed += max(0, len(before) - len(cleaned))

        # Cells are cleaned through the grid, not the rendered markdown, or the
        # two would desynchronise — the markdown is regenerated from the grid
        # afterwards. Cells get the full treatment: a cell is prose in a box,
        # and leaving its spacing broken is what produced "(Process, Index,
        # RAG)  ." where a citation used to sit.
        grid = el.meta.get("grid")
        if grid:
            new_grid: list[list[str]] = []
            cell_counts: Counter = Counter()
            for row in grid:
                new_row = []
                for cell in row:
                    c, cc = clean_cell(cell)
                    cell_counts.update(cc)
                    new_row.append(c)
                new_grid.append(new_row)
            if cell_counts:
                el.meta["grid"] = new_grid
                if el.meta.get("header_row"):
                    el.meta["header_row"] = [clean_cell(h)[0] for h in el.meta["header_row"]]
                el.text = _table_markdown(new_grid)
                report.merge(cell_counts)
    return report
