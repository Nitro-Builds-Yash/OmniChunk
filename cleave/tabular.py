"""Tabular understanding: column profiling and row-group chunking.

Spreadsheets break the assumptions prose chunking is built on. A row carries no
meaning without its header, a row group carries none without its sheet, and
there is no narrative for topic drift to follow. So the tabular path does two
things nothing else in Cleave does:

  * profiles every column (type, nulls, range, distinct values) into a schema
    card — deterministic extraction that makes the dataset searchable by shape,
    not just by cell contents;
  * cuts only between rows, repeating the header in every chunk, so each unit
    is a self-describing table rather than an orphaned block of values.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .markdown import body_rows, row_md
from .models import count_tokens

log = logging.getLogger(__name__)

#: Rows are dense and repetitive, so tabular chunks get a larger budget than
#: prose — a chunk that holds only three rows is not a useful retrieval unit.
TABULAR_TARGET_TOKENS = 900
TABULAR_MAX_TOKENS = 1200

_INT_RE = re.compile(r"^-?\d{1,3}(,\d{3})*$|^-?\d+$")
_DEC_RE = re.compile(r"^-?\d*\.\d+$|^-?\d{1,3}(,\d{3})*\.\d+$")
_DATE_RE = re.compile(
    r"^(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-Q[1-4]|\d{4}/\d{1,2})"
    r"([ T]\d{1,2}:\d{2}(:\d{2})?)?$"
)
_BOOL_VALUES = {"true", "false", "yes", "no", "y", "n", "0", "1"}
_PCT_RE = re.compile(r"^-?\d+(\.\d+)?\s*%$")
_NULLISH = {"", "-", "n/a", "na", "null", "none", "nan"}


@dataclass(slots=True)
class ColumnProfile:
    name: str
    dtype: str  # integer|decimal|percentage|date|boolean|categorical|text
    non_null: int = 0
    nulls: int = 0
    distinct: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    examples: list[str] = field(default_factory=list)

    def describe(self) -> str:
        bits = [f"{self.name} ({self.dtype}"]
        if self.nulls:
            bits.append(f", {self.nulls} empty")
        bits.append(")")
        head = "".join(bits)
        if self.minimum is not None and self.maximum is not None:
            dp = self.dtype in ("decimal", "percentage")
            return (
                f"{head}: {_fmt(self.minimum, dp)} … {_fmt(self.maximum, dp)}, "
                f"mean {_fmt(self.mean, dp)}"
            )
        if self.dtype in ("categorical", "boolean") and self.examples:
            shown = ", ".join(self.examples[:6])
            more = f" (+{self.distinct - len(self.examples[:6])} more)" if self.distinct > 6 else ""
            return f"{head}: {shown}{more}"
        if self.dtype == "identifier":
            eg = f", e.g. {self.examples[0][:40]!r}" if self.examples else ""
            return f"{head}: {self.distinct:,} unique values{eg}"
        if self.examples:
            return f"{head}: e.g. {self.examples[0][:60]!r}"
        return head


def _fmt(v: float | None, force_decimals: bool = False) -> str:
    if v is None:
        return "?"
    if not force_decimals and abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
        return f"{round(v):,}"
    return f"{v:,.2f}"


def _as_number(s: str) -> float | None:
    t = s.strip().replace(",", "").rstrip("%")
    try:
        return float(t)
    except ValueError:
        return None


def profile_column(name: str, values: list[str]) -> ColumnProfile:
    """Infer a column's type from its values. Deterministic, no model: the
    dominant well-formed pattern wins, ties resolve toward the looser type."""
    present = [v.strip() for v in values if v.strip().lower() not in _NULLISH]
    nulls = len(values) - len(present)
    p = ColumnProfile(
        name=name or "(unnamed)",
        dtype="text",
        non_null=len(present),
        nulls=nulls,
        distinct=len(set(present)),
    )
    if not present:
        p.dtype = "empty"
        return p

    n = len(present)
    # Single pass: classify each value once instead of running 4 separate loops.
    # Order matters: percentage before decimal (both match \d+\.\d+), date before
    # integer (a date like 2024-01-15 could partially match integer patterns).
    ints = 0
    decs = 0
    pcts = 0
    dates = 0
    for v in present:
        if _PCT_RE.match(v):
            pcts += 1
        elif _DATE_RE.match(v):
            dates += 1
        elif _INT_RE.match(v):
            ints += 1
        elif _DEC_RE.match(v):
            decs += 1

    if pcts >= 0.9 * n:
        p.dtype = "percentage"
    elif dates >= 0.9 * n:
        p.dtype = "date"
    elif ints + decs >= 0.9 * n:
        p.dtype = "decimal" if decs else "integer"
    else:
        lowered = {v.lower() for v in present}
        avg_len = sum(len(v) for v in present) / n
        if lowered <= _BOOL_VALUES and p.distinct <= 2:
            p.dtype = "boolean"
        elif p.distinct <= max(12, n * 0.1) and avg_len <= 60:
            # long values that happen to be distinct are prose, not a category set
            p.dtype = "categorical"
        elif p.distinct == n and n > 1:
            # every value unique and non-numeric — a key, not free text
            p.dtype = "identifier"

    if p.dtype in ("integer", "decimal", "percentage"):
        nums = [x for x in (_as_number(v) for v in present) if x is not None]
        if nums:
            p.minimum, p.maximum = min(nums), max(nums)
            p.mean = sum(nums) / len(nums)
    if p.dtype in ("categorical", "boolean"):
        seen: list[str] = []
        for v in present:
            if v not in seen:
                seen.append(v)
            if len(seen) >= 8:
                break
        p.examples = seen
    elif p.dtype != "empty":
        p.examples = present[:2]
    return p


@dataclass(slots=True)
class TableProfile:
    sheet: str | None
    header: list[str]
    row_count: int
    column_count: int
    columns: list[ColumnProfile]

    def schema_card(self, source_name: str) -> str:
        """The human- and machine-readable summary emitted as its own unit."""
        where = f"{source_name}" + (f" · sheet “{self.sheet}”" if self.sheet else "")
        lines = [
            f"Dataset: {where}",
            f"{self.row_count:,} rows × {self.column_count} columns",
            "",
            "Columns:",
        ]
        lines += [f"  - {c.describe()}" for c in self.columns]
        return "\n".join(lines)

    def to_meta(self) -> dict[str, Any]:
        return {
            "sheet": self.sheet,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "nulls": c.nulls,
                    "distinct": c.distinct,
                    **(
                        {"min": c.minimum, "max": c.maximum, "mean": round(c.mean, 4)}
                        if c.mean is not None
                        else {}
                    ),
                }
                for c in self.columns
            ],
        }


def profile_table(grid: list[list[str]], header: list[str], sheet: str | None) -> TableProfile:
    body = body_rows(grid, header)
    ncols = len(header) if header else (len(grid[0]) if grid else 0)
    cols = []
    for i in range(ncols):
        name = header[i] if i < len(header) else f"column_{i + 1}"
        cols.append(profile_column(name, [r[i] if i < len(r) else "" for r in body]))
    return TableProfile(
        sheet=sheet, header=header, row_count=len(body), column_count=ncols, columns=cols
    )


def row_groups(
    grid: list[list[str]], header: list[str], target_tokens: int = TABULAR_TARGET_TOKENS
) -> list[tuple[int, list[list[str]]]]:
    """Split body rows into groups that fit the budget. Returns (start_row_index,
    rows) with 1-based row numbers as they appear in the source, so provenance
    can point back at the spreadsheet."""
    body = body_rows(grid, header)
    header_cost = count_tokens(row_md(header)) if header else 0
    groups: list[tuple[int, list[list[str]]]] = []
    cur: list[list[str]] = []
    start = 0
    acc = header_cost
    for i, row in enumerate(body):
        cost = count_tokens(row_md(row))
        if cur and acc + cost > target_tokens:
            groups.append((start, cur))
            cur, start, acc = [], i, header_cost
        cur.append(row)
        acc += cost
    if cur:
        groups.append((start, cur))
    return groups


def render_group(header: list[str], rows: list[list[str]]) -> str:
    """Markdown for one row group — header always repeated, so the chunk is a
    valid, self-describing table on its own."""
    out = []
    if header:
        out.append(row_md(header))
        out.append("|" + "---|" * len(header))
    out += [row_md(r) for r in rows]
    return "\n".join(out)
