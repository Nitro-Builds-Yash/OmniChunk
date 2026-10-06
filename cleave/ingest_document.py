"""Document ingestion: Docling → list[ContentElement].

Docling is the AST parser for documents; this module flattens its typed tree
into Cleave's universal intermediate representation, keeping reading order,
heading ancestry, table grids, and caption references — everything the graph
and router need, nothing else.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .markdown import table_markdown as _table_markdown
from .models import ContentElement, count_tokens, sha256_of

log = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
_DOC_EXTS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".html",
    ".htm",
    ".md",
    ".txt",
    ".asciidoc",
} | _IMAGE_EXTS
_SPREADSHEET_EXTS = {".xlsx", ".csv"}


@dataclass(slots=True)
class IngestResult:
    elements: list[ContentElement]
    title: str | None
    source_uri: str
    sha256: str
    warnings: list[str] = field(default_factory=list)
    cleaning: dict | None = None  # what normalisation changed, by rule


_converter = None


def _get_converter():
    """Lazy singleton: Docling pulls torch on import and loads CV models on
    first PDF, so both happen once and only when a document actually arrives."""
    global _converter
    if _converter is None:
        from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
        from docling.datamodel.pipeline_options import (  # noqa: PLC0415
            AcceleratorOptions,
            PdfPipelineOptions,
        )
        from docling.document_converter import (  # noqa: PLC0415
            DocumentConverter,
            PdfFormatOption,
        )

        # Default num_threads is 4; this is a 14-core machine.
        pipeline = PdfPipelineOptions()
        pipeline.accelerator_options = AcceleratorOptions(num_threads=10)
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
        )
    return _converter


_NUMBERING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")


def _heading_number(text: str) -> str | None:
    """The section number a heading declares, if it declares one: "3.2.1"."""
    m = _NUMBERING_RE.match(text.strip())
    return m.group(1) if m else None


def _prov(item) -> tuple[int | None, tuple[float, float, float, float] | None]:
    """Page and bbox, when Docling recorded them.

    Debug rather than warning: plenty of items legitimately carry no
    provenance, and the caller has a designed fallback."""
    try:
        p = item.prov[0]
        bbox = p.bbox
        return p.page_no, (float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b))
    except Exception as exc:
        log.debug("no provenance for %s: %s", type(item).__name__, exc)
        return None, None


def _grid_texts(table) -> list[list[str]]:
    try:
        return [[(c.text or "").strip() for c in row] for row in table.data.grid]
    except Exception as exc:
        # This one loses data: the table survives as text but its cells do not.
        log.warning("table grid unreadable (%s) — table kept as text only", exc)
        return []


def _header_row(table) -> list[str]:
    """First row of column headers if Docling marked them, else first row."""
    try:
        for row in table.data.grid:
            if any(getattr(c, "column_header", False) for c in row):
                return [(c.text or "").strip() for c in row]
    except Exception as exc:
        log.debug("no marked header row (%s); falling back to the first row", exc)
    grid = _grid_texts(table)
    return grid[0] if grid else []


def ingest_document(path: str | Path) -> IngestResult:
    path = Path(path)
    if path.suffix.lower() not in _DOC_EXTS:
        raise ValueError(f"unsupported document type: {path.suffix}")

    result = _get_converter().convert(str(path))
    doc = result.document
    warnings: list[str] = []

    elements: list[ContentElement] = []
    by_selfref: dict[str, str] = {}  # docling "#/texts/3" → our element id
    heading_stack: list[tuple[int, str, str | None]] = []  # (level, id, section number)
    title: str | None = None
    counter = 0

    def new_id() -> str:
        nonlocal counter
        eid = f"el_{counter:04d}"
        counter += 1
        return eid

    def current_parent() -> str | None:
        return heading_stack[-1][1] if heading_stack else None

    from docling_core.types.doc import (  # noqa: PLC0415
        DocItemLabel,
        PictureItem,
        TableItem,
        TextItem,
    )

    SKIP = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.FOOTNOTE}

    for item, _level in doc.iterate_items():
        page, bbox = _prov(item)

        if isinstance(item, TableItem):
            grid = _grid_texts(item)
            eid = new_id()
            el = ContentElement(
                id=eid,
                kind="table",
                text=_table_markdown(grid),
                parent_id=current_parent(),
                page=page,
                bbox=bbox,
                meta={
                    "grid": grid,
                    "header_row": _header_row(item),
                    "caption_crefs": [getattr(c, "cref", str(c)) for c in item.captions],
                },
            )
            elements.append(el)
            by_selfref[item.self_ref] = eid
            continue

        if isinstance(item, PictureItem):
            eid = new_id()
            el = ContentElement(
                id=eid,
                kind="figure",
                text="",
                parent_id=current_parent(),
                page=page,
                bbox=bbox,
                meta={"caption_crefs": [getattr(c, "cref", str(c)) for c in item.captions]},
            )
            elements.append(el)
            by_selfref[item.self_ref] = eid
            continue

        if not isinstance(item, TextItem):
            continue
        label = item.label
        text = (item.text or "").strip()
        if not text or label in SKIP:
            continue

        if label == DocItemLabel.TITLE:
            title = title or text
            eid = new_id()
            heading_stack.clear()
            elements.append(
                ContentElement(
                    id=eid,
                    kind="heading",
                    text=text,
                    level=1,
                    parent_id=None,
                    page=page,
                    bbox=bbox,
                )
            )
            heading_stack.append((1, eid, None))
            by_selfref[item.self_ref] = eid
        elif label == DocItemLabel.SECTION_HEADER:
            eid = new_id()
            number = _heading_number(text)
            if number:
                # A numbered heading belongs under the heading whose number is
                # its prefix — "3.2.1" under "3.2" — regardless of anything
                # detected in between.
                while heading_stack and not (
                    heading_stack[-1][2] and number.startswith(heading_stack[-1][2] + ".")
                ):
                    heading_stack.pop()
                lvl = number.count(".") + 2  # level 1 is the document title
                push = True
            else:
                # Unnumbered headings are siblings of each other, so drop any
                # unnumbered heading still open. What remains is a numbered
                # ancestor, if any: this heading sits under it as a leaf and is
                # NOT pushed, so a figure label promoted to a heading by layout
                # detection cannot adopt the numbered sections that follow it.
                while heading_stack and heading_stack[-1][2] is None:
                    heading_stack.pop()
                lvl = (heading_stack[-1][0] + 1) if heading_stack else 2
                push = not heading_stack
            elements.append(
                ContentElement(
                    id=eid,
                    kind="heading",
                    text=text,
                    level=lvl,
                    parent_id=current_parent(),
                    page=page,
                    bbox=bbox,
                )
            )
            if push:
                heading_stack.append((lvl, eid, number))
            by_selfref[item.self_ref] = eid
        else:
            kind = {
                DocItemLabel.LIST_ITEM: "list_item",
                DocItemLabel.CODE: "code",
                DocItemLabel.CAPTION: "caption",
            }.get(label, "paragraph")
            eid = new_id()
            elements.append(
                ContentElement(
                    id=eid,
                    kind=kind,
                    text=text,
                    parent_id=current_parent(),
                    page=page,
                    bbox=bbox,
                )
            )
            by_selfref[item.self_ref] = eid

    # Resolve caption crefs → element ids; synthesize a caption element when the
    # referenced text never surfaced in traversal (rare, but cheap to cover).
    for el in elements:
        if el.kind not in ("table", "figure"):
            continue
        cap_ids: list[str] = []
        for cref in el.meta.pop("caption_crefs", []):
            cid = by_selfref.get(cref)
            if cid:
                cap_ids.append(cid)
            else:
                warnings.append(f"{el.id}: caption ref {cref} not found in traversal")
        el.meta["caption_ids"] = cap_ids

    if path.suffix.lower() in _SPREADSHEET_EXTS:
        title = title or path.stem
        _label_sheets(path, elements, warnings)
    elif path.suffix.lower() in _IMAGE_EXTS:
        title = title or path.stem

    # Normalise before anything measures or splits: token counts, routing
    # signals and boundaries should all describe the text that will actually be
    # stored and embedded.
    from .cleaning import clean_elements  # noqa: PLC0415

    report = clean_elements(elements)
    elements = [e for e in elements if e.text or e.kind in ("figure", "table")]
    if not elements and path.suffix.lower() in _IMAGE_EXTS:
        elements.append(
            ContentElement(
                id="el_0000",
                kind="figure",
                text=f"[{path.name} visual diagram / image]",
                page=1,
                meta={"is_image": True},
            )
        )
    if title:
        title = clean_text_value(title)

    if not elements:
        warnings.append("document produced no elements")
    log.info(
        "ingested %s: %d elements, title=%r — %s", path.name, len(elements), title, report.summary()
    )

    return IngestResult(
        elements=elements,
        title=title,
        source_uri=str(path),
        sha256=sha256_of(str(path)),
        warnings=warnings,
        cleaning=report.to_dict(),
    )


def clean_text_value(text: str) -> str:
    from .cleaning import clean_text  # noqa: PLC0415

    return clean_text(text)[0]


def _label_sheets(path: Path, elements: list[ContentElement], warnings: list[str]) -> None:
    """Recover sheet identity, which Docling drops.

    A row group is meaningless without knowing which sheet it came from, but
    Docling represents each XLSX sheet only as a page number. openpyxl gives us
    the names in workbook order, so page N maps to sheet N-1.
    """
    names: list[str] = []
    if path.suffix.lower() == ".xlsx":
        try:
            import openpyxl  # noqa: PLC0415

            wb = openpyxl.load_workbook(path, read_only=True)
            names = list(wb.sheetnames)
            wb.close()
        except Exception as exc:
            warnings.append(f"could not read sheet names ({exc}); tables stay unlabelled")

    for el in elements:
        if el.kind != "table":
            continue
        if names and el.page and 1 <= el.page <= len(names):
            el.meta["sheet"] = names[el.page - 1]
        elif path.suffix.lower() == ".csv":
            el.meta["sheet"] = None  # a CSV is a single unnamed table
    if names:
        log.info("labelled %d sheet(s) in %s: %s", len(names), path.name, names)


def total_tokens(elements: list[ContentElement]) -> int:
    return sum(count_tokens(e.text) for e in elements)
