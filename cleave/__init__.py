"""Cleave — universal chunking and information extraction.

Understand information before you cut it: profile the input, map the
relationships inside it, split along the grain, and spend AI only where
structure cannot supply the missing context.

    from cleave import chunk_file
    units = chunk_file("report.pdf")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"
__all__ = ["ChunkResult", "chunk_file", "chunk_file_detailed", "chunk_files"]

if TYPE_CHECKING:
    from .api import ChunkResult, chunk_file, chunk_file_detailed, chunk_files


def __getattr__(name: str) -> Any:
    # Lazy: `import cleave` must not pull in Docling and torch.
    if name in __all__:
        from . import api  # noqa: PLC0415

        return getattr(api, name)
    raise AttributeError(f"module 'cleave' has no attribute {name!r}")
