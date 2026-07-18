"""Chunking strategies — document → chunks with provenance.

A strategy is chosen per pipeline (`chunking.strategy`), because it is a property
of the corpus: bulas are PDFs that need layout recovery; a CV is rendered from
markdown we already hold.

  plain_text     text/markdown → chunks with REAL heading ancestry. No models,
                 no geometry (regions=None), instant. For corpora where citations
                 don't need to highlight a page.
  pdf_docling    PDF/DOCX/PPTX/HTML → chunks with per-item bounding boxes.
                 Needs docling (~1.2GB of layout models). Imported lazily so a
                 deployment that never ingests PDFs pays nothing.

`markdown_render` is declared in the model but NOT implemented, and the reason is
worth recording so nobody re-litigates it from first principles:

The appeal is real — a PDF we generate ourselves comes from source that already
states its structure, so paying a layout model to infer that structure back looks
absurd. Chunking from markdown IS better and free. The problem is boxes. A PDF
stores positioned glyphs, not structure, so the only ways to get a chunk's
rectangle are (a) a layout model, or (b) matching the chunk's text to the PDF's
word positions. (b) was measured three ways on the CV and located only 46–65% of
chunks — the misses being text the markdown renders differently (label runs, link
rows, `·`-joined stacks) or not at all (HTML comments). Docling gets 71/71 with
correct regions on the same file.

So: docling, over the generated PDF, until someone funds a proper global
alignment for (b). Chunking stays per-pipeline, so this is a corpus's choice.
"""
from __future__ import annotations

from typing import Callable

from ..layers.base import Chunk


def get_chunker(strategy: str) -> Callable[..., list[Chunk]]:
    if strategy == "plain_text":
        from .plain_text import scan
        return scan
    if strategy == "pdf_docling":
        # Lazy: importing docling pulls torch. A plain_text-only deployment must
        # not pay that at startup.
        from .pdf_docling_adapter import scan
        return scan
    if strategy == "markdown_render":
        raise NotImplementedError(
            "chunking.strategy 'markdown_render' is not implemented yet. It would "
            "emit chunks and bounding boxes at render time from the document's own "
            "source, rather than recovering them from a rendered PDF. Use "
            "'pdf_docling' (bboxes, needs models) or 'plain_text' (no geometry).")
    raise ValueError(f"unknown chunking strategy '{strategy}'")
