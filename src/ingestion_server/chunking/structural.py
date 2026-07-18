"""Structural chunking — one chunk per document item, un-merged.

WHY THIS EXISTS (and why it is not `pdf_docling`)
-------------------------------------------------
`pdf_docling` uses Docling's HybridChunker, a *retrieval* chunker: it merges
adjacent doc items across page breaks and bags whole lists into a single chunk,
sized to a token budget. That is exactly right for search, and destructive for
consumers that need the document's *structure*:

  * requirement segmentation — one requirement per block; a merged list is
    unusable, and a table must stay one unit so it can be routed to a
    table-aware identifier;
  * faithful document reconstruction — replaying the original block stream in
    order, substituting corrected text, requires the blocks to exist.

So `structural` walks `DoclingDocument.iterate_items()` and emits one `Chunk`
per item, un-merged and in document order, carrying a normalized block type in
`kind` plus page/bbox provenance. `target_tokens` is accepted for interface
symmetry and deliberately ignored — structure, not token budget, decides the
boundaries.

Ported from the reqoach analyst parser (`src/reqqa/ingest/docling_adapter.py`),
which was tuned against real SRS documents. The Docling converter is reused from
`pdf_docling` so the two strategies share one cached model load.
"""
from __future__ import annotations

import logging

from ..layers.base import Chunk

logger = logging.getLogger(__name__)

# Docling `DocItemLabel` -> normalized block vocabulary carried in `Chunk.kind`.
_LABEL_TO_KIND = {
    "title": "heading",
    "section_header": "heading",
    "list_item": "list_item",
    "table": "table",
    "picture": "picture",
    "caption": "caption",
    "code": "code",
    "formula": "code",
    "document_index": "toc",
    "text": "paragraph",
    "paragraph": "paragraph",
    "footnote": "paragraph",
    "reference": "paragraph",
}


def _normalize_text(s: str) -> str:
    """Map symbol-font private-use glyphs to a real bullet.

    Docling emits Word/PDF symbol-font bullets in the U+F000–U+F0FF remap block
    (e.g. U+F0B7), which render blank downstream and read as stray spaces.
    """
    if not s:
        return s
    return "".join("•" if 0xF000 <= ord(c) <= 0xF0FF else c for c in s)


def _kind_for(label) -> str:
    name = (getattr(label, "value", None) or str(label)).lower()
    return _LABEL_TO_KIND.get(name, "other")


def _regions_for(item) -> list[dict]:
    """All provenance entries as regions: [{page_no, bbox[x0,y0,x1,y1]}, ...].

    An item can span pages, so we keep every entry rather than only the first
    (the service's convention: `page_no`/`bbox` mirror `regions[0]`).
    """
    out: list[dict] = []
    for p in (getattr(item, "prov", None) or []):
        page_no = getattr(p, "page_no", None)
        bb = getattr(p, "bbox", None)
        bbox = None
        if bb is not None:
            try:
                bbox = [float(v) for v in bb.as_tuple()]
            except Exception:  # noqa: BLE001 — a malformed box must not drop the block
                bbox = None
        if page_no is None and bbox is None:
            continue
        out.append({"page_no": int(page_no) if page_no is not None else None,
                    "bbox": bbox})
    return out


def scan(path: str, target_tokens: int = 200) -> list[Chunk]:
    """Parse `path` into un-merged per-item blocks.

    `target_tokens` is ignored — see the module docstring.
    """
    from docling_core.types.doc import DocItemLabel

    from . import pdf_docling

    pdf_docling._ensure_loaded()          # reuse the cached converter + its config
    converter = pdf_docling._converter
    doc = converter.convert(path).document

    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []   # (tree depth, title)
    index = 0

    def section_path() -> str:
        return " > ".join(t for _, t in heading_stack) or "(root)"

    for node, level in doc.iterate_items():
        label = getattr(node, "label", None)
        if label is None:
            continue
        kind = _kind_for(label)

        # Tables render to markdown so their structure survives as text.
        if label == DocItemLabel.TABLE:
            try:
                text = node.export_to_markdown(doc)
            except Exception as e:  # noqa: BLE001
                logger.warning("structural: table export failed: %s", e)
                text = ""
        else:
            text = getattr(node, "text", None) or ""
        text = _normalize_text(text).strip()

        # Heading stack drives section_path for every following block.
        if kind == "heading" and text:
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))

        if not text:
            continue

        regions = _regions_for(node)
        first = regions[0] if regions else {}
        chunks.append(Chunk(
            index=index,
            text=text,
            section_path=section_path(),
            token_count=len(text.split()),   # informative only; not a chunking knob
            page_no=first.get("page_no"),
            bbox=first.get("bbox"),
            regions=regions or None,
            kind=kind,
        ))
        index += 1

    logger.info("structural: %s -> %d blocks", path, len(chunks))
    return chunks
