"""Adapter: the ported docling parser -> the service's Chunk.

`pdf_docling.py` is carried over almost verbatim from the CV prototype, which
was itself ported from a parser tuned against this exact document over many
iterations — the heading-hierarchy rebuild, the per-bullet split for the
certifications list, the per-item (not per-page-union) regions, and the
super-parent chunks all exist because of concrete failures on it.

It is left alone deliberately. Editing 829 lines of proven parser to change a
dataclass name is how you lose that tuning. This maps its output instead.

Decision record: three attempts to replace docling with markdown-chunking +
PDF text-position matching located only 46–65% of chunks' boxes. Docling
produces 71 chunks with correct regions on the same PDF. The section text is in
the PDF; the *hierarchy* is not — a PDF stores positioned glyphs, not structure —
so recovering it needs either this model or heuristics that proved worse.
"""
from __future__ import annotations

from ..layers.base import Chunk
from .pdf_docling import scan_pdf


def scan(path: str, target_tokens: int = 200) -> list[Chunk]:
    import os
    # The parser reads its target from the environment (it predates the service's
    # config); set it here so `chunking.target_tokens` in the pipeline wins.
    os.environ["CHUNK_TARGET_TOKENS"] = str(target_tokens)

    raw = scan_pdf(path, repo_root=os.path.dirname(path))
    out: list[Chunk] = []
    for c in raw:
        out.append(Chunk(
            index=c.chunk_index,
            text=c.text,
            section_path=c.section_path,
            token_count=c.token_count,
            page_no=c.page_no,
            bbox=c.bbox,
            regions=c.regions,
            kind=c.kind,
        ))
    return out
