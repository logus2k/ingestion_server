"""plain_text — chunk a text/markdown file by structure.

No ML, no models, no container weight. For markdown it reads the real heading
hierarchy rather than inferring one, which is strictly better than recovering it
from a rendered document: a layout model flattens headings and then has to guess
ancestry back from positional cues.

No bounding boxes: a text file has no geometry. Chunks carry `regions: None`, and
a consumer that wants PDF highlighting must use a strategy that produces them.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..layers.base import Chunk

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Rough, and honest about it: the real tokenizer lives in embeddings-server. This
# only decides where to split, and being ~15% off costs nothing here.
_CHARS_PER_TOKEN = 4


def _tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def scan(path: str, target_tokens: int = 200) -> list[Chunk]:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    doc_name = Path(path).name

    # Walk once, tracking the heading stack, so section_path is the REAL ancestry.
    stack: list[tuple[int, str]] = []
    blocks: list[tuple[str, str]] = []   # (section_path, text)
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        text = "\n".join(buf).strip()
        if text:
            blocks.append((_path_of(stack), text))
        buf.clear()

    for line in raw.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
            continue
        buf.append(line)
    flush()

    # Split oversized blocks; merge undersized neighbours within a section.
    chunks: list[Chunk] = []
    for section, text in blocks:
        for part in _split(text, target_tokens):
            # The section path is prefixed into the text deliberately: the
            # extractor is measurably better at naming the employer/role when the
            # heading is IN the excerpt, not only alongside it.
            body = f"{section}\n{part}" if section and section != "(root)" else part
            chunks.append(Chunk(index=len(chunks), text=body, section_path=section,
                                token_count=_tokens(body)))
    return chunks


def _path_of(stack: list[tuple[int, str]]) -> str:
    return " > ".join(t for _, t in stack) or "(root)"


def _split(text: str, target_tokens: int) -> list[str]:
    """Split on paragraph boundaries, packing up to target_tokens."""
    if _tokens(text) <= target_tokens:
        return [text]
    out: list[str] = []
    cur: list[str] = []
    size = 0
    for para in re.split(r"\n\s*\n", text):
        p = para.strip()
        if not p:
            continue
        t = _tokens(p)
        if size and size + t > target_tokens:
            out.append("\n\n".join(cur))
            cur, size = [], 0
        cur.append(p)
        size += t
    if cur:
        out.append("\n\n".join(cur))
    return out or [text]
