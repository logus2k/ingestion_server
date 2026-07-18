"""The structural tier — what the document already states, for free.

No inference, no regexes hand-written per corpus. The document's own structure
(its section path) carries facts that an LLM would otherwise be paid to
rediscover: everything under `TECH5 · Portugal > VP Engineering` relates to that
employer and that role by *position*, not by inference.

The section path is passed to the LLM tier as context too — structure informs
the extraction rather than bypassing it. This tier only claims what the path
literally names.
"""
from __future__ import annotations

import re
from typing import Any

from .base import Context, Entity, LayerResult, Relation, digest_of

# Docling and markdown scanners both join a heading chain with ' > '.
_SEP = ">"


def _segments(section_path: str) -> list[str]:
    if not section_path or section_path == "(root)":
        return []
    return [s.strip() for s in section_path.split(_SEP) if s.strip()]


class StructuralLayer:
    """Derives entities from section headings.

    Deliberately conservative: it emits an entity per heading segment for the
    types the step asks for, and lets the LLM tier disambiguate. It does NOT try
    to guess which segment is an organization and which is a role — that is a
    semantic call, and guessing it with patterns is exactly the "replace the
    LLM's work with regexps" trap.

    What it DOES give for free, and reliably: the co-location fact. Every entity
    found in a chunk under a heading is related to that heading's entities.
    """

    name = "structural"

    async def run(self, ctx: Context) -> LayerResult:
        if ctx.is_delete:
            return LayerResult(digest={"skipped": "delete — nothing to extract"})

        wanted = set(ctx.step.entities)
        if not wanted:
            return LayerResult(digest={"skipped": "no entity types requested"})

        entities: dict[str, Entity] = {}
        relations: list[Relation] = []
        # heading text -> the chunks that sit under it
        seen_headings: dict[str, list[int]] = {}

        for ch in ctx.chunks:
            segs = _segments(ch.section_path)
            if not segs:
                continue
            for seg in segs:
                seen_headings.setdefault(seg, []).append(ch.index)

        # A heading is evidence, not a typed entity. We surface each heading once
        # per requested type ONLY when the pipeline declares exactly one type for
        # this step — otherwise we would be inventing the type assignment.
        if len(wanted) == 1:
            etype = next(iter(wanted))
            for heading, chunk_ids in seen_headings.items():
                e = Entity(type=etype, name=heading,
                           description=f"Section heading in {ctx.document.display_name()}.",
                           confidence=1.0)
                entities[e.id] = e
        else:
            # Multiple types requested: emit nothing rather than guess. The
            # headings still reach the LLM tier as chunk context.
            return LayerResult(digest={
                "skipped": (f"{len(wanted)} entity types requested "
                            f"({', '.join(sorted(wanted))}) — the structural tier "
                            "cannot assign a type to a heading without guessing; "
                            "headings are passed to the llm tier as context instead"),
                "headings_seen": len(seen_headings),
            })

        out = list(entities.values())
        d = digest_of(out, relations)
        d["headings_seen"] = len(seen_headings)
        return LayerResult(entities=out, relations=relations, digest=d)
