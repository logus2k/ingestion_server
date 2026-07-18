"""The LLM tier — extraction by inference. The expensive one.

Measured on a real 6-page CV (71 chunks): 1.02s/chunk sustained at 2 concurrent
→ ~72s. This is ~75% of a full run's wall clock. Everything here exists to make
that cost buy as much as possible.

Two findings are baked in:

1. **One prompt, all types.** Four separate per-type prompts cost 2× and found
   LESS (they missed a certification the single prompt caught). Per-type prompts
   also can't see each other's answers, so nothing forces a choice between
   overlapping types — the combined prompt at least has to pick one.
2. **Prompts are generic.** The system prompt is a plain NER analyst with no
   corpus baked in; `corpus.context` is the only coupling and it is a parameter.
   noted hardcoded "a noted MLOps platform documentation" into every extraction,
   which mispriced every chunk of a CV.
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..models import Pipeline
from .base import Context, Entity, LayerResult, Relation, digest_of


def build_prompt(pipeline: Pipeline, wanted_entities: list[str],
                 wanted_relations: list[str], chunk_text: str,
                 section_path: str) -> str:
    """The extraction instruction. Everything corpus-specific arrives as data."""
    lines: list[str] = []
    lines.append(f"Extract named entities from this excerpt of {pipeline.corpus.context}.")
    if section_path and section_path != "(root)":
        # Structure informs the extraction rather than bypassing it: the model
        # is told where the excerpt sits, and can use it.
        lines.append(f"The excerpt sits under the section: {section_path}")
    lines.append("")
    lines.append("Types:")
    for name in wanted_entities:
        t = pipeline.types.entities.get(name)
        if not t:
            continue
        bit = f"  {name} = {t.definition}"
        if t.examples:
            bit += f" Examples: {', '.join(t.examples)}."
        if t.not_:
            # The contrastive half — this is what fixed technology-vs-domain.
            bit += f" NOT: {t.not_}."
        lines.append(bit)

    if wanted_relations:
        lines.append("")
        lines.append("Relationships to extract, ONLY when the excerpt asserts them:")
        for name in wanted_relations:
            r = pipeline.types.relations.get(name)
            if not r:
                continue
            d = f" — {r.definition}" if r.definition else ""
            lines.append(f"  {name}: from a `{r.from_}` to a `{r.to}`{d}")
        # Measured: without this, the model returns the TYPE NAME as the endpoint
        # ("from":"role") instead of the entity's name ("from":"VP Engineering"),
        # and every relation fails to resolve.
        lines.append('`from` and `to` MUST be the `name` of an entity you listed in '
                     '`entities` — copied exactly, character for character. Never put '
                     'a type name there. If either end is not in your entities list, '
                     'omit the relationship.')

    lines.append("")
    lines.append('Return ONLY JSON: {"entities":[{"type":...,"name":...,'
                 '"description":"one sentence stating this entity\'s SPECIFIC facts '
                 'as written in the excerpt — e.g. its issuer, author, date, '
                 'affiliation, role, or relationship to other named things; NOT a '
                 'generic restatement of its type",'
                 '"confidence":0.0-1.0}]'
                 + (',"relations":[{"type":...,"from":...,"to":...}]' if wanted_relations else "")
                 + "}")
    lines.append("Rules:")
    lines.append("- Use the fullest proper name as written in the excerpt. The same "
                 "entity named two ways becomes two entities and breaks the graph.")
    lines.append("- Never list the same name under more than one type. If a name could "
                 "fit two types, choose the single best-fitting type and emit it once.")
    lines.append("- Never invent anything absent from the excerpt.")
    lines.append("- Omit a type entirely if the excerpt has none. An empty list is a "
                 "correct answer.")
    lines.append(f"\n---\n{chunk_text}\n---")
    return "\n".join(lines)


class LLMLayer:
    """One call per chunk, bounded by the LLM client's semaphore."""

    name = "llm"

    async def run(self, ctx: Context) -> LayerResult:
        if ctx.is_delete:
            return LayerResult(digest={"skipped": "delete — nothing to extract"})

        llm = ctx.services["llm"]
        wanted_e = list(ctx.step.entities)
        wanted_r = list(ctx.step.relations)
        if not wanted_e:
            return LayerResult(digest={"skipped": "no entity types requested"})

        valid_e = set(wanted_e)
        valid_r = set(wanted_r)
        floor = 0.0  # confidence floor: see the digest note below

        async def one(ch) -> tuple[list[Entity], list[tuple[str, str, str]]]:
            text = (ch.text or "").strip()
            if not text:
                return [], []
            parsed = await llm.chat_json(
                build_prompt(ctx.pipeline, wanted_e, wanted_r, text, ch.section_path),
                temperature=0.1, max_tokens=4096)
            # The model sometimes returns a bare list instead of the wrapper —
            # list-shaped chunks (bullets, glossaries) bias it that way.
            if isinstance(parsed, list):
                raw_e, raw_r = parsed, []
            elif isinstance(parsed, dict):
                raw_e = parsed.get("entities") or []
                raw_r = parsed.get("relations") or []
            else:
                return [], []
            ents: list[Entity] = []
            for item in raw_e if isinstance(raw_e, list) else []:
                if not isinstance(item, dict):
                    continue
                etype = str(item.get("type") or "").strip().lower()
                name = str(item.get("name") or "").strip()
                if etype not in valid_e or not name:
                    continue
                try:
                    conf = float(item.get("confidence", 1.0))
                except (TypeError, ValueError):
                    conf = 1.0
                if conf < floor:
                    continue
                ents.append(Entity(type=etype, name=name,
                                   description=str(item.get("description") or "").strip(),
                                   confidence=conf))
            rels: list[tuple[str, str, str]] = []
            for item in raw_r if isinstance(raw_r, list) else []:
                if not isinstance(item, dict):
                    continue
                rtype = str(item.get("type") or "").strip()
                src = str(item.get("from") or "").strip()
                dst = str(item.get("to") or "").strip()
                if rtype in valid_r and src and dst:
                    rels.append((rtype, src, dst))
            return ents, rels

        results = await asyncio.gather(*(one(ch) for ch in ctx.chunks))

        # Dedupe across chunks, keeping the highest-confidence description.
        merged: dict[str, Entity] = {}
        relations: list[Relation] = []
        by_name: dict[str, str] = {}   # lowercased name -> entity id, for relation resolution
        mentions = 0

        for ch, (ents, _rels) in zip(ctx.chunks, results):
            chunk_entity = f"markdown_chunk:{ctx.services['chunk_hex'](ch.index)}"
            for e in ents:
                prev = merged.get(e.id)
                if prev is None or e.confidence > prev.confidence:
                    merged[e.id] = e
                by_name[e.name.strip().lower()] = e.id
                relations.append(Relation(src=chunk_entity, dst=e.id, type="mentions",
                                          properties={"confidence": e.confidence},
                                          provenance="llm"))
                mentions += 1

        # Resolve typed relations by name AFTER every chunk is known — a relation
        # can name an entity first seen in a different chunk.
        unresolved = 0
        for ch, (_e, rels) in zip(ctx.chunks, results):
            for rtype, src, dst in rels:
                sid = by_name.get(src.strip().lower())
                did = by_name.get(dst.strip().lower())
                if not sid or not did:
                    unresolved += 1
                    continue
                relations.append(Relation(src=sid, dst=did, type=rtype, provenance="llm"))

        out = list(merged.values())
        d = digest_of(out, relations)
        d["chunks_processed"] = len(ctx.chunks)
        d["mentions"] = mentions
        d["mentions_per_entity"] = round(mentions / max(len(out), 1), 2)
        if unresolved:
            # Surfaced, not swallowed: a relation naming an entity the extractor
            # never returned is a real signal the prompt is inconsistent.
            d["unresolved_relations"] = unresolved
        # NOTE (measured): every entity came back at confidence 1.0 across 14
        # runs of the same chunk, so a confidence floor filters nothing today.
        # The field is kept because it is free and a future model may calibrate;
        # the floor stays at 0.0 rather than pretending it does work.
        d["confidence_floor"] = floor
        return LayerResult(entities=out, relations=relations, digest=d)
