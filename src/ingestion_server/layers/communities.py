"""Communities — PageRank + Louvain, then an LLM summary per cluster.

Not "derived" despite being downstream of it: the summaries need an LLM.

Two things here are hard-won and easy to get wrong:

1. **Louvain, not Leiden.** ArcadeDB's `algo.leiden` does not accept
   `weightProperty`, and unweighted Leiden over-fragments (a real corpus produced
   130 singletons and 3 mega-clusters). Louvain accepts the weight.
2. **`CALL` is Cypher, not SQL.** The SQL parser rejects it outright — "no viable
   alternative at input 'CALL'". Verified on 26.7.2. And Cypher map keys are bare
   identifiers: `{weightProperty: 'weight'}` parses, `{'weightProperty': 'weight'}`
   is a syntax error.

This layer is the one place a layer must touch the database before commit: the
graph algorithms run server-side, over persisted vertices. It therefore operates
on a **staging graph** the engine has already written under a run-scoped tag, and
the engine drops that tag if the run is abandoned.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from .base import Context, Entity, LayerResult, Relation, digest_of

_SUMMARY_SCHEMA = {
    "name": "community_summary",
    "schema": {
        "type": "object",
        "properties": {"title": {"type": "string"}, "summary": {"type": "string"}},
        "required": ["title", "summary"],
    },
}


class CommunitiesLayer:
    name = "communities"

    async def run(self, ctx: Context) -> LayerResult:
        if ctx.is_delete:
            return LayerResult(digest={"skipped": "delete — communities recomputed at commit"})

        ents = list(ctx.entities.values())
        if len(ents) < 2:
            return LayerResult(digest={"skipped": f"{len(ents)} entities — nothing to cluster"})

        db = ctx.services.get("db")
        if db is None:
            return LayerResult(digest={"skipped": "no database — clustering needs the staged graph"})

        assign = await _louvain(db, ctx.services.get("run_tag", ""))
        # Thematic entities only, renumbered 0-indexed and contiguous.
        theme = {e.id for e in ents}
        t_assign = {k: v for k, v in assign.items() if k in theme}
        if not t_assign:
            return LayerResult(digest={"skipped": "louvain assigned no known entities"})

        renum = {old: new for new, old in enumerate(sorted(set(t_assign.values())))}
        members: dict[int, list[Entity]] = {}
        by_id = {e.id: e for e in ents}
        for eid, old in t_assign.items():
            members.setdefault(renum[old], []).append(by_id[eid])

        out_entities: list[Entity] = []
        out_relations: list[Relation] = []
        for cnum, mem in members.items():
            cid = f"community:{cnum}"
            out_entities.append(Entity(type="community", name=f"Community {cnum}",
                                       description=f"{len(mem)} members"))
            for m in mem:
                out_relations.append(Relation(src=m.id, dst=cid, type="member_of",
                                              properties={"weight": 0.0},
                                              provenance="derived"))

        summaries = await _summarise(ctx, members)
        # Communities with fewer than 2 members are skipped deliberately — that is
        # why a proven corpus had 14 communities but only 13 summaries.
        for cnum, v in summaries.items():
            out_entities.append(Entity(type="community_summary", name=v["title"],
                                       description=v["summary"]))
            out_relations.append(Relation(src=f"community_summary:{v['title']}",
                                          dst=f"community:{cnum}", type="summarizes",
                                          properties={"weight": 0.0}, provenance="llm"))

        d = digest_of(out_entities, out_relations)
        d["communities"] = len(members)
        d["summaries"] = len(summaries)
        d["skipped_singletons"] = sum(1 for m in members.values() if len(m) < 2)
        d["sizes"] = sorted((len(m) for m in members.values()), reverse=True)[:10]
        return LayerResult(entities=out_entities, relations=out_relations, digest=d)


async def _louvain(db, run_tag: str) -> dict[str, int]:
    """PageRank (stored on the vertex) + Louvain (community assignment).

    Both procedures run over the WHOLE database — they take no projection filter.
    That is safe only because a pipeline owns its database; it is the reason
    `corpus.target_db` is required rather than optional.
    """
    try:
        await db.cypher("CALL algo.pagerank({}) YIELD node, rank SET node.pagerank = rank")
    except Exception:
        # PageRank is enrichment, not structure — a failure here must not cost
        # the clustering. Deliberately swallowed, and reported in the digest.
        pass
    rows = await db.cypher(
        "CALL algo.louvain({weightProperty: 'weight'}) "
        "YIELD node, communityId RETURN node.id AS id, communityId AS c") or []
    return {r["id"]: int(r["c"]) for r in rows if r.get("id") and r.get("c") is not None}


async def _summarise(ctx: Context, members: dict[int, list[Entity]]) -> dict[int, dict]:
    llm = ctx.services.get("summary_llm") or ctx.services["llm"]
    targets = {c: m for c, m in members.items() if len(m) >= 2}
    if not targets:
        return {}

    async def one(item):
        cnum, mem = item
        ent_lines = "\n".join(
            f"- ({e.type}) {e.name}: {e.description}".rstrip() for e in mem[:60])
        prompt = (
            f"These entities form one tightly-connected cluster of a knowledge graph "
            f"built over {ctx.pipeline.corpus.context}.\n\n"
            f"ENTITIES:\n{ent_lines}\n\n"
            'Return ONLY JSON with:\n'
            '  `title` — a short noun phrase naming what binds this cluster\n'
            '  `summary` — 2-4 sentences on what this part of the material covers, '
            'grounded strictly in the entities above. Never invent.')
        parsed = await llm.chat_json(prompt, temperature=0.2, max_tokens=1024,
                                     json_schema=_SUMMARY_SCHEMA)
        if not isinstance(parsed, dict):
            return cnum, None
        title = str(parsed.get("title") or "").strip()
        summary = str(parsed.get("summary") or "").strip()
        if not summary:
            return cnum, None
        return cnum, {"title": title or f"Community {cnum}", "summary": summary}

    results = await asyncio.gather(*(one(i) for i in targets.items()))
    return {c: v for c, v in results if v}
