"""The derived tier — embeddings and graph maths. No inference, no LLM.

Two jobs:
  * embed each entity so graph seed search works (`vector.neighbors` over
    Entity[embedding]), and
  * relate entities to each other by cosine similarity, banded.

The community/summary pass lives in `communities.py` — it needs an LLM, so it is
not "derived" despite being downstream of this.
"""
from __future__ import annotations

import math
from typing import Any

from .base import Context, Entity, LayerResult, Relation, digest_of

# Cosine bands. A band is a real edge property, not a substring of a JSON blob —
# noted matched `properties_json CONTAINS '"confidence": "high"'`, which is a
# string match against json.dumps' exact spacing and breaks if anything reformats.
BANDS = (("high", 0.85, 1.0), ("medium", 0.75, 0.85), ("low", 0.65, 0.75))
BAND_WEIGHT = {"high": 1.0, "medium": 0.5, "low": 0.2}


class DerivedLayer:
    name = "derived"

    async def run(self, ctx: Context) -> LayerResult:
        if ctx.is_delete:
            return LayerResult(digest={"skipped": "delete — nothing to derive"})

        # Every entity accumulated so far, from every previous layer.
        ents: list[Entity] = list(ctx.entities.values())
        if not ents:
            return LayerResult(digest={"skipped": "no entities to relate"})

        embedder = ctx.services["embedder"]

        # "name. description" is the entity's searchable surface.
        texts = [f"{e.name}. {e.description}".strip() for e in ents]
        vectors, _ = await embedder.embed_batched(texts, dense=True, sparse=False)
        if len(vectors) != len(ents):
            raise RuntimeError(
                f"entity embed returned {len(vectors)} vectors for {len(ents)} entities")
        for e, v in zip(ents, vectors):
            e.embedding = [float(x) for x in v]

        threshold = float(ctx.step.threshold)
        floor = min(threshold, BANDS[-1][1])

        relations: list[Relation] = []
        if "SIMILAR_TO" in (ctx.step.relations or []):
            relations = _similar_to(ents, floor)

        d = digest_of(ents, relations)
        d["embedded"] = len(ents)
        d["threshold"] = threshold
        d["similar_to"] = len(relations)
        # The ratio is the signal worth watching: a proven corpus ran ~0.9
        # similarity edges per entity; a first attempt at 0.65 produced ~5 per
        # entity (1935 edges over 395), which floods Louvain and doubles the
        # community count. This is what a judge should catch.
        d["similar_to_per_entity"] = round(len(relations) / max(len(ents), 1), 2)
        by_band: dict[str, int] = {}
        for r in relations:
            b = r.properties.get("similarity_band", "?")
            by_band[b] = by_band.get(b, 0) + 1
        d["similar_to_by_band"] = by_band
        return LayerResult(entities=ents, relations=relations, digest=d)


def _similar_to(ents: list[Entity], floor: float) -> list[Relation]:
    """Pairwise cosine over entity embeddings.

    embeddings-server returns L2-normalised vectors, so a dot product IS the
    cosine — no division. O(n²) is fine at these sizes (hundreds of entities);
    if a corpus ever pushes this into thousands, it becomes an ANN query against
    the Entity vector index instead.
    """
    out: list[Relation] = []
    n = len(ents)
    for i in range(n):
        vi = ents[i].embedding
        if not vi:
            continue
        for j in range(i + 1, n):
            vj = ents[j].embedding
            if not vj:
                continue
            dot = sum(a * b for a, b in zip(vi, vj))
            if math.isnan(dot) or dot < floor:
                continue
            band = next((name for name, lo, hi in BANDS if lo <= dot < hi), None)
            if band is None:
                band = "high" if dot >= BANDS[0][1] else None
            if band is None:
                continue
            out.append(Relation(
                src=ents[i].id, dst=ents[j].id, type="SIMILAR_TO",
                properties={"similarity": round(float(dot), 5),
                            "similarity_band": band,
                            "weight": BAND_WEIGHT[band]},
                provenance="derived"))
    return out
