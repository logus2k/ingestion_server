"""The layer contract.

Every layer — built-in tier or custom module — implements the same interface,
which is what makes them orderable and interchangeable:

    async def run(ctx: Context) -> LayerResult

**Layers are pure: they write nothing.** They read the context and return what
they found. The engine accumulates results in staging and commits ONCE, at the
end, atomically. That is what makes "repeat this layer" safe, lets a suspended
run sit for hours without leaving a half-built graph on disk, and keeps a
document's additions and removals in a single transaction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from ..models import Change, Document, Pipeline, Step


@dataclass
class Chunk:
    """One unit of retrievable text, with provenance back to the source."""
    index: int
    text: str
    section_path: str
    token_count: int = 0
    page_no: Optional[int] = None
    bbox: Optional[list[float]] = None
    # A chunk can span pages, and one page can hold several disjoint items, so
    # regions is a list — not a single box. page_no/bbox mirror regions[0].
    regions: Optional[list[dict]] = None
    kind: str = "text"


@dataclass
class Entity:
    """Deduped by (type, canonical name) across the whole document."""
    type: str
    name: str
    description: str = ""
    confidence: float = 1.0
    # Set by the derived tier; used for graph seed search.
    embedding: Optional[list[float]] = None

    @property
    def id(self) -> str:
        import re
        key = re.sub(r"\s+", " ", (self.name or "").strip().lower())
        return f"{self.type}:{key}"


@dataclass
class Relation:
    src: str            # entity id, or a chunk entity id
    dst: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)
    # Which layer asserted this. Lets you see where a fact came from, re-run one
    # tier without the others, and A/B whether the LLM tier adds anything over
    # what structure already gives for free.
    provenance: str = ""


@dataclass
class Context:
    """Everything a layer can see. Later layers see earlier layers' output, so
    ordering is meaningful: structural runs first and hands the LLM tier the
    organizations it already found, rather than paying to rediscover them."""
    pipeline: Pipeline
    document: Document
    step: Step
    chunks: list[Chunk]
    # Accumulated so far, across previous layers.
    entities: dict[str, Entity]
    relations: list[Relation]
    # Services, injected — a custom layer gets the same ones the built-ins use.
    services: dict[str, Any] = field(default_factory=dict)

    @property
    def is_delete(self) -> bool:
        return self.document.change == Change.deleted


@dataclass
class LayerResult:
    """What a layer contributes. Additive: the engine merges, layers never
    mutate the corpus or each other's output."""
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    # Entity-typed values promoted to indexed chunk properties (see index:).
    chunk_props: dict[int, dict[str, Any]] = field(default_factory=dict)
    # What the judge sees: counts, distributions, and a real sample.
    digest: dict[str, Any] = field(default_factory=dict)


class Layer(Protocol):
    name: str

    async def run(self, ctx: Context) -> LayerResult: ...


def digest_of(entities: list[Entity], relations: list[Relation],
              sample_size: int = 12) -> dict[str, Any]:
    """The standard digest. Counts AND distribution AND a sample — the judge
    needs all three. A type distribution is where a degenerate extraction shows
    up instantly (200 `concept` vs 25 `organization`); a raw dump hides it, and
    counts alone can't show quality."""
    by_type: dict[str, int] = {}
    for e in entities:
        by_type[e.type] = by_type.get(e.type, 0) + 1
    by_rel: dict[str, int] = {}
    for r in relations:
        by_rel[r.type] = by_rel.get(r.type, 0) + 1
    return {
        "entities": len(entities),
        "entities_by_type": by_type,
        "relations": len(relations),
        "relations_by_type": by_rel,
        "sample": [
            {"type": e.type, "name": e.name, "description": e.description[:120],
             "confidence": e.confidence}
            for e in entities[:sample_size]
        ],
    }
