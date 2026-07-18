"""Pipeline builder — the declarative config, assembled in Python.

A pipeline is plain JSON and you can write it by hand. This exists so you get
argument names and a validate() before you spend minutes on a run.

    from ingestion_client import Pipeline

    p = (Pipeline("a curriculum vitae", target_db="cv", language="en")
         .entity("organization", "a named company, employer or certifying body",
                 examples=["Acme Corp"])
         .entity("role", "a job title held by the person", examples=["Senior Engineer"])
         .relation("AT_ORGANIZATION", "role", "organization")
         .llm(entities=["organization", "role"], relations=["AT_ORGANIZATION"])
         .derived()
         .build())
"""
from __future__ import annotations

from typing import Any, Optional


class Pipeline:
    def __init__(self, context: str, target_db: str, language: str = "en",
                 strategy: str = "pdf_docling", target_tokens: int = 200,
                 extraction_agent: str = "ingest_extractor") -> None:
        """
        context   what the corpus IS, e.g. "a curriculum vitae". This is injected
                  into the extraction prompt — it is the only corpus-specific
                  thing the generic extractor knows.
        target_db the ArcadeDB database. One pipeline = one corpus = one graph
                  namespace, so two corpora never collide.
        strategy  pdf_docling (chunks + bounding boxes) | plain_text (no boxes)
        """
        self._d: dict[str, Any] = {
            "corpus": {"context": context, "language": language, "target_db": target_db},
            "chunking": {"strategy": strategy, "target_tokens": target_tokens},
            "types": {"entities": {}, "relations": {}},
            "index": {},
            "steps": [],
            "extraction_agent": extraction_agent,
        }

    # ---- vocabulary --------------------------------------------------
    def entity(self, name: str, definition: str, examples: list[str] | None = None,
               not_: str | None = None) -> "Pipeline":
        """Declare an entity type.

        `definition` and `examples` carry the whole burden of precision, and
        `not_` is the contrastive half — "NOT a named product (that's technology)".
        Measured: types that contrast cleanly separate in one prompt; types that
        overlap (term vs concept) produced 100% duplication.
        """
        e: dict[str, Any] = {"definition": definition, "examples": examples or []}
        if not_:
            e["not"] = not_
        self._d["types"]["entities"][name] = e
        return self

    def relation(self, name: str, from_: str, to: str, definition: str = "") -> "Pipeline":
        self._d["types"]["relations"][name] = {"from": from_, "to": to,
                                               "definition": definition}
        return self

    def index(self, field: str, type_: str = "string") -> "Pipeline":
        """Promote an extracted entity to a first-class, INDEXED chunk property —
        for filtering, not for the graph."""
        self._d["index"][field] = {"type": type_}
        return self

    # ---- steps -------------------------------------------------------
    def structural(self, entities: list[str] | None = None,
                   relations: list[str] | None = None) -> "Pipeline":
        """Free and deterministic: what the document's own structure states."""
        self._d["steps"].append({"tier": "structural", "entities": entities or [],
                                 "relations": relations or []})
        return self

    def llm(self, entities: list[str], relations: list[str] | None = None) -> "Pipeline":
        """The expensive tier: one LLM call per chunk."""
        self._d["steps"].append({"tier": "llm", "entities": entities,
                                 "relations": relations or []})
        return self

    def derived(self, threshold: float = 0.75) -> "Pipeline":
        """Embeddings + graph maths. No inference, no LLM."""
        self._d["steps"].append({"tier": "derived", "relations": ["SIMILAR_TO"],
                                 "threshold": threshold})
        return self

    def communities(self) -> "Pipeline":
        """PageRank + Louvain clustering, then an LLM summary per community."""
        self._d["steps"].append({"tier": "communities"})
        return self

    def custom(self, ref: str, **config: Any) -> "Pipeline":
        """A custom layer mounted on the Agent, referenced by name. List the
        available ones with `client.layers()`."""
        self._d["steps"].append({"layer": "custom", "ref": ref, "config": config})
        return self

    def build(self) -> dict:
        return self._d

    def __repr__(self) -> str:
        c = self._d["corpus"]
        return (f"<Pipeline {c['target_db']!r} ({c['context']!r}) "
                f"entities={list(self._d['types']['entities'])} "
                f"steps={[s.get('tier') or s.get('ref') for s in self._d['steps']]}>")
