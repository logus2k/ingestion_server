"""The wire contract: pipelines, runs, results.

The pipeline is declarative and arrives **inline** with every run — the service
stores none. That keeps the Agent stateless about pipelines and lets a caller
(Jenkins, a Patron block, a script) be entirely self-contained.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── vocabulary ────────────────────────────────────────────────────────
class EntityType(BaseModel):
    """One entity type the extractor may return.

    `definition` and `examples` carry the whole burden of precision. Measured on
    a CV: a `term`/`concept` split was undecidable and produced 100% overlap
    (both claimed 'telecom', 'SaaS', 'biometrics' at confidence 1.0), while
    organization/role/technology/domain separated cleanly with zero overlap in
    a single prompt. Definitions that contrast with the nearest neighbour are
    what makes the difference — hence `not_`.
    """
    definition: str
    examples: list[str] = Field(default_factory=list)
    # The contrastive half: "NOT a named product or company (that's technology)".
    not_: Optional[str] = Field(default=None, alias="not")

    model_config = {"populate_by_name": True}


class RelationType(BaseModel):
    definition: str = ""
    from_: str = Field(alias="from")
    to: str

    model_config = {"populate_by_name": True}


class Types(BaseModel):
    entities: dict[str, EntityType] = Field(default_factory=dict)
    relations: dict[str, RelationType] = Field(default_factory=dict)


# ── the pipeline ──────────────────────────────────────────────────────
class Corpus(BaseModel):
    """What the run is *about* — not steps."""
    context: str = Field(description="e.g. 'a curriculum vitae'. Injected into prompts.")
    language: str = "en"
    target_db: str


class Chunking(BaseModel):
    # `structural` emits one chunk per document item (un-merged, block type in
    # `kind`) for consumers that need document structure rather than retrieval
    # units — see chunking/structural.py.
    strategy: Literal["pdf_docling", "markdown_render", "plain_text", "structural"] = "pdf_docling"
    # Docling-style chunkers honour only max_tokens; this is fed to it directly.
    target_tokens: int = 200
    # Opt-in: each chunk's text starts with "<title> > <section path>" (a line) before it is embedded and
    # stored. Docling keeps headings out of the chunk text, so a passage whose words never name its subject
    # (a campaign's conditions under the heading "Campanha TOIC", the second half of a short list) was not
    # found by search: on 37 questions over 16 Word documents, 18 answers correct without it, 22 with the
    # section path (Cortex, 2026-09-30). `title`: the document's title (default: none, the path only).
    context: bool = False
    title: str | None = None


class IndexField(BaseModel):
    """An extracted entity promoted to a first-class, indexed chunk property.

    job2cool filters its candidate browser on english_level / experience_years —
    those must be indexed columns, not graph vertices.
    """
    type: Literal["string", "int", "float"] = "string"


class Tier(str, Enum):
    structural = "structural"     # free, deterministic: what the document states
    llm = "llm"                   # inference — the expensive one
    derived = "derived"           # embeddings + graph maths, no inference
    communities = "communities"   # PageRank + Louvain, then an LLM summary per cluster


class Step(BaseModel):
    """One layer of the pipeline. Either a built-in tier or a custom module."""
    tier: Optional[Tier] = None
    layer: Optional[Literal["custom"]] = None
    ref: Optional[str] = None            # custom only: a module in layers_dir
    entities: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    # derived only
    threshold: float = 0.75
    # custom only
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("layer")
    @classmethod
    def _custom_needs_ref(cls, v, info):
        return v

    def key(self) -> str:
        if self.layer == "custom":
            return f"custom:{self.ref}"
        return str(self.tier.value if self.tier else "?")


class Pipeline(BaseModel):
    corpus: Corpus
    chunking: Chunking = Field(default_factory=Chunking)
    types: Types = Field(default_factory=Types)
    index: dict[str, IndexField] = Field(default_factory=dict)
    steps: list[Step] = Field(default_factory=list)
    # The agent_server preset the `llm` tier extracts with. Generic by default:
    # the document type, the entity types and their definitions all travel in the
    # prompt, so one preset serves every corpus. Distinct from the judge's preset
    # — they are different roles.
    extraction_agent: str = "ingest_extractor"
    # When true, after extraction the engine collapses entities that share a
    # normalized NAME but were emitted under different TYPES into a single
    # canonical entity (highest-confidence type wins), re-pointing every edge.
    # OFF by default: for many corpora the same name legitimately spans types
    # (e.g. "Deep Learning" as both a technique and a field), so this is opt-in
    # per pipeline where dual-typing is judged noise rather than signal.
    merge_cross_type: bool = False

    def validate_semantics(self) -> list[str]:
        """Cross-field rules the schema can't express. Human-aimed strings;
        empty list == valid."""
        errors: list[str] = []
        known = set(self.types.entities)
        for i, s in enumerate(self.steps):
            where = f"steps[{i}] ({s.key()})"
            if s.layer == "custom":
                if not s.ref:
                    errors.append(f"{where}: custom layer needs a `ref`")
                continue
            if s.tier is None:
                errors.append(f"{where}: needs `tier` or `layer: custom`")
                continue
            for e in s.entities:
                if e not in known:
                    errors.append(f"{where}: entity type '{e}' is not declared in types.entities")
            for r in s.relations:
                if r not in self.types.relations and r != "SIMILAR_TO":
                    errors.append(f"{where}: relation '{r}' is not declared in types.relations")
        for name, rel in self.types.relations.items():
            for side, val in (("from", rel.from_), ("to", rel.to)):
                if val not in known:
                    errors.append(f"relations.{name}.{side}: '{val}' is not a declared entity type")
        for field in self.index:
            if field not in known:
                errors.append(f"index.{field}: not a declared entity type — nothing would populate it")
        if not self.steps:
            errors.append("pipeline has no steps: it would only write chunks")
        return errors


# ── documents + runs ──────────────────────────────────────────────────
class Change(str, Enum):
    created = "created"
    modified = "modified"
    deleted = "deleted"


class Document(BaseModel):
    """What to reconcile with the corpus.

    `change` mirrors folder_watch's vocabulary so a File Initiator event maps
    straight through. On a delete there is no content to read, so `path` is all
    there is — which is why the caller must pass `change` rather than let us
    guess from the payload.
    """
    path: str
    name: Optional[str] = None       # display name; defaults to basename
    change: Change = Change.created

    def display_name(self) -> str:
        return self.name or self.path.rsplit("/", 1)[-1]


class RunState(str, Enum):
    pending = "pending"
    running = "running"
    suspended = "suspended"     # judge flagged, or a document failed — resumable
    completed = "completed"
    failed = "failed"           # a crash, not a judgement
    cancelled = "cancelled"


class Decision(str, Enum):
    retry = "retry"
    skip = "skip"
    abort = "abort"


class Verdict(BaseModel):
    ok: bool = True
    note: str = ""
    suspicion: str = ""


class LayerReport(BaseModel):
    name: str
    state: Literal["pending", "running", "completed", "skipped", "failed"] = "pending"
    seconds: float = 0.0
    # Counts + distributions + a sample. The judge needs both: a degenerate type
    # distribution (200 concept vs 25 organization) is invisible in a dump and
    # obvious in a histogram.
    digest: dict[str, Any] = Field(default_factory=dict)
    judge: Optional[Verdict] = None
    error: Optional[str] = None


class DocumentReport(BaseModel):
    document: str
    change: Change
    state: Literal["pending", "running", "completed", "skipped", "failed"] = "pending"
    layers: list[LayerReport] = Field(default_factory=list)
    committed: dict[str, int] = Field(default_factory=dict)
    deleted: dict[str, int] = Field(default_factory=dict)
    error: Optional[str] = None


class Run(BaseModel):
    run_id: str
    state: RunState = RunState.pending
    pipeline: Pipeline
    documents: list[Document]
    # Index of the document the run halted on; resume continues from here.
    cursor: int = 0
    reports: list[DocumentReport] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


class RunRequest(BaseModel):
    pipeline: Pipeline
    documents: list[Document]
    judge: Optional["JudgeConfig"] = None
    # True: block until terminal and return the finished Run (the block's mode).
    # False: return immediately; poll GET or stream /events.
    wait: bool = True


class JudgeConfig(BaseModel):
    """Block-level: the judge watches the pipeline, so it sits outside it."""
    persona: str = "ingest_judge"
    template: Optional[str] = None
    on_suspicion: Literal["notify", "suspend"] = "notify"
    enabled: bool = True


class ResumeRequest(BaseModel):
    decision: Decision
    note: str = ""


RunRequest.model_rebuild()


# ── ids ───────────────────────────────────────────────────────────────
def chunk_hex(doc_name: str, chunk_index: int) -> str:
    """The one hex per chunk.

    Consumers derive a citation tag as sha1(chunk_id)[:12] for corpus chunks and
    read the suffix directly from `markdown_chunk:<hex>` ids. Keying both off the
    same canonical string is what makes the two resolve to each other — noted ran
    two chunkings with two id spaces that never met (verified live: 69 vs 69,
    zero overlap), so graph-sourced citations silently lost their PDF regions.
    """
    return hashlib.sha1(f"{doc_name}#{chunk_index}".encode("utf-8")).hexdigest()[:12]


def canonical_chunk_id(doc_name: str, chunk_index: int) -> str:
    return f"{doc_name}#{chunk_index}"
