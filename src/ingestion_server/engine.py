"""The run engine: documents in, corpus out.

Shape of a run:
  * The engine is single-document; the API accepts a list and the run walks it
    one at a time.
  * Layers are PURE — they return results, they never write. The engine stages
    everything and commits ONCE per document, atomically. That is what makes a
    judge's "repeat this layer" safe, lets a suspended run sit for hours without
    leaving a half-built graph, and keeps a document's additions and removals in
    one transaction.
  * A run halts on the first failure or judge flag with state `suspended` — not
    terminal, resumable with retry/skip/abort. A crash is `failed` instead.
  * Re-ingest REPLACES. Deletion removes and reports; the judge never gates it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Callable, Optional

from .chunking import get_chunker
from .clients import LLM, Arcade, Embedder, IngestError
from .config import settings
from .judge import Judge
from .layers.base import Chunk, Context, Entity, LayerResult, Relation
from .layers.registry import resolve
from .models import (Change, Decision, Document, DocumentReport, JudgeConfig, Tier,
                     LayerReport, Pipeline, Run, RunState, Verdict,
                     canonical_chunk_id, chunk_hex)
from .store import RunStore, now_iso

logger = logging.getLogger(__name__)

# Structural entity types the engine itself creates. A pipeline never declares
# them, and the judge is told they are normal.
STRUCTURAL = ("markdown_doc", "markdown_chunk", "community", "community_summary")


class Cancelled(Exception):
    """Cooperative cancel — raised between layers, never mid-write."""


class Engine:
    def __init__(self, store: RunStore, emit: Optional[Callable] = None) -> None:
        self.store = store
        self._emit = emit or (lambda *_a, **_k: None)
        self._cancels: set[str] = set()

    # ── public ────────────────────────────────────────────────────────
    def cancel(self, run_id: str) -> None:
        self._cancels.add(run_id)

    async def execute(self, run: Run, judge_cfg: JudgeConfig | None = None) -> Run:
        """Walk the run's documents from its cursor. Returns the run in a
        terminal-or-suspended state; always persisted before returning."""
        run.state = RunState.running
        self.store.save(run)
        await self._event(run, "ingestion.progress", {"state": "running"})

        db = Arcade(run.pipeline.corpus.target_db)
        emb = Embedder()
        # The extractor and the judge are DIFFERENT roles and therefore different
        # presets. An earlier version used the judge's persona for extraction and
        # fell back to a preset that didn't exist — every chunk 404'd, `chat_json`
        # returned None (by design, so one chunk can't abort a document), the run
        # committed zero entities, and reported `completed`. The judge would have
        # caught it, except it was using the same broken preset.
        llm = LLM(run.pipeline.extraction_agent)
        judge_llm = LLM((judge_cfg or JudgeConfig()).persona)
        judge = Judge(judge_cfg or JudgeConfig(enabled=False), judge_llm)

        try:
            while run.cursor < len(run.documents):
                doc = run.documents[run.cursor]
                report = self._report_for(run, run.cursor, doc)
                try:
                    await self._one_document(run, doc, report, db, emb, llm, judge)
                except Cancelled:
                    run.state = RunState.cancelled
                    self.store.save(run)
                    await self._event(run, "ingestion.failed", {"reason": "cancelled"})
                    return run
                except _Suspend as s:
                    report.state = "failed" if s.fatal else "pending"
                    report.error = s.reason
                    run.state = RunState.suspended
                    self.store.save(run)
                    await self._event(run, "ingestion.suspended", {
                        "document": doc.path, "cursor": run.cursor, "reason": s.reason})
                    return run
                except Exception as e:                       # a real crash
                    logger.exception("run %s crashed on %s", run.run_id, doc.path)
                    report.state = "failed"
                    report.error = f"{type(e).__name__}: {e}"
                    run.state = RunState.failed
                    run.error = report.error
                    self.store.save(run)
                    await self._event(run, "ingestion.failed", {"error": run.error})
                    return run

                report.state = "completed"
                run.cursor += 1
                self.store.save(run)

            run.state = RunState.completed
            self.store.save(run)
            await self._event(run, "ingestion.completed", {
                "documents": len(run.documents),
                "committed": _totals(run, "committed"),
                "deleted": _totals(run, "deleted")})
            return run
        finally:
            self._cancels.discard(run.run_id)
            await db.aclose(); await emb.aclose()
            await llm.aclose(); await judge_llm.aclose()

    async def resume(self, run: Run, decision: Decision,
                     judge_cfg: JudgeConfig | None = None) -> Run:
        if run.state != RunState.suspended:
            raise IngestError(f"run {run.run_id} is {run.state.value}, not suspended")
        if decision == Decision.abort:
            # Staged work is discarded: nothing was committed for the halted
            # document, and earlier documents committed independently.
            run.state = RunState.cancelled
            run.error = "aborted by decision"
            self.store.save(run)
            await self._event(run, "ingestion.failed", {"reason": "aborted"})
            return run
        if decision == Decision.skip:
            r = self._report_for(run, run.cursor, run.documents[run.cursor])
            r.state = "skipped"
            run.cursor += 1
        # retry: leave the cursor where it is and run the document again.
        return await self.execute(run, judge_cfg)

    # ── one document ──────────────────────────────────────────────────
    async def _one_document(self, run: Run, doc: Document, report: DocumentReport,
                            db: Arcade, emb: Embedder, llm: LLM, judge: Judge) -> None:
        name = doc.display_name()
        report.state = "running"
        await self._event(run, "ingestion.progress",
                          {"document": name, "change": doc.change.value})

        await ensure_schema(db)

        # A delete is a removal, not a pipeline run.
        if doc.change == Change.deleted:
            report.deleted = await self._remove(db, name)
            await self._event(run, "ingestion.progress",
                              {"document": name, "deleted": report.deleted})
            return

        # ── parse ─────────────────────────────────────────────────────
        t0 = time.time()
        chunks = await asyncio.to_thread(
            get_chunker(run.pipeline.chunking.strategy), doc.path,
            run.pipeline.chunking.target_tokens)
        if not chunks:
            raise _Suspend(f"parse produced 0 chunks from {name}", fatal=True)
        report.layers.append(LayerReport(
            name="parse", state="completed", seconds=round(time.time() - t0, 1),
            digest={"chunks": len(chunks),
                    "with_regions": sum(1 for c in chunks if c.regions),
                    "sections": len({c.section_path for c in chunks})}))
        self.store.save(run)

        # ── embed ─────────────────────────────────────────────────────
        t0 = time.time()
        dense, sparse = await emb.embed_batched([c.text for c in chunks],
                                                dense=True, sparse=True)
        if len(dense) != len(chunks) or len(sparse) != len(chunks):
            raise _Suspend(f"embed returned {len(dense)} dense / {len(sparse)} sparse "
                           f"for {len(chunks)} chunks", fatal=True)
        report.layers.append(LayerReport(
            name="embed", state="completed", seconds=round(time.time() - t0, 1),
            digest={"vectors": len(dense), "sparse": len(sparse)}))
        self.store.save(run)

        # ── layers (staged: nothing is written yet) ───────────────────
        entities: dict[str, Entity] = {}
        relations: list[Relation] = []
        chunk_props: dict[int, dict[str, Any]] = {}

        for step in run.pipeline.steps:
            self._check_cancel(run)
            layer = resolve(step)
            ctx = Context(pipeline=run.pipeline, document=doc, step=step, chunks=chunks,
                          entities=entities, relations=relations,
                          services={"llm": llm, "embedder": emb, "db": db,
                                    "chunk_hex": lambda i, n=name: chunk_hex(n, i)})
            t0 = time.time()
            lr = LayerResult()
            lrep = LayerReport(name=layer.name, state="running")
            report.layers.append(lrep)
            try:
                lr = await layer.run(ctx)
                lrep.state = "completed"
            except Exception as e:
                lrep.state = "failed"
                lrep.error = f"{type(e).__name__}: {e}"
                lrep.seconds = round(time.time() - t0, 1)
                self.store.save(run)
                raise _Suspend(f"layer {layer.name} failed: {e}")
            lrep.seconds = round(time.time() - t0, 1)
            lrep.digest = lr.digest

            # An `llm` step that asked for entities and produced NONE is a failure,
            # not an empty document. It means the model, the preset or the prompt is
            # broken — and it must not commit an empty corpus and report success.
            # This is exactly how a dead extractor preset went unnoticed: 404 on
            # every chunk, `chat_json` returning None per design, run `completed`
            # with zero entities.
            if (step.tier == Tier.llm and step.entities and not lr.entities
                    and not lr.digest.get("skipped")):
                raise _Suspend(
                    f"layer {layer.name} asked for {', '.join(step.entities)} and "
                    f"extracted nothing from {len(chunks)} chunks — the extractor "
                    f"({run.pipeline.extraction_agent}) is not working", fatal=True)

            for e in lr.entities:
                prev = entities.get(e.id)
                if prev is None or e.confidence > prev.confidence:
                    entities[e.id] = e
            relations.extend(lr.relations)
            for idx, props in lr.chunk_props.items():
                chunk_props.setdefault(idx, {}).update(props)
            self.store.save(run)

            verdict = await judge.assess(run.pipeline, step, layer.name,
                                         lr.digest, lrep.seconds)
            lrep.judge = verdict
            self.store.save(run)
            if verdict and not verdict.ok:
                await self._event(run, "ingestion.progress", {
                    "document": name, "layer": layer.name, "judge": "flagged",
                    "suspicion": verdict.suspicion, "note": verdict.note})
                if judge.should_suspend(verdict):
                    raise _Suspend(f"judge flagged {layer.name}: "
                                   f"{verdict.suspicion or verdict.note}")

        # ── commit: one atomic write, additions and removals together ─
        self._check_cancel(run)
        t0 = time.time()
        report.committed = await self._commit(db, run.pipeline, name, doc.path,
                                              chunks, dense, sparse, entities,
                                              relations, chunk_props)
        report.layers.append(LayerReport(
            name="commit", state="completed", seconds=round(time.time() - t0, 1),
            digest=report.committed))

    # ── persistence ───────────────────────────────────────────────────
    async def _commit(self, db: Arcade, pipeline: Pipeline, name: str, path: str,
                      chunks: list[Chunk], dense, sparse,
                      entities: dict[str, Entity], relations: list[Relation],
                      chunk_props: dict[int, dict]) -> dict[str, int]:
        """Replace this document wholesale. Runs AFTER every expensive, failure-prone
        stage has succeeded, so a bad parse or a dead embeddings-server can never
        leave the corpus empty."""
        removed = await self._remove(db, name)

        rows = []
        for i, c in enumerate(chunks):
            hexid = chunk_hex(name, c.index)
            row = {
                "chunk_id": canonical_chunk_id(name, c.index),
                "text": c.text, "source_path": name, "section_path": c.section_path,
                "page_no": int(c.page_no) if c.page_no is not None else None,
                "regions_json": json.dumps(c.regions or [], ensure_ascii=False),
                "cite_hex": hexid,
                "embedding": [float(x) for x in dense[i]],
                "sidx": [int(x) for x in (sparse[i].get("indices") or [])],
                "swt": [float(x) for x in (sparse[i].get("weights") or [])],
            }
            row.update(chunk_props.get(c.index, {}))
            rows.append(row)
        await _insert(db, "Chunk", rows)

        doc_id = f"markdown_doc:{name}"
        ents = [{"id": doc_id, "label": name, "type": "markdown_doc",
                 "project_ids": [pipeline.corpus.target_db],
                 "properties_json": json.dumps({"path": name, "chunk_count": len(chunks)},
                                               ensure_ascii=False)}]
        edges = []
        for c in chunks:
            cid = f"markdown_chunk:{chunk_hex(name, c.index)}"
            ents.append({
                "id": cid, "label": f"{name}#{c.index}", "type": "markdown_chunk",
                "project_ids": [pipeline.corpus.target_db],
                "properties_json": json.dumps({
                    "doc_path": name, "chunk_index": c.index,
                    "section_path": c.section_path, "text": c.text,
                    "page_no": c.page_no, "regions": c.regions,
                    # The hex tying this entity to its Chunk row — one id space,
                    # so a graph-sourced citation resolves with its PDF regions.
                    "cite_hex": chunk_hex(name, c.index)}, ensure_ascii=False)})
            edges.append({"src": doc_id, "dst": cid, "type": "chunked_into"})

        for e in entities.values():
            row = {"id": e.id, "label": e.name, "type": e.type,
                   "project_ids": [pipeline.corpus.target_db],
                   "properties_json": json.dumps(
                       {"description": e.description, "confidence": e.confidence},
                       ensure_ascii=False)}
            if e.embedding:
                row["embedding"] = [float(x) for x in e.embedding]
            ents.append(row)
        await _upsert_entities(db, ents)

        edges.extend({"src": r.src, "dst": r.dst, "type": r.type,
                      "provenance": r.provenance, **r.properties} for r in relations)
        await _edges(db, edges)

        return {"chunks": len(rows), "entities": len(ents), "relations": len(edges),
                **({"replaced": removed["chunks"]} if removed.get("chunks") else {})}

    async def _remove(self, db: Arcade, name: str) -> dict[str, int]:
        """Remove a document: its chunks, its chunk entities, and any thematic
        entity left with ZERO remaining mentions.

        The orphan check is the whole subtlety: `technology:postgresql` may be
        mentioned by documents that still exist. Removing entities wholesale
        would corrupt the graph for every other document.
        """
        before = await db.query("SELECT count(*) AS n FROM Chunk WHERE source_path = :p",
                                {"p": name})
        n_chunks = (before[0]["n"] if before else 0) or 0
        await db.command("DELETE FROM Chunk WHERE source_path = :p", {"p": name})
        # The parameter must NOT be called `:like` — LIKE is a reserved keyword and
        # the parser dies with a misleading "mismatched input 'WHERE'" pointing at
        # the start of the clause. Verified: `:like` fails, `:pat` parses.
        await db.command(
            "DELETE FROM Entity WHERE type IN ['markdown_doc','markdown_chunk'] "
            "AND properties_json LIKE :pat", {"pat": f'%"{name}"%'})
        # Orphans: thematic entities with no remaining `mentions` edge.
        #
        # Two traps here, both found the hard way:
        #
        # 1. `NOT IN` is BROKEN in ArcadeDB 26.7.2 — it returns the same rows as
        #    `IN`, silently, with a plausible count (measured: IN -> 100, NOT IN ->
        #    100, when the correct answer was 180). Use `NOT (x IN [...])`.
        # 2. Count `mentions` specifically, not all RELATES. Thematic entities hold
        #    SIMILAR_TO edges to EACH OTHER, so after this document's chunks are
        #    deleted they still have incoming RELATES and never look orphaned —
        #    they then collide on re-insert with a DuplicatedKeyException.
        # 3. Use `inE()`, NOT `in()`. `in()` returns the connected VERTICES, so
        #    `in('RELATES')[type='mentions']` filters a vertex's type — and a
        #    vertex is a `markdown_chunk`, never a `mentions`. It matches nothing
        #    and reports EVERY entity as an orphan, which silently deletes another
        #    document's entities. `inE()` returns the edges, whose type is right.
        #    Measured on one entity: in()->0, inE()->40.
        #
        # A `technology:postgresql` mentioned by other documents keeps its mentions
        # and correctly survives; only entities this document alone introduced go.
        st = ", ".join(f"'{t}'" for t in STRUCTURAL)
        orphans = await db.query(
            f"SELECT id FROM Entity WHERE NOT (type IN [{st}]) "
            f"AND inE('RELATES')[type = 'mentions'].size() = 0")
        n_ent = 0
        if orphans:
            ids = [o["id"] for o in orphans]
            for i in range(0, len(ids), 200):
                await db.command("DELETE FROM Entity WHERE id IN :ids",
                                 {"ids": ids[i:i + 200]})
            n_ent = len(ids)
        return {"chunks": n_chunks, "entities": n_ent}

    # ── plumbing ──────────────────────────────────────────────────────
    def _report_for(self, run: Run, i: int, doc: Document) -> DocumentReport:
        while len(run.reports) <= i:
            d = run.documents[len(run.reports)]
            run.reports.append(DocumentReport(document=d.path, change=d.change))
        r = run.reports[i]
        r.layers = []          # a retry re-reports from scratch
        r.error = None
        return r

    def _check_cancel(self, run: Run) -> None:
        if run.run_id in self._cancels:
            raise Cancelled()

    async def _event(self, run: Run, kind: str, data: dict) -> None:
        try:
            await self._emit(kind, {"run_id": run.run_id, **data})
        except Exception:
            # A dead bus must never fail a run — the API is always the truth.
            logger.warning("event %s not published", kind, exc_info=True)


class _Suspend(Exception):
    def __init__(self, reason: str, fatal: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.fatal = fatal


def _totals(run: Run, field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in run.reports:
        for k, v in (getattr(r, field) or {}).items():
            out[k] = out.get(k, 0) + v
    return out


async def _insert(db: Arcade, vtype: str, rows: list[dict]) -> None:
    """Per-row `INSERT ... CONTENT :pN` in one sqlscript. `CONTENT :list` with a
    list parameter inserts ONE row, not one per element.

    Only for rows this document exclusively owns (Chunk) — its old ones are
    deleted first, so an insert can't collide. Entities are shared; see _upsert.
    """
    for i in range(0, len(rows), settings.write_batch):
        grp = rows[i:i + settings.write_batch]
        stmts, params = [], {}
        for k, d in enumerate(grp):
            stmts.append(f"INSERT INTO {vtype} CONTENT :p{k};")
            params[f"p{k}"] = d
        await db.script(stmts, params)


async def _upsert_entities(db: Arcade, rows: list[dict]) -> None:
    """Entities are SHARED ACROSS DOCUMENTS — that is the point of a graph.

    `substance:anticoagulantes` appears in two different bulas; it must be ONE
    vertex that both documents mention, not two. Inserting blindly raises
    DuplicatedKeyException on the second document (measured), which would make a
    multi-document corpus impossible.

    `UPDATE ... UPSERT WHERE id = :id` creates or updates. Verified: the same id
    twice leaves exactly one row.
    """
    props = ("id", "label", "type", "properties_json", "project_ids", "embedding")
    for i in range(0, len(rows), settings.write_batch):
        grp = rows[i:i + settings.write_batch]
        stmts, params = [], {}
        for k, d in enumerate(grp):
            sets = []
            for p in props:
                if d.get(p) is not None:
                    sets.append(f"{p} = :{p}{k}")
                    params[f"{p}{k}"] = d[p]
            stmts.append(f"UPDATE Entity SET {', '.join(sets)} UPSERT WHERE id = :id{k};")
        await db.script(stmts, params)


async def _edges(db: Arcade, edges: list[dict]) -> None:
    for i in range(0, len(edges), settings.write_batch):
        grp = edges[i:i + settings.write_batch]
        stmts, params = [], {}
        for k, e in enumerate(grp):
            sets = [f"type = :y{k}"]
            params[f"s{k}"], params[f"d{k}"], params[f"y{k}"] = e["src"], e["dst"], e["type"]
            for prop in ("provenance", "similarity", "similarity_band", "weight"):
                if e.get(prop) is not None:
                    sets.append(f"{prop} = :{prop}{k}")
                    params[f"{prop}{k}"] = e[prop]
            stmts.append(
                f"CREATE EDGE RELATES FROM (SELECT FROM Entity WHERE id = :s{k}) "
                f"TO (SELECT FROM Entity WHERE id = :d{k}) SET {', '.join(sets)};")
        await db.script(stmts, params)


async def ensure_schema(db: Arcade) -> None:
    """Idempotent. Index order matters: LSM_SPARSE_VECTOR does NOT backfill
    pre-existing rows (unlike LSM_VECTOR), so it must exist before any insert."""
    await db.command("CREATE VERTEX TYPE Chunk IF NOT EXISTS")
    for p, t in [("chunk_id", "STRING"), ("text", "STRING"), ("source_path", "STRING"),
                 ("section_path", "STRING"), ("page_no", "INTEGER"),
                 ("regions_json", "STRING"), ("cite_hex", "STRING"),
                 ("embedding", "ARRAY_OF_FLOATS"), ("sidx", "ARRAY_OF_INTEGERS"),
                 ("swt", "ARRAY_OF_FLOATS")]:
        await db.command(f"CREATE PROPERTY Chunk.{p} IF NOT EXISTS {t}")
    await db.command("CREATE INDEX IF NOT EXISTS ON Chunk (chunk_id) NOTUNIQUE")
    await db.command("CREATE INDEX IF NOT EXISTS ON Chunk (cite_hex) NOTUNIQUE")
    await db.command("CREATE INDEX IF NOT EXISTS ON Chunk (source_path) NOTUNIQUE")

    await db.command("CREATE VERTEX TYPE Entity IF NOT EXISTS")
    for p, t in [("id", "STRING"), ("label", "STRING"), ("type", "STRING"),
                 ("properties_json", "STRING"), ("project_ids", "LIST"),
                 ("embedding", "ARRAY_OF_FLOATS")]:
        await db.command(f"CREATE PROPERTY Entity.{p} IF NOT EXISTS {t}")
    await db.command("CREATE INDEX IF NOT EXISTS ON Entity (id) UNIQUE")
    await db.command("CREATE INDEX IF NOT EXISTS ON Entity (type) NOTUNIQUE")

    await db.command("CREATE EDGE TYPE RELATES IF NOT EXISTS")
    for p, t in [("type", "STRING"), ("weight", "FLOAT"), ("similarity", "FLOAT"),
                 ("similarity_band", "STRING"), ("provenance", "STRING")]:
        await db.command(f"CREATE PROPERTY RELATES.{p} IF NOT EXISTS {t}")
    await db.command("CREATE INDEX IF NOT EXISTS ON RELATES (type) NOTUNIQUE")

    for ddl in (f'CREATE INDEX ON Chunk (sidx, swt) LSM_SPARSE_VECTOR '
                f'METADATA {{"dimensions":{settings.sparse_dims},"modifier":"IDF"}}',
                f'CREATE INDEX ON Chunk (embedding) LSM_VECTOR '
                f'METADATA {{"dimensions":{settings.embed_dims},"similarity":"COSINE"}}',
                f'CREATE INDEX ON Entity (embedding) LSM_VECTOR '
                f'METADATA {{"dimensions":{settings.embed_dims},"similarity":"COSINE"}}'):
        try:
            await db.command(ddl)
        except Exception as e:
            logger.debug("index exists: %s", str(e)[:70])


def new_run(pipeline: Pipeline, documents: list[Document]) -> Run:
    return Run(run_id=uuid.uuid4().hex[:12], pipeline=pipeline, documents=documents,
               created_at=now_iso(), updated_at=now_iso())
