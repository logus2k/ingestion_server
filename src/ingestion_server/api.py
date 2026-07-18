"""The API — and therefore the SDK.

Every endpoint is usable without Patron, without agent_runtime, from curl. That
is the point: the Agent is the product, the block is a client.

Pipelines arrive INLINE with every run; the service stores none. A caller is
self-contained, and there is no pipeline registry to keep in sync.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse

from . import bus
from .config import settings
from .engine import Engine, new_run
from .layers.registry import list_layers
from .models import (Decision, Pipeline, ResumeRequest, Run, RunRequest,
                     RunState)
from .store import RunStore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("ingestion")

app = FastAPI(
    title="Ingestion Agent",
    version="0.1.0",
    description=(
        "Turns documents into a searchable corpus and a knowledge graph.\n\n"
        "A **pipeline** is declarative and travels inline with each run: the corpus "
        "context, the entity/relation vocabulary, and an ordered list of layers "
        "(`structural` / `llm` / `derived` / `custom`). The engine is "
        "single-document; the API takes a list and walks it one at a time.\n\n"
        "Layers are pure — they write nothing. Results are staged and committed "
        "once per document, atomically, so a suspended run leaves no half-built "
        "graph and 'repeat this layer' is always safe."),
)

store = RunStore()
engine = Engine(store, emit=bus.publish)


def auth(authorization: Optional[str] = Header(default=None)) -> None:
    """Bearer token, if one is configured. This is a WRITE api to a graph —
    an empty token means open, which is a dev-only posture."""
    if not settings.api_token:
        return
    if authorization != f"Bearer {settings.api_token}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


@app.on_event("startup")
async def _startup() -> None:
    n = store.reap_orphans()
    if n:
        logger.warning("marked %d orphaned run(s) failed (service restarted mid-run)", n)
    logger.info("ingestion agent up | arcadedb=%s embed=%s llm=%s parallelism=%d",
                settings.arcadedb_url, settings.embed_url,
                settings.agent_server_url, settings.llm_parallelism)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await bus.aclose()


@app.get("/v1/healthz", tags=["ops"])
async def healthz() -> dict:
    return {"status": "ok", "service": "ingestion_server",
            "arcadedb": settings.arcadedb_url, "embed": settings.embed_url,
            "llm": settings.agent_server_url,
            "layers": [l.name for l in list_layers()]}


@app.get("/v1/layers", tags=["pipelines"])
async def layers() -> dict:
    """Built-in tiers plus any custom layer mounted in `layers_dir` — the
    picker's data source."""
    return {"layers": [l.to_json() for l in list_layers()]}


@app.post("/v1/pipelines/validate", tags=["pipelines"])
async def validate(pipeline: Pipeline) -> dict:
    """Cross-field rules the JSON schema cannot express: entity types referenced
    by a step must be declared, relation endpoints must be real types, indexed
    fields must be extractable."""
    errors = pipeline.validate_semantics()
    return {"ok": not errors, "errors": errors}


@app.post("/v1/runs", tags=["runs"], dependencies=[Depends(auth)])
async def create_run(req: RunRequest) -> Run:
    errors = req.pipeline.validate_semantics()
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    if not req.documents:
        raise HTTPException(status_code=422, detail="no documents")

    run = new_run(req.pipeline, req.documents)
    store.save(run)

    if req.wait:
        # The block's mode: block until terminal and return the finished Run.
        return await engine.execute(run, req.judge)
    asyncio.create_task(engine.execute(run, req.judge))
    return run


@app.get("/v1/runs/{run_id}", tags=["runs"])
async def get_run(run_id: str) -> Run:
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"no run {run_id}")
    return run


@app.get("/v1/runs", tags=["runs"])
async def list_runs(state: Optional[RunState] = None, limit: int = 50) -> dict:
    return {"runs": [r.model_dump() for r in store.list(state, limit)]}


@app.post("/v1/runs/{run_id}/resume", tags=["runs"], dependencies=[Depends(auth)])
async def resume_run(run_id: str, req: ResumeRequest) -> Run:
    """A suspended run waits for a human decision that may arrive minutes or
    hours later — from a frontend, another agent, or straight from the bus."""
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"no run {run_id}")
    if run.state != RunState.suspended:
        raise HTTPException(status_code=409,
                            detail=f"run is {run.state.value}, not suspended")
    return await engine.resume(run, req.decision)


@app.post("/v1/runs/{run_id}/cancel", tags=["runs"], dependencies=[Depends(auth)])
async def cancel_run(run_id: str) -> Run:
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"no run {run_id}")
    engine.cancel(run_id)
    if run.state == RunState.suspended:
        run.state = RunState.cancelled
        store.save(run)
    return run


@app.get("/v1/runs/{run_id}/events", tags=["runs"])
async def run_events(run_id: str) -> StreamingResponse:
    """Server-sent progress. Polls the store rather than holding engine state, so
    it works across processes and survives a restart."""
    async def gen():
        last = None
        for _ in range(3600):
            run = store.get(run_id)
            if not run:
                yield f"data: {json.dumps({'error': 'no such run'})}\n\n"
                return
            snap = run.model_dump_json()
            if snap != last:
                yield f"data: {snap}\n\n"
                last = snap
            if run.state in (RunState.completed, RunState.failed,
                             RunState.cancelled, RunState.suspended):
                return
            await asyncio.sleep(1)
    return StreamingResponse(gen(), media_type="text/event-stream")
