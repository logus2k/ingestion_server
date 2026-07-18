"""Ingestion Agent SDK.

    from ingestion_client import IngestionClient, Pipeline

    c = IngestionClient("http://localhost:8700")
    p = (Pipeline("a curriculum vitae", target_db="cv")
         .entity("organization", "a named employer or certifying body")
         .llm(entities=["organization"])
         .build())
    run = c.ingest(p, ["/watched/in/cv.pdf"])
    print(run.state, run.committed)

Stdlib only. See ../../../documents/sdk.md for the full reference and
../../../examples/ for runnable scripts.
"""
from .client import IngestionClient, IngestionError, Run
from .pipeline import Pipeline

__all__ = ["IngestionClient", "IngestionError", "Run", "Pipeline"]
__version__ = "0.1.0"
