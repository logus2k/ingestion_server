"""Configuration — every external service, one place.

Nothing here points at noted. The Agent talks to exactly three services:
graph-server-arcadedb (the corpus), embeddings-server (vectors), agent_server
(the LLM).
"""
from __future__ import annotations

import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


class Settings:
    # ── the corpus ────────────────────────────────────────────────────
    arcadedb_url: str = os.environ.get("ARCADEDB_URL", "http://graph-server-arcadedb:2480").rstrip("/")
    arcadedb_user: str = os.environ.get("ARCADEDB_USER", "root")
    arcadedb_password: str = os.environ.get("ARCADEDB_PASSWORD", "poc-dev-pass")
    # ArcadeDB caps HTTP result sets around 20k rows; every read is paginated.
    arcade_page: int = _int("ARCADE_PAGE", 5000)
    # Per-row INSERT ... CONTENT :pN batches inside one sqlscript.
    write_batch: int = _int("WRITE_BATCH", 50)

    # ── vectors ───────────────────────────────────────────────────────
    embed_url: str = os.environ.get("EMBED_URL", "http://embeddings-server:8600").rstrip("/")
    embed_batch: int = _int("EMBED_BATCH", 32)
    embed_dims: int = _int("EMBED_DIMS", 1024)
    # bge-m3's lexical vocabulary — the LSM_SPARSE_VECTOR index dimension.
    sparse_dims: int = _int("SPARSE_DIMS", 250002)

    # ── the LLM ───────────────────────────────────────────────────────
    agent_server_url: str = os.environ.get("AGENT_SERVER_URL", "http://agent_server:7701").rstrip("/")
    # llama-vision runs `--parallel 2`. Measured: 2 concurrent gives ~1.02s/chunk
    # sustained; 4 shows a clean two-wave queue and no throughput gain. Raise
    # only if --parallel is raised with it.
    llm_parallelism: int = _int("LLM_PARALLELISM", 2)
    llm_timeout_s: float = float(os.environ.get("LLM_TIMEOUT_S", "300"))
    llm_retries: int = _int("LLM_RETRIES", 2)

    # ── the bus ───────────────────────────────────────────────────────
    bus_url: str = os.environ.get("BUS_URL", "http://agent-bus-app:6815").rstrip("/")
    bus_enabled: bool = _bool("BUS_ENABLED", True)
    bus_stream_id: str = os.environ.get("BUS_STREAM_ID", "ingestion")

    # ── runs ──────────────────────────────────────────────────────────
    # Run state is persisted so a suspended run survives a restart and outlives
    # any caller. SQLite: one file, no extra service, and runs are low-volume.
    state_db: str = os.environ.get("STATE_DB", "/data/runs.db")
    # Custom layers are mounted here and referenced by name (the Skills pattern:
    # the code lives in the service that owns it; callers hold a reference).
    layers_dir: str = os.environ.get("LAYERS_DIR", "/data/layers")

    # ── models (chunking) ─────────────────────────────────────────────
    docling_artifacts: str = os.environ.get("DOCLING_ARTIFACTS_ROOT", "/models/docling/models")
    # Tokenizer only — the chunker counts tokens with it. Actual embedding runs
    # in embeddings-server, so the 4.3GB of weights are not needed here.
    bge_m3_local: str = os.environ.get("BGE_M3_LOCAL", "/models/bge-m3")

    # ── api ───────────────────────────────────────────────────────────
    api_token: str = os.environ.get("API_TOKEN", "")   # empty = open (dev only)
    port: int = _int("PORT", 8700)


settings = Settings()
