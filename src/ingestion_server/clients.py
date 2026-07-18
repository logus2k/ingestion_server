"""Clients for the three external services.

Ported from the CV prototype (`~/env/assets/cv/ingest`), which ran green
end-to-end against a real corpus. Every non-obvious constant here is a measured
fact, not a guess — see the comments.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger(__name__)


class IngestError(RuntimeError):
    """Anything that should fail a run rather than be silently swallowed."""


# ── ArcadeDB ──────────────────────────────────────────────────────────
class Arcade:
    """The corpus. One instance, many databases — a pipeline names its own."""

    def __init__(self, db: str, client: httpx.AsyncClient | None = None) -> None:
        self.db = db
        self._c = client or httpx.AsyncClient(
            timeout=600.0, auth=(settings.arcadedb_user, settings.arcadedb_password))

    async def _post(self, kind: str, command: str, params: dict | None = None,
                    language: str = "sql") -> Any:
        body: dict[str, Any] = {"language": language, "command": command}
        if params:
            body["params"] = params
        r = await self._c.post(f"{settings.arcadedb_url}/api/v1/{kind}/{self.db}", json=body)
        if r.status_code >= 400:
            raise IngestError(f"ArcadeDB {kind} HTTP {r.status_code}: {r.text[:400]}")
        return r.json().get("result")

    async def query(self, command: str, params: dict | None = None) -> list[dict]:
        return await self._post("query", command, params) or []

    async def command(self, command: str, params: dict | None = None) -> Any:
        return await self._post("command", command, params)

    async def cypher(self, command: str, params: dict | None = None) -> Any:
        """Graph algorithms (algo.pagerank / algo.louvain) are reachable ONLY
        through the Cypher parser — the SQL parser rejects `CALL` outright
        ("no viable alternative at input 'CALL'"). Verified on 26.7.2.

        Note Cypher map keys are bare identifiers: `{weightProperty: 'weight'}`
        parses, `{'weightProperty': 'weight'}` is a syntax error.
        """
        return await self._post("command", command, params, "cypher")

    async def script(self, statements: list[str], params: dict | None = None) -> Any:
        """Many statements, one round trip.

        The per-row `INSERT ... CONTENT :pN` shape is deliberate: `CONTENT :list`
        with a list parameter inserts ONE row, not one per element.
        """
        if not statements:
            return None
        return await self._post("command", "\n".join(statements), params, "sqlscript")

    async def query_all(self, select: str, params: dict | None = None) -> list[dict]:
        """Paginated read. ArcadeDB caps HTTP result sets around 20k rows — the
        job2cool migration lost edges to exactly this before it was noticed."""
        out: list[dict] = []
        skip = 0
        while True:
            page = await self.query(f"{select} SKIP {skip} LIMIT {settings.arcade_page}", params)
            out.extend(page)
            if len(page) < settings.arcade_page:
                return out
            skip += settings.arcade_page

    async def aclose(self) -> None:
        await self._c.aclose()


# ── embeddings-server ─────────────────────────────────────────────────
class Embedder:
    """bge-m3 dense (1024) + sparse (lexical). Sparse comes back as
    {indices, weights} — already the shape LSM_SPARSE_VECTOR wants."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._c = client or httpx.AsyncClient(timeout=300.0)

    async def embed(self, texts: list[str], dense: bool = True,
                    sparse: bool = False) -> tuple[list[list[float]], list[dict]]:
        if not texts:
            return [], []
        r = await self._c.post(f"{settings.embed_url}/embed",
                               json={"texts": texts, "dense": dense, "sparse": sparse})
        if r.status_code >= 400:
            raise IngestError(f"embeddings-server HTTP {r.status_code}: {r.text[:300]}")
        d = r.json()
        return (d.get("vectors") or []), (d.get("sparse") or [])

    async def embed_batched(self, texts: list[str], dense: bool = True,
                            sparse: bool = False,
                            on_progress=None) -> tuple[list[list[float]], list[dict]]:
        """Batched so one oversized text cannot fail the whole document.

        noted embedded a document's chunks in ONE atomic call, so a single bad
        chunk cost every vector for that document — and the failure was swallowed
        into `out['rag'] = {'error': ...}` where nobody saw it.
        """
        dv: list[list[float]] = []
        sv: list[dict] = []
        for i in range(0, len(texts), settings.embed_batch):
            d, s = await self.embed(texts[i:i + settings.embed_batch], dense=dense, sparse=sparse)
            dv.extend(d)
            sv.extend(s)
            if on_progress:
                on_progress(min(i + settings.embed_batch, len(texts)), len(texts))
        return dv, sv

    async def aclose(self) -> None:
        await self._c.aclose()


# ── agent_server (the LLM) ────────────────────────────────────────────
class LLM:
    """OpenAI-compatible chat against an agent_server preset.

    Concurrency is capped by a semaphore rather than left to the caller: the
    model server runs `--parallel 2`, so more concurrent requests just queue and
    add scheduling overhead. Measured: 2 → 1.92s wall for 2 calls (both slots
    busy); 4 → 3.45s in a clean two-wave pattern, i.e. no gain.
    """

    def __init__(self, agent: str, parallelism: int | None = None,
                 client: httpx.AsyncClient | None = None) -> None:
        self.agent = agent
        self._c = client or httpx.AsyncClient(timeout=settings.llm_timeout_s)
        self._sem = asyncio.Semaphore(parallelism or settings.llm_parallelism)

    @staticmethod
    def _json_payload(content: str) -> str:
        """Peel the model's wrapper off the JSON.

        The active model emits a `<think>…</think>` block before its answer —
        verified identical for the long-running `noted_graph` preset and a fresh
        one, so it is normal model behaviour, not a misconfiguration. It also
        sometimes fences the JSON. `response_format` constrains the answer, not
        the preamble.

        Presets should set `chat_template_kwargs: {enable_thinking: false}` —
        measured 3.8s → 1.2s per extraction call, with 65% of generated tokens
        being reasoning we discard. This peel is the belt to that braces.
        """
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1]
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        return content.strip()

    async def chat_json(self, user_prompt: str, *, temperature: float = 0.1,
                        max_tokens: int = 4096,
                        json_schema: dict | None = None) -> Any | None:
        """Chat completion parsed as JSON. Returns None rather than raising — one
        chunk failing to extract must not abort a document."""
        fmt: dict[str, Any] = {"type": "json_object"}
        if json_schema:
            fmt = {"type": "json_schema", "json_schema": json_schema}
        body = {
            "model": self.agent,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": fmt,
        }
        async with self._sem:
            for attempt in range(settings.llm_retries + 1):
                try:
                    r = await self._c.post(
                        f"{settings.agent_server_url}/v1/chat/completions", json=body)
                    if r.status_code >= 400:
                        raise IngestError(
                            f"agent_server HTTP {r.status_code}: {r.text[:200]}")
                    content = r.json()["choices"][0]["message"]["content"]
                    return json.loads(self._json_payload(content))
                except Exception as e:
                    if attempt >= settings.llm_retries:
                        logger.warning("LLM call failed after %d attempts: %s",
                                       attempt + 1, e)
                        return None
                    await asyncio.sleep(1.5 * (attempt + 1))
        return None

    async def aclose(self) -> None:
        await self._c.aclose()
