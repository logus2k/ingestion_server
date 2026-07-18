"""Bus events — fire-and-forget.

The API is always the truth: `GET /v1/runs/{id}` is authoritative and persisted.
Events are a notification channel on top, so a dead bus must never fail a run.
Every publish is wrapped and logged, never raised.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def _c() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=10.0)
    return _client


async def publish(event_type: str, data: dict[str, Any]) -> None:
    if not settings.bus_enabled:
        return
    try:
        await _c().post(f"{settings.bus_url}/publish", json={
            "stream_id": settings.bus_stream_id,
            "sender": "ingestion_server",
            "event_type": event_type,
            "data": data,
        })
    except Exception as e:
        logger.info("bus publish %s skipped: %s", event_type, type(e).__name__)


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
