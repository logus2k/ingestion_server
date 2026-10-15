"""GPU memory given back when idle.

PyTorch keeps the memory a conversion needed reserved after it ends, so the next one is fast: docling's layout and
table models and the OCR retry leave it at their peak. Measured 2026-10-15 on Cortex's machine: this service held
3,452 MiB of the card after a day of ingestion, nearly all of it free blocks kept for later. After `idle_seconds` with
no conversion, those free blocks are handed back to the card (`torch.cuda.empty_cache()`); the models stay loaded, so
the next document pays only a little allocation.

The release takes the conversion lock without waiting (pdf_docling._CONVERT_LOCK): never during a conversion.

    GET  /v1/admin/gpu-memory   {release_when_idle, idle_seconds, reserved_mib, allocated_mib, released_mib}
    PUT  /v1/admin/gpu-memory   {release_when_idle?, idle_seconds?}   (bearer token, as the other write routes)
Defaults from the environment: GPU_RELEASE_IDLE (true), GPU_RELEASE_IDLE_SECONDS (120). Cortex sets them from its
settings (Settings > Transcription, "GPU memory").
"""
from __future__ import annotations

import gc
import logging
import os
import threading
import time

logger = logging.getLogger("ingestion.gpu_memory")

_state = {
    "release_when_idle": os.environ.get("GPU_RELEASE_IDLE", "true").strip().lower() not in ("0", "false", "no", "off"),
    "idle_seconds": max(10, int(os.environ.get("GPU_RELEASE_IDLE_SECONDS", "120"))),
}
_last_use = [time.monotonic()]
_released = [True]          # nothing used since the last release (or since start): nothing to give back
_total_released_mib = [0.0]
_lock: list = [None]        # the conversion lock, set by start()


def _torch():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - no torch, no GPU: nothing to release
        return None


def used() -> None:
    """A conversion just ended: the clock starts again."""
    _last_use[0] = time.monotonic()
    _released[0] = False


def snapshot() -> dict:
    t = _torch()
    reserved = t.cuda.memory_reserved() / 2**20 if t else 0.0
    allocated = t.cuda.memory_allocated() / 2**20 if t else 0.0
    return {**_state, "reserved_mib": round(reserved), "allocated_mib": round(allocated),
            "released_mib": round(_total_released_mib[0]), "idle_for_seconds": round(time.monotonic() - _last_use[0])}


def configure(release_when_idle: bool | None = None, idle_seconds: int | None = None) -> dict:
    if release_when_idle is not None:
        _state["release_when_idle"] = bool(release_when_idle)
    if idle_seconds is not None:
        _state["idle_seconds"] = max(10, int(idle_seconds))
    logger.info("gpu memory: release when idle %s, after %d s", _state["release_when_idle"], _state["idle_seconds"])
    return snapshot()


def release() -> float | None:
    """Hand the free blocks back; None when a conversion holds the lock (tried again later). MiB given back."""
    t = _torch()
    if t is None:
        _released[0] = True
        return 0.0
    lock = _lock[0]
    if lock is not None and not lock.acquire(blocking=False):
        return None
    try:
        before = t.cuda.memory_reserved()
        gc.collect()
        t.cuda.empty_cache()
        freed = (before - t.cuda.memory_reserved()) / 2**20
    finally:
        if lock is not None:
            lock.release()
    _released[0] = True
    _total_released_mib[0] += max(0.0, freed)
    logger.info("gpu memory: idle %d s, %.0f MiB given back to the card (%.0f MiB still reserved)",
                time.monotonic() - _last_use[0], freed, t.cuda.memory_reserved() / 2**20)
    return freed


def _loop() -> None:
    while True:
        time.sleep(15)
        try:
            if (_state["release_when_idle"] and not _released[0]
                    and time.monotonic() - _last_use[0] >= _state["idle_seconds"]):
                release()
        except Exception:  # noqa: BLE001 - the watcher never dies
            logger.exception("gpu memory: release failed")


def start(conversion_lock) -> None:
    _lock[0] = conversion_lock
    threading.Thread(target=_loop, name="gpu-memory", daemon=True).start()
