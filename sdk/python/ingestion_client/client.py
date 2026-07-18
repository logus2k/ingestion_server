"""Ingestion Agent — Python client.

The supported way to drive the Agent. Laid out like the existing
`agent_bus/sdk/python/agent_bus_client` so the two feel the same.

    from ingestion_client import IngestionClient, Pipeline

    c = IngestionClient("http://localhost:8700")
    run = c.ingest(pipeline, ["/watched/in/report.pdf"])
    print(run.state, run.committed)

Everything here is stdlib — no dependency on the service, and nothing to install
beyond this package.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


class IngestionError(RuntimeError):
    """The Agent rejected the request or a run failed."""

    def __init__(self, message: str, status: int = 0, detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


@dataclass
class Run:
    """A run, as the Agent reports it. `raw` keeps the full payload so a client
    is never blocked by a field this dataclass hasn't caught up with."""
    run_id: str
    state: str
    cursor: int = 0
    documents: list[dict] = field(default_factory=list)
    reports: list[dict] = field(default_factory=list)
    error: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: dict) -> "Run":
        return cls(run_id=d.get("run_id", ""), state=d.get("state", ""),
                   cursor=d.get("cursor", 0), documents=d.get("documents", []),
                   reports=d.get("reports", []), error=d.get("error"), raw=d)

    # ---- convenience over `reports` ----------------------------------
    @property
    def ok(self) -> bool:
        return self.state == "completed"

    @property
    def suspended(self) -> bool:
        return self.state == "suspended"

    @property
    def committed(self) -> dict[str, int]:
        """Totals across every document in the run."""
        out: dict[str, int] = {}
        for r in self.reports:
            for k, v in (r.get("committed") or {}).items():
                out[k] = out.get(k, 0) + v
        return out

    @property
    def deleted(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.reports:
            for k, v in (r.get("deleted") or {}).items():
                out[k] = out.get(k, 0) + v
        return out

    @property
    def flags(self) -> list[dict]:
        """Every layer the judge was unhappy with, across all documents."""
        out = []
        for r in self.reports:
            for L in r.get("layers", []):
                j = L.get("judge") or {}
                if j and not j.get("ok", True):
                    out.append({"document": r.get("document"), "layer": L.get("name"),
                                "suspicion": j.get("suspicion"), "note": j.get("note")})
        return out

    @property
    def blocked_on(self) -> Optional[str]:
        """The document a suspended run halted on."""
        if not self.suspended or self.cursor >= len(self.documents):
            return None
        return self.documents[self.cursor].get("path")


class IngestionClient:
    def __init__(self, base_url: str = "http://localhost:8700",
                 token: str | None = None, timeout: float = 1800) -> None:
        self.base = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    # ---- plumbing ----------------------------------------------------
    def _call(self, method: str, path: str, body: Any = None,
              timeout: float | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(f"{self.base}{path}", data=data, method=method,
                                     headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                detail = json.loads(raw).get("detail", raw)
            except Exception:
                detail = raw
            raise IngestionError(f"{method} {path} -> HTTP {e.code}: "
                                 f"{str(detail)[:300]}", e.code, detail) from None
        except urllib.error.URLError as e:
            raise IngestionError(f"cannot reach the Agent at {self.base}: {e.reason}") from None

    # ---- ops ---------------------------------------------------------
    def health(self) -> dict:
        return self._call("GET", "/v1/healthz", timeout=15)

    def layers(self) -> list[dict]:
        """Built-in tiers plus any custom layer mounted on the Agent."""
        return self._call("GET", "/v1/layers", timeout=15)["layers"]

    # ---- pipelines ---------------------------------------------------
    def validate(self, pipeline: dict) -> tuple[bool, list[str]]:
        """Check a pipeline WITHOUT running it. Returns (ok, errors)."""
        r = self._call("POST", "/v1/pipelines/validate", pipeline, timeout=30)
        return bool(r.get("ok")), list(r.get("errors") or [])

    # ---- runs --------------------------------------------------------
    def ingest(self, pipeline: dict, documents: list[Any], *,
               judge: dict | None = None, wait: bool = True,
               timeout: float | None = None) -> Run:
        """Ingest documents. `documents` may be paths or full document dicts.

        wait=True blocks until the run reaches a terminal-or-suspended state and
        returns the finished Run. wait=False returns immediately with a run_id —
        poll `get()` or stream `events()`.
        """
        docs = [{"path": d} if isinstance(d, str) else d for d in documents]
        body: dict[str, Any] = {"pipeline": pipeline, "documents": docs, "wait": wait}
        if judge is not None:
            body["judge"] = judge
        return Run.from_json(self._call("POST", "/v1/runs", body, timeout=timeout))

    def remove(self, pipeline: dict, documents: list[str], **kw) -> Run:
        """Remove documents from the corpus. The Agent drops their chunks and any
        entity left with no remaining mentions — an entity other documents still
        mention correctly survives."""
        docs = [{"path": p, "change": "deleted"} for p in documents]
        return self.ingest(pipeline, docs, **kw)

    def get(self, run_id: str) -> Run:
        return Run.from_json(self._call("GET", f"/v1/runs/{run_id}", timeout=30))

    def list(self, state: str | None = None, limit: int = 50) -> list[Run]:
        q = f"?limit={limit}" + (f"&state={state}" if state else "")
        return [Run.from_json(r) for r in
                self._call("GET", f"/v1/runs{q}", timeout=30)["runs"]]

    def resume(self, run_id: str, decision: str, note: str = "") -> Run:
        """decision: 'retry' | 'skip' | 'abort'.

        retry re-runs the document that halted the run; skip moves past it; abort
        ends the run and discards its staged work.
        """
        if decision not in ("retry", "skip", "abort"):
            raise ValueError(f"decision must be retry|skip|abort, got {decision!r}")
        return Run.from_json(self._call(
            "POST", f"/v1/runs/{run_id}/resume", {"decision": decision, "note": note}))

    def cancel(self, run_id: str) -> Run:
        return Run.from_json(self._call("POST", f"/v1/runs/{run_id}/cancel", {},
                                        timeout=30))

    def events(self, run_id: str) -> Iterator[dict]:
        """Stream a run's state as it changes (SSE). Ends when the run reaches a
        terminal-or-suspended state."""
        req = urllib.request.Request(f"{self.base}/v1/runs/{run_id}/events")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            for line in r:
                s = line.decode(errors="replace").strip()
                if s.startswith("data: "):
                    try:
                        yield json.loads(s[6:])
                    except Exception:
                        continue

    def wait_for(self, run_id: str, poll: float = 2.0,
                 timeout: float = 3600) -> Run:
        """Poll until the run stops moving. Use when you started with wait=False
        and don't want the SSE stream."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            run = self.get(run_id)
            if run.state in ("completed", "failed", "cancelled", "suspended"):
                return run
            time.sleep(poll)
        raise IngestionError(f"run {run_id} did not settle within {timeout}s")
