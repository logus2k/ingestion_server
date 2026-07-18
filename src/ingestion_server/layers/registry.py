"""Layer resolution — built-in tiers and custom modules, one interface.

Custom layers follow the Skills pattern already proven in this fleet: the code
lives in the service that owns it, mounted from the host, discovered by scan, and
referenced **by name**. Callers hold a reference, never code. That means no
upload endpoint, no code in the DSL, and nothing arbitrary executed from an
editor.
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Optional

from ..config import settings
from ..models import Step
from .base import Layer
from .communities import CommunitiesLayer
from .derived import DerivedLayer
from .llm_extract import LLMLayer
from .structural import StructuralLayer

logger = logging.getLogger(__name__)

BUILTIN: dict[str, type] = {
    "structural": StructuralLayer,
    "llm": LLMLayer,
    "derived": DerivedLayer,
    "communities": CommunitiesLayer,
}


class LayerRef:
    """What `GET /v1/layers` returns — the picker's data source."""

    def __init__(self, name: str, kind: str, description: str = "") -> None:
        self.name = name
        self.kind = kind          # "builtin" | "custom"
        self.description = description

    def to_json(self) -> dict[str, str]:
        return {"name": self.name, "kind": self.kind, "description": self.description}


def list_layers() -> list[LayerRef]:
    out = [LayerRef(name, "builtin", (cls.__doc__ or "").strip().split("\n")[0])
           for name, cls in BUILTIN.items()]
    for name, doc in _scan_custom().items():
        out.append(LayerRef(name, "custom", doc))
    return out


def _scan_custom() -> dict[str, str]:
    """Discover mounted custom layers. A layer is `<layers_dir>/<name>.py`
    exposing `run(ctx) -> LayerResult`."""
    found: dict[str, str] = {}
    root = Path(settings.layers_dir)
    if not root.is_dir():
        return found
    for f in sorted(root.glob("*.py")):
        if f.name.startswith("_"):
            continue
        found[f.stem] = _docline(f)
    return found


def _docline(path: Path) -> str:
    try:
        head = path.read_text(encoding="utf-8")[:400]
        if '"""' in head:
            return head.split('"""')[1].strip().split("\n")[0]
    except Exception:
        pass
    return ""


def _load_custom(ref: str) -> Layer:
    """Import a mounted module by name. Loud on failure — a pipeline naming a
    layer that isn't there must not silently do nothing."""
    safe = Path(ref).name  # never traverse out of layers_dir
    path = Path(settings.layers_dir) / f"{safe}.py"
    if not path.is_file():
        available = ", ".join(sorted(_scan_custom())) or "(none mounted)"
        raise ValueError(f"custom layer '{ref}' not found at {path}. Available: {available}")
    spec = importlib.util.spec_from_file_location(f"custom_layer_{safe}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"custom layer '{ref}' could not be loaded from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run"):
        raise ValueError(f"custom layer '{ref}' has no `run(ctx)` — it cannot be a layer")

    class _Wrapped:
        name = f"custom:{safe}"

        async def run(self, ctx):
            return await mod.run(ctx)

    return _Wrapped()


def resolve(step: Step) -> Layer:
    if step.layer == "custom":
        if not step.ref:
            raise ValueError("custom layer step has no `ref`")
        return _load_custom(step.ref)
    tier = step.tier.value if step.tier else None
    cls = BUILTIN.get(tier or "")
    if cls is None:
        raise ValueError(f"unknown tier '{tier}'. Known: {', '.join(sorted(BUILTIN))}")
    return cls()
