"""The judge — a monitoring operator, not a rule engine.

It runs after every layer and sees two things: the pipeline configuration (so it
knows what we are trying to extract and why) and a **real sample of that layer's
output** alongside its distributions — not statistics alone.

Deliberately NOT a set of thresholds in code. The point is that it reasons about
what is in front of it, the way an operator would, rather than us trying to
enumerate every way an ingestion can go wrong.

It publishes a notification and returns a verdict. It does not gate deletion.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .models import JudgeConfig, Pipeline, Step, Verdict

logger = logging.getLogger(__name__)

# Relation types the engine itself emits. A pipeline never declares these, so a
# judge told only about `types.relations` will report them as undeclared and flag
# every healthy run. Measured: without this, a clean digest was judged ok=False
# with "output includes a relation type 'mentions' which was not declared".
INTRINSIC_TYPES = ("mentions", "SIMILAR_TO", "member_of", "summarizes", "chunked_into")

_SCHEMA = {
    "name": "ingest_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "suspicion": {"type": "string"},
            "note": {"type": "string"},
        },
        "required": ["ok", "note"],
    },
}

# The recurring operator instincts. Written as things to LOOK for, not thresholds
# to compare against — every one of these is drawn from a real failure:
#   * a silent vector-upsert failure went unnoticed for 24h
#   * a first similarity threshold produced 5 edges/entity where a proven corpus
#     had 0.9, flooding the clustering
#   * an extractor leaned so hard on one type it returned 200 vs 25
DEFAULT_TEMPLATE = """\
You are monitoring an automated ingestion pipeline that turns documents into a
searchable corpus and a knowledge graph. A layer has just finished. Decide whether
its outcome is what a careful operator would expect.

**Default to ok=true.** Only report ok=false when you can name a concrete defect a
human should look at. A monitor that flags healthy runs gets ignored, which is
worse than no monitor at all.

Flag ONLY these:
- output that is empty or near-empty when the input clearly had content
- one entity type swallowing nearly everything (e.g. 90% of entities share a type),
  or a declared type that never appears at all
- obvious near-duplicates that should be one entity (the same thing named twice
  splits the graph)
- entities that plainly could not come from this kind of document
- a stage that produced nothing despite the input being substantial

NEVER flag these — they are how the pipeline works, not defects:
- structural edges outnumbering declared ones. Every chunk-to-entity link is a
  `mentions` edge, so `mentions` ALWAYS dominates. This is correct and expected.
- a layer emitting edge types the corpus does not declare, when those types are
  listed as structural below. `SIMILAR_TO`, `member_of` and `summarizes` are
  produced by the engine, not by the corpus vocabulary.
- a layer reporting the entity count accumulated by EARLIER layers. Later layers
  see and re-report everything found so far; that is the design.
- counts you have no baseline for. You have no history. Absence of evidence is not
  evidence of a problem.
- a layer doing exactly what its step asked for.

Return ONLY JSON: {"ok": true|false, "suspicion": "<short label, empty if ok>",
"note": "<one or two sentences a human would find useful>"}
"""


class Judge:
    def __init__(self, cfg: JudgeConfig, llm) -> None:
        self.cfg = cfg
        self._llm = llm

    async def assess(self, pipeline: Pipeline, step: Step, layer_name: str,
                     digest: dict[str, Any], seconds: float) -> Optional[Verdict]:
        if not self.cfg.enabled:
            return None
        # A skipped layer has nothing to judge.
        if digest.get("skipped"):
            return Verdict(ok=True, note=f"skipped: {digest['skipped']}")

        instruction = self.cfg.template or DEFAULT_TEMPLATE
        prompt = (
            f"{instruction}\n\n"
            f"=== WHAT THIS PIPELINE IS DOING ===\n"
            f"Corpus: {pipeline.corpus.context} (language: {pipeline.corpus.language})\n"
            f"Entity types declared: {_types(pipeline)}\n"
            f"Relations declared: {', '.join(pipeline.types.relations) or '(none)'}\n"
            # Without this the judge flags a HEALTHY run: it sees `mentions` in
            # the output, cannot find it in the declared relations, and reports an
            # undeclared type. These are structural — the engine always emits
            # them and no pipeline declares them.
            f"\nThe following are ALWAYS present and are NOT declared by a pipeline. "
            f"Their presence is normal and must never be flagged as undeclared:\n"
            f"  {', '.join(INTRINSIC_TYPES)}\n"
            f"Entities of type `markdown_doc`, `markdown_chunk`, `community` and "
            f"`community_summary` are likewise structural, not declared.\n\n"
            f"=== THE LAYER THAT JUST RAN ===\n"
            f"Layer: {layer_name}\n"
            f"It was asked for entities: {', '.join(step.entities) or '(none)'}\n"
            f"It was asked for relations: {', '.join(step.relations) or '(none)'}\n"
            f"Wall time: {seconds:.1f}s\n\n"
            f"=== ITS OUTCOME ===\n"
            f"{json.dumps(digest, ensure_ascii=False, indent=2)[:6000]}\n"
        )
        parsed = await self._llm.chat_json(prompt, temperature=0.1, max_tokens=512,
                                           json_schema=_SCHEMA)
        if not isinstance(parsed, dict):
            # A judge that cannot answer must not halt a run — it is an observer.
            logger.warning("judge returned no usable verdict for layer %s", layer_name)
            return Verdict(ok=True, note="judge unavailable — not assessed")
        return Verdict(ok=bool(parsed.get("ok", True)),
                       note=str(parsed.get("note") or "").strip(),
                       suspicion=str(parsed.get("suspicion") or "").strip())

    def should_suspend(self, verdict: Optional[Verdict]) -> bool:
        return (verdict is not None and not verdict.ok
                and self.cfg.on_suspicion == "suspend")


def _types(p: Pipeline) -> str:
    return ", ".join(f"{n} ({t.definition[:60]}…)" if len(t.definition) > 60
                     else f"{n} ({t.definition})"
                     for n, t in p.types.entities.items()) or "(none)"
