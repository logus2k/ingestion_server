#!/usr/bin/env python3
"""One-shot ingest — the simplest integration, and the one the Patron block uses.

    python examples/run_once.py /corpora/bulas/AAS\\ -\\ 178170936_profissional.pdf

Blocks until the run finishes and prints what landed in the corpus.
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/sdk/python")
from ingestion_client import IngestionClient, Pipeline   # noqa: E402

DOC = sys.argv[1] if len(sys.argv) > 1 else \
    "/corpora/bulas/AAS - 178170936_profissional.pdf"

client = IngestionClient("http://localhost:8700")
print("agent:", client.health()["status"], "| layers:", client.health()["layers"])

# The corpus context is the ONLY corpus-specific thing the extractor is told.
# Everything else — the types, their definitions — travels in the pipeline.
pipeline = (
    Pipeline("a Brazilian medicine package leaflet (bula)",
             target_db="example_bulas", language="pt")
    .entity("substance", "an active pharmaceutical ingredient or excipient",
            examples=["ácido acetilsalicílico", "lactose"])
    .entity("indication", "a condition the medicine is used to treat or prevent",
            examples=["hipertensão", "dor de cabeça"])
    .entity("adverse_reaction", "an unwanted effect the medicine may cause",
            examples=["náusea", "erupção cutânea"])
    .relation("TREATS", "substance", "indication",
              "the substance is indicated for this condition")
    .llm(entities=["substance", "indication", "adverse_reaction"],
         relations=["TREATS"])
    .derived()
    .build()
)

ok, errors = client.validate(pipeline)
if not ok:
    print("pipeline is invalid:", errors)
    raise SystemExit(1)

print(f"\ningesting {DOC} ...")
run = client.ingest(pipeline, [DOC],
                    judge={"persona": "ingest_judge", "on_suspicion": "notify"})

print(f"\nrun {run.run_id}: {run.state}")
if run.error:
    print("error:", run.error)
for rep in run.reports:
    for L in rep.get("layers", []):
        print(f"  {L['name']:12} {L['state']:10} {L['seconds']:6.1f}s")
print("committed:", run.committed)
for f in run.flags:
    print(f"JUDGE FLAG [{f['layer']}] {f['suspicion']}: {f['note']}")
