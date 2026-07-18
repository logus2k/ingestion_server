#!/usr/bin/env python3
"""Fire and follow — for a long batch you don't want to block on.

    python examples/stream_progress.py

Starts a run with wait=False, then streams its state. Useful for many documents:
a run walks them one at a time, so a 50-bula batch is minutes of work you want to
watch rather than sit inside a single HTTP call.
"""
import glob
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/sdk/python")
from ingestion_client import IngestionClient, Pipeline   # noqa: E402

client = IngestionClient("http://localhost:8700")

pipeline = (
    Pipeline("a Brazilian medicine package leaflet (bula)",
             target_db="example_batch", language="pt")
    .entity("substance", "an active pharmaceutical ingredient or excipient",
            examples=["ácido acetilsalicílico"])
    .entity("indication", "a condition the medicine treats or prevents",
            examples=["hipertensão"])
    .llm(entities=["substance", "indication"])
    .build()
)

docs = sorted(glob.glob("/corpora/bulas/*.pdf"))[:3]
print(f"queueing {len(docs)} documents")

run = client.ingest(pipeline, docs, wait=False)     # returns immediately
print(f"run {run.run_id} started\n")

seen = None
for snap in client.events(run.run_id):
    state = snap.get("state")
    cursor = snap.get("cursor", 0)
    reports = snap.get("reports", [])
    # Report only what changed — the stream emits on every state transition.
    layers = []
    if cursor < len(reports):
        layers = [f"{L['name']}={L['state']}" for L in reports[cursor].get("layers", [])]
    line = f"[{state}] doc {cursor + 1}/{len(docs)}  {' '.join(layers[-3:])}"
    if line != seen:
        print(line)
        seen = line

final = client.get(run.run_id)
print(f"\nfinal: {final.state}")
print(f"committed: {final.committed}")
if final.suspended:
    print(f"halted on: {final.blocked_on}")
    print("resume with:  client.resume(run_id, 'retry'|'skip'|'abort')")
