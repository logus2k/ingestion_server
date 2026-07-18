#!/usr/bin/env python3
"""Human-in-the-loop: the judge suspends, a person decides, the run continues.

    python examples/resume_suspended.py

A run halts on the first failure or judge flag with state `suspended` — NOT
terminal. Its earlier documents are already committed; the halted one is staged
and uncommitted. The decision can arrive minutes or hours later, from here, from
a frontend, or from another agent — the run is persisted and survives a restart.
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/sdk/python")
from ingestion_client import IngestionClient, Pipeline   # noqa: E402

client = IngestionClient("http://localhost:8700")

# on_suspicion="suspend" makes the judge HALT rather than just notify.
JUDGE = {"persona": "ingest_judge", "on_suspicion": "suspend"}

pipeline = (
    Pipeline("a Brazilian medicine package leaflet (bula)",
             target_db="example_resume", language="pt")
    .entity("substance", "an active pharmaceutical ingredient or excipient",
            examples=["ácido acetilsalicílico"])
    .llm(entities=["substance"])
    .build()
)

docs = ["/corpora/bulas/AAS - 178170936_profissional.pdf",
        "/corpora/bulas/A SAÚDE DA MULHER - 102351059_profissional.pdf"]

run = client.ingest(pipeline, docs, judge=JUDGE)
print(f"run {run.run_id}: {run.state}")

if not run.suspended:
    print(f"nothing to resume — the run ended {run.state}")
    print(f"committed: {run.committed}")
    raise SystemExit(0)

# ---- a human would look at this and decide --------------------------
print(f"\nSUSPENDED on: {run.blocked_on}")
for f in run.flags:
    print(f"  [{f['layer']}] {f['suspicion']}: {f['note']}")

# retry — re-run that document (e.g. after fixing the model or the prompt)
# skip  — accept the loss and move to the next document
# abort — end the run; staged work for the halted document is discarded
print("\nresuming with 'skip'")
run = client.resume(run.run_id, "skip", note="triaged by examples/resume_suspended.py")

print(f"\nrun {run.run_id}: {run.state}")
print(f"committed: {run.committed}")
