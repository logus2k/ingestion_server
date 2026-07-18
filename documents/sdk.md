# Ingestion Agent — SDK reference

The Agent turns documents into a searchable corpus and a knowledge graph. It is
**complete on its own**: this API is the product, and every other consumer — a
Patron block, an agent_runtime node, a Jenkins job — is a client of it.

- Base URL: `http://ingestion-server:8700` (host: `http://localhost:8700`)
- Python client: `sdk/python/ingestion_client` (stdlib only)
- Runnable examples: `examples/`
- OpenAPI: `GET /openapi.json`, interactive docs at `/docs`

---

## The model in one page

A **pipeline** is declarative and travels **inline with every run** — the Agent
stores none. That means a caller is self-contained and there is no registry to
keep in sync.

```
pipeline = corpus + chunking + types + index + steps
run      = pipeline + documents
```

- **The engine is single-document.** The API takes a *list* and walks it one at a
  time.
- **Layers are pure.** They return results; they never write. The engine stages
  everything and commits **once per document, atomically**. So a suspended run
  leaves no half-built graph, "repeat this layer" is always safe, and a
  document's additions and removals land in one transaction.
- **A run halts** on the first failure or judge flag with state `suspended` —
  resumable with `retry` / `skip` / `abort`. A crash is `failed` instead.
- **Re-ingest replaces.** The same document twice is wiped and rewritten, never
  duplicated.
- **Entities are shared.** `substance:anticoagulantes` mentioned by two documents
  is ONE vertex both mention. Removing a document deletes only entities left with
  no remaining mentions.

---

## Quick start

```python
from ingestion_client import IngestionClient, Pipeline

c = IngestionClient("http://localhost:8700")

pipeline = (
    Pipeline("a curriculum vitae", target_db="cv", language="en")
    .entity("organization", "a named company, employer or certifying body",
            examples=["Acme Corp", "ISCTE"])
    .entity("role", "a job title held by the person", examples=["VP Engineering"])
    .entity("technology", "a named product, standard or certification",
            examples=["Kubernetes", "AI-901"])
    .entity("domain", "a field or discipline worked in", examples=["biometrics"],
            not_="a named product or company (that is a technology)")
    .relation("AT_ORGANIZATION", "role", "organization")
    .llm(entities=["organization", "role", "technology", "domain"],
         relations=["AT_ORGANIZATION"])
    .derived()
    .build()
)

run = c.ingest(pipeline, ["/watched/in/cv.pdf"])
print(run.state, run.committed)     # completed {'chunks': 82, 'entities': ...}
```

Same thing in curl: `examples/ci_ingest.sh`.

---

## Endpoints

| | |
|---|---|
| `GET /v1/healthz` | liveness + which layers are available |
| `GET /v1/layers` | built-in tiers + any custom layer mounted on the Agent |
| `POST /v1/pipelines/validate` | check a pipeline **without** running it |
| `POST /v1/runs` | ingest. `wait=true` blocks and returns the finished run |
| `GET /v1/runs/{id}` | the run — always the truth, persisted |
| `GET /v1/runs` | recent runs, filterable by `state` |
| `POST /v1/runs/{id}/resume` | `{decision: retry\|skip\|abort}` |
| `POST /v1/runs/{id}/cancel` | cooperative cancel, between layers |
| `GET /v1/runs/{id}/events` | SSE stream of the run's state |

Auth: set `API_TOKEN` on the Agent and pass `IngestionClient(..., token=...)`.
This is a **write API to your graph** — an empty token means open, which is a
dev-only posture.

---

## The pipeline

### corpus

```yaml
corpus:
  context: "a curriculum vitae"   # injected into the extraction prompt
  language: en
  target_db: cv                   # one pipeline = one corpus = one namespace
```

`context` is the **only** corpus-specific thing the extractor is told. The preset
itself is generic; a Portuguese medicine leaflet and an English CV differ by this
line and the type vocabulary, nothing else.

### chunking

| strategy | gives | needs |
|---|---|---|
| `pdf_docling` | chunks + per-item bounding boxes (citations can highlight a page) | docling models |
| `plain_text` | chunks with real heading ancestry, no geometry | nothing |

`target_tokens` is fed to the chunker directly.

### types

```yaml
types:
  entities:
    domain:
      definition: "a field or discipline worked in"
      examples: ["biometrics", "computer vision"]
      not: "a named product or company (that is a technology)"
  relations:
    AT_ORGANIZATION: { from: role, to: organization }
```

`definition` and `examples` carry the whole burden of precision. **`not` is the
contrastive half and it matters**: types that contrast cleanly separate in a
single prompt, while overlapping types (a `term` vs `concept` split) produced 100%
duplication — both claimed the same entities at confidence 1.0.

### steps

Ordered. Each is a tier or a custom layer.

| tier | what it does | cost |
|---|---|---|
| `structural` | what the document's own structure states | free, deterministic |
| `llm` | one call per chunk → entities + typed relations | the expensive one |
| `derived` | embeddings, similarity bands | no inference |
| `communities` | PageRank + Louvain, then a summary per cluster | LLM per community |
| `custom` | a module mounted on the Agent, referenced by `ref` | yours |

Every edge carries `provenance` (`structural` / `derived` / `llm` / `custom:<ref>`)
so a fact's origin is visible and one tier can be re-run without the others.

### index

```yaml
index:
  english_level: { type: string }
  experience_years: { type: int }
```

Promotes an extracted entity to a first-class **indexed chunk property** — for
filtering, not for the graph.

---

## Runs

```python
run = c.ingest(pipeline, docs, judge=..., wait=True)
run.ok            # state == "completed"
run.suspended
run.committed     # {'chunks': 100, 'entities': 179, 'relations': 399}
run.deleted       # {'chunks': 100, 'entities': 78}
run.flags         # every layer the judge was unhappy with
run.blocked_on    # the document a suspended run halted on
run.raw           # the full payload, always
```

### Deleting

```python
c.remove(pipeline, ["/watched/in/old.pdf"])
```

Drops the document's chunks and its `mentions`, then garbage-collects only
entities left with **zero** remaining mentions. An entity other documents still
mention survives.

### Suspend and resume

```python
run = c.ingest(pipeline, docs, judge={"persona": "ingest_judge",
                                      "on_suspicion": "suspend"})
if run.suspended:
    for f in run.flags:
        print(f["layer"], f["suspicion"], f["note"])
    c.resume(run.run_id, "skip")        # retry | skip | abort
```

A suspended run is **persisted** — it survives a service restart and the decision
can arrive hours later, from anywhere.

---

## The judge

A monitoring operator, not a rule engine. After each layer it sees the pipeline
config and a **real sample** of that layer's output plus its distributions, and
returns a verdict.

```python
judge = {"persona": "ingest_judge",
         "template": None,             # inline instruction overrides the default
         "on_suspicion": "notify"}     # notify | suspend
```

- `notify` — publishes to the bus, never halts. **The default.**
- `suspend` — halts the run for a human decision.

It defaults to approving and only reports a concrete defect. It never gates a
deletion.

---

## Events

Published to the bus (`BUS_ENABLED`, `BUS_STREAM_ID`), mirroring the API:

`ingestion.progress` · `ingestion.suspended` · `ingestion.completed` ·
`ingestion.failed`

The API is authoritative; events are a notification channel, so a dead bus never
fails a run.

---

## Custom layers

Code lives in the service that owns it; callers hold a reference — the same
pattern as Skills.

1. Drop `mylayer.py` into the Agent's mounted `layers/` directory, exposing
   `async def run(ctx) -> LayerResult`.
2. `c.layers()` lists it.
3. Reference it: `.custom("mylayer", some_option=1)`.

The pipeline holds `{"layer": "custom", "ref": "mylayer"}` — a name, never code.

---

## Errors

```python
from ingestion_client import IngestionError

try:
    c.ingest(pipeline, docs)
except IngestionError as e:
    e.status     # HTTP status, 0 if unreachable
    e.detail     # parsed server detail
```

`POST /v1/runs` returns **422** with the validation errors when a pipeline is
invalid — the same list `validate()` gives you, so check first and save the
round trip.
