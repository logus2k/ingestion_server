#!/usr/bin/env bash
# CI ingest — pure curl, no SDK, no Python.
#
# This is the CV Jenkins case: build the PDF, then hand it to the Agent. The
# pipeline travels inline with the request, so the caller is self-contained and
# the Agent stores nothing about it.
#
#   ./examples/ci_ingest.sh /app/static/cv.pdf "António Cruz - Curriculum Vitae.pdf"
#
# Exits non-zero if the run does not complete, so a broken ingest fails the build
# instead of silently leaving the knowledge base stale.
set -euo pipefail

AGENT="${INGESTION_URL:-http://localhost:8700}"
DOC="${1:-/app/static/cv.pdf}"
NAME="${2:-$(basename "$DOC")}"
DB="${CV_DB:-cv}"

read -r -d '' PIPELINE <<JSON || true
{
  "corpus": {
    "context": "a curriculum vitae",
    "language": "en",
    "target_db": "${DB}"
  },
  "chunking": { "strategy": "pdf_docling", "target_tokens": 200 },
  "types": {
    "entities": {
      "organization": {
        "definition": "a named company, employer, client, partner, university or certifying body",
        "examples": ["Acme Corp", "ISCTE", "Microsoft"]
      },
      "role": {
        "definition": "a job title or position held by the person",
        "examples": ["VP Engineering", "CTO"]
      },
      "technology": {
        "definition": "a named product, system, tool, standard, protocol or certification",
        "examples": ["Kubernetes", "AI-901"]
      },
      "domain": {
        "definition": "a field, industry or discipline worked in",
        "examples": ["biometrics", "computer vision"],
        "not": "a named product or company (that is a technology)"
      }
    },
    "relations": {
      "AT_ORGANIZATION": { "from": "role", "to": "organization",
                           "definition": "the role was held at this organization" }
    }
  },
  "steps": [
    { "tier": "llm",
      "entities": ["organization", "role", "technology", "domain"],
      "relations": ["AT_ORGANIZATION"] },
    { "tier": "derived", "relations": ["SIMILAR_TO"], "threshold": 0.75 }
  ]
}
JSON

echo "agent: ${AGENT}"
curl -sf -m 15 "${AGENT}/v1/healthz" >/dev/null || { echo "agent unreachable"; exit 1; }

echo "validating pipeline..."
VALID=$(curl -sf -m 30 -X POST "${AGENT}/v1/pipelines/validate" \
          -H 'Content-Type: application/json' -d "${PIPELINE}")
echo "${VALID}" | grep -q '"ok":true' || { echo "invalid pipeline: ${VALID}"; exit 1; }

echo "ingesting ${NAME} ..."
BODY=$(python3 - "$DOC" "$NAME" <<'PY'
import json, sys, os
pipeline = json.loads(os.environ["PIPELINE"])
print(json.dumps({
    "pipeline": pipeline,
    "documents": [{"path": sys.argv[1], "name": sys.argv[2], "change": "created"}],
    "judge": {"persona": "ingest_judge", "on_suspicion": "notify"},
    "wait": True,
}))
PY
)
export PIPELINE

RUN=$(curl -sf -m 1800 -X POST "${AGENT}/v1/runs" \
        -H 'Content-Type: application/json' -d "${BODY}")

STATE=$(echo "${RUN}" | python3 -c 'import sys,json; print(json.load(sys.stdin)["state"])')
echo "run state: ${STATE}"
echo "${RUN}" | python3 -c '
import sys, json
r = json.load(sys.stdin)
for rep in r.get("reports", []):
    for L in rep.get("layers", []):
        j = L.get("judge") or {}
        flag = "" if not j or j.get("ok", True) else f'"'"'  JUDGE: {j.get("suspicion")}'"'"'
        print(f"  {L[\"name\"]:12} {L[\"state\"]:10} {L[\"seconds\"]:6.1f}s{flag}")
    print("  committed:", rep.get("committed"))
'

# A stale knowledge base is worse than a red build: fail loudly.
[ "${STATE}" = "completed" ] || { echo "INGEST DID NOT COMPLETE (${STATE})"; exit 1; }
echo "OK"
