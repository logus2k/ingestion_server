# The Ingestion Agent.
#
# Talks to exactly three services — graph-server-arcadedb, embeddings-server,
# agent_server — and to none of the noted stack.
#
# Models are bind-mounted, not baked: they are host assets, they are large, and
# baking them would make every image rebuild carry 1.2GB.
#   /models/docling  ~1.2GB  layout + tableformer weights
#   /models/bge-m3    ~22MB  TOKENIZER ONLY — the chunker counts tokens with it;
#                            the actual embedding runs in embeddings-server, so
#                            the 4.3GB of weights are not needed here.
# Base: the image that already carries the heavy, slow-moving half — docling
# 2.102.1, torch, and the headless-opencv fix (docling pulls opencv-python, which
# links against X11 that a slim image doesn't have; it fails at import with
# "libxcb.so.1: cannot open shared object file").
#
# Reinstalling those from scratch re-downloads torch for no reason. Only the
# service's own deps are added on top.
ARG BASE_IMAGE=logus2k/cv-ingest:test
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DOCLING_ARTIFACTS_ROOT=/models/docling/models \
    BGE_M3_LOCAL=/models/bge-m3 \
    STATE_DB=/data/runs.db \
    LAYERS_DIR=/data/layers \
    PORT=8700

WORKDIR /app

# Just the API layer — docling/torch/httpx are already in the base, and pip
# leaves them alone. Do NOT use --no-deps here: fastapi pins a starlette range,
# and installing it without its deps resolves to an incompatible starlette that
# fails at import with "Router.__init__() got an unexpected keyword argument
# 'on_startup'". The deps are small (starlette, anyio, click).
# python-multipart: required by the stateless /v1/parse endpoint (UploadFile);
# FastAPI raises at import time without it.
RUN pip install --no-cache-dir fastapi==0.116.2 uvicorn==0.35.0 python-multipart==0.0.20

COPY src/ /app/src/
ENTRYPOINT []

EXPOSE 8700
CMD ["uvicorn", "ingestion_server.api:app", "--host", "0.0.0.0", "--port", "8700"]
