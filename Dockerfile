# syntax=docker/dockerfile:1.7
#
# Cloud Run image for the medgpt MCP server.
#
# Build:
#   gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT/medgpt/serve:latest .
#
# Runtime expectations:
#   - Cloud Build downloads the FAISS artifacts into `cloudrun_index/`,
#     which this image bakes into `/data/index`.
#   - 32 GB RAM / 8 vCPU recommended (index + ids + bge-m3 + working set).

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf \
    TRANSFORMERS_CACHE=/opt/hf \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# libgomp1 is required by faiss-cpu's OpenMP runtime on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

# Pre-download the embedding model into the image so the first request
# doesn't pay a 2.3 GB download. This is the slowest single step of the
# build — leave it as its own layer so requirements changes don't
# invalidate it.
ARG EMBED_MODEL=BAAI/bge-m3
ENV EMBED_MODEL=${EMBED_MODEL}
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('${EMBED_MODEL}', device='cpu')"

COPY cloudrun_index /data/index
COPY openalex_med /app/openalex_med

ENV INDEX_DIR=/data/index \
    NPROBE=32 \
    DEFAULT_K=50 \
    MAX_K=200 \
    PORT=8080 \
    LOG_LEVEL=INFO

EXPOSE 8080

# Cloud Run sets PORT; uvicorn picks it up via PORT env var.
CMD ["python", "-m", "openalex_med.serve"]
