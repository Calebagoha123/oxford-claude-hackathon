# syntax=docker/dockerfile:1
#
# MediSnap EHR — for deploying the local MedGemma OCR path on a LINUX host with a
# CUDA GPU. Docker on macOS cannot access the Mac GPU — run locally with `uv run`
# there instead. (For the Claude OCR path you don't need the GPU image at all.)
#
# Build:  docker build -t medisnap .
# Run:    docker run --rm --gpus all -p 8000:8000 -e HF_TOKEN=hf_... medisnap
# Or, simpler, with Compose:  HF_TOKEN=hf_... docker compose up --build
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ca-certificates for HTTPS; tesseract-ocr is the OSD backend pytesseract shells
# out to for image orientation (best-effort — the app degrades without it).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock* ./
# This image is the on-device MedGemma path, so pull the heavy ML extra.
RUN uv sync --extra medgemma --no-install-project --no-dev

# All top-level modules (app.py, ocr.py, data.py, labs.py, patients.py, and — once
# the two-stage PR lands — vllm_engine.py). A glob keeps the image in step with the
# code without editing this list. .dockerignore excludes tests/serve helpers.
COPY *.py ./
COPY templates ./templates

ENV PATH="/app/.venv/bin:$PATH"
# Cache model weights on a mounted volume so they survive container restarts:
#   docker run ... -v hf-cache:/root/.cache/huggingface ...   (Compose does this)
ENV HF_HOME=/root/.cache/huggingface
EXPOSE 8000

# The venv python is on PATH; no curl/wget in the base image. Generous start
# period — first boot downloads weights and warms the model.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
