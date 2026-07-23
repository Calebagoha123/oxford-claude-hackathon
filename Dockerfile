# Review-only service for Cloud Run. CPU-only, no model — see serve_eval.py.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-eval.txt .
RUN pip install --no-cache-dir -r requirements-eval.txt

# Only the files the review UI needs. Note images + runs live in the mounted
# GCS bucket ($EVAL_DATA_DIR), not in the image, so this stays small.
COPY serve_eval.py eval_ui.py eval_pipeline.py data.py ./
COPY templates/ templates/
COPY static/ static/

ENV PORT=8080
# Shell form so $PORT is expanded at runtime (Cloud Run injects it).
CMD exec uvicorn serve_eval:app --host 0.0.0.0 --port $PORT
