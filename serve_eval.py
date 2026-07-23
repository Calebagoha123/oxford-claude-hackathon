"""Review-only entrypoint for Cloud Run — serves just the eval review UI.

The full app (app.py) bundles the live scan/OCR demo, which needs a GPU. The
clinician review page does not: it only reads eval/runs/* and writes reviews.json
(see eval_ui.py). This entrypoint mounts that router alone, so it runs on a tiny
CPU container.

Data lives in a GCS bucket mounted at $EVAL_DATA_DIR (see eval_pipeline.EVAL_DIR):
predictions + note images are read from it, and each clinician verdict is written
straight back to reviews.json in the bucket — durable and shared across the team.

Run locally:   uvicorn serve_eval:app --reload
On Cloud Run:  the container runs `uvicorn serve_eval:app --host 0.0.0.0 --port $PORT`
"""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from eval_ui import router as eval_router

app = FastAPI(title="De-paperfy — Eval Review")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(eval_router)


@app.get("/")
async def root():
    return RedirectResponse("/eval")


@app.get("/healthz")
async def healthz():
    return {"ok": True}
