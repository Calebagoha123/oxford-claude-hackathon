"""Review UI for the eval pipeline — a thin reader over eval/runs/*.

Clinicians scroll the extracted notes one image at a time, see the note photo
next to the transcript MedGemma actually retrieved and the fields it filled, and
mark each field **correct** or **incorrect**. There is no LLM judge — the human
is the only judge. Verdicts are written to eval/runs/<id>/reviews.json and rolled
up into a simple correct/incorrect tally.

Mounted on the main app: `app.include_router(eval_ui.router)`. Reads
predictions.json (what the 4B extracted) and writes reviews.json; loads no model.
"""

import csv
import io
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

import eval_pipeline as ep
from data import NOTE_FIELDS

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# The only verdicts — the clinician's call. No judge, no in-between.
HUMAN_VERDICTS = ("correct", "incorrect")


def _run_dir(run_id: str):
    d = ep.RUNS_DIR / run_id
    if not d.is_dir():
        raise HTTPException(404, f"Unknown run {run_id}")
    return d


def _preds(run_id: str) -> list:
    return ep._load_json(_run_dir(run_id) / "predictions.json", [])


def _reviews(run_id: str) -> dict:
    """image -> {field_key: "correct"|"incorrect"}."""
    return ep._load_json(_run_dir(run_id) / "reviews.json", {})


def _summary(preds: list, reviews: dict) -> dict:
    """Human-only tally: how many fields reviewed, and of those how many correct."""
    reviewed = correct = incorrect = 0
    per_image = []
    for p in preds:
        rv = reviews.get(p["image"], {})
        c = sum(1 for v in rv.values() if v == "correct")
        w = sum(1 for v in rv.values() if v == "incorrect")
        reviewed += c + w
        correct += c
        incorrect += w
        per_image.append({"image": p["image"], "reviewed": c + w, "correct": c, "incorrect": w})

    def rate(num, den):
        return round(num / den, 3) if den else None

    return {
        "n_images": len(preds),
        "reviewed": reviewed,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": rate(correct, reviewed),  # of fields a clinician judged, share correct
        "per_image": per_image,
    }


@router.get("/eval", response_class=HTMLResponse)
async def eval_index(request: Request):
    runs = []
    if ep.RUNS_DIR.is_dir():
        for d in sorted((p for p in ep.RUNS_DIR.iterdir() if p.is_dir()), reverse=True):
            preds = _preds(d.name)
            if not preds:
                continue
            runs.append({"id": d.name, "n": len(preds), "summary": _summary(preds, _reviews(d.name))})
    return templates.TemplateResponse(request, "eval_index.html", {"runs": runs})


@router.get("/eval/{run_id}", response_class=HTMLResponse)
async def eval_run(request: Request, run_id: str):
    preds = _preds(run_id)
    return templates.TemplateResponse(
        request, "eval_index.html",
        {"runs": [{"id": run_id, "n": len(preds), "summary": _summary(preds, _reviews(run_id)), "open": True}]},
    )


@router.get("/eval/{run_id}/{idx}", response_class=HTMLResponse)
async def eval_image(request: Request, run_id: str, idx: int):
    preds = _preds(run_id)
    if not preds or idx < 0 or idx >= len(preds):
        raise HTTPException(404, "No such image index in this run")
    p = preds[idx]
    fields = p.get("fields", {})
    human = _reviews(run_id).get(p["image"], {})

    rows = []
    for key, label, kind in NOTE_FIELDS:
        verdict = human.get(key, "")
        rows.append({
            "key": key, "label": label, "kind": kind,
            "value": str(fields.get(key, "") or ""),
            "human": verdict,
            "wrong": verdict == "incorrect",
        })
    summary = _summary(preds, _reviews(run_id))
    pi = next((x for x in summary["per_image"] if x["image"] == p["image"]), {"reviewed": 0, "correct": 0})
    return templates.TemplateResponse(request, "eval_image.html", {
        "run_id": run_id, "idx": idx, "n": len(preds),
        "image": p["image"],
        "transcript": p.get("raw_transcript", "") or "",
        "rows": rows, "human_verdicts": HUMAN_VERDICTS,
        "img_reviewed": pi["reviewed"], "img_correct": pi["correct"],
        "prev": idx - 1 if idx > 0 else None,
        "next": idx + 1 if idx < len(preds) - 1 else None,
    })


@router.get("/api/eval/{run_id}/export.csv")
async def export_csv(run_id: str):
    """Download the run as a CSV — one row per field per note, with the value the
    model extracted and the clinician's verdict (correct/incorrect/blank)."""
    preds = _preds(run_id)
    if not preds:
        raise HTTPException(404, f"No predictions for run {run_id}")
    reviews = _reviews(run_id)
    labels = {key: label for key, label, _kind in NOTE_FIELDS}

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["note", "field", "field_label", "extracted_value", "verdict"])
    for p in preds:
        image = p["image"]
        fields = p.get("fields", {})
        rv = reviews.get(image, {})
        for key, label, _kind in NOTE_FIELDS:
            w.writerow([image, key, labels[key], str(fields.get(key, "") or ""), rv.get(key, "")])

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="depaperfy-review-{run_id}.csv"'},
    )


@router.get("/eval/{run_id}/{idx}/image")
async def eval_image_file(run_id: str, idx: int):
    preds = _preds(run_id)
    if not preds or idx < 0 or idx >= len(preds):
        raise HTTPException(404, "No such image")
    path = ep.IMAGES_DIR / preds[idx]["image"]
    if not path.is_file():
        raise HTTPException(404, "Image file missing")
    return FileResponse(path)


@router.post("/api/eval/{run_id}/{idx}/review")
async def save_review(run_id: str, idx: int, request: Request):
    """Persist one clinician verdict: body {"field": <key>, "verdict": correct|incorrect|""}.
    Sending verdict "" clears the verdict for that field."""
    body = await request.json()
    field, verdict = body.get("field"), body.get("verdict", "")
    if field is None:
        raise HTTPException(400, "Missing 'field'")
    if verdict and verdict not in HUMAN_VERDICTS:
        raise HTTPException(400, f"Bad verdict {verdict!r}")

    preds = _preds(run_id)
    if idx < 0 or idx >= len(preds):
        raise HTTPException(404, "No such image index")
    image = preds[idx]["image"]

    path = _run_dir(run_id) / "reviews.json"
    reviews = ep._load_json(path, {})
    per_img = reviews.setdefault(image, {})
    if verdict:
        per_img[field] = verdict
    else:
        per_img.pop(field, None)
    path.write_text(json.dumps(reviews, indent=2, ensure_ascii=False))

    summary = _summary(preds, reviews)
    return JSONResponse({
        "ok": True,
        "accuracy": summary["accuracy"],
        "reviewed": summary["reviewed"],
        "correct": summary["correct"],
    })
