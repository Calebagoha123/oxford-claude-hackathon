"""De-paperfy EHR — mock EHR that turns a photo of a paper LAB REPORT into a
structured, cross-checkable results table.

A real-looking chart with a "Scan a lab report" action. The clinician either
uploads a photo in the browser or scans a QR to capture one on their phone; the
two-stage extractor (Qwen transcribe -> MedGemma table-routing) reads it, and the
structured header + analyte table appears back on the desktop next to the source
photo so every row can be verified. (Clinical notes / facesheets come later.)

Flow:
  desktop  POST /api/scan/session               -> {id, qr, mobile_url}
  desktop  uploads directly, OR shows QR and waits for the phone
  desktop  polls GET /api/scan/session/{id}
  phone    GET  /m/{id}                          -> camera capture page
  either   POST /api/scan/session/{id}/upload    (image) -> OCR runs in background
  desktop  poll sees status "done" + {meta, results}; fetches the photo to compare

Sessions are in-memory; fine for a demo. The QR encodes the server's LAN URL so
a phone on the same Wi-Fi can reach it. Set PUBLIC_BASE_URL to use a tunnel
(ngrok/cloudflared) when Wi-Fi client isolation blocks phone -> laptop.
"""

import base64
import io
import json
import os
import socket
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import qrcode

import ocr
from data import LAB_COLUMNS, LAB_META, PATIENT
from eval_ui import router as eval_router
from integrations.medical_toolkit.za_lab_report_fhir_generator import (
    build_bundle_from_depaperfy,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the local model in the background so the first scan is fast.
    if os.getenv("WARMUP", "1") == "1":
        threading.Thread(target=ocr.warmup, daemon=True).start()
    yield


app = FastAPI(title="Depaperfy EHR", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Field-extraction eval review UI (reads eval/runs/*; loads no model itself).
app.include_router(eval_router)

# session_id -> {status, text, meta, results, error, created}
# status: waiting -> uploaded -> processing -> done | error
_sessions: dict[str, dict] = {}
# session_id -> raw image bytes, kept out of the polled JSON and served on demand
# so the desktop can show the source photo next to the extracted table.
_session_images: dict[str, bytes] = {}
_sessions_lock = threading.Lock()
_SESSION_TTL = 30 * 60  # seconds; scan sessions are short-lived


def _prune_sessions():
    """Drop scan sessions older than the TTL so the in-memory store can't grow
    without bound. Called on each new session."""
    cutoff = time.time() - _SESSION_TTL
    with _sessions_lock:
        for sid in [s for s, v in _sessions.items() if v.get("created", 0) < cutoff]:
            _sessions.pop(sid, None)
            _session_images.pop(sid, None)


def _lan_ip() -> str:
    """Best-effort LAN IP so a phone on the same network can reach us."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _base_url() -> str:
    explicit = os.getenv("PUBLIC_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    port = os.getenv("PORT", "8000")
    return f"http://{_lan_ip()}:{port}"


def _qr_data_uri(url: str) -> str:
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------- pages
@app.get("/")
async def index():
    return RedirectResponse("/labs")


@app.get("/labs", response_class=HTMLResponse)
async def labs(request: Request):
    return templates.TemplateResponse(
        request,
        "labs.html",
        {
            "patient": PATIENT,
            "active": "labs",
            "meta_fields": LAB_META,
            "columns": LAB_COLUMNS,
            "today": date.today().isoformat(),
        },
    )


@app.get("/facesheet", response_class=HTMLResponse)
async def facesheet(request: Request):
    # Facesheet is a later milestone; the chart entry point is the lab report.
    return templates.TemplateResponse(
        request, "facesheet.html", {"patient": PATIENT, "active": "facesheet"}
    )


@app.get("/m/{session_id}", response_class=HTMLResponse)
async def mobile_capture(request: Request, session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Unknown or expired scan session.")
    return templates.TemplateResponse(
        request, "mobile.html", {"session_id": session_id}
    )


# ----------------------------------------------------------------- scan API
@app.post("/api/scan/session")
async def create_scan_session():
    _prune_sessions()
    session_id = uuid.uuid4().hex[:10]
    with _sessions_lock:
        _sessions[session_id] = {
            "status": "waiting", "text": None, "meta": None, "results": None,
            "error": None, "created": time.time(),
        }
    mobile_url = f"{_base_url()}/m/{session_id}"
    return {"id": session_id, "mobile_url": mobile_url, "qr": _qr_data_uri(mobile_url)}


@app.get("/api/scan/session/{session_id}")
async def scan_status(session_id: str):
    sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Unknown scan session.")
    return sess


def _run_ocr(session_id: str, raw: bytes):
    with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id]["status"] = "processing"
    try:
        result = ocr.extract_labs(raw)  # {text, meta, results}
        with _sessions_lock:
            if session_id in _sessions:
                _sessions[session_id].update(
                    status="done", text=result["text"],
                    meta=result["meta"], results=result["results"],
                )
    except Exception as e:  # noqa: BLE001 - surface to the UI
        with _sessions_lock:
            if session_id in _sessions:
                _sessions[session_id].update(status="error", error=str(e))


@app.post("/api/scan/session/{session_id}/upload")
async def upload_photo(session_id: str, image: UploadFile = File(...)):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Unknown or expired scan session.")
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    with _sessions_lock:
        _sessions[session_id]["status"] = "uploaded"
        _session_images[session_id] = raw  # keep for the cross-check photo
    # OCR can be slow (MedGemma) — run off the request thread; desktop polls.
    threading.Thread(target=_run_ocr, args=(session_id, raw), daemon=True).start()
    return JSONResponse({"ok": True})


@app.get("/api/scan/session/{session_id}/image")
async def scan_image(session_id: str):
    """The photo captured for this session, so the desktop can lay it beside the
    extracted table for row-by-row cross-check. 404 until an image is uploaded."""
    raw = _session_images.get(session_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="No image for this session yet.")
    return Response(content=raw, media_type="image/jpeg")


@app.get("/api/scan/session/{session_id}/fhir.json")
async def scan_fhir(session_id: str):
    """The extracted lab report as a base FHIR R4 bundle (South-African profile).

    Sidecar path: built directly from this session's structured results — no
    external toolkit, so the Observations carry the test name but no LOINC codes
    (LOINC comes only via the full medical-data-toolkit pipeline). See
    docs/medical-toolkit-za-integration.md."""
    sess = _sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Unknown scan session.")
    if sess.get("status") != "done":
        raise HTTPException(status_code=409, detail="Extraction not finished for this session.")
    bundle = build_bundle_from_depaperfy(sess.get("meta"), sess.get("results"))
    return Response(
        content=json.dumps(bundle, indent=2),
        media_type="application/fhir+json",
        headers={"Content-Disposition": f'attachment; filename="depaperfy-labs-{session_id}.fhir.json"'},
    )


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "transcribe_provider": ocr.TRANSCRIBE_PROVIDER,
        "extract_provider": ocr.EXTRACT_PROVIDER,
        "base_url": _base_url(),
    }
