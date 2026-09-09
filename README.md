# depaperfiy

**Photograph a handwritten note or a printed lab report, and it lands in the patient's chart — structured, in seconds.**

In many clinics, especially in lower-resource settings, notes are still written
by hand and lab results arrive on paper. Someone then re-types all of it into the
computer later — slow, error-prone, and often skipped. MediSnap removes that step:
the clinician photographs what they already wrote (or the report they were
handed), and the information appears in the right place in the record.

---

## What it does

depaperfy is a mock electronic health record (EHR) with a **phone-camera scanner**
bolted on. You open a patient screen on the computer, tap the camera icon, and a
QR code appears. Scan it with your phone, take a photo, and the data shows up back
on the computer — no typing.

It handles two kinds of paper:

- **Handwritten clinical notes** → the chief complaint, history, exam, assessment
  and plan are read out and dropped into the matching fields of a progress note.
- **Printed lab reports** → the results are turned into a clean, editable table.

### Highlights

- **Scan with your phone** — a QR code hands the camera off to your phone; the
  photo never has to be saved or emailed anywhere.
- **Rotate before sending** — upside-down or sideways photos happen. Spin the
  image on the phone before it's uploaded.
- **Single or batch** — scan one page, or photograph a whole stack at once. In a
  batch you **label each page** as a note or a lab report on the phone, and each
  one is filed in the correct tab automatically.
- **Finds the right patient** — the name / ID read off a note is matched against
  the patient list, and the note header fills in from the official record rather
  than from whatever the camera happened to read.
- **Lab results as a table** — a printed report becomes a structured grid whose
  columns match the report itself. You can edit any cell, or add rows and columns
  by hand.
- **Works online or fully offline** — two interchangeable text-reading engines
  (see below): one runs on the device with nothing leaving the machine, the other
  uses a cloud API. You choose.

---

## Try it

You'll need [uv](https://docs.astral.sh/uv/) (a fast Python package manager).

```sh
# 1. Install the app
uv sync

# 2. Run it (bind to 0.0.0.0 so your phone can reach it)
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**, go to **Medical Note** or **Lab Reports**, click the
camera icon, and scan the QR code with your phone.

> **Phone can't connect?** The QR points at your computer's local network address,
> so the phone must be on the **same Wi-Fi**. On locked-down networks, use a tunnel
> (e.g. `cloudflared tunnel --url http://localhost:8000`) and set `PUBLIC_BASE_URL`
> to its URL — see [Configuration](#configuration).

### Choosing the text-reading engine

By default MediSnap uses the **local, on-device** model. For a fast demo on a
laptop without a GPU, the **cloud** option is smoother:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
OCR_PROVIDER=claude uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

To run the on-device model, install its (heavier) dependencies once with
`uv sync --extra medgemma` — details below.

---

## How it works

```
   PHONE                          SERVER                         DESKTOP
 ┌────────┐   photo + label    ┌──────────┐   read & match   ┌───────────┐
 │ camera │ ─────────────────▶ │  OCR +   │ ───────────────▶ │  note /   │
 │ rotate │                    │ structure│                  │ lab table │
 └────────┘ ◀── QR session ─── └──────────┘ ◀── poll ─────── └───────────┘
```

1. The desktop creates a short-lived **scan session** and shows its QR code.
2. The phone opens the session link, captures the photo(s), and uploads them.
3. The server reads each image, turns it into structured data, and (for notes)
   matches the patient.
4. The desktop polls the session and fills the form or table when it's ready.

Scan sessions live in memory with a 30-minute expiry — fine for a demo.

---

## Technical details

### The reading pipeline

Handwritten notes use a **two-stage** split — the right model for each half,
because MedGemma is a strong clinical *reasoner* but a weak *reader*:

```
photo ──▶ [stage 1: Qwen3-VL transcribes]  ──▶  [stage 2: MedGemma routes the
                                                 transcript into note fields
                                                 + patient identifiers]
```

Stage 1 (`TRANSCRIBE_PROVIDER`, default `qwen`) does the OCR; stage 2
(`EXTRACT_PROVIDER`, default `medgemma`) reasons over the clean text — never the
image. On a single 24GB card the two are loaded one at a time (the vision model
is freed before the router loads). Lab reports are printed, not handwritten, so
they skip stage 1 and the extraction model reads the image directly.

For lab reports the structuring step doesn't assume a fixed layout: the table's
columns are derived from whatever the report actually contains (a chemistry panel
has result/unit/range; a differential count adds percentage and absolute columns;
a microbiology line is free text). Extracted results are also emitted as a FHIR
`Observation` bundle server-side, so the data is portable to any FHIR-aware system.

### Text-reading engines (swappable)

Set with the `OCR_PROVIDER` environment variable:

| Provider | `OCR_PROVIDER` | Runs | Notes |
|---|---|---|---|
| **MedGemma** | `medgemma` (default) | On-device | Google's medical vision-language model. Nothing leaves the machine — the offline / data-sovereignty option. Heavy: needs the `medgemma` extra and, realistically, a GPU. |
| **Claude** | `claude` | Cloud API | Strong on messy handwriting and reliable structured output. Needs `ANTHROPIC_API_KEY` and connectivity. |
| **Ollama GGUF** | `ollama` | On-device CPU | Experimental Windows/CPU path using Qwen3-VL 4B Q4_K_M. Runs on 16 GB RAM, but is slower and less accurate than the GPU/cloud paths. |

Both implement the same small interface (`extract`, `extract_labs`, `transcribe`),
so swapping is a one-line config change.

To use MedGemma, install its dependencies (PyTorch + Transformers, ~2GB+):

```sh
uv sync --extra medgemma
```

The first scan downloads the model weights (gated — set `HF_TOKEN`).

### Configuration

All via environment variables (or a `.env` file — see `.env.example`):

| Variable | Purpose |
|---|---|
| `OCR_PROVIDER` | Back-compat shortcut: forces BOTH stages onto one provider (`medgemma` or `claude`) |
| `TRANSCRIBE_PROVIDER` | Stage 1 (OCR): `qwen` (default), `medgemma`, or `claude` |
| `EXTRACT_PROVIDER` | Stage 2 (field routing): `medgemma` (default) or `claude` |
| `OCR_ENGINE` | Local serving engine: `transformers` (default) or `vllm`. vLLM accelerates the **batch** path (`extract_batch`) with paged attention, continuous batching, and JSON-schema-constrained stage-2 decoding; the single-scan path stays on transformers. Needs the separate `vllm` extra (`uv sync --extra medgemma --extra vllm`; Linux/CUDA). |
| `ANTHROPIC_API_KEY` | Required for the Claude provider |
| `CLAUDE_MODEL` | Override the Claude model (default `claude-opus-4-8`) |
| `HF_TOKEN` | Hugging Face token for the gated MedGemma download |
| `PUBLIC_BASE_URL` | URL to encode in the QR (use with a tunnel when the phone can't reach the LAN) |

### Project layout

| Path | Role |
|---|---|
| `app.py` | FastAPI app: EHR pages, scan-session API, QR + phone handoff |
| `ocr.py` | Swappable text-reading: MedGemma (local) or Claude (cloud) |
| `patients.py` | Mock patient database + forgiving identifier matching |
| `labs.py` | Lab report → dynamic table + FHIR `Observation` bundle |
| `data.py` | Mock patient record + the medical-note field schema |
| `templates/` | Facesheet, Medical Note, Lab Reports, and the phone capture page |
| `tests/` | `pytest` suite — pure logic + API, with the model stubbed |
| `tools/` | Standalone dev utilities (not part of the app) |

### Tests

The suite stubs the vision model, so it runs in seconds with no GPU or API key:

```sh
uv run pytest
```

### Docker (on-device path)

`Dockerfile` builds a CUDA image for serving the local MedGemma model on a Linux
GPU host. Docker on macOS can't reach the Mac GPU — run with `uv` there instead.
(The Claude path needs no GPU and no special image.)

```sh
docker build -t medisnap .
docker run --rm --gpus all -p 8000:8000 -e HF_TOKEN=hf_... medisnap
```

---

## OpenMRS 3 demo integration

The repository includes a reproducible OpenMRS 3 Reference Application stack.
It seeds two synthetic patients on first startup; UUIDs are generated by each
installation and are never committed. Start the complete demo with:

```sh
cp .env.example .env       # Windows PowerShell: Copy-Item .env.example .env
docker compose -f docker-compose.openmrs.yml up --build
```

Open OpenMRS at <http://localhost:8080/openmrs/spa> and De-paperfy at
<http://localhost:8000>. The default credentials are `admin` / `Admin123` and
are suitable only for a local synthetic-data demonstration.

The Compose app defaults to the CPU Ollama provider and reaches Ollama on the
host through `host.docker.internal`. Install/pull `qwen3-vl:4b-instruct` first,
or set `DEPAPERFY_OCR_PROVIDER=claude` and `ANTHROPIC_API_KEY` in `.env`.

The bootstrap container safely searches before creating `SYNTH PATIENT 001`
and `SYNTH PATIENT 002`, so restarting the stack does not duplicate them. To
find the installation-specific UUID for one of them:

```sh
curl "http://localhost:8000/api/openmrs/patients?q=SYNTH%20PATIENT%20001"
```

Create a scan session bound to that patient by including
`openmrs_patient_uuid` in `POST /api/scan/session`. After OCR and clinician
review, publish one record explicitly:

```sh
curl -X POST http://localhost:8000/api/scan/session/SESSION_ID/approve \
  -H "Content-Type: application/json" \
  -d '{"openmrs_patient_uuid":"PATIENT_UUID","record_index":0}'
```

Approval validates the patient, maps the reviewed note or lab values to text
concepts, and creates an OpenMRS encounter. It is intentionally separate from
OCR: unreviewed model output is never written automatically. The response
contains the encounter UUID and patient-chart URL.

Previously reviewed JSON artifacts can use the same adapter from the command
line. The confirmation flag is mandatory:

```sh
uv run python -m tools.publish_to_openmrs reviewed.json \
  --identifier "SYNTH PATIENT 001" --yes
```

Implementation lives under `integrations/openmrs/`. `client.py` owns REST API
calls, `mapper.py` maps De-paperfy records, and `bootstrap.py` owns idempotent
demo setup. Set `EHR_MODE=mock` when developing the original mock screens.

Do not use this Compose configuration for real patient data. A production
deployment needs TLS, secret management, non-default accounts, access control,
audit review, backups, pinned/validated images, and local clinical concept
governance.
