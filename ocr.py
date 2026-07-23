"""OCR + structured extraction — a two-stage, independently-swappable pipeline.

MedGemma-4B is a strong clinical *reasoner* but a weak *reader*: doing OCR and
field-routing in one shot, its transcription dragged the whole output down. So we
split the job and use the right model for each half:

  stage 1  image  -> transcript   TRANSCRIBE_PROVIDER (default "qwen")
  stage 2  transcript -> fields   EXTRACT_PROVIDER    (default "medgemma")

Transcription backends (TRANSCRIBE_PROVIDER):
  - "qwen" (default): local Qwen3.6-27B vision model. Far better handwriting OCR.
    27B won't fit a 24GB L4 in bf16, so it loads 4-bit by default (QWEN_QUANT=1).
  - "medgemma": the old single-model behaviour (MedGemma-4B reads the image too).
  - "claude": Claude vision via the Anthropic API; needs ANTHROPIC_API_KEY.

Extraction backends (EXTRACT_PROVIDER) — text-only, they never see the image:
  - "medgemma" (default): local MedGemma-4B routing the clean transcript into
    note fields. This is what MedGemma is good at.
  - "claude": Claude via the Anthropic API; needs ANTHROPIC_API_KEY.

Serving engine (OCR_ENGINE, local models only):
  - "transformers" (default): HF generate(). Used everywhere, including the
    latency-sensitive single-scan live path.
  - "vllm": paged-attention + continuous batching for the BATCH path
    (extract_batch, i.e. the eval pipeline). Stage 2 gets JSON-schema-constrained
    decoding. One engine resident at a time (swapped between stages); see
    vllm_engine.py. Not used for single-scan — rebuilding an engine per request
    would cost more than it saves.

Public functions:
  transcribe(image_bytes)          -> str             # stage 1 only
  extract_fields_from_text(text)   -> {note_key: value}  # stage 2 only
  extract(image_bytes)             -> {text, fields}  # both stages
  extract_batch(image_list)        -> [{text, fields}]   # batch (vLLM-accelerated)
  warmup()                                            # preload the local models
"""

import base64
import io
import json
import os
import re
import threading

# Reduce CUDA fragmentation before torch binds — the 4-bit 27B leaves only ~6GB
# of headroom on a 24GB L4, so fragmentation is the difference between fitting and
# OOM. setdefault so an explicit env still wins. (torch is imported lazily below,
# so this lands before the CUDA allocator is created.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PIL import Image, ImageOps

from data import LAB_COLUMNS, LAB_FLAGS, LAB_META, NOTE_FIELDS

# Two independently-swappable stages. OCR_PROVIDER is kept as a back-compat
# shortcut: if set, it forces BOTH stages onto that provider (the old behaviour).
_LEGACY = os.getenv("OCR_PROVIDER")
TRANSCRIBE_PROVIDER = (_LEGACY or os.getenv("TRANSCRIBE_PROVIDER", "qwen")).lower()
EXTRACT_PROVIDER = (_LEGACY or os.getenv("EXTRACT_PROVIDER", "medgemma")).lower()
# Legacy alias some callers still read (e.g. the /healthz endpoint).
PROVIDER = EXTRACT_PROVIDER

# Local-inference engine: "transformers" (default) or "vllm". vLLM only kicks in
# on the BATCH path (extract_batch, i.e. the eval pipeline) where its continuous
# batching + paged attention pay off and the one-engine-at-a-time model swap is
# amortized over many images. The single-scan live path stays on transformers —
# rebuilding a vLLM engine per request would dwarf the inference it saves. Only
# the default local combo (qwen -> medgemma) is vLLM-accelerated; other provider
# mixes fall through to transformers.
OCR_ENGINE = os.getenv("OCR_ENGINE", "transformers").lower()
# vLLM engine-build knobs (per model, shared defaults). max_model_len bounds the
# KV cache; keep it just above vision-tokens + generated tokens for headroom.
_VLLM_GPU_UTIL = float(os.getenv("VLLM_GPU_UTIL", "0.90"))
_VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
# Upper bound on generated tokens. generate() requires *a* cap (without one,
# transformers falls back to max_length=20). It's a ceiling, not a target —
# the model stops at EOS when the JSON is done, so a high value is ~free.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))

_FIELD_KEYS = [key for key, _label, _kind in NOTE_FIELDS]

TRANSCRIBE_PROMPT = (
    "You are transcribing a handwritten clinical note from a physician. "
    "Transcribe ALL handwritten and printed text in this image exactly as written, "
    "preserving line breaks and reading order. Expand nothing, infer nothing, add no "
    "commentary. Rotate the image if needed to read it. If a word is illegible, write "
    "[illegible]. Output only the transcription."
)

# Stage-2 prompt: strict field-routing rules, but the input is a clean TEXT
# transcript (from stage 1) instead of the image, and we don't ask for
# raw_transcript back — we already have it. MedGemma reasons over text here, which
# is what it's good at.
_TEXT_FIELD_TEMPLATE = ",\n".join(f'  "{k}": ""' for k in _FIELD_KEYS)
_EXTRACT_FROM_TEXT_HEAD = (
    "Below is the full transcript of a handwritten clinical note. Return ONLY a JSON "
    "object — no thinking, no explanation, no markdown fences, nothing but the JSON. "
    "Use exactly these keys:\n"
    "{\n"
    f"{_TEXT_FIELD_TEMPLATE}\n"
    "}\n"
    "Fill the section fields (chief_complaint, hpi, pmhx, fmhx, shx, ros, pe, assessment, "
    "plan, note_type) ONLY from information explicitly present in the transcript.\n"
    "STRICT RULES — accuracy matters far more than completeness:\n"
    '- Most notes fill only a FEW sections. Leaving a field as "" is correct and expected; '
    "an empty field is better than a wrong one.\n"
    '- If the transcript contains nothing for a section, set it to "". Do NOT guess, infer, '
    "or substitute the closest-looking text from elsewhere.\n"
    "- Never copy one field's content into another (e.g. do not repeat the chief complaint "
    "in pmhx or assessment).\n"
    "- Put each piece of information in the single most appropriate field; never duplicate "
    "it across fields.\n"
    "- Do not pad, extend, or repeat list items; include only what is actually written.\n"
    "Do not invent information. Your entire response must start with { and end with }.\n"
)


def _extract_from_text_prompt(transcript: str) -> str:
    return (
        f"{_EXTRACT_FROM_TEXT_HEAD}\n"
        "=== TRANSCRIPT ===\n"
        f"{transcript.strip()}\n"
        "=== END TRANSCRIPT ==="
    )


# Auto-orient before the model reads. MedGemma cannot read sideways/upside-down
# handwriting — accuracy collapses to ~0 and it won't self-rotate — so we rotate
# the photo upright first. Tesseract's Orientation-and-Script Detection (OSD)
# picks the 0/90/180/270 that stands the text up. Best-effort: if tesseract is
# missing or unsure, we leave the image untouched.
ORIENT = os.getenv("ORIENT", "1") == "1"
# OSD is trained on printed text and gets shaky on pure handwriting; ignore a
# rotation call unless its confidence clears this bar.
ORIENT_MIN_CONF = float(os.getenv("ORIENT_MIN_CONF", "2.0"))
# OSD (the tesseract call) is the expensive, unreliable half of orientation.
# Phone cameras almost always stamp a real orientation into EXIF, which is
# accurate — so when that's present we skip OSD entirely. Only fall back to OSD
# when EXIF is absent/normal (a note laid down sideways on a flatbed, or a
# stripped JPEG). Set OSD_SKIP_IF_EXIF=0 to always run OSD.
OSD_SKIP_IF_EXIF = os.getenv("OSD_SKIP_IF_EXIF", "1") == "1"


def _exif_orientation(img: Image.Image) -> int:
    """The image's EXIF orientation tag (1 = normal/none; 3/6/8 = 180/270/90)."""
    try:
        return int(img.getexif().get(0x0112, 1))
    except Exception:  # noqa: BLE001 - no/broken EXIF
        return 1


def _orient_upright(img: Image.Image) -> Image.Image:
    if not ORIENT:
        return img
    # 1) Honor the camera's own EXIF orientation (reliable; matters for phone
    #    captures in the live app — sips already baked it into the eval JPGs).
    had_exif_rotation = _exif_orientation(img) in (3, 6, 8)
    img = ImageOps.exif_transpose(img)
    # If EXIF already carried a real rotation, it's stood the text up correctly —
    # skip the slow, handwriting-shaky OSD pass entirely.
    if OSD_SKIP_IF_EXIF and had_exif_rotation:
        return img
    # 2) Then use OSD for paper-relative rotation (note laid down sideways).
    #    Detect on a downscaled probe — orientation doesn't need 12MP, and
    #    full-res OSD is several seconds per image.
    try:
        import pytesseract

        probe = img
        if max(img.size) > 1600:
            probe = img.copy()
            probe.thumbnail((1600, 1600))
        osd = pytesseract.image_to_osd(probe, output_type=pytesseract.Output.DICT)
        rot = int(osd.get("rotate", 0)) % 360
        if rot and float(osd.get("orientation_conf", 0)) >= ORIENT_MIN_CONF:
            # OSD 'rotate' is the clockwise degrees needed to make text upright;
            # PIL rotate() is counter-clockwise, so negate. Applied to full res.
            img = img.rotate(-rot, expand=True)
    except Exception:  # noqa: BLE001 - tesseract absent / OSD failed: use as-is
        pass
    return img


def _pil(image_bytes: bytes) -> Image.Image:
    return _orient_upright(Image.open(io.BytesIO(image_bytes)).convert("RGB"))


# ---------------------------------------------------------------- MedGemma (local)
_MODEL_ID = "google/medgemma-1.5-4b-it"
# 4-bit the router (MEDGEMMA_QUANT=1) so it co-resides with a *bf16* Qwen on a 24GB
# L4: MedGemma-4b is ~3GB in nf4 vs ~8GB in bf16, which is the difference between
# both models fitting resident (fast, no per-scan model reload) and OOM. Stage 2 is
# a text-routing task, so 4-bit costs little accuracy. Default off (bf16).
_MEDGEMMA_QUANT = os.getenv("MEDGEMMA_QUANT", "0") == "1"
_model = None
_proc = None
_device = None
_model_lock = threading.Lock()


def _load_model():
    """Load MedGemma directly (model + processor) rather than the high-level
    pipeline: we need to prefill the assistant turn to suppress the model's
    trained-in 'thinking', which the pipeline can't do."""
    global _model, _proc, _device
    if _model is None:
        with _model_lock:
            if _model is None:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor

                kwargs: dict = {}
                if torch.cuda.is_available():
                    _device = "cuda"
                    if _MEDGEMMA_QUANT:
                        from transformers import BitsAndBytesConfig

                        kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                        )
                        kwargs["device_map"] = "auto"
                    else:
                        kwargs["dtype"] = torch.bfloat16
                        kwargs["device_map"] = _device
                elif torch.backends.mps.is_available():
                    _device = "mps"
                    kwargs["dtype"], kwargs["device_map"] = torch.float16, "mps"
                else:
                    _device = "cpu"
                    kwargs["dtype"] = torch.float32
                _proc = AutoProcessor.from_pretrained(_MODEL_ID)
                _model = AutoModelForImageTextToText.from_pretrained(
                    _MODEL_ID, **kwargs).eval()
    return _model, _proc


def _messages(image_bytes: bytes, prompt: str) -> list:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": _pil(image_bytes)},
            {"type": "text", "text": prompt},
        ],
    }]


# Repetition control + thinking suppression. MedGemma-1.5 greedy-decodes into
# repetition loops on handwriting, and its trained-in "thinking" (<unused94>thought
# …) spirals on dense forms — burning the whole token budget before any JSON is
# emitted. So on the extract path we PREFILL the assistant turn with "{": the model
# starts the JSON immediately and never enters the thinking block (~5x faster, far
# more reliable). A repetition_penalty still guards against loops; no_repeat_ngram
# is NOT used — it corrupts JSON and wrecks free-text transcription.
TRANSCRIBE_GEN = {"repetition_penalty": 1.3}
EXTRACT_GEN = {"repetition_penalty": 1.3}
EXTRACT_PREFILL = "{"


def _medgemma_generate(msgs: list, prefill: str | None = None, **gen) -> str:
    """Core generate loop, shared by the image and text-only paths."""
    import torch

    model, proc = _load_model()
    if prefill is not None:
        # Continue a partial assistant message so generation resumes after `prefill`.
        msgs = msgs + [{"role": "assistant", "content": [{"type": "text", "text": prefill}]}]
        inputs = proc.apply_chat_template(
            msgs, add_generation_prompt=False, continue_final_message=True,
            tokenize=True, return_dict=True, return_tensors="pt")
    else:
        inputs = proc.apply_chat_template(
            msgs, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    n_in = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, **gen)
    text = proc.decode(out[0][n_in:], skip_special_tokens=True)
    return (prefill + text) if prefill else text


def _medgemma_run(image_bytes: bytes, prompt: str, prefill: str | None = None, **gen) -> str:
    return _medgemma_generate(_messages(image_bytes, prompt), prefill, **gen)


def _medgemma_text_run(prompt: str, prefill: str | None = None, **gen) -> str:
    """Text-only MedGemma (stage 2): reasons over the transcript, no image."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return _medgemma_generate(msgs, prefill, **gen)


# ---------------------------------------------------------------- Qwen (local, stage 1)
# Qwen3-VL is a vision model with markedly better handwriting OCR than MedGemma.
# It's an image-text-to-text model, so the same AutoModelForImageTextToText /
# AutoProcessor pair that loads MedGemma loads it too — just a different repo.
#
# Default is the dense 8B: it fits a 24GB L4 in bf16 (~16GB) with headroom, runs
# standard (fast) attention, and is the non-thinking Instruct variant. The 27B
# (Qwen/Qwen3.6-27B) is stronger but needs 4-bit on an L4 AND runs a slow torch
# fallback for its Gated-DeltaNet layers (~10x slower) — use it only on a big card
# (set QWEN_MODEL + QWEN_QUANT=1). See the two-model note in extract_batch.
_QWEN_ID = os.getenv("QWEN_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
# bf16 by default (the 8B fits). Set QWEN_QUANT=1 to 4-bit a bigger checkpoint.
_QWEN_QUANT = os.getenv("QWEN_QUANT", "0") == "1"
# Vision tokens scale with pixel count, and on a 24GB L4 they're the main OOM
# risk (a 12MP phone photo is thousands of image tokens). Cap the long side so a
# full-page note still fits alongside the 4-bit weights. 1600px keeps handwriting
# legible while staying in budget; raise on a bigger card.
_QWEN_MAX_SIDE = int(os.getenv("QWEN_MAX_SIDE", "1600"))
# A note transcript is rarely more than ~1500 tokens; capping generation keeps the
# KV cache small (more headroom) and avoids runaway decoding.
_QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "1536"))
_qwen_model = None
_qwen_proc = None
_qwen_device = None
_qwen_lock = threading.Lock()

# Qwen greedy-decodes cleanly on handwriting; it doesn't need MedGemma's heavy
# repetition_penalty (which would hurt verbatim transcription). A light nudge only.
QWEN_TRANSCRIBE_GEN = {"repetition_penalty": 1.05}


def _qwen_image(image_bytes: bytes) -> Image.Image:
    """Orient upright, then cap the long side to bound the vision-token count."""
    img = _pil(image_bytes)
    if max(img.size) > _QWEN_MAX_SIDE:
        img = img.copy()
        img.thumbnail((_QWEN_MAX_SIDE, _QWEN_MAX_SIDE))
    return img


def _load_qwen():
    global _qwen_model, _qwen_proc, _qwen_device
    if _qwen_model is None:
        with _qwen_lock:
            if _qwen_model is None:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor

                kwargs: dict = {}
                if torch.cuda.is_available():
                    _qwen_device = "cuda"
                    if _QWEN_QUANT:
                        from transformers import BitsAndBytesConfig

                        kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                        )
                        # bitsandbytes needs an accelerate device_map; "auto" lands on
                        # the single GPU made visible via CUDA_VISIBLE_DEVICES.
                        kwargs["device_map"] = "auto"
                    else:
                        kwargs["dtype"] = torch.bfloat16
                        kwargs["device_map"] = _qwen_device
                elif torch.backends.mps.is_available():
                    _qwen_device = "mps"
                    kwargs["dtype"], kwargs["device_map"] = torch.float16, "mps"
                else:
                    _qwen_device = "cpu"
                    kwargs["dtype"] = torch.float32
                _qwen_proc = AutoProcessor.from_pretrained(_QWEN_ID)
                _qwen_model = AutoModelForImageTextToText.from_pretrained(
                    _QWEN_ID, **kwargs).eval()
    return _qwen_model, _qwen_proc


def _free_qwen() -> None:
    """Release the vision model from the GPU so the extractor can load without the
    two co-residing on a single card. Safe to call when nothing is loaded."""
    global _qwen_model
    if _qwen_model is not None:
        import gc

        import torch

        _qwen_model = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - no CUDA / already gone
            pass


def _qwen_transcribe(image_bytes: bytes) -> str:
    import torch

    model, proc = _load_qwen()
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image", "image": _qwen_image(image_bytes)},
            {"type": "text", "text": TRANSCRIBE_PROMPT},
        ],
    }]
    # Qwen3.6 is a hybrid *thinking* model: left on, it spends the whole token
    # budget reasoning ("The user wants me to transcribe...") and never reaches the
    # transcript. enable_thinking=False injects an empty <think></think> so it
    # answers directly — correct output AND far fewer tokens (critical: decode runs
    # the slow torch fallback here, the fast DeltaNet kernels won't build on CUDA 13).
    inputs = proc.apply_chat_template(
        msgs, add_generation_prompt=True, enable_thinking=False,
        tokenize=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(_qwen_device) for k, v in inputs.items()}
    n_in = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=_QWEN_MAX_NEW_TOKENS,
                             do_sample=False, **QWEN_TRANSCRIBE_GEN)
    del inputs
    text = proc.decode(out[0][n_in:], skip_special_tokens=True).strip()
    torch.cuda.empty_cache()
    # Defensive: if a think block still slips through, keep only the answer after it.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    return text


# ---------------------------------------------------------------- Claude (cloud)
def _claude_client():
    from anthropic import Anthropic

    return Anthropic(), os.getenv("CLAUDE_MODEL", "claude-opus-4-8")


def _claude_image_block(image_bytes: bytes) -> dict:
    buf = io.BytesIO()
    _pil(image_bytes).save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}


def _claude_run(image_bytes: bytes, prompt: str) -> str:
    client, model = _claude_client()
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": [_claude_image_block(image_bytes), {"type": "text", "text": prompt}]}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _claude_fields_from_text(transcript: str) -> str:
    """Stage-2 Claude: fields-only JSON from a text transcript (no image)."""
    client, model = _claude_client()
    schema = {
        "type": "object",
        "properties": {k: {"type": "string"} for k in _FIELD_KEYS},
        "required": list(_FIELD_KEYS),
        "additionalProperties": False,
    }
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": _extract_from_text_prompt(transcript)}]}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


# ---------------------------------------------------------------- public API
def _parse_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*", "", t).strip().rstrip("`").strip()
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        t = t[i:j + 1]
    return json.loads(t)


def _json_str(s: str) -> str:
    """Decode a captured JSON string body (handle escapes; tolerate raw newlines)."""
    try:
        return json.loads('"' + s + '"')
    except Exception:  # noqa: BLE001
        return s.replace('\\n', '\n').replace('\\"', '"').replace('\\t', '\t').strip()


def _regex_value(text: str, key: str) -> str:
    """Pull one "key": "value" out of (possibly truncated/invalid) JSON-ish text."""
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    return _json_str(m.group(1)).strip() if m else ""


def _salvage_fields(text: str) -> dict:
    return {k: _regex_value(text, k) for k in _FIELD_KEYS}


def warmup():
    """Preload whichever local models the two stages use AND run a tiny generation
    so CUDA kernels are compiled at startup — the first real scan is then fast, not
    cold. Best-effort; API-backed stages are no-ops."""
    try:
        import torch

        if TRANSCRIBE_PROVIDER == "qwen":
            model, proc = _load_qwen()
            inputs = proc.apply_chat_template(
                _messages_stub(), add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt")
            inputs = {k: v.to(_qwen_device) for k, v in inputs.items()}
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=1, do_sample=False)
        if TRANSCRIBE_PROVIDER == "medgemma" or EXTRACT_PROVIDER == "medgemma":
            model, proc = _load_model()
            inputs = proc.apply_chat_template(
                _messages_stub(), add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt")
            inputs = {k: v.to(_device) for k, v in inputs.items()}
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=1, do_sample=False)
    except Exception:  # noqa: BLE001 - warmup is best-effort
        pass


def _messages_stub() -> list:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": Image.new("RGB", (64, 64), "white")},
            {"type": "text", "text": "ok"},
        ],
    }]


def transcribe(image_bytes: bytes) -> str:
    """Stage 1: image -> verbatim transcript."""
    if TRANSCRIBE_PROVIDER == "claude":
        return _claude_run(image_bytes, TRANSCRIBE_PROMPT)
    if TRANSCRIBE_PROVIDER == "medgemma":
        return _medgemma_run(image_bytes, TRANSCRIBE_PROMPT, **TRANSCRIBE_GEN)
    return _qwen_transcribe(image_bytes)


def extract_fields_from_text(transcript: str) -> dict:
    """Stage 2: transcript text -> {note_key: value}. Text-only; no image."""
    if not transcript.strip():
        return {k: "" for k in _FIELD_KEYS}
    if EXTRACT_PROVIDER == "claude":
        raw = _claude_fields_from_text(transcript)
    else:
        raw = _medgemma_text_run(
            _extract_from_text_prompt(transcript), prefill=EXTRACT_PREFILL, **EXTRACT_GEN)
    return _assemble_fields(raw)


def extract(image_bytes: bytes) -> dict:
    """Both stages: image -> {text: transcript, fields: {note_key: value}}.

    Stage 1 transcribes the image (Qwen by default); stage 2 routes that clean
    transcript into fields (MedGemma by default). Field JSON that can't be parsed
    degrades to empty fields, so the UI can still show the transcription.
    """
    import time as _time

    _t0 = _time.time()
    transcript = transcribe(image_bytes)
    _t1 = _time.time()
    # Fallback residency lever: a bf16 Qwen (~16GB) + bf16 MedGemma (~8GB) don't
    # co-reside on a 24GB L4. Preferred fix is MEDGEMMA_QUANT=1 (both resident,
    # ~19GB, no reload). If MedGemma must stay bf16, set FREE_QWEN_AFTER_TRANSCRIBE=1
    # to release the vision model before the router loads — sequential residency at
    # the cost of reloading Qwen on the next scan.
    if os.getenv("FREE_QWEN_AFTER_TRANSCRIBE") == "1":
        _free_qwen()
    fields = extract_fields_from_text(transcript)
    _t2 = _time.time()
    print(
        f"[timing] transcribe={_t1 - _t0:.1f}s extract={_t2 - _t1:.1f}s "
        f"total={_t2 - _t0:.1f}s chars={len(transcript)}",
        flush=True,
    )
    return {"text": transcript, "fields": fields}


def _flat(v) -> str:
    """Model sometimes returns a field as a JSON array of items — join to text."""
    if isinstance(v, list):
        return "\n".join(str(x).strip() for x in v if str(x).strip())
    return str(v or "").strip()


def _assemble_fields(raw: str) -> dict:
    """Turn one stage-2 response into {note_key: value}, salvaging broken JSON."""
    try:
        data = _parse_json(raw)
        return {k: _flat(data.get(k)) for k in _FIELD_KEYS}
    except Exception:  # noqa: BLE001 - JSON malformed/truncated: salvage per-key
        return _salvage_fields(raw)


# ---------------------------------------------------------------- lab reports
# The demo's live path. Same two-stage split as notes — stage 1 transcribes the
# image (Qwen), stage 2 routes the clean text into structured fields (MedGemma) —
# but a lab report is a HEADER + a TABLE, so stage 2 emits {meta, results:[...]}
# instead of a flat field dict. transcribe() is shared verbatim; only the stage-2
# prompt/parse differ, so the note eval pipeline is completely untouched.
_LAB_META_KEYS = [k for k, _label in LAB_META]
_LAB_ROW_KEYS = [k for k, _label in LAB_COLUMNS]

_LAB_EXTRACT_HEAD = (
    "Below is the full transcript of a laboratory test report. Return ONLY a JSON "
    "object — no thinking, no explanation, no markdown fences, nothing but the JSON. "
    "Use exactly this shape:\n"
    "{\n"
    '  "meta": {' + ", ".join(f'"{k}": ""' for k in _LAB_META_KEYS) + "},\n"
    '  "results": [\n'
    '    {"test": "", "value": "", "unit": "", "reference_range": "", "flag": ""}\n'
    "  ]\n"
    "}\n"
    "STRICT RULES — accuracy matters far more than completeness:\n"
    '- Add one object to "results" for EVERY analyte/measurement actually present in '
    "the report. Copy the test name, value, unit and reference range exactly as written.\n"
    '- "value" is the measured result as written — a number like "13.2", or a word like '
    '"Positive"/"Not detected".\n'
    '- Leave any field "" when the report does not state it. Do NOT guess, infer, or '
    "invent tests, values, units or ranges. An empty field is better than a wrong one.\n"
    '- "flag": use "high", "low", "critical" or "abnormal" ONLY when the report itself '
    "marks the result that way (an H / L / * / HIGH / LOW / CRIT flag beside the value, or "
    'a value plainly outside the stated reference range). Otherwise "".\n'
    '- Fill "meta" from the report header only; leave any unknown header field "".\n'
    "Your entire response must start with { and end with }.\n"
)


def _extract_labs_prompt(transcript: str) -> str:
    return (
        f"{_LAB_EXTRACT_HEAD}\n"
        "=== TRANSCRIPT ===\n"
        f"{transcript.strip()}\n"
        "=== END TRANSCRIPT ==="
    )


# Single-pass variant: the model reads the IMAGE directly and emits {meta, results}
# in one go, skipping the verbatim transcript. Same shape + rules as the two-stage
# head, only the framing changes (image, not transcript). Far fewer generated tokens
# than transcribe+route, so it's markedly faster on clean printed reports — where
# MedGemma reads well enough that a separate OCR stage buys little.
_LAB_EXTRACT_IMAGE_HEAD = (
    "The image is a photograph or scan of a laboratory test report. Read it and "
    "return ONLY a JSON object — no thinking, no explanation, no markdown fences, "
    "nothing but the JSON. Use exactly this shape:\n"
    + _LAB_EXTRACT_HEAD.split("Use exactly this shape:\n", 1)[1]
)


def _norm_flag(v) -> str:
    """Fold the model's flag onto our vocabulary (LAB_FLAGS); normal/unknown -> ""."""
    s = str(v or "").strip().lower()
    if s in LAB_FLAGS:
        return s
    return {
        "h": "high", "hi": "high",
        "l": "low", "lo": "low",
        "crit": "critical", "panic": "critical", "critical high": "critical",
        "critical low": "critical", "*": "critical",
        "a": "abnormal", "abn": "abnormal", "pos": "abnormal", "positive": "abnormal",
    }.get(s, "")


def _norm_row(row: dict) -> dict:
    r = {k: _flat(row.get(k)) for k in _LAB_ROW_KEYS}
    r["flag"] = _norm_flag(row.get("flag"))
    return r


def _assemble_labs(raw: str) -> dict:
    """Turn one stage-2 response into {meta, results}, salvaging broken JSON as best
    we can (a truncated results array still yields the rows that parsed)."""
    try:
        data = _parse_json(raw)
    except Exception:  # noqa: BLE001 - malformed/truncated JSON
        data = _salvage_labs(raw)
    meta_in = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    meta = {k: _flat(meta_in.get(k)) for k in _LAB_META_KEYS}
    rows_in = data.get("results") if isinstance(data.get("results"), list) else []
    results = [_norm_row(r) for r in rows_in if isinstance(r, dict)]
    # Drop fully-empty rows (a stray {} the model sometimes appends).
    results = [r for r in results if any(r[k] for k in _LAB_ROW_KEYS)]
    return {"meta": meta, "results": results}


def _salvage_labs(raw: str) -> dict:
    """Best-effort recovery when the whole object won't parse: pull the meta values
    by key, and parse whatever result objects are individually well-formed."""
    meta = {k: _regex_value(raw, k) for k in _LAB_META_KEYS}
    results = []
    # Each result object spans "test" .. the next "}" — parse them one at a time so
    # a single broken row doesn't lose the rest.
    for m in re.finditer(r"\{[^{}]*\"test\"[^{}]*\}", raw, re.DOTALL):
        try:
            results.append(json.loads(m.group(0)))
        except Exception:  # noqa: BLE001 - skip an unparseable row
            pass
    return {"meta": meta, "results": results}


def _claude_labs_from_text(transcript: str) -> str:
    """Stage-2 Claude for labs: {meta, results} JSON from a text transcript."""
    client, model = _claude_client()
    row_schema = {
        "type": "object",
        "properties": {k: {"type": "string"} for k in _LAB_ROW_KEYS},
        "required": list(_LAB_ROW_KEYS),
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "meta": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in _LAB_META_KEYS},
                "required": list(_LAB_META_KEYS),
                "additionalProperties": False,
            },
            "results": {"type": "array", "items": row_schema},
        },
        "required": ["meta", "results"],
        "additionalProperties": False,
    }
    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": _extract_labs_prompt(transcript)}]}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def extract_labs_from_text(transcript: str) -> dict:
    """Stage 2 for labs: transcript text -> {meta, results}. Text-only; no image."""
    if not transcript.strip():
        return {"meta": {k: "" for k in _LAB_META_KEYS}, "results": []}
    if EXTRACT_PROVIDER == "claude":
        raw = _claude_labs_from_text(transcript)
    else:
        raw = _medgemma_text_run(
            _extract_labs_prompt(transcript), prefill=EXTRACT_PREFILL, **EXTRACT_GEN)
    return _assemble_labs(raw)


def extract_labs_single_pass(image_bytes: bytes) -> dict:
    """One MedGemma pass: image -> {meta, results} directly, skipping the verbatim
    transcript. Roughly halves generated tokens vs transcribe+route, so it's much
    faster; the trade is no transcript for the side-by-side (the source photo is
    still shown). Best on clean printed reports."""
    raw = _medgemma_run(
        image_bytes, _LAB_EXTRACT_IMAGE_HEAD, prefill=EXTRACT_PREFILL, **EXTRACT_GEN)
    out = _assemble_labs(raw)
    return {"text": "", "meta": out["meta"], "results": out["results"]}


def extract_labs(image_bytes: bytes) -> dict:
    """Lab report: image -> {text, meta, results}.

    Default is two stages: transcribe (Qwen) then route the transcript into a
    header + analyte table (MedGemma). With LAB_SINGLE_PASS=1 (MedGemma extractor
    only) it's one image->{meta, results} pass — far fewer tokens, much faster on
    clean printed reports, at the cost of the transcript. Unparseable output
    degrades to empty results so the UI can still show the source photo.
    """
    import time as _time

    _t0 = _time.time()
    if os.getenv("LAB_SINGLE_PASS") == "1" and EXTRACT_PROVIDER == "medgemma":
        out = extract_labs_single_pass(image_bytes)
        print(
            f"[timing] lab single-pass total={_time.time() - _t0:.1f}s "
            f"n_results={len(out['results'])}",
            flush=True,
        )
        return out
    transcript = transcribe(image_bytes)
    _t1 = _time.time()
    out = extract_labs_from_text(transcript)
    _t2 = _time.time()
    print(
        f"[timing] lab transcribe={_t1 - _t0:.1f}s extract={_t2 - _t1:.1f}s "
        f"total={_t2 - _t0:.1f}s chars={len(transcript)}",
        flush=True,
    )
    return {"text": transcript, "meta": out["meta"], "results": out["results"]}


# ---------------------------------------------------------------- vLLM (local, batch)
# Prompt strings for vLLM are built with the HF processor (cheap: tokenizer +
# image-processor config, no model weights) and handed to the engine as
# {"prompt": str, "multi_modal_data": {"image": pil}}. The model itself is served
# by vllm_engine, which keeps one engine resident and swaps on model change.
_qwen_proc_only = None
_mg_proc_only = None


def _proc_only(model_id: str, cache_attr: str):
    from transformers import AutoProcessor

    globals()[cache_attr] = globals().get(cache_attr) or AutoProcessor.from_pretrained(model_id)
    return globals()[cache_attr]


def _qwen_transcribe_batch_vllm(image_list: list[bytes]) -> list[str]:
    """Stage 1 over a batch on vLLM: one engine, continuous-batched generation."""
    import vllm_engine
    from vllm import SamplingParams

    proc = _proc_only(_QWEN_ID, "_qwen_proc_only")
    build: dict = {
        "gpu_memory_utilization": _VLLM_GPU_UTIL,
        "max_model_len": _VLLM_MAX_MODEL_LEN,
        "limit_mm_per_prompt": {"image": 1},
        "dtype": "bfloat16",
    }
    if _QWEN_QUANT:
        build["quantization"] = "bitsandbytes"
    llm = vllm_engine.get(_QWEN_ID, **build)
    # enable_thinking=False injected into the template, same as the transformers
    # path — Qwen3.6 otherwise spends the whole budget "thinking" and never
    # reaches the transcript. repetition_penalty is a light nudge (verbatim OCR).
    sp = SamplingParams(temperature=0.0, max_tokens=_QWEN_MAX_NEW_TOKENS,
                        repetition_penalty=QWEN_TRANSCRIBE_GEN["repetition_penalty"])
    reqs = []
    for b in image_list:
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": TRANSCRIBE_PROMPT}]}]
        prompt = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        reqs.append({"prompt": prompt, "multi_modal_data": {"image": _qwen_image(b)}})
    outs = llm.generate(reqs, sp)
    texts = []
    for o in outs:
        t = o.outputs[0].text.strip()
        if "</think>" in t:  # defensive, mirrors _qwen_transcribe
            t = t.rsplit("</think>", 1)[-1].strip()
        texts.append(t)
    return texts


def _medgemma_fields_batch_vllm(transcripts: list[str]) -> list[dict]:
    """Stage 2 over a batch on vLLM, with JSON-schema-constrained decoding. The
    schema guarantees valid JSON matching our keys, so the prefill hack, the high
    token ceiling's runaway risk, and the regex-salvage path all fall away."""
    import vllm_engine
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    proc = _proc_only(_MODEL_ID, "_mg_proc_only")
    # get() frees the Qwen engine and builds MedGemma — the two never co-reside.
    llm = vllm_engine.get(_MODEL_ID, gpu_memory_utilization=_VLLM_GPU_UTIL,
                          max_model_len=_VLLM_MAX_MODEL_LEN, dtype="bfloat16")
    schema = {
        "type": "object",
        "properties": {k: {"type": "string"} for k in _FIELD_KEYS},
        "required": list(_FIELD_KEYS),
        "additionalProperties": False,
    }
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS,
                        repetition_penalty=EXTRACT_GEN["repetition_penalty"],
                        structured_outputs=StructuredOutputsParams(json=schema))
    # Skip empty transcripts (nothing to route) but keep positions aligned.
    idx = [i for i, t in enumerate(transcripts) if t.strip()]
    prompts = []
    for i in idx:
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": _extract_from_text_prompt(transcripts[i])}]}]
        prompts.append(proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False))
    outs = llm.generate(prompts, sp) if prompts else []
    results = [{k: "" for k in _FIELD_KEYS} for _ in transcripts]
    for i, o in zip(idx, outs):
        results[i] = _assemble_fields(o.outputs[0].text)
    return results


def extract_batch(image_list: list[bytes]) -> list[dict]:
    """extract() over several images, in two passes: transcribe ALL of them, free
    the vision model, then extract fields for ALL of them. This keeps the two
    models from co-residing on a single 24GB card (which OOMs) and is why the batch
    path exists separately from per-image extract().

    With OCR_ENGINE=vllm (and the default qwen->medgemma combo) each pass is a
    single continuous-batched vLLM call and stage 2 uses JSON-schema-constrained
    decoding; otherwise it's the transformers path (per-image within each stage —
    batched generation there isn't worth the padding complexity on this hardware).
    """
    if OCR_ENGINE == "vllm" and TRANSCRIBE_PROVIDER == "qwen" and EXTRACT_PROVIDER == "medgemma":
        transcripts = _qwen_transcribe_batch_vllm(image_list)
        fields = _medgemma_fields_batch_vllm(transcripts)  # swaps engine, frees Qwen
        return [{"text": t, "fields": f} for t, f in zip(transcripts, fields)]
    transcripts = [transcribe(b) for b in image_list]
    if TRANSCRIBE_PROVIDER == "qwen" and EXTRACT_PROVIDER == "medgemma":
        _free_qwen()  # release ~16GB before MedGemma loads
    return [{"text": t, "fields": extract_fields_from_text(t)} for t in transcripts]
