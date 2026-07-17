"""In-process vLLM engines for the two local stages, with single-active-engine
swapping so the Qwen reader (stage 1) and MedGemma router (stage 2) never
co-reside on one card.

Why a swap and not two persistent engines: on a 24GB L4 both models plus their
KV-cache pools don't fit. So we keep exactly ONE vLLM engine alive at a time —
`get(model_id)` tears down the previous engine before building the requested one.
That maps cleanly onto the batch extractor's two passes (transcribe ALL, then
route ALL): the cost of the swap is paid once per batch, not per image.

Only used when OCR_ENGINE=vllm (default is transformers). vLLM is a Linux/CUDA
dependency, imported lazily here so macOS/dev installs are unaffected.

The engine is a batch-throughput tool: `generate` takes the whole list and vLLM
schedules it with continuous batching. It is NOT meant for the latency-sensitive
single-scan live path (rebuilding an engine per request would dwarf inference) —
that path stays on transformers. For low-latency single-scan serving, run one
persistent vLLM server per model on its own GPU instead.
"""

import gc
import threading

_engine = None
_engine_id = None
_lock = threading.Lock()


def _free_locked() -> None:
    """Tear down the live engine and reclaim its VRAM. Caller holds _lock."""
    global _engine, _engine_id
    if _engine is None:
        return
    import torch
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    del _engine
    _engine = None
    _engine_id = None
    # vLLM leaves the model-parallel + distributed state standing; without
    # tearing it down the next LLM() build trips over the stale process group.
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - no CUDA / already gone
        pass


def get(model_id: str, **build_kwargs):
    """Return a vLLM `LLM` for `model_id`, building it (and freeing whatever was
    resident) only when the requested model differs from the live one. Repeated
    calls for the same model reuse the engine."""
    global _engine, _engine_id
    with _lock:
        if _engine_id != model_id:
            _free_locked()
            from vllm import LLM

            _engine = LLM(model=model_id, **build_kwargs)
            _engine_id = model_id
        return _engine


def free() -> None:
    """Release the live engine (if any). Safe to call when nothing is loaded."""
    with _lock:
        _free_locked()
