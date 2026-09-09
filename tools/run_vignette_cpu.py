"""Run real vignette photos through De-paperfy's configured OCR pipeline."""

import argparse
import json
import time
from pathlib import Path

import ocr


DEFAULT_IMAGES = [
    Path(r"E:\De-paperfy\vignettes david\IMG_2531.jpeg"),
    Path(r"E:\De-paperfy\vignettes sergej\vignettes sergej\IMG_5999.jpg"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs") / "cpu-gguf")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image in args.images:
        started = time.perf_counter()
        extraction = ocr.extract(image.read_bytes())
        result = {
            "source": str(image),
            "provider": ocr.PROVIDER,
            "transcribe_provider": ocr.TRANSCRIBE_PROVIDER,
            "extract_provider": ocr.EXTRACT_PROVIDER,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "extraction": extraction,
        }
        destination = args.output_dir / f"{image.stem}.json"
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"{image.name}: {result['elapsed_seconds']}s -> {destination}", flush=True)


if __name__ == "__main__":
    main()
