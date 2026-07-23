#!/usr/bin/env bash
# Vendors the ZA lab FHIR generator into a checkout of Google Health's
# medical-data-toolkit and registers it for the LABORATORY_REPORT document type.
#
# This does NOT run the toolkit — it prepares the checkout. Actually serving it
# still needs the deferred VM pieces: build the LOINC KB and run `vllm serve`
# (see README.md). Idempotent; safe to re-run.
#
# Usage:
#   ./install_into_toolkit.sh [/path/to/medical-data-toolkit]
# With no arg it clones the repo next to this project.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLKIT="${1:-$HERE/../../.medical-data-toolkit}"
REPO="https://github.com/Google-Health/medical-data-toolkit"

if [ ! -d "$TOOLKIT/.git" ]; then
  echo "==> cloning $REPO -> $TOOLKIT"
  git clone --depth 1 "$REPO" "$TOOLKIT"
else
  echo "==> using existing checkout at $TOOLKIT"
fi

DEST="$TOOLKIT/src/document_to_fhir/core/fhir/za"
mkdir -p "$DEST"
touch "$DEST/__init__.py"
cp "$HERE/za_lab_report_fhir_generator.py" "$DEST/za_lab_report_fhir_generator.py"
echo "==> copied generator -> ${DEST#$TOOLKIT/}/za_lab_report_fhir_generator.py"

cp "$HERE/config.za.yaml" "$TOOLKIT/src/config.yaml"
echo "==> installed ZA config -> src/config.yaml (edit <vllm-host> before serving)"

# Register the generator: flip fhir_generator_class for LABORATORY_REPORT.
python3 - "$TOOLKIT" <<'PY'
import re, sys, pathlib
rs = pathlib.Path(sys.argv[1]) / "src" / "rest_server.py"
src = rs.read_text()

imp = ("from src.document_to_fhir.core.fhir.za import "
       "za_lab_report_fhir_generator\n")
if imp not in src:
    anchor = ("from src.document_to_fhir.core.fhir.abdm import "
              "abdm_lab_report_fhir_generator\n")
    if anchor not in src:
        sys.exit("!! could not find the abdm generator import to anchor to; "
                 "register the generator manually (see README).")
    src = src.replace(anchor, anchor + imp)

# swap the class used in _DOCUMENT_TYPE_MAPPING
src2 = src.replace(
    "abdm_lab_report_fhir_generator.AbdmLabReportFhirGenerator",
    "za_lab_report_fhir_generator.ZaLabReportFhirGenerator",
)
# keep the import line pointing at the real abdm module intact
src2 = src2.replace(
    "from src.document_to_fhir.core.fhir.abdm import za_lab_report_fhir_generator",
    "from src.document_to_fhir.core.fhir.abdm import abdm_lab_report_fhir_generator",
)
rs.write_text(src2)
print("==> registered ZaLabReportFhirGenerator for LABORATORY_REPORT")
PY

cat <<EOF

Done. Next (on the VM, the deferred pieces):
  1. Build the LOINC knowledge base -> CSVs at /data
     (see \$TOOLKIT/src/document_to_fhir/core/medical_coding/loinc/README.md)
  2. Serve the models:  vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
                        vllm serve google/medgemma-1.5-4b-it --port 8001
  3. Edit src/config.yaml: replace <vllm-host> with the server address.
  4. Build & run:  docker build -t mdt . && docker run -p 8080:8080 -v /data:/data mdt
  5. Test:         curl -X POST --data-binary @eval/images/20231129_120507340_iOS.jpg \\
                        http://127.0.0.1:8080/document_to_fhir
EOF
