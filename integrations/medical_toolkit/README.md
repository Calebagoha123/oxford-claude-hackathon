# medical-data-toolkit integration (South Africa)

Two ways De-paperfy produces standards output from a lab report. See the design +
field mapping in [`docs/medical-toolkit-za-integration.md`](../../docs/medical-toolkit-za-integration.md).

## 1. Sidecar (live now, no toolkit)

`za_lab_report_fhir_generator.build_bundle_from_depaperfy(meta, results)` turns a
De-paperfy scan straight into a base **FHIR R4** bundle (ZA profile). It's wired
into the app:

- **Endpoint:** `GET /api/scan/session/{id}/fhir.json` → `application/fhir+json`
- **UI:** a **"FHIR bundle ↓"** button on the results view.

Caveat: the sidecar has no LOINC codes (Observations carry `code.text` only) — LOINC
comes from the full toolkit pipeline below.

## 2. Full toolkit (adds LOINC coding, runs on-prem)

The toolkit does classify → extract → **LOINC** → FHIR, driven by our on-prem
MedGemma/Qwen via LiteLLM→vLLM. We swap its India (ABDM) FHIR generator for the ZA
one.

```bash
./install_into_toolkit.sh              # clones the toolkit + registers the ZA generator
# then, on the VM (deferred pieces):
#   - build the LOINC KB (CSVs -> /data)   [toolkit's loinc/README.md]
#   - vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
#   - vllm serve google/medgemma-1.5-4b-it --port 8001
#   - edit src/config.yaml: set <vllm-host>
#   - docker build -t mdt . && docker run -p 8080:8080 -v /data:/data mdt
```

### What the installer changes (for a manual patch)

In the toolkit's `src/rest_server.py`, `_DOCUMENT_TYPE_MAPPING['LABORATORY_REPORT']`
sets `fhir_generator_class`. Point it at ours (instantiated with no args, defaults
from `ZaFhirConfig`):

```diff
+ from src.document_to_fhir.core.fhir.za import za_lab_report_fhir_generator
  ...
  'LABORATORY_REPORT': {
      'extractor_class': lab_report_extractor.LabReportExtractor,
      'schema': abdm_medical_documents.AbdmLabReport,   # extraction schema unchanged
-     'fhir_generator_class': abdm_lab_report_fhir_generator.AbdmLabReportFhirGenerator,
+     'fhir_generator_class': za_lab_report_fhir_generator.ZaLabReportFhirGenerator,
  }
```

The extraction schema stays `AbdmLabReport`; our generator duck-types it, so only
the FHIR-emit layer changes. `config.za.yaml` (installed as `src/config.yaml`)
carries the on-prem LiteLLM→vLLM client config and LOINC KB paths.

## Files

| File | Purpose |
|---|---|
| `za_lab_report_fhir_generator.py` | ZA base-R4 generator + De-paperfy adapter (stdlib-only, unit-tested) |
| `config.za.yaml` | toolkit config: on-prem LiteLLM→vLLM clients, LAB_REPORT, LOINC KB paths |
| `install_into_toolkit.sh` | clone toolkit + vendor generator + register it |
