# South-African FHIR generator for the Medical Data Toolkit

**Goal.** Reuse Google Health's `medical-data-toolkit` (classify → extract → LOINC
code) but emit **base FHIR R4** shaped for a South-African deployment instead of
the shipped **ABDM** (India) profile.

**Scope of the change is one layer.** The toolkit already separates the
India-specific bits from everything else:

| Layer | India-specific? | What we do |
|---|---|---|
| Classification | No | keep |
| Extraction (`LabReport` schema) | No — generic `common/schema/medical_documents.py` | keep |
| LOINC coding | No — LOINC is an international standard, used by NHLS too | keep |
| **FHIR generation** | **Yes** — `abdm/abdm_lab_report_fhir_generator.py` (ABHA ids, ABDM profiles) | **swap for a ZA generator** |

So we neither touch extraction nor the input model. We add **one `IFhirGenerator`
implementation** and register it in `config.yaml`.

> Note: the demo images are already South African (NHLS / George Laboratory,
> Western Cape). Only the *output profile* needs localising, not the input.

---

## The interface we implement

From `core/fhir/fhir_generator.py`:

```python
class IFhirGenerator(abc.ABC):
    def __init__(self, version: str = "", fhir_profile: str = ""): ...
    @abstractmethod
    def generate_fhir(self, medical_document: medical_documents.MedicalDocument
                      ) -> fhir_pb2.Bundle: ...
```

The input `medical_documents.LabReport` (generic) gives us, per the toolkit schema:

- `patient`: `name`, `identifiers.mr` (MRN), `dob`, `gender`
- `service_provider`: `Organization` — `name`, `address`, `contact`
- `practitioner`: `name`, `qualification`
- `sample_collection_time`
- `lab_tests[]`: `name`, `core_analyte`, `result`, `unit`, `specimen`, `method`,
  `panel_name`, `reference_range[] {low, high, text}`, **`loinc_code`**,
  **`loinc_common_name`** (the last two filled by the LOINC stage)

---

## Field → FHIR R4 mapping (ZA target)

| Extracted field | FHIR R4 element | ZA / standards decision |
|---|---|---|
| `patient.name` | `Patient.name[0].text` | — |
| `patient.identifiers.mr` | `Patient.identifier[]` | **system = a ZA MRN namespace** (see open questions); drop ABHA |
| `patient.dob` / `patient.gender` | `Patient.birthDate` / `Patient.gender` | gender normalised to `male\|female\|other\|unknown` |
| `service_provider` | `Organization` | `identifier.system` = NHLS/practice-number namespace |
| `practitioner` | `Practitioner` (+ `qualification`) | drop India registration ids |
| `sample_collection_time` | `Observation.effectiveDateTime`, `DiagnosticReport.effective` | — |
| `lab_tests[]` (whole report) | one `DiagnosticReport` | `category` = LAB (`http://terminology.hl7.org/CodeSystem/v2-0074`) |
| `lab_test.loinc_code` / `loinc_common_name` | `Observation.code.coding[]` | **`system = http://loinc.org`** (unchanged — global) |
| `lab_test.name` | `Observation.code.text` | keeps the report's own wording |
| `lab_test.result` | `Observation.valueQuantity.value` (numeric) or `valueString` | parse leading number; non-numeric → `valueString` |
| `lab_test.unit` | `valueQuantity.unit` + `.code` | **`system = http://unitsofmeasure.org` (UCUM)** |
| `reference_range.low/high` | `Observation.referenceRange.low/high` | numeric where parseable, else `.text` |
| _derived_ abnormal flag | `Observation.interpretation` | **`http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation`** (H/L/HH/LL/A) — see below |
| `panel_name` (+ `panel_loinc_code`) | panel `Observation` with `hasMember[]` | optional; mirrors ABDM panel grouping |

Everything in **bold** is an international code system (LOINC, UCUM, HL7 v3
interpretation) that SA uses as-is — the only genuinely local choices are the
**identifier `system` URIs** and whether we stamp a ZA `meta.profile`.

### Deriving the abnormal flag (ties into the demo's highlighting)

The generic `LabTest` schema has **no explicit H/L field** — the cue lives in the
`result` string and/or `reference_range.text`. The ZA generator computes it
deterministically and emits standard `Observation.interpretation`:

```python
def interpretation(result: str, rr) -> str | None:
    v = _num(result)                       # leading float, else None
    if v is None or rr is None: return None
    lo, hi = _num(rr.low), _num(rr.high)
    if hi is not None and v > hi: return "H"
    if lo is not None and v < lo: return "L"
    return "N"
# -> map to v3-ObservationInterpretation codes H / L / N (extend: HH/LL = critical)
```

This is the same flag your UI already highlights — now expressed as a coded,
EHR-ingestable element.

---

## Target bundle (for demo image `#11`, `20231129_120507340`)

A **`collection` Bundle** (portable — HAPI/OpenHIE/most EHR endpoints ingest it
directly; no Composition/document profile required). Trimmed to two tests:

```json
{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    { "fullUrl": "urn:uuid:pat-1", "resource": {
        "resourceType": "Patient",
        "name": [{ "text": "Minentle Hlaluminami Ramncwana" }],
        "gender": "female",
        "birthDate": "2023-09-21",
        "identifier": [{ "system": "http://health.gov.za/fhir/sid/mrn", "value": "MRN161282266" }]
    }},
    { "fullUrl": "urn:uuid:org-1", "resource": {
        "resourceType": "Organization",
        "name": "George Laboratory (NHLS)",
        "address": [{ "city": "George", "state": "Western Cape", "country": "ZA" }]
    }},
    { "fullUrl": "urn:uuid:obs-bili", "resource": {
        "resourceType": "Observation",
        "status": "final",
        "category": [{ "coding": [{ "system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory" }]}],
        "code": {
          "coding": [{ "system": "http://loinc.org", "code": "1975-2", "display": "Bilirubin.total [Mass/Vol] in Serum or Plasma" }],
          "text": "Total bilirubin"
        },
        "subject": { "reference": "urn:uuid:pat-1" },
        "effectiveDateTime": "2023-11-29T11:18:00Z",
        "valueQuantity": { "value": 34, "unit": "umol/L", "system": "http://unitsofmeasure.org", "code": "umol/L" },
        "interpretation": [{ "coding": [{ "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "H", "display": "High" }]}],
        "referenceRange": [{ "low": { "value": 5, "unit": "umol/L" }, "high": { "value": 21, "unit": "umol/L" } }]
    }},
    { "fullUrl": "urn:uuid:obs-alt", "resource": {
        "resourceType": "Observation",
        "status": "final",
        "code": { "coding": [{ "system": "http://loinc.org", "code": "1742-6", "display": "Alanine aminotransferase [Enzymatic activity/Vol]" }], "text": "Alanine transaminase (ALT)" },
        "subject": { "reference": "urn:uuid:pat-1" },
        "valueQuantity": { "value": 64, "unit": "U/L", "system": "http://unitsofmeasure.org", "code": "U/L" },
        "interpretation": [{ "coding": [{ "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation", "code": "H", "display": "High" }]}],
        "referenceRange": [{ "low": { "value": 3 }, "high": { "value": 30 } }]
    }},
    { "fullUrl": "urn:uuid:dr-1", "resource": {
        "resourceType": "DiagnosticReport",
        "status": "final",
        "category": [{ "coding": [{ "system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "LAB" }]}],
        "code": { "text": "Chemical Pathology — Liver function tests" },
        "subject": { "reference": "urn:uuid:pat-1" },
        "performer": [{ "reference": "urn:uuid:org-1" }],
        "effectiveDateTime": "2023-11-29T11:18:00Z",
        "result": [{ "reference": "urn:uuid:obs-bili" }, { "reference": "urn:uuid:obs-alt" }]
    }}
  ]
}
```

(LOINC codes above are illustrative — the toolkit's coder assigns the real ones.)

---

## Generator skeleton

> Full, unit-tested implementation lives in
> `integrations/medical_toolkit/za_lab_report_fhir_generator.py`. The outline below
> explains the shape.

Two implementation routes; **route B is recommended** for us.

**Route A — in-toolkit, proto-native.** Mirror `AbdmLabReportFhirGenerator` but
build base-R4 resources (no ABDM converter). Most faithful, but fights the
verbose `google.fhir` proto builders.

**Route B — dict builder + a thin proto wrap.** Build valid FHIR **JSON dicts**
(portable, trivially testable) and, only if the toolkit's proto return type is
needed, convert once with `google.fhir.r4.json_format`. This keeps the mapping
logic readable and lets De-paperfy reuse the exact same builder in the sidecar.

```python
# za_lab_report_fhir_generator.py  (Route B)
import uuid
from src.document_to_fhir.common.schema import medical_documents
from src.document_to_fhir.core.fhir import fhir_generator

LOINC = "http://loinc.org"
UCUM  = "http://unitsofmeasure.org"
INTERP = "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"
MRN_SYS = "http://health.gov.za/fhir/sid/mrn"        # <-- ZA choice (see open questions)

def build_za_lab_bundle(lab: medical_documents.LabReport) -> dict:
    pid = f"urn:uuid:{uuid.uuid4()}"
    entries, result_refs = [], []

    patient = {"resourceType": "Patient", "name": [{"text": lab.patient.name}]}
    if lab.patient.gender: patient["gender"] = _gender(lab.patient.gender)
    if lab.patient.dob:    patient["birthDate"] = lab.patient.dob.isoformat()
    mrn = getattr(lab.patient.identifiers, "mr", None) if lab.patient.identifiers else None
    if mrn: patient["identifier"] = [{"system": MRN_SYS, "value": mrn}]
    entries.append((pid, patient))

    org_ref = None
    if lab.service_provider:
        org_ref = f"urn:uuid:{uuid.uuid4()}"
        entries.append((org_ref, {"resourceType": "Organization", "name": lab.service_provider.name}))

    eff = lab.sample_collection_time.isoformat() if lab.sample_collection_time else None
    for t in lab.lab_tests:
        oref = f"urn:uuid:{uuid.uuid4()}"; result_refs.append(oref)
        obs = {
            "resourceType": "Observation", "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                       "code": "laboratory"}]}],
            "code": _code(t), "subject": {"reference": pid},
        }
        if eff: obs["effectiveDateTime"] = eff
        obs.update(_value(t))                       # valueQuantity | valueString
        interp = _interpretation(t)
        if interp: obs["interpretation"] = [{"coding": [{"system": INTERP, **interp}]}]
        rr = _reference_range(t)
        if rr: obs["referenceRange"] = rr
        entries.append((oref, obs))

    dr = {"resourceType": "DiagnosticReport", "status": "final",
          "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "LAB"}]}],
          "code": {"text": _report_title(lab)}, "subject": {"reference": pid},
          "result": [{"reference": r} for r in result_refs]}
    if org_ref: dr["performer"] = [{"reference": org_ref}]
    if eff: dr["effectiveDateTime"] = eff
    entries.append((f"urn:uuid:{uuid.uuid4()}", dr))

    return {"resourceType": "Bundle", "type": "collection",
            "entry": [{"fullUrl": u, "resource": r} for u, r in entries]}

def _code(t):
    coding = []
    if t.loinc_code:
        coding.append({"system": LOINC, "code": t.loinc_code, "display": t.loinc_common_name or t.name})
    return {"coding": coding, "text": t.name} if coding else {"text": t.name}

# _value / _interpretation / _reference_range / _num / _gender / _report_title: small pure helpers


class ZaLabReportFhirGenerator(fhir_generator.IFhirGenerator):
    """Base FHIR R4 (ZA) generator. Returns a proto Bundle to satisfy the toolkit."""
    def generate_fhir(self, medical_document):
        if not isinstance(medical_document, medical_documents.LabReport):
            raise TypeError("ZaLabReportFhirGenerator only processes LabReport documents.")
        bundle_dict = build_za_lab_bundle(medical_document)
        from google.fhir.r4 import json_format          # dict -> proto Bundle
        import json
        return json_format.json_fhir_string_to_proto(
            json.dumps(bundle_dict), fhir_pb2.Bundle)
```

For the **sidecar** architecture, De-paperfy can call `build_za_lab_bundle(...)`
directly on the toolkit's standardized `LabReport` and never touch protos at all.

---

## Wiring (`config.yaml`)

Register the ZA generator for the `LAB_REPORT` document type in the standardizer
map (replacing the ABDM entry), alongside the on-prem LLM clients:

```yaml
extractor_llm_client:  { type: LiteLLMClient, parameters: { model: hosted_vllm/Qwen/Qwen3-VL-8B-Instruct, api_base: http://<vm>:8000/v1, api_key_env: DUMMY, supports_pdf: false } }
classifier_llm_client: { type: LiteLLMClient, parameters: { model: hosted_vllm/google/medgemma-1.5-4b-it, api_base: http://<vm>:8000/v1, api_key_env: DUMMY } }
# standardizer_map: LAB_REPORT -> ZaLabReportFhirGenerator   (was AbdmLabReportFhirGenerator)
```

---

## Chosen defaults (all configurable via `ZaFhirConfig`)

Implemented in `integrations/medical_toolkit/za_lab_report_fhir_generator.py` (a
dependency-light, unit-tested reference module — runs without the toolkit). Every
choice below is a field on `ZaFhirConfig`; override to retarget.

| # | Decision | Default | Config field |
|---|---|---|---|
| 1 | Patient identifier | facility MRN namespace; a **13-digit numeric MRN auto-emits as SA National ID** | `mrn_system`, `sa_id_system`, `detect_sa_id` |
| 2 | Profile target | **plain FHIR R4** (no `meta.profile`) — portable; set a canonical to stamp a named ZA IG | `fhir_profile` |
| 3 | Org / practitioner id | NHLS facility-code / HPCSA-registration namespaces | `org_system`, `practitioner_system` |
| 4 | Critical flags | **`critical` → `HH`/`LL`** (by side of range); `high/low → H/L`; normals omitted | `interpretation_codes`, `emit_normal_interpretation` |

The identifier `system` URIs are **placeholder ZA namespaces** — swap them for the
receiving system's canonical URIs when a target EHR/HIE is chosen. LOINC / UCUM /
HL7-interpretation systems are international and are deliberately **not** config knobs.

Example override:

```python
from za_lab_report_fhir_generator import ZaFhirConfig, build_za_lab_bundle
cfg = ZaFhirConfig(
    mrn_system="https://<province>.health.gov.za/fhir/sid/mrn",
    fhir_profile="https://fhir.health.gov.za/StructureDefinition/za-lab-report",  # opt into a ZA IG
)
bundle = build_za_lab_bundle(lab_report, cfg)
```

### Still genuinely open (deployment-time, not code)
- The **canonical identifier URIs** above become real once a receiving system is
  picked (facility EHR, an OpenHIE/HAPI endpoint, or a national exchange).
- Whether to author/adopt a **named ZA FHIR IG** vs. staying on base R4.
