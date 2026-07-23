"""South-African base-FHIR-R4 generator for the Google Health medical-data-toolkit.

The toolkit classifies + extracts + LOINC-codes a lab report into a generic
`LabReport`; only its *FHIR generation* layer is India-specific (ABDM). This module
is the drop-in replacement that emits **base FHIR R4** shaped for a South-African
deployment instead.

Two ways to use it:

  1. In-toolkit:  register `ZaLabReportFhirGenerator` for the LAB_REPORT document
     type (in place of AbdmLabReportFhirGenerator). It returns a proto Bundle to
     satisfy `IFhirGenerator`.
  2. Sidecar:     call `build_za_lab_bundle(lab_report, cfg)` directly to get a
     plain FHIR-JSON dict — no toolkit / proto dependency at all. De-paperfy uses
     this path (it already has structured, flagged results to feed in).

Everything South-African-specific is on `ZaFhirConfig` with sensible defaults and
is overridable. LOINC / UCUM / HL7 interpretation code systems are international
and are NOT configurable knobs — SA uses them as-is.

This module is intentionally dependency-light (stdlib only) and duck-types its
input, so it runs and is testable without the toolkit installed. It accepts either
the toolkit's pydantic `LabReport` or any object with the same attribute shape
(e.g. De-paperfy's own extracted results).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

# ---- fixed, international code systems (not localisation knobs) ----
LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"
INTERP_SYS = "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"
OBS_CATEGORY_SYS = "http://terminology.hl7.org/CodeSystem/observation-category"
DR_CATEGORY_SYS = "http://terminology.hl7.org/CodeSystem/v2-0074"


@dataclass
class ZaFhirConfig:
    """SA defaults for the bits that genuinely vary by deployment. Override any
    field to point at a different identifier authority or FHIR profile."""

    # Identifier authorities (open question 1 & 3). Defaults are placeholder ZA
    # namespaces — swap for the receiving system's canonical URIs.
    mrn_system: str = "http://health.gov.za/fhir/sid/mrn"          # facility / hospital MRN
    sa_id_system: str = "http://health.gov.za/fhir/sid/sa-id"      # SA National ID (13 digits)
    org_system: str = "http://health.gov.za/fhir/sid/nhls"         # NHLS facility code
    practitioner_system: str = "http://hpcsa.co.za/fhir/sid/registration"  # HPCSA number
    detect_sa_id: bool = True   # a 13-digit all-numeric MRN is emitted as an SA ID

    # Profile target (open question 2). Default: plain R4, most portable. Set a
    # canonical to stamp `meta.profile` on the clinical resources for a named ZA IG.
    fhir_profile: str = ""

    default_country: str = "ZA"
    bundle_type: str = "collection"     # portable; HAPI/OpenHIE ingest directly
    emit_normal_interpretation: bool = False   # usually only abnormals get a flag

    # Abnormal-flag -> v3 ObservationInterpretation (open question 4). "critical"
    # resolves to HH/LL by which side of the range it falls on.
    interpretation_codes: dict = field(default_factory=lambda: {
        "high": ("H", "High"),
        "low": ("L", "Low"),
        "critical_high": ("HH", "Critical high"),
        "critical_low": ("LL", "Critical low"),
        "abnormal": ("A", "Abnormal"),
        "normal": ("N", "Normal"),
    })


DEFAULT_ZA_CONFIG = ZaFhirConfig()


# ---------------------------------------------------------------- helpers
def _num(s: Any) -> Optional[float]:
    """Leading number from a value/range string, else None ("13.2 H" -> 13.2)."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group()) if m else None


def _gender(g: Any) -> Optional[str]:
    s = str(g or "").strip().lower()
    if s in ("m", "male"):
        return "male"
    if s in ("f", "female"):
        return "female"
    if s in ("o", "other"):
        return "other"
    return "unknown" if s else None


def _iso(dt: Any) -> Optional[str]:
    if dt is None:
        return None
    iso = getattr(dt, "isoformat", None)
    return iso() if callable(iso) else str(dt)


def _meta(cfg: ZaFhirConfig) -> dict:
    return {"meta": {"profile": [cfg.fhir_profile]}} if cfg.fhir_profile else {}


def _patient_identifier(mrn: str, cfg: ZaFhirConfig) -> dict:
    if cfg.detect_sa_id and re.fullmatch(r"\d{13}", mrn.strip()):
        return {"system": cfg.sa_id_system, "value": mrn.strip()}
    return {"system": cfg.mrn_system, "value": mrn}


def _code(test: Any) -> dict:
    """Observation.code: LOINC coding (from the toolkit's coder) + the report's text."""
    loinc = getattr(test, "loinc_code", None)
    name = getattr(test, "name", None) or getattr(test, "test", None) or ""
    if loinc:
        display = getattr(test, "loinc_common_name", None) or name
        return {"coding": [{"system": LOINC, "code": loinc, "display": display}], "text": name}
    return {"text": name}


def _value(test: Any) -> dict:
    """valueQuantity when the result is numeric, else valueString."""
    result = getattr(test, "result", None)
    if result is None:
        result = getattr(test, "value", None)
    unit = getattr(test, "unit", None) or ""
    v = _num(result)
    if v is not None:
        q: dict = {"value": v}
        if unit:
            q.update({"unit": unit, "system": UCUM, "code": unit})
        return {"valueQuantity": q}
    return {"valueString": str(result or "").strip()}


def _reference_ranges(test: Any) -> list:
    """Map the toolkit's ReferenceRange[] (or a single string) to FHIR referenceRange[]."""
    rrs = getattr(test, "reference_range", None)
    if rrs is None:
        return []
    if isinstance(rrs, str):  # De-paperfy stores a single string like "13.0-17.0"
        rrs = [_parse_range_string(rrs)]
    out = []
    unit = getattr(test, "unit", None) or ""
    for rr in rrs:
        low = getattr(rr, "low", None) if not isinstance(rr, dict) else rr.get("low")
        high = getattr(rr, "high", None) if not isinstance(rr, dict) else rr.get("high")
        text = getattr(rr, "text", None) if not isinstance(rr, dict) else rr.get("text")
        entry: dict = {}
        lo, hi = _num(low), _num(high)
        if lo is not None:
            entry["low"] = {"value": lo, **({"unit": unit} if unit else {})}
        if hi is not None:
            entry["high"] = {"value": hi, **({"unit": unit} if unit else {})}
        if text:
            entry["text"] = str(text)
        if entry:
            out.append(entry)
    return out


class _R:  # tiny holder so a "13.0-17.0" string reuses the same reference-range path
    def __init__(self, low, high, text):
        self.low, self.high, self.text = low, high, text


def _parse_range_string(s: str) -> _R:
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*$", str(s))
    if m:
        return _R(m.group(1), m.group(2), None)
    return _R(None, None, s)


def _interpretation(test: Any, cfg: ZaFhirConfig) -> Optional[dict]:
    """Coded Observation.interpretation. Honours an explicit `flag` when present
    (De-paperfy supplies one); otherwise derives H/L numerically from the range.
    The generic toolkit LabTest has no flag field, so derivation is the default."""
    codes = cfg.interpretation_codes
    v = _num(getattr(test, "result", None) if getattr(test, "result", None) is not None
             else getattr(test, "value", None))

    flag = str(getattr(test, "flag", "") or "").strip().lower()
    if flag in ("high", "low", "abnormal"):
        key = flag
    elif flag == "critical":
        # critical high vs low depends on which side of the range it sits
        lo, hi = _range_bounds(test)
        key = "critical_low" if (v is not None and lo is not None and v < lo) else "critical_high"
    else:
        # derive from value vs reference range
        lo, hi = _range_bounds(test)
        if v is None or (lo is None and hi is None):
            return None
        if hi is not None and v > hi:
            key = "high"
        elif lo is not None and v < lo:
            key = "low"
        else:
            key = "normal"

    if key == "normal" and not cfg.emit_normal_interpretation:
        return None
    code, display = codes.get(key, codes["abnormal"])
    return {"coding": [{"system": INTERP_SYS, "code": code, "display": display}]}


def _range_bounds(test: Any):
    for rr in _reference_ranges(test):
        lo = rr.get("low", {}).get("value")
        hi = rr.get("high", {}).get("value")
        if lo is not None or hi is not None:
            return lo, hi
    return None, None


def _report_title(lab: Any) -> str:
    panels = []
    for t in getattr(lab, "lab_tests", None) or getattr(lab, "results", None) or []:
        p = getattr(t, "panel_name", None)
        if p and p not in panels:
            panels.append(p)
    return " · ".join(panels) if panels else "Laboratory report"


# ---------------------------------------------------------------- builder
def build_za_lab_bundle(lab: Any, cfg: ZaFhirConfig = DEFAULT_ZA_CONFIG) -> dict:
    """A base FHIR R4 `collection` Bundle (ZA-shaped) from a LabReport-like object.

    Accepts the toolkit's `LabReport` OR any duck-typed equivalent: `patient`
    (name/dob/gender/identifiers.mr), optional `service_provider`/`practitioner`,
    `sample_collection_time`, and `lab_tests` (or `results`) of test-like objects.
    """
    entries: list[tuple[str, dict]] = []
    patient = getattr(lab, "patient", None)
    if patient is None:
        raise ValueError("LabReport must have a patient.")

    # ---- Patient ----
    pid = f"urn:uuid:{uuid.uuid4()}"
    p: dict = {"resourceType": "Patient", **_meta(cfg),
               "name": [{"text": getattr(patient, "name", "") or "Unknown"}]}
    g = _gender(getattr(patient, "gender", None))
    if g:
        p["gender"] = g
    dob = _iso(getattr(patient, "dob", None))
    if dob:
        p["birthDate"] = dob
    idents = getattr(patient, "identifiers", None)
    mrn = getattr(idents, "mr", None) if idents else None
    if mrn:
        p["identifier"] = [_patient_identifier(str(mrn), cfg)]
    entries.append((pid, p))

    # ---- Organization (performer) ----
    org_ref = None
    org = getattr(lab, "service_provider", None)
    if org and getattr(org, "name", None):
        org_ref = f"urn:uuid:{uuid.uuid4()}"
        o: dict = {"resourceType": "Organization", **_meta(cfg), "name": org.name}
        addr = getattr(org, "address", None)
        if addr:
            o["address"] = [{k: v for k, v in {
                "city": getattr(addr, "city", None),
                "state": getattr(addr, "state", None),
                "country": getattr(addr, "country", None) or cfg.default_country,
            }.items() if v}]
        entries.append((org_ref, o))

    # ---- Practitioner ----
    prac_ref = None
    prac = getattr(lab, "practitioner", None)
    if prac and getattr(prac, "name", None):
        prac_ref = f"urn:uuid:{uuid.uuid4()}"
        entries.append((prac_ref, {"resourceType": "Practitioner", **_meta(cfg),
                                   "name": [{"text": prac.name}]}))

    eff = _iso(getattr(lab, "sample_collection_time", None))
    tests = getattr(lab, "lab_tests", None)
    if tests is None:
        tests = getattr(lab, "results", None) or []

    # ---- Observations ----
    result_refs = []
    for t in tests:
        oref = f"urn:uuid:{uuid.uuid4()}"
        result_refs.append(oref)
        obs: dict = {
            "resourceType": "Observation", **_meta(cfg), "status": "final",
            "category": [{"coding": [{"system": OBS_CATEGORY_SYS, "code": "laboratory"}]}],
            "code": _code(t),
            "subject": {"reference": pid},
        }
        if eff:
            obs["effectiveDateTime"] = eff
        obs.update(_value(t))
        interp = _interpretation(t, cfg)
        if interp:
            obs["interpretation"] = [interp]
        rrs = _reference_ranges(t)
        if rrs:
            obs["referenceRange"] = rrs
        entries.append((oref, obs))

    # ---- DiagnosticReport ----
    dr: dict = {
        "resourceType": "DiagnosticReport", **_meta(cfg), "status": "final",
        "category": [{"coding": [{"system": DR_CATEGORY_SYS, "code": "LAB"}]}],
        "code": {"text": _report_title(lab)},
        "subject": {"reference": pid},
        "result": [{"reference": r} for r in result_refs],
    }
    if org_ref:
        dr["performer"] = [{"reference": org_ref}]
    if prac_ref:
        dr.setdefault("performer", []).append({"reference": prac_ref})
    if eff:
        dr["effectiveDateTime"] = eff
    entries.append((f"urn:uuid:{uuid.uuid4()}", dr))

    return {"resourceType": "Bundle", "type": cfg.bundle_type,
            "entry": [{"fullUrl": u, "resource": r} for u, r in entries]}


# ---------------------------------------------------------------- De-paperfy adapter
def _gender_from_age_sex(age_sex: Any) -> Optional[str]:
    """Pull a gender out of De-paperfy's free-text 'age / sex' meta ("34 / F")."""
    s = str(age_sex or "")
    m = re.search(r"\b(male|female|[MF])\b", s, re.IGNORECASE)
    return _gender(m.group(1)) if m else None


def build_bundle_from_depaperfy(meta: Optional[dict], results: Optional[list],
                                cfg: ZaFhirConfig = DEFAULT_ZA_CONFIG) -> dict:
    """Build a ZA FHIR bundle straight from De-paperfy's extracted session data
    (`ocr.extract_labs` output: `meta` dict + `results` list). This is the sidecar
    path — no toolkit, no LOINC codes (those only come via the full toolkit
    pipeline); the resulting Observations carry `code.text` but no LOINC coding."""
    from types import SimpleNamespace as _NS

    meta = meta or {}
    patient = _NS(
        name=meta.get("patient_name") or "Unknown",
        gender=_gender_from_age_sex(meta.get("age_sex")),
        dob=None,
        identifiers=_NS(mr=meta.get("patient_id") or None),
    )
    org = _NS(name=meta.get("performing_lab"), address=None) if meta.get("performing_lab") else None
    panel = meta.get("panel") or None
    tests = [
        _NS(test=r.get("test"), name=r.get("test"),
            value=r.get("value"), result=r.get("value"),
            unit=r.get("unit"), reference_range=r.get("reference_range"),
            flag=r.get("flag"), panel_name=panel,
            loinc_code=None, loinc_common_name=None)
        for r in (results or [])
    ]
    lab = _NS(patient=patient, service_provider=org, practitioner=None,
              sample_collection_time=meta.get("collected") or None,
              lab_tests=tests, results=tests)
    return build_za_lab_bundle(lab, cfg)


# ---------------------------------------------------------------- toolkit adapter
class ZaLabReportFhirGenerator:
    """`IFhirGenerator` implementation. Import-guarded so this module stays usable
    (and testable) without the toolkit + google-fhir protos installed."""

    def __init__(self, version: str = "", fhir_profile: str = "",
                 config: Optional[ZaFhirConfig] = None):
        self.cfg = config or ZaFhirConfig()
        if fhir_profile:
            self.cfg.fhir_profile = fhir_profile

    def generate_fhir(self, medical_document: Any):
        bundle_dict = build_za_lab_bundle(medical_document, self.cfg)
        # dict -> proto Bundle, only when actually running inside the toolkit
        import json

        from google.fhir.r4 import json_format  # type: ignore
        from google.fhir.r4.proto.core.resources import (  # type: ignore
            bundle_and_contained_resource_pb2 as fhir_pb2,
        )
        return json_format.json_fhir_string_to_proto(
            json.dumps(bundle_dict), fhir_pb2.Bundle)
