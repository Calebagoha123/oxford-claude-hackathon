"""A curated LOINC lookup for common lab analytes — the sidecar LOINC stage.

The full Google Health medical-data-toolkit has a LOINC coder; the De-paperfy
sidecar path doesn't run the toolkit, so this module supplies LOINC codes for the
common analytes a primary-care lab report contains (chemistry, haematology,
serology, urinalysis). It is deliberately dependency-free and deterministic.

Coverage is the common panel, not all ~100k LOINC terms. `lookup(name)` returns
(loinc_code, loinc_common_name) or (None, None) for an unrecognised test — the
FHIR generator then falls back to a text-only `Observation.code`, so an unknown
analyte degrades gracefully rather than emitting a wrong code.

Codes are the widely-used serum/plasma or blood variants; where a report's unit
implies a different molecular vs mass variant we still emit the common code (a
demo-grade approximation — the toolkit's coder is unit-aware and authoritative).
"""

from __future__ import annotations

import re

# canonical normalised name -> (LOINC code, LOINC common/long name)
_LOINC: dict[str, tuple[str, str]] = {
    # ---- renal / electrolytes ----
    "sodium": ("2951-2", "Sodium [Moles/volume] in Serum or Plasma"),
    "potassium": ("2823-3", "Potassium [Moles/volume] in Serum or Plasma"),
    "chloride": ("2075-0", "Chloride [Moles/volume] in Serum or Plasma"),
    "bicarbonate": ("1963-8", "Bicarbonate [Moles/volume] in Serum or Plasma"),
    "urea": ("22664-7", "Urea [Moles/volume] in Serum or Plasma"),
    "creatinine": ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma"),
    "egfr": ("62238-1", "Glomerular filtration rate/1.73 sq M.predicted (CKD-EPI)"),
    "calcium": ("17861-6", "Calcium [Mass/volume] in Serum or Plasma"),
    "calcium corrected": ("18281-6", "Calcium.corrected for albumin [Mass/volume]"),
    "phosphate": ("2777-1", "Phosphate [Mass/volume] in Serum or Plasma"),
    "magnesium": ("2601-3", "Magnesium [Moles/volume] in Serum or Plasma"),
    # ---- liver ----
    "total bilirubin": ("1975-2", "Bilirubin.total [Mass/volume] in Serum or Plasma"),
    "direct bilirubin": ("1968-7", "Bilirubin.direct [Mass/volume] in Serum or Plasma"),
    "alt": ("1742-6", "Alanine aminotransferase [Enzymatic activity/volume]"),
    "ast": ("1920-8", "Aspartate aminotransferase [Enzymatic activity/volume]"),
    "alp": ("6768-6", "Alkaline phosphatase [Enzymatic activity/volume]"),
    "ggt": ("2324-2", "Gamma glutamyl transferase [Enzymatic activity/volume]"),
    "total protein": ("2885-2", "Protein [Mass/volume] in Serum or Plasma"),
    "albumin": ("1751-7", "Albumin [Mass/volume] in Serum or Plasma"),
    # ---- haematology (FBC) ----
    "haemoglobin": ("718-7", "Hemoglobin [Mass/volume] in Blood"),
    "haematocrit": ("4544-3", "Hematocrit [Volume Fraction] of Blood by Automated count"),
    "wbc": ("6690-2", "Leukocytes [#/volume] in Blood by Automated count"),
    "rbc": ("789-8", "Erythrocytes [#/volume] in Blood by Automated count"),
    "platelets": ("777-3", "Platelets [#/volume] in Blood by Automated count"),
    "mcv": ("787-2", "MCV [Entitic volume] by Automated count"),
    "mch": ("785-6", "MCH [Entitic mass] by Automated count"),
    "mchc": ("786-4", "MCHC [Mass/volume] by Automated count"),
    "rdw": ("788-0", "Erythrocyte distribution width [Ratio] by Automated count"),
    "neutrophils": ("751-8", "Neutrophils [#/volume] in Blood by Automated count"),
    "lymphocytes": ("731-0", "Lymphocytes [#/volume] in Blood by Automated count"),
    "monocytes": ("742-7", "Monocytes [#/volume] in Blood by Automated count"),
    "eosinophils": ("711-2", "Eosinophils [#/volume] in Blood by Automated count"),
    "basophils": ("704-7", "Basophils [#/volume] in Blood by Automated count"),
    # ---- inflammatory / metabolic ----
    "crp": ("1988-5", "C reactive protein [Mass/volume] in Serum or Plasma"),
    "lactate": ("2524-7", "Lactate [Moles/volume] in Serum or Plasma"),
    "glucose": ("2345-7", "Glucose [Mass/volume] in Serum or Plasma"),
    "random blood glucose": ("2345-7", "Glucose [Mass/volume] in Serum or Plasma"),
    "hba1c": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood"),
    # ---- serology / antenatal ----
    "blood group": ("883-9", "ABO group [Type] in Blood"),
    "rhesus d factor": ("10331-7", "Rh [Type] in Blood"),
    "hiv": ("75622-1", "HIV 1 and 2 Ab+Ag panel in Serum or Plasma"),
    "syphilis rpr": ("20507-0", "Reagin Ab [Titer] in Serum by RPR"),
    "syphilis tpha": ("22592-0", "Treponema pallidum Ab [Presence] in Serum"),
    "hbsag": ("5195-3", "Hepatitis B virus surface Ag [Presence] in Serum"),
    "malaria": ("70802-4", "Plasmodium sp Ag [Presence] in Blood by Rapid test"),
    # ---- urinalysis ----
    "urine protein": ("5804-0", "Protein [Mass/volume] in Urine by Test strip"),
    "urine glucose": ("5792-7", "Glucose [Mass/volume] in Urine by Test strip"),
    "urine nitrites": ("5802-4", "Nitrite [Presence] in Urine by Test strip"),
}

# variant spellings / abbreviations -> canonical key above
_ALIASES: dict[str, str] = {
    "na": "sodium", "k": "potassium", "cl": "chloride",
    "hco3": "bicarbonate", "co2": "bicarbonate", "bicarb": "bicarbonate",
    "bun": "urea", "creat": "creatinine",
    "egfr ckd epi": "egfr", "egfr ckdepi": "egfr", "gfr": "egfr",
    "corrected calcium": "calcium corrected", "ca corrected": "calcium corrected",
    "ca": "calcium", "po4": "phosphate", "phosphorus": "phosphate", "mg": "magnesium",
    "tbili": "total bilirubin", "t bilirubin": "total bilirubin", "bilirubin total": "total bilirubin",
    "dbili": "direct bilirubin", "bilirubin direct": "direct bilirubin",
    "sgpt": "alt", "sgot": "ast", "alk phos": "alp", "alkaline phosphatase": "alp",
    "protein total": "total protein", "alb": "albumin",
    "hb": "haemoglobin", "hgb": "haemoglobin", "hemoglobin": "haemoglobin",
    "hct": "haematocrit", "hematocrit": "haematocrit", "pcv": "haematocrit",
    "white cell count": "wbc", "leukocytes": "wbc", "white blood cells": "wbc",
    "red cell count": "rbc", "erythrocytes": "rbc", "red blood cells": "rbc",
    "platelet count": "platelets", "plt": "platelets",
    "neutrophil": "neutrophils", "lymphocyte": "lymphocytes",
    "monocyte": "monocytes", "eosinophil": "eosinophils", "basophil": "basophils",
    "c reactive protein": "crp",
    "rbg": "random blood glucose", "rbs": "random blood glucose",
    "blood sugar": "glucose", "fbg": "glucose", "fasting glucose": "glucose",
    "abo group": "blood group", "abo": "blood group",
    "rhesus": "rhesus d factor", "rh factor": "rhesus d factor", "rh d": "rhesus d factor",
    "rhesus factor": "rhesus d factor", "rh": "rhesus d factor",
    "hiv 1 2": "hiv", "hiv12": "hiv", "hiv 1 and 2": "hiv",
    "rpr": "syphilis rpr", "syphilis": "syphilis rpr",
    "tpha": "syphilis tpha", "syphilis confirm": "syphilis tpha",
    "hepatitis b": "hbsag", "hep b": "hbsag", "hbv sag": "hbsag",
    "malaria rdt": "malaria", "malaria thick film": "malaria", "mp": "malaria",
    "urinalysis protein": "urine protein", "urine analysis protein": "urine protein",
    "urinalysis glucose": "urine glucose", "urine analysis glucose": "urine glucose",
    "urinalysis nitrites": "urine nitrites", "urine analysis nitrites": "urine nitrites",
    "nitrites": "urine nitrites",
}


def _normalise(name: str) -> str:
    """Lowercase, drop parenthetical qualifiers and punctuation, collapse spaces.
    'eGFR (CKD-EPI)' -> 'egfr'; 'Calcium (corrected)' -> 'calcium corrected'."""
    s = str(name or "").lower()
    # keep the word inside parens only for the corrected-calcium case; otherwise drop
    paren = re.findall(r"\(([^)]*)\)", s)
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    # fold a trailing "corrected" that lived in parens back on (calcium corrected)
    if paren and "corrected" in " ".join(paren).lower() and "corrected" not in s:
        s = f"{s} corrected".strip()
    return s


def lookup(name: str) -> tuple[str | None, str | None]:
    """(loinc_code, loinc_common_name) for a test name, or (None, None)."""
    key = _normalise(name)
    if not key:
        return (None, None)
    canon = _ALIASES.get(key, key)
    hit = _LOINC.get(canon)
    if hit:
        return hit
    # last resort: alias may itself need the normaliser applied (e.g. multi-word)
    canon = _ALIASES.get(re.sub(r"\s+", " ", key))
    if canon and canon in _LOINC:
        return _LOINC[canon]
    return (None, None)
