"""Map reviewed De-paperfy records into OpenMRS encounters."""

from __future__ import annotations

from .client import OpenMRSClient, OpenMRSError


_NOTE_LABELS = {
    "note_type": "Note Type",
    "chief_complaint": "Chief Complaint",
    "hpi": "History of Present Illness",
    "pmhx": "Past Medical History",
    "fmhx": "Family History",
    "shx": "Social History",
    "ros": "Review of Systems",
    "pe": "Physical Examination",
    "assessment": "Assessment",
    "plan": "Plan",
}


def _text_obs(client: OpenMRSClient, label: str, value: object) -> dict | None:
    text = str(value or "").strip()
    if not text:
        return None
    concept = client.ensure_text_concept(f"De-paperfy {label}")
    return {"concept": concept["uuid"], "value": text}


def record_observations(client: OpenMRSClient, record: dict) -> list[dict]:
    observations: list[dict] = []
    if record.get("doc_type") == "note":
        for key, label in _NOTE_LABELS.items():
            obs = _text_obs(client, label, (record.get("fields") or {}).get(key))
            if obs:
                observations.append(obs)
        raw = _text_obs(client, "Source Transcription", record.get("text"))
        if raw:
            observations.append(raw)
    elif record.get("doc_type") == "lab":
        for panel in (record.get("report") or {}).get("panels", []):
            for row in panel.get("rows", []):
                test = str(row.get("test") or "Unnamed Result").strip()
                result = str(row.get("result") or row.get("result_abs") or row.get("result_pct") or "").strip()
                unit = str(row.get("unit") or "").strip()
                value = " ".join(x for x in (result, unit) if x)
                obs = _text_obs(client, f"Lab {test}", value)
                if obs:
                    observations.append(obs)
    else:
        raise OpenMRSError(f"Unsupported record type: {record.get('doc_type')!r}")
    if not observations:
        raise OpenMRSError("The reviewed record contains no values to publish")
    return observations


def publish_record(client: OpenMRSClient, patient_uuid: str, record: dict) -> dict:
    client.get_patient(patient_uuid)  # fail before creating metadata/clinical data
    encounter = client.create_encounter(patient_uuid, record_observations(client, record))
    return {
        "patient_uuid": patient_uuid,
        "encounter_uuid": encounter["uuid"],
        "display": encounter.get("display", "De-paperfy encounter"),
        "chart_url": f"{client.settings.base_url}/spa/patient/{patient_uuid}/chart",
    }
