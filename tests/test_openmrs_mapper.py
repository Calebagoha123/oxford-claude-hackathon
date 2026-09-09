import pytest

from integrations.openmrs.client import OpenMRSError
from integrations.openmrs.mapper import publish_record, record_observations


class FakeClient:
    class Settings:
        base_url = "http://openmrs.test/openmrs"

    settings = Settings()

    def __init__(self):
        self.concepts = {}
        self.encounter_args = None

    def get_patient(self, uuid):
        if uuid == "missing":
            raise OpenMRSError("missing patient")
        return {"uuid": uuid}

    def ensure_text_concept(self, name):
        self.concepts.setdefault(name, {"uuid": f"concept-{len(self.concepts) + 1}"})
        return self.concepts[name]

    def create_encounter(self, patient_uuid, observations):
        self.encounter_args = (patient_uuid, observations)
        return {"uuid": "encounter-1", "display": "Consultation"}


def test_note_maps_non_empty_reviewed_fields_and_transcription():
    client = FakeClient()
    record = {
        "doc_type": "note", "text": "original text",
        "fields": {"chief_complaint": "cough", "plan": "CXR", "hpi": ""},
    }
    observations = record_observations(client, record)
    assert [o["value"] for o in observations] == ["cough", "CXR", "original text"]
    assert "De-paperfy Chief Complaint" in client.concepts


def test_lab_maps_each_result_with_unit():
    client = FakeClient()
    record = {"doc_type": "lab", "report": {"panels": [{"rows": [
        {"test": "Haemoglobin", "result": "11.9", "unit": "g/dL"},
    ]}]}}
    assert record_observations(client, record)[0]["value"] == "11.9 g/dL"


def test_publish_validates_patient_then_creates_encounter():
    client = FakeClient()
    result = publish_record(client, "patient-1", {
        "doc_type": "note", "text": "", "fields": {"assessment": "Asthma"},
    })
    assert client.encounter_args[0] == "patient-1"
    assert result["encounter_uuid"] == "encounter-1"
    assert result["chart_url"].endswith("/spa/patient/patient-1/chart")


def test_empty_record_is_rejected():
    with pytest.raises(OpenMRSError, match="no values"):
        record_observations(FakeClient(), {"doc_type": "note", "fields": {}})
