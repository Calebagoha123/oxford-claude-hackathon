import app as app_module


class FakeOpenMRS:
    class Settings:
        base_url = "http://openmrs.test/openmrs"

    settings = Settings()
    closed = False

    def close(self):
        self.closed = True

    def search_patients(self, query):
        return [{"uuid": "patient-1", "display": f"Synth {query}"}]

    def get_patient(self, uuid):
        return {"uuid": uuid}

    def ensure_text_concept(self, name):
        return {"uuid": "concept-1"}

    def create_encounter(self, patient_uuid, observations):
        return {"uuid": "encounter-1", "display": "Consultation"}


def test_patient_search_uses_openmrs_adapter(client, monkeypatch):
    monkeypatch.setattr(app_module, "_openmrs_client", FakeOpenMRS)
    response = client.get("/api/openmrs/patients?q=001")
    assert response.status_code == 200
    assert response.json()["results"][0]["uuid"] == "patient-1"


def test_approval_requires_completed_scan(client):
    sid = client.post("/api/scan/session", json={
        "openmrs_patient_uuid": "patient-1",
    }).json()["id"]
    response = client.post(f"/api/scan/session/{sid}/approve", json={})
    assert response.status_code == 409


def test_reviewed_record_is_published_once(client, monkeypatch):
    monkeypatch.setattr(app_module, "_openmrs_client", FakeOpenMRS)
    sid = "approval-test"
    app_module._sessions[sid] = {
        "status": "done", "openmrs_patient_uuid": "patient-1",
        "records": [{"doc_type": "note", "text": "", "fields": {"plan": "Review"}}],
        "created": 0,
    }
    response = client.post(f"/api/scan/session/{sid}/approve", json={})
    assert response.status_code == 200
    assert response.json()["encounter_uuid"] == "encounter-1"
    assert client.post(f"/api/scan/session/{sid}/approve", json={}).status_code == 409
