"""Small synchronous client for the OpenMRS REST API.

The adapter deliberately owns all OpenMRS-specific payloads. OCR and UI code do
not need to know about UUIDs for locations, identifier types, concepts, or
encounter types.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from .settings import OpenMRSSettings


class OpenMRSError(RuntimeError):
    """A useful, non-secret error returned by OpenMRS."""


class OpenMRSClient:
    def __init__(self, settings: OpenMRSSettings | None = None,
                 transport: httpx.BaseTransport | None = None):
        self.settings = settings or OpenMRSSettings.from_env()
        self._http = httpx.Client(
            base_url=f"{self.settings.base_url}/ws/rest/v1",
            auth=(self.settings.username, self.settings.password),
            timeout=self.settings.timeout,
            headers={"Accept": "application/json"},
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = self._http.request(method, path.lstrip("/"), **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise OpenMRSError(
                f"OpenMRS {method} {path} returned {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OpenMRSError(f"Cannot reach OpenMRS at {self.settings.base_url}: {exc}") from exc
        return response.json() if response.content else None

    def health(self) -> dict:
        return self._request("GET", "session")

    def get_patient(self, patient_uuid: str) -> dict:
        return self._request("GET", f"patient/{patient_uuid}", params={"v": "full"})

    def find_patient(self, identifier: str) -> dict | None:
        data = self._request("GET", "patient", params={"identifier": identifier, "v": "full"})
        results = data.get("results", [])
        return results[0] if results else None

    def search_patients(self, query: str) -> list[dict]:
        data = self._request("GET", "patient", params={"q": query, "v": "default"})
        return data.get("results", [])

    def _find_metadata(self, resource: str, names: tuple[str, ...]) -> dict:
        for name in names:
            data = self._request("GET", resource, params={"q": name, "v": "default"})
            results = data.get("results", [])
            exact = next((x for x in results if (x.get("display") or "").lower() == name.lower()), None)
            if exact:
                return exact
        # Some OpenMRS metadata resources (notably conceptdatatype/class) do
        # not implement `q`; enumerate their small collection and match it.
        data = self._request("GET", resource, params={"v": "default", "limit": 100})
        results = data.get("results", [])
        for name in names:
            exact = next((x for x in results if (x.get("display") or "").lower() == name.lower()), None)
            if exact:
                return exact
        raise OpenMRSError(f"OpenMRS has no usable {resource}; tried {', '.join(names)}")

    def create_demo_patient(self, identifier: str, given_name: str,
                            family_name: str, gender: str = "M") -> dict:
        existing = self.find_patient(identifier)
        if existing:
            return existing
        identifier_type = self._find_metadata(
            "patientidentifiertype", ("Old Identification Number", "OpenMRS ID")
        )
        location = self._find_metadata("location", ("Unknown Location", "Outpatient Clinic"))
        payload = {
            "person": {
                "names": [{"givenName": given_name, "familyName": family_name}],
                "gender": gender,
            },
            "identifiers": [{
                "identifier": identifier,
                "identifierType": identifier_type["uuid"],
                "location": location["uuid"],
                "preferred": True,
            }],
        }
        return self._request("POST", "patient", json=payload)

    def ensure_text_concept(self, name: str) -> dict:
        data = self._request("GET", "concept", params={"q": name, "v": "default"})
        for concept in data.get("results", []):
            if (concept.get("display") or "").lower() == name.lower():
                return concept
        datatype = self._find_metadata("conceptdatatype", ("Text",))
        concept_class = self._find_metadata("conceptclass", ("Miscellaneous", "Misc"))
        return self._request("POST", "concept", json={
            "names": [{"name": name, "locale": "en", "conceptNameType": "FULLY_SPECIFIED"}],
            "datatype": datatype["uuid"],
            "conceptClass": concept_class["uuid"],
        })

    def create_encounter(self, patient_uuid: str, observations: list[dict]) -> dict:
        encounter_type = self._find_metadata(
            "encountertype", ("Consultation", "Visit Note", "Vitals")
        )
        location = self._find_metadata("location", ("Unknown Location", "Outpatient Clinic"))
        now = datetime.now(timezone.utc)
        # OpenMRS REST expects its ISO8601 Long form: millisecond precision and
        # a compact numeric offset (Python's isoformat uses +00:00).
        encounter_datetime = (
            now.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{now.microsecond // 1000:03d}"
            + now.strftime("%z")
        )
        payload = {
            "patient": patient_uuid,
            "encounterDatetime": encounter_datetime,
            "encounterType": encounter_type["uuid"],
            "location": location["uuid"],
            "obs": observations,
        }
        return self._request("POST", "encounter", json=payload)
