"""Publish a reviewed De-paperfy JSON artifact to OpenMRS.

This is a demo/migration utility; the web API uses the same mapper. Publication
requires --yes so inspecting a file can never create clinical data by accident.
"""

import argparse
import json
from pathlib import Path

from integrations.openmrs import OpenMRSClient, publish_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=Path)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--patient-uuid")
    target.add_argument("--identifier")
    parser.add_argument("--yes", action="store_true", help="confirm creation of an OpenMRS encounter")
    args = parser.parse_args()
    if not args.yes:
        parser.error("publication creates an encounter; pass --yes after reviewing the JSON")

    artifact = json.loads(args.json_file.read_text(encoding="utf-8"))
    extraction = artifact.get("extraction", artifact)
    record = {
        "doc_type": "lab" if extraction.get("report") or extraction.get("panels") else "note",
        **extraction,
    }
    client = OpenMRSClient()
    try:
        patient_uuid = args.patient_uuid or (artifact.get("openmrs") or {}).get("patient_uuid")
        if args.identifier:
            patient = client.find_patient(args.identifier)
            if not patient:
                raise SystemExit(f"No OpenMRS patient has identifier {args.identifier!r}")
            patient_uuid = patient["uuid"]
        if not patient_uuid:
            raise SystemExit("Specify --patient-uuid or --identifier; no UUID is stored in the artifact")
        print(json.dumps(publish_record(client, patient_uuid, record), indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
