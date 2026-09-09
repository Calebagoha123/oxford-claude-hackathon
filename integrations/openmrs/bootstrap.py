"""Idempotently seed synthetic patients used by the public demo."""

import json
import os
import time

from .client import OpenMRSClient


DEMO_PATIENTS = (
    ("SYNTH PATIENT 001", "Synth", "Patient 001", "M"),
    ("SYNTH PATIENT 002", "Synth", "Patient 002", "M"),
)


def bootstrap(client: OpenMRSClient) -> list[dict]:
    return [client.create_demo_patient(*patient) for patient in DEMO_PATIENTS]


def main() -> None:
    deadline = time.monotonic() + float(os.getenv("OPENMRS_BOOTSTRAP_TIMEOUT", "300"))
    while True:
        client = OpenMRSClient()
        try:
            patients = bootstrap(client)
            print(json.dumps({"seeded": [p.get("uuid") for p in patients]}))
            return
        except Exception:
            if time.monotonic() >= deadline:
                raise
            time.sleep(5)
        finally:
            client.close()


if __name__ == "__main__":
    main()
