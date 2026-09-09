"""Environment-backed OpenMRS configuration."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class OpenMRSSettings:
    base_url: str
    username: str
    password: str
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "OpenMRSSettings":
        return cls(
            base_url=os.getenv("OPENMRS_BASE_URL", "http://localhost:8080/openmrs").rstrip("/"),
            username=os.getenv("OPENMRS_USERNAME", "admin"),
            password=os.getenv("OPENMRS_PASSWORD", "Admin123"),
            timeout=float(os.getenv("OPENMRS_TIMEOUT", "30")),
        )
