"""FastAPI wrapper around the long-lived EvidenceGapEngine."""

from evidencegap_backend.api.app import create_app
from evidencegap_backend.api.config import (
    ApiConfig,
    backend_config_from_env,
    load_config_document,
)

__all__ = [
    "ApiConfig",
    "backend_config_from_env",
    "create_app",
    "load_config_document",
]
