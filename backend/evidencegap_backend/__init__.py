"""Independent EvidenceGap 07.7 backend runtime."""

from evidencegap_backend.config import (
    AgentConfig,
    BackendConfig,
    LLMStageConfig,
    PipelineConfig,
)
from evidencegap_backend.engine import (
    EvidenceGapEngine,
    LocalizationResult,
    StatementAnalysisResult,
)
from evidencegap_backend.resources import RuntimeResources
from evidencegap_backend.common import EvidenceGapError

__all__ = [
    "AgentConfig",
    "BackendConfig",
    "EvidenceGapEngine",
    "LLMStageConfig",
    "PipelineConfig",
    "EvidenceGapError",
    "RuntimeResources",
    "LocalizationResult",
    "StatementAnalysisResult",
]
