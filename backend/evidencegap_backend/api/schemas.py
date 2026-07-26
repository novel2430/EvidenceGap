from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RunStatus = Literal["queued", "running", "succeeded", "failed"]
RunStage = Literal[
    "statement_decomposition",
    "claim_analysis",
    "statement_bundle",
    "inference_gap_analysis",
    "output_generation",
]


class RunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1, max_length=100_000)
    language: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("statement", "language")
    @classmethod
    def strip_non_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class RunAcceptedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: Literal["queued"]
    created_at: datetime


class RunErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class RunProgressResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: RunStage
    stage_index: int = Field(ge=1)
    total_stages: int = Field(ge=1)
    message: str = Field(min_length=1)
    completed_units: int | None = Field(default=None, ge=0)
    total_units: int | None = Field(default=None, ge=0)
    updated_at: datetime


class RunStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    language: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: RunProgressResponse | None = None
    execution_summary: dict[str, Any] | None = None
    error: RunErrorResponse | None = None
    result: dict[str, Any] | None = None


class RunListItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    statement_preview: str
    language: str
    status: RunStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    total_seconds: float | None = None
    summary: dict[str, Any] | None = None
    error: RunErrorResponse | None = None


class RunListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: list[RunListItemResponse]
    next_cursor: str | None = None


class ArticleSectionSpanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence_type: str
    section: str
    section_index: int = Field(ge=0)
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=0)


class ArticleEvidenceSpanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    claim_id: str
    section: str | None = None
    section_index: int = Field(ge=0)
    sentence_index: int = Field(ge=0)
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=0)
    text: str


class ArticleContextResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article_node_id: str
    article_id: str
    claim_id: str
    pmid: str | None = None
    title: str | None = None
    canonical_text: str
    source_text_fingerprint: str
    fingerprint_verified: bool
    sections: list[ArticleSectionSpanResponse]
    evidence_spans: list[ArticleEvidenceSpanResponse]


class LocalizationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = Field(min_length=1, max_length=100)

    @field_validator("language")
    @classmethod
    def strip_language(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class LocalizationAcceptedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    localization_id: str
    source_run_id: str
    language: str
    status: Literal["queued"]
    created_at: datetime


class LocalizationStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    localization_id: str
    source_run_id: str
    language: str
    status: RunStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: RunErrorResponse | None = None
    result: dict[str, Any] | None = None


class LocalizationListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    localizations: list[LocalizationStatusResponse]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    engine_loaded: bool
    worker_alive: bool
    active_run_id: str | None
    queued_runs: int
    load_count: int
    analysis_runs: int
