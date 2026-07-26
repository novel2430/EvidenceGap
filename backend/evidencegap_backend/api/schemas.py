from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RunStatus = Literal["queued", "running", "succeeded", "failed"]


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


class RunStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    language: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: RunErrorResponse | None = None
    result: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    engine_loaded: bool
    worker_alive: bool
    active_run_id: str | None
    queued_runs: int
    load_count: int
    analysis_runs: int
