from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentAction(StrEnum):
    SEARCH = "SEARCH"
    RESOLVE = "RESOLVE"
    ABSTAIN = "ABSTAIN"
    FINISH = "FINISH"


class GapAction(StrEnum):
    REQUEST_MORE_EVIDENCE = "REQUEST_MORE_EVIDENCE"
    ACCEPT_GAPS = "ACCEPT_GAPS"
    ABSTAIN = "ABSTAIN"


class GapDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: GapAction
    target_gap_id: str | None = None
    claim_id: str | None = None
    query: str | None = None
    remaining_problem: str | None = None
    reason: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_action_fields(self) -> "GapDecision":
        if self.action is GapAction.REQUEST_MORE_EVIDENCE:
            if not all(
                (self.target_gap_id, self.claim_id, (self.query or "").strip())
            ):
                raise ValueError(
                    "REQUEST_MORE_EVIDENCE requires target_gap_id, claim_id, and query"
                )
        elif any(
            (self.target_gap_id, self.claim_id, self.query, self.remaining_problem)
        ):
            raise ValueError(
                f"{self.action} does not accept gap, claim, query, or remaining problem fields"
            )
        return self


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: AgentAction
    claim_id: str | None = None
    query: str | None = None
    selected_attempt_id: str | None = None
    remaining_problem: str | None = None
    reason: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_action_fields(self) -> "AgentDecision":
        if self.action is AgentAction.SEARCH:
            if not (self.claim_id or "").strip() or not (self.query or "").strip():
                raise ValueError("SEARCH requires claim_id and query")
            if self.selected_attempt_id is not None:
                raise ValueError("SEARCH does not accept selected_attempt_id")
        elif self.action in {AgentAction.RESOLVE, AgentAction.ABSTAIN}:
            if not (self.claim_id or "").strip():
                raise ValueError(f"{self.action} requires claim_id")
            if self.query is not None:
                raise ValueError(f"{self.action} does not accept query")
        elif any(
            (
                self.claim_id,
                self.query,
                self.selected_attempt_id,
                self.remaining_problem,
            )
        ):
            raise ValueError(
                "FINISH does not accept claim, query, attempt, or remaining problem fields"
            )
        return self


class SearchEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str = Field(min_length=1)
    canonical_claim: str = Field(min_length=1)
    query: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)


class SearchAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attempt_id: str
    claim_id: str
    query: str
    normalized_query: str
    artifact_dir: str | None = None
    graph_bundle_path: str | None = None
    verdict: str | None = None
    article_counts: dict[str, int] = Field(default_factory=dict)
    article_ids: list[str] = Field(default_factory=list)
    new_article_ids: list[str] = Field(default_factory=list)
    top_article_titles: list[str] = Field(default_factory=list)
    direct_evidence_articles: int = 0
    utility_score: int = 0
    status: Literal["successful", "failed"]
    error: str | None = None
    started_at: str
    finished_at: str


class ClaimWorkspace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    source_text: str
    source_spans: list[dict[str, Any]]
    canonical_claim_en: str
    status: Literal["pending", "resolved", "abstained", "failed"] = "pending"
    verdict: str | None = None
    used_queries: list[str] = Field(default_factory=list)
    normalized_queries: list[str] = Field(default_factory=list)
    seen_article_ids: list[str] = Field(default_factory=list)
    attempts: list[SearchAttempt] = Field(default_factory=list)
    selected_attempt_id: str | None = None
    remaining_problem: str | None = None
    terminal_reason: str | None = None
    last_error: str | None = None
    reopen_count: int = 0
    reopened_for_gap_id: str | None = None


class EvidenceWorkspace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0.0"
    run_name: str
    statement: str
    language: str
    decomposition: dict[str, Any]
    claims: dict[str, ClaimWorkspace]
    claim_order: list[str]
    active_claim_id: str | None = None
    decision: AgentDecision | None = None
    decision_source: str | None = None
    gap_decision: GapDecision | None = None
    gap_decision_source: str | None = None
    last_action_result: dict[str, Any] | None = None
    step_count: int = 0
    max_steps: int = 20
    initial_search_budget: int = 8
    remaining_search_budget: int = 8
    per_claim_search_budget: int = 3
    action_counts: dict[str, int] = Field(
        default_factory=lambda: {action.value: 0 for action in AgentAction}
    )
    rejected_decisions: int = 0
    rejected_gap_decisions: int = 0
    evidence_controller_decision_count: int = 0
    gap_controller_decision_count: int = 0
    deterministic_fallback_decisions: int = 0
    phase: Literal[
        "evidence",
        "materializing",
        "bundle",
        "gap_analysis",
        "gap_decision",
        "output",
        "finished",
        "failed",
    ] = "evidence"
    evidence_cycle: int = 1
    gap_round: int = 0
    gap_remediation_count: int = 0
    max_gap_rounds: int = 2
    initial_gap_remediation_budget: int = 2
    remaining_gap_remediation_budget: int = 2
    gap_history: list[dict[str, Any]] = Field(default_factory=list)
    latest_gap_summary: dict[str, Any] | None = None
    final_output_path: str | None = None
    status: Literal["running", "finished", "failed"] = "running"
    finish_reason: str | None = None
    started_at: str
    finished_at: str | None = None


class AgentGraphState(dict):
    """Marker type retained for API discoverability; graph uses a TypedDict."""
