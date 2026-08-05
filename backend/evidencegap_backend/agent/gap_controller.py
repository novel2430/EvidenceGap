from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from pydantic import ValidationError

from evidencegap_backend.agent.schemas import EvidenceWorkspace, GapDecision
from evidencegap_backend.common import EvidenceGapError, sha256_text
from evidencegap_backend.config import LLMStageConfig
from evidencegap_backend.stance.llm_judge import call_structured_llm

GapControllerCallable = Callable[
    [EvidenceWorkspace, Mapping[str, Any], str | None], GapDecision
]


def compact_gap_summary(
    workspace: EvidenceWorkspace,
    gap_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    gaps: list[dict[str, Any]] = []
    for index, analysis in enumerate(
        gap_bundle.get("inference_gap_analyses", []), start=1
    ):
        if not isinstance(analysis, Mapping):
            continue
        inference_step_id = str(analysis.get("inference_step_id") or "")
        for gap_type in ("scope_gap", "causal_gap"):
            raw = analysis.get(gap_type)
            if not isinstance(raw, Mapping) or not raw.get("detected"):
                continue
            identity = json.dumps(
                {
                    "index": index,
                    "inference_step_id": inference_step_id,
                    "gap_type": gap_type,
                    "gap": dict(raw),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            gaps.append(
                {
                    "gap_id": "gap_" + sha256_text(identity)[:16],
                    "inference_step_id": inference_step_id,
                    "gap_type": gap_type,
                    "subtype": raw.get("subtype"),
                    "affected_dimensions": list(
                        raw.get("affected_dimensions") or []
                    ),
                    "supported_basis": raw.get("supported_basis"),
                    "unsupported_extension": raw.get("unsupported_extension"),
                    "reason": raw.get("reason"),
                    "closure_requirement": raw.get("closure_requirement"),
                }
            )
    claims = [
        {
            "claim_id": claim.claim_id,
            "canonical_claim": claim.canonical_claim_en,
            "status": claim.status,
            "verdict": claim.verdict,
            "selected_attempt_id": claim.selected_attempt_id,
            "used_queries": claim.used_queries,
            "remaining_problem": claim.remaining_problem,
        }
        for claim in (workspace.claims[cid] for cid in workspace.claim_order)
    ]
    remediation_history = [
        {
            key: row.get(key)
            for key in (
                "gap_round",
                "evidence_cycle",
                "action",
                "target_gap_id",
                "claim_id",
                "query",
                "reason",
                "decision_source",
            )
        }
        for row in workspace.gap_history
    ]
    return {
        "original_statement": workspace.statement,
        "inference_steps": workspace.decomposition.get("inference_steps", []),
        "claims": claims,
        "formal_gap_summary": dict(gap_bundle.get("summary") or {}),
        "gaps": gaps,
        "gap_round": workspace.gap_round,
        "max_gap_rounds": workspace.max_gap_rounds,
        "remaining_search_budget": workspace.remaining_search_budget,
        "remaining_gap_remediation_budget": (
            workspace.remaining_gap_remediation_budget
        ),
        "previous_remediation_history": remediation_history,
    }


class GapController:
    """Structured controller deciding whether formal gaps justify more search."""

    def __init__(self, config: LLMStageConfig) -> None:
        self.config = config
        default_prompt = (
            Path(__file__).parents[1] / "prompts" / "agent_gap_controller.txt"
        )
        self.system_prompt = (
            config.prompt.system_prompt
            or default_prompt.read_text(encoding="utf-8").strip()
        )
        if config.prompt.additional_instructions:
            self.system_prompt += "\n\n" + config.prompt.additional_instructions

    def __call__(
        self,
        workspace: EvidenceWorkspace,
        gap_bundle: Mapping[str, Any],
        validation_note: str | None = None,
    ) -> GapDecision:
        cfg = self.config
        api_key_env = cfg.api_key_env or ""
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise EvidenceGapError(
                f"Missing gap controller API key environment variable: {api_key_env}"
            )
        prompt: dict[str, Any] = {
            "gap_workspace": compact_gap_summary(workspace, gap_bundle)
        }
        if validation_note:
            prompt["validation_note"] = validation_note
        response = call_structured_llm(
            provider=cfg.provider,
            api_key=api_key,
            base_url=str(cfg.base_url or ""),
            model=str(cfg.model or ""),
            system_prompt=self.system_prompt,
            user_prompt=json.dumps(prompt, ensure_ascii=False, sort_keys=True),
            response_schema=GapDecision.model_json_schema(),
            max_tokens=cfg.max_tokens,
            timeout_seconds=cfg.timeout_seconds,
            thinking=bool(cfg.thinking),
        )
        try:
            return GapDecision.model_validate(response.payload)
        except ValidationError as exc:
            raise EvidenceGapError(f"Invalid gap controller decision: {exc}") from exc
