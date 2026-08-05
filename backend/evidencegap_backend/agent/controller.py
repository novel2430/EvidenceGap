from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from evidencegap_backend.agent.schemas import AgentDecision, EvidenceWorkspace
from evidencegap_backend.agent.workspace import compact_summary
from evidencegap_backend.common import EvidenceGapError
from evidencegap_backend.config import LLMStageConfig
from evidencegap_backend.stance.llm_judge import call_structured_llm

ControllerCallable = Callable[[EvidenceWorkspace, str | None], AgentDecision]


class EvidenceController:
    """Strict structured controller using the project's existing LLM transport."""

    def __init__(self, config: LLMStageConfig) -> None:
        self.config = config
        default_prompt = Path(__file__).parents[1] / "prompts" / "agent_controller.txt"
        self.system_prompt = (
            config.prompt.system_prompt
            or default_prompt.read_text(encoding="utf-8").strip()
        )
        if config.prompt.additional_instructions:
            self.system_prompt += "\n\n" + config.prompt.additional_instructions

    def __call__(
        self, workspace: EvidenceWorkspace, validation_note: str | None = None
    ) -> AgentDecision:
        cfg = self.config
        api_key_env = cfg.api_key_env or ""
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise EvidenceGapError(
                f"Missing controller API key environment variable: {api_key_env}"
            )
        prompt = {"workspace": compact_summary(workspace)}
        if validation_note:
            prompt["validation_note"] = validation_note
        response = call_structured_llm(
            provider=cfg.provider,
            api_key=api_key,
            base_url=str(cfg.base_url or ""),
            model=str(cfg.model or ""),
            system_prompt=self.system_prompt,
            user_prompt=json.dumps(prompt, ensure_ascii=False, sort_keys=True),
            response_schema=AgentDecision.model_json_schema(),
            max_tokens=cfg.max_tokens,
            timeout_seconds=cfg.timeout_seconds,
            thinking=bool(cfg.thinking),
        )
        try:
            return AgentDecision.model_validate(response.payload)
        except ValidationError as exc:
            raise EvidenceGapError(f"Invalid controller decision: {exc}") from exc
