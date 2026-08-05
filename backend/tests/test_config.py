from __future__ import annotations

import json
from pathlib import Path

from evidencegap_backend.api.config import (
    ApiConfig,
    backend_config_from_env,
    load_config_document,
)


def test_config_json_env_precedence_and_prompt_loading(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    prompt_path = prompt_dir / "decomposition.txt"
    prompt_path.write_text("Preserve every claim qualifier.\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "config_version": 1,
                "workspace_root": ".",
                "runtime": {"device": "cuda:7", "article_cache_size": 123},
                "llm": {
                    "defaults": {
                        "provider": "deepseek",
                        "model": "deepseek-v4-flash",
                        "timeout_seconds": 90,
                    },
                    "stages": {
                        "statement_decomposition": {
                            "model": "deepseek-v4-pro",
                            "prompt": {
                                "system_file": "prompts/decomposition.txt",
                                "version": "custom-decomposition-v1",
                            },
                        },
                        "article_evidence": {"request_batch_size": 2},
                    },
                },
                "pipeline": {
                    "retrieval": {"final_article_top_k": 8},
                    "article_evidence": {"max_evidence_sentences": 4},
                },
                "output": {"default_language": "繁體中文（台灣）"},
                "api": {"max_queue_size": 5, "cors_origins": ["http://localhost:5173"]},
            }
        ),
        encoding="utf-8",
    )
    env = {
        "EVIDENCEGAP_CONFIG": str(config_path),
        "EVIDENCEGAP_MODEL": "deepseek-v4-flash-env",
        "EVIDENCEGAP_INFERENCE_GAP_MODEL": "deepseek-v4-pro-gap",
        "EVIDENCEGAP_DEVICE": "cuda:0",
    }
    document = load_config_document(env)
    config = backend_config_from_env(env, document=document)
    api = ApiConfig.from_env(config, env, document=document)

    assert config.workspace_root == tmp_path
    assert config.device == "cuda:0"
    assert config.article_cache_size == 123
    assert config.decomposition_llm is not None
    assert config.decomposition_llm.model == "deepseek-v4-flash-env"
    assert config.decomposition_llm.prompt.system_prompt == (
        "Preserve every claim qualifier."
    )
    assert config.decomposition_llm.prompt.version == "custom-decomposition-v1"
    assert config.article_evidence_llm is not None
    assert config.article_evidence_llm.request_batch_size == 2
    assert config.inference_gap_llm is not None
    assert config.inference_gap_llm.model == "deepseek-v4-pro-gap"
    assert config.pipeline.final_article_top_k == 8
    assert config.pipeline.max_evidence_sentences == 4
    assert config.default_language == "繁體中文（台灣）"
    assert api.max_queue_size == 5
    assert api.cors_origins == ("http://localhost:5173",)


def test_safe_snapshot_never_contains_api_key_value(tmp_path: Path) -> None:
    config = backend_config_from_env(
        {
            "EVIDENCEGAP_WORKSPACE_ROOT": str(tmp_path),
            "EVIDENCEGAP_PROVIDER": "deepseek",
            "EVIDENCEGAP_API_KEY_ENV": "CUSTOM_DEEPSEEK_KEY",
            "CUSTOM_DEEPSEEK_KEY": "secret-value-must-not-be-recorded",
        }
    )

    rendered = json.dumps(config.safe_dict(), ensure_ascii=False)
    assert "CUSTOM_DEEPSEEK_KEY" in rendered
    assert "secret-value-must-not-be-recorded" not in rendered


def test_agent_config_and_controller_stage_environment(tmp_path: Path) -> None:
    config = backend_config_from_env(
        {
            "EVIDENCEGAP_WORKSPACE_ROOT": str(tmp_path),
            "EVIDENCEGAP_AGENT_ENABLED": "false",
            "EVIDENCEGAP_AGENT_MAX_STEPS": "9",
            "EVIDENCEGAP_AGENT_TOTAL_SEARCH_BUDGET": "4",
            "EVIDENCEGAP_AGENT_PER_CLAIM_SEARCH_BUDGET": "2",
            "EVIDENCEGAP_AGENT_CHECKPOINT_ENABLED": "false",
            "EVIDENCEGAP_AGENT_CONTROLLER_MODEL": "deepseek-v4-pro",
        }
    )
    assert not config.agent.enabled
    assert config.agent.max_steps == 9
    assert config.agent.total_search_budget == 4
    assert config.agent.per_claim_search_budget == 2
    assert not config.agent.checkpoint_enabled
    assert config.agent_controller_llm is not None
    assert config.agent_controller_llm.model == "deepseek-v4-pro"
    assert "agent" in config.safe_dict()


def test_prompt_override_is_used_and_recorded(tmp_path: Path, monkeypatch) -> None:
    from evidencegap_backend.pipeline.statement_decomposition import (
        run_statement_decomposition,
    )
    from evidencegap_backend.prompting import PromptOverride
    from evidencegap_backend.stance.llm_judge import ProviderResponse

    captured: dict[str, str] = {}

    def fake_call(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        return ProviderResponse(
            payload={
                "source_language": "en",
                "claims": [
                    {
                        "claim_ref": "C1",
                        "source_text": "Vitamin D reduces infection risk.",
                        "canonical_claim_en": "Vitamin D reduces infection risk.",
                    }
                ],
                "inference_steps": [],
            },
            request_id="request-1",
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            raw_response_sha256="a" * 64,
            finish_reason="stop",
        )

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "evidencegap_backend.pipeline.statement_decomposition.call_structured_llm",
        fake_call,
    )
    result = run_statement_decomposition(
        tmp_path,
        statement="Vitamin D reduces infection risk.",
        provider="deepseek",
        run_name="prompt-test",
        model="deepseek-v4-flash",
        prompt_override=PromptOverride(
            system_prompt="Custom decomposition prompt.",
            additional_instructions="Preserve modality exactly.",
            version="custom-v1",
            source="config.json:inline",
        ),
        artifact_root=tmp_path / "artifacts",
    )

    assert captured["system_prompt"] == (
        "Custom decomposition prompt.\n\n"
        "Additional configured instructions:\nPreserve modality exactly."
    )
    manifest = json.loads(
        (tmp_path / "artifacts/prompt-test/run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["prompt_version"] == "custom-v1"
    assert manifest["prompt_source"] == "config.json:inline"
    assert len(manifest["prompt_sha256"]) == 64
    assert result["claims"] == 1


def test_stage_provider_uses_provider_specific_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "config_version": 1,
                "workspace_root": ".",
                "llm": {
                    "defaults": {
                        "provider": "deepseek",
                        "model": "deepseek-v4-flash",
                        "api_key_env": "DEEPSEEK_API_KEY",
                    },
                    "stages": {
                        "inference_gap": {"provider": "anthropic"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    config = backend_config_from_env({"EVIDENCEGAP_CONFIG": str(config_path)})

    assert config.inference_gap_llm is not None
    assert config.inference_gap_llm.provider == "anthropic"
    assert config.inference_gap_llm.model == "claude-sonnet-4-6"
    assert config.inference_gap_llm.api_key_env == "ANTHROPIC_API_KEY"
    assert config.inference_gap_llm.base_url == "https://api.anthropic.com"


def test_builtin_prompts_are_packaged_and_rendered() -> None:
    from evidencegap_backend.prompting import load_builtin_prompt

    decomposition = load_builtin_prompt("statement_decomposition.txt")
    article = load_builtin_prompt(
        "article_evidence.txt",
        substitutions={"max_evidence_sentences": 7},
    )
    gap = load_builtin_prompt("inference_gap.txt")
    localization = load_builtin_prompt("localization.txt")

    assert decomposition.startswith("You perform argument-preserving decomposition")
    assert "at most 7 sentences" in article
    assert "{{max_evidence_sentences}}" not in article
    assert gap.startswith("You analyze evidence gaps")
    assert localization.startswith("You localize an already validated")


def test_pipeline_modules_do_not_embed_system_prompt_constants() -> None:
    import inspect

    from evidencegap_backend.output import presentation
    from evidencegap_backend.pipeline import (
        article_evidence,
        inference_gap_analysis,
        statement_decomposition,
    )

    for module in (
        statement_decomposition,
        article_evidence,
        inference_gap_analysis,
        presentation,
    ):
        source = inspect.getsource(module)
        assert "SYSTEM_PROMPT =" not in source
        assert "resolve_builtin_prompt(" in source
