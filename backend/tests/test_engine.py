from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evidencegap_backend import BackendConfig, EvidenceGapEngine, EvidenceGapError


class FakeResources:
    def __init__(self) -> None:
        self.loaded = False
        self.load_count = 0
        self.analysis_runs = 0

    def load(self, *, validate_paths: bool = True) -> None:
        if self.loaded:
            return
        self.loaded = True
        self.load_count += 1

    def close(self) -> None:
        self.loaded = False

    def record_analysis_run(self) -> None:
        self.analysis_runs += 1

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "load_count": self.load_count,
            "analysis_runs": self.analysis_runs,
            "resource_ids": {"fake": id(self)},
        }


def test_config_resolves_runtime_paths_from_workspace(tmp_path: Path) -> None:
    config = BackendConfig(workspace_root=tmp_path, provider="deepseek")

    assert config.artifact_root == tmp_path / "artifacts/v1/pipeline/statement_run"
    assert config.bm25_index_dir == tmp_path / "artifacts/v1/bm25_index"
    assert config.cross_encoder_model_dir == tmp_path / "models/v1/medcpt-cross"


def test_engine_requires_load_before_analysis(tmp_path: Path) -> None:
    engine = EvidenceGapEngine(
        BackendConfig(workspace_root=tmp_path, provider="deepseek"),
        resources=FakeResources(),  # type: ignore[arg-type]
    )

    with pytest.raises(EvidenceGapError, match=r"load\(\)"):
        engine.analyze_statement(statement="claim", run_name="demo")


def test_engine_reuses_loaded_resources_and_returns_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = BackendConfig(workspace_root=tmp_path, provider="deepseek")
    resources = FakeResources()
    engine = EvidenceGapEngine(
        config,
        resources=resources,  # type: ignore[arg-type]
    )
    engine.load(validate_resources=False)
    engine.load(validate_resources=False)

    artifact_dir = config.artifact_root / "demo"
    presentation_path = artifact_dir / "output/presentation_bundle.json"
    presentation_path.parent.mkdir(parents=True)
    presentation_path.write_text(
        '{"contract_id":"phase077.presentation-bundle.v1"}\n',
        encoding="utf-8",
    )

    def fake_run(*args: object, **kwargs: object) -> dict[str, object]:
        assert kwargs["runtime_resources"] is resources
        assert kwargs["stage_configs"] == config.llm_stages
        assert kwargs["pipeline_config"] == config.pipeline
        assert kwargs["resolved_config_snapshot"] == config.safe_dict()
        return {
            "artifact_dir": str(artifact_dir.relative_to(tmp_path)),
            "presentation_bundle_path": str(presentation_path.relative_to(tmp_path)),
            "presentation_bundle": {
                "contract_id": "phase077.presentation-bundle.v1"
            },
        }

    monkeypatch.setattr("evidencegap_backend.engine.run_statement_pipeline", fake_run)
    monkeypatch.setattr(
        "evidencegap_backend.engine.validate_statement_pipeline_artifact",
        lambda path: {"status": "PASS"},
    )

    result = engine.analyze_statement(
        statement="Vitamin D supplementation prevents respiratory infections.",
        run_name="demo",
    )

    assert resources.load_count == 1
    assert resources.analysis_runs == 1
    assert engine.runtime_status["resource_ids"]["fake"] == id(resources)
    assert result.artifact_dir == artifact_dir
    assert result.presentation_bundle_path == presentation_path
    assert result.presentation_bundle["contract_id"] == (
        "phase077.presentation-bundle.v1"
    )

    engine.close()
    assert not engine.loaded


def test_engine_uses_configured_default_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = BackendConfig(
        workspace_root=tmp_path,
        provider="deepseek",
        default_language="繁體中文（台灣）",
    )
    resources = FakeResources()
    engine = EvidenceGapEngine(
        config,
        resources=resources,  # type: ignore[arg-type]
    )
    engine.load(validate_resources=False)
    artifact_dir = config.artifact_root / "default-language"
    presentation_path = artifact_dir / "output/presentation_bundle.json"
    presentation_path.parent.mkdir(parents=True)
    presentation_path.write_text('{}\n', encoding="utf-8")

    def fake_run(*args: object, **kwargs: object) -> dict[str, object]:
        assert kwargs["language"] == "繁體中文（台灣）"
        return {
            "artifact_dir": str(artifact_dir.relative_to(tmp_path)),
            "presentation_bundle_path": str(presentation_path.relative_to(tmp_path)),
            "presentation_bundle": {},
        }

    monkeypatch.setattr("evidencegap_backend.engine.run_statement_pipeline", fake_run)
    monkeypatch.setattr(
        "evidencegap_backend.engine.validate_statement_pipeline_artifact",
        lambda path: {"status": "PASS"},
    )
    result = engine.analyze_statement(statement="claim", run_name="default-language")
    assert result.presentation_bundle == {}


def test_dense_runtime_metadata_uses_configured_nprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evidencegap_backend.config import PipelineConfig
    from evidencegap_backend.resources import RuntimeResources

    config = BackendConfig(
        workspace_root=tmp_path,
        provider="deepseek",
        pipeline=PipelineConfig(dense_nprobe=37),
    )
    resources = RuntimeResources(config)

    class FakeBM25:
        article_ids = ["pmid:1"]

    class FakeEncoderSpec:
        query_model = tmp_path / "query-model"

    class FakeEncoder:
        spec = FakeEncoderSpec()

    class FakeIndex:
        ntotal = 1

    class FakeBackend:
        manifest = {
            "model_key": "medcpt",
            "index": {"sha256": "index-sha"},
            "article_embedding_manifest": "embedding_manifest.json",
        }
        index = FakeIndex()
        nprobe = 37

    resources.bm25 = FakeBM25()  # type: ignore[assignment]
    resources.expected_article_input_sha256 = "article-input-sha"

    monkeypatch.setattr(
        "evidencegap_backend.resources.sha256_file",
        lambda path: "index-sha" if Path(path).name == "index.faiss" else "sha",
    )
    monkeypatch.setattr(
        "evidencegap_backend.resources.load_json",
        lambda path: {"article_input_sha256": "article-input-sha"},
    )
    monkeypatch.setattr(
        "evidencegap_backend.resources.model_fingerprint",
        lambda spec, article: "model-sha",
    )

    metadata = resources._validate_dense_runtime(  # noqa: SLF001
        "medcpt",
        FakeEncoder(),  # type: ignore[arg-type]
        FakeBackend(),  # type: ignore[arg-type]
    )

    assert metadata["requested_nprobe"] == 37
    assert metadata["actual_nprobe"] == 37
