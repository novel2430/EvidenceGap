from __future__ import annotations

import copy
import threading
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from evidencegap_backend import (
    BackendConfig,
    LocalizationResult,
    StatementAnalysisResult,
)
from evidencegap_backend.api import ApiConfig, create_app


class FakeEngine:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.loaded = False
        self.load_count = 0
        self.analysis_runs = 0
        self.closed = False
        self.load_thread_id: int | None = None
        self.analysis_thread_ids: list[int] = []
        self.calls: list[dict[str, Any]] = []
        self.localization_calls: list[dict[str, Any]] = []

    @property
    def runtime_status(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "load_count": self.load_count,
            "analysis_runs": self.analysis_runs,
        }

    def load(self, *, validate_resources: bool = True) -> None:
        self.loaded = True
        self.load_count += 1
        self.load_thread_id = threading.get_ident()

    def close(self) -> None:
        self.loaded = False
        self.closed = True

    @staticmethod
    def _bundle(statement: str, language: str) -> dict[str, Any]:
        return {
            "schema_version": "1.2.0",
            "contract_id": "phase077.presentation-bundle.v1",
            "output_language": language,
            "localized": language != "English",
            "analysis_context": {
                "scope": "retrieved_top_articles",
                "is_systematic_review": False,
                "is_clinical_recommendation": False,
                "is_final_medical_truth": False,
                "aggregation_method": "deterministic_article_count",
                "uses_confidence_weighting": False,
                "article_top_k": 10,
            },
            "statement": {
                "original_text": statement,
                "display_text": statement,
            },
            "claims": [
                {
                    "claim_id": "claim_1",
                    "canonical_claim_en": statement,
                    "display_text": statement,
                    "display_rationale": "One retrieved article supports the claim.",
                    "analysis_status": "completed",
                    "evidence_state": "SUPPORTED",
                    "argument_role": "STANDALONE",
                }
            ],
            "inference_steps": [],
            "articles": [
                {
                    "article_node_id": "article_1",
                    "claim_id": "claim_1",
                    "article_id": "pmid:123",
                    "pmid": "123",
                    "rank": 1,
                    "title": "Vitamin D trial",
                    "display_title": "Vitamin D trial",
                    "rationale": "The article reports a direct result.",
                    "display_rationale": "The article reports a direct result.",
                    "stance": "support",
                    "confidence": 0.91,
                    "applicability": {
                        "population_or_species": "MATCH",
                        "intervention_or_exposure": "MATCH",
                        "comparator": "NOT_REPORTED",
                        "outcome": "MATCH",
                        "direction": "MATCH",
                        "timeframe": "NOT_REPORTED",
                        "causal_strength": "MATCH",
                        "prevention_treatment_scope": "NOT_APPLICABLE",
                    },
                    "applicability_issues": [],
                }
            ],
            "evidence": [
                {
                    "evidence_id": "evidence_1",
                    "claim_id": "claim_1",
                    "article_node_id": "article_1",
                    "section": "results",
                    "section_index": 1,
                    "sentence_index": 0,
                    "text": "Vitamin D reduced infections.",
                    "display_text": "Vitamin D reduced infections.",
                    "character_start": 17,
                    "character_end": 46,
                    "source_text_fingerprint": "f" * 64,
                }
            ],
            "summary": {
                "total_claims": 1,
                "evidence_states": {
                    "SUPPORTED": 1,
                    "REFUTED": 0,
                    "CONFLICTED": 0,
                    "INSUFFICIENT": 0,
                    "ERROR": 0,
                },
                "argument_roles": {
                    "PREMISE": 0,
                    "INTERMEDIATE": 0,
                    "CONCLUSION": 0,
                    "STANDALONE": 1,
                },
                "total_inference_steps": 0,
                "gaps": {"SCOPE_GAP": 0, "CAUSAL_GAP": 0},
                "articles": 1,
                "evidence": 1,
            },
        }

    def analyze_statement(
        self,
        *,
        statement: str,
        run_name: str,
        language: str = "English",
        force: bool = False,
        validate: bool = True,
        progress_callback=None,
    ) -> StatementAnalysisResult:
        self.analysis_thread_ids.append(threading.get_ident())
        self.calls.append(
            {
                "statement": statement,
                "run_name": run_name,
                "language": language,
                "force": force,
                "validate": validate,
            }
        )
        if progress_callback is not None:
            for index, stage in enumerate(
                (
                    "statement_decomposition",
                    "claim_analysis",
                    "statement_bundle",
                    "inference_gap_analysis",
                    "output_generation",
                ),
                start=1,
            ):
                progress_callback(
                    {
                        "stage": stage,
                        "stage_index": index,
                        "total_stages": 5,
                        "message": stage.replace("_", " "),
                        "completed_units": 1 if stage == "claim_analysis" else None,
                        "total_units": 1 if stage == "claim_analysis" else None,
                    }
                )
        artifact_dir = self.tmp_path / "pipeline" / run_name
        bundle_path = artifact_dir / "output/presentation_bundle.json"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = self._bundle(statement, language)
        self.analysis_runs += 1
        return StatementAnalysisResult(
            run={
                "run_name": run_name,
                "execution_summary": {
                    "total_seconds": 1.25,
                    "stages": {
                        "statement_decomposition": {"seconds": 0.1},
                        "claim_analysis": {"seconds": 0.8},
                        "statement_bundle": {"seconds": 0.05},
                        "inference_gap_analysis": {"seconds": 0.1},
                        "output_generation": {"seconds": 0.2},
                    },
                },
            },
            artifact_dir=artifact_dir,
            presentation_bundle_path=bundle_path,
            presentation_bundle=bundle,
        )

    def get_article_context(
        self,
        *,
        presentation_bundle: dict[str, Any],
        article_node_id: str,
    ) -> dict[str, Any]:
        if article_node_id != "article_1":
            raise RuntimeError("unknown article")
        canonical = "Vitamin D trial\n\nVitamin D reduced infections."
        return {
            "article_node_id": article_node_id,
            "article_id": "pmid:123",
            "claim_id": "claim_1",
            "pmid": "123",
            "title": "Vitamin D trial",
            "canonical_text": canonical,
            "source_text_fingerprint": "f" * 64,
            "fingerprint_verified": True,
            "sections": [
                {
                    "sentence_type": "title",
                    "section": "title",
                    "section_index": 0,
                    "character_start": 0,
                    "character_end": 15,
                },
                {
                    "sentence_type": "abstract",
                    "section": "results",
                    "section_index": 1,
                    "character_start": 17,
                    "character_end": len(canonical),
                },
            ],
            "evidence_spans": [
                {
                    "evidence_id": "evidence_1",
                    "claim_id": "claim_1",
                    "section": "results",
                    "section_index": 1,
                    "sentence_index": 0,
                    "character_start": 17,
                    "character_end": len(canonical),
                    "text": "Vitamin D reduced infections.",
                }
            ],
        }

    def localize_statement_run(
        self,
        *,
        artifact_dir: Path,
        localization_name: str,
        language: str,
        artifact_root: Path,
        force: bool = False,
        validate: bool = True,
    ) -> LocalizationResult:
        self.localization_calls.append(
            {
                "artifact_dir": artifact_dir,
                "localization_name": localization_name,
                "language": language,
            }
        )
        target = artifact_root / localization_name
        target.mkdir(parents=True, exist_ok=True)
        path = target / "presentation_bundle.json"
        bundle = self._bundle("維生素 D 可預防感染。", language)
        return LocalizationResult(
            run={"run_name": localization_name},
            artifact_dir=target,
            presentation_bundle_path=path,
            presentation_bundle=bundle,
        )


def _make_client(tmp_path: Path) -> tuple[TestClient, FakeEngine]:
    backend_config = BackendConfig(
        workspace_root=tmp_path,
        provider="deepseek",
    )
    api_config = ApiConfig(
        run_store_root=tmp_path / "api_runs",
        max_queue_size=2,
        validate_resources=False,
    )
    engine = FakeEngine(tmp_path)
    app = create_app(
        backend_config=backend_config,
        api_config=api_config,
        engine=engine,
    )
    return TestClient(app), engine


def _wait_for_terminal(client: TestClient, run_id: str) -> dict[str, Any]:
    for _ in range(100):
        response = client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"succeeded", "failed"}:
            return body
        time.sleep(0.01)
    raise AssertionError("run did not finish")


def test_lifespan_loads_one_engine_and_health_reports_runtime(
    tmp_path: Path,
) -> None:
    client, engine = _make_client(tmp_path)

    with client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "engine_loaded": True,
            "worker_alive": True,
            "active_run_id": None,
            "queued_runs": 0,
            "load_count": 1,
            "analysis_runs": 0,
        }

    assert engine.closed is True
    assert engine.load_count == 1


def test_submit_and_poll_returns_complete_presentation_bundle(
    tmp_path: Path,
) -> None:
    client, engine = _make_client(tmp_path)

    with client:
        accepted = client.post(
            "/api/v1/runs",
            json={
                "statement": "Vitamin D supplementation prevents infections.",
                "language": "English",
            },
        )
        assert accepted.status_code == 202
        assert accepted.headers["location"].startswith("/api/v1/runs/run_")
        run_id = accepted.json()["run_id"]

        finished = _wait_for_terminal(client, run_id)
        assert finished["status"] == "succeeded"
        assert finished["error"] is None
        assert finished["result"]["contract_id"] == (
            "phase077.presentation-bundle.v1"
        )
        assert finished["result"]["output_language"] == "English"

    assert engine.calls == [
        {
            "statement": "Vitamin D supplementation prevents infections.",
            "run_name": run_id,
            "language": "English",
            "force": False,
            "validate": True,
        }
    ]
    assert engine.load_thread_id == engine.analysis_thread_ids[0]


def test_unknown_run_and_blank_statement_are_rejected(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)

    with client:
        missing = client.get("/api/v1/runs/run_00000000000000000000000000000000")
        assert missing.status_code == 404

        invalid = client.post(
            "/api/v1/runs",
            json={"statement": "   ", "language": "English"},
        )
        assert invalid.status_code == 422


def test_missing_language_uses_backend_default(tmp_path: Path) -> None:
    backend_config = BackendConfig(
        workspace_root=tmp_path,
        provider="deepseek",
        default_language="繁體中文（台灣）",
    )
    api_config = ApiConfig(
        run_store_root=tmp_path / "api_runs",
        max_queue_size=2,
        validate_resources=False,
    )
    engine = FakeEngine(tmp_path)
    app = create_app(
        backend_config=backend_config,
        api_config=api_config,
        engine=engine,
    )

    with TestClient(app) as client:
        accepted = client.post(
            "/api/v1/runs",
            json={"statement": "Vitamin D supplementation prevents infections."},
        )
        finished = _wait_for_terminal(client, accepted.json()["run_id"])

    assert finished["language"] == "繁體中文（台灣）"
    assert engine.calls[0]["language"] == "繁體中文（台灣）"


def test_phase82_run_history_progress_article_and_exports(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)

    with client:
        run_ids: list[str] = []
        for statement in (
            "Claim one is supported.",
            "Claim two is supported.",
        ):
            accepted = client.post(
                "/api/v1/runs",
                json={"statement": statement, "language": "English"},
            )
            assert accepted.status_code == 202
            run_id = accepted.json()["run_id"]
            run_ids.append(run_id)
            finished = _wait_for_terminal(client, run_id)
            assert finished["status"] == "succeeded"
            assert finished["progress"] == {
                "stage": "output_generation",
                "stage_index": 5,
                "total_stages": 5,
                "message": "output generation",
                "completed_units": None,
                "total_units": None,
                "updated_at": finished["progress"]["updated_at"],
            }
            assert finished["execution_summary"]["total_seconds"] == 1.25

        first_page = client.get("/api/v1/runs?limit=1")
        assert first_page.status_code == 200
        page_body = first_page.json()
        assert len(page_body["runs"]) == 1
        assert page_body["runs"][0]["run_id"] == run_ids[-1]
        assert page_body["runs"][0]["summary"]["evidence_states"]["SUPPORTED"] == 1
        assert page_body["next_cursor"] == run_ids[-1]

        second_page = client.get(
            "/api/v1/runs",
            params={"limit": 1, "cursor": page_body["next_cursor"]},
        )
        assert second_page.status_code == 200
        assert second_page.json()["runs"][0]["run_id"] == run_ids[0]
        assert second_page.json()["next_cursor"] is None

        article = client.get(
            f"/api/v1/runs/{run_ids[0]}/articles/article_1"
        )
        assert article.status_code == 200
        assert article.json()["fingerprint_verified"] is True
        assert article.json()["canonical_text"].endswith(
            "Vitamin D reduced infections."
        )

        result_export = client.get(
            f"/api/v1/runs/{run_ids[0]}/exports/result.json"
        )
        assert result_export.status_code == 200
        assert "attachment" in result_export.headers["content-disposition"]
        assert result_export.json()["summary"]["total_claims"] == 1

        markdown_export = client.get(
            f"/api/v1/runs/{run_ids[0]}/exports/report.md"
        )
        assert markdown_export.status_code == 200
        assert "text/markdown" in markdown_export.headers["content-type"]
        assert "# EvidenceGap Analysis" in markdown_export.text
        assert "## Methodological Boundary" in markdown_export.text
        assert "Vitamin D trial" in markdown_export.text


def test_phase82_localization_variant_is_derived_and_immutable(
    tmp_path: Path,
) -> None:
    client, engine = _make_client(tmp_path)

    with client:
        accepted = client.post(
            "/api/v1/runs",
            json={
                "statement": "Vitamin D supplementation prevents infections.",
                "language": "English",
            },
        )
        run_id = accepted.json()["run_id"]
        source = _wait_for_terminal(client, run_id)
        assert source["result"]["output_language"] == "English"

        localization = client.post(
            f"/api/v1/runs/{run_id}/localizations",
            json={"language": "繁體中文（台灣）"},
        )
        assert localization.status_code == 202
        localization_id = localization.json()["localization_id"]
        assert localization.headers["location"].endswith(localization_id)

        for _ in range(100):
            localized = client.get(
                f"/api/v1/runs/{run_id}/localizations/{localization_id}"
            )
            assert localized.status_code == 200
            localized_body = localized.json()
            if localized_body["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("localization did not finish")

        assert localized_body["status"] == "succeeded"
        assert localized_body["result"]["output_language"] == "繁體中文（台灣）"
        assert localized_body["source_run_id"] == run_id

        listed = client.get(f"/api/v1/runs/{run_id}/localizations")
        assert listed.status_code == 200
        assert [
            row["localization_id"] for row in listed.json()["localizations"]
        ] == [localization_id]

        source_after = client.get(f"/api/v1/runs/{run_id}").json()
        assert source_after["result"]["output_language"] == "English"

    assert len(engine.localization_calls) == 1
    assert engine.localization_calls[0]["language"] == "繁體中文（台灣）"
