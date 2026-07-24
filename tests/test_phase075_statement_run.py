from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evidencegap.common import EvidenceGapError, atomic_write_json, relative_path  # noqa: E402
from evidencegap.pipeline.statement_run import (  # noqa: E402
    STATEMENT_RUN_CONTRACT_ID,
    run_statement_pipeline,
    validate_statement_pipeline_artifact,
)
from run_v1_phase075 import build_parser  # noqa: E402


class StatementRunTests(unittest.TestCase):
    def test_run_cli_defaults_match_stage_workloads(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--statement",
                "fixture",
                "--run-name",
                "fixture",
                "--provider",
                "deepseek",
            ]
        )
        self.assertFalse(args.decomposition_thinking)
        self.assertIsNone(args.analysis_thinking)
        self.assertEqual(args.request_batch_size, 1)
        self.assertEqual(args.max_tokens, 8192)

        compatibility_args = parser.parse_args(
            [
                "run",
                "--statement",
                "fixture",
                "--run-name",
                "fixture",
                "--provider",
                "deepseek",
                "--thinking",
            ]
        )
        self.assertTrue(compatibility_args.analysis_thinking)

    def _run_fixture(
        self,
        root: Path,
        *,
        analysis_status: str = "completed",
        provider: str = "deepseek",
        decomposition_thinking: bool = False,
        analysis_thinking: bool | None = None,
    ) -> Path:
        artifact_root = root / "artifacts/statement_run"
        statement_id = "statement_fixture"

        def fake_decomposition(*args: object, **kwargs: object) -> dict[str, object]:
            target = Path(kwargs["artifact_root"]) / str(kwargs["run_name"])
            target.mkdir(parents=True)
            atomic_write_json(target / "decomposition.json", {"statement_id": statement_id})
            atomic_write_json(target / "run_manifest.json", {"stage": "decomposition"})
            return {
                "status": "PASS",
                "statement_id": statement_id,
                "claims": 1,
                "inference_steps": 0,
                "empty_claims": False,
            }

        def fake_analysis(*args: object, **kwargs: object) -> dict[str, object]:
            target = Path(kwargs["artifact_root"]) / str(kwargs["run_name"])
            target.mkdir(parents=True)
            atomic_write_json(
                target / "request.json",
                {
                    "decomposition_artifact_dir": relative_path(
                        root, Path(kwargs["decomposition_artifact_dir"])
                    )
                },
            )
            atomic_write_json(target / "run_manifest.json", {"stage": "analysis"})
            return {
                "status": "PASS",
                "statement_id": statement_id,
                "analysis_status": analysis_status,
                "total_claims": 1,
                "completed_claims": int(analysis_status == "completed"),
                "failed_claims": int(analysis_status != "completed"),
            }

        def fake_bundle(*args: object, **kwargs: object) -> dict[str, object]:
            target = Path(kwargs["artifact_root"]) / str(kwargs["run_name"])
            target.mkdir(parents=True)
            atomic_write_json(target / "statement_bundle.json", {"fixture": True})
            atomic_write_json(
                target / "run_manifest.json",
                {
                    "source": {
                        "statement_analysis_artifact_dir": relative_path(
                            root, Path(kwargs["statement_analysis_artifact_dir"])
                        )
                    }
                },
            )
            return {
                "status": "PASS",
                "statement_id": statement_id,
                "analysis_status": analysis_status,
                "total_claims": 1,
                "completed_claims": int(analysis_status == "completed"),
                "failed_claims": int(analysis_status != "completed"),
                "articles": 10 if analysis_status == "completed" else 0,
                "evidence": 3 if analysis_status == "completed" else 0,
            }

        with (
            patch(
                "evidencegap.pipeline.statement_run.run_statement_decomposition",
                side_effect=fake_decomposition,
            ) as decompose,
            patch(
                "evidencegap.pipeline.statement_run.run_statement_analysis",
                side_effect=fake_analysis,
            ) as analyze,
            patch(
                "evidencegap.pipeline.statement_run.run_statement_bundle",
                side_effect=fake_bundle,
            ) as bundle,
        ):
            result = run_statement_pipeline(
                root,
                statement="維生素D能預防呼吸道感染。",
                run_name="full",
                provider=provider,
                artifact_root=artifact_root,
                decomposition_thinking=decomposition_thinking,
                analysis_thinking=analysis_thinking,
            )

        self.assertEqual(decompose.call_count, 1)
        self.assertEqual(analyze.call_count, 1)
        self.assertEqual(bundle.call_count, 1)
        self.assertEqual(
            decompose.call_args.kwargs["thinking"],
            decomposition_thinking if provider == "deepseek" else False,
        )
        expected_analysis_thinking = (
            (True if analysis_thinking is None else analysis_thinking)
            if provider == "deepseek"
            else False
        )
        self.assertEqual(
            analyze.call_args.kwargs["thinking"], expected_analysis_thinking
        )
        self.assertEqual(result["analysis_status"], analysis_status)
        self.assertEqual(result["status"], analysis_status.upper())
        self.assertEqual(result["artifact_status"], "PASS")
        self.assertEqual(
            result["statement_bundle_path"],
            "artifacts/statement_run/full/bundle/statement_bundle.json",
        )
        return artifact_root / "full"

    def test_run_orchestrates_three_stages_and_writes_top_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)
            artifact_dir = self._run_fixture(root)
            manifest = json.loads(
                (artifact_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["contract_id"], STATEMENT_RUN_CONTRACT_ID)
            self.assertEqual(
                set(manifest["stages"]), {"decomposition", "analysis", "bundle"}
            )
            self.assertEqual(manifest["counts"]["articles"], 10)
            self.assertFalse(manifest["execution"]["decomposition_thinking"])
            self.assertTrue(manifest["execution"]["analysis_thinking"])

    def test_run_allows_stage_specific_thinking_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)
            artifact_dir = self._run_fixture(
                root,
                decomposition_thinking=True,
                analysis_thinking=False,
            )
            manifest = json.loads(
                (artifact_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["execution"]["decomposition_thinking"])
            self.assertFalse(manifest["execution"]["analysis_thinking"])

    def test_anthropic_run_defaults_to_no_thinking(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)
            artifact_dir = self._run_fixture(root, provider="anthropic")
            manifest = json.loads(
                (artifact_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(manifest["execution"]["decomposition_thinking"])
            self.assertIsNone(manifest["execution"]["analysis_thinking"])

    def test_validate_run_checks_nested_artifacts_and_stage_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)
            artifact_dir = self._run_fixture(root)
            with (
                patch(
                    "evidencegap.pipeline.statement_run.validate_statement_decomposition_artifact",
                    return_value={"status": "PASS", "statement_id": "statement_fixture"},
                ),
                patch(
                    "evidencegap.pipeline.statement_run.validate_statement_analysis_artifact",
                    return_value={
                        "status": "PASS",
                        "statement_id": "statement_fixture",
                        "analysis_status": "completed",
                    },
                ),
                patch(
                    "evidencegap.pipeline.statement_run.validate_statement_bundle_artifact",
                    return_value={
                        "status": "PASS",
                        "statement_id": "statement_fixture",
                        "analysis_status": "completed",
                        "total_claims": 1,
                        "completed_claims": 1,
                        "failed_claims": 0,
                        "articles": 10,
                        "evidence": 3,
                    },
                ),
            ):
                validation = validate_statement_pipeline_artifact(artifact_dir)
            self.assertEqual(validation["status"], "PASS")
            self.assertEqual(validation["checksums"], "PASS")
            self.assertEqual(validation["articles"], 10)

    def test_validate_run_rejects_analysis_linked_to_another_decomposition(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)
            artifact_dir = self._run_fixture(root)
            atomic_write_json(
                artifact_dir / "analysis/request.json",
                {"decomposition_artifact_dir": "artifacts/other"},
            )
            with (
                patch(
                    "evidencegap.pipeline.statement_run.validate_statement_decomposition_artifact",
                    return_value={"status": "PASS", "statement_id": "statement_fixture"},
                ),
                patch(
                    "evidencegap.pipeline.statement_run.validate_statement_analysis_artifact",
                    return_value={
                        "status": "PASS",
                        "statement_id": "statement_fixture",
                        "analysis_status": "completed",
                    },
                ),
                patch(
                    "evidencegap.pipeline.statement_run.validate_statement_bundle_artifact",
                    return_value={
                        "status": "PASS",
                        "statement_id": "statement_fixture",
                        "analysis_status": "completed",
                        "total_claims": 1,
                        "completed_claims": 1,
                        "failed_claims": 0,
                        "articles": 10,
                        "evidence": 3,
                    },
                ),
            ):
                with self.assertRaisesRegex(
                    EvidenceGapError, "analysis source mismatch"
                ):
                    validate_statement_pipeline_artifact(artifact_dir)


if __name__ == "__main__":
    unittest.main()
