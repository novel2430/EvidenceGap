from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from evidencegap_backend import BackendConfig, StatementAnalysisResult
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

    def analyze_statement(
        self,
        *,
        statement: str,
        run_name: str,
        language: str = "English",
        force: bool = False,
        validate: bool = True,
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
        artifact_dir = self.tmp_path / "pipeline" / run_name
        bundle_path = artifact_dir / "output/presentation_bundle.json"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "contract_id": "phase077.presentation-bundle.v1",
            "statement": {"display_text": statement},
            "output_language": language,
        }
        self.analysis_runs += 1
        return StatementAnalysisResult(
            run={"run_name": run_name},
            artifact_dir=artifact_dir,
            presentation_bundle_path=bundle_path,
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
