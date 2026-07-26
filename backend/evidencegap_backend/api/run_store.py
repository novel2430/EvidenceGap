from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evidencegap_backend.common import (
    EvidenceGapError,
    atomic_write_json,
    load_json,
    relative_path,
)

RUN_STATUS_SCHEMA_VERSION = "1.0.0"
_RUN_ID_RE = re.compile(r"^run_[0-9a-f]{32}$")


class RunNotFoundError(EvidenceGapError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise RunNotFoundError(f"Unknown run: {run_id}")
    return run_id


class JsonRunStore:
    """Small atomic filesystem store for API status and final presentation data."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _run_dir(self, run_id: str) -> Path:
        return self.root / _validate_run_id(run_id)

    def _status_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "status.json"

    def create(
        self,
        *,
        run_id: str,
        statement: str,
        language: str,
    ) -> dict[str, Any]:
        with self._lock:
            run_dir = self._run_dir(run_id)
            if run_dir.exists():
                raise EvidenceGapError(f"API run already exists: {run_id}")
            run_dir.mkdir(parents=True)
            created_at = _now()
            atomic_write_json(
                run_dir / "request.json",
                {
                    "schema_version": RUN_STATUS_SCHEMA_VERSION,
                    "run_id": run_id,
                    "statement": statement,
                    "language": language,
                    "created_at": created_at,
                },
            )
            record = {
                "schema_version": RUN_STATUS_SCHEMA_VERSION,
                "run_id": run_id,
                "status": "queued",
                "language": language,
                "created_at": created_at,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "result_path": None,
                "artifact_dir": None,
                "presentation_bundle_path": None,
            }
            atomic_write_json(self._status_path(run_id), record)
            return self.get(run_id, include_result=False)

    def mark_running(self, run_id: str) -> None:
        self._update(
            run_id,
            status="running",
            started_at=_now(),
            finished_at=None,
            error=None,
        )

    def mark_succeeded(
        self,
        run_id: str,
        *,
        result: Mapping[str, Any],
        artifact_dir: Path,
        presentation_bundle_path: Path,
    ) -> None:
        with self._lock:
            run_dir = self._run_dir(run_id)
            result_path = run_dir / "result.json"
            atomic_write_json(result_path, dict(result))
            self._update(
                run_id,
                status="succeeded",
                finished_at=_now(),
                error=None,
                result_path=relative_path(run_dir, result_path),
                artifact_dir=artifact_dir.resolve().as_posix(),
                presentation_bundle_path=(
                    presentation_bundle_path.resolve().as_posix()
                ),
            )

    def mark_failed(self, run_id: str, *, code: str, message: str) -> None:
        self._update(
            run_id,
            status="failed",
            finished_at=_now(),
            error={"code": code, "message": message},
        )

    def _update(self, run_id: str, **changes: Any) -> None:
        with self._lock:
            path = self._status_path(run_id)
            record = load_json(path)
            if not isinstance(record, dict):
                raise EvidenceGapError(f"Invalid API run status: {path}")
            record.update(changes)
            atomic_write_json(path, record)

    def get(
        self,
        run_id: str,
        *,
        include_result: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            path = self._status_path(run_id)
            if not path.is_file():
                raise RunNotFoundError(f"Unknown run: {run_id}")
            record = load_json(path)
            if not isinstance(record, dict):
                raise EvidenceGapError(f"Invalid API run status: {path}")
            public = {
                "run_id": record["run_id"],
                "status": record["status"],
                "language": record["language"],
                "created_at": record["created_at"],
                "started_at": record.get("started_at"),
                "finished_at": record.get("finished_at"),
                "error": record.get("error"),
                "result": None,
            }
            result_path_value = record.get("result_path")
            if (
                include_result
                and record.get("status") == "succeeded"
                and isinstance(result_path_value, str)
                and result_path_value
            ):
                result_path = self._run_dir(run_id) / result_path_value
                result = load_json(result_path)
                if not isinstance(result, dict):
                    raise EvidenceGapError(
                        f"Invalid API result artifact: {result_path}"
                    )
                public["result"] = result
            return public

    def recover_interrupted(self) -> int:
        """A restarted in-process service cannot resume queued/running work."""

        recovered = 0
        with self._lock:
            for path in sorted(self.root.glob("run_*/status.json")):
                record = load_json(path)
                if not isinstance(record, dict):
                    continue
                if record.get("status") not in {"queued", "running"}:
                    continue
                run_id = str(record.get("run_id", ""))
                try:
                    self.mark_failed(
                        run_id,
                        code="SERVICE_RESTARTED",
                        message=(
                            "The API process restarted before this run completed."
                        ),
                    )
                except RunNotFoundError:
                    continue
                recovered += 1
        return recovered
