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

RUN_STATUS_SCHEMA_VERSION = "1.1.0"
LOCALIZATION_STATUS_SCHEMA_VERSION = "1.0.0"
_RUN_ID_RE = re.compile(r"^run_[0-9a-f]{32}$")
_LOCALIZATION_ID_RE = re.compile(r"^loc_[0-9a-f]{32}$")
_RUN_STAGE_INDEX = {
    "statement_decomposition": 1,
    "claim_analysis": 2,
    "statement_bundle": 3,
    "inference_gap_analysis": 4,
    "output_generation": 5,
}


class RunNotFoundError(EvidenceGapError):
    pass


class LocalizationNotFoundError(EvidenceGapError):
    pass


class InvalidRunCursorError(EvidenceGapError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise RunNotFoundError(f"Unknown run: {run_id}")
    return run_id


def _validate_localization_id(localization_id: str) -> str:
    if not _LOCALIZATION_ID_RE.fullmatch(localization_id):
        raise LocalizationNotFoundError(
            f"Unknown localization: {localization_id}"
        )
    return localization_id


def _statement_preview(statement: str, *, limit: int = 240) -> str:
    compact = " ".join(statement.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def _result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        return {}
    evidence_states = summary.get("evidence_states")
    claim_inference_integrity = summary.get("claim_inference_integrity")
    inference_step_integrity = summary.get("inference_step_integrity")
    gaps = summary.get("gaps")
    return {
        "total_claims": int(summary.get("total_claims", 0)),
        "evidence_states": (
            {str(key): int(value) for key, value in evidence_states.items()}
            if isinstance(evidence_states, Mapping)
            else {}
        ),
        "claim_inference_integrity": (
            {
                str(key): int(value)
                for key, value in claim_inference_integrity.items()
            }
            if isinstance(claim_inference_integrity, Mapping)
            else {}
        ),
        "total_inference_steps": int(summary.get("total_inference_steps", 0)),
        "inference_step_integrity": (
            {
                str(key): int(value)
                for key, value in inference_step_integrity.items()
            }
            if isinstance(inference_step_integrity, Mapping)
            else {}
        ),
        "gaps": (
            {str(key): int(value) for key, value in gaps.items()}
            if isinstance(gaps, Mapping)
            else {}
        ),
        "articles": int(summary.get("articles", 0)),
        "evidence": int(summary.get("evidence", 0)),
    }


class JsonRunStore:
    """Atomic filesystem store for API runs and derived localizations."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _run_dir(self, run_id: str) -> Path:
        return self.root / _validate_run_id(run_id)

    def _status_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "status.json"

    def _request_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "request.json"

    def _localizations_dir(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "localizations"

    def _localization_dir(self, run_id: str, localization_id: str) -> Path:
        return self._localizations_dir(run_id) / _validate_localization_id(
            localization_id
        )

    def _localization_status_path(
        self, run_id: str, localization_id: str
    ) -> Path:
        return self._localization_dir(run_id, localization_id) / "status.json"

    def localization_artifact_root(self, run_id: str) -> Path:
        root = self._run_dir(run_id) / "localization_artifacts"
        root.mkdir(parents=True, exist_ok=True)
        return root

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
                self._request_path(run_id),
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
                "progress": None,
                "execution_summary": None,
                "summary": None,
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

    def update_progress(
        self,
        run_id: str,
        *,
        stage: str,
        stage_index: int,
        total_stages: int,
        message: str,
        completed_units: int | None = None,
        total_units: int | None = None,
    ) -> None:
        expected_index = _RUN_STAGE_INDEX.get(stage)
        if (
            expected_index is None
            or stage_index != expected_index
            or total_stages != len(_RUN_STAGE_INDEX)
        ):
            raise EvidenceGapError("Invalid run progress stage position")
        if completed_units is not None and completed_units < 0:
            raise EvidenceGapError("completed_units cannot be negative")
        if total_units is not None and total_units < 0:
            raise EvidenceGapError("total_units cannot be negative")
        if (
            completed_units is not None
            and total_units is not None
            and completed_units > total_units
        ):
            raise EvidenceGapError("completed_units cannot exceed total_units")
        self._update(
            run_id,
            progress={
                "stage": stage,
                "stage_index": stage_index,
                "total_stages": total_stages,
                "message": message,
                "completed_units": completed_units,
                "total_units": total_units,
                "updated_at": _now(),
            },
        )

    def mark_succeeded(
        self,
        run_id: str,
        *,
        result: Mapping[str, Any],
        artifact_dir: Path,
        presentation_bundle_path: Path,
        execution_summary: Mapping[str, Any] | None = None,
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
                summary=_result_summary(result),
                execution_summary=(
                    dict(execution_summary)
                    if isinstance(execution_summary, Mapping)
                    else None
                ),
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

    def _load_run_record(self, run_id: str) -> dict[str, Any]:
        path = self._status_path(run_id)
        if not path.is_file():
            raise RunNotFoundError(f"Unknown run: {run_id}")
        record = load_json(path)
        if not isinstance(record, dict):
            raise EvidenceGapError(f"Invalid API run status: {path}")
        return record

    def get(
        self,
        run_id: str,
        *,
        include_result: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._load_run_record(run_id)
            public = {
                "run_id": record["run_id"],
                "status": record["status"],
                "language": record["language"],
                "created_at": record["created_at"],
                "started_at": record.get("started_at"),
                "finished_at": record.get("finished_at"),
                "progress": record.get("progress"),
                "execution_summary": record.get("execution_summary"),
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

    def get_internal(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._load_run_record(run_id))

    def get_request(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            path = self._request_path(run_id)
            if not path.is_file():
                raise RunNotFoundError(f"Unknown run: {run_id}")
            value = load_json(path)
            if not isinstance(value, dict):
                raise EvidenceGapError(f"Invalid API run request: {path}")
            return value

    def get_result(self, run_id: str) -> dict[str, Any]:
        value = self.get(run_id, include_result=True)
        if value["status"] != "succeeded" or not isinstance(value["result"], dict):
            raise EvidenceGapError(f"Run is not complete: {run_id}")
        return dict(value["result"])

    def list_runs(
        self,
        *,
        limit: int,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        if limit <= 0 or limit > 100:
            raise EvidenceGapError("Run history limit must be between 1 and 100")
        with self._lock:
            records: list[dict[str, Any]] = []
            for path in self.root.glob("run_*/status.json"):
                try:
                    value = load_json(path)
                except EvidenceGapError:
                    continue
                if isinstance(value, dict) and _RUN_ID_RE.fullmatch(
                    str(value.get("run_id") or "")
                ):
                    records.append(value)
            records.sort(
                key=lambda row: (
                    str(row.get("created_at") or ""),
                    str(row.get("run_id") or ""),
                ),
                reverse=True,
            )
            start = 0
            if cursor is not None:
                if not _RUN_ID_RE.fullmatch(cursor):
                    raise InvalidRunCursorError("Invalid run history cursor")
                positions = [
                    index
                    for index, row in enumerate(records)
                    if row.get("run_id") == cursor
                ]
                if not positions:
                    raise InvalidRunCursorError("Unknown run history cursor")
                start = positions[0] + 1
            page = records[start : start + limit]
            has_more = start + limit < len(records)
            items: list[dict[str, Any]] = []
            for record in page:
                run_id = str(record["run_id"])
                try:
                    request_value = load_json(self._request_path(run_id))
                except EvidenceGapError:
                    request_value = {}
                request = request_value if isinstance(request_value, Mapping) else {}
                execution = record.get("execution_summary")
                total_seconds = None
                if isinstance(execution, Mapping):
                    raw = execution.get("total_seconds")
                    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                        total_seconds = float(raw)
                items.append(
                    {
                        "run_id": run_id,
                        "statement_preview": _statement_preview(
                            str(request.get("statement") or "")
                        ),
                        "language": str(record.get("language") or ""),
                        "status": str(record.get("status") or "failed"),
                        "created_at": record.get("created_at"),
                        "started_at": record.get("started_at"),
                        "finished_at": record.get("finished_at"),
                        "total_seconds": total_seconds,
                        "summary": record.get("summary"),
                        "error": record.get("error"),
                    }
                )
            return {
                "runs": items,
                "next_cursor": (
                    str(page[-1]["run_id"]) if has_more and page else None
                ),
            }

    def create_localization(
        self,
        *,
        run_id: str,
        localization_id: str,
        language: str,
    ) -> dict[str, Any]:
        with self._lock:
            source = self._load_run_record(run_id)
            if source.get("status") != "succeeded":
                raise EvidenceGapError(
                    "Localization requires a successfully completed source run"
                )
            target = self._localization_dir(run_id, localization_id)
            if target.exists():
                raise EvidenceGapError(
                    f"Localization already exists: {localization_id}"
                )
            target.mkdir(parents=True)
            created_at = _now()
            atomic_write_json(
                target / "request.json",
                {
                    "schema_version": LOCALIZATION_STATUS_SCHEMA_VERSION,
                    "localization_id": localization_id,
                    "source_run_id": run_id,
                    "language": language,
                    "created_at": created_at,
                },
            )
            record = {
                "schema_version": LOCALIZATION_STATUS_SCHEMA_VERSION,
                "localization_id": localization_id,
                "source_run_id": run_id,
                "language": language,
                "status": "queued",
                "created_at": created_at,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "result_path": None,
                "artifact_dir": None,
                "presentation_bundle_path": None,
            }
            atomic_write_json(
                self._localization_status_path(run_id, localization_id), record
            )
            return self.get_localization(
                run_id, localization_id, include_result=False
            )

    def _load_localization_record(
        self, run_id: str, localization_id: str
    ) -> dict[str, Any]:
        self._load_run_record(run_id)
        path = self._localization_status_path(run_id, localization_id)
        if not path.is_file():
            raise LocalizationNotFoundError(
                f"Unknown localization: {localization_id}"
            )
        value = load_json(path)
        if not isinstance(value, dict):
            raise EvidenceGapError(f"Invalid localization status: {path}")
        return value

    def _update_localization(
        self, run_id: str, localization_id: str, **changes: Any
    ) -> None:
        with self._lock:
            path = self._localization_status_path(run_id, localization_id)
            value = self._load_localization_record(run_id, localization_id)
            value.update(changes)
            atomic_write_json(path, value)

    def mark_localization_running(
        self, run_id: str, localization_id: str
    ) -> None:
        self._update_localization(
            run_id,
            localization_id,
            status="running",
            started_at=_now(),
            finished_at=None,
            error=None,
        )

    def mark_localization_succeeded(
        self,
        run_id: str,
        localization_id: str,
        *,
        result: Mapping[str, Any],
        artifact_dir: Path,
        presentation_bundle_path: Path,
    ) -> None:
        with self._lock:
            directory = self._localization_dir(run_id, localization_id)
            result_path = directory / "result.json"
            atomic_write_json(result_path, dict(result))
            self._update_localization(
                run_id,
                localization_id,
                status="succeeded",
                finished_at=_now(),
                error=None,
                result_path=relative_path(directory, result_path),
                artifact_dir=artifact_dir.resolve().as_posix(),
                presentation_bundle_path=(
                    presentation_bundle_path.resolve().as_posix()
                ),
            )

    def mark_localization_failed(
        self,
        run_id: str,
        localization_id: str,
        *,
        code: str,
        message: str,
    ) -> None:
        self._update_localization(
            run_id,
            localization_id,
            status="failed",
            finished_at=_now(),
            error={"code": code, "message": message},
        )

    def get_localization(
        self,
        run_id: str,
        localization_id: str,
        *,
        include_result: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            value = self._load_localization_record(run_id, localization_id)
            public = {
                "localization_id": value["localization_id"],
                "source_run_id": value["source_run_id"],
                "language": value["language"],
                "status": value["status"],
                "created_at": value["created_at"],
                "started_at": value.get("started_at"),
                "finished_at": value.get("finished_at"),
                "error": value.get("error"),
                "result": None,
            }
            result_path_value = value.get("result_path")
            if (
                include_result
                and value.get("status") == "succeeded"
                and isinstance(result_path_value, str)
                and result_path_value
            ):
                path = self._localization_dir(
                    run_id, localization_id
                ) / result_path_value
                result = load_json(path)
                if not isinstance(result, dict):
                    raise EvidenceGapError(
                        f"Invalid localization result artifact: {path}"
                    )
                public["result"] = result
            return public

    def list_localizations(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            self._load_run_record(run_id)
            rows: list[dict[str, Any]] = []
            root = self._localizations_dir(run_id)
            if root.is_dir():
                for path in root.glob("loc_*/status.json"):
                    try:
                        value = load_json(path)
                    except EvidenceGapError:
                        continue
                    if isinstance(value, dict):
                        rows.append(value)
            rows.sort(
                key=lambda row: (
                    str(row.get("created_at") or ""),
                    str(row.get("localization_id") or ""),
                ),
                reverse=True,
            )
            return {
                "localizations": [
                    self.get_localization(
                        run_id,
                        str(row["localization_id"]),
                        include_result=False,
                    )
                    for row in rows
                ]
            }

    def recover_interrupted(self) -> int:
        """A restarted in-process service cannot resume queued/running work."""

        recovered = 0
        with self._lock:
            for path in sorted(self.root.glob("run_*/status.json")):
                try:
                    record = load_json(path)
                except EvidenceGapError:
                    continue
                if not isinstance(record, dict):
                    continue
                run_id = str(record.get("run_id", ""))
                if record.get("status") in {"queued", "running"}:
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
                for localization_path in sorted(
                    path.parent.glob("localizations/loc_*/status.json")
                ):
                    try:
                        localization = load_json(localization_path)
                    except EvidenceGapError:
                        continue
                    if (
                        not isinstance(localization, dict)
                        or localization.get("status") not in {"queued", "running"}
                    ):
                        continue
                    localization_id = str(
                        localization.get("localization_id") or ""
                    )
                    try:
                        self.mark_localization_failed(
                            run_id,
                            localization_id,
                            code="SERVICE_RESTARTED",
                            message=(
                                "The API process restarted before this localization completed."
                            ),
                        )
                    except (RunNotFoundError, LocalizationNotFoundError):
                        continue
                    recovered += 1
        return recovered
