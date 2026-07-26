from __future__ import annotations

import logging
import queue
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from evidencegap_backend.common import EvidenceGapError
from evidencegap_backend.engine import LocalizationResult, StatementAnalysisResult
from evidencegap_backend.output.report import render_markdown_report
from evidencegap_backend.api.run_store import JsonRunStore

LOGGER = logging.getLogger(__name__)
_STOP = object()


class RunQueueFullError(EvidenceGapError):
    pass


class EngineProtocol(Protocol):
    @property
    def loaded(self) -> bool: ...

    @property
    def runtime_status(self) -> dict[str, Any]: ...

    def load(self, *, validate_resources: bool = True) -> None: ...

    def close(self) -> None: ...

    def analyze_statement(
        self,
        *,
        statement: str,
        run_name: str,
        language: str | None = None,
        force: bool = False,
        validate: bool = True,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> StatementAnalysisResult: ...

    def get_article_context(
        self,
        *,
        presentation_bundle: Mapping[str, Any],
        article_node_id: str,
    ) -> dict[str, Any]: ...

    def localize_statement_run(
        self,
        *,
        artifact_dir: Path,
        localization_name: str,
        language: str,
        artifact_root: Path,
        force: bool = False,
        validate: bool = True,
    ) -> LocalizationResult: ...


@dataclass(frozen=True)
class _RunJob:
    run_id: str
    statement: str
    language: str


@dataclass(frozen=True)
class _LocalizationJob:
    source_run_id: str
    localization_id: str
    language: str


Job = _RunJob | _LocalizationJob


class RunManager:
    """One in-process worker that serially calls only EvidenceGapEngine."""

    def __init__(
        self,
        *,
        engine: EngineProtocol,
        store: JsonRunStore,
        max_queue_size: int,
        validate_resources: bool = True,
    ) -> None:
        self.engine = engine
        self.store = store
        self._validate_resources = validate_resources
        self._queue: queue.Queue[Job | object] = queue.Queue(
            maxsize=max_queue_size
        )
        self._thread: threading.Thread | None = None
        self._startup_event = threading.Event()
        self._startup_error: BaseException | None = None
        self._accepting = False
        self._active_run_id: str | None = None
        self._state_lock = threading.RLock()
        self._submit_lock = threading.Lock()

    @property
    def worker_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def active_run_id(self) -> str | None:
        with self._state_lock:
            return self._active_run_id

    @property
    def queued_runs(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        with self._state_lock:
            if self.worker_alive:
                return
            self.store.recover_interrupted()
            self._startup_event.clear()
            self._startup_error = None
            self._accepting = True
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="evidencegap-runtime-worker",
                daemon=True,
            )
            self._thread.start()
        self._startup_event.wait()
        if self._startup_error is not None:
            error = self._startup_error
            self._thread = None
            raise error

    def submit(self, *, statement: str, language: str) -> dict[str, Any]:
        with self._submit_lock:
            self._ensure_accepting()
            run_id = f"run_{uuid.uuid4().hex}"
            record = self.store.create(
                run_id=run_id,
                statement=statement,
                language=language,
            )
            self._queue.put_nowait(
                _RunJob(
                    run_id=run_id,
                    statement=statement,
                    language=language,
                )
            )
            return record

    def submit_localization(
        self,
        *,
        run_id: str,
        language: str,
    ) -> dict[str, Any]:
        with self._submit_lock:
            self._ensure_accepting()
            localization_id = f"loc_{uuid.uuid4().hex}"
            record = self.store.create_localization(
                run_id=run_id,
                localization_id=localization_id,
                language=language,
            )
            self._queue.put_nowait(
                _LocalizationJob(
                    source_run_id=run_id,
                    localization_id=localization_id,
                    language=language,
                )
            )
            return record

    def _ensure_accepting(self) -> None:
        if not self._accepting or not self.worker_alive:
            raise EvidenceGapError("EvidenceGap API worker is not running")
        if self._queue.full():
            raise RunQueueFullError("EvidenceGap API run queue is full")

    def get(self, run_id: str) -> dict[str, Any]:
        return self.store.get(run_id)

    def list_runs(self, *, limit: int, cursor: str | None) -> dict[str, Any]:
        return self.store.list_runs(limit=limit, cursor=cursor)

    def get_article_context(
        self,
        *,
        run_id: str,
        article_node_id: str,
    ) -> dict[str, Any]:
        result = self.store.get_result(run_id)
        return self.engine.get_article_context(
            presentation_bundle=result,
            article_node_id=article_node_id,
        )

    def export_result(self, run_id: str) -> dict[str, Any]:
        return self.store.get_result(run_id)

    def export_markdown(self, run_id: str) -> str:
        result = self.store.get_result(run_id)
        record = self.store.get_internal(run_id)
        execution = record.get("execution_summary")
        return render_markdown_report(
            result,
            run_id=run_id,
            execution_summary=(
                execution if isinstance(execution, Mapping) else None
            ),
        )

    def get_localization(
        self, run_id: str, localization_id: str
    ) -> dict[str, Any]:
        return self.store.get_localization(run_id, localization_id)

    def list_localizations(self, run_id: str) -> dict[str, Any]:
        return self.store.list_localizations(run_id)

    def status(self) -> dict[str, Any]:
        runtime = self.engine.runtime_status
        return {
            "engine_loaded": bool(self.engine.loaded),
            "worker_alive": self.worker_alive,
            "active_run_id": self.active_run_id,
            "queued_runs": self.queued_runs,
            "load_count": int(runtime.get("load_count", 0)),
            "analysis_runs": int(runtime.get("analysis_runs", 0)),
        }

    def close(self) -> None:
        with self._submit_lock:
            self._accepting = False
            while True:
                try:
                    queued = self._queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(queued, _RunJob):
                    self.store.mark_failed(
                        queued.run_id,
                        code="SERVICE_SHUTDOWN",
                        message=(
                            "The API service stopped before this queued run started."
                        ),
                    )
                elif isinstance(queued, _LocalizationJob):
                    self.store.mark_localization_failed(
                        queued.source_run_id,
                        queued.localization_id,
                        code="SERVICE_SHUTDOWN",
                        message=(
                            "The API service stopped before this queued localization started."
                        ),
                    )
                self._queue.task_done()
            if self.worker_alive:
                self._queue.put(_STOP)
        thread = self._thread
        if thread is not None:
            thread.join()
        self._thread = None

    def _worker_loop(self) -> None:
        try:
            self.engine.load(validate_resources=self._validate_resources)
        except BaseException as exc:
            self._startup_error = exc
            self._accepting = False
            try:
                self.engine.close()
            finally:
                self._startup_event.set()
            return
        self._startup_event.set()

        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        return
                    if isinstance(item, _RunJob):
                        self._execute_run(item)
                    elif isinstance(item, _LocalizationJob):
                        self._execute_localization(item)
                    else:
                        raise AssertionError("Unexpected EvidenceGap job")
                finally:
                    self._queue.task_done()
        finally:
            with self._state_lock:
                self._active_run_id = None
            self.engine.close()

    def _execute_run(self, job: _RunJob) -> None:
        with self._state_lock:
            self._active_run_id = job.run_id
        self.store.mark_running(job.run_id)

        def progress(value: Mapping[str, Any]) -> None:
            self.store.update_progress(
                job.run_id,
                stage=str(value["stage"]),
                stage_index=int(value["stage_index"]),
                total_stages=int(value["total_stages"]),
                message=str(value["message"]),
                completed_units=(
                    None
                    if value.get("completed_units") is None
                    else int(value["completed_units"])
                ),
                total_units=(
                    None
                    if value.get("total_units") is None
                    else int(value["total_units"])
                ),
            )

        try:
            result = self.engine.analyze_statement(
                statement=job.statement,
                run_name=job.run_id,
                language=job.language,
                force=False,
                validate=True,
                progress_callback=progress,
            )
            self.store.mark_succeeded(
                job.run_id,
                result=result.presentation_bundle,
                artifact_dir=Path(result.artifact_dir),
                presentation_bundle_path=Path(
                    result.presentation_bundle_path
                ),
                execution_summary=(
                    result.run.get("execution_summary")
                    if isinstance(result.run.get("execution_summary"), Mapping)
                    else None
                ),
            )
        except EvidenceGapError as exc:
            LOGGER.exception("EvidenceGap run failed: %s", job.run_id)
            self.store.mark_failed(
                job.run_id,
                code="PIPELINE_FAILED",
                message=str(exc),
            )
        except Exception:
            LOGGER.exception("Unexpected EvidenceGap run failure: %s", job.run_id)
            self.store.mark_failed(
                job.run_id,
                code="INTERNAL_ERROR",
                message="The analysis failed because of an internal error.",
            )
        finally:
            with self._state_lock:
                self._active_run_id = None

    def _execute_localization(self, job: _LocalizationJob) -> None:
        with self._state_lock:
            self._active_run_id = job.source_run_id
        self.store.mark_localization_running(
            job.source_run_id, job.localization_id
        )
        try:
            source = self.store.get_internal(job.source_run_id)
            artifact_value = source.get("artifact_dir")
            if not isinstance(artifact_value, str) or not artifact_value:
                raise EvidenceGapError("Source run artifact directory is unavailable")
            result = self.engine.localize_statement_run(
                artifact_dir=Path(artifact_value),
                localization_name=job.localization_id,
                language=job.language,
                artifact_root=self.store.localization_artifact_root(
                    job.source_run_id
                ),
                force=False,
                validate=True,
            )
            self.store.mark_localization_succeeded(
                job.source_run_id,
                job.localization_id,
                result=result.presentation_bundle,
                artifact_dir=result.artifact_dir,
                presentation_bundle_path=result.presentation_bundle_path,
            )
        except EvidenceGapError as exc:
            LOGGER.exception(
                "EvidenceGap localization failed: %s/%s",
                job.source_run_id,
                job.localization_id,
            )
            self.store.mark_localization_failed(
                job.source_run_id,
                job.localization_id,
                code="LOCALIZATION_FAILED",
                message=str(exc),
            )
        except Exception:
            LOGGER.exception(
                "Unexpected EvidenceGap localization failure: %s/%s",
                job.source_run_id,
                job.localization_id,
            )
            self.store.mark_localization_failed(
                job.source_run_id,
                job.localization_id,
                code="INTERNAL_ERROR",
                message="The localization failed because of an internal error.",
            )
        finally:
            with self._state_lock:
                self._active_run_id = None
