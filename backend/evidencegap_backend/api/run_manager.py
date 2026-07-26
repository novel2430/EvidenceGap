from __future__ import annotations

import logging
import queue
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from evidencegap_backend.common import EvidenceGapError
from evidencegap_backend.engine import StatementAnalysisResult
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
    ) -> StatementAnalysisResult: ...


@dataclass(frozen=True)
class _RunJob:
    run_id: str
    statement: str
    language: str


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
        self._queue: queue.Queue[_RunJob | object] = queue.Queue(
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
            if not self._accepting or not self.worker_alive:
                raise EvidenceGapError("EvidenceGap API worker is not running")
            if self._queue.full():
                raise RunQueueFullError("EvidenceGap API run queue is full")
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

    def get(self, run_id: str) -> dict[str, Any]:
        return self.store.get(run_id)

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
                    assert isinstance(item, _RunJob)
                    self._execute(item)
                finally:
                    self._queue.task_done()
        finally:
            with self._state_lock:
                self._active_run_id = None
            self.engine.close()

    def _execute(self, job: _RunJob) -> None:
        with self._state_lock:
            self._active_run_id = job.run_id
        self.store.mark_running(job.run_id)
        try:
            result = self.engine.analyze_statement(
                statement=job.statement,
                run_name=job.run_id,
                language=job.language,
                force=False,
                validate=True,
            )
            self.store.mark_succeeded(
                job.run_id,
                result=result.presentation_bundle,
                artifact_dir=Path(result.artifact_dir),
                presentation_bundle_path=Path(
                    result.presentation_bundle_path
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
