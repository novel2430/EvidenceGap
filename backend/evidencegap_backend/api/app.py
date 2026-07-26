from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from evidencegap_backend.config import BackendConfig
from evidencegap_backend.engine import EvidenceGapEngine
from evidencegap_backend.api.config import (
    ApiConfig,
    backend_config_from_env,
    load_config_document,
)
from evidencegap_backend.api.run_manager import (
    EngineProtocol,
    RunManager,
    RunQueueFullError,
)
from evidencegap_backend.api.run_store import JsonRunStore, RunNotFoundError
from evidencegap_backend.api.schemas import (
    HealthResponse,
    RunAcceptedResponse,
    RunCreateRequest,
    RunStatusResponse,
)


def create_app(
    *,
    backend_config: BackendConfig | None = None,
    api_config: ApiConfig | None = None,
    engine: EngineProtocol | None = None,
) -> FastAPI:
    document = (
        None
        if backend_config is not None and api_config is not None
        else load_config_document()
    )
    runtime_config = backend_config or backend_config_from_env(document=document)
    http_config = api_config or ApiConfig.from_env(runtime_config, document=document)
    runtime_engine = engine or EvidenceGapEngine(runtime_config)
    store = JsonRunStore(http_config.run_store_root)
    manager = RunManager(
        engine=runtime_engine,
        store=store,
        max_queue_size=http_config.max_queue_size,
        validate_resources=http_config.validate_resources,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.start()
        app.state.evidencegap_engine = runtime_engine
        app.state.run_manager = manager
        try:
            yield
        finally:
            manager.close()

    app = FastAPI(
        title="EvidenceGap API",
        version="0.4.0",
        lifespan=lifespan,
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    if http_config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(http_config.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict[str, Any]:
        return {"status": "ok", **manager.status()}

    @app.post(
        "/api/v1/runs",
        response_model=RunAcceptedResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_run(
        request: RunCreateRequest,
        response: Response,
    ) -> dict[str, Any]:
        if len(request.statement) > http_config.max_statement_chars:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "statement exceeds the configured maximum of "
                    f"{http_config.max_statement_chars} characters"
                ),
            )
        try:
            record = manager.submit(
                statement=request.statement,
                language=request.language or runtime_config.default_language,
            )
        except RunQueueFullError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        response.headers["Location"] = f"/api/v1/runs/{record['run_id']}"
        return {
            "run_id": record["run_id"],
            "status": record["status"],
            "created_at": record["created_at"],
        }

    @app.get(
        "/api/v1/runs/{run_id}",
        response_model=RunStatusResponse,
    )
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return manager.get(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="run not found",
            ) from exc

    return app
