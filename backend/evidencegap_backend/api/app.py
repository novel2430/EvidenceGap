from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from evidencegap_backend.common import EvidenceGapError
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
from evidencegap_backend.api.run_store import (
    InvalidRunCursorError,
    JsonRunStore,
    LocalizationNotFoundError,
    RunNotFoundError,
)
from evidencegap_backend.api.schemas import (
    ArticleContextResponse,
    HealthResponse,
    LocalizationAcceptedResponse,
    LocalizationCreateRequest,
    LocalizationListResponse,
    LocalizationStatusResponse,
    RunAcceptedResponse,
    RunCreateRequest,
    RunListResponse,
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
        version="0.5.0",
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

    @app.get("/api/v1/runs", response_model=RunListResponse)
    def list_runs(
        limit: int = Query(default=20, ge=1, le=100),
        cursor: str | None = Query(default=None),
    ) -> dict[str, Any]:
        try:
            return manager.list_runs(limit=limit, cursor=cursor)
        except InvalidRunCursorError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

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

    @app.get(
        "/api/v1/runs/{run_id}/articles/{article_node_id}",
        response_model=ArticleContextResponse,
    )
    def get_article_context(
        run_id: str, article_node_id: str
    ) -> dict[str, Any]:
        try:
            return manager.get_article_context(
                run_id=run_id,
                article_node_id=article_node_id,
            )
        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="run not found",
            ) from exc
        except EvidenceGapError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

    @app.get("/api/v1/runs/{run_id}/exports/result.json")
    def export_result(run_id: str) -> Response:
        try:
            result = manager.export_result(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="run not found",
            ) from exc
        except EvidenceGapError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return Response(
            content=json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="evidencegap-{run_id}.json"'
                )
            },
        )

    @app.get("/api/v1/runs/{run_id}/exports/report.md")
    def export_markdown(run_id: str) -> Response:
        try:
            report = manager.export_markdown(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="run not found",
            ) from exc
        except EvidenceGapError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return Response(
            content=report,
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="evidencegap-{run_id}.md"'
                )
            },
        )

    @app.post(
        "/api/v1/runs/{run_id}/localizations",
        response_model=LocalizationAcceptedResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_localization(
        run_id: str,
        request: LocalizationCreateRequest,
        response: Response,
    ) -> dict[str, Any]:
        try:
            record = manager.submit_localization(
                run_id=run_id,
                language=request.language,
            )
        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="run not found",
            ) from exc
        except RunQueueFullError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except EvidenceGapError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        location = (
            f"/api/v1/runs/{run_id}/localizations/"
            f"{record['localization_id']}"
        )
        response.headers["Location"] = location
        return {
            "localization_id": record["localization_id"],
            "source_run_id": record["source_run_id"],
            "language": record["language"],
            "status": record["status"],
            "created_at": record["created_at"],
        }

    @app.get(
        "/api/v1/runs/{run_id}/localizations",
        response_model=LocalizationListResponse,
    )
    def list_localizations(run_id: str) -> dict[str, Any]:
        try:
            return manager.list_localizations(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="run not found",
            ) from exc

    @app.get(
        "/api/v1/runs/{run_id}/localizations/{localization_id}",
        response_model=LocalizationStatusResponse,
    )
    def get_localization(
        run_id: str, localization_id: str
    ) -> dict[str, Any]:
        try:
            return manager.get_localization(run_id, localization_id)
        except RunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="run not found",
            ) from exc
        except LocalizationNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="localization not found",
            ) from exc

    return app
