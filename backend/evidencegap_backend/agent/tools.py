from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from langchain_core.tools import StructuredTool

from evidencegap_backend.agent.schemas import SearchAttempt, SearchEvidenceInput
from evidencegap_backend.agent.workspace import normalize_query
from evidencegap_backend.common import EvidenceGapError
from evidencegap_backend.pipeline.analysis import run_analysis

SearchExecutor = Callable[[SearchEvidenceInput], SearchAttempt | Mapping[str, Any]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_analysis_result(
    value: Mapping[str, Any], *, request: SearchEvidenceInput, started_at: str
) -> SearchAttempt:
    graph = value.get("graph_bundle")
    nodes = graph.get("nodes", []) if isinstance(graph, Mapping) else []
    articles = [
        node
        for node in nodes
        if isinstance(node, Mapping) and node.get("node_type") == "article"
    ]
    article_ids = [
        str(node["article_id"]) for node in articles if node.get("article_id")
    ]
    titles = [str(node.get("label") or "") for node in articles if node.get("label")]
    support = sum(node.get("stance") == "support" for node in articles)
    refute = sum(node.get("stance") == "refute" for node in articles)
    insufficient = sum(node.get("stance") == "insufficient" for node in articles)
    direct = support + refute
    verdict = str(value.get("verdict") or "insufficient")
    utility = direct * 100 + (20 if verdict != "insufficient" else 0) + len(article_ids)
    return SearchAttempt(
        attempt_id=request.attempt_id,
        claim_id=request.claim_id,
        query=request.query,
        normalized_query=normalize_query(request.query),
        artifact_dir=str(value.get("artifact_dir") or ""),
        graph_bundle_path=str(value.get("graph_bundle_path") or ""),
        verdict=verdict,
        article_counts={
            "total": len(article_ids),
            "support": support,
            "refute": refute,
            "insufficient": insufficient,
        },
        article_ids=article_ids,
        new_article_ids=article_ids,
        top_article_titles=titles[:3],
        direct_evidence_articles=direct,
        utility_score=utility,
        status="successful",
        started_at=started_at,
        finished_at=_now(),
    )


def create_analysis_executor(
    *, root: Any, analysis_kwargs: Mapping[str, Any], attempts_root: Any
) -> SearchExecutor:
    def execute(request: SearchEvidenceInput) -> SearchAttempt:
        started = _now()
        try:
            result = run_analysis(
                root,
                claim=request.canonical_claim,
                retrieval_query=request.query,
                run_name=request.attempt_id,
                artifact_root=attempts_root / request.claim_id,
                **dict(analysis_kwargs),
            )
            return summarize_analysis_result(
                result, request=request, started_at=started
            )
        except Exception as exc:
            if not isinstance(exc, EvidenceGapError):
                exc = EvidenceGapError(str(exc))
            return SearchAttempt(
                attempt_id=request.attempt_id,
                claim_id=request.claim_id,
                query=request.query,
                normalized_query=normalize_query(request.query),
                status="failed",
                error=str(exc),
                started_at=started,
                finished_at=_now(),
            )

    return execute


def create_search_evidence_tool(executor: SearchExecutor) -> StructuredTool:
    def invoke(**kwargs: Any) -> dict[str, Any]:
        request = SearchEvidenceInput.model_validate(kwargs)
        result = executor(request)
        attempt = (
            result
            if isinstance(result, SearchAttempt)
            else SearchAttempt.model_validate(result)
        )
        return attempt.model_dump(mode="json")

    return StructuredTool.from_function(
        func=invoke,
        name="search_evidence",
        description="Run EvidenceGap's deterministic retrieval, reranking, article judge, aggregation, and graph pipeline for one canonical claim and retrieval query.",
        args_schema=SearchEvidenceInput,
    )
