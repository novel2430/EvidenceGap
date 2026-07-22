from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Iterator, Mapping, Sequence

from evidencegap.common import (
    EvidenceGapError,
    atomic_write_json,
    relative_path,
    sha256_file,
)
from evidencegap.evaluation import run_article_retrieval

TRACKS = ("independent", "origin", "overall")
CANDIDATE_KINDS = ("judged", "open")
DEFAULT_CORPUS_DIR = Path("artifacts/v1/article_corpus")
DEFAULT_INPUT_RUN_DIR = Path("artifacts/v1/article_retrieval_runs")
DEFAULT_CANDIDATE_DIR = Path("artifacts/v1/reranking/candidates")
DEFAULT_RUN_DIR = Path("artifacts/v1/reranking/runs")
DEFAULT_REPORT_DIR = Path("reports/v1")
FUSION_SCHEMA_VERSION = "1.0.0"
_ALIAS_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class SourceRun:
    alias: str
    run_name: str


@dataclass(frozen=True)
class TrecRow:
    article_id: str
    rank: int
    score: float


def _duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing duckdb dependency. Install requirements/v1-phase04.txt"
        ) from exc
    return duckdb


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing pyarrow dependency. Install requirements/v1-phase04.txt"
        ) from exc
    return pa, pq


def _quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def safe_run_name(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    ).strip("_")
    if not safe:
        raise EvidenceGapError("run_name must contain at least one safe character")
    return safe


def parse_source_run(value: str) -> SourceRun:
    if "=" not in value:
        raise EvidenceGapError(f"Invalid --source {value!r}; expected ALIAS=RUN_NAME")
    alias, run_name = value.split("=", 1)
    alias = alias.strip()
    run_name = run_name.strip()
    if not _ALIAS_RE.fullmatch(alias):
        raise EvidenceGapError(
            f"Invalid source alias {alias!r}; use letters, digits, and underscores"
        )
    if not run_name:
        raise EvidenceGapError(f"Missing run name in --source {value!r}")
    return SourceRun(alias=alias, run_name=run_name)


def parse_weight(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise EvidenceGapError(f"Invalid --weight {value!r}; expected ALIAS=FLOAT")
    alias, raw = value.split("=", 1)
    alias = alias.strip()
    try:
        weight = float(raw)
    except ValueError as exc:
        raise EvidenceGapError(f"Invalid fusion weight in {value!r}") from exc
    if not math.isfinite(weight) or weight <= 0:
        raise EvidenceGapError(f"Fusion weight must be finite and > 0: {value!r}")
    return alias, weight


def trec_path(
    run_dir: Path,
    *,
    split: str,
    candidate_kind: str,
    track: str,
    run_name: str,
) -> Path:
    return run_dir / f"{split}_{candidate_kind}_{track}_{run_name}.trec"


def iter_trec_groups(
    path: Path,
    *,
    allowed_claim_ids: set[str] | None = None,
) -> Iterator[tuple[str, list[TrecRow]]]:
    if not path.exists():
        raise EvidenceGapError(f"Missing source TREC run: {path}")

    current_claim: str | None = None
    rows: list[TrecRow] = []
    seen_claims: set[str] = set()

    def emit() -> tuple[str, list[TrecRow]] | None:
        nonlocal rows
        if current_claim is None:
            return None
        if current_claim in seen_claims:
            raise EvidenceGapError(
                f"TREC query appears in multiple non-contiguous groups: "
                f"{current_claim} in {path}"
            )
        seen_claims.add(current_claim)
        expected = list(range(1, len(rows) + 1))
        actual = [row.rank for row in rows]
        if actual != expected:
            raise EvidenceGapError(
                f"Non-contiguous ranks for {current_claim} in {path}: "
                f"expected 1..{len(rows)}"
            )
        article_ids = [row.article_id for row in rows]
        if len(article_ids) != len(set(article_ids)):
            raise EvidenceGapError(
                f"Duplicate article within query {current_claim} in {path}"
            )
        result = (current_claim, rows)
        rows = []
        return result

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split()
            if len(fields) != 6:
                raise EvidenceGapError(
                    f"Malformed TREC row in {path}:{line_number}; expected 6 fields"
                )
            claim_id, q0, article_id, raw_rank, raw_score, _tag = fields
            if q0 != "Q0":
                raise EvidenceGapError(
                    f"Malformed TREC row in {path}:{line_number}; field 2 must be Q0"
                )
            try:
                rank = int(raw_rank)
                score = float(raw_score)
            except ValueError as exc:
                raise EvidenceGapError(
                    f"Malformed rank/score in {path}:{line_number}"
                ) from exc
            if rank <= 0 or not math.isfinite(score):
                raise EvidenceGapError(f"Invalid rank/score in {path}:{line_number}")

            if current_claim is not None and claim_id != current_claim:
                group = emit()
                if group is not None and (
                    allowed_claim_ids is None or group[0] in allowed_claim_ids
                ):
                    yield group
            current_claim = claim_id
            rows.append(TrecRow(article_id=article_id, rank=rank, score=score))

    group = emit()
    if group is not None and (
        allowed_claim_ids is None or group[0] in allowed_claim_ids
    ):
        yield group


def _load_allowed_claim_ids(
    corpus_dir: Path,
    *,
    split: str,
    max_queries: int | None,
) -> list[str]:
    claims = corpus_dir / "claims.parquet"
    judgments = corpus_dir / "judgments.parquet"
    for path in (claims, judgments):
        if not path.exists():
            raise EvidenceGapError(f"Missing Phase 02 corpus asset: {path}")

    duckdb = _duckdb()
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""
            SELECT DISTINCT CAST(c.claim_id AS VARCHAR) AS claim_id
            FROM read_parquet('{_quote(claims)}') c
            JOIN read_parquet('{_quote(judgments)}') j USING(claim_id)
            WHERE c.split = ? AND j.eligible_for_qrels
            ORDER BY claim_id
            """,
            [split],
        ).fetchall()
    finally:
        connection.close()
    values = [str(row[0]) for row in rows]
    if max_queries is not None:
        if max_queries <= 0:
            raise EvidenceGapError("max_queries must be positive")
        values = values[:max_queries]
    return values


def _expected_track_query_counts(
    corpus_dir: Path,
    *,
    selected_claim_ids: Sequence[str],
) -> dict[str, int]:
    if not selected_claim_ids:
        return {track: 0 for track in TRACKS}
    pa, _pq = _pyarrow()
    selected = pa.table({"claim_id": list(selected_claim_ids)})
    judgments = corpus_dir / "judgments.parquet"
    duckdb = _duckdb()
    connection = duckdb.connect()
    try:
        connection.register("selected_claims", selected)
        row = connection.execute(
            f"""
            SELECT
                count(DISTINCT CASE
                    WHEN j.relevance_grade > 0 AND NOT j.is_origin_source
                    THEN CAST(j.claim_id AS VARCHAR)
                END) AS independent_queries,
                count(DISTINCT CASE
                    WHEN j.relevance_grade > 0 AND j.is_origin_source
                    THEN CAST(j.claim_id AS VARCHAR)
                END) AS origin_queries,
                count(DISTINCT CASE
                    WHEN j.relevance_grade > 0
                    THEN CAST(j.claim_id AS VARCHAR)
                END) AS overall_queries
            FROM read_parquet('{_quote(judgments)}') j
            JOIN selected_claims s
              ON CAST(j.claim_id AS VARCHAR) = s.claim_id
            WHERE j.eligible_for_qrels
            """
        ).fetchone()
    finally:
        connection.close()
    return {
        "independent": int(row[0]),
        "origin": int(row[1]),
        "overall": int(row[2]),
    }


def _merge_source_groups(
    paths: Mapping[str, Path],
    *,
    allowed_claim_ids: set[str],
) -> Iterator[tuple[str, dict[str, list[TrecRow]]]]:
    aliases = list(paths)
    iterators = [
        iter_trec_groups(paths[alias], allowed_claim_ids=allowed_claim_ids)
        for alias in aliases
    ]
    for groups in zip_longest(*iterators):
        if any(group is None for group in groups):
            counts = ", ".join(
                f"{alias}={'done' if group is None else group[0]}"
                for alias, group in zip(aliases, groups)
            )
            raise EvidenceGapError(
                "Source runs have different query coverage after filtering: " + counts
            )
        assert all(group is not None for group in groups)
        claim_ids = [group[0] for group in groups if group is not None]
        if len(set(claim_ids)) != 1:
            detail = ", ".join(
                f"{alias}={claim_id}" for alias, claim_id in zip(aliases, claim_ids)
            )
            raise EvidenceGapError(
                "Source runs are not query-aligned. Regenerate them with the same "
                f"split/query limit: {detail}"
            )
        yield (
            claim_ids[0],
            {
                alias: group[1]
                for alias, group in zip(aliases, groups)
                if group is not None
            },
        )


def _fuse_one_query(
    source_rows: Mapping[str, Sequence[TrecRow]],
    *,
    aliases: Sequence[str],
    method: str,
    rrf_k: int,
    weights: Mapping[str, float],
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for source_index, alias in enumerate(aliases):
        for row in source_rows[alias]:
            item = candidates.setdefault(
                row.article_id,
                {
                    "article_id": row.article_id,
                    "source_mask": 0,
                    "source_count": 0,
                    "best_rank": row.rank,
                    "rank_sum": 0,
                    "rrf_score": 0.0,
                },
            )
            if f"{alias}_rank" in item:
                raise EvidenceGapError(
                    f"Duplicate article {row.article_id} in source {alias}"
                )
            item[f"{alias}_rank"] = row.rank
            item[f"{alias}_score"] = row.score
            item["source_mask"] |= 1 << source_index
            item["source_count"] += 1
            item["best_rank"] = min(item["best_rank"], row.rank)
            item["rank_sum"] += row.rank
            item["rrf_score"] += weights[alias] / (rrf_k + row.rank)

    values = list(candidates.values())
    if method == "rrf":
        values.sort(
            key=lambda row: (
                -float(row["rrf_score"]),
                -int(row["source_count"]),
                int(row["best_rank"]),
                int(row["rank_sum"]),
                str(row["article_id"]),
            )
        )
        for row in values:
            row["fusion_score"] = float(row["rrf_score"])
    elif method == "union":
        values.sort(
            key=lambda row: (
                int(row["best_rank"]),
                -int(row["source_count"]),
                int(row["rank_sum"]),
                str(row["article_id"]),
            )
        )
        # Union has no natural cross-retriever score. The deterministic ordering is
        # best rank, source agreement, then rank sum; the synthetic score merely
        # preserves that order in TREC format.
        total = len(values)
        for index, row in enumerate(values):
            row["fusion_score"] = float(total - index)
    else:
        raise EvidenceGapError(f"Unsupported fusion method: {method}")

    for rank, row in enumerate(values, start=1):
        row["fusion_rank"] = rank
        for alias in aliases:
            row.setdefault(f"{alias}_rank", None)
            row.setdefault(f"{alias}_score", None)
    return values


def _candidate_schema(aliases: Sequence[str]) -> Any:
    pa, _pq = _pyarrow()
    fields = [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("candidate_kind", pa.string(), nullable=False),
        pa.field("track", pa.string(), nullable=False),
        pa.field("claim_id", pa.string(), nullable=False),
        pa.field("article_id", pa.string(), nullable=False),
        pa.field("fusion_method", pa.string(), nullable=False),
        pa.field("source_mask", pa.int64(), nullable=False),
        pa.field("source_count", pa.int16(), nullable=False),
        pa.field("best_rank", pa.int32(), nullable=False),
        pa.field("rank_sum", pa.int32(), nullable=False),
        pa.field("rrf_score", pa.float64(), nullable=False),
        pa.field("fusion_score", pa.float64(), nullable=False),
        pa.field("fusion_rank", pa.int32(), nullable=False),
    ]
    for alias in aliases:
        fields.extend(
            [
                pa.field(f"{alias}_rank", pa.int32()),
                pa.field(f"{alias}_score", pa.float64()),
            ]
        )
    return pa.schema(fields)


def _write_candidate_batch(
    writer: Any, schema: Any, rows: list[dict[str, Any]]
) -> None:
    if not rows:
        return
    pa, _pq = _pyarrow()
    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    rows.clear()


def _write_trec_row(
    handle: Any,
    *,
    claim_id: str,
    article_id: str,
    rank: int,
    score: float,
    run_name: str,
) -> None:
    handle.write(f"{claim_id} Q0 {article_id} {rank} {score:.10f} {run_name}\n")


def _source_paths(
    input_run_dir: Path,
    *,
    sources: Sequence[SourceRun],
    split: str,
    candidate_kind: str,
    track: str,
) -> dict[str, Path]:
    return {
        source.alias: trec_path(
            input_run_dir,
            split=split,
            candidate_kind=candidate_kind,
            track=track,
            run_name=source.run_name,
        )
        for source in sources
    }


def _ensure_outputs_available(paths: Iterable[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        formatted = "\n".join(f"- {path}" for path in existing[:10])
        raise EvidenceGapError(
            "Phase 04 output already exists. Use --force to replace it:\n" + formatted
        )
    if force:
        for path in existing:
            path.unlink()


def _union_ceiling(
    candidate_path: Path,
    corpus_dir: Path,
    *,
    split: str,
) -> dict[str, dict[str, float | int | None]]:
    duckdb = _duckdb()
    claims = corpus_dir / "claims.parquet"
    judgments = corpus_dir / "judgments.parquet"
    connection = duckdb.connect()
    results: dict[str, dict[str, float | int | None]] = {}
    try:
        for track in TRACKS:
            condition = {
                "independent": "NOT j.is_origin_source",
                "origin": "j.is_origin_source",
                "overall": "TRUE",
            }[track]
            row = connection.execute(
                f"""
                WITH selected AS (
                    SELECT DISTINCT claim_id
                    FROM read_parquet('{_quote(candidate_path)}')
                    WHERE candidate_kind = 'open' AND track = ?
                ), positives AS (
                    SELECT
                        CAST(c.claim_id AS VARCHAR) AS claim_id,
                        CAST(j.article_id AS VARCHAR) AS article_id
                    FROM read_parquet('{_quote(claims)}') c
                    JOIN read_parquet('{_quote(judgments)}') j USING(claim_id)
                    JOIN selected s ON CAST(c.claim_id AS VARCHAR) = s.claim_id
                    WHERE c.split = ?
                      AND j.eligible_for_qrels
                      AND j.relevance_grade > 0
                      AND {condition}
                ), eligible AS (
                    SELECT claim_id, count(*) AS positive_count
                    FROM positives
                    GROUP BY claim_id
                ), found AS (
                    SELECT p.claim_id, count(*) AS found_count
                    FROM positives p
                    JOIN (
                        SELECT DISTINCT claim_id, article_id
                        FROM read_parquet('{_quote(candidate_path)}')
                        WHERE candidate_kind = 'open' AND track = ?
                    ) c USING(claim_id, article_id)
                    GROUP BY p.claim_id
                )
                SELECT
                    count(*) AS queries,
                    avg(coalesce(found_count, 0)::DOUBLE / positive_count) AS recall,
                    avg(CASE WHEN coalesce(found_count, 0) > 0 THEN 1.0 ELSE 0.0 END) AS hitrate
                FROM eligible
                LEFT JOIN found USING(claim_id)
                """,
                [track, split, track],
            ).fetchone()
            queries = int(row[0])
            results[track] = {
                "eligible_queries": queries,
                "known_positive_recall_full_union": (
                    round(float(row[1]), 8) if row[1] is not None else None
                ),
                "hitrate_full_union": (
                    round(float(row[2]), 8) if row[2] is not None else None
                ),
            }
    finally:
        connection.close()
    return results


def _phase04_markdown(report: Mapping[str, Any]) -> str:
    fusion = report["fusion"]
    lines = [
        f"# Phase 04 Fusion — {report['split']}",
        "",
        f"Run: `{report['run_name']}`  ",
        f"Method: `{fusion['method']}`  ",
        f"Sources: `{', '.join(source['alias'] for source in fusion['sources'])}`",
        "",
        "| Track | MRR | nDCG@5 | Top-1 positive | Pairwise acc. | KP Recall@10 | KP Recall@100 | Full-union recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ceilings = fusion["full_union_ceiling"]
    for track in TRACKS:
        data = report["tracks"][track]
        judged = data["judged_candidate_ranking"]
        opened = data["open_corpus_incomplete_qrels"]
        pairwise = judged.get("pairwise_ordering_accuracy")
        ceiling = ceilings[track].get("known_positive_recall_full_union")
        lines.append(
            "| {track} | {mrr:.4f} | {ndcg:.4f} | {top1:.4f} | {pairwise} | "
            "{r10:.4f} | {r100:.4f} | {ceiling} |".format(
                track=track,
                mrr=judged.get("mrr") or 0.0,
                ndcg=judged.get("ndcg@5") or 0.0,
                top1=judged.get("top1_positive_rate") or 0.0,
                pairwise="n/a" if pairwise is None else f"{pairwise:.4f}",
                r10=opened.get("known_positive_recall@10") or 0.0,
                r100=opened.get("known_positive_recall@100") or 0.0,
                ceiling="n/a" if ceiling is None else f"{ceiling:.4f}",
            )
        )
    lines.extend(
        [
            "",
            "The full-union column measures known-positive coverage before the fused ranking is truncated to Top-100.",
            "Independent-source remains the primary track. Open-corpus qrels are incomplete.",
            "",
        ]
    )
    return "\n".join(lines)


def run_fusion(
    root: Path,
    *,
    split: str,
    sources: Sequence[SourceRun],
    method: str = "rrf",
    rrf_k: int = 60,
    weights: Mapping[str, float] | None = None,
    input_run_dir: Path | None = None,
    corpus_dir: Path | None = None,
    candidate_dir: Path | None = None,
    run_dir: Path | None = None,
    report_dir: Path | None = None,
    run_name: str | None = None,
    top_k: int = 100,
    max_queries: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if split not in {"dev", "test"}:
        raise EvidenceGapError("split must be dev or test")
    if len(sources) < 2:
        raise EvidenceGapError("Fusion requires at least two --source runs")
    aliases = [source.alias for source in sources]
    if len(set(aliases)) != len(aliases):
        raise EvidenceGapError("Source aliases must be unique")
    if len(sources) > 63:
        raise EvidenceGapError("At most 63 source runs are supported by source_mask")
    if method not in {"rrf", "union"}:
        raise EvidenceGapError("method must be rrf or union")
    if rrf_k <= 0:
        raise EvidenceGapError("rrf_k must be positive")
    if top_k < 100:
        raise EvidenceGapError("top_k must be at least 100 for Recall@100")

    root = root.resolve()
    input_run_dir = (root / (input_run_dir or DEFAULT_INPUT_RUN_DIR)).resolve()
    corpus_dir = (root / (corpus_dir or DEFAULT_CORPUS_DIR)).resolve()
    candidate_dir = (root / (candidate_dir or DEFAULT_CANDIDATE_DIR)).resolve()
    run_dir = (root / (run_dir or DEFAULT_RUN_DIR)).resolve()
    report_dir = (root / (report_dir or DEFAULT_REPORT_DIR)).resolve()

    normalized_weights = {alias: 1.0 for alias in aliases}
    if weights:
        unknown = set(weights) - set(aliases)
        if unknown:
            raise EvidenceGapError(
                "Weights reference unknown source aliases: "
                + ", ".join(sorted(unknown))
            )
        normalized_weights.update(weights)

    default_name = f"{method}_{'_'.join(aliases)}" + (
        f"_k{rrf_k}" if method == "rrf" else ""
    )
    run_name = safe_run_name(run_name or default_name)
    candidate_path = candidate_dir / f"{split}_{run_name}.parquet"
    manifest_path = candidate_dir / f"{split}_{run_name}.manifest.json"
    output_paths = [candidate_path, manifest_path]
    for candidate_kind in CANDIDATE_KINDS:
        for track in TRACKS:
            output_paths.append(
                trec_path(
                    run_dir,
                    split=split,
                    candidate_kind=candidate_kind,
                    track=track,
                    run_name=run_name,
                )
            )
    _ensure_outputs_available(output_paths, force=force)

    allowed_claim_ids_list = _load_allowed_claim_ids(
        corpus_dir, split=split, max_queries=max_queries
    )
    allowed_claim_ids = set(allowed_claim_ids_list)
    if not allowed_claim_ids:
        raise EvidenceGapError("No eligible claims selected for fusion")
    expected_track_queries = _expected_track_query_counts(
        corpus_dir,
        selected_claim_ids=allowed_claim_ids_list,
    )

    source_files: dict[str, str] = {}
    source_fingerprints: dict[str, str] = {}
    for candidate_kind in CANDIDATE_KINDS:
        for track in TRACKS:
            for source in sources:
                path = trec_path(
                    input_run_dir,
                    split=split,
                    candidate_kind=candidate_kind,
                    track=track,
                    run_name=source.run_name,
                )
                if not path.exists():
                    raise EvidenceGapError(f"Missing source TREC run: {path}")
                key = f"{source.alias}:{candidate_kind}:{track}"
                source_files[key] = relative_path(root, path)
                source_fingerprints[key] = sha256_file(path)

    candidate_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    schema = _candidate_schema(aliases)
    _pa, pq = _pyarrow()
    candidate_temp = candidate_path.with_name(candidate_path.name + ".tmp")
    trec_temps: dict[tuple[str, str], Path] = {}
    trec_finals: dict[tuple[str, str], Path] = {}
    handles: dict[tuple[str, str], Any] = {}
    stats: dict[str, dict[str, Any]] = {}
    candidate_buffer: list[dict[str, Any]] = []

    try:
        writer = pq.ParquetWriter(
            candidate_temp,
            schema,
            compression="zstd",
            use_dictionary=[
                "schema_version",
                "split",
                "candidate_kind",
                "track",
                "fusion_method",
            ],
        )
        try:
            for candidate_kind in CANDIDATE_KINDS:
                for track in TRACKS:
                    final_path = trec_path(
                        run_dir,
                        split=split,
                        candidate_kind=candidate_kind,
                        track=track,
                        run_name=run_name,
                    )
                    temp_path = final_path.with_name(final_path.name + ".tmp")
                    temp_path.unlink(missing_ok=True)
                    trec_finals[(candidate_kind, track)] = final_path
                    trec_temps[(candidate_kind, track)] = temp_path
                    handles[(candidate_kind, track)] = temp_path.open(
                        "w", encoding="utf-8"
                    )

                    source_paths = _source_paths(
                        input_run_dir,
                        sources=sources,
                        split=split,
                        candidate_kind=candidate_kind,
                        track=track,
                    )
                    query_count = 0
                    union_sizes: list[int] = []
                    source_masks: Counter[int] = Counter()
                    rows_written = 0
                    trec_rows_written = 0
                    for claim_id, grouped in _merge_source_groups(
                        source_paths, allowed_claim_ids=allowed_claim_ids
                    ):
                        fused = _fuse_one_query(
                            grouped,
                            aliases=aliases,
                            method=method,
                            rrf_k=rrf_k,
                            weights=normalized_weights,
                        )
                        query_count += 1
                        union_sizes.append(len(fused))
                        for row in fused:
                            source_masks[int(row["source_mask"])] += 1
                            candidate_buffer.append(
                                {
                                    "schema_version": FUSION_SCHEMA_VERSION,
                                    "split": split,
                                    "candidate_kind": candidate_kind,
                                    "track": track,
                                    "claim_id": claim_id,
                                    "article_id": row["article_id"],
                                    "fusion_method": method,
                                    "source_mask": row["source_mask"],
                                    "source_count": row["source_count"],
                                    "best_rank": row["best_rank"],
                                    "rank_sum": row["rank_sum"],
                                    "rrf_score": row["rrf_score"],
                                    "fusion_score": row["fusion_score"],
                                    "fusion_rank": row["fusion_rank"],
                                    **{
                                        f"{alias}_rank": row[f"{alias}_rank"]
                                        for alias in aliases
                                    },
                                    **{
                                        f"{alias}_score": row[f"{alias}_score"]
                                        for alias in aliases
                                    },
                                }
                            )
                            rows_written += 1
                            if len(candidate_buffer) >= 100_000:
                                _write_candidate_batch(writer, schema, candidate_buffer)

                        emitted = fused if candidate_kind == "judged" else fused[:top_k]
                        handle = handles[(candidate_kind, track)]
                        for rank, row in enumerate(emitted, start=1):
                            _write_trec_row(
                                handle,
                                claim_id=claim_id,
                                article_id=str(row["article_id"]),
                                rank=rank,
                                score=float(row["fusion_score"]),
                                run_name=run_name,
                            )
                            trec_rows_written += 1

                    expected_queries = expected_track_queries[track]
                    if query_count != expected_queries:
                        raise EvidenceGapError(
                            "Source runs do not cover the expected eligible queries for "
                            f"{candidate_kind}/{track}: "
                            f"{query_count:,}/{expected_queries:,}"
                        )

                    stats[f"{candidate_kind}:{track}"] = {
                        "queries": query_count,
                        "candidate_rows": rows_written,
                        "trec_rows": trec_rows_written,
                        "union_size_min": min(union_sizes) if union_sizes else None,
                        "union_size_mean": (
                            round(fmean(union_sizes), 4) if union_sizes else None
                        ),
                        "union_size_max": max(union_sizes) if union_sizes else None,
                        "source_mask_counts": {
                            str(mask): count
                            for mask, count in sorted(source_masks.items())
                        },
                    }
            _write_candidate_batch(writer, schema, candidate_buffer)
        finally:
            writer.close()
            for handle in handles.values():
                handle.close()

        os.replace(candidate_temp, candidate_path)
        for key, temp_path in trec_temps.items():
            os.replace(temp_path, trec_finals[key])
    except Exception:
        candidate_temp.unlink(missing_ok=True)
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        for temp_path in trec_temps.values():
            temp_path.unlink(missing_ok=True)
        raise

    ceiling = _union_ceiling(candidate_path, corpus_dir, split=split)
    manifest = {
        "schema_version": FUSION_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "run_name": run_name,
        "method": method,
        "rrf_k": rrf_k,
        "top_k": top_k,
        "query_limit": max_queries,
        "selected_claims": len(allowed_claim_ids_list),
        "sources": [
            {
                "alias": source.alias,
                "run_name": source.run_name,
                "weight": normalized_weights[source.alias],
            }
            for source in sources
        ],
        "source_files": source_files,
        "source_sha256": source_fingerprints,
        "candidate_parquet": {
            "path": relative_path(root, candidate_path),
            "sha256": sha256_file(candidate_path),
            "bytes": candidate_path.stat().st_size,
        },
        "stats": stats,
        "full_union_ceiling": ceiling,
    }
    atomic_write_json(manifest_path, manifest)

    report = run_article_retrieval(
        root,
        split=split,
        corpus_dir=corpus_dir,
        run_dir=run_dir,
        report_dir=report_dir,
        top_k=top_k,
        max_queries=max_queries,
        run_name=run_name,
        reuse_run=True,
    )
    report["retriever"] = {"family": "hybrid_fusion"}
    report["fusion"] = {
        "method": method,
        "rrf_k": rrf_k,
        "sources": manifest["sources"],
        "candidate_manifest": relative_path(root, manifest_path),
        "candidate_parquet": relative_path(root, candidate_path),
        "stats": stats,
        "full_union_ceiling": ceiling,
    }
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report_stem = f"article_retrieval_{run_name}_{split}"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{report_stem}.json"
    atomic_write_json(report_path, report)
    (report_dir / f"{report_stem}.md").write_text(
        _phase04_markdown(report), encoding="utf-8"
    )
    print(f"Fusion candidate manifest: {manifest_path}")
    print(f"Fusion report: {report_path}")
    return report
