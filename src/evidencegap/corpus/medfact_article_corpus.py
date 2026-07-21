from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    load_json,
    manifest_fingerprint,
    relative_path,
    sha256_file,
)

CORPUS_SCHEMA_VERSION = "1.0.0"
PHASE01_DIR = Path("data/processed/v1/manifests")
RAW_GLOB = "data/raw/v1/medfact_synth/data/*.parquet"
DEFAULT_OUTPUT = Path("artifacts/v1/article_corpus")


def _duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing dependency duckdb. Install requirements/v1-phase02.txt"
        ) from exc
    return duckdb


def _quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _query_scalar(connection: Any, sql: str) -> Any:
    row = connection.execute(sql).fetchone()
    return row[0] if row else None


def _parquet_rows(path: Path) -> int:
    return pq.read_metadata(path).num_rows


def _configure_duckdb(
    connection: Any,
    *,
    threads: int,
    memory_limit: str | None,
    temp_directory: Path,
) -> None:
    connection.execute(f"SET threads={max(1, threads)}")
    connection.execute(f"SET temp_directory='{_quote(temp_directory)}'")
    connection.execute("SET preserve_insertion_order=false")
    if memory_limit:
        safe = memory_limit.replace("'", "''")
        connection.execute(f"SET memory_limit='{safe}'")


def _copy_parquet(connection: Any, query: str, path: Path) -> None:
    connection.execute(
        f"COPY ({query}) TO '{_quote(path)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )


def build_medfact_article_corpus(
    root: Path,
    *,
    output_dir: Path | None = None,
    force: bool = False,
    threads: int = 8,
    memory_limit: str | None = None,
    quick_rows: int | None = None,
) -> dict[str, Any]:
    """Build engine-agnostic article, claim, and judgment Parquet tables."""
    root = root.resolve()
    output_dir = (root / (output_dir or DEFAULT_OUTPUT)).resolve()
    phase01_dir = root / PHASE01_DIR
    manifest_paths = [
        phase01_dir / "medfact_train.jsonl",
        phase01_dir / "medfact_dev.jsonl",
        phase01_dir / "medfact_test.jsonl",
    ]
    for path in manifest_paths:
        if not path.exists():
            raise EvidenceGapError(f"Missing Phase 01 manifest: {path}")

    phase01_meta_path = phase01_dir / "phase01_manifest.json"
    phase01_meta = load_json(phase01_meta_path)
    if phase01_meta.get("schema_version") != "1.0.0":
        raise EvidenceGapError(
            "Phase 01 manifest schema must be 1.0.0 before Phase 02"
        )

    raw_paths = sorted(root.glob(RAW_GLOB))
    if not raw_paths:
        raise EvidenceGapError(f"No raw MedFact shards found: {RAW_GLOB}")

    duckdb = _duckdb()
    with atomic_directory(output_dir, force=force) as staging:
        db_path = staging / "phase02_build.duckdb"
        temp_dir = staging / "duckdb_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(db_path))
        _configure_duckdb(
            connection,
            threads=threads,
            memory_limit=memory_limit,
            temp_directory=temp_dir,
        )

        manifest_list = ",".join(f"'{_quote(path)}'" for path in manifest_paths)
        raw_glob = _quote(root / RAW_GLOB)
        limit_clause = f" LIMIT {int(quick_rows)}" if quick_rows else ""

        connection.execute(
            f"""
            CREATE TABLE phase01_rows AS
            SELECT *
            FROM read_json_auto(
                [{manifest_list}],
                format='newline_delimited',
                union_by_name=true
            )
            {limit_clause}
            """
        )
        connection.execute(
            f"""
            CREATE VIEW raw_medfact AS
            SELECT * FROM read_parquet('{raw_glob}', union_by_name=true)
            """
        )
        connection.execute(
            """
            CREATE TABLE joined_rows AS
            SELECT
                m.*,
                r.claim AS raw_claim,
                r.source AS raw_source
            FROM phase01_rows m
            INNER JOIN raw_medfact r
              ON CAST(r.idx AS VARCHAR) = CAST(m.raw_locator.record_id AS VARCHAR)
            """
        )

        manifest_count = _query_scalar(connection, "SELECT count(*) FROM phase01_rows")
        joined_count = _query_scalar(connection, "SELECT count(*) FROM joined_rows")
        if manifest_count != joined_count:
            raise EvidenceGapError(
                "Phase 01/raw join is not one-to-one: "
                f"manifest={manifest_count:,}, joined={joined_count:,}"
            )

        # A text variant is identified by the Phase 01 normalized-text hash.
        # We retain one deterministic representative string per hash.
        connection.execute(
            r"""
            CREATE TABLE article_variants AS
            WITH normalized AS (
                SELECT
                    article_id,
                    source_pmid,
                    article_text_hash AS text_hash,
                    regexp_replace(trim(CAST(raw_source AS VARCHAR)), '\s+', ' ', 'g') AS text
                FROM joined_rows
            ), collapsed AS (
                SELECT
                    article_id,
                    any_value(source_pmid) AS source_pmid,
                    text_hash,
                    max(text) AS text,
                    count(*)::BIGINT AS variant_occurrence_count
                FROM normalized
                GROUP BY article_id, text_hash
            )
            SELECT * FROM collapsed
            """
        )

        articles_path = staging / "articles.parquet"
        _copy_parquet(
            connection,
            """
            WITH ranked AS (
                SELECT
                    article_id,
                    source_pmid,
                    text_hash,
                    text,
                    variant_occurrence_count,
                    sum(variant_occurrence_count) OVER (
                        PARTITION BY article_id
                    )::BIGINT AS occurrence_count,
                    count(*) OVER (
                        PARTITION BY article_id
                    )::INTEGER AS text_variant_count,
                    row_number() OVER (
                        PARTITION BY article_id
                        ORDER BY
                            variant_occurrence_count DESC,
                            length(text) DESC,
                            text_hash ASC
                    ) AS variant_rank
                FROM article_variants
            ), selected AS (
                SELECT * FROM ranked WHERE variant_rank = 1
            )
            SELECT
                (row_number() OVER (ORDER BY article_id) - 1)::BIGINT AS doc_idx,
                article_id,
                source_pmid AS pmid,
                text,
                text_hash,
                occurrence_count,
                variant_occurrence_count AS selected_variant_occurrence_count,
                text_variant_count
            FROM selected
            ORDER BY article_id
            """,
            articles_path,
        )

        conflicts_path = staging / "corpus_conflicts.parquet"
        _copy_parquet(
            connection,
            """
            WITH stats AS (
                SELECT
                    *,
                    count(*) OVER (PARTITION BY article_id) AS text_variant_count,
                    row_number() OVER (
                        PARTITION BY article_id
                        ORDER BY
                            variant_occurrence_count DESC,
                            length(text) DESC,
                            text_hash ASC
                    ) AS variant_rank
                FROM article_variants
            )
            SELECT
                article_id,
                source_pmid AS pmid,
                text_hash,
                variant_occurrence_count,
                length(text)::BIGINT AS text_char_count,
                variant_rank = 1 AS selected
            FROM stats
            WHERE text_variant_count > 1
            ORDER BY article_id, variant_rank
            """,
            conflicts_path,
        )

        claims_path = staging / "claims.parquet"
        _copy_parquet(
            connection,
            r"""
            WITH variants AS (
                SELECT
                    claim_id,
                    any_value(claim_pmid) AS claim_pmid,
                    claim_text_hash,
                    split_group_id,
                    split,
                    regexp_replace(trim(CAST(raw_claim AS VARCHAR)), '\s+', ' ', 'g') AS claim_text,
                    count(*)::BIGINT AS occurrence_count
                FROM joined_rows
                GROUP BY
                    claim_id,
                    claim_text_hash,
                    split_group_id,
                    split,
                    regexp_replace(trim(CAST(raw_claim AS VARCHAR)), '\s+', ' ', 'g')
            ), ranked AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY claim_id
                        ORDER BY occurrence_count DESC, length(claim_text) DESC,
                                 claim_text ASC
                    ) AS text_rank,
                    count(*) OVER (PARTITION BY claim_id) AS text_variant_count
                FROM variants
            )
            SELECT
                claim_id,
                claim_pmid,
                claim_text,
                claim_text_hash,
                split_group_id,
                split,
                occurrence_count,
                text_variant_count::INTEGER AS text_variant_count
            FROM ranked
            WHERE text_rank = 1
            ORDER BY claim_id
            """,
            claims_path,
        )

        judgments_path = staging / "judgments.parquet"
        _copy_parquet(
            connection,
            """
            WITH grouped AS (
                SELECT
                    claim_id,
                    article_id,
                    any_value(split) AS split,
                    min(stance_label)::INTEGER AS min_stance_label,
                    max(stance_label)::INTEGER AS max_stance_label,
                    count(*)::BIGINT AS raw_row_count,
                    bool_or(is_origin_source) AS is_origin_source,
                    count(DISTINCT split) AS split_count
                FROM joined_rows
                GROUP BY claim_id, article_id
            )
            SELECT
                claim_id,
                article_id,
                split,
                CASE WHEN min_stance_label = max_stance_label
                     THEN min_stance_label ELSE NULL END AS stance_label,
                CASE WHEN min_stance_label = max_stance_label
                     THEN abs(min_stance_label) ELSE NULL END AS relevance_grade,
                is_origin_source,
                raw_row_count,
                min_stance_label != max_stance_label AS label_conflict,
                min_stance_label AS conflict_min_label,
                max_stance_label AS conflict_max_label,
                min_stance_label = max_stance_label AS eligible_for_qrels,
                split_count
            FROM grouped
            ORDER BY claim_id, article_id
            """,
            judgments_path,
        )

        # Validate key relational invariants before finalizing the directory.
        article_count = _parquet_rows(articles_path)
        claim_count = _parquet_rows(claims_path)
        judgment_count = _parquet_rows(judgments_path)
        conflict_variant_rows = _parquet_rows(conflicts_path)
        doc_idx_issue = _query_scalar(
            connection,
            f"""
            SELECT count(*) FROM (
                SELECT doc_idx,
                       row_number() OVER (ORDER BY doc_idx) - 1 AS expected
                FROM read_parquet('{_quote(articles_path)}')
            ) WHERE doc_idx != expected
            """,
        )
        split_issue = _query_scalar(
            connection,
            f"""
            SELECT count(*)
            FROM read_parquet('{_quote(judgments_path)}')
            WHERE split_count != 1
            """,
        )
        missing_article = _query_scalar(
            connection,
            f"""
            SELECT count(*)
            FROM read_parquet('{_quote(judgments_path)}') j
            LEFT JOIN read_parquet('{_quote(articles_path)}') a USING(article_id)
            WHERE a.article_id IS NULL
            """,
        )
        missing_claim = _query_scalar(
            connection,
            f"""
            SELECT count(*)
            FROM read_parquet('{_quote(judgments_path)}') j
            LEFT JOIN read_parquet('{_quote(claims_path)}') c USING(claim_id)
            WHERE c.claim_id IS NULL
            """,
        )
        if any([doc_idx_issue, split_issue, missing_article, missing_claim]):
            raise EvidenceGapError(
                "Corpus relational validation failed: "
                f"doc_idx={doc_idx_issue}, split={split_issue}, "
                f"missing_article={missing_article}, missing_claim={missing_claim}"
            )

        label_conflicts = _query_scalar(
            connection,
            f"SELECT count(*) FROM read_parquet('{_quote(judgments_path)}') "
            "WHERE label_conflict"
        )
        article_text_conflicts = _query_scalar(
            connection,
            f"SELECT count(DISTINCT article_id) "
            f"FROM read_parquet('{_quote(conflicts_path)}')"
        )
        split_counts = dict(
            connection.execute(
                f"SELECT split, count(*) FROM read_parquet('{_quote(claims_path)}') "
                "GROUP BY split ORDER BY split"
            ).fetchall()
        )
        judgment_split_counts = dict(
            connection.execute(
                f"SELECT split, count(*) FROM read_parquet('{_quote(judgments_path)}') "
                "GROUP BY split ORDER BY split"
            ).fetchall()
        )
        label_counts = {
            str(label): count
            for label, count in connection.execute(
                f"SELECT stance_label, count(*) "
                f"FROM read_parquet('{_quote(judgments_path)}') "
                "WHERE eligible_for_qrels GROUP BY stance_label ORDER BY stance_label"
            ).fetchall()
        }

        connection.close()
        # DuckDB scratch state is not part of the canonical artifact.
        db_path.unlink(missing_ok=True)
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

        output_files = [articles_path, claims_path, judgments_path, conflicts_path]
        manifest = {
            "schema_version": CORPUS_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "quick" if quick_rows else "full",
            "quick_rows": quick_rows,
            "phase01": {
                "schema_version": phase01_meta.get("schema_version"),
                "manifest_fingerprint": manifest_fingerprint(manifest_paths),
                "files": {
                    path.name: sha256_file(path) for path in manifest_paths
                },
            },
            "raw_sources": [
                {
                    "path": relative_path(root, path),
                    "bytes": path.stat().st_size,
                    "rows": pq.read_metadata(path).num_rows,
                }
                for path in raw_paths
            ],
            "selection_policy": {
                "article_key": "Phase 01 article_id",
                "article_variant": [
                    "highest occurrence count",
                    "longest normalized text",
                    "lexicographically smallest text hash",
                ],
                "duplicate_pair": "collapse only when labels agree",
                "label_conflict": "excluded from formal qrels",
            },
            "counts": {
                "phase01_rows": manifest_count,
                "articles": article_count,
                "claims": claim_count,
                "judgments": judgment_count,
                "article_text_conflicts": article_text_conflicts,
                "article_conflict_variant_rows": conflict_variant_rows,
                "judgment_label_conflicts": label_conflicts,
                "claims_by_split": split_counts,
                "judgments_by_split": judgment_split_counts,
                "eligible_stance_labels": label_counts,
            },
            "files": {
                path.name: {
                    "rows": _parquet_rows(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in output_files
            },
        }
        atomic_write_json(staging / "corpus_manifest.json", manifest)

    return manifest


def validate_corpus(root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    output_dir = (root / (output_dir or DEFAULT_OUTPUT)).resolve()
    manifest_path = output_dir / "corpus_manifest.json"
    manifest = load_json(manifest_path)
    errors: list[str] = []
    if manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        errors.append("corpus_manifest schema_version mismatch")

    for name in (
        "articles.parquet",
        "claims.parquet",
        "judgments.parquet",
        "corpus_conflicts.parquet",
    ):
        path = output_dir / name
        metadata = manifest.get("files", {}).get(name, {})
        if not path.exists():
            errors.append(f"missing file: {name}")
            continue
        if metadata.get("sha256") != sha256_file(path):
            errors.append(f"fingerprint mismatch: {name}")
        if metadata.get("rows") != _parquet_rows(path):
            errors.append(f"row count mismatch: {name}")

    report = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "manifest": relative_path(root, manifest_path),
    }
    if errors:
        raise EvidenceGapError("; ".join(errors))
    return report
