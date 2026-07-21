from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    load_json,
    relative_path,
    sha256_file,
)

ARTICLE_INPUT_SCHEMA_VERSION = "1.0.0"
DEFAULT_CORPUS_DIR = Path("artifacts/v1/article_corpus")
DEFAULT_OUTPUT_DIR = Path("artifacts/v1/dense/article_inputs")
PHASE01_DIR = Path("data/processed/v1/manifests")
RAW_GLOB = "data/raw/v1/medfact_synth/data/*.parquet"


def _duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing duckdb dependency. Install requirements/v1-phase03.txt"
        ) from exc
    return duckdb


def _quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _configure(
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


def build_dense_article_inputs(
    root: Path,
    *,
    corpus_dir: Path | None = None,
    output_dir: Path | None = None,
    threads: int = 8,
    memory_limit: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Recover title/abstract pairs without changing the Phase 02 corpus.

    Phase 02 intentionally normalizes article text for lexical retrieval. MedCPT's
    article encoder is trained on title/abstract pairs, so Phase 03 creates a
    separate, immutable input view joined back to the selected raw source variant.
    """
    root = root.resolve()
    corpus_dir = (root / (corpus_dir or DEFAULT_CORPUS_DIR)).resolve()
    output_dir = (root / (output_dir or DEFAULT_OUTPUT_DIR)).resolve()
    articles_path = corpus_dir / "articles.parquet"
    corpus_manifest_path = corpus_dir / "corpus_manifest.json"
    if not articles_path.exists() or not corpus_manifest_path.exists():
        raise EvidenceGapError(
            f"Missing Phase 02 corpus under {corpus_dir}; build Phase 02 first"
        )

    phase01_paths = [
        root / PHASE01_DIR / "medfact_train.jsonl",
        root / PHASE01_DIR / "medfact_dev.jsonl",
        root / PHASE01_DIR / "medfact_test.jsonl",
    ]
    for path in phase01_paths:
        if not path.exists():
            raise EvidenceGapError(f"Missing Phase 01 manifest: {path}")
    raw_paths = sorted(root.glob(RAW_GLOB))
    if not raw_paths:
        raise EvidenceGapError(f"No raw MedFact shards found: {RAW_GLOB}")

    corpus_manifest = load_json(corpus_manifest_path)
    expected_articles = int(corpus_manifest["counts"]["articles"])
    duckdb = _duckdb()

    with atomic_directory(output_dir, force=force) as staging:
        temp_dir = staging / "duckdb_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(staging / "build.duckdb"))
        try:
            _configure(
                connection,
                threads=threads,
                memory_limit=memory_limit,
                temp_directory=temp_dir,
            )
            manifest_list = ",".join(
                f"'{_quote(path)}'" for path in phase01_paths
            )
            raw_glob = _quote(root / RAW_GLOB)
            output_path = staging / "article_inputs.parquet"
            connection.execute(
                f"""
                COPY (
                    WITH phase01 AS (
                        SELECT *
                        FROM read_json_auto(
                            [{manifest_list}],
                            format='newline_delimited',
                            union_by_name=true
                        )
                    ), raw_medfact AS (
                        SELECT idx, source
                        FROM read_parquet('{raw_glob}', union_by_name=true)
                    ), matched AS (
                        SELECT
                            a.doc_idx,
                            a.article_id,
                            a.text_hash,
                            CAST(r.source AS VARCHAR) AS raw_text,
                            row_number() OVER (
                                PARTITION BY a.doc_idx
                                ORDER BY
                                    length(CAST(r.source AS VARCHAR)) DESC,
                                    CAST(r.source AS VARCHAR) ASC
                            ) AS source_rank
                        FROM read_parquet('{_quote(articles_path)}') a
                        JOIN phase01 m
                          ON a.article_id = m.article_id
                         AND a.text_hash = m.article_text_hash
                        JOIN raw_medfact r
                          ON CAST(r.idx AS VARCHAR) =
                             CAST(m.raw_locator.record_id AS VARCHAR)
                    ), selected AS (
                        SELECT doc_idx, article_id, text_hash, raw_text
                        FROM matched
                        WHERE source_rank = 1
                    ), segmented AS (
                        SELECT
                            doc_idx,
                            article_id,
                            text_hash,
                            CASE
                                WHEN strpos(replace(raw_text, chr(13), ''), chr(10)) > 0
                                THEN trim(substr(
                                    replace(raw_text, chr(13), ''),
                                    1,
                                    strpos(replace(raw_text, chr(13), ''), chr(10)) - 1
                                ))
                                ELSE ''
                            END AS title,
                            CASE
                                WHEN strpos(replace(raw_text, chr(13), ''), chr(10)) > 0
                                THEN trim(substr(
                                    replace(raw_text, chr(13), ''),
                                    strpos(replace(raw_text, chr(13), ''), chr(10)) + 1
                                ))
                                ELSE trim(replace(raw_text, chr(13), ''))
                            END AS abstract
                        FROM selected
                    )
                    SELECT *
                    FROM segmented
                    ORDER BY doc_idx
                ) TO '{_quote(output_path)}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
                """
            )
        finally:
            connection.close()

        actual_articles = pq.read_metadata(output_path).num_rows
        if actual_articles != expected_articles:
            raise EvidenceGapError(
                "Dense article input count mismatch: "
                f"expected {expected_articles:,}, got {actual_articles:,}"
            )

        manifest = {
            "schema_version": ARTICLE_INPUT_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rows": actual_articles,
            "source_corpus": relative_path(root, corpus_dir),
            "source_corpus_manifest_sha256": sha256_file(corpus_manifest_path),
            "output": {
                "path": relative_path(root, output_dir / "article_inputs.parquet"),
                "sha256": sha256_file(output_path),
                "bytes": output_path.stat().st_size,
            },
            "segmentation": {
                "title": "text before the first raw newline",
                "abstract": "remaining raw text; full raw text when no newline exists",
                "fallback": "empty title plus full text as abstract",
            },
        }
        atomic_write_json(staging / "article_inputs_manifest.json", manifest)
        for transient in (staging / "build.duckdb", temp_dir):
            if transient.is_file():
                transient.unlink()
            elif transient.is_dir():
                import shutil

                shutil.rmtree(transient, ignore_errors=True)

    return manifest


def validate_dense_article_inputs(
    root: Path,
    *,
    corpus_dir: Path | None = None,
    input_dir: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    corpus_dir = (root / (corpus_dir or DEFAULT_CORPUS_DIR)).resolve()
    input_dir = (root / (input_dir or DEFAULT_OUTPUT_DIR)).resolve()
    errors: list[str] = []
    manifest_path = input_dir / "article_inputs_manifest.json"
    parquet_path = input_dir / "article_inputs.parquet"
    try:
        manifest = load_json(manifest_path)
    except EvidenceGapError as exc:
        return {"status": "FAIL", "errors": [str(exc)]}

    if manifest.get("schema_version") != ARTICLE_INPUT_SCHEMA_VERSION:
        errors.append("article input schema version mismatch")
    if not parquet_path.exists():
        errors.append(f"missing {parquet_path}")
    else:
        if sha256_file(parquet_path) != manifest["output"]["sha256"]:
            errors.append("article input fingerprint mismatch")
        rows = pq.read_metadata(parquet_path).num_rows
        if rows != int(manifest["rows"]):
            errors.append("article input row count mismatch")
        connection = _duckdb().connect()
        try:
            stats = connection.execute(
                f"SELECT count(*), count(DISTINCT doc_idx), min(doc_idx), max(doc_idx) "
                f"FROM read_parquet('{_quote(parquet_path)}')"
            ).fetchone()
        finally:
            connection.close()
        if stats != (rows, rows, 0, rows - 1):
            errors.append(f"article input doc_idx invariant failed: {stats}")

    corpus_manifest_path = corpus_dir / "corpus_manifest.json"
    if not corpus_manifest_path.exists():
        errors.append("missing Phase 02 corpus manifest")
    elif (
        sha256_file(corpus_manifest_path)
        != manifest.get("source_corpus_manifest_sha256")
    ):
        errors.append("Phase 02 corpus fingerprint drift")

    return {
        "schema_version": ARTICLE_INPUT_SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "manifest": relative_path(root, manifest_path),
    }
