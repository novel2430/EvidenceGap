from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    load_json,
    manifest_fingerprint,
    relative_path,
    sha256_file,
)
from evidencegap.evaluation import run_article_retrieval
from evidencegap.reranking.fusion import (
    CANDIDATE_KINDS,
    DEFAULT_CORPUS_DIR,
    DEFAULT_REPORT_DIR,
    DEFAULT_RUN_DIR,
    TRACKS,
    safe_run_name,
    trec_path,
)

DEFAULT_ARTICLE_INPUT_DIR = Path("artifacts/v1/dense/article_inputs")
DEFAULT_MODEL_DIR = Path("models/v1/medcpt-cross")
DEFAULT_SCORE_ROOT = Path("artifacts/v1/reranking/cross_encoder_scores")
DEFAULT_RERANKED_CANDIDATE_DIR = Path("artifacts/v1/reranking/reranked_candidates")
CROSS_ENCODER_SCHEMA_VERSION = "1.1.0"


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


def _normalize_devices(values: Sequence[str]) -> list[str]:
    devices: list[str] = []
    for raw in values:
        raw = raw.strip()
        if not raw:
            continue
        if raw == "cpu" or raw.startswith("cuda:"):
            device = raw
        elif raw.isdigit():
            device = f"cuda:{raw}"
        else:
            raise EvidenceGapError(
                f"Invalid device {raw!r}; use 0,1,..., cuda:N, or cpu"
            )
        devices.append(device)
    if not devices:
        raise EvidenceGapError("At least one device is required")
    if len(set(devices)) != len(devices):
        raise EvidenceGapError("Devices must be unique")
    return devices


def _model_files(model_dir: Path) -> list[Path]:
    names = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
    }
    files = [path for path in model_dir.iterdir() if path.name in names]
    files.extend(sorted(model_dir.glob("*.safetensors")))
    files.extend(sorted(model_dir.glob("pytorch_model*.bin")))
    files.extend(sorted(model_dir.glob("*.index.json")))
    unique = sorted({path.resolve() for path in files if path.is_file()})
    if not unique:
        raise EvidenceGapError(f"No model files found in {model_dir}")
    return unique


def _model_fingerprint(model_dir: Path) -> str:
    return manifest_fingerprint(_model_files(model_dir))


def _recorded_asset_sha256(path: Path) -> str:
    """Prefer immutable upstream manifest fingerprints over rehashing multi-GB files."""
    sibling_manifest = path.with_suffix(".manifest.json")
    if sibling_manifest.exists():
        manifest = load_json(sibling_manifest)
        value = manifest.get("candidate_parquet", {}).get("sha256")
        if value:
            return str(value)
    if path.name == "claims.parquet":
        manifest_path = path.parent / "corpus_manifest.json"
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            value = manifest.get("files", {}).get(path.name, {}).get("sha256")
            if value:
                return str(value)
    if path.name == "article_inputs.parquet":
        manifest_path = path.parent / "article_inputs_manifest.json"
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            value = manifest.get("output", {}).get("sha256")
            if value:
                return str(value)
    return sha256_file(path)


def _input_manifest_signature(
    *,
    candidate_path: Path,
    corpus_dir: Path,
    article_input_dir: Path,
    split: str,
    num_shards: int,
    rerank_depth: int,
) -> dict[str, Any]:
    files = [
        candidate_path,
        corpus_dir / "claims.parquet",
        article_input_dir / "article_inputs.parquet",
    ]
    for path in files:
        if not path.exists():
            raise EvidenceGapError(f"Missing reranking input: {path}")
    return {
        "schema_version": CROSS_ENCODER_SCHEMA_VERSION,
        "split": split,
        "num_shards": num_shards,
        "candidate_sha256": _recorded_asset_sha256(candidate_path),
        "claims_sha256": _recorded_asset_sha256(corpus_dir / "claims.parquet"),
        "article_inputs_sha256": _recorded_asset_sha256(
            article_input_dir / "article_inputs.parquet"
        ),
        "rerank_depth": rerank_depth,
        "pair_selection": (
            "all judged pairs plus open pairs with fusion_rank <= rerank_depth, "
            "deduplicated across tracks"
        ),
        "assignment": "dense_rank(claim_id)-1 modulo num_shards",
    }


def _input_schema() -> Any:
    pa, _pq = _pyarrow()
    return pa.schema(
        [
            pa.field("claim_id", pa.string(), nullable=False),
            pa.field("article_id", pa.string(), nullable=False),
            pa.field("claim_text", pa.string(), nullable=False),
            pa.field("title", pa.string(), nullable=False),
            pa.field("abstract", pa.string(), nullable=False),
        ]
    )


def _flush_rows(writer: Any, schema: Any, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    pa, _pq = _pyarrow()
    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    rows.clear()


def _prepare_inputs(
    root: Path,
    *,
    candidate_path: Path,
    corpus_dir: Path,
    article_input_dir: Path,
    split: str,
    num_shards: int,
    rerank_depth: int,
    input_dir: Path,
    force: bool,
) -> tuple[dict[str, Any], list[Path]]:
    expected = _input_manifest_signature(
        candidate_path=candidate_path,
        corpus_dir=corpus_dir,
        article_input_dir=article_input_dir,
        split=split,
        num_shards=num_shards,
        rerank_depth=rerank_depth,
    )
    manifest_path = input_dir / "input_manifest.json"
    if input_dir.exists() and not force:
        manifest = load_json(manifest_path)
        comparable = {key: manifest.get(key) for key in expected}
        if comparable != expected:
            raise EvidenceGapError(
                f"Existing cross-encoder inputs are stale: {input_dir}. "
                "Use --force to rebuild them."
            )
        shard_paths = [
            input_dir / f"shard-{index:05d}-of-{num_shards:05d}.parquet"
            for index in range(num_shards)
        ]
        missing = [path for path in shard_paths if not path.exists()]
        if missing:
            raise EvidenceGapError(
                f"Input manifest exists but {len(missing)} shard files are missing"
            )
        return manifest, shard_paths

    schema = _input_schema()
    _pa, pq = _pyarrow()
    claims = corpus_dir / "claims.parquet"
    articles = article_input_dir / "article_inputs.parquet"
    duckdb = _duckdb()

    with atomic_directory(input_dir, force=force) as staging:
        shard_paths = [
            staging / f"shard-{index:05d}-of-{num_shards:05d}.parquet"
            for index in range(num_shards)
        ]
        writers = [
            pq.ParquetWriter(path, schema, compression="zstd") for path in shard_paths
        ]
        buffers: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
        row_counts = [0 for _ in range(num_shards)]
        claim_counts = [0 for _ in range(num_shards)]
        connection = duckdb.connect()
        try:
            expected_pairs = int(
                connection.execute(
                    f"""
                    SELECT count(*)
                    FROM (
                        SELECT DISTINCT
                            CAST(claim_id AS VARCHAR) AS claim_id,
                            CAST(article_id AS VARCHAR) AS article_id
                        FROM read_parquet('{_quote(candidate_path)}')
                        WHERE candidate_kind = 'judged'
                           OR (candidate_kind = 'open' AND fusion_rank <= ?)
                    )
                    """,
                    [rerank_depth],
                ).fetchone()[0]
            )
            query = f"""
                WITH pairs AS (
                    SELECT DISTINCT
                        CAST(claim_id AS VARCHAR) AS claim_id,
                        CAST(article_id AS VARCHAR) AS article_id
                    FROM read_parquet('{_quote(candidate_path)}')
                    WHERE candidate_kind = 'judged'
                       OR (candidate_kind = 'open' AND fusion_rank <= ?)
                )
                SELECT
                    p.claim_id,
                    p.article_id,
                    CAST(c.claim_text AS VARCHAR) AS claim_text,
                    coalesce(CAST(a.title AS VARCHAR), '') AS title,
                    coalesce(CAST(a.abstract AS VARCHAR), '') AS abstract
                FROM pairs p
                JOIN read_parquet('{_quote(claims)}') c USING(claim_id)
                JOIN read_parquet('{_quote(articles)}') a USING(article_id)
                WHERE c.split = ?
                ORDER BY p.claim_id, p.article_id
            """
            reader = connection.execute(
                query, [rerank_depth, split]
            ).fetch_record_batch(8192)
            current_claim: str | None = None
            query_index = -1
            for batch in reader:
                for row in batch.to_pylist():
                    claim_id = str(row["claim_id"])
                    if claim_id != current_claim:
                        current_claim = claim_id
                        query_index += 1
                        claim_counts[query_index % num_shards] += 1
                    shard_id = query_index % num_shards
                    buffers[shard_id].append(
                        {
                            "claim_id": claim_id,
                            "article_id": str(row["article_id"]),
                            "claim_text": str(row["claim_text"]),
                            "title": str(row["title"]),
                            "abstract": str(row["abstract"]),
                        }
                    )
                    row_counts[shard_id] += 1
                    if len(buffers[shard_id]) >= 4096:
                        _flush_rows(writers[shard_id], schema, buffers[shard_id])
            joined_pairs = sum(row_counts)
            if joined_pairs != expected_pairs:
                raise EvidenceGapError(
                    "Cross-encoder input join dropped candidates: "
                    f"{joined_pairs:,}/{expected_pairs:,} unique claim-article pairs. "
                    "Check claims.parquet and article_inputs.parquet coverage."
                )
        finally:
            connection.close()
            for shard_id, writer in enumerate(writers):
                _flush_rows(writer, schema, buffers[shard_id])
                writer.close()

        manifest = {
            **expected,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "candidate_path": relative_path(root, candidate_path),
            "corpus_dir": relative_path(root, corpus_dir),
            "article_input_dir": relative_path(root, article_input_dir),
            "rows": sum(row_counts),
            "claims": sum(claim_counts),
            "shards": [
                {
                    "shard_id": index,
                    "path": relative_path(
                        root,
                        input_dir / f"shard-{index:05d}-of-{num_shards:05d}.parquet",
                    ),
                    "rows": row_counts[index],
                    "claims": claim_counts[index],
                }
                for index in range(num_shards)
            ],
        }
        atomic_write_json(staging / "input_manifest.json", manifest)

    final_paths = [
        input_dir / f"shard-{index:05d}-of-{num_shards:05d}.parquet"
        for index in range(num_shards)
    ]
    return load_json(manifest_path), final_paths


def _score_signature(
    *,
    input_path: Path,
    model_fingerprint: str,
    max_length: int,
    amp: str,
) -> dict[str, Any]:
    return {
        "schema_version": CROSS_ENCODER_SCHEMA_VERSION,
        "input_sha256": sha256_file(input_path),
        "model_fingerprint": model_fingerprint,
        "max_length": max_length,
        "amp": amp,
        "score_semantics": "raw_single_logit_higher_is_more_relevant",
    }


def _score_paths(score_dir: Path, input_path: Path) -> tuple[Path, Path]:
    stem = input_path.stem
    return score_dir / f"{stem}.scores.parquet", score_dir / f"{stem}.scores.json"


def _article_text(title: str, abstract: str) -> str:
    title = title.strip()
    abstract = abstract.strip()
    if not title:
        return abstract
    if not abstract:
        return title
    separator = " " if title.endswith((".", "!", "?", ":", ";")) else ". "
    return title + separator + abstract


def _load_cross_encoder(
    model_dir: Path, *, device: str, amp: str
) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing torch/transformers dependency for cross-encoder reranking"
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise EvidenceGapError(
            f"CUDA device requested but CUDA is unavailable: {device}"
        )
    if device == "cpu" and amp == "fp16":
        raise EvidenceGapError("fp16 cross-encoder inference is not supported on CPU")

    safe_weights = tuple(model_dir.glob("*.safetensors"))
    if not safe_weights:
        raise EvidenceGapError(
            "MedCPT cross encoder requires model.safetensors in "
            f"{model_dir}. The original pytorch_model.bin cannot be loaded "
            "with torch<2.6 after CVE-2025-32434. Re-run: "
            "python scripts/download_v1_models.py --root . "
            "--model medcpt-cross"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=True
    )
    if int(getattr(model.config, "num_labels", 0)) != 1:
        raise EvidenceGapError(
            "MedCPT cross encoder must expose one relevance logit; "
            f"model reports num_labels={model.config.num_labels}"
        )
    model.eval()
    model.to(device)
    if amp == "fp16":
        model.half()
    return torch, tokenizer, model


def _score_one_shard(
    *,
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    model_fingerprint: str,
    device: str,
    batch_size: int,
    max_length: int,
    amp: str,
    torch: Any,
    tokenizer: Any,
    model: Any,
) -> dict[str, Any]:
    signature = _score_signature(
        input_path=input_path,
        model_fingerprint=model_fingerprint,
        max_length=max_length,
        amp=amp,
    )

    _pa, pq = _pyarrow()
    score_schema = _pa.schema(
        [
            _pa.field("claim_id", _pa.string(), nullable=False),
            _pa.field("article_id", _pa.string(), nullable=False),
            _pa.field("cross_encoder_score", _pa.float32(), nullable=False),
        ]
    )
    temp_output = output_path.with_name(output_path.name + ".tmp")
    temp_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
    temp_output.unlink(missing_ok=True)
    temp_metadata.unlink(missing_ok=True)

    started = time.perf_counter()
    rows_scored = 0
    score_sum = 0.0
    score_min: float | None = None
    score_max: float | None = None
    writer = pq.ParquetWriter(temp_output, score_schema, compression="zstd")
    try:
        parquet = pq.ParquetFile(input_path)
        for record_batch in parquet.iter_batches(batch_size=batch_size):
            rows = record_batch.to_pylist()
            claims = [str(row["claim_text"]) for row in rows]
            articles = [
                _article_text(str(row["title"]), str(row["abstract"])) for row in rows
            ]
            encoded = tokenizer(
                claims,
                articles,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise EvidenceGapError(
                    f"Unexpected cross-encoder logits shape: {tuple(logits.shape)}"
                )
            values = logits[:, 0].float().cpu().tolist()
            if any(not math.isfinite(float(value)) for value in values):
                raise EvidenceGapError("Cross encoder produced a non-finite score")
            output_rows = [
                {
                    "claim_id": str(row["claim_id"]),
                    "article_id": str(row["article_id"]),
                    "cross_encoder_score": float(score),
                }
                for row, score in zip(rows, values)
            ]
            writer.write_table(_pa.Table.from_pylist(output_rows, schema=score_schema))
            rows_scored += len(output_rows)
            numeric_values = [float(value) for value in values]
            score_sum += sum(numeric_values)
            if numeric_values:
                batch_min = min(numeric_values)
                batch_max = max(numeric_values)
                score_min = (
                    batch_min if score_min is None else min(score_min, batch_min)
                )
                score_max = (
                    batch_max if score_max is None else max(score_max, batch_max)
                )
    finally:
        writer.close()

    elapsed = time.perf_counter() - started
    metadata = {
        **signature,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "device": device,
        "batch_size": batch_size,
        "rows": rows_scored,
        "seconds": round(elapsed, 4),
        "pairs_per_second": round(rows_scored / elapsed, 4) if elapsed > 0 else None,
        "score_min": score_min,
        "score_mean": score_sum / rows_scored if rows_scored else None,
        "score_max": score_max,
    }
    metadata["output_sha256"] = sha256_file(temp_output)
    temp_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_output, output_path)
    os.replace(temp_metadata, metadata_path)
    return metadata


def _cross_encoder_worker(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    device = str(payload["device"])
    amp = str(payload["amp"])
    torch, tokenizer, model = _load_cross_encoder(
        Path(payload["model_dir"]), device=device, amp=amp
    )
    results: list[dict[str, Any]] = []
    try:
        for task in payload["tasks"]:
            results.append(
                _score_one_shard(
                    input_path=Path(task["input_path"]),
                    output_path=Path(task["output_path"]),
                    metadata_path=Path(task["metadata_path"]),
                    model_fingerprint=str(payload["model_fingerprint"]),
                    device=device,
                    batch_size=int(payload["batch_size"]),
                    max_length=int(payload["max_length"]),
                    amp=amp,
                    torch=torch,
                    tokenizer=tokenizer,
                    model=model,
                )
            )
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return results


def _validate_or_plan_scores(
    *,
    input_paths: Sequence[Path],
    score_dir: Path,
    model_fingerprint: str,
    max_length: int,
    amp: str,
    force: bool,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    tasks: list[dict[str, str]] = []
    reused: list[dict[str, Any]] = []
    score_dir.mkdir(parents=True, exist_ok=True)
    _pa, pq = _pyarrow()
    for input_path in input_paths:
        output_path, metadata_path = _score_paths(score_dir, input_path)
        expected = _score_signature(
            input_path=input_path,
            model_fingerprint=model_fingerprint,
            max_length=max_length,
            amp=amp,
        )
        if force:
            output_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
        if output_path.exists() or metadata_path.exists():
            if not output_path.exists() or not metadata_path.exists():
                # A process may have stopped between the two atomic renames. Treat
                # the orphan as an incomplete shard and recompute it automatically.
                output_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
            else:
                metadata = load_json(metadata_path)
                comparable = {key: metadata.get(key) for key in expected}
                if comparable != expected:
                    raise EvidenceGapError(
                        f"Stale score shard for {input_path.name}; use --force"
                    )
                actual_rows = pq.read_metadata(output_path).num_rows
                if actual_rows != int(metadata.get("rows", -1)):
                    raise EvidenceGapError(
                        f"Score shard row count mismatch for {input_path.name}; "
                        "use --force"
                    )
                recorded_output_sha = metadata.get("output_sha256")
                if recorded_output_sha and recorded_output_sha != sha256_file(
                    output_path
                ):
                    raise EvidenceGapError(
                        f"Score shard fingerprint mismatch for {input_path.name}; "
                        "use --force"
                    )
                reused.append(metadata)
                continue
        tasks.append(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "metadata_path": str(metadata_path),
            }
        )
    return tasks, reused


def _run_scoring(
    *,
    tasks: Sequence[Mapping[str, str]],
    devices: Sequence[str],
    model_dir: Path,
    model_fingerprint: str,
    batch_size: int,
    max_length: int,
    amp: str,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    assignments: list[list[Mapping[str, str]]] = [[] for _ in devices]
    for index, task in enumerate(tasks):
        assignments[index % len(devices)].append(task)
    payloads = [
        {
            "tasks": assignment,
            "device": device,
            "model_dir": str(model_dir),
            "model_fingerprint": model_fingerprint,
            "batch_size": batch_size,
            "max_length": max_length,
            "amp": amp,
        }
        for device, assignment in zip(devices, assignments)
        if assignment
    ]
    if len(payloads) == 1:
        return _cross_encoder_worker(payloads[0])
    context = mp.get_context("spawn")
    with context.Pool(processes=len(payloads)) as pool:
        nested = pool.map(_cross_encoder_worker, payloads)
    return [item for group in nested for item in group]


def _score_glob(score_dir: Path) -> str:
    return _quote(score_dir / "shard-*.scores.parquet")


def _write_reranked_candidates(
    *,
    candidate_path: Path,
    score_dir: Path,
    output_path: Path,
    rerank_depth: int,
    force: bool,
) -> dict[str, int]:
    if output_path.exists() and not force:
        raise EvidenceGapError(
            f"Reranked candidate parquet already exists: {output_path}. Use --force."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + ".tmp")
    temp.unlink(missing_ok=True)
    duckdb = _duckdb()
    connection = duckdb.connect()
    selected_condition = (
        "p.candidate_kind = 'judged' OR "
        f"(p.candidate_kind = 'open' AND p.fusion_rank <= {rerank_depth})"
    )
    try:
        selected_rows = int(
            connection.execute(
                f"""
                SELECT count(*)
                FROM read_parquet('{_quote(candidate_path)}') p
                WHERE {selected_condition}
                """
            ).fetchone()[0]
        )
        joined_selected_rows = int(
            connection.execute(
                f"""
                SELECT count(*)
                FROM read_parquet('{_quote(candidate_path)}') p
                JOIN read_parquet('{_score_glob(score_dir)}') s
                  USING(claim_id, article_id)
                WHERE {selected_condition}
                """
            ).fetchone()[0]
        )
        if selected_rows != joined_selected_rows:
            raise EvidenceGapError(
                "Cross-encoder score join is incomplete for selected candidates: "
                f"{joined_selected_rows:,}/{selected_rows:,} rows"
            )

        connection.execute(
            f"""
            COPY (
                SELECT
                    p.*,
                    CASE
                        WHEN {selected_condition}
                        THEN CAST(s.cross_encoder_score AS DOUBLE)
                        ELSE NULL
                    END AS cross_encoder_score,
                    row_number() OVER (
                        PARTITION BY p.candidate_kind, p.track, p.claim_id
                        ORDER BY
                            CASE WHEN {selected_condition} THEN 0 ELSE 1 END ASC,
                            CASE WHEN {selected_condition}
                                 THEN s.cross_encoder_score ELSE NULL END DESC NULLS LAST,
                            p.fusion_rank ASC,
                            p.fusion_score DESC,
                            p.article_id ASC
                    )::INTEGER AS final_rank
                FROM read_parquet('{_quote(candidate_path)}') p
                LEFT JOIN read_parquet('{_score_glob(score_dir)}') s
                  USING(claim_id, article_id)
                ORDER BY p.candidate_kind, p.track, p.claim_id, final_rank
            ) TO '{_quote(temp)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
            """
        )
        source_rows = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet('{_quote(candidate_path)}')"
            ).fetchone()[0]
        )
        output_rows = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet('{_quote(temp)}')"
            ).fetchone()[0]
        )
        if source_rows != output_rows:
            raise EvidenceGapError(
                "Reranked candidate output did not preserve the full union: "
                f"{output_rows:,}/{source_rows:,} rows"
            )
        os.replace(temp, output_path)
        return {
            "source_candidate_rows": source_rows,
            "output_candidate_rows": output_rows,
            "selected_candidate_rows": selected_rows,
        }
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    finally:
        connection.close()


def _write_reranked_trec(
    *,
    reranked_path: Path,
    run_dir: Path,
    split: str,
    run_name: str,
    top_k: int,
    force: bool,
) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    duckdb = _duckdb()
    connection = duckdb.connect()
    temp_paths: list[Path] = []
    try:
        for candidate_kind in CANDIDATE_KINDS:
            for track in TRACKS:
                final = trec_path(
                    run_dir,
                    split=split,
                    candidate_kind=candidate_kind,
                    track=track,
                    run_name=run_name,
                )
                if final.exists() and not force:
                    raise EvidenceGapError(
                        f"Reranked TREC run already exists: {final}. Use --force."
                    )
                temp = final.with_name(final.name + ".tmp")
                temp.unlink(missing_ok=True)
                temp_paths.append(temp)
                limit_clause = "AND final_rank <= ?" if candidate_kind == "open" else ""
                params: list[Any] = [candidate_kind, track]
                if candidate_kind == "open":
                    params.append(top_k)
                reader = connection.execute(
                    f"""
                    SELECT claim_id, article_id, final_rank, cross_encoder_score
                    FROM read_parquet('{_quote(reranked_path)}')
                    WHERE candidate_kind = ? AND track = ? {limit_clause}
                    ORDER BY claim_id, final_rank
                    """,
                    params,
                ).fetch_record_batch(8192)
                with temp.open("w", encoding="utf-8") as handle:
                    for batch in reader:
                        for row in batch.to_pylist():
                            handle.write(
                                f"{row['claim_id']} Q0 {row['article_id']} "
                                f"{int(row['final_rank'])} "
                                f"{float(row['cross_encoder_score']):.10f} {run_name}\n"
                            )
                os.replace(temp, final)
                outputs[f"{candidate_kind}_{track}"] = str(final)
    except Exception:
        for temp in temp_paths:
            temp.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
    return outputs


def _candidate_set_diagnostics(
    *,
    candidate_path: Path,
    reranked_path: Path,
    top_k: int,
) -> dict[str, dict[str, int | bool]]:
    duckdb = _duckdb()
    connection = duckdb.connect()
    result: dict[str, dict[str, int | bool]] = {}
    try:
        for track in TRACKS:
            row = connection.execute(
                f"""
                WITH baseline AS (
                    SELECT DISTINCT claim_id, article_id
                    FROM read_parquet('{_quote(candidate_path)}')
                    WHERE candidate_kind = 'open'
                      AND track = ?
                      AND fusion_rank <= ?
                ), reranked AS (
                    SELECT DISTINCT claim_id, article_id
                    FROM read_parquet('{_quote(reranked_path)}')
                    WHERE candidate_kind = 'open'
                      AND track = ?
                      AND final_rank <= ?
                ), compared AS (
                    SELECT
                        coalesce(b.claim_id, r.claim_id) AS claim_id,
                        coalesce(b.article_id, r.article_id) AS article_id,
                        b.article_id IS NOT NULL AS in_baseline,
                        r.article_id IS NOT NULL AS in_reranked
                    FROM baseline b
                    FULL OUTER JOIN reranked r USING(claim_id, article_id)
                )
                SELECT
                    (SELECT count(*) FROM baseline) AS baseline_rows,
                    (SELECT count(*) FROM reranked) AS reranked_rows,
                    count(*) FILTER (WHERE in_baseline AND NOT in_reranked)
                        AS missing_from_reranked,
                    count(*) FILTER (WHERE in_reranked AND NOT in_baseline)
                        AS added_to_reranked
                FROM compared
                """,
                [track, top_k, track, top_k],
            ).fetchone()
            missing = int(row[2])
            added = int(row[3])
            result[track] = {
                "baseline_top_k_rows": int(row[0]),
                "reranked_top_k_rows": int(row[1]),
                "missing_from_reranked": missing,
                "added_to_reranked": added,
                "candidate_set_preserved": missing == 0 and added == 0,
            }
    finally:
        connection.close()
    return result


def _score_diagnostics(
    *,
    reranked_path: Path,
    corpus_dir: Path,
    split: str,
) -> dict[str, Any]:
    duckdb = _duckdb()
    judgments = corpus_dir / "judgments.parquet"
    claims = corpus_dir / "claims.parquet"
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""
            SELECT
                j.relevance_grade,
                count(*) AS rows,
                avg(r.cross_encoder_score) AS mean_score,
                stddev_pop(r.cross_encoder_score) AS std_score,
                min(r.cross_encoder_score) AS min_score,
                max(r.cross_encoder_score) AS max_score
            FROM read_parquet('{_quote(reranked_path)}') r
            JOIN read_parquet('{_quote(judgments)}') j USING(claim_id, article_id)
            JOIN read_parquet('{_quote(claims)}') c USING(claim_id)
            WHERE r.candidate_kind = 'judged'
              AND r.track = 'overall'
              AND c.split = ?
              AND j.eligible_for_qrels
            GROUP BY j.relevance_grade
            ORDER BY j.relevance_grade
            """,
            [split],
        ).fetchall()
        by_grade = {
            str(int(row[0])): {
                "rows": int(row[1]),
                "mean_score": round(float(row[2]), 8),
                "std_score": round(float(row[3]), 8) if row[3] is not None else None,
                "min_score": round(float(row[4]), 8),
                "max_score": round(float(row[5]), 8),
            }
            for row in rows
        }
        ordered_means = [
            by_grade[key]["mean_score"] for key in ("0", "1", "2") if key in by_grade
        ]
        monotonic = all(
            right >= left for left, right in zip(ordered_means, ordered_means[1:])
        )
        pairwise = connection.execute(
            f"""
            WITH judged AS (
                SELECT
                    r.claim_id,
                    r.article_id,
                    r.cross_encoder_score,
                    j.relevance_grade
                FROM read_parquet('{_quote(reranked_path)}') r
                JOIN read_parquet('{_quote(judgments)}') j USING(claim_id, article_id)
                JOIN read_parquet('{_quote(claims)}') c USING(claim_id)
                WHERE r.candidate_kind = 'judged'
                  AND r.track = 'overall'
                  AND c.split = ?
                  AND j.eligible_for_qrels
            ), pairs AS (
                SELECT
                    CASE
                        WHEN high.cross_encoder_score > low.cross_encoder_score THEN 1.0
                        WHEN high.cross_encoder_score = low.cross_encoder_score THEN 0.5
                        ELSE 0.0
                    END AS correct
                FROM judged high
                JOIN judged low
                  ON high.claim_id = low.claim_id
                 AND high.relevance_grade > low.relevance_grade
            )
            SELECT count(*), avg(correct) FROM pairs
            """,
            [split],
        ).fetchone()
    finally:
        connection.close()
    return {
        "score_semantics": "raw single logit; higher is more relevant",
        "by_relevance_grade": by_grade,
        "mean_score_monotonic_by_grade": monotonic,
        "pairwise_grade_pairs": int(pairwise[0]),
        "pairwise_score_direction_accuracy": (
            round(float(pairwise[1]), 8) if pairwise[1] is not None else None
        ),
    }


def _phase04_markdown(report: Mapping[str, Any]) -> str:
    reranker = report["reranker"]
    lines = [
        f"# Phase 04 Cross-Encoder Reranking — {report['split']}",
        "",
        f"Run: `{report['run_name']}`  ",
        f"Model: `{reranker['model_path']}`  ",
        f"Candidate run: `{reranker['candidate_run_name']}`  ",
        f"Rerank depth: `{reranker['rerank_depth']}`",
        "",
        "| Track | MRR | nDCG@5 | Top-1 positive | Pairwise acc. | KP Recall@10 | KP Recall@100 | HitRate@100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for track in TRACKS:
        data = report["tracks"][track]
        judged = data["judged_candidate_ranking"]
        opened = data["open_corpus_incomplete_qrels"]
        pairwise = judged.get("pairwise_ordering_accuracy")
        lines.append(
            "| {track} | {mrr:.4f} | {ndcg:.4f} | {top1:.4f} | {pairwise} | "
            "{r10:.4f} | {r100:.4f} | {h100:.4f} |".format(
                track=track,
                mrr=judged.get("mrr") or 0.0,
                ndcg=judged.get("ndcg@5") or 0.0,
                top1=judged.get("top1_positive_rate") or 0.0,
                pairwise="n/a" if pairwise is None else f"{pairwise:.4f}",
                r10=opened.get("known_positive_recall@10") or 0.0,
                r100=opened.get("known_positive_recall@100") or 0.0,
                h100=opened.get("hitrate@100") or 0.0,
            )
        )
    diagnostics = reranker["score_diagnostics"]
    lines.extend(
        [
            "",
            "## Score direction diagnostic",
            "",
            f"Pairwise grade direction accuracy: `{diagnostics['pairwise_score_direction_accuracy']}`  ",
            f"Mean score monotonic by grade: `{diagnostics['mean_score_monotonic_by_grade']}`",
            "",
            "Raw MedCPT logits are ranked descending. Independent-source is the primary track.",
            "Open-corpus qrels are incomplete; Recall@100 must be reported with ranking metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def run_cross_encoder_reranking(
    root: Path,
    *,
    split: str,
    candidate_path: Path,
    candidate_run_name: str,
    run_name: str | None = None,
    corpus_dir: Path | None = None,
    article_input_dir: Path | None = None,
    model_dir: Path | None = None,
    score_root: Path | None = None,
    reranked_candidate_dir: Path | None = None,
    run_dir: Path | None = None,
    report_dir: Path | None = None,
    devices: Sequence[str] = ("0",),
    num_shards: int | None = None,
    batch_size: int = 16,
    max_length: int = 512,
    amp: str = "fp16",
    top_k: int = 100,
    rerank_depth: int | None = None,
    max_queries: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if split not in {"dev", "test"}:
        raise EvidenceGapError("split must be dev or test")
    if amp not in {"fp16", "fp32"}:
        raise EvidenceGapError("amp must be fp16 or fp32")
    if batch_size <= 0:
        raise EvidenceGapError("batch_size must be positive")
    if max_length <= 0:
        raise EvidenceGapError("max_length must be positive")
    if top_k < 100:
        raise EvidenceGapError("top_k must be at least 100 for Recall@100")
    rerank_depth = top_k if rerank_depth is None else rerank_depth
    if rerank_depth < top_k:
        raise EvidenceGapError(
            "rerank_depth must be at least top_k so every emitted open candidate "
            "has a cross-encoder score"
        )

    root = root.resolve()
    candidate_path = (root / candidate_path).resolve()
    corpus_dir = (root / (corpus_dir or DEFAULT_CORPUS_DIR)).resolve()
    article_input_dir = (
        root / (article_input_dir or DEFAULT_ARTICLE_INPUT_DIR)
    ).resolve()
    model_dir = (root / (model_dir or DEFAULT_MODEL_DIR)).resolve()
    score_root = (root / (score_root or DEFAULT_SCORE_ROOT)).resolve()
    reranked_candidate_dir = (
        root / (reranked_candidate_dir or DEFAULT_RERANKED_CANDIDATE_DIR)
    ).resolve()
    run_dir = (root / (run_dir or DEFAULT_RUN_DIR)).resolve()
    report_dir = (root / (report_dir or DEFAULT_REPORT_DIR)).resolve()
    if not candidate_path.exists():
        raise EvidenceGapError(f"Missing fused candidate parquet: {candidate_path}")
    candidate_manifest_path = candidate_path.with_suffix(".manifest.json")
    if not candidate_manifest_path.exists():
        raise EvidenceGapError(
            f"Missing fused candidate manifest: {candidate_manifest_path}"
        )
    candidate_manifest = load_json(candidate_manifest_path)
    if candidate_manifest.get("split") != split:
        raise EvidenceGapError(
            f"Candidate split mismatch: manifest has {candidate_manifest.get('split')!r}, "
            f"rerank requested {split!r}"
        )
    manifest_run_name = candidate_manifest.get("run_name")
    if manifest_run_name and manifest_run_name != candidate_run_name:
        raise EvidenceGapError(
            f"Candidate run mismatch: manifest has {manifest_run_name!r}, "
            f"--candidate-run-name is {candidate_run_name!r}"
        )
    candidate_query_limit = candidate_manifest.get("query_limit")
    if max_queries is None:
        max_queries = candidate_query_limit
    elif max_queries != candidate_query_limit:
        raise EvidenceGapError(
            "--max-queries must match the fused candidate manifest query_limit "
            f"({candidate_query_limit!r}); rebuild fusion for a different smoke size"
        )
    if not model_dir.exists():
        raise EvidenceGapError(f"Missing MedCPT cross encoder: {model_dir}")

    normalized_devices = _normalize_devices(devices)
    if (
        any(device == "cpu" for device in normalized_devices)
        and len(normalized_devices) > 1
    ):
        raise EvidenceGapError("CPU cannot be mixed with CUDA devices")
    num_shards = num_shards or len(normalized_devices)
    if num_shards <= 0:
        raise EvidenceGapError("num_shards must be positive")
    run_name = safe_run_name(run_name or f"medcpt_cross_{candidate_run_name}")
    work_dir = score_root / f"{split}_{run_name}"
    input_dir = work_dir / "inputs"
    score_dir = work_dir / "scores"
    reranked_path = reranked_candidate_dir / f"{split}_{run_name}.parquet"
    reranked_manifest_path = (
        reranked_candidate_dir / f"{split}_{run_name}.manifest.json"
    )

    input_manifest, input_paths = _prepare_inputs(
        root,
        candidate_path=candidate_path,
        corpus_dir=corpus_dir,
        article_input_dir=article_input_dir,
        split=split,
        num_shards=num_shards,
        rerank_depth=rerank_depth,
        input_dir=input_dir,
        force=force,
    )
    model_fingerprint = _model_fingerprint(model_dir)
    tasks, reused = _validate_or_plan_scores(
        input_paths=input_paths,
        score_dir=score_dir,
        model_fingerprint=model_fingerprint,
        max_length=max_length,
        amp=amp,
        force=force,
    )
    created = _run_scoring(
        tasks=tasks,
        devices=normalized_devices,
        model_dir=model_dir,
        model_fingerprint=model_fingerprint,
        batch_size=batch_size,
        max_length=max_length,
        amp=amp,
    )
    shard_metadata = sorted([*reused, *created], key=lambda row: str(row["input_path"]))
    if len(shard_metadata) != len(input_paths):
        raise EvidenceGapError(
            f"Expected {len(input_paths)} scored shards, got {len(shard_metadata)}"
        )
    score_rows = sum(int(row["rows"]) for row in shard_metadata)
    if score_rows != int(input_manifest["rows"]):
        raise EvidenceGapError(
            f"Scored pair count mismatch: {score_rows:,}/{input_manifest['rows']:,}"
        )

    rerank_stats = _write_reranked_candidates(
        candidate_path=candidate_path,
        score_dir=score_dir,
        output_path=reranked_path,
        rerank_depth=rerank_depth,
        force=force,
    )
    _write_reranked_trec(
        reranked_path=reranked_path,
        run_dir=run_dir,
        split=split,
        run_name=run_name,
        top_k=top_k,
        force=force,
    )
    candidate_set_diagnostics = _candidate_set_diagnostics(
        candidate_path=candidate_path,
        reranked_path=reranked_path,
        top_k=top_k,
    )
    if rerank_depth == top_k:
        broken_tracks = [
            track
            for track, values in candidate_set_diagnostics.items()
            if not values["candidate_set_preserved"]
        ]
        if broken_tracks:
            raise EvidenceGapError(
                "Top-K candidate-set preservation failed for tracks: "
                + ", ".join(broken_tracks)
            )
    diagnostics = _score_diagnostics(
        reranked_path=reranked_path, corpus_dir=corpus_dir, split=split
    )

    manifest = {
        "schema_version": CROSS_ENCODER_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "run_name": run_name,
        "candidate_run_name": candidate_run_name,
        "candidate_path": relative_path(root, candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "model_path": relative_path(root, model_dir),
        "model_fingerprint": model_fingerprint,
        "max_length": max_length,
        "top_k": top_k,
        "rerank_depth": rerank_depth,
        "rerank_policy": (
            "cross-encoder orders the fused open candidates through rerank_depth; "
            "remaining union candidates retain fusion order after that block"
        ),
        "amp": amp,
        "batch_size": batch_size,
        "devices": normalized_devices,
        "num_shards": num_shards,
        "score_semantics": "raw single logit; higher is more relevant",
        "input_manifest": relative_path(root, input_dir / "input_manifest.json"),
        "score_shards": [
            {
                "input_path": relative_path(root, Path(row["input_path"])),
                "output_path": relative_path(root, Path(row["output_path"])),
                "rows": int(row["rows"]),
                "seconds": row.get("seconds"),
                "pairs_per_second": row.get("pairs_per_second"),
                "device": row.get("device"),
            }
            for row in shard_metadata
        ],
        "rows_scored": score_rows,
        "rerank_stats": rerank_stats,
        "candidate_set_diagnostics": candidate_set_diagnostics,
        "reranked_candidates": {
            "path": relative_path(root, reranked_path),
            "sha256": sha256_file(reranked_path),
            "bytes": reranked_path.stat().st_size,
        },
        "score_diagnostics": diagnostics,
    }
    reranked_candidate_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(reranked_manifest_path, manifest)

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
    report["retriever"] = {"family": "cross_encoder_reranker"}
    report["reranker"] = {
        "model_path": relative_path(root, model_dir),
        "model_fingerprint": model_fingerprint,
        "candidate_run_name": candidate_run_name,
        "candidate_path": relative_path(root, candidate_path),
        "reranked_candidate_manifest": relative_path(root, reranked_manifest_path),
        "score_semantics": "raw single logit; higher is more relevant",
        "max_length": max_length,
        "top_k": top_k,
        "rerank_depth": rerank_depth,
        "rerank_policy": (
            "cross-encoder orders the fused open candidates through rerank_depth; "
            "remaining union candidates retain fusion order after that block"
        ),
        "amp": amp,
        "batch_size": batch_size,
        "devices": normalized_devices,
        "num_shards": num_shards,
        "rows_scored": score_rows,
        "rerank_stats": rerank_stats,
        "candidate_set_diagnostics": candidate_set_diagnostics,
        "score_diagnostics": diagnostics,
    }
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report_stem = f"article_retrieval_{run_name}_{split}"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{report_stem}.json"
    atomic_write_json(report_path, report)
    (report_dir / f"{report_stem}.md").write_text(
        _phase04_markdown(report), encoding="utf-8"
    )
    print(f"Cross-encoder manifest: {reranked_manifest_path}")
    print(f"Cross-encoder report: {report_path}")
    return report
