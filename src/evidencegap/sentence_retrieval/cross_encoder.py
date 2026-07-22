from __future__ import annotations

import math
import multiprocessing as mp
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evidencegap.common import EvidenceGapError, atomic_write_json, manifest_fingerprint, relative_path, sha256_file
from evidencegap.sentence_retrieval.artifacts import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    RUN_SCHEMA_VERSION,
    combine_shards,
    read_rows_by_query,
    reuse_or_reject_shard,
    safe_run_name,
    validate_ranking_rows,
    write_rows_atomic,
    write_shard_metadata,
)
from evidencegap.sentence_retrieval.contracts import EvidenceQuery, SCHEMA_VERSION, TASK_ID
from evidencegap.sentence_retrieval.dense import normalize_devices
from evidencegap.sentence_retrieval.evaluation import evaluate_sentence_run
from evidencegap.sentence_retrieval.evidencebench import load_canonical_queries

DEFAULT_MODEL_DIR = Path("models/v1/medcpt-cross")


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
    files = [path for path in model_dir.iterdir() if path.is_file() and path.name in names]
    files.extend(sorted(model_dir.glob("*.safetensors")))
    if not files:
        raise EvidenceGapError(f"No cross-encoder model assets found under {model_dir}")
    if not tuple(model_dir.glob("*.safetensors")):
        raise EvidenceGapError(
            f"MedCPT cross encoder requires safetensors under {model_dir}; "
            "do not fall back to pytorch_model.bin"
        )
    return sorted(set(files))


def _load_model(model_dir: Path, *, device: str, amp: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing torch/transformers. Install requirements/v1-phase05.txt"
        ) from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise EvidenceGapError(f"CUDA requested but unavailable: {device}")
    if device == "cpu" and amp == "fp16":
        raise EvidenceGapError("fp16 cross-encoder inference is not supported on CPU")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        use_safetensors=True,
    )
    if int(getattr(model.config, "num_labels", 0)) != 1:
        raise EvidenceGapError(
            f"Cross encoder must expose one relevance logit; got {model.config.num_labels}"
        )
    model.eval().to(device)
    if amp == "fp16":
        model.half()
    return torch, tokenizer, model


def _stable_shard(query_id: str, num_shards: int) -> int:
    import hashlib
    return int(hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:16], 16) % num_shards


def _validate_candidate_rows(
    query: EvidenceQuery, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["final_rank"]))
    seen: set[int] = set()
    for row in ordered:
        if str(row["paper_id"]) != query.paper_id:
            raise EvidenceGapError(f"Candidate paper mismatch for {query.query_id}")
        if str(row["pool_fingerprint"]) != query.pool_fingerprint:
            raise EvidenceGapError(f"Candidate pool mismatch for {query.query_id}")
        index = int(row["sentence_index"])
        if index in seen:
            raise EvidenceGapError(f"Duplicate candidate sentence for {query.query_id}: {index}")
        seen.add(index)
        if index < 0 or index >= len(query.candidate_sentences):
            raise EvidenceGapError(f"Invalid candidate sentence index for {query.query_id}: {index}")
        if str(row["sentence_text"]) != query.candidate_sentences[index]:
            raise EvidenceGapError(f"Candidate sentence text changed for {query.query_id}:{index}")
    if not ordered:
        raise EvidenceGapError(f"Candidate run has no rows for {query.query_id}")
    return ordered


def _score_pairs(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    hypothesis: str,
    sentences: Sequence[str],
    batch_size: int,
    max_length: int,
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(sentences), batch_size):
        batch = list(sentences[start : start + batch_size])
        encoded = tokenizer(
            [hypothesis] * len(batch),
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise EvidenceGapError(f"Unexpected cross-encoder logits shape: {tuple(logits.shape)}")
        values = [float(value) for value in logits[:, 0].float().cpu().tolist()]
        if any(not math.isfinite(value) for value in values):
            raise EvidenceGapError("Cross encoder produced a non-finite score")
        scores.extend(values)
    return scores


def _reranked_rows(
    query: EvidenceQuery,
    candidates: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    split: str,
    run_name: str,
    rerank_depth: int,
) -> Iterable[dict[str, Any]]:
    scored_count = min(rerank_depth, len(candidates))
    scored = []
    for row, score in zip(candidates[:scored_count], scores):
        scored.append((dict(row), float(score)))
    scored.sort(
        key=lambda item: (
            -item[1],
            int(item[0]["retrieval_rank"]),
            int(item[0]["sentence_index"]),
        )
    )
    final_items: list[tuple[dict[str, Any], float | None]] = scored + [
        (dict(row), None) for row in candidates[scored_count:]
    ]
    for final_rank, (row, cross_score) in enumerate(final_items, start=1):
        yield {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "split": split,
            "run_name": run_name,
            "query_id": query.query_id,
            "paper_id": query.paper_id,
            "pool_fingerprint": query.pool_fingerprint,
            "sentence_index": int(row["sentence_index"]),
            "sentence_type": str(row["sentence_type"]),
            "sentence_text": str(row["sentence_text"]),
            "retrieval_model": str(row["retrieval_model"]),
            "retrieval_score": float(row["retrieval_score"]),
            "retrieval_rank": int(row["retrieval_rank"]),
            "cross_encoder_score": cross_score,
            "final_score": cross_score,
            "final_rank": final_rank,
        }


def _worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    output_path = Path(payload["output_path"])
    metadata_path = Path(payload["metadata_path"])
    signature = dict(payload["signature"])
    reused = reuse_or_reject_shard(
        output_path=output_path,
        metadata_path=metadata_path,
        expected_signature=signature,
        force=bool(payload["force"]),
    )
    if reused is not None:
        return reused
    device = str(payload["device"])
    amp = str(payload["amp"])
    entries = payload["entries"]
    if not entries:
        row_count = write_rows_atomic(output_path, ())
        metadata = {
            **signature,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "queries": 0,
            "rows": row_count,
            "output_sha256": sha256_file(output_path),
        }
        write_shard_metadata(metadata_path, metadata)
        return metadata
    torch, tokenizer, model = _load_model(Path(payload["model_dir"]), device=device, amp=amp)

    def rows() -> Iterable[dict[str, Any]]:
        for entry in entries:
            query = EvidenceQuery.from_dict(entry["query"])
            candidates = _validate_candidate_rows(query, entry["candidates"])
            scored_count = min(int(payload["rerank_depth"]), len(candidates))
            scores = _score_pairs(
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                hypothesis=query.hypothesis,
                sentences=[str(row["sentence_text"]) for row in candidates[:scored_count]],
                batch_size=int(payload["batch_size"]),
                max_length=int(payload["max_length"]),
            )
            yield from _reranked_rows(
                query,
                candidates,
                scores,
                split=str(payload["split"]),
                run_name=str(payload["run_name"]),
                rerank_depth=int(payload["rerank_depth"]),
            )

    row_count = write_rows_atomic(output_path, rows())
    metadata = {
        **signature,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "queries": len(entries),
        "rows": row_count,
        "output_sha256": sha256_file(output_path),
    }
    write_shard_metadata(metadata_path, metadata)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return metadata


def _worker_group(payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_worker(payload) for payload in payloads]


def run_cross_encoder_sentence_reranking(
    root: Path,
    *,
    split: str,
    canonical_dir: Path,
    candidate_path: Path,
    devices: Sequence[str],
    run_name: str | None = None,
    candidate_run_name: str | None = None,
    model_dir: Path | None = None,
    rerank_depth: int = 20,
    num_shards: int | None = None,
    batch_size: int = 16,
    max_length: int = 512,
    amp: str = "fp16",
    artifact_root: Path | None = None,
    report_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if rerank_depth <= 0 or batch_size <= 0 or max_length <= 0:
        raise EvidenceGapError("rerank_depth/batch_size/max_length must be positive")
    root = root.resolve()
    canonical_dir = canonical_dir.resolve()
    candidate_path = candidate_path.resolve()
    queries, canonical_manifest = load_canonical_queries(canonical_dir)
    candidate_rows = read_rows_by_query(candidate_path)
    expected_ids = {query.query_id for query in queries}
    if set(candidate_rows) != expected_ids:
        raise EvidenceGapError(
            f"Candidate run query coverage mismatch: missing={len(expected_ids-set(candidate_rows))}, "
            f"extra={len(set(candidate_rows)-expected_ids)}"
        )
    normalized_devices = normalize_devices(devices)
    if any(device == "cpu" for device in normalized_devices) and amp == "fp16":
        raise EvidenceGapError("fp16 cross-encoder inference is not supported on CPU")
    shard_count = num_shards or len(normalized_devices)
    model_path = (model_dir.resolve() if model_dir else root / DEFAULT_MODEL_DIR)
    model_fp = manifest_fingerprint(_model_files(model_path))
    source_name = candidate_run_name or candidate_path.parent.name
    name = safe_run_name(run_name or f"medcpt_cross_{source_name}")
    base_root = artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT
    run_base = base_root / "reranked" / name
    if force and run_base.exists():
        shutil.rmtree(run_base)
    shard_dir = run_base / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    partitions: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for query in queries:
        partition = _stable_shard(query.query_id, shard_count)
        partitions[partition].append(
            {"query": query.to_dict(), "candidates": candidate_rows[query.query_id]}
        )

    payloads: list[dict[str, Any]] = []
    shard_paths: list[Path] = []
    for shard_index, entries in enumerate(partitions):
        output_path = shard_dir / f"shard-{shard_index:05d}-of-{shard_count:05d}.parquet"
        metadata_path = output_path.with_suffix(".json")
        shard_paths.append(output_path)
        signature = {
            "schema_version": RUN_SCHEMA_VERSION,
            "task_id": TASK_ID,
            "run_name": name,
            "split": split,
            "model_fingerprint": model_fp,
            "canonical_sha256": canonical_manifest["canonical_sha256"],
            "candidate_sha256": sha256_file(candidate_path),
            "shard_index": shard_index,
            "num_shards": shard_count,
            "rerank_depth": rerank_depth,
            "max_length": max_length,
            "amp": amp,
            "assignment": "sha256(query_id) modulo num_shards",
        }
        payloads.append(
            {
                "output_path": str(output_path),
                "metadata_path": str(metadata_path),
                "signature": signature,
                "force": force,
                "device": normalized_devices[shard_index % len(normalized_devices)],
                "amp": amp,
                "model_dir": str(model_path),
                "entries": entries,
                "rerank_depth": rerank_depth,
                "batch_size": batch_size,
                "max_length": max_length,
                "split": split,
                "run_name": name,
            }
        )
    groups = [
        [payload for payload in payloads if payload["device"] == device]
        for device in normalized_devices
    ]
    groups = [group for group in groups if group]
    if len(groups) == 1:
        nested_metadata = [_worker_group(groups[0])]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=len(groups)) as pool:
            nested_metadata = pool.map(_worker_group, groups)
    shard_metadata = [item for group in nested_metadata for item in group]
    shard_metadata.sort(key=lambda item: int(item["shard_index"]))

    output_path = run_base / "ranked_sentences.parquet"
    validation = combine_shards(
        shard_paths,
        output_path,
        expected_queries={query.query_id: len(query.candidate_sentences) for query in queries},
        run_name=name,
        force=force,
    )
    validation = validate_ranking_rows(
        output_path,
        expected_queries={query.query_id: len(query.candidate_sentences) for query in queries},
        expected_depths={
            query.query_id: len(candidate_rows[query.query_id]) for query in queries
        },
        expected_run_name=name,
    )
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "run_name": name,
        "split": split,
        "candidate_run_name": source_name,
        "candidate_path": relative_path(root, candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "canonical_dir": relative_path(root, canonical_dir),
        "canonical_sha256": canonical_manifest["canonical_sha256"],
        "model_dir": relative_path(root, model_path),
        "model_fingerprint": model_fp,
        "parameters": {
            "rerank_depth": rerank_depth,
            "batch_size": batch_size,
            "max_length": max_length,
            "amp": amp,
            "devices": normalized_devices,
            "num_shards": shard_count,
            "pair_format": "tokenizer(hypothesis, exact_candidate_sentence)",
            "score_semantics": "raw_single_logit_higher_is_more_relevant",
            "candidate_contract": (
                "only candidate retrieval Top-N is reranked; remaining rows preserve retrieval order"
            ),
        },
        "queries": len(queries),
        "rows": validation["rows"],
        "output_path": relative_path(root, output_path),
        "output_sha256": validation["sha256"],
        "shards": shard_metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = run_base / "run_manifest.json"
    atomic_write_json(manifest_path, manifest)
    report_root = report_dir.resolve() if report_dir else root / DEFAULT_REPORT_ROOT
    report_path = report_root / f"evidence_sentence_retrieval_{name}_{split}.json"
    evaluation = evaluate_sentence_run(
        root,
        canonical_dir=canonical_dir,
        run_path=output_path,
        report_path=report_path,
    )
    return {
        "run_name": name,
        "run_path": relative_path(root, output_path),
        "manifest_path": relative_path(root, manifest_path),
        "report_path": relative_path(root, report_path),
        "validation": validation,
        "metrics": evaluation["metrics"],
    }
