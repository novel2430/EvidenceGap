from __future__ import annotations

import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    manifest_fingerprint,
    relative_path,
    sha256_file,
)
from evidencegap.stance.artifacts import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    RUN_SCHEMA_VERSION,
    iter_inputs,
    validate_input_artifact,
    validate_prediction_artifact,
    write_predictions_atomic,
)
from evidencegap.stance.contracts import (
    SCHEMA_VERSION,
    STANCE_LABELS,
    TASK_ID,
    StanceInput,
    StancePrediction,
)
from evidencegap.stance.evaluation import (
    evaluate_prediction_rows,
    render_evaluation_markdown,
)

DEFAULT_MODEL_DIR = Path("models/v1/verifier-deberta-v3-base")
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"


def _safe_name(value: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise EvidenceGapError(f"Invalid run name: {value!r}")
    return cleaned


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
        "sentencepiece.bpe.model",
        "spiece.model",
        "spm.model",
        "tokenizer.model",
    }
    try:
        files = [
            path for path in model_dir.iterdir() if path.is_file() and path.name in names
        ]
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing verifier model directory: {model_dir}") from exc
    weight_files = sorted(model_dir.glob("*.safetensors"))
    if not weight_files:
        raise EvidenceGapError(
            f"Verifier requires safetensors under {model_dir}; "
            "do not fall back to pytorch_model.bin"
        )
    files.extend(weight_files)
    if not any(path.name == "config.json" for path in files):
        raise EvidenceGapError(f"Verifier model has no config.json: {model_dir}")
    return sorted(set(files))


def _normalize_label_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def resolve_nli_label_indices(config: Any) -> dict[str, int]:
    raw = getattr(config, "id2label", None)
    if not isinstance(raw, Mapping):
        raise EvidenceGapError("NLI model config has no id2label mapping")
    resolved: dict[str, int] = {}
    aliases = {
        "entailment": "entailment",
        "entails": "entailment",
        "contradiction": "contradiction",
        "contradictory": "contradiction",
        "neutral": "neutral",
    }
    for raw_index, raw_label in raw.items():
        normalized = _normalize_label_name(raw_label)
        canonical = aliases.get(normalized)
        if canonical is not None:
            resolved[canonical] = int(raw_index)
    expected = {"entailment", "contradiction", "neutral"}
    if set(resolved) != expected:
        raise EvidenceGapError(
            "Verifier config must explicitly identify entailment, contradiction, and neutral; "
            f"got id2label={dict(raw)!r}"
        )
    if len(set(resolved.values())) != 3:
        raise EvidenceGapError("NLI label indices are not unique")
    return resolved


def _load_model(
    model_dir: Path, *, device: str, amp: str
) -> tuple[Any, Any, Any, dict[str, int]]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing torch/transformers. Install requirements/v1-phase06.txt"
        ) from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise EvidenceGapError(f"CUDA requested but unavailable: {device}")
    if device == "cpu" and amp != "none":
        raise EvidenceGapError("CPU zero-shot inference requires --amp none")
    if amp not in {"none", "fp16", "bf16"}:
        raise EvidenceGapError("amp must be none, fp16, or bf16")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        use_safetensors=True,
    )
    if int(getattr(model.config, "num_labels", 0)) != 3:
        raise EvidenceGapError(
            f"Verifier must expose three NLI logits; got {model.config.num_labels}"
        )
    label_indices = resolve_nli_label_indices(model.config)
    model.eval().to(device)
    if amp == "fp16":
        model.half()
    elif amp == "bf16":
        model.to(dtype=torch.bfloat16)
    return torch, tokenizer, model, label_indices


def _score_batch(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    label_indices: Mapping[str, int],
    device: str,
    inputs: Sequence[StanceInput],
    max_length: int,
) -> list[tuple[float, float, float]]:
    # NLI contract: evidence is the premise (sequence A), claim is the
    # hypothesis (sequence B). Reversing this order changes the task.
    encoded = tokenizer(
        [item.evidence_text for item in inputs],
        [item.claim_text for item in inputs],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits.float()
        probabilities = torch.softmax(logits, dim=-1).cpu()
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise EvidenceGapError(
            f"Unexpected verifier probability shape: {tuple(probabilities.shape)}"
        )
    results: list[tuple[float, float, float]] = []
    for row in probabilities.tolist():
        support = float(row[label_indices["entailment"]])
        refute = float(row[label_indices["contradiction"]])
        insufficient = float(row[label_indices["neutral"]])
        values = (support, refute, insufficient)
        if any(not math.isfinite(value) for value in values):
            raise EvidenceGapError("Verifier produced a non-finite probability")
        results.append(values)
    return results


def _prediction(
    stance_input: StanceInput,
    probabilities: tuple[float, float, float],
    *,
    run_name: str,
    model_fingerprint: str,
    stance_input_artifact_sha256: str,
) -> StancePrediction:
    support, refute, insufficient = probabilities
    by_label = {
        "support": support,
        "refute": refute,
        "insufficient": insufficient,
    }
    label_order = {label: index for index, label in enumerate(STANCE_LABELS)}
    ordered = sorted(
        by_label.items(), key=lambda item: (-item[1], label_order[item[0]])
    )
    predicted_label, confidence = ordered[0]
    margin = confidence - ordered[1][1]
    return StancePrediction(
        stance_input=stance_input,
        run_name=run_name,
        model_name=MODEL_NAME,
        model_fingerprint=model_fingerprint,
        stance_input_artifact_sha256=stance_input_artifact_sha256,
        predicted_label=predicted_label,
        probability_support=support,
        probability_refute=refute,
        probability_insufficient=insufficient,
        confidence=confidence,
        probability_margin=margin,
        abstained=False,
    )


def run_deberta_zero_shot(
    root: Path,
    *,
    input_path: Path,
    run_name: str | None = None,
    model_dir: Path | None = None,
    device: str = "cuda:0",
    batch_size: int = 16,
    max_length: int = 512,
    amp: str = "fp16",
    artifact_root: Path | None = None,
    report_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    input_path = input_path.resolve()
    model_dir = (model_dir or (root / DEFAULT_MODEL_DIR)).resolve()
    if batch_size <= 0:
        raise EvidenceGapError("batch_size must be positive")
    if max_length <= 0:
        raise EvidenceGapError("max_length must be positive")
    input_validation = validate_input_artifact(input_path)
    input_sha = input_validation["sha256"]
    name = _safe_name(run_name or f"deberta_nli_zero_shot_{input_path.parent.name}")
    base = artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT / "zero_shot"
    target = base / name
    model_files = _model_files(model_dir)
    model_fp = manifest_fingerprint(model_files)
    torch, tokenizer, model, label_indices = _load_model(
        model_dir, device=device, amp=amp
    )
    started = time.perf_counter()
    prediction_counts: Counter[str] = Counter()
    gold_rows: list[dict[str, Any]] = []
    confidence_sum = 0.0
    margin_sum = 0.0

    def predictions() -> Iterator[StancePrediction]:
        nonlocal confidence_sum, margin_sum
        batch: list[StanceInput] = []
        for stance_input in iter_inputs(input_path):
            batch.append(stance_input)
            if len(batch) < batch_size:
                continue
            for prediction in score_and_yield(batch):
                yield prediction
            batch.clear()
        if batch:
            yield from score_and_yield(batch)

    def score_and_yield(batch: Sequence[StanceInput]) -> Iterable[StancePrediction]:
        nonlocal confidence_sum, margin_sum
        values = _score_batch(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            label_indices=label_indices,
            device=device,
            inputs=batch,
            max_length=max_length,
        )
        for stance_input, probabilities in zip(batch, values):
            prediction = _prediction(
                stance_input,
                probabilities,
                run_name=name,
                model_fingerprint=model_fp,
                stance_input_artifact_sha256=input_sha,
            )
            prediction_counts[prediction.predicted_label] += 1
            confidence_sum += prediction.confidence
            margin_sum += prediction.probability_margin
            if stance_input.gold_label is not None:
                gold_rows.append(prediction.to_dict())
            yield prediction

    with atomic_directory(target, force=force) as staging:
        output_path = staging / "stance_predictions.parquet"
        row_count = write_predictions_atomic(output_path, predictions())
        elapsed = time.perf_counter() - started
        validation = validate_prediction_artifact(
            output_path,
            expected_input_sha256=input_sha,
            expected_run_name=name,
        )
        metrics = evaluate_prediction_rows(gold_rows) if gold_rows else None
        report = {
            "schema_version": RUN_SCHEMA_VERSION,
            "stance_schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "run_name": name,
            "run_type": "deberta_nli_zero_shot",
            "model": {
                "name": MODEL_NAME,
                "path": relative_path(root, model_dir),
                "fingerprint": model_fp,
                "id2label": {
                    key: int(value) for key, value in sorted(label_indices.items())
                },
                "input_contract": "premise=evidence_text; hypothesis=claim_text",
            },
            "parameters": {
                "device": device,
                "batch_size": batch_size,
                "max_length": max_length,
                "amp": amp,
            },
            "source_input_path": relative_path(root, input_path),
            "source_input_sha256": input_sha,
            "rows": row_count,
            "prediction_counts": dict(sorted(prediction_counts.items())),
            "mean_confidence": confidence_sum / row_count,
            "mean_probability_margin": margin_sum / row_count,
            "elapsed_seconds": elapsed,
            "rows_per_second": row_count / elapsed if elapsed > 0 else None,
            "metrics": metrics,
            "output_path": relative_path(root, target / "stance_predictions.parquet"),
            "output_sha256": validation["sha256"],
            "validation": validation,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(staging / "run_manifest.json", report)

    report_root = report_dir.resolve() if report_dir else root / DEFAULT_REPORT_ROOT
    report_root.mkdir(parents=True, exist_ok=True)
    json_report_path = report_root / f"stance_zero_shot_{name}.json"
    atomic_write_json(json_report_path, report)
    markdown_path: Path | None = None
    if report["metrics"] is not None:
        markdown_path = report_root / f"stance_zero_shot_{name}.md"
        markdown_result = {
            "prediction_path": report["output_path"],
            "metrics": report["metrics"],
        }
        markdown_path.write_text(
            render_evaluation_markdown(markdown_result), encoding="utf-8"
        )
    return {
        "run_name": name,
        "prediction_path": report["output_path"],
        "manifest_path": relative_path(root, target / "run_manifest.json"),
        "report_path": relative_path(root, json_report_path),
        "markdown_report_path": (
            None if markdown_path is None else relative_path(root, markdown_path)
        ),
        "validation": validation,
        "metrics": report["metrics"],
    }
