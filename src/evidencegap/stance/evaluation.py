from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from evidencegap.common import EvidenceGapError, atomic_write_json, relative_path
from evidencegap.stance.artifacts import (
    iter_prediction_rows,
    validate_prediction_artifact,
)
from evidencegap.stance.contracts import STANCE_LABELS, canonical_stance_label


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_prediction_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    gold_rows = [row for row in rows if row.get("gold_label") is not None]
    if not gold_rows:
        raise EvidenceGapError("Stance predictions contain no gold labels")
    labels = list(STANCE_LABELS)
    matrix = {gold: {predicted: 0 for predicted in labels} for gold in labels}
    for row in gold_rows:
        gold = canonical_stance_label(str(row["gold_label"]))
        predicted = canonical_stance_label(str(row["predicted_label"]))
        matrix[gold][predicted] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    total_correct = 0
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[gold][label] for gold in labels if gold != label)
        fn = sum(matrix[label][predicted] for predicted in labels if predicted != label)
        support = sum(matrix[label].values())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)
        recall_values.append(recall)
        total_correct += tp

    prediction_counts = Counter(
        canonical_stance_label(str(row["predicted_label"])) for row in gold_rows
    )
    gold_counts = Counter(
        canonical_stance_label(str(row["gold_label"])) for row in gold_rows
    )
    return {
        "eligible_rows": len(gold_rows),
        "accuracy": total_correct / len(gold_rows),
        "macro_f1": sum(f1_values) / len(f1_values),
        "balanced_accuracy": sum(recall_values) / len(recall_values),
        "per_class": per_class,
        "gold_counts": {label: gold_counts[label] for label in labels},
        "prediction_counts": {label: prediction_counts[label] for label in labels},
        "confusion_matrix": {
            "label_order": labels,
            "rows_are_gold": True,
            "values": [[matrix[gold][predicted] for predicted in labels] for gold in labels],
        },
    }


def evaluate_stance_predictions(
    root: Path,
    *,
    prediction_path: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    prediction_path = prediction_path.resolve()
    validation = validate_prediction_artifact(prediction_path)
    rows = list(iter_prediction_rows(prediction_path))
    metrics = evaluate_prediction_rows(rows)
    result = {
        "task": "stance_verification_3class",
        "prediction_path": relative_path(root, prediction_path),
        "validation": validation,
        "metrics": metrics,
    }
    if report_path is not None:
        report_path = report_path.resolve()
        atomic_write_json(report_path, result)
    return result


def render_evaluation_markdown(
    result: Mapping[str, Any],
    *,
    title: str = "Phase 06 Zero-shot Stance Evaluation",
) -> str:
    metrics = result["metrics"]
    lines = [
        f"# {title}",
        "",
        f"- Eligible rows: {metrics['eligible_rows']:,}",
        f"- Accuracy: {metrics['accuracy']:.5f}",
        f"- Macro-F1: {metrics['macro_f1']:.5f}",
        f"- Balanced Accuracy: {metrics['balanced_accuracy']:.5f}",
        "",
        "| Label | Support | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in STANCE_LABELS:
        row = metrics["per_class"][label]
        lines.append(
            f"| {label} | {row['support']:,} | {row['precision']:.5f} | "
            f"{row['recall']:.5f} | {row['f1']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Confusion matrix",
            "",
            "Rows are gold labels; columns are predicted labels.",
            "",
            "```json",
            json.dumps(metrics["confusion_matrix"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
