#!/usr/bin/env python3
"""
Audit the three raw EvidenceGap V1 datasets without loading them fully into RAM.

Checks:
- MedFact-Synth: 17 readable Parquet shards, consistent schema, total row count.
- EvidenceBench-100k: valid streaming JSON, expected train/test record counts.
- HealthFC: readable CSV, row count, columns, label distributions.
- Writes data/raw/v1/audit_report.json.

Dependencies:
  python -m pip install pyarrow pandas ijson
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import ijson
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "Missing dependency.\n"
        "Install with:\n"
        "  python -m pip install pyarrow pandas ijson"
    ) from exc


EXPECTED_MEDFACT_SHARDS = 17
EXPECTED_MEDFACT_ROWS = 1_497_981
EXPECTED_EVIDENCEBENCH_TRAIN = 87_461
EXPECTED_EVIDENCEBENCH_TEST = 20_000
EXPECTED_HEALTHFC_ROWS = 750


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def audit_medfact(root: Path) -> dict[str, Any]:
    data_dir = root / "data/raw/v1/medfact_synth/data"
    shards = sorted(data_dir.glob("*.parquet"))
    result: dict[str, Any] = {
        "path": str(data_dir),
        "expected_shards": EXPECTED_MEDFACT_SHARDS,
        "actual_shards": len(shards),
        "expected_rows": EXPECTED_MEDFACT_ROWS,
        "shards": [],
        "errors": [],
    }

    schema_signatures: Counter[str] = Counter()
    total_rows = 0
    sample: list[dict[str, Any]] = []

    for shard in shards:
        try:
            metadata = pq.read_metadata(shard)
            schema = pq.read_schema(shard)
            names = schema.names
            signature = json.dumps(names, ensure_ascii=False)
            schema_signatures[signature] += 1
            total_rows += metadata.num_rows

            result["shards"].append(
                {
                    "name": shard.name,
                    "bytes": shard.stat().st_size,
                    "rows": metadata.num_rows,
                    "row_groups": metadata.num_row_groups,
                    "columns": names,
                }
            )

            if not sample:
                table = pq.read_table(shard, columns=names[: min(8, len(names))])
                sample = clean_json(table.slice(0, 2).to_pylist())
        except Exception as exc:
            result["errors"].append(f"{shard.name}: {type(exc).__name__}: {exc}")

    result["actual_rows"] = total_rows
    result["schema_variants"] = [
        {"columns": json.loads(signature), "shards": count}
        for signature, count in schema_signatures.items()
    ]
    result["sample"] = sample
    result["ok"] = (
        len(shards) == EXPECTED_MEDFACT_SHARDS
        and total_rows == EXPECTED_MEDFACT_ROWS
        and len(schema_signatures) == 1
        and not result["errors"]
    )
    return result


def stream_json_records(
    path: Path,
    *,
    sample_limit: int = 3,
) -> tuple[int, Counter[tuple[str, ...]], list[dict[str, Any]], str]:
    """
    Stream either:
    - a top-level JSON array; or
    - a top-level JSON object mapping record IDs to datapoints.

    EvidenceBench uses the second form:
      {"evidencebench_train_id_0": {...}, ...}
    """
    count = 0
    key_signatures: Counter[tuple[str, ...]] = Counter()
    sample: list[dict[str, Any]] = []

    with path.open("rb") as handle:
        first = handle.read(4096).lstrip()[:1]

    if first == b"[":
        container_type = "array"
        with path.open("rb") as handle:
            iterator = ((None, item) for item in ijson.items(handle, "item"))
            for record_id, item in iterator:
                count += 1
                if isinstance(item, dict):
                    keys = tuple(sorted(str(k) for k in item.keys()))
                    key_signatures[keys] += 1
                    if len(sample) < sample_limit:
                        sample.append(clean_json(item))
                elif len(sample) < sample_limit:
                    sample.append({"_non_object_value": clean_json(item)})

    elif first == b"{":
        container_type = "object"
        with path.open("rb") as handle:
            for record_id, item in ijson.kvitems(handle, ""):
                count += 1
                if isinstance(item, dict):
                    keys = tuple(sorted(str(k) for k in item.keys()))
                    key_signatures[keys] += 1
                    if len(sample) < sample_limit:
                        row = {"_record_id": str(record_id), **clean_json(item)}
                        sample.append(row)
                elif len(sample) < sample_limit:
                    sample.append(
                        {
                            "_record_id": str(record_id),
                            "_non_object_value": clean_json(item),
                        }
                    )
    else:
        raise RuntimeError(
            f"{path.name} is not a top-level JSON array or object"
        )

    return count, key_signatures, sample, container_type


def audit_evidencebench_file(path: Path, expected_rows: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": file_size(path),
        "expected_rows": expected_rows,
        "errors": [],
    }
    try:
        count, signatures, sample, container_type = stream_json_records(path)
        result["actual_rows"] = count
        result["container_type"] = container_type
        result["key_variants"] = [
            {"keys": list(keys), "rows": rows}
            for keys, rows in signatures.most_common()
        ]
        result["sample"] = sample
        result["ok"] = count == expected_rows and bool(signatures)
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["actual_rows"] = 0
        result["container_type"] = "unknown"
        result["key_variants"] = []
        result["sample"] = []
        result["ok"] = False
    return result


def audit_evidencebench(root: Path) -> dict[str, Any]:
    data_dir = root / "data/raw/v1/evidencebench_100k"
    train = audit_evidencebench_file(
        data_dir / "evidencebench_100k_train_set.json",
        EXPECTED_EVIDENCEBENCH_TRAIN,
    )
    test = audit_evidencebench_file(
        data_dir / "evidencebench_100k_test_set.json",
        EXPECTED_EVIDENCEBENCH_TEST,
    )
    return {
        "path": str(data_dir),
        "train": train,
        "test": test,
        "actual_rows": train["actual_rows"] + test["actual_rows"],
        "expected_rows": EXPECTED_EVIDENCEBENCH_TRAIN + EXPECTED_EVIDENCEBENCH_TEST,
        "ok": bool(train["ok"] and test["ok"]),
    }


def detect_csv(path: Path) -> tuple[str, csv.Dialect]:
    encodings = ("utf-8-sig", "utf-8", "latin-1")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                sample = handle.read(64 * 1024)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            return encoding, dialect
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not detect CSV encoding/dialect: {last_error}")


def likely_label_columns(columns: Iterable[str]) -> list[str]:
    tokens = ("label", "verdict", "rating", "class", "category")
    return [
        column
        for column in columns
        if any(token in column.lower() for token in tokens)
    ]


def audit_healthfc(root: Path) -> dict[str, Any]:
    path = root / "data/raw/v1/healthfc/Datensatz.csv"
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": file_size(path),
        "expected_rows": EXPECTED_HEALTHFC_ROWS,
        "errors": [],
    }
    try:
        encoding, dialect = detect_csv(path)
        frame = pd.read_csv(
            path,
            encoding=encoding,
            sep=dialect.delimiter,
            low_memory=False,
        )
        label_counts = {}
        for column in likely_label_columns(frame.columns):
            label_counts[column] = {
                str(key): int(value)
                for key, value in frame[column]
                .fillna("<NA>")
                .astype(str)
                .value_counts(dropna=False)
                .to_dict()
                .items()
            }

        result.update(
            {
                "encoding": encoding,
                "delimiter": dialect.delimiter,
                "actual_rows": len(frame),
                "columns": [str(column) for column in frame.columns],
                "null_counts": {
                    str(column): int(frame[column].isna().sum())
                    for column in frame.columns
                },
                "label_counts": label_counts,
                "sample": clean_json(frame.head(3).to_dict(orient="records")),
                "ok": len(frame) == EXPECTED_HEALTHFC_ROWS,
            }
        )
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["actual_rows"] = 0
        result["columns"] = []
        result["ok"] = False
    return result


def summarize(report: dict[str, Any]) -> None:
    med = report["datasets"]["medfact_synth"]
    ev = report["datasets"]["evidencebench_100k"]
    hf = report["datasets"]["healthfc"]

    print("\n=== EvidenceGap V1 raw-data audit ===")
    print(
        f"MedFact-Synth: {med['actual_shards']} shards, "
        f"{med['actual_rows']:,} rows — {'OK' if med['ok'] else 'FAIL'}"
    )
    print(
        f"EvidenceBench-100k: {ev['actual_rows']:,} rows — "
        f"{'OK' if ev['ok'] else 'FAIL'}"
    )
    print(
        f"HealthFC: {hf['actual_rows']:,} rows — "
        f"{'OK' if hf['ok'] else 'FAIL'}"
    )
    print(f"Overall: {'OK' if report['ok'] else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.root.resolve()
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "medfact_synth": audit_medfact(root),
            "evidencebench_100k": audit_evidencebench(root),
            "healthfc": audit_healthfc(root),
        },
    }
    report["ok"] = all(
        dataset["ok"] for dataset in report["datasets"].values()
    )

    output = root / "data/raw/v1/audit_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(clean_json(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summarize(report)
    print(f"Report: {output}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

