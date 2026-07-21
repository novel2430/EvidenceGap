#!/usr/bin/env python3
"""
Read-only V1 dataset inspection.

Outputs:
  data/analysis/v1/dataset_inventory.json
  data/analysis/v1/dataset_inventory.md

Dependencies:
  python -m pip install pandas pyarrow ijson
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ijson
import pandas as pd
import pyarrow.parquet as pq


def short(value: Any, limit: int = 400) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, list):
        return [short(item, limit) for item in value[:8]]
    if isinstance(value, dict):
        return {str(k): short(v, limit) for k, v in list(value.items())[:30]}
    return value


def analyze_medfact(root: Path) -> dict[str, Any]:
    paths = sorted(
        (root / "data/raw/v1/medfact_synth/data").glob("*.parquet")
    )
    if not paths:
        raise FileNotFoundError("MedFact-Synth parquet shards not found")

    rows = 0
    label_counts: Counter[str] = Counter()
    potential_counts: Counter[str] = Counter()
    claim_pmids: set[str] = set()
    source_pmids: set[str] = set()
    same_pmid_rows = 0
    null_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    columns: list[str] = []

    wanted = [
        "idx",
        "claim_pmid",
        "claim_potential",
        "claim",
        "source_pmid",
        "source",
        "synthetic_label",
        "assistant_prompt",
    ]

    for path in paths:
        parquet = pq.ParquetFile(path)
        if not columns:
            columns = parquet.schema_arrow.names
        available = [name for name in wanted if name in parquet.schema_arrow.names]

        for batch in parquet.iter_batches(batch_size=8192, columns=available):
            frame = batch.to_pandas()
            rows += len(frame)

            for column in frame.columns:
                null_counts[column] += int(frame[column].isna().sum())

            for record in frame.to_dict(orient="records"):
                label_counts[str(record.get("synthetic_label"))] += 1
                potential_counts[str(record.get("claim_potential"))] += 1

                claim_pmid = str(record.get("claim_pmid") or "").strip()
                source_pmid = str(record.get("source_pmid") or "").strip()
                if claim_pmid:
                    claim_pmids.add(claim_pmid)
                if source_pmid:
                    source_pmids.add(source_pmid)
                if claim_pmid and claim_pmid == source_pmid:
                    same_pmid_rows += 1

                if len(samples) < 3:
                    samples.append(short(record))

    return {
        "rows": rows,
        "shards": len(paths),
        "columns": columns,
        "null_counts": dict(null_counts),
        "synthetic_label_counts": dict(sorted(label_counts.items())),
        "claim_potential_counts": dict(sorted(potential_counts.items())),
        "unique_claim_pmids": len(claim_pmids),
        "unique_source_pmids": len(source_pmids),
        "same_claim_source_pmid_rows": same_pmid_rows,
        "same_claim_source_pmid_ratio": round(same_pmid_rows / rows, 6),
        "samples": samples,
    }


def json_container(path: Path) -> str:
    with path.open("rb") as handle:
        first = handle.read(4096).lstrip()[:1]
    if first == b"{":
        return "object"
    if first == b"[":
        return "array"
    raise RuntimeError(f"Unsupported JSON top-level format: {path}")


def iter_json_records(path: Path):
    container = json_container(path)
    with path.open("rb") as handle:
        if container == "object":
            yield from ijson.kvitems(handle, "")
        else:
            for index, value in enumerate(ijson.items(handle, "item")):
                yield str(index), value


def analyze_evidencebench_file(path: Path) -> dict[str, Any]:
    rows = 0
    field_presence: Counter[str] = Counter()
    field_types: dict[str, Counter[str]] = {}
    key_signatures: Counter[tuple[str, ...]] = Counter()
    samples: list[dict[str, Any]] = []

    for record_id, record in iter_json_records(path):
        rows += 1
        if not isinstance(record, dict):
            continue

        key_signatures[tuple(sorted(str(key) for key in record))] += 1
        for key, value in record.items():
            key = str(key)
            field_presence[key] += 1
            field_types.setdefault(key, Counter())[type(value).__name__] += 1

        if len(samples) < 3:
            samples.append({"_record_id": record_id, **short(record)})

    return {
        "rows": rows,
        "bytes": path.stat().st_size,
        "container": json_container(path),
        "field_presence": dict(field_presence.most_common()),
        "field_types": {
            key: dict(counter.most_common())
            for key, counter in sorted(field_types.items())
        },
        "key_signatures": [
            {"keys": list(keys), "rows": count}
            for keys, count in key_signatures.most_common(10)
        ],
        "samples": samples,
    }


def analyze_evidencebench(root: Path) -> dict[str, Any]:
    directory = root / "data/raw/v1/evidencebench_100k"
    train = analyze_evidencebench_file(
        directory / "evidencebench_100k_train_set.json"
    )
    test = analyze_evidencebench_file(
        directory / "evidencebench_100k_test_set.json"
    )

    common_fields = sorted(
        set(train["field_presence"]) & set(test["field_presence"])
    )
    return {
        "rows": train["rows"] + test["rows"],
        "common_fields": common_fields,
        "train": train,
        "test": test,
    }


def analyze_healthfc(root: Path) -> dict[str, Any]:
    path = root / "data/raw/v1/healthfc/Datensatz.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    return {
        "rows": len(frame),
        "columns": [str(column) for column in frame.columns],
        "null_counts": {
            str(column): int(frame[column].isna().sum())
            for column in frame.columns
        },
        "label_counts": {
            str(key): int(value)
            for key, value in frame["label"].value_counts().sort_index().items()
        },
        "duplicate_en_claim_rows": int(frame["en_claim"].duplicated().sum()),
        "samples": short(frame.head(3).to_dict(orient="records")),
    }


def markdown(report: dict[str, Any]) -> str:
    med = report["datasets"]["medfact_synth"]
    ev = report["datasets"]["evidencebench_100k"]
    hf = report["datasets"]["healthfc"]

    ev_fields = ", ".join(ev["common_fields"])

    return f"""# EvidenceGap V1 Dataset Inventory

Generated: {report["created_at"]}

## MedFact-Synth

- Rows: {med["rows"]:,}
- Shards: {med["shards"]}
- Unique claim PMIDs: {med["unique_claim_pmids"]:,}
- Unique source PMIDs: {med["unique_source_pmids"]:,}
- Claim PMID = source PMID: {med["same_claim_source_pmid_rows"]:,} ({med["same_claim_source_pmid_ratio"]:.2%})

Labels:

```json
{json.dumps(med["synthetic_label_counts"], ensure_ascii=False, indent=2)}
```

Preliminary role: large synthetic **claim–article stance** corpus. It is useful for scale and stress-testing, but cannot be the sole final medical-quality proof.

## EvidenceBench-100k

- Rows: {ev["rows"]:,}
- Train: {ev["train"]["rows"]:,}
- Test: {ev["test"]["rows"]:,}
- Common fields: {ev_fields}

Preliminary role: **hypothesis-to-evidence-sentence retrieval/extraction**. Review the samples and field names before fixing the exact input/output contract.

## HealthFC

- Rows: {hf["rows"]:,}
- Duplicate English claims: {hf["duplicate_en_claim_rows"]}
- Missing English explanations: {hf["null_counts"].get("en_explanation", 0)}

Labels:

```json
{json.dumps(hf["label_counts"], ensure_ascii=False, indent=2)}
```

Preliminary role: small expert-annotated **final evaluation set**, not the main training or indexing corpus.

## Decision gate

Before writing model code, fix these five items:

1. Exact meaning of every label.
2. Exact EvidenceBench hypothesis, paper-text, candidate-sentence and gold-evidence fields.
3. Which dataset evaluates article retrieval.
4. Which dataset evaluates evidence-sentence extraction.
5. Which dataset evaluates final stance/verdict.

After reviewing this report, write a one-page V1 task contract. Do not clean or transform the datasets before that contract is fixed.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.root.resolve()
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "medfact_synth": analyze_medfact(root),
            "evidencebench_100k": analyze_evidencebench(root),
            "healthfc": analyze_healthfc(root),
        },
    }

    output_dir = root / "data/analysis/v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "dataset_inventory.json"
    md_path = output_dir / "dataset_inventory.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(markdown(report), encoding="utf-8")

    print("Written:")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
