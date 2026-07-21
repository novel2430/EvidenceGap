#!/usr/bin/env python3
"""
Build and validate EvidenceGap V1 Phase 01 split manifests.

Phase 01 responsibilities:
- keep data/raw/v1 immutable;
- create deterministic MedFact-Synth train/dev/test splits grouped by Claim;
- preserve EvidenceBench official test and derive a leakage-resistant dev split
  from official train using connected systematic-review/paper components;
- keep all HealthFC rows as external evaluation only;
- write metadata-only JSONL manifests (no claim/article/evidence text copies);
- write reproducibility metadata, output hashes, statistics, and validation.

Outputs (full mode):
  data/processed/v1/manifests/
    medfact_train.jsonl
    medfact_dev.jsonl
    medfact_test.jsonl
    evidencebench_train.jsonl
    evidencebench_dev.jsonl
    evidencebench_test.jsonl
    healthfc_eval.jsonl
    phase01_manifest.json
    validation_report.json

Dependencies:
  python -m pip install pyarrow ijson

Examples:
  python scripts/build_v1_phase01_manifests.py --root .
  python scripts/build_v1_phase01_manifests.py --root . --quick
  python scripts/build_v1_phase01_manifests.py --root . --validate-only
  python scripts/build_v1_phase01_manifests.py --root . --force
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator

try:
    import ijson
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install with:\n"
        "  python -m pip install pyarrow ijson"
    ) from exc


PHASE01_SCHEMA_VERSION = "1.0.0"
EXPECTED_CONTRACT_VERSION = "1.0.0"
DEFAULT_SEED = 20260721

MEDFACT_TRAIN_RATIO = 0.90
MEDFACT_DEV_RATIO = 0.05
MEDFACT_TEST_RATIO = 0.05
EVIDENCEBENCH_DEV_RATIO = 0.05

MEDFACT_GLOB = "data/raw/v1/medfact_synth/data/*.parquet"
EVIDENCEBENCH_TRAIN = Path(
    "data/raw/v1/evidencebench_100k/evidencebench_100k_train_set.json"
)
EVIDENCEBENCH_TEST = Path(
    "data/raw/v1/evidencebench_100k/evidencebench_100k_test_set.json"
)
HEALTHFC_PATH = Path("data/raw/v1/healthfc/Datensatz.csv")
CONTRACT_DIR = Path("data/contracts/v1")

FULL_OUTPUT_DIR = Path("data/processed/v1/manifests")
QUICK_OUTPUT_DIR = Path("data/processed/v1/manifests_quick")

EXPECTED_MEDFACT_ROWS = 1_497_981
EXPECTED_EVIDENCEBENCH_TRAIN_ROWS = 87_461
EXPECTED_EVIDENCEBENCH_TEST_ROWS = 20_000
EXPECTED_HEALTHFC_ROWS = 750

QUICK_MEDFACT_LIMIT = 50_000
QUICK_EVIDENCEBENCH_TRAIN_LIMIT = 5_000
QUICK_EVIDENCEBENCH_TEST_LIMIT = 2_000

OUTPUT_FILES = (
    "medfact_train.jsonl",
    "medfact_dev.jsonl",
    "medfact_test.jsonl",
    "evidencebench_train.jsonl",
    "evidencebench_dev.jsonl",
    "evidencebench_test.jsonl",
    "healthfc_eval.jsonl",
    "phase01_manifest.json",
    "validation_report.json",
)

MEDFACT_ALLOWED_LABELS = {-2, -1, 0, 1, 2}
HEALTHFC_ALLOWED_LABELS = {0, 1, 2}

FORBIDDEN_TEXT_KEYS = {
    "claim",
    "claim_text",
    "source",
    "source_text",
    "article_text",
    "hypothesis",
    "candidate_sentences",
    "paper_as_candidate_pool",
    "evidence",
    "evidence_text",
    "en_claim",
    "en_top_sentences",
    "en_explanation",
}


class Phase01Error(RuntimeError):
    """Raised when Phase 01 contracts or data invariants are violated."""


@dataclass
class JsonlWriter:
    path: Path
    handle: BinaryIO = field(init=False)
    digest: Any = field(init=False)
    records: int = 0
    bytes_written: int = 0

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("wb")
        self.digest = hashlib.sha256()

    def write(self, record: dict[str, Any]) -> None:
        payload = (
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        self.handle.write(payload)
        self.digest.update(payload)
        self.records += 1
        self.bytes_written += len(payload)

    def close(self) -> None:
        if self.handle.closed:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()

    def summary(self) -> dict[str, Any]:
        self.close()
        return {
            "path": self.path.name,
            "records": self.records,
            "bytes": self.bytes_written,
            "sha256": self.digest.hexdigest(),
        }


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> str:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return left_root


@dataclass
class BuildState:
    mode: str
    seed: int
    output_dir: Path
    contract_hashes: dict[str, str]
    source_descriptors: dict[str, Any]
    files: dict[str, Any] = field(default_factory=dict)
    datasets: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationState:
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def passed(self, name: str, details: Any = None) -> None:
        item: dict[str, Any] = {"name": name, "status": "PASS"}
        if details is not None:
            item["details"] = details
        self.checks.append(item)

    def failed(self, name: str, message: str) -> None:
        self.errors.append(f"{name}: {message}")
        self.checks.append(
            {"name": name, "status": "FAIL", "details": message}
        )

    def warned(self, name: str, message: str) -> None:
        self.warnings.append(f"{name}: {message}")
        self.checks.append(
            {"name": name, "status": "WARN", "details": message}
        )


# ---------------------------------------------------------------------------
# Stable normalization, IDs, hashes, and splits
# ---------------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


def clean_identifier(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    if text.endswith(".0"):
        integer = text[:-2]
        if integer.isdigit():
            return integer
    return text


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def text_hash(value: Any, length: int = 16) -> str:
    return sha256_text(normalize_text(value))[:length]


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_unit_interval(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}\x1f{key}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big", signed=False)
    return integer / float(1 << 64)


def medfact_split(seed: int, group_id: str) -> str:
    value = deterministic_unit_interval(seed, group_id)
    if value < MEDFACT_TRAIN_RATIO:
        return "train"
    if value < MEDFACT_TRAIN_RATIO + MEDFACT_DEV_RATIO:
        return "dev"
    return "test"


def stable_component_id(component_key: str) -> str:
    return f"evidencebench-component:{sha256_text(component_key)[:16]}"


def relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


# ---------------------------------------------------------------------------
# Contract guards
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Phase01Error(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Phase01Error(f"Invalid JSON file {path}: {exc}") from exc


def validate_phase00_contracts(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    mappings_path = root / CONTRACT_DIR / "dataset_mappings.json"
    labels_path = root / CONTRACT_DIR / "label_maps.json"
    mappings = load_json(mappings_path)
    labels = load_json(labels_path)

    mapping_version = str(mappings.get("schema_version", ""))
    label_version = str(labels.get("schema_version", ""))
    if mapping_version != EXPECTED_CONTRACT_VERSION:
        raise Phase01Error(
            f"dataset_mappings.json schema_version must be "
            f"{EXPECTED_CONTRACT_VERSION}, got {mapping_version!r}"
        )
    if label_version != EXPECTED_CONTRACT_VERSION:
        raise Phase01Error(
            f"label_maps.json schema_version must be "
            f"{EXPECTED_CONTRACT_VERSION}, got {label_version!r}"
        )

    expected_mappings = {
        ("medfact_synth", "row_id"): "idx",
        ("medfact_synth", "claim_source_id"): "claim_pmid",
        ("medfact_synth", "claim_text"): "claim",
        ("medfact_synth", "article_source_id"): "source_pmid",
        ("medfact_synth", "article_text"): "source",
        ("medfact_synth", "stance_label"): "synthetic_label",
        ("evidencebench_100k", "query_text"): "hypothesis",
        ("evidencebench_100k", "candidate_sentences"): "paper_as_candidate_pool",
        ("evidencebench_100k", "paper_id"): "paper_id",
        ("evidencebench_100k", "group_id"): "systematic_review_id",
        ("healthfc", "claim_text"): "en_claim",
        ("healthfc", "evidence_text"): "en_top_sentences",
        ("healthfc", "label"): "label",
    }
    datasets = mappings.get("datasets", {})
    for (dataset, field_name), expected_raw in expected_mappings.items():
        actual = (
            datasets.get(dataset, {})
            .get("fields", {})
            .get(field_name)
        )
        if actual != expected_raw:
            raise Phase01Error(
                f"Contract mapping {dataset}.{field_name} must be "
                f"{expected_raw!r}, got {actual!r}"
            )

    medfact_labels = labels.get("medfact_stance_5", {}).get("labels", {})
    if {int(key) for key in medfact_labels} != MEDFACT_ALLOWED_LABELS:
        raise Phase01Error("Phase 00 MedFact label domain is not {-2,-1,0,1,2}")
    healthfc_labels = labels.get("healthfc_verdict_3", {}).get("labels", {})
    if {int(key) for key in healthfc_labels} != HEALTHFC_ALLOWED_LABELS:
        raise Phase01Error("Phase 00 HealthFC label domain is not {0,1,2}")

    hashes = {
        "dataset_mappings.json": file_sha256(mappings_path),
        "label_maps.json": file_sha256(labels_path),
    }
    return mappings, labels, hashes


# ---------------------------------------------------------------------------
# Raw dataset iterators
# ---------------------------------------------------------------------------


def json_container(path: Path) -> str:
    with path.open("rb") as handle:
        first = handle.read(4096).lstrip()[:1]
    if first == b"{":
        return "object"
    if first == b"[":
        return "array"
    raise Phase01Error(f"Unsupported JSON top-level container: {path}")


def iter_json_records(
    path: Path,
    limit: int | None = None,
) -> Iterator[tuple[int, str, Any]]:
    container = json_container(path)
    with path.open("rb") as handle:
        if container == "object":
            iterator = ijson.kvitems(handle, "")
        else:
            iterator = ((str(index), item) for index, item in enumerate(ijson.items(handle, "item")))
        for ordinal, (record_id, record) in enumerate(iterator):
            if limit is not None and ordinal >= limit:
                break
            yield ordinal, str(record_id), record


def medfact_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob(MEDFACT_GLOB))
    if not paths:
        raise Phase01Error(f"No MedFact parquet shards found for {MEDFACT_GLOB}")
    return paths


def source_descriptors(root: Path) -> dict[str, Any]:
    medfact = medfact_paths(root)
    evidence_train = root / EVIDENCEBENCH_TRAIN
    evidence_test = root / EVIDENCEBENCH_TEST
    healthfc = root / HEALTHFC_PATH
    for path in [evidence_train, evidence_test, healthfc]:
        if not path.exists():
            raise Phase01Error(f"Missing raw dataset file: {path}")

    return {
        "medfact_synth": {
            "files": [
                {
                    "path": relative_path(root, path),
                    "bytes": path.stat().st_size,
                    "parquet_rows": pq.read_metadata(path).num_rows,
                }
                for path in medfact
            ]
        },
        "evidencebench_100k": {
            "train": {
                "path": relative_path(root, evidence_train),
                "bytes": evidence_train.stat().st_size,
            },
            "test": {
                "path": relative_path(root, evidence_test),
                "bytes": evidence_test.stat().st_size,
            },
        },
        "healthfc": {
            "path": relative_path(root, healthfc),
            "bytes": healthfc.stat().st_size,
        },
    }


# ---------------------------------------------------------------------------
# MedFact-Synth
# ---------------------------------------------------------------------------


def parse_stance_label(value: Any) -> int:
    if value is None:
        raise Phase01Error("MedFact synthetic_label is null")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Phase01Error(f"Invalid MedFact synthetic_label: {value!r}") from exc
    if not number.is_integer():
        raise Phase01Error(f"MedFact synthetic_label is non-integral: {value!r}")
    label = int(number)
    if label not in MEDFACT_ALLOWED_LABELS:
        raise Phase01Error(f"Unexpected MedFact label: {label}")
    return label


def build_medfact_manifests(
    root: Path,
    writers: dict[str, JsonlWriter],
    seed: int,
    limit: int | None,
) -> dict[str, Any]:
    required = [
        "idx",
        "claim_pmid",
        "claim_potential",
        "claim",
        "source_pmid",
        "source",
        "synthetic_label",
    ]

    row_counts: Counter[str] = Counter()
    group_sets: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    label_counts: dict[str, Counter[int]] = {
        "train": Counter(),
        "dev": Counter(),
        "test": Counter(),
    }
    origin_counts: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    total = 0

    for shard_path in medfact_paths(root):
        parquet = pq.ParquetFile(shard_path)
        missing = sorted(set(required) - set(parquet.schema_arrow.names))
        if missing:
            raise Phase01Error(f"{shard_path.name} misses fields: {missing}")

        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=required)
            values = {name: table[name].to_pylist() for name in required}
            for row_in_group in range(table.num_rows):
                if limit is not None and total >= limit:
                    break

                raw_idx = values["idx"][row_in_group]
                claim_pmid = clean_identifier(values["claim_pmid"][row_in_group])
                source_pmid = clean_identifier(values["source_pmid"][row_in_group])
                claim = normalize_text(values["claim"][row_in_group])
                source = normalize_text(values["source"][row_in_group])
                if not claim:
                    raise Phase01Error(
                        f"Empty MedFact claim at {shard_path.name} "
                        f"row_group={row_group} row={row_in_group}"
                    )
                if not source:
                    raise Phase01Error(
                        f"Empty MedFact source at {shard_path.name} "
                        f"row_group={row_group} row={row_in_group}"
                    )

                claim_text_hash = text_hash(claim)
                article_text_hash = text_hash(source)
                claim_id_fallback = not bool(claim_pmid)
                article_id_fallback = not bool(source_pmid)
                if claim_id_fallback:
                    fallback_counts["claim_id"] += 1
                if article_id_fallback:
                    fallback_counts["article_id"] += 1

                claim_source_key = claim_pmid or f"missing-{claim_text_hash}"
                claim_id = f"medfact:{claim_source_key}:{claim_text_hash}"
                group_id = f"medfact-group:{claim_source_key}:{claim_text_hash}"
                article_id = (
                    f"pmid:{source_pmid}"
                    if source_pmid
                    else f"medfact-article:{article_text_hash}"
                )
                judgment_id = (
                    f"medfact-pair:{clean_identifier(raw_idx)}"
                    if clean_identifier(raw_idx)
                    else (
                        f"medfact-pair:{shard_path.stem}:"
                        f"{row_group}:{row_in_group}"
                    )
                )
                split = medfact_split(seed, group_id)
                stance_label = parse_stance_label(
                    values["synthetic_label"][row_in_group]
                )
                is_origin = bool(
                    claim_pmid
                    and source_pmid
                    and claim_pmid == source_pmid
                )

                record = {
                    "manifest_schema_version": PHASE01_SCHEMA_VERSION,
                    "contract_version": EXPECTED_CONTRACT_VERSION,
                    "dataset": "medfact_synth",
                    "task_ids": ["AR-MEDFACT-JUDGED", "STANCE-MEDFACT-5"],
                    "split": split,
                    "judgment_id": judgment_id,
                    "claim_id": claim_id,
                    "split_group_id": group_id,
                    "article_id": article_id,
                    "claim_pmid": claim_pmid or None,
                    "source_pmid": source_pmid or None,
                    "claim_text_hash": claim_text_hash,
                    "article_text_hash": article_text_hash,
                    "stance_label": stance_label,
                    "relevance_grade": abs(stance_label),
                    "is_origin_source": is_origin,
                    "claim_potential": clean_identifier(
                        values["claim_potential"][row_in_group]
                    )
                    or None,
                    "id_fallback": {
                        "claim_id": claim_id_fallback,
                        "article_id": article_id_fallback,
                    },
                    "raw_locator": {
                        "path": relative_path(root, shard_path),
                        "record_id": clean_identifier(raw_idx) or None,
                        "row_group": row_group,
                        "row_in_row_group": row_in_group,
                    },
                }
                writers[f"medfact_{split}.jsonl"].write(record)

                total += 1
                row_counts[split] += 1
                group_sets[split].add(group_id)
                label_counts[split][stance_label] += 1
                if is_origin:
                    origin_counts[split] += 1

            if limit is not None and total >= limit:
                break
        if limit is not None and total >= limit:
            break
        print(
            f"  MedFact {shard_path.name}: cumulative {total:,} rows",
            flush=True,
        )

    overlap = {
        "train_dev": len(group_sets["train"] & group_sets["dev"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "dev_test": len(group_sets["dev"] & group_sets["test"]),
    }
    if any(overlap.values()):
        raise Phase01Error(f"MedFact Claim groups leak across splits: {overlap}")

    if limit is None and total != EXPECTED_MEDFACT_ROWS:
        raise Phase01Error(
            f"MedFact row count mismatch: expected {EXPECTED_MEDFACT_ROWS:,}, "
            f"got {total:,}"
        )

    return {
        "rows": total,
        "split_method": {
            "unit": "split_group_id = hash(claim_pmid, normalized_claim)",
            "assignment": "sha256(seed, split_group_id) unit interval",
            "seed": seed,
            "target_ratios": {
                "train": MEDFACT_TRAIN_RATIO,
                "dev": MEDFACT_DEV_RATIO,
                "test": MEDFACT_TEST_RATIO,
            },
        },
        "rows_by_split": dict(row_counts),
        "groups_by_split": {
            split: len(groups) for split, groups in group_sets.items()
        },
        "actual_row_ratios": {
            split: round(row_counts[split] / total, 8)
            for split in ("train", "dev", "test")
        },
        "label_counts_by_split": {
            split: {str(label): count for label, count in sorted(counter.items())}
            for split, counter in label_counts.items()
        },
        "origin_source_rows_by_split": dict(origin_counts),
        "id_fallback_counts": dict(fallback_counts),
        "group_overlap": overlap,
    }


# ---------------------------------------------------------------------------
# EvidenceBench-100k
# ---------------------------------------------------------------------------


def evidence_nodes(record_id: str, record: dict[str, Any]) -> tuple[str, str]:
    review_id = clean_identifier(record.get("systematic_review_id"))
    paper_id = clean_identifier(record.get("paper_id"))
    review_node = (
        f"review:{review_id}" if review_id else f"record-review:{record_id}"
    )
    paper_node = (
        f"paper:{paper_id}" if paper_id else f"record-paper:{record_id}"
    )
    return review_node, paper_node


def prepare_evidencebench_dev_components(
    path: Path,
    seed: int,
    limit: int | None,
) -> tuple[set[str], dict[str, str], dict[str, Any], set[str], set[str]]:
    union_find = UnionFind()
    records: list[tuple[str, str, str]] = []
    review_ids: set[str] = set()
    paper_ids: set[str] = set()

    for ordinal, record_id, record in iter_json_records(path, limit=limit):
        if not isinstance(record, dict):
            raise Phase01Error(f"EvidenceBench {record_id} is not an object")
        review_node, paper_node = evidence_nodes(record_id, record)
        union_find.union(review_node, paper_node)
        records.append((record_id, review_node, paper_node))
        review = clean_identifier(record.get("systematic_review_id"))
        paper = clean_identifier(record.get("paper_id"))
        if review:
            review_ids.add(review)
        if paper:
            paper_ids.add(paper)
        if (ordinal + 1) % 10_000 == 0:
            print(
                f"  EvidenceBench component scan: {ordinal + 1:,} records",
                flush=True,
            )

    members_by_root: defaultdict[str, list[str]] = defaultdict(list)
    for node in union_find.parent:
        members_by_root[union_find.find(node)].append(node)
    canonical_key_by_root = {
        root: min(members) for root, members in members_by_root.items()
    }

    component_counts: Counter[str] = Counter()
    record_component: dict[str, str] = {}
    for record_id, review_node, _paper_node in records:
        root = union_find.find(review_node)
        component_key = canonical_key_by_root[root]
        component_id = stable_component_id(component_key)
        component_counts[component_id] += 1
        record_component[record_id] = component_id

    target_dev_rows = round(len(records) * EVIDENCEBENCH_DEV_RATIO)
    ordered_components = sorted(
        component_counts,
        key=lambda component_id: (
            deterministic_unit_interval(seed, component_id),
            component_id,
        ),
    )

    cumulative = 0
    best_k = 0
    best_distance = abs(target_dev_rows)
    for index, component_id in enumerate(ordered_components, start=1):
        cumulative += component_counts[component_id]
        distance = abs(cumulative - target_dev_rows)
        if distance < best_distance:
            best_distance = distance
            best_k = index
    dev_components = set(ordered_components[:best_k])
    actual_dev_rows = sum(component_counts[c] for c in dev_components)

    stats = {
        "official_train_rows": len(records),
        "connected_components": len(component_counts),
        "target_dev_rows": target_dev_rows,
        "actual_dev_rows": actual_dev_rows,
        "actual_train_rows": len(records) - actual_dev_rows,
        "target_dev_ratio": EVIDENCEBENCH_DEV_RATIO,
        "actual_dev_ratio": round(actual_dev_rows / max(1, len(records)), 8),
        "largest_component_rows": max(component_counts.values(), default=0),
        "split_unit": "connected component of systematic_review_id and paper_id",
        "selection": "deterministic hash order with closest whole-component cutoff",
        "seed": seed,
    }
    return dev_components, record_component, stats, review_ids, paper_ids


def build_evidencebench_manifests(
    root: Path,
    writers: dict[str, JsonlWriter],
    seed: int,
    train_limit: int | None,
    test_limit: int | None,
) -> dict[str, Any]:
    train_path = root / EVIDENCEBENCH_TRAIN
    test_path = root / EVIDENCEBENCH_TEST

    (
        dev_components,
        record_component,
        split_stats,
        official_train_reviews,
        official_train_papers,
    ) = prepare_evidencebench_dev_components(train_path, seed, train_limit)

    rows_by_split: Counter[str] = Counter()
    groups_by_split: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    reviews_by_split: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    papers_by_split: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    missing_counts: Counter[str] = Counter()

    for ordinal, record_id, record in iter_json_records(train_path, limit=train_limit):
        if not isinstance(record, dict):
            raise Phase01Error(f"EvidenceBench {record_id} is not an object")
        component_id = record_component[record_id]
        split = "dev" if component_id in dev_components else "train"
        review_id = clean_identifier(record.get("systematic_review_id"))
        paper_id = clean_identifier(record.get("paper_id"))
        if not review_id:
            missing_counts["systematic_review_id"] += 1
        if not paper_id:
            missing_counts["paper_id"] += 1

        candidate_pool = record.get("paper_as_candidate_pool")
        aspect_ids = record.get("aspect_list_ids")
        results_aspects = record.get("results_aspect_list_ids")
        if not isinstance(candidate_pool, list):
            raise Phase01Error(
                f"EvidenceBench {record_id} candidate pool is not a list"
            )
        if not isinstance(aspect_ids, list):
            raise Phase01Error(
                f"EvidenceBench {record_id} aspect_list_ids is not a list"
            )
        if results_aspects is not None and not isinstance(results_aspects, list):
            raise Phase01Error(
                f"EvidenceBench {record_id} results_aspect_list_ids "
                f"is neither null nor list"
            )

        manifest_record = {
            "manifest_schema_version": PHASE01_SCHEMA_VERSION,
            "contract_version": EXPECTED_CONTRACT_VERSION,
            "dataset": "evidencebench_100k",
            "task_ids": ["ESR-EVIDENCEBENCH"],
            "split": split,
            "official_split": "train",
            "query_id": f"evidencebench:{record_id}",
            "split_group_id": component_id,
            "systematic_review_id": review_id or None,
            "paper_id": paper_id or None,
            "candidate_sentence_count": len(candidate_pool),
            "aspect_count": len(aspect_ids),
            "results_aspect_count": len(results_aspects or []),
            "raw_locator": {
                "path": relative_path(root, train_path),
                "record_id": record_id,
                "ordinal": ordinal,
            },
        }
        writers[f"evidencebench_{split}.jsonl"].write(manifest_record)
        rows_by_split[split] += 1
        groups_by_split[split].add(component_id)
        if review_id:
            reviews_by_split[split].add(review_id)
        if paper_id:
            papers_by_split[split].add(paper_id)

    official_test_review_overlap: set[str] = set()
    official_test_paper_overlap: set[str] = set()
    for ordinal, record_id, record in iter_json_records(test_path, limit=test_limit):
        if not isinstance(record, dict):
            raise Phase01Error(f"EvidenceBench {record_id} is not an object")
        review_id = clean_identifier(record.get("systematic_review_id"))
        paper_id = clean_identifier(record.get("paper_id"))
        if not review_id:
            missing_counts["systematic_review_id"] += 1
        if not paper_id:
            missing_counts["paper_id"] += 1
        if review_id in official_train_reviews:
            official_test_review_overlap.add(review_id)
        if paper_id in official_train_papers:
            official_test_paper_overlap.add(paper_id)

        candidate_pool = record.get("paper_as_candidate_pool")
        aspect_ids = record.get("aspect_list_ids")
        results_aspects = record.get("results_aspect_list_ids")
        if not isinstance(candidate_pool, list) or not isinstance(aspect_ids, list):
            raise Phase01Error(
                f"EvidenceBench test {record_id} has invalid candidate/aspect fields"
            )
        if results_aspects is not None and not isinstance(results_aspects, list):
            raise Phase01Error(
                f"EvidenceBench test {record_id} results_aspect_list_ids invalid"
            )

        test_group_key = (
            f"review:{review_id}"
            if review_id
            else (f"paper:{paper_id}" if paper_id else f"record:{record_id}")
        )
        component_id = stable_component_id(test_group_key)
        manifest_record = {
            "manifest_schema_version": PHASE01_SCHEMA_VERSION,
            "contract_version": EXPECTED_CONTRACT_VERSION,
            "dataset": "evidencebench_100k",
            "task_ids": ["ESR-EVIDENCEBENCH"],
            "split": "test",
            "official_split": "test",
            "query_id": f"evidencebench:{record_id}",
            "split_group_id": component_id,
            "systematic_review_id": review_id or None,
            "paper_id": paper_id or None,
            "candidate_sentence_count": len(candidate_pool),
            "aspect_count": len(aspect_ids),
            "results_aspect_count": len(results_aspects or []),
            "raw_locator": {
                "path": relative_path(root, test_path),
                "record_id": record_id,
                "ordinal": ordinal,
            },
        }
        writers["evidencebench_test.jsonl"].write(manifest_record)
        rows_by_split["test"] += 1
        groups_by_split["test"].add(component_id)
        if review_id:
            reviews_by_split["test"].add(review_id)
        if paper_id:
            papers_by_split["test"].add(paper_id)
        if (ordinal + 1) % 10_000 == 0:
            print(
                f"  EvidenceBench test write: {ordinal + 1:,} records",
                flush=True,
            )

    train_dev_group_overlap = groups_by_split["train"] & groups_by_split["dev"]
    train_dev_review_overlap = reviews_by_split["train"] & reviews_by_split["dev"]
    train_dev_paper_overlap = papers_by_split["train"] & papers_by_split["dev"]
    if train_dev_group_overlap or train_dev_review_overlap or train_dev_paper_overlap:
        raise Phase01Error(
            "EvidenceBench derived train/dev leakage: "
            f"groups={len(train_dev_group_overlap)}, "
            f"reviews={len(train_dev_review_overlap)}, "
            f"papers={len(train_dev_paper_overlap)}"
        )

    if train_limit is None:
        official_train_total = rows_by_split["train"] + rows_by_split["dev"]
        if official_train_total != EXPECTED_EVIDENCEBENCH_TRAIN_ROWS:
            raise Phase01Error(
                "EvidenceBench official train count mismatch: "
                f"expected {EXPECTED_EVIDENCEBENCH_TRAIN_ROWS:,}, "
                f"got {official_train_total:,}"
            )
    if test_limit is None and rows_by_split["test"] != EXPECTED_EVIDENCEBENCH_TEST_ROWS:
        raise Phase01Error(
            "EvidenceBench official test count mismatch: "
            f"expected {EXPECTED_EVIDENCEBENCH_TEST_ROWS:,}, "
            f"got {rows_by_split['test']:,}"
        )

    return {
        "rows_by_split": dict(rows_by_split),
        "groups_by_split": {
            split: len(groups) for split, groups in groups_by_split.items()
        },
        "reviews_by_split": {
            split: len(groups) for split, groups in reviews_by_split.items()
        },
        "papers_by_split": {
            split: len(groups) for split, groups in papers_by_split.items()
        },
        "derived_train_dev": split_stats,
        "train_dev_overlap": {
            "groups": len(train_dev_group_overlap),
            "systematic_reviews": len(train_dev_review_overlap),
            "papers": len(train_dev_paper_overlap),
        },
        "official_train_test_overlap": {
            "systematic_reviews": len(official_test_review_overlap),
            "papers": len(official_test_paper_overlap),
            "note": (
                "Official test is preserved even if source dataset contains "
                "cross-split overlap; overlaps are reported, never repaired."
            ),
        },
        "missing_id_counts": dict(missing_counts),
    }


# ---------------------------------------------------------------------------
# HealthFC
# ---------------------------------------------------------------------------


def build_healthfc_manifest(
    root: Path,
    writer: JsonlWriter,
    label_names: dict[str, str],
) -> dict[str, Any]:
    path = root / HEALTHFC_PATH
    label_counts: Counter[int] = Counter()
    case_ids: set[str] = set()
    rows = 0
    duplicate_case_ids = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"en_claim", "en_top_sentences", "label", "url"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise Phase01Error(f"HealthFC misses fields: {missing}")

        for row_index, row in enumerate(reader):
            claim = normalize_text(row.get("en_claim"))
            evidence = normalize_text(row.get("en_top_sentences"))
            if not claim:
                raise Phase01Error(f"HealthFC row {row_index} has empty en_claim")
            if not evidence:
                raise Phase01Error(
                    f"HealthFC row {row_index} has empty en_top_sentences"
                )
            try:
                label = int(clean_identifier(row.get("label")))
            except ValueError as exc:
                raise Phase01Error(
                    f"HealthFC row {row_index} has invalid label {row.get('label')!r}"
                ) from exc
            if label not in HEALTHFC_ALLOWED_LABELS:
                raise Phase01Error(
                    f"HealthFC row {row_index} has unexpected label {label}"
                )

            claim_hash = text_hash(claim)
            case_id = f"healthfc:{claim_hash}"
            if case_id in case_ids:
                duplicate_case_ids += 1
            case_ids.add(case_id)

            record = {
                "manifest_schema_version": PHASE01_SCHEMA_VERSION,
                "contract_version": EXPECTED_CONTRACT_VERSION,
                "dataset": "healthfc",
                "task_ids": ["VERDICT-HEALTHFC-3"],
                "split": "eval",
                "usage": "external_expert_evaluation_only",
                "case_id": case_id,
                "claim_text_hash": claim_hash,
                "evidence_text_hash": text_hash(evidence),
                "label": label,
                "label_name": label_names[str(label)],
                "source_url": clean_identifier(row.get("url")) or None,
                "raw_locator": {
                    "path": relative_path(root, path),
                    "row_index": row_index,
                },
            }
            writer.write(record)
            rows += 1
            label_counts[label] += 1

    if rows != EXPECTED_HEALTHFC_ROWS:
        raise Phase01Error(
            f"HealthFC row count mismatch: expected {EXPECTED_HEALTHFC_ROWS}, "
            f"got {rows}"
        )
    if duplicate_case_ids:
        raise Phase01Error(
            f"HealthFC has {duplicate_case_ids} duplicate canonical case IDs"
        )

    return {
        "rows": rows,
        "split": "eval",
        "usage": "external expert evaluation only",
        "label_counts": {
            str(label): count for label, count in sorted(label_counts.items())
        },
        "duplicate_case_ids": duplicate_case_ids,
    }


# ---------------------------------------------------------------------------
# Existing manifest validation
# ---------------------------------------------------------------------------


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Phase01Error(
                    f"Invalid JSONL {path.name}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise Phase01Error(
                    f"JSONL record must be object: {path.name}:{line_number}"
                )
            yield line_number, record


def scan_manifest_file(
    path: Path,
    expected_dataset: str,
    expected_split: str,
) -> tuple[dict[str, Any], set[str], set[str], set[str]]:
    records = 0
    groups: set[str] = set()
    reviews: set[str] = set()
    papers: set[str] = set()
    label_counts: Counter[int] = Counter()

    for line_number, record in iter_jsonl(path):
        records += 1
        if record.get("dataset") != expected_dataset:
            raise Phase01Error(
                f"{path.name}:{line_number} dataset mismatch"
            )
        if record.get("split") != expected_split:
            raise Phase01Error(
                f"{path.name}:{line_number} split mismatch"
            )
        leaked = FORBIDDEN_TEXT_KEYS & set(record)
        if leaked:
            raise Phase01Error(
                f"{path.name}:{line_number} copies forbidden text fields: "
                f"{sorted(leaked)}"
            )

        group_id = record.get("split_group_id")
        if group_id:
            groups.add(str(group_id))
        review = record.get("systematic_review_id")
        paper = record.get("paper_id")
        if review:
            reviews.add(str(review))
        if paper:
            papers.add(str(paper))

        if expected_dataset == "medfact_synth":
            label = record.get("stance_label")
            if label not in MEDFACT_ALLOWED_LABELS:
                raise Phase01Error(
                    f"{path.name}:{line_number} invalid stance_label {label!r}"
                )
            if record.get("relevance_grade") != abs(label):
                raise Phase01Error(
                    f"{path.name}:{line_number} relevance_grade mismatch"
                )
            label_counts[label] += 1
        elif expected_dataset == "healthfc":
            label = record.get("label")
            if label not in HEALTHFC_ALLOWED_LABELS:
                raise Phase01Error(
                    f"{path.name}:{line_number} invalid HealthFC label {label!r}"
                )
            label_counts[label] += 1

    return (
        {
            "records": records,
            "groups": len(groups),
            "labels": {str(k): v for k, v in sorted(label_counts.items())},
        },
        groups,
        reviews,
        papers,
    )


def validate_output_dir(
    output_dir: Path,
    expected_contract_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    state = ValidationState()
    manifest_path = output_dir / "phase01_manifest.json"
    try:
        manifest = load_json(manifest_path)
    except Phase01Error as exc:
        state.failed("Phase 01 manifest", str(exc))
        return validation_payload(output_dir, state, {})

    if manifest.get("schema_version") != PHASE01_SCHEMA_VERSION:
        state.failed(
            "Phase 01 schema version",
            f"expected {PHASE01_SCHEMA_VERSION}, got {manifest.get('schema_version')!r}",
        )
    else:
        state.passed("Phase 01 schema version")

    if expected_contract_hashes is not None:
        actual_hashes = manifest.get("contract_hashes", {})
        if actual_hashes != expected_contract_hashes:
            state.failed(
                "Phase 00 contract fingerprints",
                "Phase 01 outputs were not built from current Phase 00 contracts",
            )
        else:
            state.passed("Phase 00 contract fingerprints")

    expected_file_meta = manifest.get("files", {})
    for name in OUTPUT_FILES:
        if name == "validation_report.json":
            continue
        path = output_dir / name
        if not path.exists():
            state.failed("Output files", f"missing {name}")
            continue
        if name in expected_file_meta:
            expected = expected_file_meta[name]
            actual_hash = file_sha256(path)
            if actual_hash != expected.get("sha256"):
                state.failed("Output hashes", f"SHA-256 mismatch for {name}")
            elif path.stat().st_size != expected.get("bytes"):
                state.failed("Output sizes", f"size mismatch for {name}")
    if not any(check["name"] == "Output files" and check["status"] == "FAIL" for check in state.checks):
        state.passed("Output files")
    if not any(check["name"] in {"Output hashes", "Output sizes"} and check["status"] == "FAIL" for check in state.checks):
        state.passed("Output fingerprints")

    scans: dict[str, Any] = {}
    sets: dict[str, tuple[set[str], set[str], set[str]]] = {}
    specs = {
        "medfact_train.jsonl": ("medfact_synth", "train"),
        "medfact_dev.jsonl": ("medfact_synth", "dev"),
        "medfact_test.jsonl": ("medfact_synth", "test"),
        "evidencebench_train.jsonl": ("evidencebench_100k", "train"),
        "evidencebench_dev.jsonl": ("evidencebench_100k", "dev"),
        "evidencebench_test.jsonl": ("evidencebench_100k", "test"),
        "healthfc_eval.jsonl": ("healthfc", "eval"),
    }
    try:
        for name, (dataset, split) in specs.items():
            scan, groups, reviews, papers = scan_manifest_file(
                output_dir / name, dataset, split
            )
            scans[name] = scan
            sets[name] = (groups, reviews, papers)
        state.passed("JSONL schema and label domains")
        state.passed("Metadata-only manifests")
    except Phase01Error as exc:
        state.failed("JSONL manifests", str(exc))

    if sets:
        med_train = sets["medfact_train.jsonl"][0]
        med_dev = sets["medfact_dev.jsonl"][0]
        med_test = sets["medfact_test.jsonl"][0]
        med_overlap = {
            "train_dev": len(med_train & med_dev),
            "train_test": len(med_train & med_test),
            "dev_test": len(med_dev & med_test),
        }
        if any(med_overlap.values()):
            state.failed("MedFact split leakage", str(med_overlap))
        else:
            state.passed("MedFact split leakage", med_overlap)

        ev_train_groups, ev_train_reviews, ev_train_papers = sets[
            "evidencebench_train.jsonl"
        ]
        ev_dev_groups, ev_dev_reviews, ev_dev_papers = sets[
            "evidencebench_dev.jsonl"
        ]
        ev_overlap = {
            "groups": len(ev_train_groups & ev_dev_groups),
            "systematic_reviews": len(ev_train_reviews & ev_dev_reviews),
            "papers": len(ev_train_papers & ev_dev_papers),
        }
        if any(ev_overlap.values()):
            state.failed("EvidenceBench train/dev leakage", str(ev_overlap))
        else:
            state.passed("EvidenceBench train/dev leakage", ev_overlap)

        health_scan = scans.get("healthfc_eval.jsonl", {})
        if health_scan.get("records") != EXPECTED_HEALTHFC_ROWS:
            state.failed(
                "HealthFC external evaluation",
                f"expected {EXPECTED_HEALTHFC_ROWS} eval rows, "
                f"got {health_scan.get('records')}",
            )
        else:
            state.passed("HealthFC external evaluation")

    expected_counts = {
        name: meta.get("records") for name, meta in expected_file_meta.items()
        if name.endswith(".jsonl")
    }
    actual_counts = {
        name: scan.get("records") for name, scan in scans.items()
    }
    if expected_counts and expected_counts != actual_counts:
        state.failed(
            "Manifest record counts",
            f"expected {expected_counts}, got {actual_counts}",
        )
    elif actual_counts:
        state.passed("Manifest record counts")

    return validation_payload(output_dir, state, scans)


def validation_payload(
    output_dir: Path,
    state: ValidationState,
    scans: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PHASE01_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": output_dir.as_posix(),
        "status": "PASS" if not state.errors else "FAIL",
        "checks": state.checks,
        "errors": state.errors,
        "warnings": state.warnings,
        "manifest_scans": scans,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def print_validation(report: dict[str, Any]) -> None:
    print("\n=== EvidenceGap V1 Phase 01 validation ===")
    for check in report.get("checks", []):
        print(f"{check['name']}: {check['status']}")
    print(f"Errors: {len(report.get('errors', []))}")
    print(f"Warnings: {len(report.get('warnings', []))}")
    print(f"Status: {report.get('status')}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def managed_paths(output_dir: Path) -> list[Path]:
    return [output_dir / name for name in OUTPUT_FILES]


def guard_output_dir(output_dir: Path, force: bool) -> None:
    existing = [path for path in managed_paths(output_dir) if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing[:5])
        raise Phase01Error(
            f"Phase 01 outputs already exist in {output_dir}: {names}. "
            "Use --validate-only to check them or --force to rebuild."
        )


def create_writers(temp_dir: Path) -> dict[str, JsonlWriter]:
    names = [name for name in OUTPUT_FILES if name.endswith(".jsonl")]
    return {name: JsonlWriter(temp_dir / name) for name in names}


def finalize_writers(writers: dict[str, JsonlWriter]) -> dict[str, Any]:
    return {name: writer.summary() for name, writer in writers.items()}


def close_writers_safely(writers: dict[str, JsonlWriter]) -> None:
    for writer in writers.values():
        try:
            writer.close()
        except Exception:
            pass


def install_outputs(temp_dir: Path, output_dir: Path, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        source = temp_dir / name
        if not source.exists():
            raise Phase01Error(f"Internal error: temporary output missing {name}")
        target = output_dir / name
        if target.exists() and force:
            target.unlink()
        os.replace(source, target)
    temp_dir.rmdir()


def build(
    root: Path,
    output_dir: Path,
    seed: int,
    quick: bool,
    force: bool,
) -> dict[str, Any]:
    _mappings, labels, contract_hashes = validate_phase00_contracts(root)
    descriptors = source_descriptors(root)
    guard_output_dir(output_dir, force)

    temp_dir = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    writers = create_writers(temp_dir)

    state = BuildState(
        mode="quick" if quick else "full",
        seed=seed,
        output_dir=output_dir,
        contract_hashes=contract_hashes,
        source_descriptors=descriptors,
    )

    medfact_limit = QUICK_MEDFACT_LIMIT if quick else None
    evidence_train_limit = QUICK_EVIDENCEBENCH_TRAIN_LIMIT if quick else None
    evidence_test_limit = QUICK_EVIDENCEBENCH_TEST_LIMIT if quick else None

    try:
        print("Building MedFact manifests...", flush=True)
        state.datasets["medfact_synth"] = build_medfact_manifests(
            root,
            writers,
            seed,
            medfact_limit,
        )

        print("Building EvidenceBench manifests...", flush=True)
        state.datasets["evidencebench_100k"] = build_evidencebench_manifests(
            root,
            writers,
            seed,
            evidence_train_limit,
            evidence_test_limit,
        )

        print("Building HealthFC evaluation manifest...", flush=True)
        healthfc_label_names = labels["healthfc_verdict_3"]["labels"]
        state.datasets["healthfc"] = build_healthfc_manifest(
            root,
            writers["healthfc_eval.jsonl"],
            healthfc_label_names,
        )

        state.files = finalize_writers(writers)

        phase_manifest = {
            "schema_version": PHASE01_SCHEMA_VERSION,
            "phase": "EvidenceGap V1 Phase 01",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": state.mode,
            "seed": seed,
            "contract_version": EXPECTED_CONTRACT_VERSION,
            "contract_hashes": contract_hashes,
            "raw_sources": descriptors,
            "policy": {
                "raw_data_immutable": True,
                "manifests_copy_raw_text": False,
                "medfact_grouping": "claim_pmid + normalized claim",
                "medfact_ratios": {
                    "train": MEDFACT_TRAIN_RATIO,
                    "dev": MEDFACT_DEV_RATIO,
                    "test": MEDFACT_TEST_RATIO,
                },
                "evidencebench_official_test_preserved": True,
                "evidencebench_dev_grouping": (
                    "connected components of systematic_review_id and paper_id"
                ),
                "healthfc_usage": "external expert evaluation only",
            },
            "datasets": state.datasets,
            "files": state.files,
            "warnings": state.warnings,
        }
        write_json(temp_dir / "phase01_manifest.json", phase_manifest)
        state.files["phase01_manifest.json"] = {
            "path": "phase01_manifest.json",
            "bytes": (temp_dir / "phase01_manifest.json").stat().st_size,
            "sha256": file_sha256(temp_dir / "phase01_manifest.json"),
        }

        # Re-write once so the manifest includes its own descriptor except its hash
        # is intentionally not embedded recursively. The authoritative file list for
        # validation excludes self-hash requirements.
        phase_manifest["files"] = {
            name: meta
            for name, meta in state.files.items()
            if name != "phase01_manifest.json"
        }
        write_json(temp_dir / "phase01_manifest.json", phase_manifest)

        validation = validate_output_dir(
            temp_dir,
            expected_contract_hashes=contract_hashes,
        )
        write_json(temp_dir / "validation_report.json", validation)
        print_validation(validation)
        if validation["status"] != "PASS":
            raise Phase01Error(
                "Generated manifests failed validation; temporary files retained at "
                f"{temp_dir}"
            )

        install_outputs(temp_dir, output_dir, force)
        print(f"\nPhase 01 outputs written to: {output_dir}")
        return phase_manifest
    except Exception:
        close_writers_safely(writers)
        raise


def validate_only(root: Path, output_dir: Path) -> dict[str, Any]:
    _mappings, _labels, contract_hashes = validate_phase00_contracts(root)
    report = validate_output_dir(
        output_dir,
        expected_contract_hashes=contract_hashes,
    )
    write_json(output_dir / "validation_report.json", report)
    print_validation(report)
    print(f"Report: {output_dir / 'validation_report.json'}")
    if report["status"] != "PASS":
        raise SystemExit(1)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Build a non-official smoke-test subset under "
            "data/processed/v1/manifests_quick"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing manifests without rebuilding them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing managed Phase 01 output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = root / (QUICK_OUTPUT_DIR if args.quick else FULL_OUTPUT_DIR)

    try:
        if args.validate_only:
            validate_only(root, output_dir)
        else:
            build(
                root=root,
                output_dir=output_dir,
                seed=args.seed,
                quick=args.quick,
                force=args.force,
            )
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. Official output files were not replaced.")
    except Phase01Error as exc:
        raise SystemExit(f"Phase 01 error: {exc}") from exc


if __name__ == "__main__":
    main()
