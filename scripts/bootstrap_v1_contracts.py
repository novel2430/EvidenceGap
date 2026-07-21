#!/usr/bin/env python3
"""
Generate and validate EvidenceGap V1 Phase 00 machine-readable contracts.

This script completes:
  Step 2: write dataset_mappings.json and label_maps.json
  Step 3: build one canonical example from each raw dataset
  Step 4: validate mappings, label domains, indices, reverse mappings, and examples

It never modifies data/raw/v1.

Dependencies:
  python -m pip install pyarrow pandas ijson

Usage:
  python scripts/bootstrap_v1_contracts.py --root .
  python scripts/bootstrap_v1_contracts.py --root . --quick
  python scripts/bootstrap_v1_contracts.py --root . --validate-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

try:
    import ijson
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install with:\n"
        "  python -m pip install pyarrow pandas ijson"
    ) from exc


CONTRACT_VERSION = "1.0.0"

MEDFACT_GLOB = "data/raw/v1/medfact_synth/data/*.parquet"
EVIDENCEBENCH_TRAIN = Path(
    "data/raw/v1/evidencebench_100k/evidencebench_100k_train_set.json"
)
EVIDENCEBENCH_TEST = Path(
    "data/raw/v1/evidencebench_100k/evidencebench_100k_test_set.json"
)
HEALTHFC_PATH = Path("data/raw/v1/healthfc/Datensatz.csv")

CONTRACT_DIR = Path("data/contracts/v1")
EXAMPLE_DIR = CONTRACT_DIR / "examples"
REPORT_PATH = CONTRACT_DIR / "validation_report.json"

MEDFACT_REQUIRED_FIELDS = {
    "idx",
    "claim_pmid",
    "claim_potential",
    "claim",
    "source_pmid",
    "source",
    "synthetic_label",
}

EVIDENCEBENCH_REQUIRED_FIELDS = {
    "hypothesis",
    "paper_as_candidate_pool",
    "aspect_list_ids",
    "aspect_id2aspect",
    "aspect2sentence_indices",
    "sentence_index2aspects",
    "sentence_types_in_candidate_pool",
    "paper_id",
    "systematic_review_id",
}

HEALTHFC_REQUIRED_FIELDS = {
    "en_claim",
    "en_explanation",
    "en_top_sentences",
    "label",
    "authors",
    "date",
    "url",
}

DATASET_MAPPINGS: dict[str, Any] = {
    "schema_version": CONTRACT_VERSION,
    "description": "EvidenceGap V1 raw-to-canonical field mappings.",
    "datasets": {
        "medfact_synth": {
            "raw_format": "parquet",
            "raw_path_glob": MEDFACT_GLOB,
            "roles": [
                "judged_article_ranking",
                "open_corpus_retrieval",
                "five_level_stance",
            ],
            "fields": {
                "row_id": "idx",
                "claim_source_id": "claim_pmid",
                "claim_potential": "claim_potential",
                "claim_text": "claim",
                "article_source_id": "source_pmid",
                "article_text": "source",
                "stance_label": "synthetic_label",
            },
            "derived_fields": {
                "relevance_grade": "abs(int(synthetic_label))",
                "is_origin_source": "str(claim_pmid) == str(source_pmid)",
                "claim_id": "medfact:{claim_pmid}:{sha256(normalized_claim)[:16]}",
                "split_group_id": "medfact-group:{claim_pmid}:{sha256(normalized_claim)[:16]}",
                "article_id": "pmid:{source_pmid}",
                "judgment_id": "medfact-pair:{idx}",
            },
            "input_exclusions": [
                "system_prompt",
                "user_prompt",
                "assistant_prompt",
            ],
            "judgment_completeness": {
                "judged_candidate_ranking": "complete only within observed pairs for each claim",
                "open_corpus": "incomplete; unseen claim-article pairs are UNJUDGED",
            },
        },
        "evidencebench_100k": {
            "raw_format": "json_object",
            "raw_paths": {
                "train": str(EVIDENCEBENCH_TRAIN),
                "test": str(EVIDENCEBENCH_TEST),
            },
            "roles": ["evidence_sentence_ranking"],
            "fields": {
                "query_text": "hypothesis",
                "candidate_sentences": "paper_as_candidate_pool",
                "aspect_ids": "aspect_list_ids",
                "aspect_text": "aspect_id2aspect",
                "aspect_to_sentence_indices": "aspect2sentence_indices",
                "sentence_to_aspects": "sentence_index2aspects",
                "results_aspect_ids": "results_aspect_list_ids",
                "sentence_types": "sentence_types_in_candidate_pool",
                "paper_id": "paper_id",
                "group_id": "systematic_review_id",
            },
            "input_exclusions": [
                "results_evidence_retrieval_at_5_evaluation",
                "results_evidence_retrieval_at_optimal_evaluation",
            ],
            "index_invariants": [
                "candidate sentence order is immutable",
                "sentence indices are zero-based",
                "all gold indices must be inside candidate pool",
            ],
        },
        "healthfc": {
            "raw_format": "csv",
            "raw_path": str(HEALTHFC_PATH),
            "roles": ["expert_verdict_evaluation"],
            "fields": {
                "claim_text": "en_claim",
                "evidence_text": "en_top_sentences",
                "label": "label",
                "explanation": "en_explanation",
                "authors": "authors",
                "date": "date",
                "source_url": "url",
            },
            "input_fields": ["en_claim", "en_top_sentences"],
            "input_exclusions": ["en_explanation", "label", "de_verdict"],
            "usage": "external expert evaluation only",
        },
    },
}

LABEL_MAPS: dict[str, Any] = {
    "schema_version": CONTRACT_VERSION,
    "medfact_stance_5": {
        "task_id": "STANCE-MEDFACT-5",
        "ordered": True,
        "labels": {
            "-2": "STRONG_REFUTE",
            "-1": "PARTIAL_REFUTE",
            "0": "NEUTRAL_OR_INSUFFICIENT",
            "1": "PARTIAL_SUPPORT",
            "2": "STRONG_SUPPORT",
        },
    },
    "medfact_relevance_3": {
        "task_id": "AR-MEDFACT-JUDGED",
        "ordered": True,
        "labels": {
            "0": "NO_POSITIVE_JUDGMENT",
            "1": "PARTIALLY_RELEVANT",
            "2": "STRONGLY_RELEVANT",
        },
        "derivation": "abs(medfact_stance_5)",
        "note": "Grade 0 combines unrelated, unaddressed, and insufficient evidence under the source labeling prompt.",
    },
    "healthfc_verdict_3": {
        "task_id": "VERDICT-HEALTHFC-3",
        "ordered": False,
        "labels": {
            "0": "SUPPORTED",
            "1": "NOT_ENOUGH_INFORMATION",
            "2": "REFUTED",
        },
    },
    "medfact_to_healthfc_probability_fold": {
        "SUPPORTED": [1, 2],
        "NOT_ENOUGH_INFORMATION": [0],
        "REFUTED": [-2, -1],
    },
}


@dataclass
class ValidationState:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)

    def pass_check(self, name: str, details: Any = None) -> None:
        item: dict[str, Any] = {"name": name, "status": "PASS"}
        if details is not None:
            item["details"] = details
        self.checks.append(item)

    def warn(self, name: str, message: str) -> None:
        self.warnings.append(f"{name}: {message}")
        self.checks.append(
            {"name": name, "status": "WARN", "details": message}
        )

    def fail(self, name: str, message: str) -> None:
        self.errors.append(f"{name}: {message}")
        self.checks.append(
            {"name": name, "status": "FAIL", "details": message}
        )


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


def text_hash(value: Any) -> str:
    normalized = normalize_text(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def detect_csv(path: Path) -> tuple[str, str]:
    encodings = ("utf-8-sig", "utf-8", "latin-1")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            sample = path.read_text(encoding=encoding)[:65536]
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            return encoding, dialect.delimiter
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_error = exc
    raise RuntimeError(f"Could not detect CSV format: {last_error}")


def json_container(path: Path) -> str:
    with path.open("rb") as handle:
        first = handle.read(4096).lstrip()[:1]
    if first == b"{":
        return "object"
    if first == b"[":
        return "array"
    raise RuntimeError(f"Unsupported JSON top-level format: {path}")


def iter_json_records(path: Path) -> Iterator[tuple[str, Any]]:
    container = json_container(path)
    with path.open("rb") as handle:
        if container == "object":
            yield from ijson.kvitems(handle, "")
        else:
            for index, item in enumerate(ijson.items(handle, "item")):
                yield str(index), item


def first_medfact_row(root: Path) -> tuple[Path, dict[str, Any]]:
    paths = sorted(root.glob(MEDFACT_GLOB))
    if not paths:
        raise FileNotFoundError(f"No MedFact shards matched {MEDFACT_GLOB}")
    path = paths[0]
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1):
        rows = batch.to_pylist()
        if rows:
            return path, rows[0]
    raise RuntimeError(f"Empty MedFact shard: {path}")


def first_evidencebench_row(root: Path) -> tuple[Path, str, dict[str, Any]]:
    for relative_path in (EVIDENCEBENCH_TRAIN, EVIDENCEBENCH_TEST):
        path = root / relative_path
        if not path.exists():
            continue
        for record_id, record in iter_json_records(path):
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"EvidenceBench first record is not an object: {record_id}"
                )
            return path, str(record_id), record
    raise FileNotFoundError("No EvidenceBench train/test file found")


def first_healthfc_row(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root / HEALTHFC_PATH
    encoding, delimiter = detect_csv(path)
    frame = pd.read_csv(
        path,
        encoding=encoding,
        sep=delimiter,
        nrows=1,
        low_memory=False,
    )
    if frame.empty:
        raise RuntimeError(f"Empty HealthFC CSV: {path}")
    row = frame.iloc[0].where(pd.notna(frame.iloc[0]), None).to_dict()
    return path, row


def as_int_label(value: Any) -> int:
    if value is None:
        raise ValueError("label is null")
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"label is not an integer value: {value!r}")
    return int(number)


def build_medfact_example(root: Path) -> dict[str, Any]:
    path, row = first_medfact_row(root)
    claim = str(row["claim"])
    claim_pmid = str(row["claim_pmid"])
    source_pmid = str(row["source_pmid"])
    row_id = str(row["idx"])
    digest = text_hash(claim)
    stance = as_int_label(row["synthetic_label"])
    locator = {
        "path": relative(path, root),
        "record_id": row_id,
        "shard": path.name,
    }

    claim_record = {
        "record_type": "ClaimRecord",
        "claim_id": f"medfact:{claim_pmid}:{digest}",
        "dataset": "medfact_synth",
        "text": claim,
        "source_reference": {"type": "pmid", "value": claim_pmid},
        "split_group_id": f"medfact-group:{claim_pmid}:{digest}",
        "contract_version": CONTRACT_VERSION,
        "raw_locator": locator,
    }
    article_record = {
        "record_type": "ArticleRecord",
        "article_id": f"pmid:{source_pmid}",
        "dataset": "medfact_synth",
        "pmid": source_pmid,
        "text": str(row["source"]),
        "contract_version": CONTRACT_VERSION,
        "raw_locator": locator,
    }
    judgment = {
        "record_type": "ClaimArticleJudgment",
        "judgment_id": f"medfact-pair:{row_id}",
        "claim_id": claim_record["claim_id"],
        "article_id": article_record["article_id"],
        "stance_label": stance,
        "relevance_grade": abs(stance),
        "is_origin_source": claim_pmid == source_pmid,
        "claim_potential": None
        if row.get("claim_potential") is None
        else str(row.get("claim_potential")),
        "dataset": "medfact_synth",
        "contract_version": CONTRACT_VERSION,
        "raw_locator": locator,
    }
    return {
        "schema_version": CONTRACT_VERSION,
        "task_ids": ["AR-MEDFACT-JUDGED", "STANCE-MEDFACT-5"],
        "claim": claim_record,
        "article": article_record,
        "judgment": judgment,
    }


def build_evidencebench_example(root: Path) -> dict[str, Any]:
    path, record_id, row = first_evidencebench_row(root)
    return {
        "schema_version": CONTRACT_VERSION,
        "task_id": "ESR-EVIDENCEBENCH",
        "record_type": "EvidenceQueryRecord",
        "query_id": f"evidencebench:{record_id}",
        "dataset": "evidencebench_100k",
        "hypothesis": row.get("hypothesis"),
        "paper_id": row.get("paper_id"),
        "systematic_review_id": row.get("systematic_review_id"),
        "candidate_sentences": row.get("paper_as_candidate_pool"),
        "sentence_types": row.get("sentence_types_in_candidate_pool"),
        "aspect_ids": row.get("aspect_list_ids"),
        "aspect_text": row.get("aspect_id2aspect"),
        "aspect_to_sentence_indices": row.get("aspect2sentence_indices"),
        "sentence_to_aspects": row.get("sentence_index2aspects"),
        "results_aspect_ids": row.get("results_aspect_list_ids"),
        "contract_version": CONTRACT_VERSION,
        "raw_locator": {
            "path": relative(path, root),
            "record_id": record_id,
        },
    }


def build_healthfc_example(root: Path) -> dict[str, Any]:
    path, row = first_healthfc_row(root)
    claim = str(row["en_claim"])
    label = as_int_label(row["label"])
    verdict = LABEL_MAPS["healthfc_verdict_3"]["labels"][str(label)]
    return {
        "schema_version": CONTRACT_VERSION,
        "task_id": "VERDICT-HEALTHFC-3",
        "record_type": "ExpertVerdictRecord",
        "case_id": f"healthfc:{text_hash(claim)}",
        "dataset": "healthfc",
        "claim": claim,
        "evidence_text": str(row["en_top_sentences"]),
        "label_id": label,
        "verdict": verdict,
        "explanation": row.get("en_explanation"),
        "source_url": row.get("url"),
        "authors": row.get("authors"),
        "date": row.get("date"),
        "contract_version": CONTRACT_VERSION,
        "raw_locator": {
            "path": relative(path, root),
            "record_id": "0",
        },
    }


def write_contracts(root: Path) -> None:
    contract_dir = root / CONTRACT_DIR
    example_dir = root / EXAMPLE_DIR
    contract_dir.mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)

    json_dump(contract_dir / "dataset_mappings.json", DATASET_MAPPINGS)
    json_dump(contract_dir / "label_maps.json", LABEL_MAPS)
    json_dump(
        example_dir / "medfact_claim_article.json",
        build_medfact_example(root),
    )
    json_dump(
        example_dir / "evidencebench_query.json",
        build_evidencebench_example(root),
    )
    json_dump(
        example_dir / "healthfc_verdict.json",
        build_healthfc_example(root),
    )


def validate_contract_files(root: Path, state: ValidationState) -> None:
    expected = [
        root / CONTRACT_DIR / "dataset_mappings.json",
        root / CONTRACT_DIR / "label_maps.json",
        root / EXAMPLE_DIR / "medfact_claim_article.json",
        root / EXAMPLE_DIR / "evidencebench_query.json",
        root / EXAMPLE_DIR / "healthfc_verdict.json",
    ]
    missing = [relative(path, root) for path in expected if not path.exists()]
    if missing:
        state.fail("Contract files", f"Missing: {missing}")
        return

    for path in expected:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            state.fail("Contract JSON", f"{relative(path, root)}: {exc}")
            return
    state.pass_check("Contract files", [relative(path, root) for path in expected])


def validate_medfact(
    root: Path,
    state: ValidationState,
    quick: bool,
) -> None:
    paths = sorted(root.glob(MEDFACT_GLOB))
    if not paths:
        state.fail("MedFact mapping", f"No files matched {MEDFACT_GLOB}")
        return

    schemas: set[tuple[str, ...]] = set()
    labels: Counter[int] = Counter()
    rows = 0
    max_rows = 50_000 if quick else None

    try:
        for path in paths:
            parquet = pq.ParquetFile(path)
            fields = set(parquet.schema_arrow.names)
            missing = MEDFACT_REQUIRED_FIELDS - fields
            if missing:
                state.fail(
                    "MedFact mapping",
                    f"{path.name} missing fields: {sorted(missing)}",
                )
                return
            schemas.add(tuple(parquet.schema_arrow.names))

            for batch in parquet.iter_batches(
                batch_size=16384,
                columns=["synthetic_label"],
            ):
                for value in batch.column(0).to_pylist():
                    label = as_int_label(value)
                    labels[label] += 1
                    rows += 1
                    if max_rows is not None and rows >= max_rows:
                        break
                if max_rows is not None and rows >= max_rows:
                    break
            if max_rows is not None and rows >= max_rows:
                break
    except Exception as exc:
        state.fail("MedFact mapping", str(exc))
        return

    invalid = sorted(set(labels) - {-2, -1, 0, 1, 2})
    if invalid:
        state.fail("MedFact label domain", f"Unexpected labels: {invalid}")
        return
    if len(schemas) != 1 and not quick:
        state.fail("MedFact schema consistency", f"Variants: {len(schemas)}")
        return

    state.pass_check(
        "MedFact mapping",
        {
            "files_scanned": 1 if quick and rows >= 50_000 else len(paths),
            "rows_scanned": rows,
            "label_counts": dict(sorted(labels.items())),
            "mode": "quick" if quick else "full",
        },
    )
    state.pass_check("MedFact label domain", sorted(labels))


def _aspect_mapping_errors(
    record_id: str,
    row: dict[str, Any],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    candidates = row.get("paper_as_candidate_pool")
    sentence_types = row.get("sentence_types_in_candidate_pool")
    aspect_ids = row.get("aspect_list_ids")
    aspect_text = row.get("aspect_id2aspect")
    aspect_to_indices = row.get("aspect2sentence_indices")
    sentence_to_aspects = row.get("sentence_index2aspects")

    if not isinstance(candidates, list):
        return [f"{record_id}: candidate pool is not a list"], warnings
    if not isinstance(sentence_types, list):
        errors.append(f"{record_id}: sentence types is not a list")
    elif len(sentence_types) != len(candidates):
        errors.append(
            f"{record_id}: {len(sentence_types)} sentence types for "
            f"{len(candidates)} candidates"
        )

    if not isinstance(aspect_ids, list):
        errors.append(f"{record_id}: aspect_list_ids is not a list")
        aspect_ids = []
    if not isinstance(aspect_text, dict):
        errors.append(f"{record_id}: aspect_id2aspect is not an object")
        aspect_text = {}
    if not isinstance(aspect_to_indices, dict):
        errors.append(f"{record_id}: aspect2sentence_indices is not an object")
        aspect_to_indices = {}
    if not isinstance(sentence_to_aspects, dict):
        errors.append(f"{record_id}: sentence_index2aspects is not an object")
        sentence_to_aspects = {}

    for aspect_id in aspect_ids:
        if aspect_id not in aspect_text:
            errors.append(f"{record_id}: missing aspect text for {aspect_id}")
        if aspect_id not in aspect_to_indices:
            errors.append(f"{record_id}: missing indices for {aspect_id}")

    reverse_pairs: set[tuple[str, int]] = set()
    for sentence_index, aspects in sentence_to_aspects.items():
        try:
            index = int(sentence_index)
        except (TypeError, ValueError):
            errors.append(
                f"{record_id}: non-integer sentence index key {sentence_index!r}"
            )
            continue
        if index < 0 or index >= len(candidates):
            errors.append(
                f"{record_id}: reverse index {index} out of range "
                f"for {len(candidates)} candidates"
            )
        if not isinstance(aspects, list):
            errors.append(
                f"{record_id}: reverse aspects at sentence {index} is not a list"
            )
            continue
        for aspect_id in aspects:
            reverse_pairs.add((str(aspect_id), index))

    forward_pairs: set[tuple[str, int]] = set()
    for aspect_id, indices in aspect_to_indices.items():
        if not isinstance(indices, list):
            errors.append(f"{record_id}: indices for {aspect_id} is not a list")
            continue
        for raw_index in indices:
            if isinstance(raw_index, bool):
                errors.append(
                    f"{record_id}: boolean index for {aspect_id}: {raw_index}"
                )
                continue
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                errors.append(
                    f"{record_id}: non-integer index for {aspect_id}: {raw_index!r}"
                )
                continue
            if index < 0 or index >= len(candidates):
                errors.append(
                    f"{record_id}: index {index} for {aspect_id} out of range "
                    f"for {len(candidates)} candidates"
                )
            forward_pairs.add((str(aspect_id), index))

    missing_reverse = forward_pairs - reverse_pairs
    extra_reverse = reverse_pairs - forward_pairs
    if missing_reverse:
        warnings.append(
            f"{record_id}: {len(missing_reverse)} forward pairs absent from reverse map"
        )
    if extra_reverse:
        warnings.append(
            f"{record_id}: {len(extra_reverse)} reverse pairs absent from forward map"
        )

    return errors, warnings


def validate_evidencebench(
    root: Path,
    state: ValidationState,
    quick: bool,
) -> None:
    files = [root / EVIDENCEBENCH_TRAIN, root / EVIDENCEBENCH_TEST]
    missing = [relative(path, root) for path in files if not path.exists()]
    if missing:
        state.fail("EvidenceBench mapping", f"Missing: {missing}")
        return

    limit = 1_000 if quick else None
    total_rows = 0
    hard_errors: list[str] = []
    reverse_warning_count = 0
    reverse_warning_samples: list[str] = []
    split_counts: dict[str, int] = {}

    try:
        for path in files:
            split_rows = 0
            for record_id, row in iter_json_records(path):
                split_rows += 1
                total_rows += 1
                if not isinstance(row, dict):
                    hard_errors.append(f"{record_id}: record is not an object")
                    continue
                missing_fields = EVIDENCEBENCH_REQUIRED_FIELDS - set(row)
                if missing_fields:
                    hard_errors.append(
                        f"{record_id}: missing fields {sorted(missing_fields)}"
                    )
                row_errors, row_warnings = _aspect_mapping_errors(
                    str(record_id), row
                )
                hard_errors.extend(row_errors)
                if row_warnings:
                    reverse_warning_count += len(row_warnings)
                    remaining = 10 - len(reverse_warning_samples)
                    if remaining > 0:
                        reverse_warning_samples.extend(row_warnings[:remaining])

                if len(hard_errors) >= 50:
                    break
                if limit is not None and split_rows >= limit:
                    break
                if split_rows % 10_000 == 0:
                    print(
                        f"  EvidenceBench {path.name}: {split_rows:,} records",
                        flush=True,
                    )
            split_counts[path.name] = split_rows
            if len(hard_errors) >= 50:
                break
    except Exception as exc:
        state.fail("EvidenceBench mapping", str(exc))
        return

    if hard_errors:
        preview = hard_errors[:20]
        state.fail(
            "EvidenceBench mapping",
            f"{len(hard_errors)} errors; first entries: {preview}",
        )
        return

    state.pass_check(
        "EvidenceBench mapping",
        {
            "rows_scanned": total_rows,
            "split_counts": split_counts,
            "mode": "quick" if quick else "full",
        },
    )
    state.pass_check("EvidenceBench index bounds")

    if reverse_warning_count:
        state.warn(
            "EvidenceBench reverse mapping",
            f"{reverse_warning_count} forward/reverse differences; "
            f"first entries: {reverse_warning_samples}",
        )
    else:
        state.pass_check("EvidenceBench reverse mapping")


def validate_healthfc(root: Path, state: ValidationState) -> None:
    path = root / HEALTHFC_PATH
    if not path.exists():
        state.fail("HealthFC mapping", f"Missing: {relative(path, root)}")
        return

    try:
        encoding, delimiter = detect_csv(path)
        frame = pd.read_csv(
            path,
            encoding=encoding,
            sep=delimiter,
            usecols=lambda column: column in HEALTHFC_REQUIRED_FIELDS,
            low_memory=False,
        )
    except Exception as exc:
        state.fail("HealthFC mapping", str(exc))
        return

    missing_fields = HEALTHFC_REQUIRED_FIELDS - set(frame.columns)
    if missing_fields:
        state.fail("HealthFC mapping", f"Missing fields: {sorted(missing_fields)}")
        return

    labels: Counter[int] = Counter()
    try:
        for value in frame["label"]:
            labels[as_int_label(value)] += 1
    except Exception as exc:
        state.fail("HealthFC label domain", str(exc))
        return

    invalid = sorted(set(labels) - {0, 1, 2})
    if invalid:
        state.fail("HealthFC label domain", f"Unexpected labels: {invalid}")
        return

    state.pass_check(
        "HealthFC mapping",
        {
            "rows": len(frame),
            "encoding": encoding,
            "delimiter": delimiter,
        },
    )
    state.pass_check("HealthFC label domain", dict(sorted(labels.items())))


def validate_examples(root: Path, state: ValidationState) -> None:
    try:
        medfact = json.loads(
            (root / EXAMPLE_DIR / "medfact_claim_article.json").read_text(
                encoding="utf-8"
            )
        )
        evidence = json.loads(
            (root / EXAMPLE_DIR / "evidencebench_query.json").read_text(
                encoding="utf-8"
            )
        )
        healthfc = json.loads(
            (root / EXAMPLE_DIR / "healthfc_verdict.json").read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        state.fail("Contract examples", str(exc))
        return

    errors: list[str] = []

    if medfact.get("schema_version") != CONTRACT_VERSION:
        errors.append("MedFact example schema_version mismatch")
    judgment = medfact.get("judgment", {})
    if judgment.get("relevance_grade") != abs(
        int(judgment.get("stance_label", 999))
    ):
        errors.append("MedFact relevance_grade != abs(stance_label)")
    if judgment.get("claim_id") != medfact.get("claim", {}).get("claim_id"):
        errors.append("MedFact judgment claim_id mismatch")
    if judgment.get("article_id") != medfact.get("article", {}).get(
        "article_id"
    ):
        errors.append("MedFact judgment article_id mismatch")

    candidates = evidence.get("candidate_sentences")
    sentence_types = evidence.get("sentence_types")
    if not isinstance(candidates, list) or not isinstance(sentence_types, list):
        errors.append("EvidenceBench example candidate/sentence types are not lists")
    elif len(candidates) != len(sentence_types):
        errors.append("EvidenceBench example sentence types length mismatch")

    label_id = str(healthfc.get("label_id"))
    expected_verdict = LABEL_MAPS["healthfc_verdict_3"]["labels"].get(label_id)
    if healthfc.get("verdict") != expected_verdict:
        errors.append("HealthFC example verdict does not match label map")
    forbidden = {"en_explanation", "label", "de_verdict"}
    if forbidden & set(healthfc):
        errors.append("HealthFC example leaks raw forbidden input field names")

    if errors:
        state.fail("Contract examples", errors)
    else:
        state.pass_check("Contract examples")


def validate_all(root: Path, quick: bool) -> ValidationState:
    state = ValidationState()
    validate_contract_files(root, state)
    validate_medfact(root, state, quick=quick)
    validate_evidencebench(root, state, quick=quick)
    validate_healthfc(root, state)
    validate_examples(root, state)
    return state


def write_report(root: Path, state: ValidationState, quick: bool) -> None:
    payload = {
        "schema_version": CONTRACT_VERSION,
        "mode": "quick" if quick else "full",
        "ok": not state.errors,
        "checks": state.checks,
        "errors": state.errors,
        "warnings": state.warnings,
    }
    json_dump(root / REPORT_PATH, payload)


def print_summary(root: Path, state: ValidationState) -> None:
    print("\n=== EvidenceGap V1 Phase 00 contract validation ===")
    for check in state.checks:
        print(f"{check['name']}: {check['status']}")
    print(f"Errors: {len(state.errors)}")
    print(f"Warnings: {len(state.warnings)}")
    print(f"Report: {root / REPORT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Do not rewrite machine-readable contracts/examples.",
    )
    parser.add_argument(
        "--write-only",
        action="store_true",
        help="Write contracts/examples but skip validation.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Scan up to 50k MedFact rows and 1k EvidenceBench rows per split.",
    )
    args = parser.parse_args()
    if args.validate_only and args.write_only:
        parser.error("--validate-only and --write-only cannot be combined")
    return args


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    if not args.validate_only:
        write_contracts(root)
        print(f"Written machine-readable contracts under {root / CONTRACT_DIR}")

    if args.write_only:
        return

    state = validate_all(root, quick=args.quick)
    write_report(root, state, quick=args.quick)
    print_summary(root, state)

    if state.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
