#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

CIVIC_SPLITS = {"train", "dev", "test"}
CLINI_LABELS = {"0": "inconclusive", "1": "evidence", "2": "NEI"}


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def number(value: Any) -> int | None:
    try:
        return int(float(value)) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def passage_id(pmid: str, pmcid: str, section: str, start: int | None, end: int | None, body: str) -> str:
    key = "|".join([pmid, pmcid, section, str(start), str(end), body])
    return "civicfact:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]


def overlap_count(df: pd.DataFrame, column: str) -> tuple[int, list[dict[str, Any]]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for value, split in zip(df[column], df["split"], strict=False):
        value = text(value)
        if value:
            groups[value].add(text(split))
    overlaps = sorted((value, sorted(splits)) for value, splits in groups.items() if len(splits) > 1)
    return len(overlaps), [{"value": value, "splits": splits} for value, splits in overlaps[:10]]


def prepare_civicfact(source: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    benchmark: list[dict[str, Any]] = []
    generated_train: list[dict[str, Any]] = []
    passages: dict[str, dict[str, Any]] = {}
    raw_rows = 0

    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            raw_rows += 1

            split = text(row.get("partition"))
            generated = bool(row.get("generated"))
            flagged = bool(row.get("flagged"))
            include = split in CIVIC_SPLITS and not flagged and (not generated or split == "train")

            pmid = text(row.get("document.pmid"))
            pmcid = text(row.get("document.pmcid"))
            doi = text(row.get("document.doi"))
            title = text(row.get("document.title"))
            license_name = text(row.get("document.license"))

            gold_ids: list[str] = []
            evidence_items = row.get("evidence") or []

            for i, item in enumerate(evidence_items):
                if not isinstance(item, dict):
                    continue
                body = text(item.get("content"))
                if not body:
                    continue
                meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
                section = text(meta.get("section"))
                start = number(meta.get("start"))
                end = number(meta.get("end"))
                pid = passage_id(pmid, pmcid, section, start, end, body)
                gold_ids.append(pid)

                if include:
                    passages.setdefault(
                        pid,
                        {
                            "passage_id": pid,
                            "document_id": pmcid or pmid or doi or title,
                            "pmid": pmid,
                            "pmcid": pmcid,
                            "doi": doi,
                            "document_title": title,
                            "document_license": license_name,
                            "section": section,
                            "start": start,
                            "end": end,
                            "evidence_type": text(item.get("type")),
                            "in_table": bool(row.get("in_table")),
                            "text": body,
                            "source_example_id": text(row.get("id")),
                            "source_evidence_index": i,
                        },
                    )

            normalized = {
                "dataset": "civicfact",
                "example_id": text(row.get("id")),
                "display_id": text(row.get("display_id")),
                "claim_id": text(row.get("claim.id")),
                "claim_text": text(row.get("claim.flat")),
                "evidence_text": text(row.get("evidence.flat")),
                "raw_label": text(row.get("gold_label_name")),
                "split": split,
                "generated": generated,
                "flagged": flagged,
                "claim_generated": bool(row.get("claim.generated")),
                "evidence_generated": bool(row.get("evidence.generated")),
                "in_table": bool(row.get("in_table")),
                "pmid": pmid,
                "pmcid": pmcid,
                "doi": doi,
                "document_title": title,
                "document_license": license_name,
                "gold_passage_ids_json": json_text(gold_ids),
                "gold_passage_count": len(gold_ids),
                "claim_structure_json": json_text(row.get("claim") or []),
                "evidence_structure_json": json_text(evidence_items),
                "source_line": line_no,
            }

            if split in CIVIC_SPLITS and not generated and not flagged:
                benchmark.append(normalized)
            elif split == "train" and generated and not flagged:
                generated_train.append(normalized)

    benchmark_df = pd.DataFrame(benchmark).sort_values(["split", "example_id"]).reset_index(drop=True)
    generated_df = pd.DataFrame(generated_train).sort_values("example_id").reset_index(drop=True)
    passages_df = pd.DataFrame(passages.values()).sort_values(["document_id", "start", "passage_id"]).reset_index(drop=True)

    audit = {
        "raw_rows": raw_rows,
        "benchmark_rows": len(benchmark_df),
        "generated_train_rows": len(generated_df),
        "passage_rows": len(passages_df),
        "split_counts": benchmark_df["split"].value_counts().sort_index().to_dict(),
        "label_counts": benchmark_df["raw_label"].value_counts().sort_index().to_dict(),
        "missing_claim": int((benchmark_df["claim_text"] == "").sum()),
        "missing_evidence": int((benchmark_df["evidence_text"] == "").sum()),
        "missing_gold_passage": int((benchmark_df["gold_passage_count"] == 0).sum()),
    }
    for column in ("claim_text", "claim_id", "pmid", "pmcid"):
        count, sample = overlap_count(benchmark_df, column)
        audit[f"cross_split_{column}"] = {"count": count, "sample": sample}

    return benchmark_df, generated_df, passages_df, audit


def prepare_clinifact(source_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []

    for split, filename in (
        ("train", "train_set.csv"),
        ("validation", "validation_set.csv"),
        ("test", "test_set.csv"),
    ):
        path = source_dir / filename
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        required = {"index", "nctId", "claim", "PMID", "article_title", "article_abstract", "label"}
        missing = required - set(raw.columns)
        if missing:
            raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
        unknown = set(raw["label"]) - set(CLINI_LABELS)
        if unknown:
            raise RuntimeError(f"{path} unknown labels: {sorted(unknown)}")

        out = raw.rename(
            columns={
                "index": "source_index",
                "nctId": "nct_id",
                "claim": "claim_text",
                "PMID": "pmid",
            }
        ).copy()
        out.insert(0, "dataset", "clinifact")
        out.insert(1, "example_id", [f"clinifact:{split}:{x}" for x in out["source_index"]])
        out.insert(2, "split", split)
        out["raw_label_id"] = out["label"]
        out["raw_label"] = out["label"].map(CLINI_LABELS)
        out = out.drop(columns=["label"])
        frames.append(out)

    pairs = pd.concat(frames, ignore_index=True).sort_values(["split", "source_index"]).reset_index(drop=True)
    audit = {
        "rows": len(pairs),
        "split_counts": pairs["split"].value_counts().sort_index().to_dict(),
        "label_counts": pairs["raw_label"].value_counts().sort_index().to_dict(),
        "missing_claim": int((pairs["claim_text"].str.strip() == "").sum()),
        "missing_abstract": int((pairs["article_abstract"].str.strip() == "").sum()),
        "missing_pmid": int((pairs["pmid"].str.strip() == "").sum()),
    }
    for column in ("claim_text", "pmid", "nct_id"):
        count, sample = overlap_count(pairs, column)
        audit[f"cross_split_{column}"] = {"count": count, "sample": sample}

    return pairs, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    civic_source = root / "data/raw/civicfact/data.jsonl.gz"
    clini_source = root / "data/raw/clinifact"

    for path in (
        civic_source,
        clini_source / "train_set.csv",
        clini_source / "validation_set.csv",
        clini_source / "test_set.csv",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    print("Preparing CIViC-Fact...")
    civic, civic_generated, passages, civic_audit = prepare_civicfact(civic_source)
    print("Preparing CliniFact...")
    clini, clini_audit = prepare_clinifact(clini_source)

    expected = {"civic": 4541, "civic_generated": 3544, "clini": 1970}
    actual = {"civic": len(civic), "civic_generated": len(civic_generated), "clini": len(clini)}
    if expected != actual:
        message = f"unexpected row counts: expected={expected}, actual={actual}"
        if args.strict:
            raise RuntimeError(message)
        print("WARNING:", message)

    civic_dir = root / "data/processed/civicfact"
    clini_dir = root / "data/processed/clinifact"
    audit_dir = root / "data/processed/audit"
    civic_dir.mkdir(parents=True, exist_ok=True)
    clini_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    civic.to_parquet(civic_dir / "benchmark.parquet", index=False)
    civic_generated.to_parquet(civic_dir / "generated_train.parquet", index=False)
    passages.to_parquet(civic_dir / "passages.parquet", index=False)
    clini.to_parquet(clini_dir / "pairs.parquet", index=False)

    audit = {"civicfact": civic_audit, "clinifact": clini_audit}
    (audit_dir / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone")
    print(f"CIViC benchmark:       {len(civic)}")
    print(f"CIViC generated train: {len(civic_generated)}")
    print(f"CIViC passages:        {len(passages)}")
    print(f"CliniFact pairs:       {len(clini)}")
    print(f"Audit: {audit_dir / 'data_audit.json'}")


if __name__ == "__main__":
    main()

