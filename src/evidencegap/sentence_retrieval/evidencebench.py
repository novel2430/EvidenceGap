from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    load_json,
    relative_path,
    sha256_file,
    sha256_text,
)
from evidencegap.sentence_retrieval.contracts import (
    SCHEMA_VERSION,
    EvidenceQuery,
    canonicalize_raw_record,
)

DEFAULT_MANIFEST_DIR = Path("data/processed/v1/manifests")
DEFAULT_CANONICAL_ROOT = Path("artifacts/v1/evidence_sentence_retrieval/canonical")


def _ijson() -> Any:
    try:
        import ijson
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing ijson. Install requirements/v1-phase05.txt"
        ) from exc
    return ijson


def canonical_subset_name(split: str, max_queries: int | None) -> str:
    return f"{split}_full" if max_queries is None else f"{split}_first_{max_queries}"


def canonical_dir_for(
    root: Path, *, split: str, max_queries: int | None, output_dir: Path | None = None
) -> Path:
    return (
        output_dir.resolve()
        if output_dir is not None
        else (root / DEFAULT_CANONICAL_ROOT / canonical_subset_name(split, max_queries)).resolve()
    )


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing EvidenceBench manifest: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceGapError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise EvidenceGapError(f"Manifest record is not an object: {path}:{line_number}")
            yield value


def select_manifest_records(
    root: Path,
    *,
    split: str,
    max_queries: int | None,
    manifest_dir: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    if split not in {"train", "dev", "test"}:
        raise EvidenceGapError("split must be train, dev, or test")
    if max_queries is not None and max_queries <= 0:
        raise EvidenceGapError("max_queries must be positive")
    base = manifest_dir.resolve() if manifest_dir else (root / DEFAULT_MANIFEST_DIR)
    path = base / f"evidencebench_{split}.jsonl"
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in _iter_jsonl(path):
        query_id = str(record.get("query_id", ""))
        if not query_id:
            raise EvidenceGapError(f"Manifest record in {path} has no query_id")
        if query_id in seen:
            raise EvidenceGapError(f"Duplicate query_id in manifest: {query_id}")
        seen.add(query_id)
        if record.get("dataset") != "evidencebench_100k":
            raise EvidenceGapError(f"Unexpected dataset for {query_id}")
        if record.get("split") != split:
            raise EvidenceGapError(f"Split mismatch for {query_id}")
        locator = record.get("raw_locator")
        if not isinstance(locator, dict) or not locator.get("path") or locator.get("record_id") is None:
            raise EvidenceGapError(f"Invalid raw_locator for {query_id}")
        records.append(record)
        if max_queries is not None and len(records) >= max_queries:
            break
    if not records:
        raise EvidenceGapError(f"No EvidenceBench records selected from {path}")
    return path, records


def _json_container(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            while True:
                byte = handle.read(1)
                if not byte:
                    raise EvidenceGapError(f"Empty raw JSON file: {path}")
                if byte in b" \t\r\n":
                    continue
                if byte == b"{":
                    return "object"
                if byte == b"[":
                    return "array"
                raise EvidenceGapError(f"Raw JSON root must be object or array: {path}")
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing raw EvidenceBench file: {path}") from exc


def _stream_selected_raw(
    path: Path, selected_ids: set[str]
) -> dict[str, Mapping[str, Any]]:
    found: dict[str, Mapping[str, Any]] = {}
    container = _json_container(path)
    try:
        ijson = _ijson()
    except EvidenceGapError:
        # Tiny-fixture fallback keeps contract/audit development usable in minimal
        # environments. Official multi-GB datasets still require streaming ijson.
        if path.stat().st_size > 16 * 1024 * 1024:
            raise
        value = json.loads(path.read_text(encoding="utf-8"))
        iterator = (
            ((str(key), item) for key, item in value.items())
            if container == "object"
            else ((str(index), item) for index, item in enumerate(value))
        )
        for record_id, item in iterator:
            if record_id in selected_ids:
                if not isinstance(item, dict):
                    raise EvidenceGapError(
                        f"Raw EvidenceBench record {record_id} is not an object"
                    )
                found[record_id] = item
    else:
        with path.open("rb") as handle:
            if container == "object":
                iterator = ((str(key), value) for key, value in ijson.kvitems(handle, ""))
            else:
                iterator = ((str(index), value) for index, value in enumerate(ijson.items(handle, "item")))
            for record_id, value in iterator:
                if record_id not in selected_ids:
                    continue
                if record_id in found:
                    raise EvidenceGapError(f"Duplicate raw EvidenceBench record ID: {record_id}")
                if not isinstance(value, dict):
                    raise EvidenceGapError(f"Raw EvidenceBench record {record_id} is not an object")
                found[record_id] = value
                if len(found) == len(selected_ids):
                    break
    missing = selected_ids - set(found)
    if missing:
        sample = sorted(missing)[:10]
        raise EvidenceGapError(f"Raw EvidenceBench records not found in {path}: {sample}")
    return found


def _raw_source_provenance(root: Path, path: Path) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": relative_path(root, path),
        "bytes": path.stat().st_size,
    }
    download_manifest = path.parent / "download_manifest.json"
    if download_manifest.exists():
        manifest = load_json(download_manifest)
        value["download_manifest_path"] = relative_path(root, download_manifest)
        value["download_manifest_sha256"] = sha256_file(download_manifest)
        value["dataset_revision"] = manifest.get("revision")
        expected_size = manifest.get("files", {}).get(path.name)
        if expected_size is not None and int(expected_size) != path.stat().st_size:
            raise EvidenceGapError(
                f"Raw file size differs from download manifest: {path}"
            )
    return value


def load_selected_queries(
    root: Path,
    *,
    split: str,
    max_queries: int | None = None,
    manifest_dir: Path | None = None,
) -> tuple[Path, list[EvidenceQuery], dict[str, Any]]:
    root = root.resolve()
    manifest_path, manifest_records = select_manifest_records(
        root,
        split=split,
        max_queries=max_queries,
        manifest_dir=manifest_dir,
    )
    by_raw_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for record in manifest_records:
        raw_path = (root / str(record["raw_locator"]["path"])).resolve()
        by_raw_path[raw_path].append(record)

    raw_by_locator: dict[tuple[Path, str], Mapping[str, Any]] = {}
    for raw_path, records in by_raw_path.items():
        ids = {str(record["raw_locator"]["record_id"]) for record in records}
        values = _stream_selected_raw(raw_path, ids)
        raw_by_locator.update({(raw_path, key): value for key, value in values.items()})

    queries: list[EvidenceQuery] = []
    for manifest_record in manifest_records:
        raw_path = (root / str(manifest_record["raw_locator"]["path"])).resolve()
        raw_id = str(manifest_record["raw_locator"]["record_id"])
        queries.append(
            canonicalize_raw_record(
                manifest_record=manifest_record,
                raw_record_id=raw_id,
                raw_record=raw_by_locator[(raw_path, raw_id)],
            )
        )

    query_ids = [query.query_id for query in queries]
    stats = _audit_stats(queries)
    stats.update(
        {
            "manifest_path": relative_path(root, manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "selected_query_id_fingerprint": sha256_text("\n".join(query_ids) + "\n"),
            "selected_queries": len(queries),
            "max_queries": max_queries,
            "split": split,
            "raw_sources": [
                _raw_source_provenance(root, path) for path in sorted(by_raw_path)
            ],
        }
    )
    return manifest_path, queries, stats


def _audit_stats(queries: Sequence[EvidenceQuery]) -> dict[str, Any]:
    sentence_counts = [len(query.candidate_sentences) for query in queries]
    aspect_counts = [len(query.aspect_ids) for query in queries]
    coverable_aspect_counts = [len(query.coverable_aspect_ids) for query in queries]
    unmapped_aspect_counts = [len(query.unmapped_aspect_ids) for query in queries]
    pool_to_papers: dict[str, set[str]] = defaultdict(set)
    paper_to_pools: dict[str, set[str]] = defaultdict(set)
    type_counts: Counter[str] = Counter()
    optimal_source_counts: Counter[str] = Counter()
    for query in queries:
        pool_to_papers[query.pool_fingerprint].add(query.paper_id)
        paper_to_pools[query.paper_id].add(query.pool_fingerprint)
        type_counts.update(query.sentence_types)
        if query.optimal_sentence_budget_source is not None:
            optimal_source_counts[query.optimal_sentence_budget_source] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "queries": len(queries),
        "unique_papers": len({query.paper_id for query in queries}),
        "unique_pools": len(pool_to_papers),
        "paper_pool_conflicts": sum(1 for values in paper_to_pools.values() if len(values) > 1),
        "candidate_sentences": sum(sentence_counts),
        "candidate_count_min": min(sentence_counts, default=0),
        "candidate_count_max": max(sentence_counts, default=0),
        "candidate_count_mean": (sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0.0),
        "candidate_pool_lt5_queries": sum(count < 5 for count in sentence_counts),
        "aspects": sum(aspect_counts),
        "coverable_aspects": sum(coverable_aspect_counts),
        "unmapped_aspects": sum(unmapped_aspect_counts),
        "queries_with_unmapped_aspects": sum(count > 0 for count in unmapped_aspect_counts),
        "empty_aspect_queries": sum(count == 0 for count in aspect_counts),
        "empty_coverable_aspect_queries": sum(
            count == 0 for count in coverable_aspect_counts
        ),
        "empty_results_aspect_queries": sum(not query.results_aspect_ids for query in queries),
        "queries_with_no_coverable_results_aspects": sum(
            not any(
                aspect_id in set(query.coverable_aspect_ids)
                for aspect_id in (query.results_aspect_ids or ())
            )
            for query in queries
        ),
        "optimal_budget_max": max((query.optimal_sentence_budget or 0 for query in queries), default=0),
        "optimal_budget_mean": (
            sum(query.optimal_sentence_budget or 0 for query in queries if query.optimal_sentence_budget is not None)
            / max(1, sum(query.optimal_sentence_budget is not None for query in queries))
        ),
        "sentence_type_counts": dict(sorted(type_counts.items())),
        "optimal_budget_source_counts": dict(sorted(optimal_source_counts.items())),
    }


def audit_evidencebench(
    root: Path,
    *,
    split: str,
    max_queries: int | None = None,
    manifest_dir: Path | None = None,
) -> dict[str, Any]:
    _manifest_path, _queries, stats = load_selected_queries(
        root,
        split=split,
        max_queries=max_queries,
        manifest_dir=manifest_dir,
    )
    stats["status"] = "PASS"
    return stats


def prepare_evidencebench_canonical(
    root: Path,
    *,
    split: str,
    max_queries: int | None = None,
    manifest_dir: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path, queries, stats = load_selected_queries(
        root,
        split=split,
        max_queries=max_queries,
        manifest_dir=manifest_dir,
    )
    target = canonical_dir_for(
        root, split=split, max_queries=max_queries, output_dir=output_dir
    )
    with atomic_directory(target, force=force) as staging:
        records_path = staging / "queries.jsonl"
        with records_path.open("w", encoding="utf-8") as handle:
            for query in queries:
                handle.write(json.dumps(query.to_dict(), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        manifest = {
            **stats,
            "status": "PASS",
            "canonical_path": relative_path(root, target / "queries.jsonl"),
            "canonical_sha256": sha256_file(records_path),
            "source_manifest": relative_path(root, manifest_path),
        }
        atomic_write_json(staging / "canonical_manifest.json", manifest)
    return manifest


def load_canonical_queries(canonical_dir: Path) -> tuple[list[EvidenceQuery], dict[str, Any]]:
    canonical_dir = canonical_dir.resolve()
    manifest_path = canonical_dir / "canonical_manifest.json"
    records_path = canonical_dir / "queries.jsonl"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing canonical manifest: {manifest_path}") from exc
    if sha256_file(records_path) != manifest.get("canonical_sha256"):
        raise EvidenceGapError(f"Canonical query checksum mismatch: {records_path}")
    queries = [EvidenceQuery.from_dict(value) for value in _iter_jsonl(records_path)]
    if len(queries) != int(manifest.get("selected_queries", -1)):
        raise EvidenceGapError("Canonical query count differs from manifest")
    if len({query.query_id for query in queries}) != len(queries):
        raise EvidenceGapError("Canonical artifact contains duplicate query IDs")
    return queries, manifest


def ensure_canonical(
    root: Path,
    *,
    split: str,
    max_queries: int | None,
    canonical_dir: Path | None,
) -> tuple[Path, list[EvidenceQuery], dict[str, Any]]:
    path = canonical_dir_for(
        root.resolve(), split=split, max_queries=max_queries, output_dir=canonical_dir
    )
    if not path.exists():
        prepare_evidencebench_canonical(
            root,
            split=split,
            max_queries=max_queries,
            output_dir=path,
            force=False,
        )
    queries, manifest = load_canonical_queries(path)
    source_manifest_value = manifest.get("manifest_path") or manifest.get("source_manifest")
    if source_manifest_value:
        source_manifest = (root.resolve() / str(source_manifest_value)).resolve()
        if not source_manifest.exists():
            raise EvidenceGapError(
                f"Canonical source manifest is missing: {source_manifest}"
            )
        if sha256_file(source_manifest) != manifest.get("manifest_sha256"):
            raise EvidenceGapError(
                f"Canonical subset is stale because its source manifest changed: {path}. "
                "Re-run prepare with --force."
            )
    if manifest.get("split") != split or manifest.get("max_queries") != max_queries:
        raise EvidenceGapError(
            f"Canonical subset does not match requested split/max_queries: {path}"
        )
    return path, queries, manifest
