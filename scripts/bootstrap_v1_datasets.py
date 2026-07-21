#!/usr/bin/env python3
"""
Clean obsolete EvidenceGap V1 dataset artifacts and download the new V1 datasets.

Datasets:
  - ncbi/MedFact-Synth
  - EvidenceBench/EvidenceBench-100k
  - jvladika/HealthFC

The script is:
  - idempotent: completed files are reused on repeated runs;
  - resumable: Hugging Face downloads resume automatically, HTTP downloads use .part files;
  - retryable: each network operation has exponential-backoff retries;
  - explicit about cleanup: only known obsolete CIViC-Fact / CliniFact / PMC artifacts are removed.

Examples:
  python scripts/bootstrap_v1_datasets.py \
    --root . \
    --clean-old \
    --proxy http://127.0.0.1:7899

  python scripts/bootstrap_v1_datasets.py --root . --dataset medfact-synth
  python scripts/bootstrap_v1_datasets.py --root . --verify-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, TypeVar

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: requests\n"
        "Install with:\n"
        "  python -m pip install -U requests 'huggingface_hub[hf_xet]'"
    ) from exc

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: huggingface_hub\n"
        "Install with:\n"
        "  python -m pip install -U requests 'huggingface_hub[hf_xet]'"
    ) from exc


T = TypeVar("T")

MEDFACT_REPO = "ncbi/MedFact-Synth"
EVIDENCEBENCH_REPO = "EvidenceBench/EvidenceBench-100k"

HEALTHFC_RAW_BASE = (
    "https://raw.githubusercontent.com/jvladika/HealthFC/main"
)

DATASET_CHOICES = (
    "all",
    "medfact-synth",
    "evidencebench-100k",
    "healthfc",
)

# Delete only artifacts created by the abandoned CIViC-Fact / CliniFact path.
# The V0 static fixtures, contracts, guidelines and frontend data are untouched.
OBSOLETE_PATHS = (
    "data/raw/civicfact",
    "data/raw/clinifact",
    "data/processed/civicfact",
    "data/processed/clinifact",
    "data/cache/europe_pmc",
    "data/processed/audit/strict_split_audit.json",
    "scripts/add_strict_splits.py",
    "scripts/build_civicfact_retrieval_corpus.py",
    "scripts/build_civicfact_retrieval_corpus_v1.py",
    "scripts/build_civicfact_retrieval_corpus_v2.py",
)

HEALTHFC_FILES = (
    "Datensatz.csv",
    "README.md",
    "LICENSE-CC-BY-NC-ND",
)


@dataclass(frozen=True)
class DownloadResult:
    name: str
    target: str
    revision: str
    files: int
    bytes: int
    status: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def human_bytes(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def directory_stats(path: Path) -> tuple[int, int]:
    files = 0
    total = 0
    if not path.exists():
        return files, total
    for item in path.rglob("*"):
        if item.is_file() and "/.cache/" not in item.as_posix():
            files += 1
            total += item.stat().st_size
    return files, total


def retry_call(
    label: str,
    operation: Callable[[], T],
    *,
    retries: int,
    base_delay: float,
) -> T:
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            return operation()
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            last_error = exc
            if attempt >= retries:
                break
            delay = min(90.0, base_delay * (2 ** (attempt - 1)))
            delay *= random.uniform(0.8, 1.2)
            print(
                f"[retry] {label} failed "
                f"({type(exc).__name__}: {exc}); "
                f"attempt {attempt}/{retries}, retrying in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    assert last_error is not None
    raise RuntimeError(
        f"{label} failed after {retries} attempts"
    ) from last_error


def configure_network(proxy: str, timeout: int) -> None:
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy

    # huggingface_hub reads these environment variables.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(timeout))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(timeout))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def append_gitignore(root: Path) -> None:
    gitignore = root / ".gitignore"
    current = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    entries = (
        "data/raw/v1/",
        "data/processed/v1/",
    )
    missing = [entry for entry in entries if entry not in current.splitlines()]
    if not missing:
        return

    with gitignore.open("a", encoding="utf-8") as handle:
        if current and not current.endswith("\n"):
            handle.write("\n")
        handle.write("\n# Downloaded V1 datasets\n")
        for entry in missing:
            handle.write(f"{entry}\n")


def remove_path(path: Path) -> str:
    if not path.exists() and not path.is_symlink():
        return "absent"

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    return "removed"


def cleanup_obsolete(root: Path) -> dict[str, str]:
    print("\n=== Cleaning obsolete V1 artifacts ===")
    results: dict[str, str] = {}
    for relative in OBSOLETE_PATHS:
        path = root / relative
        status = remove_path(path)
        results[relative] = status
        print(f"[{status}] {relative}")

    # Remove empty parent directories, but never remove data/, scripts/, or
    # any curated fixture / contract / guideline directory.
    for relative in (
        "data/processed/audit",
        "data/cache",
        "data/processed",
        "data/raw",
    ):
        path = root / relative
        try:
            path.rmdir()
        except OSError:
            pass

    return results


def hf_revision(
    repo_id: str,
    *,
    retries: int,
    base_delay: float,
) -> str:
    api = HfApi()
    info = retry_call(
        f"resolve revision for {repo_id}",
        lambda: api.dataset_info(repo_id=repo_id, revision="main"),
        retries=retries,
        base_delay=base_delay,
    )
    return info.sha


def download_hf_dataset(
    *,
    name: str,
    repo_id: str,
    target: Path,
    allow_patterns: list[str],
    workers: int,
    retries: int,
    base_delay: float,
) -> DownloadResult:
    target.mkdir(parents=True, exist_ok=True)
    revision = hf_revision(
        repo_id,
        retries=retries,
        base_delay=base_delay,
    )

    print(f"\n=== Downloading {name} ===")
    print(f"repo:     {repo_id}")
    print(f"revision: {revision}")
    print(f"target:   {target}")

    def do_download() -> str:
        return snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=target,
            allow_patterns=allow_patterns,
            max_workers=workers,
            token=os.environ.get("HF_TOKEN") or None,
        )

    retry_call(
        f"download {repo_id}",
        do_download,
        retries=retries,
        base_delay=base_delay,
    )

    files, total = directory_stats(target)
    if files == 0 or total == 0:
        raise RuntimeError(f"{name} download produced no files in {target}")

    print(f"[ok] {files} files, {human_bytes(total)}")
    return DownloadResult(
        name=name,
        target=str(target),
        revision=revision,
        files=files,
        bytes=total,
        status="ok",
    )


def make_http_session(proxy: str, retries: int) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=1.0,
        status_forcelist=(408, 425, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=4,
        pool_maxsize=4,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
    session.headers.update(
        {
            "User-Agent": "EvidenceGap-V1-Dataset-Bootstrap/1.0",
            "Accept": "*/*",
        }
    )
    return session


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_http_file(
    *,
    session: requests.Session,
    url: str,
    target: Path,
    timeout: int,
    retries: int,
    base_delay: float,
) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"[reuse] {target.name} ({human_bytes(target.stat().st_size)})")
        return

    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)

    def do_download() -> None:
        existing = part.stat().st_size if part.exists() else 0
        headers: dict[str, str] = {}
        if existing:
            headers["Range"] = f"bytes={existing}-"

        with session.get(
            url,
            headers=headers,
            stream=True,
            timeout=(timeout, timeout),
        ) as response:
            if response.status_code == 416 and existing:
                part.replace(target)
                return

            response.raise_for_status()
            append = existing > 0 and response.status_code == 206
            mode = "ab" if append else "wb"

            if existing and not append:
                existing = 0

            with part.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())

        if not part.exists() or part.stat().st_size == 0:
            raise RuntimeError(f"empty download: {url}")
        part.replace(target)

    retry_call(
        f"download {url}",
        do_download,
        retries=retries,
        base_delay=base_delay,
    )
    print(f"[ok] {target.name} ({human_bytes(target.stat().st_size)})")


def healthfc_revision(
    session: requests.Session,
    *,
    timeout: int,
    retries: int,
    base_delay: float,
) -> str:
    # Resolve the current main commit so the manifest records exactly what was
    # downloaded. The raw downloads still work if this metadata endpoint fails.
    url = "https://api.github.com/repos/jvladika/HealthFC/commits/main"

    def request_revision() -> str:
        response = session.get(url, timeout=(timeout, timeout))
        response.raise_for_status()
        value = response.json().get("sha")
        if not value:
            raise RuntimeError("GitHub response did not contain commit SHA")
        return str(value)

    try:
        return retry_call(
            "resolve HealthFC revision",
            request_revision,
            retries=retries,
            base_delay=base_delay,
        )
    except Exception as exc:
        print(
            f"[warning] could not resolve HealthFC revision: {exc}",
            file=sys.stderr,
        )
        return "main"


def download_healthfc(
    *,
    target: Path,
    proxy: str,
    timeout: int,
    retries: int,
    base_delay: float,
) -> DownloadResult:
    target.mkdir(parents=True, exist_ok=True)
    session = make_http_session(proxy, retries)
    revision = healthfc_revision(
        session,
        timeout=timeout,
        retries=retries,
        base_delay=base_delay,
    )

    print("\n=== Downloading HealthFC ===")
    print("repo:     jvladika/HealthFC")
    print(f"revision: {revision}")
    print(f"target:   {target}")

    for filename in HEALTHFC_FILES:
        url = f"{HEALTHFC_RAW_BASE}/{filename}"
        download_http_file(
            session=session,
            url=url,
            target=target / filename,
            timeout=timeout,
            retries=retries,
            base_delay=base_delay,
        )

    csv_path = target / "Datensatz.csv"
    validate_healthfc_csv(csv_path)

    files, total = directory_stats(target)
    checksums = {
        filename: sha256_file(target / filename)
        for filename in HEALTHFC_FILES
    }
    (target / "checksums.sha256.json").write_text(
        json.dumps(checksums, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    files, total = directory_stats(target)

    print(f"[ok] {files} files, {human_bytes(total)}")
    return DownloadResult(
        name="healthfc",
        target=str(target),
        revision=revision,
        files=files,
        bytes=total,
        status="ok",
    )


def validate_healthfc_csv(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty HealthFC CSV: {path}")

    # Only inspect the header; do not transform the licensed source file.
    encodings = ("utf-8-sig", "utf-8", "latin-1")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                sample = handle.read(16 * 1024)
                handle.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except csv.Error:
                    dialect = csv.excel
                reader = csv.reader(handle, dialect)
                header = next(reader)
            normalized = {column.strip().lower() for column in header}
            expected_any = {"en_claim", "de_claim", "verdict", "label"}
            if not normalized & expected_any:
                raise RuntimeError(
                    "HealthFC CSV header does not contain expected fields: "
                    f"{header[:20]}"
                )
            return
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"cannot validate HealthFC CSV: {last_error}")


def verify_medfact(path: Path) -> tuple[int, int]:
    shards = sorted((path / "data").glob("*.parquet"))
    if len(shards) < 17:
        raise RuntimeError(
            f"MedFact-Synth expected at least 17 parquet shards, found {len(shards)}"
        )
    if any(item.stat().st_size == 0 for item in shards):
        raise RuntimeError("MedFact-Synth contains an empty parquet shard")
    return directory_stats(path)


def verify_evidencebench(path: Path) -> tuple[int, int]:
    expected = (
        path / "evidencebench_100k_train_set.json",
        path / "evidencebench_100k_test_set.json",
    )
    missing = [str(item) for item in expected if not item.exists()]
    if missing:
        raise RuntimeError(
            "EvidenceBench-100k missing required files: " + ", ".join(missing)
        )
    if any(item.stat().st_size == 0 for item in expected):
        raise RuntimeError("EvidenceBench-100k contains an empty JSON file")
    return directory_stats(path)


def verify_healthfc(path: Path) -> tuple[int, int]:
    for filename in HEALTHFC_FILES:
        item = path / filename
        if not item.exists() or item.stat().st_size == 0:
            raise RuntimeError(f"HealthFC missing or empty: {item}")
    validate_healthfc_csv(path / "Datensatz.csv")
    return directory_stats(path)


def verify_selected(raw_root: Path, selected: set[str]) -> list[DownloadResult]:
    print("\n=== Verifying downloaded datasets ===")
    results: list[DownloadResult] = []

    if "medfact-synth" in selected:
        path = raw_root / "medfact_synth"
        files, total = verify_medfact(path)
        print(f"[ok] MedFact-Synth: {files} files, {human_bytes(total)}")
        results.append(
            DownloadResult(
                "medfact-synth", str(path), "unknown", files, total, "ok"
            )
        )

    if "evidencebench-100k" in selected:
        path = raw_root / "evidencebench_100k"
        files, total = verify_evidencebench(path)
        print(f"[ok] EvidenceBench-100k: {files} files, {human_bytes(total)}")
        results.append(
            DownloadResult(
                "evidencebench-100k",
                str(path),
                "unknown",
                files,
                total,
                "ok",
            )
        )

    if "healthfc" in selected:
        path = raw_root / "healthfc"
        files, total = verify_healthfc(path)
        print(f"[ok] HealthFC: {files} files, {human_bytes(total)}")
        results.append(
            DownloadResult(
                "healthfc", str(path), "unknown", files, total, "ok"
            )
        )

    return results


def selected_datasets(value: str) -> set[str]:
    if value == "all":
        return {
            "medfact-synth",
            "evidencebench-100k",
            "healthfc",
        }
    return {value}


def write_data_readme(raw_root: Path) -> None:
    content = """# EvidenceGap V1 raw datasets

This directory is generated by `scripts/bootstrap_v1_datasets.py`.

- `medfact_synth/`
  - Source: `ncbi/MedFact-Synth`
  - Role: large-scale claim–article stance and rationale data
- `evidencebench_100k/`
  - Source: `EvidenceBench/EvidenceBench-100k`
  - Role: hypothesis-to-evidence-sentence retrieval within biomedical papers
- `healthfc/`
  - Source: `jvladika/HealthFC`
  - Role: medical-expert annotated final evaluation set

Raw source files are not modified. Dataset licenses remain in their respective
directories. Generated processing outputs must go under `data/processed/v1/`.
"""
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "README.md").write_text(content, encoding="utf-8")


def write_manifest(
    raw_root: Path,
    *,
    selected: set[str],
    cleanup: dict[str, str],
    results: Iterable[DownloadResult],
    args: argparse.Namespace,
) -> None:
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "selected_datasets": sorted(selected),
        "network": {
            "proxy_configured": bool(args.proxy),
            "workers": args.workers,
            "retries": args.retries,
            "timeout_seconds": args.timeout,
        },
        "cleanup": cleanup,
        "downloads": [
            {
                "name": item.name,
                "target": item.target,
                "revision": item.revision,
                "files": item.files,
                "bytes": item.bytes,
                "status": item.status,
            }
            for item in results
        ],
        "sources": {
            "medfact-synth": {
                "repo_id": MEDFACT_REPO,
                "repo_type": "dataset",
            },
            "evidencebench-100k": {
                "repo_id": EVIDENCEBENCH_REPO,
                "repo_type": "dataset",
            },
            "healthfc": {
                "repository": "jvladika/HealthFC",
                "files": list(HEALTHFC_FILES),
            },
        },
    }
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "download_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean obsolete EvidenceGap V1 dataset artifacts and download "
            "MedFact-Synth, EvidenceBench-100k and HealthFC."
        )
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="all",
        help="Download/verify one dataset or all datasets.",
    )
    parser.add_argument(
        "--clean-old",
        action="store_true",
        help="Delete known obsolete CIViC-Fact, CliniFact and PMC artifacts.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download; verify existing dataset files only.",
    )
    parser.add_argument(
        "--proxy",
        default="",
        help="HTTP/HTTPS proxy, e.g. http://127.0.0.1:7899",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--base-delay", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    if not (root / ".gitignore").exists() and not (root / "data").exists():
        raise SystemExit(
            f"{root} does not look like the EvidenceGap repository root"
        )
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.retries < 1:
        raise SystemExit("--retries must be >= 1")
    if args.timeout < 1:
        raise SystemExit("--timeout must be >= 1")

    configure_network(args.proxy, args.timeout)
    append_gitignore(root)

    raw_root = root / "data/raw/v1"
    selected = selected_datasets(args.dataset)
    cleanup: dict[str, str] = {}

    if args.clean_old:
        cleanup = cleanup_obsolete(root)

    write_data_readme(raw_root)

    if args.verify_only:
        results = verify_selected(raw_root, selected)
        write_manifest(
            raw_root,
            selected=selected,
            cleanup=cleanup,
            results=results,
            args=args,
        )
        print("\nVerification complete.")
        return

    results: list[DownloadResult] = []

    if "medfact-synth" in selected:
        result = download_hf_dataset(
            name="medfact-synth",
            repo_id=MEDFACT_REPO,
            target=raw_root / "medfact_synth",
            allow_patterns=["data/*.parquet", "README.md"],
            workers=args.workers,
            retries=args.retries,
            base_delay=args.base_delay,
        )
        verify_medfact(raw_root / "medfact_synth")
        results.append(result)

    if "evidencebench-100k" in selected:
        result = download_hf_dataset(
            name="evidencebench-100k",
            repo_id=EVIDENCEBENCH_REPO,
            target=raw_root / "evidencebench_100k",
            allow_patterns=[
                "evidencebench_100k_train_set.json",
                "evidencebench_100k_test_set.json",
                "README.md",
            ],
            workers=args.workers,
            retries=args.retries,
            base_delay=args.base_delay,
        )
        verify_evidencebench(raw_root / "evidencebench_100k")
        results.append(result)

    if "healthfc" in selected:
        result = download_healthfc(
            target=raw_root / "healthfc",
            proxy=args.proxy,
            timeout=args.timeout,
            retries=args.retries,
            base_delay=args.base_delay,
        )
        verify_healthfc(raw_root / "healthfc")
        results.append(result)

    write_manifest(
        raw_root,
        selected=selected,
        cleanup=cleanup,
        results=results,
        args=args,
    )

    print("\n=== Complete ===")
    for item in results:
        print(
            f"{item.name}: {item.files} files, "
            f"{human_bytes(item.bytes)}, revision={item.revision}"
        )
    print(f"manifest: {raw_root / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
