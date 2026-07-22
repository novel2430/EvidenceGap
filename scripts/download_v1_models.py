#!/usr/bin/env python3
"""
Download the fixed EvidenceGap V1 retrieval/reranking model set.

Models:
- ncbi/MedCPT-Query-Encoder
- ncbi/MedCPT-Article-Encoder
- ncbi/MedCPT-Cross-Encoder
- BMRetriever/BMRetriever-410M

The downloader bypasses Hugging Face snapshot/Xet transfers:
- resolves the current repository commit through the public model API;
- downloads individual files over normal HTTPS;
- uses .part files and HTTP Range for resume;
- retries with exponential backoff;
- reuses validated completed files;
- writes models/v1/download_manifest.json.

No verifier model is included yet. The verifier is intentionally selected only
after auditing the actual label and text structure of the three V1 datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


MODELS = {
    "medcpt-query": "ncbi/MedCPT-Query-Encoder",
    "medcpt-article": "ncbi/MedCPT-Article-Encoder",
    "medcpt-cross": "ncbi/MedCPT-Cross-Encoder",
    "bmretriever-410m": "BMRetriever/BMRetriever-410M",
    "verifier-deberta-v3-base": "cross-encoder/nli-deberta-v3-base",
}

# MedCPT Cross Encoder's main branch currently exposes only pytorch_model.bin.
# Transformers refuses that pickle-based format with torch<2.6 after
# CVE-2025-32434. Hugging Face's verified conversion commit provides an
# equivalent safetensors file without requiring a PyTorch/CUDA upgrade.
SAFE_WEIGHT_OVERRIDES = {
    "medcpt-cross": {
        "revision": "75e855e5aaeda1e16da04a894207072d4b0db66a",
        "filename": "model.safetensors",
        "sha256": "b27e15c8bae944cb3cfd752e09669e447bd6282f787115ee485b484ef4657eb9",
    }
}

SMALL_MODEL_FILES = {
    "config.json",
    "generation_config.json",
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
    "modules.json",
    "sentence_bert_config.json",
}

EXCLUDED_SUFFIXES = (
    ".h5",
    ".msgpack",
    ".ot",
    ".onnx",
    ".ckpt",
)

EXCLUDED_NAMES = {
    ".gitattributes",
    "README.md",
    "README.MD",
}


@dataclass(frozen=True)
class ModelInfo:
    alias: str
    repo_id: str
    revision: str
    files: tuple[str, ...]


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def make_session(proxy: str, retries: int) -> requests.Session:
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
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
    session.headers.update(
        {"User-Agent": "EvidenceGap-V1-Model-Downloader/1.0"}
    )
    return session


def retry_call(label: str, attempts: int, operation):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            delay = min(90.0, 3.0 * (2 ** (attempt - 1)))
            delay *= random.uniform(0.8, 1.2)
            print(
                f"[retry] {label}: {type(exc).__name__}: {exc}; "
                f"sleeping {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"{label} failed after {attempts} attempts") from last_error


def resolve_model(
    session: requests.Session,
    *,
    alias: str,
    repo_id: str,
    timeout: int,
    attempts: int,
) -> ModelInfo:
    url = f"https://huggingface.co/api/models/{repo_id}"

    def request_info() -> dict[str, Any]:
        response = session.get(url, timeout=(timeout, timeout))
        response.raise_for_status()
        return response.json()

    payload = retry_call(
        f"resolve {repo_id}",
        attempts,
        request_info,
    )
    revision = str(payload.get("sha") or "main")
    siblings = [
        str(item.get("rfilename"))
        for item in payload.get("siblings", [])
        if item.get("rfilename")
    ]
    files = select_files(siblings)
    if alias in SAFE_WEIGHT_OVERRIDES:
        files = [name for name in files if not is_model_weight(name)]
    if not files:
        raise RuntimeError(f"No usable model files found for {repo_id}")
    return ModelInfo(alias, repo_id, revision, tuple(files))


def is_model_weight(name: str) -> bool:
    basename = Path(name).name
    return (
        name.endswith(".safetensors")
        or name.endswith(".safetensors.index.json")
        or basename == "pytorch_model.bin"
        or basename.startswith("pytorch_model-")
        or basename == "pytorch_model.bin.index.json"
    )


def select_files(siblings: list[str]) -> list[str]:
    names = set(siblings)

    safetensors = sorted(
        name
        for name in names
        if name.endswith(".safetensors")
        or name.endswith(".safetensors.index.json")
    )
    pytorch = sorted(
        name
        for name in names
        if name == "pytorch_model.bin"
        or name.startswith("pytorch_model-")
        or name == "pytorch_model.bin.index.json"
    )
    weights = safetensors if safetensors else pytorch

    selected: set[str] = set(weights)
    for name in names:
        basename = Path(name).name
        if basename in SMALL_MODEL_FILES:
            selected.add(name)
        elif name.endswith(".py"):
            selected.add(name)

    # Include any config JSON adjacent to sharded weights.
    for name in names:
        if name.endswith(".index.json") and (
            "model" in name or "pytorch" in name
        ):
            selected.add(name)

    selected = {
        name
        for name in selected
        if Path(name).name not in EXCLUDED_NAMES
        and not name.endswith(EXCLUDED_SUFFIXES)
    }
    return sorted(selected)


def remote_size(
    session: requests.Session,
    *,
    url: str,
    timeout: int,
) -> int | None:
    try:
        response = session.head(
            url,
            timeout=(timeout, timeout),
            allow_redirects=True,
        )
        response.raise_for_status()
        value = response.headers.get("Content-Length")
        return int(value) if value and value.isdigit() else None
    except Exception:
        return None


def download_file(
    session: requests.Session,
    *,
    repo_id: str,
    revision: str,
    filename: str,
    target: Path,
    timeout: int,
    attempts: int,
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    encoded_path = "/".join(quote(part, safe="") for part in filename.split("/"))
    url = (
        f"https://huggingface.co/{repo_id}/resolve/"
        f"{revision}/{encoded_path}?download=true"
    )
    expected = remote_size(session, url=url, timeout=timeout)

    if target.exists() and target.stat().st_size > 0:
        if expected is None or target.stat().st_size == expected:
            print(
                f"[reuse] {repo_id}/{filename}: "
                f"{human_bytes(target.stat().st_size)}",
                flush=True,
            )
            return {
                "file": filename,
                "bytes": target.stat().st_size,
                "status": "reused",
            }
        target.replace(part)

    for attempt in range(1, attempts + 1):
        existing = part.stat().st_size if part.exists() else 0
        headers: dict[str, str] = {}
        if existing:
            headers["Range"] = f"bytes={existing}-"

        try:
            print(
                f"[download] {repo_id}/{filename}: "
                f"resume at {human_bytes(existing)} "
                f"(attempt {attempt}/{attempts})",
                flush=True,
            )
            with session.get(
                url,
                headers=headers,
                stream=True,
                timeout=(timeout, timeout),
                allow_redirects=True,
            ) as response:
                if response.status_code == 416 and existing:
                    part.replace(target)
                    break

                response.raise_for_status()
                append = existing > 0 and response.status_code == 206
                mode = "ab" if append else "wb"
                written = existing if append else 0

                if existing and not append:
                    print(
                        "  server ignored Range; restarting this file",
                        flush=True,
                    )

                next_report = written + 256 * 1024 * 1024
                with part.open(mode) as handle:
                    for chunk in response.iter_content(
                        chunk_size=8 * 1024 * 1024
                    ):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
                        if written >= next_report:
                            print(
                                f"  {filename}: {human_bytes(written)}",
                                flush=True,
                            )
                            next_report = written + 256 * 1024 * 1024
                    handle.flush()
                    os.fsync(handle.fileno())

            if part.exists():
                part.replace(target)
            break
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if attempt >= attempts:
                raise RuntimeError(
                    f"Failed to download {repo_id}/{filename}"
                ) from exc
            delay = min(90.0, 3.0 * (2 ** (attempt - 1)))
            delay *= random.uniform(0.8, 1.2)
            print(
                f"[retry] {type(exc).__name__}: {exc}; "
                f"sleeping {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)

    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty downloaded file: {target}")
    if expected is not None and target.stat().st_size != expected:
        raise RuntimeError(
            f"Size mismatch for {target}: "
            f"expected {expected}, got {target.stat().st_size}"
        )

    print(
        f"[ok] {repo_id}/{filename}: "
        f"{human_bytes(target.stat().st_size)}",
        flush=True,
    )
    return {
        "file": filename,
        "bytes": target.stat().st_size,
        "status": "downloaded",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )


def validate_model_dir(path: Path) -> None:
    files = [item for item in path.rglob("*") if item.is_file()]
    names = {item.name for item in files}

    if "config.json" not in names:
        raise RuntimeError(f"{path} lacks config.json")
    if not (
        "tokenizer.json" in names
        or "vocab.txt" in names
        or "spiece.model" in names
        or "sentencepiece.bpe.model" in names
        or "tokenizer.model" in names
    ):
        raise RuntimeError(f"{path} lacks tokenizer vocabulary")
    if not any(
        item.name.endswith(".safetensors")
        or item.name == "pytorch_model.bin"
        or item.name.startswith("pytorch_model-")
        for item in files
    ):
        raise RuntimeError(f"{path} lacks model weights")


def append_gitignore(root: Path) -> None:
    path = root / ".gitignore"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    entry = "models/v1/"
    if entry in text.splitlines():
        return
    with path.open("a", encoding="utf-8") as handle:
        if text and not text.endswith("\n"):
            handle.write("\n")
        handle.write("\n# Downloaded EvidenceGap V1 models\n")
        handle.write(f"{entry}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--model",
        choices=("all", *MODELS.keys()),
        default="all",
    )
    parser.add_argument(
        "--proxy",
        default="http://127.0.0.1:7899",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--attempts", type=int, default=20)
    args = parser.parse_args()

    root = args.root.resolve()
    output_root = root / "models/v1"
    output_root.mkdir(parents=True, exist_ok=True)
    append_gitignore(root)

    session = make_session(args.proxy, retries=5)
    aliases = list(MODELS) if args.model == "all" else [args.model]

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }

    for alias in aliases:
        repo_id = MODELS[alias]
        info = resolve_model(
            session,
            alias=alias,
            repo_id=repo_id,
            timeout=args.timeout,
            attempts=args.attempts,
        )
        target_dir = output_root / alias
        print("\n===", alias, "===")
        print("repo:", info.repo_id)
        print("revision:", info.revision)
        print("files:", len(info.files))

        downloaded = []
        for filename in info.files:
            downloaded.append(
                download_file(
                    session,
                    repo_id=info.repo_id,
                    revision=info.revision,
                    filename=filename,
                    target=target_dir / filename,
                    timeout=args.timeout,
                    attempts=args.attempts,
                )
            )

        safe_weight = SAFE_WEIGHT_OVERRIDES.get(alias)
        if safe_weight is not None:
            safe_target = target_dir / safe_weight["filename"]
            result = download_file(
                session,
                repo_id=info.repo_id,
                revision=safe_weight["revision"],
                filename=safe_weight["filename"],
                target=safe_target,
                timeout=args.timeout,
                attempts=args.attempts,
            )
            validate_sha256(safe_target, safe_weight["sha256"])
            result["revision"] = safe_weight["revision"]
            result["sha256"] = safe_weight["sha256"]
            downloaded.append(result)

        validate_model_dir(target_dir)
        total = sum(
            item.stat().st_size
            for item in target_dir.rglob("*")
            if item.is_file()
        )
        manifest["models"][alias] = {
            "repo_id": info.repo_id,
            "revision": info.revision,
            "target": str(target_dir),
            "bytes": total,
            "files": downloaded,
            "safe_weight_override": safe_weight,
            "status": "ok",
        }
        print(f"[complete] {alias}: {human_bytes(total)}")

    (output_root / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nAll selected V1 models are complete.")
    print(output_root / "download_manifest.json")


if __name__ == "__main__":
    main()
