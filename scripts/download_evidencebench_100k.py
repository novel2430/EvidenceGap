#!/usr/bin/env python3
"""
Resumable direct downloader for EvidenceBench-100k.

This bypasses huggingface_hub snapshot/Xet downloads and fetches the two
published JSON files through normal HTTPS. Re-running the script resumes any
existing .part file using HTTP Range requests.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


REPO_ID = "EvidenceBench/EvidenceBench-100k"
REVISION = "9810647d26b22e39e638f822b402f5c2b1466d61"
FILES = (
    "evidencebench_100k_train_set.json",
    "evidencebench_100k_test_set.json",
    "README.md",
)


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
        pool_connections=2,
        pool_maxsize=2,
    )
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
    session.headers.update(
        {"User-Agent": "EvidenceGap-EvidenceBench-Downloader/1.0"}
    )
    return session


def last_non_whitespace_byte(path: Path) -> bytes:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        while position > 0:
            read_size = min(8192, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size).rstrip()
            if chunk:
                return chunk[-1:]
    return b""


def validate_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty file: {path}")

    if path.suffix != ".json":
        return

    with path.open("rb") as handle:
        first = handle.read(8192).lstrip()[:1]
    last = last_non_whitespace_byte(path)

    if first not in {b"[", b"{"}:
        raise RuntimeError(f"{path.name} does not start like JSON")
    expected_last = b"]" if first == b"[" else b"}"
    if last != expected_last:
        raise RuntimeError(
            f"{path.name} appears incomplete: expected final "
            f"{expected_last!r}, got {last!r}"
        )


def download_one(
    session: requests.Session,
    *,
    filename: str,
    target_dir: Path,
    timeout: int,
    attempts: int,
) -> None:
    target = target_dir / filename
    part = target_dir / f"{filename}.part"
    url = (
        f"https://huggingface.co/datasets/{REPO_ID}/resolve/"
        f"{REVISION}/{filename}?download=true"
    )

    if target.exists():
        try:
            validate_file(target)
            print(
                f"[reuse] {filename}: {human_bytes(target.stat().st_size)}",
                flush=True,
            )
            return
        except Exception:
            target.replace(part)

    for attempt in range(1, attempts + 1):
        existing = part.stat().st_size if part.exists() else 0
        headers: dict[str, str] = {}
        if existing:
            headers["Range"] = f"bytes={existing}-"

        try:
            print(
                f"[download] {filename}: resume at {human_bytes(existing)} "
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
                    validate_file(target)
                    return

                response.raise_for_status()

                append = existing > 0 and response.status_code == 206
                if existing > 0 and not append:
                    print(
                        "[warning] server ignored Range; restarting this file",
                        flush=True,
                    )

                mode = "ab" if append else "wb"
                written = existing if append else 0
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

            part.replace(target)
            validate_file(target)
            print(
                f"[ok] {filename}: {human_bytes(target.stat().st_size)}",
                flush=True,
            )
            return

        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if attempt >= attempts:
                raise RuntimeError(
                    f"{filename} failed after {attempts} attempts"
                ) from exc
            delay = min(90.0, 3.0 * (2 ** (attempt - 1)))
            delay *= random.uniform(0.8, 1.2)
            print(
                f"[retry] {type(exc).__name__}: {exc}; "
                f"sleeping {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--proxy",
        default="http://127.0.0.1:7899",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--attempts", type=int, default=20)
    args = parser.parse_args()

    root = args.root.resolve()
    target_dir = root / "data/raw/v1/evidencebench_100k"
    target_dir.mkdir(parents=True, exist_ok=True)

    session = make_session(args.proxy, retries=5)
    for filename in FILES:
        download_one(
            session,
            filename=filename,
            target_dir=target_dir,
            timeout=args.timeout,
            attempts=args.attempts,
        )

    manifest: dict[str, Any] = {
        "dataset": "evidencebench-100k",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            filename: (target_dir / filename).stat().st_size
            for filename in FILES
        },
    }
    (target_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nEvidenceBench-100k download complete.")
    print(target_dir)


if __name__ == "__main__":
    main()

