from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator


class EvidenceGapError(RuntimeError):
    """Base exception for deterministic EvidenceGap pipeline failures."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceGapError(f"Invalid JSON file {path}: {exc}") from exc


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def require_empty_or_force(path: Path, *, force: bool) -> None:
    if not path.exists():
        return
    if not force:
        raise EvidenceGapError(
            f"Output already exists: {path}. Use --force to rebuild it."
        )
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


@contextmanager
def atomic_directory(target: Path, *, force: bool) -> Iterator[Path]:
    """Build a directory beside its destination and rename only on success."""
    require_empty_or_force(target, force=force)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        yield staging
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def manifest_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


_WORKSPACE_ROOT: ContextVar[Path | None] = ContextVar(
    "evidencegap_backend_workspace_root", default=None
)


@contextmanager
def workspace_root_context(root: Path) -> Iterator[None]:
    """Bind the explicit backend workspace for nested artifact validators."""
    token = _WORKSPACE_ROOT.set(root.resolve())
    try:
        yield
    finally:
        _WORKSPACE_ROOT.reset(token)


def find_workspace_root(start: Path) -> Path:
    """Resolve artifact paths without relying on a repository or src/ marker."""
    configured = _WORKSPACE_ROOT.get()
    if configured is not None:
        return configured
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        artifacts = candidate / "artifacts"
        try:
            current.relative_to(artifacts)
        except ValueError:
            pass
        else:
            return candidate
        if (candidate / ".evidencegap-workspace").exists():
            return candidate
        if (candidate / "backend/pyproject.toml").exists():
            return candidate
    return current
