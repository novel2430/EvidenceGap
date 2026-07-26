from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from evidencegap_backend.common import (
    EvidenceGapError,
    manifest_fingerprint,
    relative_path,
)

DEFAULT_MODEL_DIR = Path("models/v1/medcpt-cross")


def _normalize_devices(values: Sequence[str]) -> list[str]:
    devices: list[str] = []
    for raw in values:
        raw = raw.strip()
        if not raw:
            continue
        if raw == "cpu" or raw.startswith("cuda:"):
            device = raw
        elif raw.isdigit():
            device = f"cuda:{raw}"
        else:
            raise EvidenceGapError(
                f"Invalid device {raw!r}; use 0,1,..., cuda:N, or cpu"
            )
        devices.append(device)
    if not devices:
        raise EvidenceGapError("At least one device is required")
    if len(set(devices)) != len(devices):
        raise EvidenceGapError("Devices must be unique")
    return devices


def _model_files(model_dir: Path) -> list[Path]:
    names = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
    }
    files = [path for path in model_dir.iterdir() if path.name in names]
    files.extend(sorted(model_dir.glob("*.safetensors")))
    files.extend(sorted(model_dir.glob("pytorch_model*.bin")))
    files.extend(sorted(model_dir.glob("*.index.json")))
    unique = sorted({path.resolve() for path in files if path.is_file()})
    if not unique:
        raise EvidenceGapError(f"No model files found in {model_dir}")
    return unique


def _model_fingerprint(model_dir: Path) -> str:
    return manifest_fingerprint(_model_files(model_dir))


def _article_text(title: str, abstract: str) -> str:
    title = title.strip()
    abstract = abstract.strip()
    if not title:
        return abstract
    if not abstract:
        return title
    separator = " " if title.endswith((".", "!", "?", ":", ";")) else ". "
    return title + separator + abstract


class CrossEncoderScorer:
    """Long-lived MedCPT cross-encoder used by the service runtime."""

    def __init__(
        self,
        root: Path,
        *,
        model_dir: Path | None = None,
        device: str = "cuda:0",
        amp: str = "fp16",
        max_length: int = 512,
    ) -> None:
        if max_length <= 0:
            raise EvidenceGapError("max_length must be positive")
        if amp not in {"fp16", "fp32"}:
            raise EvidenceGapError("amp must be fp16 or fp32")
        self.root = root.resolve()
        self.device = _normalize_devices([device])[0]
        self.amp = amp
        self.max_length = max_length
        self.model_dir = (
            model_dir.resolve()
            if model_dir is not None
            else (self.root / DEFAULT_MODEL_DIR).resolve()
        )
        if not self.model_dir.exists():
            raise EvidenceGapError(f"Missing MedCPT cross encoder: {self.model_dir}")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise EvidenceGapError(
                "Missing torch/transformers dependency for cross-encoder reranking"
            ) from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise EvidenceGapError(
                f"CUDA device requested but CUDA is unavailable: {self.device}"
            )
        if self.device == "cpu" and amp == "fp16":
            raise EvidenceGapError("fp16 cross-encoder inference is not supported on CPU")
        safe_weights = tuple(self.model_dir.glob("*.safetensors"))
        if not safe_weights:
            raise EvidenceGapError(
                "MedCPT cross encoder requires model.safetensors in "
                f"{self.model_dir}. The original pytorch_model.bin cannot be loaded "
                "with torch<2.6 after CVE-2025-32434. Re-run: "
                "python scripts/download_v1_models.py --root . "
                "--model medcpt-cross"
            )
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True, use_fast=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, local_files_only=True, use_safetensors=True
        )
        if int(getattr(self.model.config, "num_labels", 0)) != 1:
            raise EvidenceGapError(
                "MedCPT cross encoder must expose one relevance logit; "
                f"model reports num_labels={self.model.config.num_labels}"
            )
        self.model.eval().to(self.device)
        if amp == "fp16":
            self.model.half()
        self.model_fingerprint = _model_fingerprint(self.model_dir)
        self.score_calls = 0

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def score(
        self,
        *,
        claim_text: str,
        articles: Sequence[Mapping[str, Any]],
        batch_size: int = 16,
    ) -> dict[str, Any]:
        if not self.loaded:
            raise EvidenceGapError("Cross-encoder scorer is closed")
        if not claim_text.strip():
            raise EvidenceGapError("claim_text cannot be empty")
        if batch_size <= 0:
            raise EvidenceGapError("batch_size must be positive")
        normalized_articles: list[dict[str, str]] = []
        seen: set[str] = set()
        for index, article in enumerate(articles):
            article_id = str(article.get("article_id", "")).strip()
            if not article_id:
                raise EvidenceGapError(f"Runtime article {index} has no article_id")
            if article_id in seen:
                raise EvidenceGapError(f"Duplicate runtime article_id: {article_id}")
            seen.add(article_id)
            title = str(article.get("title") or "")
            abstract = str(article.get("abstract") or article.get("text") or "")
            if not title.strip() and not abstract.strip():
                raise EvidenceGapError(f"{article_id}: empty title and abstract")
            normalized_articles.append(
                {"article_id": article_id, "title": title, "abstract": abstract}
            )

        started = time.perf_counter()
        output: list[dict[str, Any]] = []
        for start in range(0, len(normalized_articles), batch_size):
            batch = normalized_articles[start : start + batch_size]
            claims = [claim_text] * len(batch)
            article_texts = [
                _article_text(row["title"], row["abstract"]) for row in batch
            ]
            encoded = self.tokenizer(
                claims,
                article_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise EvidenceGapError(
                    f"Unexpected cross-encoder logits shape: {tuple(logits.shape)}"
                )
            values = logits[:, 0].float().cpu().tolist()
            if any(not math.isfinite(float(value)) for value in values):
                raise EvidenceGapError("Cross encoder produced a non-finite score")
            output.extend(
                {
                    "article_id": row["article_id"],
                    "cross_encoder_score": float(score),
                }
                for row, score in zip(batch, values, strict=True)
            )
        self.score_calls += 1
        elapsed = time.perf_counter() - started
        return {
            "scores": output,
            "metadata": {
                "model_path": relative_path(self.root, self.model_dir),
                "model_fingerprint": self.model_fingerprint,
                "device": self.device,
                "batch_size": batch_size,
                "max_length": self.max_length,
                "amp": self.amp,
                "score_semantics": "raw_single_logit_higher_is_more_relevant",
                "pairs": len(output),
                "seconds": round(elapsed, 6),
                "pairs_per_second": (
                    round(len(output) / elapsed, 6) if elapsed > 0 else None
                ),
            },
        }

    def close(self) -> None:
        if self.model is None and self.tokenizer is None:
            return
        model = self.model
        self.model = None
        self.tokenizer = None
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
        import gc

        gc.collect()
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def score_runtime_article_pairs(
    root: Path,
    *,
    claim_text: str,
    articles: Sequence[Mapping[str, Any]],
    model_dir: Path | None = None,
    device: str = "cuda:0",
    batch_size: int = 16,
    max_length: int = 512,
    amp: str = "fp16",
    scorer: CrossEncoderScorer | None = None,
) -> dict[str, Any]:
    """Score runtime pairs, optionally using an already-loaded scorer."""
    if scorer is not None:
        return scorer.score(
            claim_text=claim_text,
            articles=articles,
            batch_size=batch_size,
        )
    owned = CrossEncoderScorer(
        root,
        model_dir=model_dir,
        device=device,
        amp=amp,
        max_length=max_length,
    )
    try:
        return owned.score(
            claim_text=claim_text,
            articles=articles,
            batch_size=batch_size,
        )
    finally:
        owned.close()
