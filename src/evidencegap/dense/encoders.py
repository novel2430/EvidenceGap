from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evidencegap.common import EvidenceGapError, sha256_file

BMR_TASK = (
    "Given a scientific claim, retrieve documents that support or refute the claim"
)


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    query_model: Path
    article_model: Path
    dimension: int
    query_max_length: int
    article_max_length: int
    default_article_batch_size: int
    default_query_batch_size: int
    pooling: str
    similarity: str = "inner_product"
    normalize: bool = False


def encoder_spec(root: Path, model_key: str) -> EncoderSpec:
    root = root.resolve()
    if model_key == "medcpt":
        return EncoderSpec(
            key=model_key,
            query_model=root / "models/v1/medcpt-query",
            article_model=root / "models/v1/medcpt-article",
            dimension=768,
            query_max_length=64,
            article_max_length=512,
            default_article_batch_size=64,
            default_query_batch_size=256,
            pooling="cls",
        )
    if model_key == "bmretriever":
        model = root / "models/v1/bmretriever-410m"
        return EncoderSpec(
            key=model_key,
            query_model=model,
            article_model=model,
            dimension=1024,
            query_max_length=512,
            article_max_length=512,
            default_article_batch_size=8,
            default_query_batch_size=32,
            pooling="last_token_with_eos",
        )
    raise EvidenceGapError(f"Unsupported dense model: {model_key}")


def model_fingerprint(spec: EncoderSpec, *, article: bool) -> str:
    model_dir = spec.article_model if article else spec.query_model
    if not model_dir.exists():
        raise EvidenceGapError(f"Missing local model directory: {model_dir}")
    candidates: list[Path] = []
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "merges.txt",
        "tokenizer.model",
        "spm.model",
    ):
        path = model_dir / name
        if path.exists():
            candidates.append(path)
    candidates.extend(sorted(model_dir.glob("*.safetensors")))
    candidates.extend(sorted(model_dir.glob("pytorch_model*.bin")))
    if not candidates:
        raise EvidenceGapError(f"No model files found under {model_dir}")
    digest = hashlib.sha256()
    for path in sorted(set(candidates)):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _imports() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing torch/transformers dependencies. "
            "Install requirements/v1-phase03.txt"
        ) from exc
    return torch, AutoModel, AutoTokenizer


class DenseEncoder:
    def __init__(
        self,
        root: Path,
        model_key: str,
        *,
        device: str,
        amp: str = "fp16",
    ) -> None:
        self.root = root.resolve()
        self.spec = encoder_spec(self.root, model_key)
        self.device = device
        self.amp = amp
        if amp not in {"fp16", "fp32"}:
            raise EvidenceGapError("amp must be fp16 or fp32")
        torch, AutoModel, AutoTokenizer = _imports()
        self.torch = torch
        self.AutoModel = AutoModel
        self.AutoTokenizer = AutoTokenizer
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise EvidenceGapError("CUDA device requested but torch CUDA is unavailable")
        self._query_bundle: tuple[Any, Any] | None = None
        self._article_bundle: tuple[Any, Any] | None = None

    def _load(self, model_dir: Path, *, article: bool) -> tuple[Any, Any]:
        bundle = self._article_bundle if article else self._query_bundle
        if bundle is not None:
            return bundle
        # BMRetriever uses the same checkpoint for query and passage encoding.
        # Reuse one in-memory model copy while retaining the two input formats;
        # otherwise a single 11 GB GPU unnecessarily holds the 410M model twice.
        other_bundle = self._query_bundle if article else self._article_bundle
        other_model_dir = self.spec.query_model if article else self.spec.article_model
        if other_bundle is not None and model_dir.resolve() == other_model_dir.resolve():
            if article:
                self._article_bundle = other_bundle
            else:
                self._query_bundle = other_bundle
            return other_bundle
        tokenizer = self.AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = self.AutoModel.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        model.to(self.device).eval()
        bundle = (tokenizer, model)
        if article:
            self._article_bundle = bundle
        else:
            self._query_bundle = bundle
        if self.spec.query_model.resolve() == self.spec.article_model.resolve():
            self._query_bundle = bundle
            self._article_bundle = bundle
        return bundle

    def _autocast(self):
        if self.amp == "fp16" and self.device.startswith("cuda"):
            return self.torch.autocast(device_type="cuda", dtype=self.torch.float16)
        from contextlib import nullcontext

        return nullcontext()

    def encode_queries(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.spec.dimension), dtype=np.float32)
        batch_size = batch_size or self.spec.default_query_batch_size
        tokenizer, model = self._load(self.spec.query_model, article=False)
        outputs: list[np.ndarray] = []
        try:
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                if self.spec.key == "medcpt":
                    encoded = tokenizer(
                        batch,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                        max_length=self.spec.query_max_length,
                    ).to(self.device)
                    with self.torch.inference_mode(), self._autocast():
                        hidden = model(**encoded).last_hidden_state[:, 0, :]
                else:
                    prompted = [f"{BMR_TASK}\nQuery: {text}" for text in batch]
                    encoded = _bmretriever_tokenize(
                        tokenizer,
                        prompted,
                        max_length=self.spec.query_max_length,
                        device=self.device,
                    )
                    with self.torch.inference_mode(), self._autocast():
                        result = model(**encoded)
                        hidden = _last_token_pool(
                            result.last_hidden_state,
                            encoded["attention_mask"],
                        )
                outputs.append(hidden.float().cpu().numpy())
        finally:
            pass
        matrix = np.concatenate(outputs, axis=0)
        if matrix.shape[1] != self.spec.dimension:
            raise EvidenceGapError(
                f"Unexpected embedding dimension: {matrix.shape[1]} "
                f"!= {self.spec.dimension}"
            )
        return matrix

    def encode_articles(
        self,
        records: Sequence[tuple[str, str]],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        if not records:
            return np.empty((0, self.spec.dimension), dtype=np.float32)
        batch_size = batch_size or self.spec.default_article_batch_size
        tokenizer, model = self._load(self.spec.article_model, article=True)
        outputs: list[np.ndarray] = []
        try:
            for start in range(0, len(records), batch_size):
                batch = list(records[start : start + batch_size])
                if self.spec.key == "medcpt":
                    pairs = [[title, abstract] for title, abstract in batch]
                    encoded = tokenizer(
                        pairs,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                        max_length=self.spec.article_max_length,
                    ).to(self.device)
                    with self.torch.inference_mode(), self._autocast():
                        hidden = model(**encoded).last_hidden_state[:, 0, :]
                else:
                    passages = []
                    for title, abstract in batch:
                        passage = "\n".join(
                            value for value in (title.strip(), abstract.strip()) if value
                        )
                        passages.append(
                            f"Represent this passage\npassage: {passage}"
                        )
                    encoded = _bmretriever_tokenize(
                        tokenizer,
                        passages,
                        max_length=self.spec.article_max_length,
                        device=self.device,
                    )
                    with self.torch.inference_mode(), self._autocast():
                        result = model(**encoded)
                        hidden = _last_token_pool(
                            result.last_hidden_state,
                            encoded["attention_mask"],
                        )
                outputs.append(hidden.float().cpu().numpy())
        finally:
            pass
        matrix = np.concatenate(outputs, axis=0)
        if matrix.shape[1] != self.spec.dimension:
            raise EvidenceGapError(
                f"Unexpected embedding dimension: {matrix.shape[1]} "
                f"!= {self.spec.dimension}"
            )
        return matrix


def _last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden.shape[0]
    return last_hidden[
        last_hidden.new_tensor(list(range(batch_size)), dtype=sequence_lengths.dtype),
        sequence_lengths,
    ]


def _bmretriever_tokenize(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
    device: str,
) -> Any:
    if tokenizer.eos_token_id is None:
        raise EvidenceGapError("BMRetriever tokenizer has no EOS token")
    encoded = tokenizer(
        list(texts),
        max_length=max_length - 1,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )
    input_ids = [ids + [tokenizer.eos_token_id] for ids in encoded["input_ids"]]
    attention_mask = [mask + [1] for mask in encoded["attention_mask"]]
    return tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    ).to(device)
