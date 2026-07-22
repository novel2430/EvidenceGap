from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from evidencegap.common import EvidenceGapError, atomic_write_json, relative_path, sha256_file
from evidencegap.sentence_retrieval.artifacts import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    RUN_SCHEMA_VERSION,
    safe_run_name,
    validate_ranking_rows,
    write_rows_atomic,
)
from evidencegap.sentence_retrieval.contracts import EvidenceQuery, SCHEMA_VERSION, TASK_ID
from evidencegap.sentence_retrieval.evaluation import evaluate_sentence_run
from evidencegap.sentence_retrieval.evidencebench import ensure_canonical


TOKEN_PATTERN = r"(?u)\b[\w]+(?:[-/][\w]+)*\b"
TOKENIZER_CONTRACT = {
    "unicode_normalization": "NFKC",
    "lowercase": True,
    "token_pattern": TOKEN_PATTERN,
    "stemming": False,
    "stopword_removal": False,
}

def normalize_for_search(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

_TOKEN_RE = re.compile(TOKEN_PATTERN)


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(normalize_for_search(text))]


class LocalBM25Pool:
    def __init__(self, sentences: Sequence[str], *, k1: float, b: float) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens = [_tokens(sentence) for sentence in sentences]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freq: Counter[str] = Counter()
        for frequencies in self.term_freqs:
            self.doc_freq.update(frequencies.keys())
        self.n_docs = len(sentences)

    def score(self, query: str) -> list[float]:
        query_terms = _tokens(query)
        if not query_terms or self.n_docs == 0:
            return [0.0] * self.n_docs
        scores: list[float] = []
        for frequencies, length in zip(self.term_freqs, self.doc_lengths):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = self.doc_freq[term]
                idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
                norm = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / self.avgdl
                ) if self.avgdl > 0 else frequency + self.k1
                score += idf * (frequency * (self.k1 + 1.0)) / norm
            scores.append(float(score))
        return scores


def required_depth(query: EvidenceQuery, requested_top_k: int) -> int:
    """Return a fixed, gold-independent output depth."""
    return min(len(query.candidate_sentences), max(5, requested_top_k))


def _rank_rows(
    query: EvidenceQuery,
    scores: Sequence[float],
    *,
    split: str,
    run_name: str,
    top_k: int,
) -> Iterable[dict[str, Any]]:
    ranking = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    depth = required_depth(query, top_k)
    for rank, index in enumerate(ranking[:depth], start=1):
        score = float(scores[index])
        yield {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "split": split,
            "run_name": run_name,
            "query_id": query.query_id,
            "paper_id": query.paper_id,
            "pool_fingerprint": query.pool_fingerprint,
            "sentence_index": index,
            "sentence_type": query.sentence_types[index],
            "sentence_text": query.candidate_sentences[index],
            "retrieval_model": "bm25",
            "retrieval_score": score,
            "retrieval_rank": rank,
            "cross_encoder_score": None,
            "final_score": score,
            "final_rank": rank,
        }


def run_bm25_sentence_retrieval(
    root: Path,
    *,
    split: str,
    max_queries: int | None = None,
    canonical_dir: Path | None = None,
    run_name: str | None = None,
    top_k: int = 20,
    k1: float = 1.5,
    b: float = 0.75,
    run_dir: Path | None = None,
    report_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if top_k <= 0 or k1 <= 0 or not 0 <= b <= 1:
        raise EvidenceGapError("top_k/k1/b are outside valid ranges")
    root = root.resolve()
    canonical_path, queries, canonical_manifest = ensure_canonical(
        root,
        split=split,
        max_queries=max_queries,
        canonical_dir=canonical_dir,
    )
    name = safe_run_name(run_name or f"bm25_{split}_{'full' if max_queries is None else max_queries}")
    base = run_dir.resolve() if run_dir else root / DEFAULT_ARTIFACT_ROOT / "runs" / name
    output_path = base / "ranked_sentences.parquet"
    manifest_path = base / "run_manifest.json"
    report_root = report_dir.resolve() if report_dir else root / DEFAULT_REPORT_ROOT
    report_path = report_root / f"evidence_sentence_retrieval_{name}_{split}.json"

    signature = {
        "schema_version": RUN_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "run_name": name,
        "split": split,
        "retrieval_model": "bm25",
        "canonical_sha256": canonical_manifest["canonical_sha256"],
        "top_k": top_k,
        "k1": k1,
        "b": b,
        "tokenizer_contract": TOKENIZER_CONTRACT,
    }
    if not force and output_path.exists() != manifest_path.exists():
        raise EvidenceGapError(
            f"Incomplete BM25 run artifact under {base}; use --force to rebuild"
        )
    if output_path.exists() and manifest_path.exists() and not force:
        existing = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        if {key: existing.get(key) for key in signature} != signature:
            raise EvidenceGapError(f"Existing BM25 run is stale: {base}; use --force")
        if existing.get("output_sha256") != sha256_file(output_path):
            raise EvidenceGapError(
                f"Existing BM25 output checksum mismatch: {output_path}; use --force"
            )
        validation = validate_ranking_rows(
            output_path,
            expected_queries={query.query_id: len(query.candidate_sentences) for query in queries},
            expected_depths={
                query.query_id: required_depth(query, top_k) for query in queries
            },
            expected_run_name=name,
        )
    else:
        if force and base.exists():
            import shutil
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=True)
        pool_cache: dict[str, LocalBM25Pool] = {}
        def rows() -> Iterable[dict[str, Any]]:
            for query in queries:
                pool = pool_cache.get(query.pool_fingerprint)
                if pool is None:
                    pool = LocalBM25Pool(query.candidate_sentences, k1=k1, b=b)
                    pool_cache[query.pool_fingerprint] = pool
                yield from _rank_rows(
                    query,
                    pool.score(query.hypothesis),
                    split=split,
                    run_name=name,
                    top_k=top_k,
                )
        write_rows_atomic(output_path, rows())
        validation = validate_ranking_rows(
            output_path,
            expected_queries={query.query_id: len(query.candidate_sentences) for query in queries},
            expected_depths={
                query.query_id: required_depth(query, top_k) for query in queries
            },
            expected_run_name=name,
        )
        manifest = {
            **signature,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "canonical_dir": relative_path(root, canonical_path),
            "output_path": relative_path(root, output_path),
            "output_sha256": sha256_file(output_path),
            "queries": len(queries),
            "unique_pools": len({query.pool_fingerprint for query in queries}),
            "rows": validation["rows"],
            "score_semantics": "higher_is_more_relevant",
            "depth_semantics": "max(requested_top_k, 5), capped by candidate count; gold optimal never affects retrieval",
        }
        atomic_write_json(manifest_path, manifest)

    evaluation = evaluate_sentence_run(
        root,
        canonical_dir=canonical_path,
        run_path=output_path,
        report_path=report_path,
    )
    return {
        "run_name": name,
        "run_path": relative_path(root, output_path),
        "manifest_path": relative_path(root, manifest_path),
        "report_path": relative_path(root, report_path),
        "validation": validation,
        "metrics": evaluation["metrics"],
    }
