#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidencegap.common import EvidenceGapError  # noqa: E402
from evidencegap.corpus import build_medfact_article_corpus, validate_corpus  # noqa: E402
from evidencegap.evaluation import run_article_retrieval  # noqa: E402
from evidencegap.retrieval import (  # noqa: E402
    BM25SBackend,
    build_bm25s_index,
    validate_bm25s_index,
)


def add_common_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=REPO_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EvidenceGap V1 Phase 02 article retrieval pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("build-corpus", help="Build canonical article corpus")
    add_common_root(corpus)
    corpus.add_argument("--output-dir", type=Path)
    corpus.add_argument("--threads", type=int, default=8)
    corpus.add_argument("--memory-limit", help="DuckDB limit, e.g. 32GB")
    corpus.add_argument("--quick-rows", type=int)
    corpus.add_argument("--force", action="store_true")

    index = sub.add_parser("build-index", help="Build BM25S index")
    add_common_root(index)
    index.add_argument("--corpus-dir", type=Path)
    index.add_argument("--index-dir", type=Path)
    index.add_argument("--k1", type=float, default=1.2)
    index.add_argument("--b", type=float, default=0.75)
    index.add_argument("--method", default="lucene")
    index.add_argument("--backend", default="auto")
    index.add_argument("--csc-backend", default="auto")
    index.add_argument("--batch-size", type=int, default=8192)
    index.add_argument("--max-docs", type=int)
    index.add_argument("--force", action="store_true")

    run = sub.add_parser("run", help="Run judged and open-corpus evaluation")
    add_common_root(run)
    run.add_argument("--split", choices=("dev", "test"), required=True)
    run.add_argument("--corpus-dir", type=Path)
    run.add_argument("--index-dir", type=Path)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--report-dir", type=Path)
    run.add_argument("--top-k", type=int, default=100)
    run.add_argument("--max-queries", type=int)
    run.add_argument("--run-name", default="bm25s_default")

    validate = sub.add_parser("validate", help="Validate corpus and index")
    add_common_root(validate)
    validate.add_argument("--corpus-dir", type=Path)
    validate.add_argument("--index-dir", type=Path)

    query = sub.add_parser("query", help="Search one claim")
    add_common_root(query)
    query.add_argument("claim")
    query.add_argument("--index-dir", type=Path)
    query.add_argument("--top-k", type=int, default=10)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()

    if args.command == "build-corpus":
        result = build_medfact_article_corpus(
            root,
            output_dir=args.output_dir,
            force=args.force,
            threads=args.threads,
            memory_limit=args.memory_limit,
            quick_rows=args.quick_rows,
        )
        print(json.dumps(result["counts"], ensure_ascii=False, indent=2))
        return

    if args.command == "build-index":
        result = build_bm25s_index(
            root,
            corpus_dir=args.corpus_dir,
            index_dir=args.index_dir,
            force=args.force,
            k1=args.k1,
            b=args.b,
            method=args.method,
            backend=args.backend,
            csc_backend=args.csc_backend,
            batch_size=args.batch_size,
            max_docs=args.max_docs,
        )
        print(json.dumps(result["parameters"], ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        result = run_article_retrieval(
            root,
            split=args.split,
            corpus_dir=args.corpus_dir,
            index_dir=args.index_dir,
            run_dir=args.run_dir,
            report_dir=args.report_dir,
            top_k=args.top_k,
            max_queries=args.max_queries,
            run_name=args.run_name,
        )
        print(json.dumps(result["tracks"], ensure_ascii=False, indent=2))
        return

    if args.command == "validate":
        corpus = validate_corpus(root, output_dir=args.corpus_dir)
        index = validate_bm25s_index(
            root,
            corpus_dir=args.corpus_dir,
            index_dir=args.index_dir,
        )
        print(json.dumps({"corpus": corpus, "index": index}, indent=2))
        return

    if args.command == "query":
        index_dir = root / (args.index_dir or Path("artifacts/v1/bm25_index"))
        backend = BM25SBackend(index_dir, mmap=True)
        for hit in backend.search(args.claim, top_k=args.top_k):
            print(f"{hit.rank:>3}  {hit.score:>10.5f}  {hit.article_id}")
        return

    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        main()
    except EvidenceGapError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
