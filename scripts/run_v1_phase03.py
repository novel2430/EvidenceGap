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
from evidencegap.dense import (  # noqa: E402
    build_dense_article_inputs,
    build_faiss_index,
    compare_retrieval_reports,
    encode_article_embeddings,
    encode_query_embeddings,
    query_dense_index,
    run_dense_article_retrieval,
    validate_article_embeddings,
    validate_dense_article_inputs,
    validate_faiss_index,
    validate_query_embeddings,
)

MODELS = ("medcpt", "bmretriever")


def add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=REPO_ROOT)


def path_or_none(value: str | None) -> Path | None:
    return Path(value) if value else None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="EvidenceGap V1 Phase 03 dense article retrieval"
    )
    sub = result.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-inputs")
    add_root(prepare)
    prepare.add_argument("--corpus-dir", type=Path)
    prepare.add_argument("--output-dir", type=Path)
    prepare.add_argument("--threads", type=int, default=16)
    prepare.add_argument("--memory-limit")
    prepare.add_argument("--force", action="store_true")

    articles = sub.add_parser("encode-articles")
    add_root(articles)
    articles.add_argument("--model", choices=MODELS, required=True)
    articles.add_argument(
        "--devices",
        default="0",
        help="Comma-separated visible CUDA indices, e.g. 0,1,2,3",
    )
    articles.add_argument("--num-shards", type=int)
    articles.add_argument("--input-dir", type=Path)
    articles.add_argument("--output-dir", type=Path)
    articles.add_argument("--batch-size", type=int)
    articles.add_argument("--amp", choices=("fp16", "fp32"), default="fp16")
    articles.add_argument("--force", action="store_true")

    queries = sub.add_parser("encode-queries")
    add_root(queries)
    queries.add_argument("--model", choices=MODELS, required=True)
    queries.add_argument("--split", choices=("dev", "test"), required=True)
    queries.add_argument("--device", default="cuda:0")
    queries.add_argument("--corpus-dir", type=Path)
    queries.add_argument("--output-dir", type=Path)
    queries.add_argument("--batch-size", type=int)
    queries.add_argument("--amp", choices=("fp16", "fp32"), default="fp16")
    queries.add_argument("--force", action="store_true")

    index = sub.add_parser("build-index")
    add_root(index)
    index.add_argument("--model", choices=MODELS, required=True)
    index.add_argument("--embedding-dir", type=Path)
    index.add_argument("--index-dir", type=Path)
    index.add_argument(
        "--index-type",
        choices=("flat", "ivf-flat", "ivf-sq-fp16"),
        default="ivf-sq-fp16",
    )
    index.add_argument("--nlist", type=int, default=4096)
    index.add_argument("--nprobe", type=int, default=64)
    index.add_argument("--train-size", type=int, default=200000)
    index.add_argument("--seed", type=int, default=20260721)
    index.add_argument("--add-batch-size", type=int, default=32768)
    index.add_argument("--threads", type=int, default=16)
    index.add_argument("--force", action="store_true")

    run = sub.add_parser("run")
    add_root(run)
    run.add_argument("--model", choices=MODELS, required=True)
    run.add_argument("--split", choices=("dev", "test"), required=True)
    run.add_argument("--corpus-dir", type=Path)
    run.add_argument("--embedding-dir", type=Path)
    run.add_argument("--query-dir", type=Path)
    run.add_argument("--index-dir", type=Path)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--report-dir", type=Path)
    run.add_argument("--top-k", type=int, default=100)
    run.add_argument("--nprobe", type=int)
    run.add_argument("--search-batch-size", type=int, default=256)
    run.add_argument("--max-queries", type=int)
    run.add_argument("--run-name")
    run.add_argument("--reuse-run", action="store_true")

    query = sub.add_parser("query")
    add_root(query)
    query.add_argument("--model", choices=MODELS, required=True)
    query.add_argument("--device", default="cuda:0")
    query.add_argument("--index-dir", type=Path)
    query.add_argument("--corpus-dir", type=Path)
    query.add_argument("--nprobe", type=int)
    query.add_argument("--top-k", type=int, default=10)
    query.add_argument("--amp", choices=("fp16", "fp32"), default="fp16")
    query.add_argument("claim")

    validate = sub.add_parser("validate")
    add_root(validate)
    validate.add_argument("--model", choices=MODELS, required=True)
    validate.add_argument("--split", choices=("dev", "test"))
    validate.add_argument("--corpus-dir", type=Path)
    validate.add_argument("--input-dir", type=Path)
    validate.add_argument("--embedding-dir", type=Path)
    validate.add_argument("--query-dir", type=Path)
    validate.add_argument("--index-dir", type=Path)

    compare = sub.add_parser("compare")
    add_root(compare)
    compare.add_argument("--split", choices=("dev", "test"), required=True)
    compare.add_argument("--report", type=Path, action="append", required=True)
    compare.add_argument("--output-stem")

    return result


def main() -> None:
    args = parser().parse_args()
    root = args.root.resolve()
    if args.command == "prepare-inputs":
        value = build_dense_article_inputs(
            root,
            corpus_dir=args.corpus_dir,
            output_dir=args.output_dir,
            threads=args.threads,
            memory_limit=args.memory_limit,
            force=args.force,
        )
    elif args.command == "encode-articles":
        devices = [f"cuda:{value.strip()}" for value in args.devices.split(",") if value.strip()]
        value = encode_article_embeddings(
            root,
            model_key=args.model,
            devices=devices,
            num_shards=args.num_shards,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            amp=args.amp,
            force=args.force,
        )
    elif args.command == "encode-queries":
        value = encode_query_embeddings(
            root,
            model_key=args.model,
            split=args.split,
            device=args.device,
            corpus_dir=args.corpus_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            amp=args.amp,
            force=args.force,
        )
    elif args.command == "build-index":
        value = build_faiss_index(
            root,
            model_key=args.model,
            embedding_dir=args.embedding_dir,
            index_dir=args.index_dir,
            index_type=args.index_type,
            nlist=args.nlist,
            nprobe=args.nprobe,
            train_size=args.train_size,
            seed=args.seed,
            add_batch_size=args.add_batch_size,
            threads=args.threads,
            force=args.force,
        )
    elif args.command == "run":
        value = run_dense_article_retrieval(
            root,
            model_key=args.model,
            split=args.split,
            corpus_dir=args.corpus_dir,
            embedding_dir=args.embedding_dir,
            query_dir=args.query_dir,
            index_dir=args.index_dir,
            run_dir=args.run_dir,
            report_dir=args.report_dir,
            top_k=args.top_k,
            nprobe=args.nprobe,
            search_batch_size=args.search_batch_size,
            max_queries=args.max_queries,
            run_name=args.run_name,
            reuse_run=args.reuse_run,
        )
    elif args.command == "query":
        value = query_dense_index(
            root,
            model_key=args.model,
            claim=args.claim,
            device=args.device,
            index_dir=args.index_dir,
            corpus_dir=args.corpus_dir,
            nprobe=args.nprobe,
            top_k=args.top_k,
            amp=args.amp,
        )
    elif args.command == "validate":
        value = {
            "article_inputs": validate_dense_article_inputs(
                root,
                corpus_dir=args.corpus_dir,
                input_dir=args.input_dir,
            ),
            "article_embeddings": validate_article_embeddings(
                root,
                model_key=args.model,
                embedding_dir=args.embedding_dir,
            ),
            "faiss_index": validate_faiss_index(
                root,
                model_key=args.model,
                embedding_dir=args.embedding_dir,
                index_dir=args.index_dir,
            ),
        }
        if args.split:
            value["query_embeddings"] = validate_query_embeddings(
                root,
                model_key=args.model,
                split=args.split,
                query_dir=args.query_dir,
            )
    elif args.command == "compare":
        value = compare_retrieval_reports(
            root,
            split=args.split,
            report_paths=args.report,
            output_stem=args.output_stem,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except EvidenceGapError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
