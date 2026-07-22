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
from evidencegap.reranking import (  # noqa: E402
    parse_source_run,
    parse_weight,
    run_cross_encoder_reranking,
    run_fusion,
)


def add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=REPO_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EvidenceGap V1 Phase 04 hybrid fusion and reranking"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fuse = sub.add_parser(
        "fuse",
        help="Fuse existing BM25/dense TREC runs without rerunning retrieval",
    )
    add_root(fuse)
    fuse.add_argument("--split", choices=("dev", "test"), required=True)
    fuse.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="ALIAS=RUN_NAME",
        help="Repeat for each source run, e.g. bm25=bm25s_default",
    )
    fuse.add_argument("--method", choices=("union", "rrf"), default="rrf")
    fuse.add_argument("--rrf-k", type=int, default=60)
    fuse.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="ALIAS=FLOAT",
        help="Optional weighted RRF source weight; defaults to 1.0",
    )
    fuse.add_argument("--input-run-dir", type=Path)
    fuse.add_argument("--corpus-dir", type=Path)
    fuse.add_argument("--candidate-dir", type=Path)
    fuse.add_argument("--run-dir", type=Path)
    fuse.add_argument("--report-dir", type=Path)
    fuse.add_argument("--run-name")
    fuse.add_argument("--top-k", type=int, default=100)
    fuse.add_argument("--max-queries", type=int)
    fuse.add_argument("--force", action="store_true")

    rerank = sub.add_parser(
        "rerank",
        help="Rerank a fused candidate parquet with the MedCPT cross encoder",
    )
    add_root(rerank)
    rerank.add_argument("--split", choices=("dev", "test"), required=True)
    rerank.add_argument("--candidate-path", type=Path, required=True)
    rerank.add_argument("--candidate-run-name", required=True)
    rerank.add_argument("--run-name")
    rerank.add_argument("--corpus-dir", type=Path)
    rerank.add_argument("--article-input-dir", type=Path)
    rerank.add_argument("--model-dir", type=Path)
    rerank.add_argument("--score-root", type=Path)
    rerank.add_argument("--reranked-candidate-dir", type=Path)
    rerank.add_argument("--run-dir", type=Path)
    rerank.add_argument("--report-dir", type=Path)
    rerank.add_argument(
        "--devices",
        default="0",
        help="Comma-separated visible device indices, e.g. 0,1,2,3; use cpu for CPU",
    )
    rerank.add_argument("--num-shards", type=int)
    rerank.add_argument("--batch-size", type=int, default=16)
    rerank.add_argument("--max-length", type=int, default=512)
    rerank.add_argument("--amp", choices=("fp16", "fp32"), default="fp16")
    rerank.add_argument("--top-k", type=int, default=100)
    rerank.add_argument(
        "--rerank-depth",
        type=int,
        help=(
            "Score only the fused open-corpus candidates up to this fusion rank; "
            "defaults to --top-k. Set equal to --top-k to preserve the retriever "
            "Top-K candidate set exactly."
        ),
    )
    rerank.add_argument("--max-queries", type=int)
    rerank.add_argument("--force", action="store_true")

    compare = sub.add_parser(
        "compare", help="Compare Phase 02/03/04 JSON reports on the independent track"
    )
    add_root(compare)
    compare.add_argument("--split", choices=("dev", "test"), required=True)
    compare.add_argument("--report", type=Path, action="append", required=True)
    compare.add_argument("--output-stem")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()

    if args.command == "fuse":
        sources = [parse_source_run(value) for value in args.source]
        weights = dict(parse_weight(value) for value in args.weight)
        result = run_fusion(
            root,
            split=args.split,
            sources=sources,
            method=args.method,
            rrf_k=args.rrf_k,
            weights=weights,
            input_run_dir=args.input_run_dir,
            corpus_dir=args.corpus_dir,
            candidate_dir=args.candidate_dir,
            run_dir=args.run_dir,
            report_dir=args.report_dir,
            run_name=args.run_name,
            top_k=args.top_k,
            max_queries=args.max_queries,
            force=args.force,
        )
    elif args.command == "rerank":
        devices = [value.strip() for value in args.devices.split(",") if value.strip()]
        result = run_cross_encoder_reranking(
            root,
            split=args.split,
            candidate_path=args.candidate_path,
            candidate_run_name=args.candidate_run_name,
            run_name=args.run_name,
            corpus_dir=args.corpus_dir,
            article_input_dir=args.article_input_dir,
            model_dir=args.model_dir,
            score_root=args.score_root,
            reranked_candidate_dir=args.reranked_candidate_dir,
            run_dir=args.run_dir,
            report_dir=args.report_dir,
            devices=devices,
            num_shards=args.num_shards,
            batch_size=args.batch_size,
            max_length=args.max_length,
            amp=args.amp,
            top_k=args.top_k,
            rerank_depth=args.rerank_depth,
            max_queries=args.max_queries,
            force=args.force,
        )
    elif args.command == "compare":
        from evidencegap.dense.evaluation import compare_retrieval_reports

        result = compare_retrieval_reports(
            root,
            split=args.split,
            report_paths=args.report,
            output_stem=args.output_stem,
        )
    else:
        raise AssertionError(args.command)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except EvidenceGapError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
