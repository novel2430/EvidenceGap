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
from evidencegap.sentence_retrieval import (  # noqa: E402
    analyze_sentence_run_complementarity,
    audit_evidencebench,
    compare_sentence_runs_paired,
    diagnose_sentence_run,
    evaluate_sentence_run,
    load_canonical_queries,
    prepare_evidencebench_canonical,
    run_bm25_sentence_retrieval,
    run_cross_encoder_sentence_reranking,
    run_dense_sentence_retrieval,
    run_sentence_rrf_fusion,
    validate_ranking_rows,
)

SPLITS = ("train", "dev", "test")
DENSE_MODELS = ("medcpt", "bmretriever")


def add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=REPO_ROOT)


def add_subset(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--canonical-dir", type=Path)


def add_devices(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--devices",
        default="0",
        help=(
            "Comma-separated process-visible devices such as 0,1,2,3 or cpu. "
            "After CUDA_VISIBLE_DEVICES, numbering starts again at cuda:0."
        ),
    )
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--amp", choices=("fp16", "fp32"), default="fp16")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EvidenceGap V1 Phase 05 EvidenceBench sentence retrieval"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Stream and validate a manifest-selected subset")
    add_root(audit)
    audit.add_argument("--split", choices=SPLITS, required=True)
    audit.add_argument("--max-queries", type=int)
    audit.add_argument("--manifest-dir", type=Path)

    prepare = sub.add_parser("prepare", help="Materialize canonical EvidenceBench records")
    add_root(prepare)
    prepare.add_argument("--split", choices=SPLITS, required=True)
    prepare.add_argument("--max-queries", type=int)
    prepare.add_argument("--manifest-dir", type=Path)
    prepare.add_argument("--output-dir", type=Path)
    prepare.add_argument("--force", action="store_true")

    bm25 = sub.add_parser("bm25", help="Run paper-local BM25 sentence retrieval")
    add_root(bm25)
    add_subset(bm25)
    bm25.add_argument("--run-name")
    bm25.add_argument("--top-k", type=int, default=20)
    bm25.add_argument("--k1", type=float, default=1.5)
    bm25.add_argument("--b", type=float, default=0.75)
    bm25.add_argument("--run-dir", type=Path)
    bm25.add_argument("--report-dir", type=Path)
    bm25.add_argument("--force", action="store_true")

    dense = sub.add_parser("dense", help="Run MedCPT or BMRetriever sentence retrieval")
    add_root(dense)
    add_subset(dense)
    dense.add_argument("--model", choices=DENSE_MODELS, required=True)
    dense.add_argument("--run-name")
    dense.add_argument("--top-k", type=int, default=20)
    add_devices(dense)
    dense.add_argument("--artifact-root", type=Path)
    dense.add_argument("--report-dir", type=Path)
    dense.add_argument("--force", action="store_true")

    rerank = sub.add_parser("rerank", help="Rerank retrieval Top-N with MedCPT cross encoder")
    add_root(rerank)
    rerank.add_argument("--split", choices=SPLITS, required=True)
    rerank.add_argument("--canonical-dir", type=Path, required=True)
    rerank.add_argument("--candidate-path", type=Path, required=True)
    rerank.add_argument("--candidate-run-name")
    rerank.add_argument("--run-name")
    rerank.add_argument("--model-dir", type=Path)
    rerank.add_argument("--rerank-depth", type=int, default=20)
    add_devices(rerank)
    rerank.set_defaults(batch_size=16)
    rerank.add_argument("--max-length", type=int, default=512)
    rerank.add_argument("--artifact-root", type=Path)
    rerank.add_argument("--report-dir", type=Path)
    rerank.add_argument("--force", action="store_true")

    complementarity = sub.add_parser(
        "complementarity",
        help="Measure sentence and aspect complementarity between two retrieval runs",
    )
    add_root(complementarity)
    complementarity.add_argument("--canonical-dir", type=Path, required=True)
    complementarity.add_argument("--left", type=Path, required=True)
    complementarity.add_argument("--right", type=Path, required=True)
    complementarity.add_argument("--left-name")
    complementarity.add_argument("--right-name")
    complementarity.add_argument(
        "--depths",
        default="5,10,20,50",
        help="Comma-separated source depths to inspect",
    )
    complementarity.add_argument("--report", type=Path)

    fuse = sub.add_parser(
        "fuse",
        help="Build an equal-weight RRF sentence candidate run from two retrieval runs",
    )
    add_root(fuse)
    fuse.add_argument("--split", choices=SPLITS, required=True)
    fuse.add_argument("--canonical-dir", type=Path, required=True)
    fuse.add_argument("--left", type=Path, required=True)
    fuse.add_argument("--right", type=Path, required=True)
    fuse.add_argument("--left-name")
    fuse.add_argument("--right-name")
    fuse.add_argument("--left-depth", type=int, default=20)
    fuse.add_argument("--right-depth", type=int, default=20)
    fuse.add_argument(
        "--output-depth",
        type=int,
        help="Keep only RRF Top-K. Omit to preserve the full unique union.",
    )
    fuse.add_argument("--rrf-k", type=int, default=60)
    fuse.add_argument("--run-name")
    fuse.add_argument("--artifact-root", type=Path)
    fuse.add_argument("--report-dir", type=Path)
    fuse.add_argument("--force", action="store_true")

    compare = sub.add_parser(
        "compare",
        help="Paired per-query comparison with bootstrap confidence intervals",
    )
    add_root(compare)
    compare.add_argument("--canonical-dir", type=Path, required=True)
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--challenger", type=Path, required=True)
    compare.add_argument("--baseline-name")
    compare.add_argument("--challenger-name")
    compare.add_argument("--bootstrap-samples", type=int, default=5000)
    compare.add_argument("--seed", type=int, default=20260722)
    compare.add_argument(
        "--bootstrap-unit",
        choices=("query", "paper", "systematic_review"),
        default="systematic_review",
    )
    compare.add_argument("--report", type=Path)

    evaluate = sub.add_parser("evaluate", help="Evaluate a ranked sentence parquet")
    add_root(evaluate)
    evaluate.add_argument("--canonical-dir", type=Path, required=True)
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--report", type=Path)

    diagnose = sub.add_parser("diagnose", help="Check score direction against sentence gold")
    add_root(diagnose)
    diagnose.add_argument("--canonical-dir", type=Path, required=True)
    diagnose.add_argument("--run", type=Path, required=True)
    diagnose.add_argument(
        "--score-field",
        choices=("retrieval_score", "cross_encoder_score", "final_score"),
        default="retrieval_score",
    )
    diagnose.add_argument("--pair-limit-per-query", type=int, default=5000)

    validate = sub.add_parser("validate", help="Validate canonical or ranking artifacts")
    add_root(validate)
    validate.add_argument("--canonical-dir", type=Path, required=True)
    validate.add_argument("--run", type=Path)
    validate.add_argument("--run-name")

    return parser


def _devices(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    if args.command == "audit":
        result = audit_evidencebench(
            root,
            split=args.split,
            max_queries=args.max_queries,
            manifest_dir=args.manifest_dir,
        )
    elif args.command == "prepare":
        result = prepare_evidencebench_canonical(
            root,
            split=args.split,
            max_queries=args.max_queries,
            manifest_dir=args.manifest_dir,
            output_dir=args.output_dir,
            force=args.force,
        )
    elif args.command == "bm25":
        result = run_bm25_sentence_retrieval(
            root,
            split=args.split,
            max_queries=args.max_queries,
            canonical_dir=args.canonical_dir,
            run_name=args.run_name,
            top_k=args.top_k,
            k1=args.k1,
            b=args.b,
            run_dir=args.run_dir,
            report_dir=args.report_dir,
            force=args.force,
        )
    elif args.command == "dense":
        result = run_dense_sentence_retrieval(
            root,
            model_key=args.model,
            split=args.split,
            devices=_devices(args.devices),
            max_queries=args.max_queries,
            canonical_dir=args.canonical_dir,
            run_name=args.run_name,
            top_k=args.top_k,
            num_shards=args.num_shards,
            batch_size=args.batch_size,
            amp=args.amp,
            artifact_root=args.artifact_root,
            report_dir=args.report_dir,
            force=args.force,
        )
    elif args.command == "rerank":
        result = run_cross_encoder_sentence_reranking(
            root,
            split=args.split,
            canonical_dir=args.canonical_dir,
            candidate_path=args.candidate_path,
            devices=_devices(args.devices),
            run_name=args.run_name,
            candidate_run_name=args.candidate_run_name,
            model_dir=args.model_dir,
            rerank_depth=args.rerank_depth,
            num_shards=args.num_shards,
            batch_size=args.batch_size,
            max_length=args.max_length,
            amp=args.amp,
            artifact_root=args.artifact_root,
            report_dir=args.report_dir,
            force=args.force,
        )
    elif args.command == "complementarity":
        depths = [int(value) for value in args.depths.split(",") if value.strip()]
        result = analyze_sentence_run_complementarity(
            canonical_dir=args.canonical_dir,
            left_path=args.left,
            right_path=args.right,
            depths=depths,
            left_name=args.left_name,
            right_name=args.right_name,
            report_path=args.report,
        )
    elif args.command == "fuse":
        result = run_sentence_rrf_fusion(
            root,
            split=args.split,
            canonical_dir=args.canonical_dir,
            left_path=args.left,
            right_path=args.right,
            left_name=args.left_name,
            right_name=args.right_name,
            left_depth=args.left_depth,
            right_depth=args.right_depth,
            output_depth=args.output_depth,
            rrf_k=args.rrf_k,
            run_name=args.run_name,
            artifact_root=args.artifact_root,
            report_dir=args.report_dir,
            force=args.force,
        )
    elif args.command == "compare":
        result = compare_sentence_runs_paired(
            canonical_dir=args.canonical_dir,
            baseline_path=args.baseline,
            challenger_path=args.challenger,
            baseline_name=args.baseline_name,
            challenger_name=args.challenger_name,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            bootstrap_unit=args.bootstrap_unit,
            report_path=args.report,
        )
    elif args.command == "evaluate":
        result = evaluate_sentence_run(
            root,
            canonical_dir=args.canonical_dir,
            run_path=args.run,
            report_path=args.report,
        )
    elif args.command == "diagnose":
        result = diagnose_sentence_run(
            canonical_dir=args.canonical_dir,
            run_path=args.run,
            score_field=args.score_field,
            pair_limit_per_query=args.pair_limit_per_query,
        )
    elif args.command == "validate":
        queries, manifest = load_canonical_queries(args.canonical_dir)
        result = {
            "canonical": {
                "status": "PASS",
                "split": manifest["split"],
                "queries": len(queries),
                "candidate_sentences": sum(len(query.candidate_sentences) for query in queries),
            }
        }
        if args.run:
            result["run"] = validate_ranking_rows(
                args.run,
                expected_queries={query.query_id: len(query.candidate_sentences) for query in queries},
                expected_run_name=args.run_name,
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
