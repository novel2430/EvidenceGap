#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidencegap.common import EvidenceGapError
from evidencegap.stance import (
    evaluate_stance_predictions,
    export_graph_ready_stance,
    export_llm_stance_cache,
    prepare_healthfc_stance_inputs,
    prepare_phase05_stance_inputs,
    run_deberta_zero_shot,
    run_llm_stance_judge,
    validate_input_artifact,
    validate_prediction_artifact,
)


def _root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EvidenceGap V1 Phase 06 stance verification"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    phase05 = sub.add_parser(
        "prepare-phase05",
        help="Materialize stance inputs from a frozen Phase 05 sentence ranking",
    )
    _root(phase05)
    phase05.add_argument("--split", choices=("dev", "test"), default="dev")
    phase05.add_argument("--canonical-dir", type=Path)
    phase05.add_argument("--ranking-path", type=Path)
    phase05.add_argument("--top-k", type=int, default=5)
    phase05.add_argument(
        "--context-window",
        type=int,
        default=1,
        help="Exact neighboring canonical sentences on each side; 0 disables context",
    )
    phase05.add_argument("--run-name")
    phase05.add_argument("--artifact-root", type=Path)
    phase05.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly permit the frozen Phase 05 test artifact for final evaluation",
    )
    phase05.add_argument("--force", action="store_true")

    healthfc = sub.add_parser(
        "prepare-healthfc",
        help="Materialize expert-labeled HealthFC evidence bundles using the stance schema",
    )
    _root(healthfc)
    healthfc.add_argument("--manifest-path", type=Path)
    healthfc.add_argument("--raw-path", type=Path)
    healthfc.add_argument("--run-name", default="healthfc_eval")
    healthfc.add_argument("--artifact-root", type=Path)
    healthfc.add_argument("--force", action="store_true")

    zero_shot = sub.add_parser(
        "zero-shot",
        help="Run cross-encoder/nli-deberta-v3-base with evidence as premise",
    )
    _root(zero_shot)
    zero_shot.add_argument("--input-path", type=Path, required=True)
    zero_shot.add_argument("--run-name")
    zero_shot.add_argument("--model-dir", type=Path)
    zero_shot.add_argument("--device", default="cuda:0")
    zero_shot.add_argument("--batch-size", type=int, default=16)
    zero_shot.add_argument("--max-length", type=int, default=512)
    zero_shot.add_argument("--amp", choices=("none", "fp16", "bf16"), default="fp16")
    zero_shot.add_argument("--artifact-root", type=Path)
    zero_shot.add_argument("--report-dir", type=Path)
    zero_shot.add_argument("--force", action="store_true")


    llm_judge = sub.add_parser(
        "llm-judge",
        help="Run the structured LLM stance judge with DeepSeek or Claude",
    )
    _root(llm_judge)
    llm_judge.add_argument("--input-path", type=Path, required=True)
    llm_judge.add_argument("--provider", choices=("deepseek", "anthropic"), required=True)
    llm_judge.add_argument("--model")
    llm_judge.add_argument("--run-name")
    llm_judge.add_argument("--api-key-env")
    llm_judge.add_argument("--base-url")
    llm_judge.add_argument("--request-batch-size", type=int, default=8)
    llm_judge.add_argument("--max-tokens", type=int, default=4096)
    llm_judge.add_argument("--timeout-seconds", type=float, default=180.0)
    llm_judge.add_argument("--max-retries", type=int, default=4)
    llm_judge.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Row-level offset; retained for bundle evaluations",
    )
    llm_judge.add_argument(
        "--limit",
        type=int,
        help="Row-level limit; for Phase 05 use query-level sampling instead",
    )
    llm_judge.add_argument(
        "--query-offset",
        type=int,
        default=0,
        help="Select complete query groups starting from this query offset",
    )
    llm_judge.add_argument(
        "--query-limit",
        type=int,
        help="Select this many complete query groups",
    )
    llm_judge.add_argument(
        "--query-sample-size",
        type=int,
        help="Deterministically sample complete query groups for a Phase 05 smoke run",
    )
    llm_judge.add_argument(
        "--query-sample-seed",
        type=int,
        default=20260722,
    )
    llm_judge.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selection and print request estimates without requiring an API key",
    )
    llm_judge.add_argument(
        "--thinking",
        action="store_true",
        help="Enable DeepSeek thinking mode; disabled by default for classification cost/latency",
    )
    llm_judge.add_argument("--cache-dir", type=Path)
    llm_judge.add_argument("--artifact-root", type=Path)
    llm_judge.add_argument("--report-dir", type=Path)
    llm_judge.add_argument("--force", action="store_true")

    export_cache = sub.add_parser(
        "export-cache",
        help="Export completed exact-match LLM cache batches without API calls",
    )
    _root(export_cache)
    export_cache.add_argument("--input-path", type=Path, required=True)
    export_cache.add_argument(
        "--provider", choices=("deepseek", "anthropic"), required=True
    )
    export_cache.add_argument("--model")
    export_cache.add_argument("--run-name")
    export_cache.add_argument("--base-url")
    export_cache.add_argument("--request-batch-size", type=int, default=8)
    export_cache.add_argument("--max-tokens", type=int, default=4096)
    export_cache.add_argument(
        "--thinking",
        action="store_true",
        help="Match a DeepSeek cache created with thinking enabled",
    )
    export_cache.add_argument("--cache-dir", type=Path)
    export_cache.add_argument("--artifact-root", type=Path)
    export_cache.add_argument("--report-dir", type=Path)
    export_cache.add_argument("--force", action="store_true")

    graph_export = sub.add_parser(
        "export-graph",
        help="Aggregate sentence stances into graph-ready query/paper artifacts",
    )
    _root(graph_export)
    graph_export.add_argument("--prediction-path", type=Path, required=True)
    graph_export.add_argument("--run-name")
    graph_export.add_argument("--artifact-root", type=Path)
    graph_export.add_argument("--report-dir", type=Path)
    graph_export.add_argument("--force", action="store_true")

    evaluate = sub.add_parser(
        "evaluate",
        help="Evaluate a stance prediction artifact containing gold labels",
    )
    _root(evaluate)
    evaluate.add_argument("--prediction-path", type=Path, required=True)
    evaluate.add_argument("--report-path", type=Path)

    validate = sub.add_parser("validate", help="Validate Phase 06 Parquet artifacts")
    _root(validate)
    group = validate.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-path", type=Path)
    group.add_argument("--prediction-path", type=Path)
    validate.add_argument("--run-name")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    if args.command == "prepare-phase05":
        result = prepare_phase05_stance_inputs(
            root,
            split=args.split,
            canonical_dir=args.canonical_dir,
            ranking_path=args.ranking_path,
            top_k=args.top_k,
            context_window=args.context_window,
            run_name=args.run_name,
            artifact_root=args.artifact_root,
            allow_test=args.allow_test,
            force=args.force,
        )
    elif args.command == "prepare-healthfc":
        result = prepare_healthfc_stance_inputs(
            root,
            manifest_path=args.manifest_path,
            raw_path=args.raw_path,
            run_name=args.run_name,
            artifact_root=args.artifact_root,
            force=args.force,
        )
    elif args.command == "zero-shot":
        result = run_deberta_zero_shot(
            root,
            input_path=args.input_path,
            run_name=args.run_name,
            model_dir=args.model_dir,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            amp=args.amp,
            artifact_root=args.artifact_root,
            report_dir=args.report_dir,
            force=args.force,
        )
    elif args.command == "llm-judge":
        result = run_llm_stance_judge(
            root,
            input_path=args.input_path,
            provider=args.provider,
            model=args.model,
            run_name=args.run_name,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            request_batch_size=args.request_batch_size,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            offset=args.offset,
            limit=args.limit,
            query_offset=args.query_offset,
            query_limit=args.query_limit,
            query_sample_size=args.query_sample_size,
            query_sample_seed=args.query_sample_seed,
            dry_run=args.dry_run,
            thinking=args.thinking,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            report_dir=args.report_dir,
            force=args.force,
        )
    elif args.command == "export-cache":
        result = export_llm_stance_cache(
            root,
            input_path=args.input_path,
            provider=args.provider,
            model=args.model,
            run_name=args.run_name,
            base_url=args.base_url,
            request_batch_size=args.request_batch_size,
            max_tokens=args.max_tokens,
            thinking=args.thinking,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            report_dir=args.report_dir,
            force=args.force,
        )
    elif args.command == "export-graph":
        result = export_graph_ready_stance(
            root,
            prediction_path=args.prediction_path,
            run_name=args.run_name,
            artifact_root=args.artifact_root,
            report_dir=args.report_dir,
            force=args.force,
        )
    elif args.command == "evaluate":
        result = evaluate_stance_predictions(
            root,
            prediction_path=args.prediction_path,
            report_path=args.report_path,
        )
    elif args.command == "validate":
        if args.input_path is not None:
            result = validate_input_artifact(args.input_path)
        else:
            result = validate_prediction_artifact(
                args.prediction_path,
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
