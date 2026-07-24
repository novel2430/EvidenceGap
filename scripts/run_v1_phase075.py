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

from evidencegap.common import EvidenceGapError  # noqa: E402
from evidencegap.pipeline import (  # noqa: E402
    run_statement_analysis,
    run_statement_bundle,
    run_statement_decomposition,
    run_statement_pipeline,
    validate_statement_analysis_artifact,
    validate_statement_bundle_artifact,
    validate_statement_decomposition_artifact,
    validate_statement_pipeline_artifact,
)


def _root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=ROOT)


def _phase07_options(
    parser: argparse.ArgumentParser,
    *,
    include_thinking: bool = True,
    request_batch_size_default: int = 2,
    max_tokens_default: int = 4096,
) -> None:
    parser.add_argument("--provider", choices=("deepseek", "anthropic"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--article-input-dir", type=Path)
    parser.add_argument("--bm25-index-dir", type=Path)
    parser.add_argument("--medcpt-index-dir", type=Path)
    parser.add_argument("--bmretriever-index-dir", type=Path)
    parser.add_argument("--cross-encoder-model-dir", type=Path)
    parser.add_argument("--stanza-model-dir", type=Path)
    parser.add_argument("--stanza-package", default="genia")
    parser.add_argument("--stanza-batch-size", type=int, default=32)
    parser.add_argument("--cross-encoder-batch-size", type=int, default=16)
    parser.add_argument("--section-mode", choices=("auto", "none"), default="auto")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--api-key-env")
    parser.add_argument("--base-url")
    parser.add_argument(
        "--request-batch-size", type=int, default=request_batch_size_default
    )
    parser.add_argument("--max-tokens", type=int, default=max_tokens_default)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=4)
    if include_thinking:
        parser.add_argument(
            "--thinking",
            action="store_true",
            help="Enable DeepSeek thinking mode; disabled by default",
        )
    parser.add_argument("--cache-dir", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EvidenceGap V1 Phase 7.5 multilingual multi-claim pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    decompose = sub.add_parser(
        "decompose",
        help=(
            "Extract only directly verifiable biomedical claims from a multilingual "
            "statement and canonicalize them in English"
        ),
    )
    _root(decompose)
    decompose.add_argument("--statement", required=True)
    decompose.add_argument("--run-name", required=True)
    decompose.add_argument(
        "--provider", choices=("deepseek", "anthropic"), required=True
    )
    decompose.add_argument("--model")
    decompose.add_argument("--api-key-env")
    decompose.add_argument("--base-url")
    decompose.add_argument("--max-tokens", type=int, default=2048)
    decompose.add_argument("--timeout-seconds", type=float, default=180.0)
    decompose.add_argument("--max-retries", type=int, default=4)
    decompose.add_argument(
        "--thinking",
        action="store_true",
        help="Enable DeepSeek thinking mode; disabled by default",
    )
    decompose.add_argument("--artifact-root", type=Path)
    decompose.add_argument("--force", action="store_true")

    validate = sub.add_parser(
        "validate-decomposition",
        help="Validate the decomposition contract, claim identities, and checksums",
    )
    _root(validate)
    validate.add_argument("--artifact-dir", type=Path, required=True)

    analyze = sub.add_parser(
        "analyze-claims",
        help=(
            "Run the existing Phase 07 pipeline sequentially for every canonical "
            "English claim in a validated Phase 7.5 decomposition artifact"
        ),
    )
    _root(analyze)
    analyze.add_argument("--decomposition-artifact-dir", type=Path, required=True)
    analyze.add_argument("--run-name", required=True)
    analyze.add_argument("--artifact-root", type=Path)
    _phase07_options(analyze)
    analyze.add_argument("--force", action="store_true")

    validate_analysis = sub.add_parser(
        "validate-claim-analysis",
        help=(
            "Validate the multi-claim result, source decomposition checksum, and "
            "every completed nested Phase 07 analysis"
        ),
    )
    _root(validate_analysis)
    validate_analysis.add_argument("--artifact-dir", type=Path, required=True)

    build_bundle = sub.add_parser(
        "build-statement-bundle",
        help=(
            "Merge the decomposition, inference relationships, and every completed "
            "Phase 07 final graph into one language-neutral statement bundle"
        ),
    )
    _root(build_bundle)
    build_bundle.add_argument(
        "--statement-analysis-artifact-dir", type=Path, required=True
    )
    build_bundle.add_argument("--run-name", required=True)
    build_bundle.add_argument("--artifact-root", type=Path)
    build_bundle.add_argument("--force", action="store_true")

    validate_bundle = sub.add_parser(
        "validate-statement-bundle",
        help=(
            "Validate the merged statement bundle against its decomposition, "
            "multi-claim analysis, and Phase 07 final graph sources"
        ),
    )
    _root(validate_bundle)
    validate_bundle.add_argument("--artifact-dir", type=Path, required=True)

    run = sub.add_parser(
        "run",
        help=(
            "Run Phase 7.5 end to end: multilingual decomposition, sequential "
            "Phase 07 analyses, and final statement bundle assembly"
        ),
    )
    _root(run)
    run.add_argument("--statement", required=True)
    run.add_argument("--run-name", required=True)
    run.add_argument("--artifact-root", type=Path)
    _phase07_options(
        run,
        include_thinking=False,
        request_batch_size_default=1,
        max_tokens_default=8192,
    )
    run.add_argument("--decomposition-max-tokens", type=int, default=2048)
    run.add_argument(
        "--decomposition-thinking",
        action="store_true",
        help="Enable DeepSeek thinking for claim decomposition; disabled by default",
    )
    analysis_thinking = run.add_mutually_exclusive_group()
    analysis_thinking.add_argument(
        "--analysis-thinking",
        "--thinking",
        dest="analysis_thinking",
        action="store_true",
        help="Enable DeepSeek thinking for Phase 07 analysis; default for DeepSeek",
    )
    analysis_thinking.add_argument(
        "--no-analysis-thinking",
        dest="analysis_thinking",
        action="store_false",
        help="Disable DeepSeek thinking for Phase 07 analysis",
    )
    run.set_defaults(analysis_thinking=None)
    run.add_argument("--force", action="store_true")

    validate_run = sub.add_parser(
        "validate-run",
        help=(
            "Validate the complete Phase 7.5 run and all nested decomposition, "
            "Phase 07 analysis, and statement bundle artifacts"
        ),
    )
    _root(validate_run)
    validate_run.add_argument("--artifact-dir", type=Path, required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    if args.command == "decompose":
        result = run_statement_decomposition(
            root,
            statement=args.statement,
            provider=args.provider,
            run_name=args.run_name,
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            thinking=args.thinking,
            artifact_root=args.artifact_root,
            force=args.force,
        )
    elif args.command == "validate-decomposition":
        result = validate_statement_decomposition_artifact(args.artifact_dir)
    elif args.command == "analyze-claims":
        result = run_statement_analysis(
            root,
            decomposition_artifact_dir=args.decomposition_artifact_dir,
            run_name=args.run_name,
            provider=args.provider,
            model=args.model,
            device=args.device,
            amp=args.amp,
            artifact_root=args.artifact_root,
            corpus_dir=args.corpus_dir,
            article_input_dir=args.article_input_dir,
            bm25_index_dir=args.bm25_index_dir,
            medcpt_index_dir=args.medcpt_index_dir,
            bmretriever_index_dir=args.bmretriever_index_dir,
            cross_encoder_model_dir=args.cross_encoder_model_dir,
            stanza_model_dir=args.stanza_model_dir,
            stanza_package=args.stanza_package,
            stanza_batch_size=args.stanza_batch_size,
            cross_encoder_batch_size=args.cross_encoder_batch_size,
            section_mode=args.section_mode,
            allow_cpu_fallback=args.allow_cpu_fallback,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            request_batch_size=args.request_batch_size,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            thinking=args.thinking,
            cache_dir=args.cache_dir,
            force=args.force,
        )
    elif args.command == "validate-claim-analysis":
        result = validate_statement_analysis_artifact(args.artifact_dir)
    elif args.command == "build-statement-bundle":
        result = run_statement_bundle(
            root,
            statement_analysis_artifact_dir=args.statement_analysis_artifact_dir,
            run_name=args.run_name,
            artifact_root=args.artifact_root,
            force=args.force,
        )
    elif args.command == "validate-statement-bundle":
        result = validate_statement_bundle_artifact(args.artifact_dir)
    elif args.command == "run":
        result = run_statement_pipeline(
            root,
            statement=args.statement,
            run_name=args.run_name,
            provider=args.provider,
            model=args.model,
            device=args.device,
            amp=args.amp,
            artifact_root=args.artifact_root,
            corpus_dir=args.corpus_dir,
            article_input_dir=args.article_input_dir,
            bm25_index_dir=args.bm25_index_dir,
            medcpt_index_dir=args.medcpt_index_dir,
            bmretriever_index_dir=args.bmretriever_index_dir,
            cross_encoder_model_dir=args.cross_encoder_model_dir,
            stanza_model_dir=args.stanza_model_dir,
            stanza_package=args.stanza_package,
            stanza_batch_size=args.stanza_batch_size,
            cross_encoder_batch_size=args.cross_encoder_batch_size,
            section_mode=args.section_mode,
            allow_cpu_fallback=args.allow_cpu_fallback,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            decomposition_max_tokens=args.decomposition_max_tokens,
            request_batch_size=args.request_batch_size,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            decomposition_thinking=args.decomposition_thinking,
            analysis_thinking=args.analysis_thinking,
            cache_dir=args.cache_dir,
            force=args.force,
        )
    elif args.command == "validate-run":
        result = validate_statement_pipeline_artifact(args.artifact_dir)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except EvidenceGapError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
