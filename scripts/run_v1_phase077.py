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
from evidencegap.output import (  # noqa: E402
    run_output_module,
    validate_output_artifact,
)
from evidencegap.pipeline import (  # noqa: E402
    run_inference_gap_analysis,
    validate_inference_gap_analysis_artifact,
)


def _root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EvidenceGap V1 Phase 7.7 gap analysis and output module"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser(
        "analyze-inference-gaps",
        help=(
            "Analyze every inference step in a validated statement bundle for "
            "SCOPE_GAP and CAUSAL_GAP"
        ),
    )
    _root(analyze)
    analyze.add_argument("--statement-bundle-artifact-dir", type=Path, required=True)
    analyze.add_argument("--run-name", required=True)
    analyze.add_argument(
        "--provider", choices=("deepseek", "anthropic"), required=True
    )
    analyze.add_argument("--model")
    analyze.add_argument("--api-key-env")
    analyze.add_argument("--base-url")
    analyze.add_argument("--max-tokens", type=int, default=4096)
    analyze.add_argument("--timeout-seconds", type=float, default=180.0)
    analyze.add_argument("--max-retries", type=int, default=4)
    thinking = analyze.add_mutually_exclusive_group()
    thinking.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        help="Enable DeepSeek thinking mode; this is the default for DeepSeek",
    )
    thinking.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Disable DeepSeek thinking mode explicitly",
    )
    analyze.set_defaults(thinking=None)
    analyze.add_argument("--artifact-root", type=Path)
    analyze.add_argument("--force", action="store_true")

    validate = sub.add_parser(
        "validate-inference-gaps",
        help="Validate inference gap identities, source links, and checksums",
    )
    _root(validate)
    validate.add_argument("--artifact-dir", type=Path, required=True)

    output = sub.add_parser(
        "build-output",
        help=(
            "Merge a validated statement bundle and inference gap artifact into "
            "a frontend-ready presentation bundle, optionally localized"
        ),
    )
    _root(output)
    output.add_argument("--statement-bundle-artifact-dir", type=Path, required=True)
    output.add_argument("--inference-gap-artifact-dir", type=Path, required=True)
    output.add_argument("--run-name", required=True)
    output.add_argument(
        "--language",
        default="English",
        help=(
            "Free-form target language, for example '繁體中文（台灣）'. "
            "English is the default and skips the translation API call."
        ),
    )
    output.add_argument(
        "--provider", choices=("deepseek", "anthropic"), default="deepseek"
    )
    output.add_argument("--model")
    output.add_argument("--api-key-env")
    output.add_argument("--base-url")
    output.add_argument("--max-tokens", type=int, default=8192)
    output.add_argument(
        "--request-batch-size",
        type=int,
        default=32,
        help=(
            "Maximum number of translation text units per LLM request. "
            "Reduce this when a provider truncates localization output."
        ),
    )
    output.add_argument("--timeout-seconds", type=float, default=180.0)
    output.add_argument("--max-retries", type=int, default=4)
    output.add_argument("--artifact-root", type=Path)
    output.add_argument("--force", action="store_true")

    validate_output = sub.add_parser(
        "validate-output",
        help="Validate presentation source links, identities, counts, and checksums",
    )
    _root(validate_output)
    validate_output.add_argument("--artifact-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    if args.command == "analyze-inference-gaps":
        result = run_inference_gap_analysis(
            root,
            statement_bundle_artifact_dir=args.statement_bundle_artifact_dir,
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
    elif args.command == "validate-inference-gaps":
        result = validate_inference_gap_analysis_artifact(args.artifact_dir)
    elif args.command == "build-output":
        result = run_output_module(
            root,
            statement_bundle_artifact_dir=args.statement_bundle_artifact_dir,
            inference_gap_artifact_dir=args.inference_gap_artifact_dir,
            run_name=args.run_name,
            language=args.language,
            provider=args.provider,
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            request_batch_size=args.request_batch_size,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            artifact_root=args.artifact_root,
            force=args.force,
        )
    elif args.command == "validate-output":
        result = validate_output_artifact(args.artifact_dir)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except EvidenceGapError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
