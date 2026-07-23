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
    check_stanza_runtime,
    download_stanza_sentence_model,
    materialize_runtime_sentences,
    run_article_evidence_extractor,
    run_claim_aggregation,
    run_retrieval_adapters,
    validate_article_evidence_artifact,
    validate_claim_aggregation_artifact,
    validate_retrieval_adapter_artifact,
    validate_runtime_sentence_artifact,
)


def _root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=ROOT)


def _stanza_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--package", default="genia")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Retry on CPU only when Stanza initialization looks like a CUDA failure",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EvidenceGap V1 Phase 07 offline runtime pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser(
        "download-sentence-model",
        help="Download only the English GENIA tokenizer used by runtime materialization",
    )
    _root(download)
    download.add_argument("--model-dir", type=Path)
    download.add_argument("--package", default="genia")
    download.add_argument(
        "--download-source",
        choices=("auto", "huggingface", "stanford"),
        default="auto",
        help=(
            "Model host to use. auto tries Hugging Face first, then the "
            "official Stanford download server."
        ),
    )

    check = sub.add_parser(
        "check-sentence-runtime",
        help="Load Stanza on the selected device and segment a biomedical smoke input",
    )
    _root(check)
    _stanza_options(check)

    materialize = sub.add_parser(
        "materialize-sentences",
        help="Convert runtime article JSON/JSONL/Parquet into stable sentence artifacts",
    )
    _root(materialize)
    materialize.add_argument("--input-path", type=Path, required=True)
    materialize.add_argument("--run-name", required=True)
    _stanza_options(materialize)
    materialize.add_argument(
        "--section-mode",
        choices=("auto", "none"),
        default="auto",
        help="Detect common structured-abstract labels, or treat the body as one section",
    )
    materialize.add_argument("--artifact-root", type=Path)
    materialize.add_argument("--force", action="store_true")

    validate = sub.add_parser(
        "validate-sentences",
        help="Validate offsets, stable indices, manifest checksums, and splitter provenance",
    )
    _root(validate)
    validate.add_argument("--artifact-dir", type=Path, required=True)

    retrieve = sub.add_parser(
        "retrieve-evidence",
        help=(
            "Run frozen Phase 04 article retrieval, Stanza runtime sentence "
            "materialization, and frozen Phase 05 per-article evidence retrieval"
        ),
    )
    _root(retrieve)
    retrieve.add_argument("--claim", required=True)
    retrieve.add_argument("--run-name", required=True)
    retrieve.add_argument("--device", default="cuda:0")
    retrieve.add_argument("--amp", choices=("fp16", "fp32"), default="fp16")
    retrieve.add_argument("--artifact-root", type=Path)
    retrieve.add_argument("--corpus-dir", type=Path)
    retrieve.add_argument("--article-input-dir", type=Path)
    retrieve.add_argument("--bm25-index-dir", type=Path)
    retrieve.add_argument("--medcpt-index-dir", type=Path)
    retrieve.add_argument("--bmretriever-index-dir", type=Path)
    retrieve.add_argument("--cross-encoder-model-dir", type=Path)
    retrieve.add_argument("--stanza-model-dir", type=Path)
    retrieve.add_argument("--stanza-package", default="genia")
    retrieve.add_argument("--stanza-batch-size", type=int, default=32)
    retrieve.add_argument("--cross-encoder-batch-size", type=int, default=16)
    retrieve.add_argument("--medcpt-sentence-batch-size", type=int, default=64)
    retrieve.add_argument("--bmretriever-sentence-batch-size", type=int, default=8)
    retrieve.add_argument(
        "--section-mode", choices=("auto", "none"), default="auto"
    )
    retrieve.add_argument("--allow-cpu-fallback", action="store_true")
    retrieve.add_argument("--force", action="store_true")

    validate_retrieval = sub.add_parser(
        "validate-retrieval-adapters",
        help="Validate Phase 07.2 article, sentence, and evidence artifacts",
    )
    _root(validate_retrieval)
    validate_retrieval.add_argument("--artifact-dir", type=Path, required=True)

    extract = sub.add_parser(
        "extract-article-evidence",
        help=(
            "Let DeepSeek or Claude read every numbered abstract sentence for "
            "each Top Article, select evidence sentence IDs, and judge article stance"
        ),
    )
    _root(extract)
    extract.add_argument("--retrieval-artifact-dir", type=Path, required=True)
    extract.add_argument(
        "--provider", choices=("deepseek", "anthropic"), required=True
    )
    extract.add_argument("--model")
    extract.add_argument("--run-name")
    extract.add_argument("--api-key-env")
    extract.add_argument("--base-url")
    extract.add_argument("--request-batch-size", type=int, default=2)
    extract.add_argument("--max-tokens", type=int, default=4096)
    extract.add_argument("--timeout-seconds", type=float, default=180.0)
    extract.add_argument("--max-retries", type=int, default=4)
    extract.add_argument(
        "--thinking",
        action="store_true",
        help="Enable DeepSeek thinking mode; disabled by default",
    )
    extract.add_argument("--dry-run", action="store_true")
    extract.add_argument("--cache-dir", type=Path)
    extract.add_argument("--artifact-root", type=Path)
    extract.add_argument("--force", action="store_true")

    validate_article = sub.add_parser(
        "validate-article-evidence",
        help="Validate article-level LLM evidence IDs, stance rows, and checksums",
    )
    _root(validate_article)
    validate_article.add_argument("--artifact-dir", type=Path, required=True)

    aggregate = sub.add_parser(
        "aggregate-claim",
        help=(
            "Aggregate article-level support/refute/insufficient results into a "
            "transparent claim verdict"
        ),
    )
    _root(aggregate)
    aggregate.add_argument(
        "--article-evidence-artifact-dir", type=Path, required=True
    )
    aggregate.add_argument("--run-name")
    aggregate.add_argument("--artifact-root", type=Path)
    aggregate.add_argument("--force", action="store_true")

    validate_aggregation = sub.add_parser(
        "validate-claim-aggregation",
        help="Validate deterministic claim aggregation and source checksums",
    )
    _root(validate_aggregation)
    validate_aggregation.add_argument("--artifact-dir", type=Path, required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    if args.command == "download-sentence-model":
        result = download_stanza_sentence_model(
            root,
            model_dir=args.model_dir,
            package=args.package,
            download_source=args.download_source,
        )
    elif args.command == "check-sentence-runtime":
        result = check_stanza_runtime(
            root,
            model_dir=args.model_dir,
            device=args.device,
            package=args.package,
            batch_size=args.batch_size,
            allow_cpu_fallback=args.allow_cpu_fallback,
        )
    elif args.command == "materialize-sentences":
        result = materialize_runtime_sentences(
            root,
            input_path=args.input_path,
            run_name=args.run_name,
            model_dir=args.model_dir,
            device=args.device,
            package=args.package,
            batch_size=args.batch_size,
            section_mode=args.section_mode,
            allow_cpu_fallback=args.allow_cpu_fallback,
            artifact_root=args.artifact_root,
            force=args.force,
        )
    elif args.command == "validate-sentences":
        result = validate_runtime_sentence_artifact(args.artifact_dir)
    elif args.command == "retrieve-evidence":
        result = run_retrieval_adapters(
            root,
            claim=args.claim,
            run_name=args.run_name,
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
            medcpt_sentence_batch_size=args.medcpt_sentence_batch_size,
            bmretriever_sentence_batch_size=args.bmretriever_sentence_batch_size,
            section_mode=args.section_mode,
            allow_cpu_fallback=args.allow_cpu_fallback,
            force=args.force,
        )
    elif args.command == "validate-retrieval-adapters":
        result = validate_retrieval_adapter_artifact(args.artifact_dir)
    elif args.command == "extract-article-evidence":
        result = run_article_evidence_extractor(
            root,
            retrieval_artifact_dir=args.retrieval_artifact_dir,
            provider=args.provider,
            model=args.model,
            run_name=args.run_name,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            request_batch_size=args.request_batch_size,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            thinking=args.thinking,
            dry_run=args.dry_run,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            force=args.force,
        )
    elif args.command == "validate-article-evidence":
        result = validate_article_evidence_artifact(args.artifact_dir)
    elif args.command == "aggregate-claim":
        result = run_claim_aggregation(
            root,
            article_evidence_artifact_dir=args.article_evidence_artifact_dir,
            run_name=args.run_name,
            artifact_root=args.artifact_root,
            force=args.force,
        )
    elif args.command == "validate-claim-aggregation":
        result = validate_claim_aggregation_artifact(args.artifact_dir)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except EvidenceGapError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
