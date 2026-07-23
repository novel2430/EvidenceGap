from __future__ import annotations

import json
import platform
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    relative_path,
    sha256_file,
    sha256_text,
)
from evidencegap.pipeline.contracts import (
    RUNTIME_SENTENCE_CONTRACT_ID,
    RUNTIME_SENTENCE_SCHEMA_VERSION,
    ArticleSection,
    RuntimeArticle,
    RuntimeSentence,
)

DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/pipeline/runtime_sentences")
DEFAULT_STANZA_MODEL_DIR = Path("models/v1/stanza")
RUN_SCHEMA_VERSION = "1.0.0"
RUN_TYPE = "phase07_runtime_sentence_materialization"
DEFAULT_STANZA_PACKAGE = "genia"
DEFAULT_PROCESSORS = {"tokenize": DEFAULT_STANZA_PACKAGE}
STANFORD_MODEL_URL = (
    "https://nlp.stanford.edu/software/stanza/"
    "{resources_version}/{lang}/{filename}"
)
DOWNLOAD_SOURCES = ("auto", "huggingface", "stanford")

# MedFact retains structured-abstract headings as inline text but does not
# retain PubMed AbstractText Label/NlmCategory metadata.  Recover only a frozen
# allowlist of common biomedical headings before Stanza sentence segmentation.
# Automatic parsing activates only when at least two headings are found, which
# avoids treating a one-off prose label as a full abstract structure.
STRUCTURED_ABSTRACT_PARSER_ID = "phase07.structured-abstract-header.v1"
_SECTION_LABELS = (
    "abstract",
    "aim",
    "aims",
    "background",
    "case",
    "case presentation",
    "case report",
    "conclusion",
    "conclusions",
    "conclusions and relevance",
    "context",
    "data extraction",
    "data sources",
    "data synthesis",
    "design",
    "design and setting",
    "discussion",
    "eligibility criteria",
    "importance",
    "intervention",
    "interventions",
    "main outcome",
    "main outcomes",
    "main outcome measure",
    "main outcome measures",
    "main outcomes and measures",
    "materials",
    "materials and methods",
    "measurements",
    "measures",
    "methodology",
    "methods",
    "objective",
    "objectives",
    "outcome",
    "outcomes",
    "participants",
    "patients",
    "results",
    "review methods",
    "selection criteria",
    "setting",
    "settings",
    "subjects",
    "trial registration",
)
# Match longer labels first so a prefix such as "conclusions" cannot obscure
# "conclusions and relevance".
_SECTION_PATTERN = re.compile(
    r"(?:(?<=^)|(?<=\n)|(?<=\s))"
    r"(?P<label>"
    + "|".join(
        re.escape(value)
        for value in sorted(_SECTION_LABELS, key=lambda value: (-len(value), value))
    )
    + r")"
    r"\s*:\s*",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SentenceSpan:
    start_char: int
    end_char: int
    text: str


@dataclass(frozen=True)
class SourceSegment:
    sentence_type: str
    section: str
    section_index: int
    text: str
    document_start: int


@dataclass(frozen=True)
class CanonicalArticleText:
    source_text: str
    segments: tuple[SourceSegment, ...]


class SentenceSplitter(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def split_many(self, texts: Sequence[str]) -> list[list[SentenceSpan]]: ...


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise EvidenceGapError("run_name cannot be empty after sanitization")
    return cleaned


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing dependency pyarrow. Install requirements/v1-phase07.txt"
        ) from exc
    return pa, pq


def _write_parquet_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    if not rows:
        raise EvidenceGapError(f"Refusing to write empty Parquet artifact: {path}")
    pa, pq = _pyarrow()
    table = pa.Table.from_pylist([dict(row) for row in rows])
    temp = path.with_name(path.name + ".tmp")
    pq.write_table(table, temp, compression="zstd")
    temp.replace(path)
    return table.num_rows


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    temp = path.with_name(path.name + ".tmp")
    count = 0
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temp.replace(path)
    return count


def _load_json_records(path: Path) -> list[Mapping[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing article input: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceGapError(f"Invalid JSON article input {path}: {exc}") from exc
    if isinstance(value, Mapping):
        records = value.get("articles")
        if records is None:
            records = [value]
    else:
        records = value
    if not isinstance(records, list) or any(not isinstance(row, Mapping) for row in records):
        raise EvidenceGapError(
            f"JSON article input {path} must be an object, list of objects, or {{articles: [...]}}"
        )
    return list(records)


def _load_jsonl_records(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvidenceGapError(
                        f"Invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise EvidenceGapError(
                        f"JSONL article row {path}:{line_number} must be an object"
                    )
                rows.append(value)
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing article input: {path}") from exc
    return rows


def _load_parquet_records(path: Path) -> list[Mapping[str, Any]]:
    _, pq = _pyarrow()
    try:
        return pq.read_table(path).to_pylist()
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing article input: {path}") from exc
    except Exception as exc:
        raise EvidenceGapError(f"Unable to read Parquet article input {path}: {exc}") from exc


def load_runtime_articles(path: Path) -> list[RuntimeArticle]:
    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = _load_jsonl_records(path)
    elif suffix == ".json":
        records = _load_json_records(path)
    elif suffix in {".parquet", ".pq"}:
        records = _load_parquet_records(path)
    else:
        raise EvidenceGapError(
            f"Unsupported article input extension {path.suffix!r}; use .jsonl, .json, or .parquet"
        )
    if not records:
        raise EvidenceGapError(f"Article input is empty: {path}")
    articles = [RuntimeArticle.from_mapping(row) for row in records]
    seen: set[str] = set()
    for article in articles:
        if article.article_id in seen:
            raise EvidenceGapError(f"Duplicate runtime article_id: {article.article_id}")
        seen.add(article.article_id)
    return articles


def _auto_sections(abstract: str) -> tuple[tuple[str, int, int], ...]:
    matches = list(_SECTION_PATTERN.finditer(abstract))
    if len(matches) < 2:
        return (("abstract", 0, len(abstract)),)
    result: list[tuple[str, int, int]] = []
    prefix = abstract[: matches[0].start()]
    if prefix.strip():
        result.append(("abstract", 0, matches[0].start()))
    for index, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(abstract)
        if abstract[body_start:body_end].strip():
            label = re.sub(r"\s+", "_", match.group("label").strip().lower())
            result.append((label, body_start, body_end))
    return tuple(result) if result else (("abstract", 0, len(abstract)),)


def canonicalize_article_text(
    article: RuntimeArticle,
    *,
    section_mode: str = "auto",
) -> CanonicalArticleText:
    article.validate()
    if section_mode not in {"auto", "none"}:
        raise EvidenceGapError("section_mode must be 'auto' or 'none'")

    chunks: list[str] = []
    segments: list[SourceSegment] = []

    def append_segment(
        *,
        sentence_type: str,
        section: str,
        section_index: int,
        text: str,
    ) -> None:
        if not text:
            return
        if chunks:
            chunks.append("\n\n")
        document_start = sum(len(chunk) for chunk in chunks)
        chunks.append(text)
        segments.append(
            SourceSegment(
                sentence_type=sentence_type,
                section=section,
                section_index=section_index,
                text=text,
                document_start=document_start,
            )
        )

    next_section_index = 0
    if article.title is not None:
        append_segment(
            sentence_type="title",
            section="title",
            section_index=next_section_index,
            text=article.title,
        )
        next_section_index += 1

    if article.sections:
        for section in article.sections:
            append_segment(
                sentence_type="abstract",
                section=section.section,
                section_index=next_section_index,
                text=section.text,
            )
            next_section_index += 1
    elif article.abstract is not None:
        if section_mode == "auto":
            section_ranges = _auto_sections(article.abstract)
        else:
            section_ranges = (("abstract", 0, len(article.abstract)),)
        for section_name, start, end in section_ranges:
            body = article.abstract[start:end]
            left_trim = len(body) - len(body.lstrip())
            right_trimmed = body.rstrip()
            if not right_trimmed.strip():
                continue
            body_start = start + left_trim
            body_end = start + len(right_trimmed)
            append_segment(
                sentence_type="abstract",
                section=section_name,
                section_index=next_section_index,
                text=article.abstract[body_start:body_end],
            )
            next_section_index += 1

    source_text = "".join(chunks)
    if not source_text:
        raise EvidenceGapError(f"{article.article_id}: no materializable text")
    return CanonicalArticleText(source_text=source_text, segments=tuple(segments))


def _splitter_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return only sentence-boundary semantics, excluding execution placement.

    Device, model directory, batching, and CPU fallback are run provenance. They
    must not change RuntimeSentence IDs when the same frozen model emits the
    same boundaries.
    """

    keys = (
        "library",
        "library_version",
        "language",
        "processors",
        "model_package",
        "model_sha256",
        "contract_id",
    )
    return {key: metadata.get(key) for key in keys if metadata.get(key) is not None}


def _splitter_fingerprint(metadata: Mapping[str, Any]) -> str:
    return sha256_text(
        json.dumps(
            _splitter_identity(metadata),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _sentence_id(
    *,
    article_id: str,
    source_text_fingerprint: str,
    splitter_fingerprint: str,
    character_start: int,
    character_end: int,
    sentence_text: str,
) -> str:
    payload = {
        "article_id": article_id,
        "source_text_fingerprint": source_text_fingerprint,
        "splitter_fingerprint": splitter_fingerprint,
        "character_start": character_start,
        "character_end": character_end,
        "sentence_text": sentence_text,
    }
    return "sent_" + sha256_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )[:24]


def materialize_article_sentences(
    article: RuntimeArticle,
    *,
    canonical: CanonicalArticleText,
    segment_spans: Sequence[Sequence[SentenceSpan]],
    splitter_metadata: Mapping[str, Any],
) -> list[RuntimeSentence]:
    if len(canonical.segments) != len(segment_spans):
        raise EvidenceGapError(
            f"{article.article_id}: splitter output count does not match source segments"
        )
    source_text_fingerprint = sha256_text(canonical.source_text)
    splitter_fingerprint = _splitter_fingerprint(splitter_metadata)
    rows: list[RuntimeSentence] = []
    sentence_index = 0
    for segment, spans in zip(canonical.segments, segment_spans, strict=True):
        for within_index, span in enumerate(spans):
            if not 0 <= span.start_char < span.end_char <= len(segment.text):
                raise EvidenceGapError(
                    f"{article.article_id}: splitter emitted invalid offsets for section {segment.section}"
                )
            sentence_text = segment.text[span.start_char : span.end_char]
            if sentence_text != span.text:
                raise EvidenceGapError(
                    f"{article.article_id}: splitter text disagrees with source offsets"
                )
            document_start = segment.document_start + span.start_char
            document_end = segment.document_start + span.end_char
            row = RuntimeSentence(
                sentence_id=_sentence_id(
                    article_id=article.article_id,
                    source_text_fingerprint=source_text_fingerprint,
                    splitter_fingerprint=splitter_fingerprint,
                    character_start=document_start,
                    character_end=document_end,
                    sentence_text=sentence_text,
                ),
                article_id=article.article_id,
                pmid=article.pmid,
                article_rank=article.article_rank,
                sentence_index=sentence_index,
                sentence_index_within_section=within_index,
                sentence_type=segment.sentence_type,
                section=segment.section,
                section_index=segment.section_index,
                sentence_text=sentence_text,
                character_start=document_start,
                character_end=document_end,
                section_character_start=span.start_char,
                section_character_end=span.end_char,
                source_text_fingerprint=source_text_fingerprint,
                splitter_fingerprint=splitter_fingerprint,
            )
            row.validate(source_text=canonical.source_text)
            rows.append(row)
            sentence_index += 1
    if not rows:
        raise EvidenceGapError(f"{article.article_id}: splitter produced no sentences")
    return rows


class StanzaSentenceSplitter:
    def __init__(
        self,
        *,
        model_dir: Path,
        device: str,
        package: str = DEFAULT_STANZA_PACKAGE,
        batch_size: int = 32,
        allow_cpu_fallback: bool = False,
    ) -> None:
        try:
            import stanza
        except ImportError as exc:
            raise EvidenceGapError(
                "Missing dependency stanza. Install requirements/v1-phase07.txt"
            ) from exc
        if batch_size <= 0:
            raise EvidenceGapError("Stanza tokenize batch_size must be positive")
        self._stanza = stanza
        self._requested_device = device
        self._fallback_used = False
        kwargs = {
            "lang": "en",
            "dir": str(model_dir.resolve()),
            "package": None,
            "processors": {"tokenize": package},
            "download_method": None,
            "device": device,
            "tokenize_batch_size": batch_size,
            "verbose": False,
        }
        try:
            self._pipeline = stanza.Pipeline(**kwargs)
        except Exception as exc:
            if not allow_cpu_fallback or not device.startswith("cuda") or not _looks_like_cuda_failure(exc):
                raise EvidenceGapError(
                    "Unable to initialize Stanza GENIA tokenizer. Run the Phase 07 "
                    "download-sentence-model command first and verify the requested device. "
                    f"Original error: {exc}"
                ) from exc
            kwargs["device"] = "cpu"
            try:
                self._pipeline = stanza.Pipeline(**kwargs)
            except Exception as fallback_exc:
                raise EvidenceGapError(
                    "Stanza CUDA initialization failed and CPU fallback also failed: "
                    f"{fallback_exc}"
                ) from fallback_exc
            self._fallback_used = True
        model_path = model_dir.resolve() / "en" / "tokenize" / f"{package}.pt"
        resources_path = model_dir.resolve() / "resources.json"
        self._metadata = {
            "library": "stanza",
            "library_version": str(getattr(stanza, "__version__", "unknown")),
            "language": "en",
            "processors": {"tokenize": package},
            "model_package": package,
            "model_dir": str(model_dir.resolve()),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path) if model_path.exists() else None,
            "resources_sha256": (
                sha256_file(resources_path) if resources_path.exists() else None
            ),
            "requested_device": self._requested_device,
            "actual_device": str(getattr(self._pipeline, "device", kwargs["device"])),
            "cpu_fallback_used": self._fallback_used,
            "tokenize_batch_size": batch_size,
            "download_method": "none",
            "contract_id": RUNTIME_SENTENCE_CONTRACT_ID,
        }

    @property
    def metadata(self) -> Mapping[str, Any]:
        return dict(self._metadata)

    def split_many(self, texts: Sequence[str]) -> list[list[SentenceSpan]]:
        if not texts:
            return []
        try:
            documents = self._pipeline.bulk_process(list(texts))
        except Exception as exc:
            raise EvidenceGapError(f"Stanza sentence segmentation failed: {exc}") from exc
        if len(documents) != len(texts):
            raise EvidenceGapError("Stanza returned an unexpected document count")
        outputs: list[list[SentenceSpan]] = []
        for text, document in zip(texts, documents, strict=True):
            spans: list[SentenceSpan] = []
            for sentence in document.sentences:
                tokens = list(sentence.tokens)
                if not tokens:
                    continue
                start = tokens[0].start_char
                end = tokens[-1].end_char
                if start is None or end is None:
                    raise EvidenceGapError("Stanza did not return character offsets")
                start_int = int(start)
                end_int = int(end)
                if not 0 <= start_int < end_int <= len(text):
                    raise EvidenceGapError("Stanza returned invalid character offsets")
                spans.append(
                    SentenceSpan(
                        start_char=start_int,
                        end_char=end_int,
                        text=text[start_int:end_int],
                    )
                )
            outputs.append(spans)
        return outputs


def _looks_like_cuda_failure(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "cuda",
        "cudnn",
        "cublas",
        "gpu",
        "nvidia driver",
        "device-side",
        "out of memory",
    )
    return any(marker in message for marker in markers)


def download_stanza_sentence_model(
    root: Path,
    *,
    model_dir: Path | None = None,
    package: str = DEFAULT_STANZA_PACKAGE,
    download_source: str = "auto",
) -> dict[str, Any]:
    try:
        import stanza
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing dependency stanza. Install requirements/v1-phase07.txt"
        ) from exc
    if download_source not in DOWNLOAD_SOURCES:
        raise EvidenceGapError(
            f"Unsupported Stanza download source: {download_source!r}. "
            f"Choose one of {DOWNLOAD_SOURCES}."
        )
    root = root.resolve()
    destination = (
        model_dir.resolve()
        if model_dir is not None
        else (root / DEFAULT_STANZA_MODEL_DIR).resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    sources = ("huggingface", "stanford") if download_source == "auto" else (download_source,)
    failures: list[dict[str, str]] = []
    download_list = None
    selected_source = None
    for source in sources:
        kwargs: dict[str, Any] = {
            "lang": "en",
            "model_dir": str(destination),
            "package": None,
            "processors": {"tokenize": package},
            "verbose": False,
        }
        if source == "stanford":
            # Stanza >=1.13 normally resolves model URLs through
            # huggingface_hub.  Some compute nodes can reach Stanford's
            # download host but not the Hub metadata/Xet endpoints, so keep an
            # official non-Hugging-Face source as a first-class fallback.
            kwargs.update(
                resources_url="stanford",
                model_url=STANFORD_MODEL_URL,
            )
        try:
            download_list = stanza.download(**kwargs)
        except Exception as exc:  # preserve both attempt errors for operators
            failures.append(
                {
                    "source": source,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        selected_source = source
        break

    if download_list is None or selected_source is None:
        detail = "; ".join(
            f"{item['source']}={item['error_type']}: {item['error']}"
            for item in failures
        )
        raise EvidenceGapError(
            "Unable to download Stanza GENIA tokenizer from any configured "
            f"official source. Attempts: {detail}"
        )

    return {
        "status": "PASS",
        "library": "stanza",
        "library_version": str(getattr(stanza, "__version__", "unknown")),
        "language": "en",
        "processors": {"tokenize": package},
        "model_dir": relative_path(root, destination),
        "download_source_requested": download_source,
        "download_source_used": selected_source,
        "failed_attempts": failures,
        "downloaded": download_list,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def _torch_runtime() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    result: dict[str, Any] = {
        "installed": True,
        "version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        result["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(
                    torch.cuda.get_device_properties(index).total_memory
                ),
            }
            for index in range(torch.cuda.device_count())
        ]
    return result


def check_stanza_runtime(
    root: Path,
    *,
    model_dir: Path | None = None,
    device: str = "cuda:0",
    package: str = DEFAULT_STANZA_PACKAGE,
    batch_size: int = 32,
    allow_cpu_fallback: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    resolved_model_dir = (
        model_dir.resolve()
        if model_dir is not None
        else (root / DEFAULT_STANZA_MODEL_DIR).resolve()
    )
    started = time.perf_counter()
    splitter = StanzaSentenceSplitter(
        model_dir=resolved_model_dir,
        device=device,
        package=package,
        batch_size=batch_size,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    load_elapsed = time.perf_counter() - started
    sample = (
        "BACKGROUND: Vitamin D has immunomodulatory effects. "
        "METHODS: We conducted a randomized trial in 1,200 adults. "
        "RESULTS: Infection incidence was 12.5% versus 15.8% (p = 0.04)."
    )
    inference_started = time.perf_counter()
    spans = splitter.split_many([sample])[0]
    inference_elapsed = time.perf_counter() - inference_started
    return {
        "status": "PASS",
        "splitter": dict(splitter.metadata),
        "python": platform.python_version(),
        "torch": _torch_runtime(),
        "model_load_seconds": round(load_elapsed, 6),
        "sample_inference_seconds": round(inference_elapsed, 6),
        "sample_sentences": [span.text for span in spans],
    }


def materialize_runtime_sentences(
    root: Path,
    *,
    input_path: Path,
    run_name: str,
    model_dir: Path | None = None,
    device: str = "cuda:0",
    package: str = DEFAULT_STANZA_PACKAGE,
    batch_size: int = 32,
    section_mode: str = "auto",
    allow_cpu_fallback: bool = False,
    artifact_root: Path | None = None,
    force: bool = False,
    splitter: SentenceSplitter | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    input_path = input_path.resolve()
    articles = load_runtime_articles(input_path)
    canonical_articles = [
        canonicalize_article_text(article, section_mode=section_mode)
        for article in articles
    ]

    resolved_model_dir = (
        model_dir.resolve()
        if model_dir is not None
        else (root / DEFAULT_STANZA_MODEL_DIR).resolve()
    )
    splitter_instance = splitter or StanzaSentenceSplitter(
        model_dir=resolved_model_dir,
        device=device,
        package=package,
        batch_size=batch_size,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    splitter_metadata = dict(splitter_instance.metadata)

    all_segments = [
        segment
        for canonical in canonical_articles
        for segment in canonical.segments
    ]
    # A title is a distinct evidence source unit by contract, not a miniature
    # abstract.  Keep it as exactly one RuntimeSentence even if it contains
    # punctuation that a statistical tokenizer might interpret as boundaries.
    all_span_groups: list[list[SentenceSpan] | None] = [None] * len(all_segments)
    model_positions: list[int] = []
    model_texts: list[str] = []
    for index, segment in enumerate(all_segments):
        if segment.sentence_type == "title":
            all_span_groups[index] = [
                SentenceSpan(0, len(segment.text), segment.text)
            ]
        else:
            model_positions.append(index)
            model_texts.append(segment.text)
    inference_started = time.perf_counter()
    model_span_groups = splitter_instance.split_many(model_texts)
    inference_elapsed = time.perf_counter() - inference_started
    if len(model_span_groups) != len(model_positions):
        raise EvidenceGapError("Splitter output does not match the model segment count")
    for position, spans in zip(model_positions, model_span_groups, strict=True):
        all_span_groups[position] = spans
    if any(group is None for group in all_span_groups):
        raise EvidenceGapError("Internal error: an article segment was not materialized")
    finalized_span_groups = [group for group in all_span_groups if group is not None]

    sentences: list[RuntimeSentence] = []
    cursor = 0
    source_text_by_article: dict[str, str] = {}
    for article, canonical in zip(articles, canonical_articles, strict=True):
        group_count = len(canonical.segments)
        article_groups = finalized_span_groups[cursor : cursor + group_count]
        cursor += group_count
        article_sentences = materialize_article_sentences(
            article,
            canonical=canonical,
            segment_spans=article_groups,
            splitter_metadata=splitter_metadata,
        )
        sentences.extend(article_sentences)
        source_text_by_article[article.article_id] = canonical.source_text

    validation = validate_runtime_sentence_rows(
        [row.to_dict() for row in sentences],
        articles=articles,
        source_text_by_article=source_text_by_article,
        expected_splitter_fingerprint=_splitter_fingerprint(splitter_metadata),
    )

    name = _safe_name(run_name)
    base = (
        artifact_root.resolve()
        if artifact_root is not None
        else (root / DEFAULT_ARTIFACT_ROOT).resolve()
    )
    target = base / name
    with atomic_directory(target, force=force) as staging:
        article_rows: list[dict[str, Any]] = []
        for article, canonical in zip(articles, canonical_articles, strict=True):
            row = article.to_dict()
            row.update(
                {
                    "canonical_source_text": canonical.source_text,
                    "canonical_source_text_sha256": sha256_text(canonical.source_text),
                    "source_segment_count": len(canonical.segments),
                }
            )
            article_rows.append(row)
        sentence_rows = [row.to_dict() for row in sentences]
        article_path = staging / "runtime_articles.parquet"
        sentence_path = staging / "runtime_sentences.parquet"
        preview_path = staging / "runtime_sentences.jsonl"
        article_count = _write_parquet_atomic(article_path, article_rows)
        sentence_count = _write_parquet_atomic(sentence_path, sentence_rows)
        preview_count = _write_jsonl_atomic(preview_path, sentence_rows)
        if sentence_count != preview_count:
            raise EvidenceGapError("Parquet and JSONL runtime sentence counts disagree")

        output_manifest = {
            "runtime_articles": {
                "path": relative_path(root, target / article_path.name),
                "sha256": sha256_file(article_path),
                "rows": article_count,
            },
            "runtime_sentences": {
                "path": relative_path(root, target / sentence_path.name),
                "sha256": sha256_file(sentence_path),
                "rows": sentence_count,
            },
            "runtime_sentences_preview": {
                "path": relative_path(root, target / preview_path.name),
                "sha256": sha256_file(preview_path),
                "rows": preview_count,
            },
        }
        manifest = {
            "schema_version": RUN_SCHEMA_VERSION,
            "contract_schema_version": RUNTIME_SENTENCE_SCHEMA_VERSION,
            "contract_id": RUNTIME_SENTENCE_CONTRACT_ID,
            "run_type": RUN_TYPE,
            "run_name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "input_path": relative_path(root, input_path),
                "input_sha256": sha256_file(input_path),
                "articles": len(articles),
            },
            "materialization": {
                "section_mode": section_mode,
                "structured_abstract_parser": {
                    "parser_id": STRUCTURED_ABSTRACT_PARSER_ID,
                    "activation_policy": "at_least_two_allowlisted_headers",
                    "header_count": len(_SECTION_LABELS),
                    "headers": list(_SECTION_LABELS),
                    "header_text_policy": "stored_as_section_metadata_not_sentence_text",
                },
                "title_policy": "separate_source_segment_when_present",
                "compatibility_text_policy": "text_is_treated_as_abstract_body",
                "content_normalization": "CRLF_to_LF_and_outer_trim_per_input_field",
                "splitter": splitter_metadata,
                "splitter_identity": _splitter_identity(splitter_metadata),
                "splitter_fingerprint": _splitter_fingerprint(splitter_metadata),
                "inference_seconds": round(inference_elapsed, 6),
            },
            "counts": {
                "articles": len(articles),
                "source_segments": len(all_segments),
                "sentences": len(sentences),
                "title_sentences": sum(
                    row.sentence_type == "title" for row in sentences
                ),
                "abstract_sentences": sum(
                    row.sentence_type == "abstract" for row in sentences
                ),
            },
            "validation": validation,
            "outputs": output_manifest,
        }
        atomic_write_json(staging / "run_manifest.json", manifest)

    return {
        "status": "PASS",
        "run_name": name,
        "artifact_dir": relative_path(root, target),
        "articles": len(articles),
        "source_segments": len(all_segments),
        "sentences": len(sentences),
        "splitter": splitter_metadata,
        "inference_seconds": round(inference_elapsed, 6),
        "validation": validation,
        "outputs": output_manifest,
    }


def validate_runtime_sentence_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    articles: Sequence[RuntimeArticle] | None = None,
    source_text_by_article: Mapping[str, str] | None = None,
    expected_splitter_fingerprint: str | None = None,
) -> dict[str, Any]:
    if not rows:
        raise EvidenceGapError("Runtime sentence artifact cannot be empty")
    sentence_ids: set[str] = set()
    by_article: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("schema_version") != RUNTIME_SENTENCE_SCHEMA_VERSION:
            raise EvidenceGapError("Unexpected runtime sentence schema_version")
        if row.get("contract_id") != RUNTIME_SENTENCE_CONTRACT_ID:
            raise EvidenceGapError("Unexpected runtime sentence contract_id")
        sentence_id = str(row.get("sentence_id", ""))
        if not sentence_id or sentence_id in sentence_ids:
            raise EvidenceGapError(f"Duplicate or empty runtime sentence_id: {sentence_id!r}")
        sentence_ids.add(sentence_id)
        article_id = str(row.get("article_id", ""))
        if not article_id:
            raise EvidenceGapError(f"{sentence_id}: article_id cannot be empty")
        if expected_splitter_fingerprint is not None and row.get(
            "splitter_fingerprint"
        ) != expected_splitter_fingerprint:
            raise EvidenceGapError(f"{sentence_id}: splitter fingerprint mismatch")
        by_article.setdefault(article_id, []).append(row)

    expected_article_ids = (
        None if articles is None else {article.article_id for article in articles}
    )
    if expected_article_ids is not None and set(by_article) != expected_article_ids:
        raise EvidenceGapError("Runtime sentence article coverage does not match input")

    for article_id, article_rows in by_article.items():
        ordered = sorted(article_rows, key=lambda row: int(row["sentence_index"]))
        actual_indices = [int(row["sentence_index"]) for row in ordered]
        if actual_indices != list(range(len(ordered))):
            raise EvidenceGapError(f"{article_id}: sentence_index values are not contiguous")
        source_text = (
            None
            if source_text_by_article is None
            else source_text_by_article.get(article_id)
        )
        previous_start = -1
        for row in ordered:
            start = int(row["character_start"])
            end = int(row["character_end"])
            sentence_text = str(row["sentence_text"])
            if not 0 <= start < end:
                raise EvidenceGapError(f"{row['sentence_id']}: invalid offsets")
            if start < previous_start:
                raise EvidenceGapError(f"{article_id}: sentences are not in source order")
            previous_start = start
            if source_text is not None and source_text[start:end] != sentence_text:
                raise EvidenceGapError(
                    f"{row['sentence_id']}: offsets do not reproduce sentence text"
                )

    return {
        "status": "PASS",
        "articles": len(by_article),
        "sentences": len(rows),
        "unique_sentence_ids": len(sentence_ids),
        "contiguous_sentence_indices": True,
        "offset_round_trip": source_text_by_article is not None,
    }


def validate_runtime_sentence_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing runtime sentence manifest: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceGapError(f"Invalid runtime sentence manifest: {exc}") from exc
    if manifest.get("contract_id") != RUNTIME_SENTENCE_CONTRACT_ID:
        raise EvidenceGapError("Runtime sentence manifest contract_id mismatch")
    sentence_path = artifact_dir / "runtime_sentences.parquet"
    article_path = artifact_dir / "runtime_articles.parquet"
    preview_path = artifact_dir / "runtime_sentences.jsonl"
    for path in (sentence_path, article_path, preview_path):
        if not path.exists():
            raise EvidenceGapError(f"Missing runtime sentence artifact file: {path}")
    _, pq = _pyarrow()
    sentence_rows = pq.read_table(sentence_path).to_pylist()
    article_rows = pq.read_table(article_path).to_pylist()
    source_text_by_article = {
        str(row["article_id"]): str(row["canonical_source_text"])
        for row in article_rows
    }
    validation = validate_runtime_sentence_rows(
        sentence_rows,
        source_text_by_article=source_text_by_article,
        expected_splitter_fingerprint=manifest["materialization"][
            "splitter_fingerprint"
        ],
    )
    expected_outputs = manifest.get("outputs", {})
    for key, path in (
        ("runtime_articles", article_path),
        ("runtime_sentences", sentence_path),
        ("runtime_sentences_preview", preview_path),
    ):
        expected_sha = expected_outputs.get(key, {}).get("sha256")
        if expected_sha != sha256_file(path):
            raise EvidenceGapError(f"Checksum mismatch for {path}")
    return {
        **validation,
        "run_name": manifest.get("run_name"),
        "artifact_dir": str(artifact_dir),
        "checksums": "PASS",
        "splitter": manifest["materialization"]["splitter"],
    }
