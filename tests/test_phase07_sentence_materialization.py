from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidencegap.common import EvidenceGapError
from evidencegap.pipeline import (
    RUNTIME_SENTENCE_CONTRACT_ID,
    RuntimeArticle,
    canonicalize_article_text,
    download_stanza_sentence_model,
    load_runtime_articles,
    materialize_article_sentences,
    materialize_runtime_sentences,
    validate_runtime_sentence_rows,
)
from evidencegap.pipeline.sentence_materialization import SentenceSpan


class FakeBiomedicalSplitter:
    def __init__(self) -> None:
        self._metadata = {
            "library": "fixture",
            "library_version": "1",
            "language": "en",
            "processors": {"tokenize": "fixture"},
            "model_package": "fixture",
            "actual_device": "cpu",
            "contract_id": RUNTIME_SENTENCE_CONTRACT_ID,
        }

    @property
    def metadata(self):
        return dict(self._metadata)

    def split_many(self, texts):
        groups = []
        for text in texts:
            spans = []
            cursor = 0
            for match in re.finditer(r".+?(?:[.!?](?=\s|$)|$)", text, re.DOTALL):
                start = match.start()
                end = match.end()
                while start < end and text[start].isspace():
                    start += 1
                while end > start and text[end - 1].isspace():
                    end -= 1
                if start < end:
                    spans.append(SentenceSpan(start, end, text[start:end]))
                cursor = end
            if cursor < len(text) and text[cursor:].strip():
                start = cursor
                while start < len(text) and text[start].isspace():
                    start += 1
                spans.append(SentenceSpan(start, len(text), text[start:]))
            groups.append(spans)
        return groups




class Phase07SentenceModelDownloadTests(unittest.TestCase):
    def test_auto_download_falls_back_to_official_stanford_source(self) -> None:
        calls = []

        class FakeStanza:
            __version__ = "1.14.0"

            @staticmethod
            def download(**kwargs):
                calls.append(dict(kwargs))
                if "model_url" not in kwargs:
                    raise RuntimeError("Hub metadata unavailable")
                return [["tokenize", "genia"]]

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            sys.modules, {"stanza": FakeStanza}
        ):
            result = download_stanza_sentence_model(
                Path(temp_dir),
                download_source="auto",
            )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["download_source_used"], "stanford")
        self.assertEqual(len(result["failed_attempts"]), 1)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("model_url", calls[0])
        self.assertEqual(calls[1]["resources_url"], "stanford")
        self.assertIn("nlp.stanford.edu", calls[1]["model_url"])

    def test_explicit_stanford_source_skips_huggingface(self) -> None:
        calls = []

        class FakeStanza:
            __version__ = "1.14.0"

            @staticmethod
            def download(**kwargs):
                calls.append(dict(kwargs))
                return [["tokenize", "genia"]]

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            sys.modules, {"stanza": FakeStanza}
        ):
            result = download_stanza_sentence_model(
                Path(temp_dir),
                download_source="stanford",
            )

        self.assertEqual(result["download_source_used"], "stanford")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["resources_url"], "stanford")

class Phase07RuntimeArticleContractTests(unittest.TestCase):
    def test_current_corpus_text_field_is_supported_without_guessing_title(self) -> None:
        article = RuntimeArticle.from_mapping(
            {
                "article_id": "pmid:1",
                "pmid": "1",
                "text": "This is the stored article text. It remains an abstract body.",
                "final_article_rank": 2,
                "cross_encoder_score": 7.4,
            }
        )
        self.assertIsNone(article.title)
        self.assertTrue(article.abstract.startswith("This is"))
        self.assertEqual(article.article_rank, 2)
        self.assertEqual(article.source_metadata["cross_encoder_score"], 7.4)

    def test_sections_are_mutually_exclusive_with_abstract(self) -> None:
        with self.assertRaises(EvidenceGapError):
            RuntimeArticle.from_mapping(
                {
                    "article_id": "a",
                    "abstract": "Body",
                    "sections": [{"section": "results", "text": "Result"}],
                }
            )

    def test_jsonl_loader_rejects_duplicate_article_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "articles.jsonl"
            path.write_text(
                json.dumps({"article_id": "a", "text": "One."})
                + "\n"
                + json.dumps({"article_id": "a", "text": "Two."})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(EvidenceGapError):
                load_runtime_articles(path)


class Phase07SentenceMaterializationContractTests(unittest.TestCase):
    def test_title_is_a_separate_source_segment(self) -> None:
        article = RuntimeArticle.from_mapping(
            {
                "article_id": "pmid:2",
                "title": "Dr. Smith et al. report a randomized trial",
                "abstract": "The intervention reduced infections. No serious events occurred.",
            }
        )
        canonical = canonicalize_article_text(article, section_mode="none")
        self.assertEqual(len(canonical.segments), 2)
        self.assertEqual(canonical.segments[0].sentence_type, "title")
        self.assertEqual(canonical.segments[0].text, article.title)
        self.assertEqual(
            canonical.source_text,
            f"{article.title}\n\n{article.abstract}",
        )

    def test_structured_abstract_labels_are_preserved_as_sections(self) -> None:
        article = RuntimeArticle.from_mapping(
            {
                "article_id": "pmid:3",
                "abstract": (
                    "BACKGROUND: Prior studies were inconsistent. "
                    "METHODS: We randomized 200 adults. "
                    "RESULTS: Infection risk was reduced."
                ),
            }
        )
        canonical = canonicalize_article_text(article, section_mode="auto")
        self.assertEqual(
            [segment.section for segment in canonical.segments],
            ["background", "methods", "results"],
        )
        self.assertTrue(canonical.segments[2].text.startswith("Infection risk"))

    def test_medfact_inline_measurements_header_is_recovered_before_stanza(self) -> None:
        article = RuntimeArticle.from_mapping(
            {
                "article_id": "pmid:27861708",
                "abstract": (
                    "OBJECTIVES: To determine efficacy. "
                    "DESIGN: Randomized controlled trial. "
                    "INTERVENTION: The high-dose group received monthly supplement of vitamin D "
                    "MEASUREMENTS: The primary outcome was incidence of ARI. "
                    "RESULTS: The high-dose group had fewer ARIs. "
                    "CONCLUSION: Monthly high-dose vitamin D"
                ),
            }
        )
        canonical = canonicalize_article_text(article, section_mode="auto")
        self.assertEqual(
            [segment.section for segment in canonical.segments],
            [
                "objectives",
                "design",
                "intervention",
                "measurements",
                "results",
                "conclusion",
            ],
        )
        self.assertEqual(
            canonical.segments[2].text,
            "The high-dose group received monthly supplement of vitamin D",
        )
        self.assertEqual(
            canonical.segments[3].text,
            "The primary outcome was incidence of ARI.",
        )
        self.assertNotIn("MEASUREMENTS:", canonical.source_text)
        self.assertNotIn("CONCLUSION:", canonical.source_text)

        splitter = FakeBiomedicalSplitter()
        span_groups = splitter.split_many(
            [segment.text for segment in canonical.segments]
        )
        rows = materialize_article_sentences(
            article,
            canonical=canonical,
            segment_spans=span_groups,
            splitter_metadata=splitter.metadata,
        )
        measurements = [row for row in rows if row.section == "measurements"]
        self.assertEqual(len(measurements), 1)
        self.assertEqual(
            measurements[0].sentence_text,
            "The primary outcome was incidence of ARI.",
        )
        for row in rows:
            self.assertNotRegex(row.sentence_text, r"^[A-Z][A-Z ]+:$")
            self.assertNotIn(" MEASUREMENTS:", row.sentence_text)

    def test_execution_device_does_not_change_sentence_ids(self) -> None:
        article = RuntimeArticle.from_mapping(
            {"article_id": "pmid:device", "abstract": "One result sentence."}
        )
        canonical = canonicalize_article_text(article)
        spans = [[SentenceSpan(0, len(canonical.segments[0].text), canonical.segments[0].text)]]
        base_metadata = FakeBiomedicalSplitter().metadata
        cuda_metadata = {**base_metadata, "requested_device": "cuda:0", "actual_device": "cuda:0", "tokenize_batch_size": 32}
        cpu_metadata = {**base_metadata, "requested_device": "cuda:0", "actual_device": "cpu", "tokenize_batch_size": 8}
        cuda_rows = materialize_article_sentences(
            article,
            canonical=canonical,
            segment_spans=spans,
            splitter_metadata=cuda_metadata,
        )
        cpu_rows = materialize_article_sentences(
            article,
            canonical=canonical,
            segment_spans=spans,
            splitter_metadata=cpu_metadata,
        )
        self.assertEqual(cuda_rows[0].sentence_id, cpu_rows[0].sentence_id)
        self.assertEqual(
            cuda_rows[0].splitter_fingerprint, cpu_rows[0].splitter_fingerprint
        )

    def test_offsets_round_trip_and_ids_are_repeatable(self) -> None:
        article = RuntimeArticle.from_mapping(
            {
                "article_id": "pmid:4",
                "pmid": "4",
                "title": "Trial title",
                "sections": [
                    {
                        "section": "results",
                        "text": "Risk fell from 15.8% to 12.5%. The difference was significant.",
                    }
                ],
            }
        )
        canonical = canonicalize_article_text(article)
        splitter = FakeBiomedicalSplitter()
        title_group = [SentenceSpan(0, len(article.title), article.title)]
        body_group = splitter.split_many([canonical.segments[1].text])[0]
        first = materialize_article_sentences(
            article,
            canonical=canonical,
            segment_spans=[title_group, body_group],
            splitter_metadata=splitter.metadata,
        )
        second = materialize_article_sentences(
            article,
            canonical=canonical,
            segment_spans=[title_group, body_group],
            splitter_metadata=splitter.metadata,
        )
        self.assertEqual([row.sentence_id for row in first], [row.sentence_id for row in second])
        self.assertEqual([row.sentence_index for row in first], [0, 1, 2])
        self.assertEqual(first[0].sentence_type, "title")
        for row in first:
            self.assertEqual(
                canonical.source_text[row.character_start : row.character_end],
                row.sentence_text,
            )
        validation = validate_runtime_sentence_rows(
            [row.to_dict() for row in first],
            articles=[article],
            source_text_by_article={article.article_id: canonical.source_text},
            expected_splitter_fingerprint=first[0].splitter_fingerprint,
        )
        self.assertEqual(validation["status"], "PASS")
        self.assertTrue(validation["offset_round_trip"])


class Phase07SentenceArtifactIntegrationTests(unittest.TestCase):
    def test_materialization_writes_reusable_artifacts_and_manifest(self) -> None:
        def fake_parquet_writer(path: Path, rows) -> int:
            values = list(rows)
            path.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
            return len(values)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "articles.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "article_id": "pmid:10",
                        "pmid": "10",
                        "title": "A title. With punctuation.",
                        "abstract": "BACKGROUND: Context sentence. RESULTS: Direct result sentence.",
                        "final_article_rank": 1,
                        "cross_encoder_score": 8.1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch(
                "evidencegap.pipeline.sentence_materialization._write_parquet_atomic",
                side_effect=fake_parquet_writer,
            ):
                result = materialize_runtime_sentences(
                    root,
                    input_path=input_path,
                    run_name="fixture",
                    splitter=FakeBiomedicalSplitter(),
                )

            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["articles"], 1)
            # Title remains exactly one sentence even though it contains punctuation.
            self.assertEqual(result["sentences"], 3)
            artifact_dir = root / "artifacts/v1/pipeline/runtime_sentences/fixture"
            manifest = json.loads(
                (artifact_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["contract_id"], RUNTIME_SENTENCE_CONTRACT_ID)
            self.assertEqual(manifest["counts"]["title_sentences"], 1)
            self.assertEqual(manifest["counts"]["abstract_sentences"], 2)
            self.assertEqual(
                manifest["materialization"]["title_policy"],
                "separate_source_segment_when_present",
            )
            preview_rows = [
                json.loads(line)
                for line in (artifact_dir / "runtime_sentences.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(preview_rows[0]["sentence_text"], "A title. With punctuation.")
            self.assertIn("cross_encoder_score", json.loads(
                json.loads((artifact_dir / "runtime_articles.parquet").read_text(encoding="utf-8"))[0]["source_metadata_json"]
            ))


if __name__ == "__main__":
    unittest.main()
