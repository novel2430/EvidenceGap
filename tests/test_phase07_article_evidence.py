from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidencegap.pipeline.article_evidence import (
    ARTICLE_EVIDENCE_PROMPT_VERSION,
    SYSTEM_PROMPT,
    ArticlePromptInput,
    PromptSentence,
    build_user_prompt,
    load_article_prompt_inputs,
    run_article_evidence_extractor,
    validate_article_evidence_artifact,
    validate_response_payload,
)
from evidencegap.stance.llm_judge import (
    ProviderResponse,
    _ProviderError,
    call_structured_llm,
)


def _json_read_parquet(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))



@contextmanager
def fake_parquet_io():
    with patch(
        "evidencegap.pipeline.article_evidence._read_parquet",
        side_effect=_json_read_parquet,
    ):
        yield



class ArticleEvidenceFixtureMixin:
    def _write_fixture(self, root: Path) -> Path:
        (root / "src/evidencegap").mkdir(parents=True)
        artifact = root / "artifacts/v1/pipeline/retrieval_adapters/fixture"
        (artifact / "article_retrieval").mkdir(parents=True)
        (artifact / "sentence_materialization").mkdir(parents=True)
        (artifact / "request.json").write_text(
            json.dumps(
                {
                    "claim_id": "claim_fixture",
                    "claim_text": "Vitamin D prevents respiratory infection.",
                }
            ),
            encoding="utf-8",
        )
        top_articles = [
            {
                "article_id": "pmid:1",
                "pmid": "1",
                "final_article_rank": 1,
                "title": "Support trial",
            },
            {
                "article_id": "pmid:2",
                "pmid": "2",
                "final_article_rank": 2,
                "title": "Related review",
            },
        ]
        (artifact / "article_retrieval/top_articles.parquet").write_text(
            json.dumps(top_articles), encoding="utf-8"
        )
        rows = []
        texts = {
            "pmid:1": [
                ("title", "title", "Support trial"),
                ("abstract", "objective", "We tested vitamin D supplementation."),
                (
                    "abstract",
                    "results",
                    "Vitamin D reduced respiratory infections compared with placebo.",
                ),
            ],
            "pmid:2": [
                ("title", "title", "Related review"),
                ("abstract", "background", "Vitamin D has immune effects."),
                ("abstract", "methods", "The literature was reviewed."),
            ],
        }
        for article_id, values in texts.items():
            section_counts: dict[str, int] = {}
            for sentence_index, (sentence_type, section, text) in enumerate(values):
                within = section_counts.get(section, 0)
                section_counts[section] = within + 1
                rows.append(
                    {
                        "article_id": article_id,
                        "pmid": article_id.split(":", 1)[1],
                        "sentence_id": f"{article_id}-s{sentence_index}",
                        "sentence_index": sentence_index,
                        "sentence_index_within_section": within,
                        "sentence_type": sentence_type,
                        "section": section,
                        "section_index": sentence_index,
                        "sentence_text": text,
                        "character_start": sentence_index * 100,
                        "character_end": sentence_index * 100 + len(text),
                        "source_text_fingerprint": f"source-{article_id}",
                        "splitter_fingerprint": "splitter-fixture",
                    }
                )
        (artifact / "sentence_materialization/runtime_sentences.parquet").write_text(
            json.dumps(rows), encoding="utf-8"
        )
        return artifact


class SharedStructuredProviderTests(unittest.TestCase):
    def test_deepseek_structured_call_reuses_json_transport(self) -> None:
        with patch(
            "evidencegap.stance.llm_judge._post_json",
            return_value=(
                {
                    "id": "deepseek-request",
                    "choices": [
                        {
                            "message": {"content": '{"results": []}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                },
                "raw",
            ),
        ) as post_json:
            response = call_structured_llm(
                provider="deepseek",
                api_key="key",
                base_url="https://example.test",
                model="deepseek-v4-pro",
                system_prompt="system",
                user_prompt="user",
                response_schema={"type": "object"},
                max_tokens=100,
                timeout_seconds=10.0,
            )
        body = post_json.call_args.kwargs["body"]
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(response.payload, {"results": []})
        self.assertEqual(response.usage["total_tokens"], 12)

    def test_anthropic_structured_call_passes_json_schema(self) -> None:
        schema = {"type": "object", "properties": {"results": {"type": "array"}}}
        with patch(
            "evidencegap.stance.llm_judge._post_json",
            return_value=(
                {
                    "id": "claude-request",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": '{"results": []}'}],
                    "usage": {"input_tokens": 9, "output_tokens": 3},
                },
                "raw",
            ),
        ) as post_json:
            response = call_structured_llm(
                provider="anthropic",
                api_key="key",
                base_url="https://example.test",
                model="claude-sonnet-4-6",
                system_prompt="system",
                user_prompt="user",
                response_schema=schema,
                max_tokens=100,
                timeout_seconds=10.0,
            )
        body = post_json.call_args.kwargs["body"]
        self.assertEqual(body["output_config"]["format"]["schema"], schema)
        self.assertEqual(response.payload, {"results": []})
        self.assertEqual(response.usage["total_tokens"], 12)


class ArticleEvidencePromptTests(unittest.TestCase, ArticleEvidenceFixtureMixin):
    def test_system_prompt_enforces_claim_applicability(self) -> None:
        self.assertEqual(
            ARTICLE_EVIDENCE_PROMPT_VERSION,
            "phase07_article_evidence_v2",
        )
        self.assertIn("Check claim applicability", SYSTEM_PROMPT)
        self.assertIn("treatment study", SYSTEM_PROMPT)
        self.assertIn("prevention claim", SYSTEM_PROMPT)
        self.assertIn("respiratory support", SYSTEM_PROMPT)
        self.assertIn("two active doses", SYSTEM_PROMPT)

    def test_prompt_uses_all_abstract_sentences_but_not_title_as_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, fake_parquet_io():
            root = Path(temp_dir)
            artifact = self._write_fixture(root)
            request, items, _paths = load_article_prompt_inputs(artifact)
            prompt = build_user_prompt(
                claim_text=request["claim_text"],
                items=items[:1],
            )
            self.assertIn("S01", prompt)
            self.assertIn("S02", prompt)
            self.assertIn("Support trial", prompt)
            self.assertNotIn('"sentence_id": "S00"', prompt)
            self.assertEqual(
                [str(value.source["sentence_text"]) for value in items[0].sentences],
                [
                    "We tested vitamin D supplementation.",
                    "Vitamin D reduced respiratory infections compared with placebo.",
                ],
            )

    def test_response_rejects_unknown_sentence_id(self) -> None:
        sentence = PromptSentence(
            alias="S01",
            source={
                "sentence_id": "stable-1",
                "sentence_index": 1,
                "section": "results",
                "sentence_text": "Result.",
            },
        )
        item = ArticlePromptInput(
            article={
                "article_id": "pmid:1",
                "pmid": "1",
                "final_article_rank": 1,
                "title": "Title",
            },
            sentences=(sentence,),
        )
        with self.assertRaises(_ProviderError):
            validate_response_payload(
                {
                    "results": [
                        {
                            "article_id": "pmid:1",
                            "label": "support",
                            "probabilities": {
                                "support": 0.9,
                                "refute": 0.05,
                                "insufficient": 0.05,
                            },
                            "evidence_sentence_ids": ["S99"],
                            "rationale": "The result supports the claim.",
                        }
                    ]
                },
                [item],
            )


class ArticleEvidenceExecutionTests(unittest.TestCase, ArticleEvidenceFixtureMixin):
    def test_dry_run_needs_no_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, fake_parquet_io():
            root = Path(temp_dir)
            artifact = self._write_fixture(root)
            with patch.dict(os.environ, {}, clear=True):
                result = run_article_evidence_extractor(
                    root,
                    retrieval_artifact_dir=artifact,
                    provider="deepseek",
                    dry_run=True,
                )
            self.assertEqual(result["status"], "DRY_RUN")
            self.assertEqual(result["articles"], 2)
            self.assertEqual(result["requests"], 1)
            self.assertEqual(result["prompt_version"], ARTICLE_EVIDENCE_PROMPT_VERSION)

    def test_run_writes_valid_grounded_article_and_evidence_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, fake_parquet_io():
            root = Path(temp_dir)
            artifact = self._write_fixture(root)

            def fake_call(**kwargs):
                self.assertEqual(kwargs["provider"], "deepseek")
                return ProviderResponse(
                    payload={
                        "results": [
                            {
                                "article_id": "pmid:1",
                                "label": "support",
                                "probabilities": {
                                    "support": 0.94,
                                    "refute": 0.01,
                                    "insufficient": 0.05,
                                },
                                "evidence_sentence_ids": ["S02"],
                                "rationale": "The reported trial result directly supports the claim.",
                            },
                            {
                                "article_id": "pmid:2",
                                "label": "insufficient",
                                "probabilities": {
                                    "support": 0.08,
                                    "refute": 0.02,
                                    "insufficient": 0.90,
                                },
                                "evidence_sentence_ids": [],
                                "rationale": "The review excerpt provides background and methods only.",
                            },
                        ]
                    },
                    request_id="req-fixture",
                    usage={"input_tokens": 100, "output_tokens": 40, "total_tokens": 140},
                    raw_response_sha256="a" * 64,
                    finish_reason="stop",
                )

            with (
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture"}, clear=True),
                patch(
                    "evidencegap.pipeline.article_evidence.call_structured_llm",
                    side_effect=fake_call,
                ) as call,
            ):
                result = run_article_evidence_extractor(
                    root,
                    retrieval_artifact_dir=artifact,
                    provider="deepseek",
                    run_name="fixture",
                    request_batch_size=2,
                    artifact_root=root / "article_evidence",
                    cache_dir=root / "cache",
                )

            self.assertEqual(call.call_count, 1)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["articles"], 2)
            self.assertEqual(result["evidence_selections"], 1)
            output_dir = root / "article_evidence/fixture"
            validation = validate_article_evidence_artifact(output_dir)
            self.assertEqual(validation["status"], "PASS")
            self.assertEqual(validation["title_evidence_count"], 0)

            rows = [
                json.loads(line)
                for line in (output_dir / "article_evidence.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            selected = rows[0]["selected_evidence"]
            self.assertEqual(selected[0]["sentence_id"], "pmid:1-s2")
            self.assertEqual(
                selected[0]["sentence_text"],
                "Vitamin D reduced respiratory infections compared with placebo.",
            )


if __name__ == "__main__":
    unittest.main()
