from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidencegap.common import EvidenceGapError
from evidencegap.stance.contracts import StanceInput
from evidencegap.stance.inputs import _adjacent_context
from evidencegap.stance.llm_judge import _execution_plan, _select_inputs


def phase05_input(query_id: str, rank: int) -> StanceInput:
    return StanceInput(
        input_id=f"stance:phase05:{query_id}:{rank}",
        dataset="evidencebench_100k",
        split="dev",
        claim_id=query_id,
        query_id=query_id,
        claim_text=f"Claim for {query_id}",
        paper_id=f"paper-{query_id}",
        sentence_index=rank - 1,
        sentence_type="normal paragraph",
        evidence_rank=rank,
        evidence_text=f"Evidence {rank} for {query_id}",
        evidence_unit="sentence",
        context_before="Previous sentence" if rank > 1 else None,
        context_after="Next sentence",
    )


class Phase05ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = [
            phase05_input(query_id, rank)
            for query_id in ("q1", "q2", "q3", "q4")
            for rank in (1, 2, 3)
        ]

    def test_adjacent_context_preserves_exact_neighbors(self) -> None:
        sentences = ["Heading", "First result.", "Second result.", "Discussion."]
        before, after, before_indices, after_indices = _adjacent_context(
            sentences,
            sentence_index=2,
            context_window=1,
        )
        self.assertEqual(before, "First result.")
        self.assertEqual(after, "Discussion.")
        self.assertEqual(before_indices, [1])
        self.assertEqual(after_indices, [3])

    def test_query_sample_keeps_complete_top_k_groups(self) -> None:
        selected, metadata = _select_inputs(
            self.inputs,
            query_sample_size=2,
            query_sample_seed=7,
        )
        selected_ids = {item.query_id for item in selected}
        self.assertEqual(len(selected_ids), 2)
        self.assertEqual(len(selected), 6)
        self.assertTrue(metadata["complete_query_groups"])
        for query_id in selected_ids:
            self.assertEqual(
                [item.evidence_rank for item in selected if item.query_id == query_id],
                [1, 2, 3],
            )

    def test_query_sample_is_deterministic(self) -> None:
        first, first_meta = _select_inputs(
            self.inputs,
            query_sample_size=2,
            query_sample_seed=20260722,
        )
        second, second_meta = _select_inputs(
            self.inputs,
            query_sample_size=2,
            query_sample_seed=20260722,
        )
        self.assertEqual(
            [item.input_id for item in first],
            [item.input_id for item in second],
        )
        self.assertEqual(
            first_meta["selection_sha256"],
            second_meta["selection_sha256"],
        )

    def test_row_and_query_selection_cannot_be_mixed(self) -> None:
        with self.assertRaises(EvidenceGapError):
            _select_inputs(self.inputs, limit=5, query_limit=1)

    def test_execution_plan_estimates_requests(self) -> None:
        selected, selection = _select_inputs(self.inputs, query_limit=2)
        plan = _execution_plan(
            selected,
            selection=selection,
            request_batch_size=5,
        )
        self.assertEqual(plan["selected_queries"], 2)
        self.assertEqual(plan["selected_rows"], 6)
        self.assertEqual(plan["estimated_api_requests"], 2)
        self.assertEqual(plan["rank_gap_queries"], 0)
        self.assertEqual(plan["duplicate_sentence_queries"], 0)
        self.assertEqual(plan["rows_per_query"], {"3": 2})


class Phase05CacheExportTests(unittest.TestCase):
    def test_collects_only_exact_cached_complete_queries(self) -> None:
        import tempfile

        from evidencegap.stance.llm_judge import (
            ProviderResponse,
            _collect_exact_cached_batches,
            _complete_cached_query_ids,
            _request_fingerprint,
            _write_cache,
        )

        inputs = [
            phase05_input(query_id, rank)
            for query_id in ("q1", "q2")
            for rank in (1, 2)
        ]
        provider = "deepseek"
        model = "deepseek-v4-pro"
        base_url = "https://api.deepseek.com"
        max_tokens = 4096
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            batch = inputs[:2]
            request_hash = _request_fingerprint(
                provider=provider,
                model=model,
                base_url=base_url,
                inputs=batch,
                max_tokens=max_tokens,
                thinking=False,
            )
            results = [
                {
                    "input_id": item.input_id,
                    "label": "support",
                    "probabilities": {
                        "support": 0.9,
                        "refute": 0.02,
                        "insufficient": 0.08,
                    },
                    "rationale": "The result supports the claim.",
                    "evidence_type": "direct_result",
                    "requires_context": False,
                }
                for item in batch
            ]
            response = ProviderResponse(
                payload={"results": results},
                request_id="request-1",
                usage={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                },
                raw_response_sha256="a" * 64,
                finish_reason="stop",
            )
            _write_cache(
                cache_dir / f"{request_hash}.json",
                request_hash=request_hash,
                provider=provider,
                model=model,
                input_ids=[item.input_id for item in batch],
                response=response,
                validated_results=results,
            )

            cached, stats = _collect_exact_cached_batches(
                inputs,
                provider=provider,
                model=model,
                base_url=base_url,
                request_batch_size=2,
                max_tokens=max_tokens,
                thinking=False,
                provider_cache=cache_dir,
            )
            complete, query_stats = _complete_cached_query_ids(
                inputs,
                set(cached),
            )

        self.assertEqual(set(cached), {item.input_id for item in inputs[:2]})
        self.assertEqual(stats["completed_batches"], 1)
        self.assertEqual(stats["missing_batches"], 1)
        self.assertEqual(stats["cached_usage"]["total_tokens"], 150)
        self.assertEqual(complete, {"q1"})
        self.assertEqual(query_stats["complete_cached_queries"], 1)
        self.assertEqual(query_stats["uncached_queries"], 1)

    def test_partial_query_is_excluded(self) -> None:
        from evidencegap.stance.llm_judge import _complete_cached_query_ids

        inputs = [phase05_input("q1", rank) for rank in (1, 2, 3)]
        complete, stats = _complete_cached_query_ids(
            inputs,
            {inputs[0].input_id, inputs[1].input_id},
        )
        self.assertEqual(complete, set())
        self.assertEqual(stats["incomplete_cached_queries"], 1)
        self.assertEqual(stats["cached_rows_excluded"], 2)


if __name__ == "__main__":
    unittest.main()
