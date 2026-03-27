from __future__ import annotations

import unittest

from local_llm.context_pipeline import _should_apply_rag_context
from local_llm.rag import ParentHit, RetrievalResult


class RagChildScoreGatingTests(unittest.TestCase):
    def test_auto_gate_uses_child_scores_when_available(self) -> None:
        retrieval = RetrievalResult(
            parent_hits=[
                ParentHit(
                    parent_id="p1",
                    heading="A",
                    page_start=1,
                    page_end=1,
                    score=0.91,
                )
            ],
            top_child_score=0.20,
            second_child_score=0.10,
            selected_child_count=2,
            candidate_child_count=12,
        )
        applied, decision = _should_apply_rag_context(
            "According to the pdf, summarize section A",
            retrieval,
            rag_mode="auto",
            min_child_score=0.30,
            min_score_margin=0.02,
        )
        self.assertFalse(applied)
        self.assertIn("top child score", decision)

    def test_auto_gate_falls_back_to_parent_scores_without_child_metadata(self) -> None:
        retrieval = RetrievalResult(
            parent_hits=[
                ParentHit(
                    parent_id="p1",
                    heading="A",
                    page_start=1,
                    page_end=1,
                    score=0.86,
                ),
                ParentHit(
                    parent_id="p2",
                    heading="B",
                    page_start=2,
                    page_end=2,
                    score=0.50,
                ),
            ],
            top_child_score=0.0,
            second_child_score=0.0,
            selected_child_count=0,
            candidate_child_count=0,
        )
        applied, decision = _should_apply_rag_context(
            "What does the document say about topic B?",
            retrieval,
            rag_mode="auto",
            min_child_score=0.30,
            min_score_margin=0.02,
        )
        self.assertTrue(applied)
        self.assertIn("top child score", decision)


if __name__ == "__main__":
    unittest.main()
