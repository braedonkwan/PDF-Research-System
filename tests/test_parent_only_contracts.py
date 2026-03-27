from __future__ import annotations

import json
import unittest

from local_llm.context_pipeline import _build_tagged_context_envelope
from local_llm.rag import ParentHit, RetrievalResult, SourceHit
from local_llm.working_memory import MemoryParentHit, MemoryRetrievalResult


class ParentOnlyContractTests(unittest.TestCase):
    def test_public_retrieval_results_have_no_child_hits_field(self) -> None:
        rag_result = RetrievalResult()
        memory_result = MemoryRetrievalResult()

        self.assertFalse(hasattr(rag_result, "child_hits"))
        self.assertFalse(hasattr(memory_result, "child_hits"))

    def test_context_envelope_uses_hard_cutover_keys(self) -> None:
        rag_result = RetrievalResult(
            source_hits=[
                SourceHit(
                    source_id="pdf_001",
                    source_path="data/pdfs/sample.pdf",
                    score=0.77,
                    parent_count=3,
                )
            ],
            parent_hits=[
                ParentHit(
                    parent_id="p000001",
                    heading="Section A",
                    page_start=1,
                    page_end=2,
                    score=0.81,
                    source_id="pdf_001",
                    source_path="data/pdfs/sample.pdf",
                    text="Parent section body.",
                )
            ],
        )
        memory_result = MemoryRetrievalResult(
            parent_hits=[
                MemoryParentHit(
                    parent_id="turn_000001",
                    score=0.69,
                    turn_start=1,
                    turn_end=1,
                    text="User: hi\\nAssistant: hello",
                )
            ]
        )
        envelope = _build_tagged_context_envelope(
            last_rounds=[{"round_index": 1, "user_query": {"text": "hi"}, "response": {"text": "hello"}}],
            memory_payload={"recent_long_term_rounds": [vars(memory_result.parent_hits[0])]},
            rag_payload={
                "sources": [vars(rag_result.source_hits[0])],
                "parent_sections": [vars(rag_result.parent_hits[0])],
            },
        )
        payload = json.loads(envelope)
        self.assertEqual(
            set(payload.keys()),
            {"working_memory", "knowledge"},
        )
        self.assertEqual(
            set(payload["working_memory"].keys()),
            {"last_n_rounds", "long_term_memory"},
        )
        self.assertNotIn("last_round_context", payload)


if __name__ == "__main__":
    unittest.main()
