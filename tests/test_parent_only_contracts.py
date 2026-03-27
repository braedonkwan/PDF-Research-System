from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from local_llm.context_pipeline import _build_tagged_context_envelope, _serialize_memory_rounds
from local_llm.rag import ParentHit, RetrievalResult, SourceHit
from local_llm.working_memory import MemoryRetrievalResult


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
        envelope = _build_tagged_context_envelope(
            last_rounds=[
                {
                    "round_index": 2,
                    "user_query": {"speaker": "User", "text": "hi"},
                    "response": {"speaker": "Assistant", "text": "hello"},
                }
            ],
            memory_payload={
                "older_long_term_rounds": [
                    {
                        "round_id": "turn_000001",
                        "score": 0.69,
                        "turn_start": 1,
                        "turn_end": 1,
                        "user_query": {"speaker": "User", "text": "earlier"},
                        "response": {"speaker": "Assistant", "text": "earlier response"},
                    }
                ]
            },
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
        self.assertIsInstance(payload["working_memory"], list)
        self.assertEqual(
            payload["working_memory"][0]["content"],
            "earlier",
        )
        self.assertEqual(
            payload["working_memory"][1]["content"],
            "earlier response",
        )
        self.assertEqual(
            payload["working_memory"][2]["content"],
            "hi",
        )
        self.assertEqual(
            payload["working_memory"][3]["content"],
            "hello",
        )
        self.assertIsInstance(payload["knowledge"], list)
        self.assertGreaterEqual(len(payload["knowledge"]), 1)
        self.assertIn("source", payload["knowledge"][0])
        self.assertIn("content", payload["knowledge"][0])
        self.assertNotIn("last_round_context", payload)

    def test_memory_round_serialization_preserves_custom_speaker_roles(self) -> None:
        hits = [
            SimpleNamespace(
                parent_id="turn_000010",
                score=0.81,
                turn_start=10,
                turn_end=10,
                text="Agent 1: draft answer\nAgent 2: critique",
            )
        ]
        payload = _serialize_memory_rounds(hits, max_chars=300)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["user_query"]["speaker"], "Agent 1")
        self.assertEqual(payload[0]["response"]["speaker"], "Agent 2")


if __name__ == "__main__":
    unittest.main()
