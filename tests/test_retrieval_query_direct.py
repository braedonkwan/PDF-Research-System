from __future__ import annotations

import unittest
from unittest.mock import patch

from local_llm.chat_context_service import RetrievedContext
from local_llm.chat_runtime import ChatRuntimeOptions
from local_llm.client import _collect_turn_context


class RetrievalQueryDirectTests(unittest.TestCase):
    def test_retrieval_query_uses_retrieval_input_directly(self) -> None:
        captured: dict[str, object] = {}

        def _fake_collect_context_with_last_rounds(*args, **kwargs):
            captured.update(kwargs)
            return RetrievedContext(
                context_text=None,
                rag_summary=None,
                memory_summary=None,
                status_lines=[],
                retrieval_query_text=str(kwargs.get("retrieval_query_text", "")),
            )

        with patch(
            "local_llm.client.collect_context_with_last_rounds",
            side_effect=_fake_collect_context_with_last_rounds,
        ):
            result = _collect_turn_context(
                query_prompt="Prompt body",
                retrieval_input="Use this exact retrieval query",
                options=ChatRuntimeOptions(),
                rag_store=None,
                working_memory_store=None,
                last_rounds_buffer=None,
                context_user_name="User",
                context_assistant_name="Assistant",
                recent_memory_exclude_last_turns=0,
            )

        self.assertEqual(
            captured.get("retrieval_query_text"),
            "Use this exact retrieval query",
        )
        self.assertEqual(result.retrieval_query_text, "Use this exact retrieval query")


if __name__ == "__main__":
    unittest.main()
