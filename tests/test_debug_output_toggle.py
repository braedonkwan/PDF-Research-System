from __future__ import annotations

import unittest
from unittest.mock import patch

from local_llm.chat_context_service import RetrievedContext
from local_llm.chat_runtime import ChatRuntimeOptions
from local_llm.client import _run_contextual_turn


def _sample_context_result() -> RetrievedContext:
    return RetrievedContext(
        context_text='{"knowledge":[],"working_memory":[]}',
        rag_summary=None,
        memory_summary=None,
        status_lines=["[RAG] using 0 source docs, 0 parent sections (off)"],
        retrieval_query_text="test query",
    )


class DebugOutputToggleTests(unittest.TestCase):
    def test_debug_output_disabled_hides_context_debug_prints(self) -> None:
        context_result = _sample_context_result()
        with patch(
            "local_llm.client._collect_turn_context",
            return_value=context_result,
        ), patch(
            "local_llm.client._stream_labeled_reply",
            return_value=(True, "assistant reply"),
        ), patch("local_llm.client._print_context_status_lines") as status_mock, patch(
            "local_llm.client._print_retrieval_query_and_context"
        ) as query_context_mock:
            ok, reply, returned_context = _run_contextual_turn(
                client=object(),  # type: ignore[arg-type]
                label="Assistant",
                prompt="hello",
                retrieval_input="hello",
                options=ChatRuntimeOptions(debug_output=False),
                rag_store=None,
                working_memory_store=None,
                last_rounds_buffer=None,
                context_user_name="User",
                context_assistant_name="Assistant",
                recent_memory_exclude_last_turns=0,
            )

        self.assertTrue(ok)
        self.assertEqual(reply, "assistant reply")
        self.assertIs(returned_context, context_result)
        status_mock.assert_not_called()
        query_context_mock.assert_not_called()

    def test_debug_output_enabled_prints_context_debug_sections(self) -> None:
        context_result = _sample_context_result()
        with patch(
            "local_llm.client._collect_turn_context",
            return_value=context_result,
        ), patch(
            "local_llm.client._stream_labeled_reply",
            return_value=(True, "assistant reply"),
        ), patch("local_llm.client._print_context_status_lines") as status_mock, patch(
            "local_llm.client._print_retrieval_query_and_context"
        ) as query_context_mock:
            _run_contextual_turn(
                client=object(),  # type: ignore[arg-type]
                label="Assistant",
                prompt="hello",
                retrieval_input="hello",
                options=ChatRuntimeOptions(debug_output=True),
                rag_store=None,
                working_memory_store=None,
                last_rounds_buffer=None,
                context_user_name="User",
                context_assistant_name="Assistant",
                recent_memory_exclude_last_turns=0,
            )

        status_mock.assert_called_once_with(context_result)
        query_context_mock.assert_called_once_with(context_result)


if __name__ == "__main__":
    unittest.main()
