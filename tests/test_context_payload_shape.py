from __future__ import annotations

import json
import unittest

from local_llm.client import _build_contextual_user_input


class ContextPayloadShapeTests(unittest.TestCase):
    def test_user_payload_uses_working_memory_knowledge_user_query(self) -> None:
        context = {
            "working_memory": {
                "last_n_rounds": [
                    {
                        "round_index": 1,
                        "user_query": {"speaker": "User", "text": "What is x?"},
                        "response": {"speaker": "Assistant", "text": "x is ..."},
                    }
                ],
                "long_term_memory": {
                    "recent_long_term_rounds": [],
                    "older_long_term_rounds": [],
                },
            },
            "knowledge": {
                "sources": [],
                "parent_sections": [],
            },
        }
        payload_text = _build_contextual_user_input(
            "Explain x",
            json.dumps(context, ensure_ascii=False),
        )
        payload = json.loads(payload_text)
        self.assertEqual(
            set(payload.keys()),
            {"working_memory", "knowledge", "user_query"},
        )
        self.assertEqual(payload["user_query"], "Explain x")
        self.assertEqual(payload["working_memory"], context["working_memory"])
        self.assertEqual(payload["knowledge"], context["knowledge"])


if __name__ == "__main__":
    unittest.main()
