from __future__ import annotations

import json
import unittest

from local_llm.client import _build_contextual_user_input


class ContextPayloadShapeTests(unittest.TestCase):
    def test_user_payload_uses_working_memory_knowledge_user_query(self) -> None:
        working_memory = [
            {
                "round_index": 1,
                "role": "User",
                "content": "Earlier context",
            },
            {
                "round_index": 1,
                "role": "Assistant",
                "content": "Earlier answer",
            },
            {
                "round_index": 2,
                "role": "User",
                "content": "What is x?",
            },
            {
                "round_index": 2,
                "role": "Assistant",
                "content": "x is ...",
            }
        ]
        context = {
            "working_memory": working_memory,
            "knowledge": [],
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
        self.assertIsInstance(payload["working_memory"], list)
        self.assertEqual(len(payload["working_memory"]), 4)
        self.assertEqual(
            payload["working_memory"][0]["role"],
            "User",
        )
        self.assertEqual(
            payload["working_memory"][0]["content"],
            "Earlier context",
        )
        self.assertEqual(
            payload["working_memory"][1]["role"],
            "Assistant",
        )
        self.assertEqual(
            payload["working_memory"][1]["content"],
            "Earlier answer",
        )
        self.assertEqual(
            payload["working_memory"][2]["role"],
            "User",
        )
        self.assertEqual(
            payload["working_memory"][2]["content"],
            "What is x?",
        )
        self.assertEqual(
            payload["working_memory"][3]["role"],
            "Assistant",
        )
        self.assertEqual(
            payload["working_memory"][3]["content"],
            "x is ...",
        )
        self.assertEqual(payload["knowledge"], [])


if __name__ == "__main__":
    unittest.main()
