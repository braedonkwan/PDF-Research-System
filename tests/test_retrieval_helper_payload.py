from __future__ import annotations

import json
import unittest

from local_llm.client import _build_retrieval_query


class _FakeHelperClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def stream_reply(self, user_text: str, context_text: str | None = None):
        self.prompts.append(user_text)
        yield self.reply


class RetrievalHelperPayloadTests(unittest.TestCase):
    def test_payload_uses_strict_hard_cutover_shape(self) -> None:
        helper = _FakeHelperClient('{"retrieval_query":"network security intro"}')
        rounds = [
            {
                "round_index": 1,
                "user_query": {"speaker": "Input", "text": "what is x"},
                "response": {"speaker": "Agent", "text": "x is ..."},
            }
        ]
        query = _build_retrieval_query(
            helper,
            latest_input="Explain network security basics",
            current_agent_system_prompt="You are Agent 1",
            last_n_rounds=rounds,
        )

        self.assertEqual(query, "network security intro")
        self.assertEqual(len(helper.prompts), 1)
        payload = json.loads(helper.prompts[0])
        self.assertEqual(
            set(payload.keys()),
            {"latest_input", "current_agent_system_prompt", "last_n_rounds"},
        )
        self.assertEqual(payload["latest_input"], "Explain network security basics")
        self.assertEqual(payload["current_agent_system_prompt"], "You are Agent 1")
        self.assertEqual(payload["last_n_rounds"], rounds)


if __name__ == "__main__":
    unittest.main()
