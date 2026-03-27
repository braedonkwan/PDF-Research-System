from __future__ import annotations

import unittest

from local_llm.last_rounds_buffer import LastRoundsBuffer


class _FakeMemoryStore:
    def __init__(self) -> None:
        self.appended: list[tuple[str, str]] = []

    def append_turn(self, user_text: str, assistant_text: str) -> None:
        self.appended.append((user_text, assistant_text))


class LastRoundsBufferTests(unittest.TestCase):
    def test_eviction_appends_oldest_round_to_memory(self) -> None:
        fake_store = _FakeMemoryStore()
        buffer = LastRoundsBuffer(memory_store=fake_store, max_context_rounds=1)

        buffer.append("first user", "first assistant")
        buffer.append("second user", "second assistant")

        self.assertEqual(fake_store.appended, [("first user", "first assistant")])
        rounds = buffer.build_rounds(user_name="Input", assistant_name="Agent")
        self.assertEqual(len(rounds), 1)
        self.assertEqual(rounds[0]["round_index"], 1)
        self.assertEqual(rounds[0]["user_query"]["text"], "second user")
        self.assertEqual(rounds[0]["response"]["text"], "second assistant")


if __name__ == "__main__":
    unittest.main()
