from __future__ import annotations

import unittest

from local_llm.last_rounds_buffer import LastRoundsBuffer


class _FakeMemoryStore:
    def __init__(self) -> None:
        self.appended: list[tuple[str, str, str, str]] = []

    def append_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        user_speaker: str = "User",
        assistant_speaker: str = "Assistant",
    ) -> None:
        self.appended.append((user_text, assistant_text, user_speaker, assistant_speaker))


class LastRoundsBufferTests(unittest.TestCase):
    def test_eviction_appends_oldest_round_to_memory(self) -> None:
        fake_store = _FakeMemoryStore()
        buffer = LastRoundsBuffer(memory_store=fake_store, max_context_rounds=1)

        buffer.append("first user", "first assistant")
        buffer.append("second user", "second assistant")

        self.assertEqual(
            fake_store.appended,
            [("first user", "first assistant", "User", "Assistant")],
        )
        rounds = buffer.build_rounds(user_name="Input", assistant_name="Agent")
        self.assertEqual(len(rounds), 1)
        self.assertEqual(rounds[0]["round_index"], 1)
        self.assertEqual(rounds[0]["user_query"]["text"], "second user")
        self.assertEqual(rounds[0]["response"]["text"], "second assistant")

    def test_rounds_preserve_per_turn_speaker_identity(self) -> None:
        buffer = LastRoundsBuffer(memory_store=None, max_context_rounds=2)
        buffer.append(
            "original user question",
            "agent1 answer",
            input_speaker="User",
            output_speaker="Agent 1",
        )
        buffer.append(
            "agent2 critique",
            "agent1 revision",
            input_speaker="Agent 2",
            output_speaker="Agent 1",
        )

        rounds = buffer.build_rounds(user_name="Input", assistant_name="Agent")
        self.assertEqual(len(rounds), 2)
        self.assertEqual(rounds[0]["user_query"]["speaker"], "User")
        self.assertEqual(rounds[0]["response"]["speaker"], "Agent 1")
        self.assertEqual(rounds[1]["user_query"]["speaker"], "Agent 2")
        self.assertEqual(rounds[1]["response"]["speaker"], "Agent 1")

    def test_eviction_passes_custom_speaker_labels_to_memory(self) -> None:
        fake_store = _FakeMemoryStore()
        buffer = LastRoundsBuffer(memory_store=fake_store, max_context_rounds=1)
        buffer.append(
            "agent1 answer",
            "agent2 critique",
            input_speaker="Agent 1",
            output_speaker="Agent 2",
        )
        buffer.append("next user", "next assistant")
        self.assertEqual(
            fake_store.appended[0],
            ("agent1 answer", "agent2 critique", "Agent 1", "Agent 2"),
        )


if __name__ == "__main__":
    unittest.main()
