from __future__ import annotations

import unittest
from pathlib import Path

from local_llm.settings import load_settings


class DefaultsContractsTests(unittest.TestCase):
    def _assert_prompt_sections(self, prompt: str) -> None:
        self.assertIn("Identity:", prompt)
        self.assertIn("Objective:", prompt)
        self.assertIn("Given context and usage:", prompt)
        self.assertIn("Conflict handling:", prompt)
        self.assertIn("Output style constraints:", prompt)

    def test_project_defaults_match_spec(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_settings(root / "settings.json")

        self.assertEqual(config.default_model, "qwen35_35b")
        self.assertEqual(config.runtime.last_n_rounds, 1)
        self.assertEqual(config.runtime.rag.top_k, 8)
        self.assertEqual(config.runtime.working_memory.recent_window_rounds, 5)
        self.assertEqual(config.runtime.working_memory.recent_top_k_rounds, 8)
        self.assertEqual(config.runtime.working_memory.older_top_k_rounds, 5)
        self.assertEqual(config.server.ctx_size, 32768)
        self.assertEqual(config.server.batch_size, 1536)
        self.assertEqual(config.server.ubatch_size, 512)

        default_model = config.get_model()
        self.assertEqual(default_model.alias, "qwen35_35b")
        self.assertIn(
            "Qwen3.5-35B-A3B-heretic-v2.Q4_K_M.gguf",
            str(default_model.gguf_path),
        )

        self.assertIn("Q&A agent", config.chat.system_prompt)
        self.assertIn("Q&A agent", config.runtime.agent_loop.agent1_system_prompt)
        self.assertIn("Critic agent", config.runtime.agent_loop.agent2_system_prompt)
        self._assert_prompt_sections(config.chat.system_prompt)
        self._assert_prompt_sections(config.runtime.agent_loop.agent1_system_prompt)
        self._assert_prompt_sections(config.runtime.agent_loop.agent2_system_prompt)
        self.assertNotIn("long_term_memory", config.chat.system_prompt)
        self.assertNotIn("long_term_memory", config.runtime.agent_loop.agent1_system_prompt)
        self.assertNotIn("long_term_memory", config.runtime.agent_loop.agent2_system_prompt)


if __name__ == "__main__":
    unittest.main()
