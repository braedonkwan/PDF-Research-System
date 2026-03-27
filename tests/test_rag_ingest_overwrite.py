from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_llm.rag import _reset_store_dir


class RagIngestOverwriteTests(unittest.TestCase):
    def test_reset_store_dir_removes_previous_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "rag_store"
            (store / "old.bin").parent.mkdir(parents=True, exist_ok=True)
            (store / "old.bin").write_text("x", encoding="utf-8")
            nested = store / "nested"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / "stale.txt").write_text("stale", encoding="utf-8")

            _reset_store_dir(store)

            self.assertTrue(store.exists())
            self.assertTrue(store.is_dir())
            self.assertEqual(list(store.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
